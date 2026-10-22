# flash-ssft

flash-ssft（Split-Step Fourier Transform）通过 **GPU 算子优化加速 Lugiato–Lefever 方程（LLE）仿真**。核心是两个 Triton 融合算子：频域线性乘法与泵浦相加、时域 Kerr 相位旋转。它们减少逐点计算中的 kernel 启动、临时张量和显存读写；FFT/IFFT 继续使用 cuFFT，CUDA Graph 负责降低多步循环的提交开销。

当前求解标准单场 Kerr LLE，采用一阶 Lie 分步傅里叶法：每步先完成线性演化与泵浦注入，再完成 Kerr 非线性演化。NumPy 实现提供数值参考，独立算子和完整循环的 benchmark 检查优化收益。

求解的归一化方程为

\[
\partial_\tau\psi=-(1+i\alpha)\psi
+i\frac{d_2}{2}\partial_\theta^2\psi
+\frac{d_3}{6}\partial_\theta^3\psi
+i|\psi|^2\psi+F.
\]

`d3=0` 时为标准单场基准。傅里叶变换采用 NumPy 约定：`fft` 不缩放，`ifft` 除以模态数。频域线性项使用步长为 `h` 的精确指数推进，随后向零模加入 `NFh`，再经 IFFT 执行 Kerr 精确相位旋转。线性与泵浦的联合推进采用一阶近似，整体时间离散为一阶；每个完整时间步做一次 IFFT 和一次 FFT。

## 单独一步与多步循环

每个 Lie 时间步接收时域（角度域）场，并返回下一步的时域场，计算顺序如下：

```text
时域状态 ψ
  │
  ▼
cuFFT：FFT
  │
  ▼
Triton：频域线性乘法 + 零模泵浦（全步）
  │
  ▼
cuFFT：IFFT
  │
  ▼
Triton：|ψ|² + 相位 + sin/cos + 复数乘法（Kerr 全步）
  │
  ▼
下一步时域状态 ψ
```

第一个 Triton 核函数把各模态的线性乘法与零模泵浦合在一次读写中；第二个核函数对每个时域采样点读取一次复数场，在寄存器中完成模平方、相位旋转和复数乘法，然后写回。**cuFFT 的 IFFT 和 FFT 仍是独立算子。**

线性与泵浦全步 `L` 在时域场上执行 `FFT → 频域仿射 → IFFT`，其中频域仿射为 `a ← linear*a + pump*δ₀`。`linear_coefficients(params)` 在循环外预计算完整步长 `h = params.h` 的传播因子与泵浦增量：

\[
\mathrm{linear}_\mu=e^{\lambda_\mu h},\qquad
\mathrm{pump}=NFh.
\]

这两个参数已经包含步长，`linear_and_pump_` 先做复数乘法，再向零模加入泵浦；泵浦增量采用一阶近似。`kerr_` 接收同一步长 `h`，根据当前场计算相位旋转。执行 `n` 步的顺序为：

```text
(L → N) 重复 n 次
```

频域仿射 kernel、Kerr、IFFT 和 FFT 均各执行 `n` 次；`n=0` 时不执行演化子步。CUDA Graph 录制并回放若干个完整的 `L → N` 循环。图回放减少 Python/CUDA 提交开销，**不等于算子融合**。

这两个融合算子也可以单独调用。下面直接组合它们完成**一步**；多步计算通常调用求解器，让它管理缓冲区和 CUDA Graph：

```python
import torch
from flashlle import LLEParams, initial_field
from flashlle.gpu import linear_coefficients
from flashlle.operators import kerr_, linear_and_pump_

params = LLEParams()
linear, pump = linear_coefficients(params)
linear_gpu = torch.as_tensor(linear, device="cuda", dtype=torch.complex64)
psi = torch.as_tensor(initial_field(params), device="cuda", dtype=torch.complex64)

a = torch.fft.fft(psi)
linear_and_pump_(a, linear_gpu, pump)
psi = torch.fft.ifft(a)
kerr_(psi, params.h)
```

## 代码组织

```text
src/flashlle/
├── __init__.py                  直接导出公开接口
├── gpu.py                       参数、系数、单步组合、CUDA Graph 与求解入口
├── model.py                     initial_field 初态
├── reference.py                 NumPy 数值参考
├── _validation.py               主机输入检查，不依赖 GPU 模块
└── operators/
    ├── __init__.py              导出两个融合算子
    ├── kerr.py                  Kerr kernel 及 Python 调用入口
    ├── linear_and_pump.py       线性与泵浦 kernel 及 Python 调用入口
    └── _validation.py           算子的 complex64 CUDA 张量检查
```

`gpu.py` 依次定义 `LLEParams`、`linear_coefficients`、`lle_step` 和执行部分。`lle_step` 只组合一次 `FFT → 线性/泵浦 → IFFT → Kerr`；`CudaGraphStepper` 录制与回放多步，`simulate_gpu` 处理主机数据、图缓存和最终结果。算子模块只负责各自的逐点更新。

依赖方向为 `model/reference → gpu → operators`。`reference.py` 从 `gpu.py` 取得公共参数和系数；`gpu.py` 与 `reference.py` 都使用独立的 `_validation.check_input(psi0, n_modes, steps)`，`gpu.py` 不依赖 NumPy 参考模块。`__init__.py` 直接导出公开函数，便于从 `flashlle` 调用。

新增局部物理项时，在 `operators/` 增加算子，并在 `lle_step` 中明确它的计算域和执行顺序；只修改色散时，调整 `linear_coefficients` 即可。系数仍在循环外计算，传给 Kerr 的 `h` 必须与预计算步长一致。执行器通过 `functools.partial` 绑定系数与步长，持有时域状态；修改算子组合后需要重新捕获图。

## 安装与使用

基础依赖为 Python 3.10 或更新版本、NumPy、PyTorch 和 Triton。PyTorch 与 Triton 均在包导入时加载，CPU 参考也需要安装，但不要求 CUDA GPU。GPU 求解需要 NVIDIA GPU 和支持 CUDA 的 PyTorch。以下命令中的 `python` 指所选环境里的解释器，安装命令在项目根目录执行：

```bash
python -m pip install -e .
```

需要运行测试时，安装 `dev` 额外依赖：

```bash
python -m pip install -e '.[dev]'
```

一个典型调用如下。`initial_field` 产生角度域初态；两个 `simulate_*` 函数都返回主机上的最终角度域场。NumPy 默认使用 `complex128`，GPU 默认使用 `complex64`。

```python
from flashlle import LLEParams, initial_field, simulate_gpu, simulate_numpy

params = LLEParams(
    n_modes=1024,
    pump_f=2.5,
    detuning_alpha=5.0,
    d2=0.02,
    h=0.005,
    d3=0.0,
)
psi0 = initial_field(params)

cpu_final = simulate_numpy(psi0, params, steps=10_000)
gpu_final = simulate_gpu(psi0, params, steps=10_000, graph_chunk=16)
```

GPU 路径使用 `cuFFT + Triton 逐点融合 + CUDA Graph`。只要 `steps > 0` 就使用图，实际分块大小为 `chunk = min(graph_chunk, steps)`；例如 `graph_chunk=16` 时，每次最多录制/回放 16 个完整时间步 `L → N`。余下的 `steps % chunk` 个时间步直接执行；`steps=0` 返回输入副本，不建图。切换模型结构或录制时固定的配置，需要重新捕获图。GPU 首次调用还包含 CUDA 初始化、Triton 编译和图捕获，因此应与预热后的求解时间分开看。

图缓存按完整参数、设备及实际分块大小 `chunk` 区分。参数扫描创建很多不同图时，可调用 `from flashlle import clear_graph_cache; clear_graph_cache()` 释放缓存，再继续下一批参数。

## 验证与性能测量

测试集中在 `tests/test_numerics.py`，覆盖算子公式与原地更新、完整 LLE 步的数值一致性，以及图分块、缓存复用和输入输出独立性：

```bash
python -m pytest
```

没有 CUDA GPU 时，GPU 测试跳过，NumPy 参考测试仍可运行。正确性测试检查计算是否符合当前算法，benchmark 则测量算子优化是否改善运行时间。

| 脚本 | 测量内容 | 用途 |
| --- | --- | --- |
| `benchmark_operators.py` | 同 GPU、同 complex64 下，Kerr 与线性/泵浦的普通 PyTorch 和 Triton 实现，各自采用普通调用或 Graph | 直接比较逐点算子优化 |
| `compare_implementations.py` | 相同 FFT、系数、精度和步数下，完整 LLE 循环的四种组合 | 检查算子优化在完整仿真中的收益 |
| `compare.py` | NumPy complex128 与公开 GPU complex64 求解器，主机输入到主机输出 | 展示整体速度和数值误差，结果包含硬件、精度与实现差异 |
| `compare_graph.py` | Triton 普通循环、缓存图和重新录图，主机输入到主机输出 | 观察实际调用成本；普通循环重新准备系数，缓存图复用系数，因此差值不等于纯 Graph 收益 |
| `compare_chunks.py` | 不同 chunk 的构造时间、缓存运行时间和显存 | 调整图分块大小 |

前两个脚本比较 `torch_eager`、`triton_eager`、`torch_graph`、`triton_graph`。固定执行方式比较 PyTorch 与 Triton，可以衡量算子实现的收益；固定算子实现比较普通调用与 Graph，可以衡量提交方式的收益。普通 PyTorch 算子集中在 `benchmarks/_torch_reference.py`，只作为基准对照，不进入产品求解路径。

算子计时使用驻留 GPU 的数据，排除传输、系数准备、输入恢复、编译、预热和录图，Graph 耗时按实际算子调用次数摊销。完整循环计时同样排除预计算、主机传输和建图，并对四组统一计入输入状态复制、循环执行、输出复制和同步。两种计时分别回答算子和仿真循环的问题。

安装绘图依赖后，运行两个核心对照：

```bash
python -m pip install -e '.[benchmark]'
python benchmarks/benchmark_operators.py --operators kerr linear_and_pump
python benchmarks/compare_implementations.py
```

算子脚本支持 `--sizes`、`--repeats`、`--chunk`、`--calls`、`--h` 和 `--warmup`；显式指定的 `--calls` 必须能被 `--chunk` 整除。结果默认写入带 UTC 时间戳的 `results/benchmark_operators_*.json/.png/.svg`，`--output` 指定输出前缀。旧的 `python benchmarks/benchmark_kerr.py` 入口仍可使用，它转入新脚本并只测 Kerr，不覆盖旧的 `results/benchmark_kerr.*`。

完整循环脚本支持 `--modes`、`--steps`、`--graph-chunk`、`--repeats` 和 `--output`；默认测 1024、2048、4096 模态各 100、1000 步，结果写入 `results/compare_implementations.json`。两种新对照均记录数值误差，性能比较应同时查看误差是否通过。

整体 CPU/GPU 对照仍可独立运行：

```bash
python benchmarks/compare.py
python benchmarks/compare.py --backends numpy
```

`compare.py` 默认测 1024、2048、4096 模态各 1 千、1 万、10 万步，区分首次调用和预热后的重复计时，结果写入 `results/compare.json`。图调用与分块实验分别使用 `compare_graph.py` 和 `compare_chunks.py`。下节历史数据对应旧算法；当前实现的性能以重新运行脚本生成的结果为准。

已有的 `optics-cuda` GPU 环境也可不安装到环境中，直接从本地同步并使用 `PYTHONPATH` 运行：

```bash
./scripts/sync_remote.sh
ssh -o RemoteCommand=none -o RequestTTY=no optics-cuda \
  'wsl.exe -d Ubuntu-24.04-CUDA -u optomachanics --cd /home/optomachanics/projects/FlashLLE -- env PYTHONPATH=/home/optomachanics/projects/FlashLLE/src /home/optomachanics/micromamba/envs/torch-cu126/bin/python -m pytest tests -q'
```

## 历史实测与复现边界

此前采用 Strang 对称分裂的求解器在 GTX 1660 上通过了 15 项数值测试。以下性能及误差数据来自该旧算法在 `optics-cuda` 上的测量，不能用于判断当前一阶 Lie 实现的速度或误差；当前结果需重新运行 benchmark。历史测量固定 `F=2.5, α=5, d2=0.02, h=0.005, d3=0`，单条轨迹运行 100,000 步；NumPy 为 `complex128`，GPU 为 `complex64`。每项是排除首次调用后 3 次主机初态到主机最终场计时的中位数，包含传输与同步。

| 模态数 | NumPy | cuFFT + Triton + CUDA Graph | 加速比 |
| ---: | ---: | ---: | ---: |
| 1024 | 5.828 s | 0.927 s | 6.28× |
| 2048 | 10.195 s | 1.583 s | 6.44× |
| 4096 | 18.083 s | 1.784 s | 10.14× |

三种 N 的 100,000 步总功率相对误差不超过 `2.49e-6`，功率谱相对 L2 误差不超过 `1.34e-5`。复场相对 L2 误差为 `1.13%–1.32%`，显示长循环后的相位或位置差异；原定长期验收以功率和谱为准。完整逐次耗时、首次调用时间与环境版本见 [`results/compare_full.json`](results/compare_full.json)。2048 点的三次 GPU 计时为 `1.631 / 1.021 / 1.583 s`，因此请把表中数字视为此设备上的测量值，而非稳定的硬件常数。

此前独立项目的 [尺寸对照](../lle-triton/SIZE_BENCHMARK.md)提供另一轮可交叉核对的结果。不同 GPU、驱动、库版本或模型参数会改变计时；请运行本项目 benchmark 获取本机结果。
