# flash-ssft

使用独立 Triton 算子、cuFFT 和 CUDA Graph 求解含损耗、色散、Kerr 非线性与自陡峭效应的 **NLSE**。Python 包名仍为 `flashlle`，当前求解器接口是 `NLSEParams` 和 `flashlle.solvers.nlse_gpu_opt`。

## 方程与离散约定

求解

\[
\frac{\partial A}{\partial z}
=-\frac{\alpha}{2}A
-\sum_{m=2}^{M}\frac{i^{m-1}\beta_m}{m!}\frac{\partial^m A}{\partial T^m}
+i\gamma_0|A|^2A
-\frac{\gamma_0}{\omega_0}\frac{\partial}{\partial T}(|A|^2A).
\]

这是给定方程中非线性括号展开后的形式。无限色散级数按提供的有限系数截断：`beta2`、`beta3`、`beta4` 对应二至四阶，`beta_higher=(beta5, beta6, ...)` 可增加更高阶，未提供的阶数为零。当前求解器没有泵浦、腔失谐或 Raman 项。

采用周期边界与均匀迟滞时间网格，窗口长度为 `n_samples * dt`，FFT 不做 shift。频率为 `w = 2*pi*fftfreq(n_samples, dt)`，因此 `∂T ↔ i*w`。线性传播因子严格按上述符号生成：

\[
L(w)=\exp\!\left[\Delta z\left(-\frac{\alpha}{2}
+i\sum_{m=2}^{M}\frac{(-1)^m\beta_m w^m}{m!}\right)\right].
\]

二至四阶分别为 `+i*beta2*w**2/2`、`-i*beta3*w**3/6`、`+i*beta4*w**4/24`。

## 单步推进

复用现有算子，按以下顺序推进：

```text
时域 A → FFT → disp_and_loss_(L) → IFFT → kerr_shock_(h=gamma0*dz, omega0)
```

若线性步后的场为 `A_L`，则非线性步为：

```text
B     = A_L * exp(i * gamma0 * dz * |A_L|²)
A_new = B - gamma0 * dz / omega0 * ∂T(|B|² B)
```

Kerr 相位旋转精确计算，自陡峭使用显式 Euler 更新，整体为**一阶 Lie 分裂**。NumPy 参考实现使用相同步进方法；CPU/GPU 一致性不能代替收敛性检查。减小 `dz` 并加密时间网格，检查所关心的场和能量是否收敛。FFT 窗口应足够大以避免脉冲绕回；当前没有额外去混叠滤波。

- `alpha` 是功率损耗系数，场幅度衰减为 `exp(-alpha*z/2)`，不是 dB 数值。
- `gamma0` 同时缩放 Kerr 和自陡峭项；`gamma0=0` 退化为线性传播。
- `omega0` 是正的载波角频率，与 `w` 使用相同单位；`omega0=float("inf")` 显式关闭自陡峭。
- `dt`、`dz`、`beta_m`、`gamma0`、`omega0` 必须采用一致的时间、长度和功率单位；默认值仅是归一化示例。
- 输入为一维复数 NumPy 数组，输出是传播到 `z=steps*dz` 的新数组。NumPy 使用 `complex128`，GPU 使用 `complex64`。

## 快速开始

Python ≥ 3.10。只运行 NumPy 参考求解器和 CPU 测试：

```bash
python -m pip install -e '.[dev]'
```

```python
import numpy as np
from flashlle.solvers import NLSEParams, time_grid, simulate_numpy

params = NLSEParams(
    n_samples=1024, dt=0.02, dz=0.001,
    alpha=0.0, beta2=-1.0, beta3=0.05, beta4=0.0,
    gamma0=1.0, omega0=20.0,
)
t = time_grid(params)
a0 = (1 / np.cosh(t)).astype(np.complex128)
final = simulate_numpy(a0, params, steps=1000)
```

在有 NVIDIA GPU、CUDA PyTorch 和 Triton 的环境中：

```bash
python -m pip install -e '.[gpu,dev,benchmark]'
```

```python
from flashlle.solvers.nlse_gpu_opt import simulate_gpu, clear_graph_cache

final_gpu = simulate_gpu(a0, params, steps=1000, graph_chunk=16)
clear_graph_cache()
```

GPU 正步数使用 CUDA Graph，每次回放最多 `graph_chunk` 步，剩余步骤直接执行。首次调用包含编译和建图；缓存按完整物理参数、网格、图分块与设备区分。零步数或 `dz=0` 返回输入副本。

示例脚本中的 `initial_field` 仅负责构造单位幅度 sech 脉冲，不属于求解器接口：

```bash
python -m examples.run_tod --backend numpy --steps 1000
python -m examples.run_tod --backend compare --steps 1000
```

`run_tod` 使用固定长度为 20 的时间窗口，包含三阶色散与自陡峭。`compare` 模式需 GPU。

## 测试与性能对照

```bash
python -m pytest
python benchmarks/compare.py --backends numpy --samples 1024 --steps 100 1000
python benchmarks/compare_implementations.py --samples 1024 2048 4096 --steps 1000 --repeats 7 --graph-chunk 16
python benchmarks/compare_graph.py --samples 1024 --steps 1 16 17 1000
python benchmarks/compare_chunks.py --samples 1024 --steps 1000
```

NLSE 测试覆盖损耗幅度/功率关系、二至六阶色散的正负 Fourier 模式、非单位 `gamma0`、自陡峭符号、Nyquist 频率，以及与独立整方程 RK4 积分的步长收敛比较。GPU 测试另检查完整单步、图分块、缓存参数隔离、输入保护和 CPU/GPU 一致性。缺少依赖或 CUDA 时，相应 GPU 测试跳过，CPU 测试正常运行。

完整仿真的 benchmark 已切换为 NLSE，普通 PyTorch 基线与 Triton 使用同样的线性、Kerr 和自陡峭推进。输出默认保存为 `*_nlse.json`，每份结果记录方程和计算参数。当前未生成新的 NLSE GPU 性能结果。

独立算子仍可使用 `benchmarks/benchmark_operators.py` 测量。`operators/` 中的 Kerr、Kerr–Raman、Kerr–shock、色散/损耗和线性/泵浦实现均独立保留；NLSE 求解器只组合色散/损耗与 Kerr–shock。

## 调参与性能优化

性能调优在目标 CUDA GPU 上进行。先固定满足精度要求的物理问题和离散方案，再比较执行配置；最佳配置取决于 GPU、采样点数、是否开启自陡峭、传播步数和计算图复用次数。下面给出的数值是初始搜索范围，尚未测定当前 NLSE 的最优配置。

### 区分性能参数与数值参数

| 参数类别 | 参数 | 选择依据 |
| --- | --- | --- |
| Kernel 执行配置 | `BLOCK`、`num_warps`、`num_stages` | 在正确性检查通过后比较耗时和资源占用 |
| 求解器执行配置 | `graph_chunk` | 比较建图成本、复用后的耗时和显存占用 |
| 数值离散与精度 | `dz`、`dt`、`n_samples`、数据类型 | 根据稳定性、窗口长度、分辨率与收敛性确定 |
| 物理参数 | `alpha`、`beta_m`、`gamma0`、`omega0` | 由仿真对象决定 |

调 `dz` 时保持目标传播距离 `z=steps*dz` 一致；调时间分辨率时，区分加密网格与扩大时间窗口这两种实验。当前 GPU 算子要求 `complex64`，不能直接切换 dtype 来完成精度或性能对照。NumPy 基线与 GPU 采用相同的一阶推进方法，二者一致性检查之外仍需做步长收敛验证。

### 优先联合搜索 BLOCK 和 num_warps

当前算子中的 `BLOCK` 就是常说的 `BLOCK_SIZE`，均固定为 `256`，`num_warps` 尚未显式指定。`BLOCK` 表示每个 Triton program 处理的复数采样点或频点数，`num_warps` 表示该 program 使用的 warp 数；NVIDIA GPU 上每个 warp 有 32 个线程，线程数不必等于元素数。[Triton Config 说明](https://triton-lang.org/main/python-api/generated/triton.Config.html)

| 参数 | 建议初始候选 | 主要权衡 |
| --- | --- | --- |
| `BLOCK` | `64, 128, 256, 512` | program 数量、每个 program 的工作量、寄存器压力及尾部掩码开销 |
| `num_warps` | `1, 2, 4`；较大规模再考虑 `8` | 线程协作方式、寄存器资源与执行效率；应与 `BLOCK` 联合测量 |
| `num_stages` | 初期保持编译器默认值 | 当前 kernel 没有显式分块加载循环，暂不作为重点；软件流水线更常用于带循环的矩阵类 kernel |

例如 `N=1024` 时，`BLOCK=256` 只启动 4 个 program，`BLOCK=64` 则启动 16 个。较小的块可能提高设备利用率，较大的块可能提高单个 program 的效率；具体收益必须实测，不能按块大小推断。

NLSE 实际使用的以下四个 kernel 可以分别选择配置，不要求共用同一个 `BLOCK` 或 `num_warps`：

| Kernel | 职责 |
| --- | --- |
| `_disp_and_loss_kernel` | 色散/损耗的频域复数乘法 |
| `_kerr_cubic_kernel` | Kerr 相位旋转及三次非线性项生成 |
| `_shock_derivative_kernel` | 自陡峭项的频域求导 |
| `_shock_update_kernel` | 自陡峭场更新 |

修改块大小时，必须同步修改启动网格，使其使用 `triton.cdiv(N, BLOCK)`；目前启动网格中的 `256` 也是硬编码。如果不同 kernel 使用不同块大小，各自的启动网格也要匹配。验证应覆盖实际规模和 `N=257` 这样的非整除尺寸，确认尾部元素没有遗漏。

### 再搜索 CUDA Graph 的 chunk

`graph_chunk` 决定一次回放执行多少个完整传播步，不改变 `dz` 或数值更新公式。先测试 `1, 4, 8, 16, 32`，必要时扩大范围；`4` 或现有默认值 `16` 都可以作为初始基线。

- 较小的 chunk 建图较快，但需要更多次回放和状态写回。
- 较大的 chunk 减少回放次数，但图规模增大，通常需要更长建图时间，也可能占用更多显存。
- 当前实现对 `steps % chunk` 个剩余步骤普通执行。总共 1,000 步时，chunk 为 4 没有余数，为 64 则有 40 步走普通调用；应按实际常用的总步数测量。
- 只运行一次时，关注包含首次编译与建图的总耗时；同一参数反复运行时，关注缓存图后的耗时，并单独记录建图成本。

已有 chunk 扫描脚本可直接运行：

```bash
python benchmarks/compare_chunks.py \
  --samples 1024 2048 4096 \
  --chunks 1 4 8 16 32 \
  --steps 1000 \
  --repeats 7 \
  --output results/tuning_chunks_nlse.json
```

脚本分别记录预热后的建图时间、缓存图后的主机到主机耗时，以及 PyTorch 预留显存增量；该显存指标不包含分配器外的全部占用，也不等同于完整峰值显存。脚本使用 `examples/run_tod.py` 的示例参数和初态，实际应用的参数不同则需换成对应工况。仓库里的旧 LLE chunk 测量只能作为历史参考，不能直接认定为当前 NLSE 的最佳选择。

### 测量顺序与验收

1. **固定工况和误差要求。** 确定 GPU、物理参数、初始场、采样点数、传播距离及数据类型，先通过数值正确性和收敛性检查。
2. **分解耗时。** 测量 FFT/IFFT、逐点 kernel 以及主机提交等开销。开启自陡峭时，每步包含线性部分的一组 FFT/IFFT 和非线性求导的一组 FFT/IFFT；如果主要时间花在 FFT 上，仅调逐点 kernel 的收益会受到限制。
3. **联合搜索 kernel 配置。** 按实际输入规模分别比较 `BLOCK × num_warps`，每个候选先验证数值结果，再测量。改变配置后完成编译和预热，再捕获 CUDA Graph。
4. **扫描 graph_chunk。** 使用选定的 kernel 配置重新建图，比较常用传播步数下的总耗时。现有图缓存的键不含 kernel 调参配置；配置改变后需清空缓存或重启进程，避免回放旧图。
5. **验收完整算子与完整仿真。** 同时记录 eager 和 CUDA Graph 模式，分别比较相同模式；以完整 `kerr_shock_` 和完整 NLSE 的耗时与误差决定是否保留优化，不能只看某个 kernel 的加速比。

计时前完成编译、FFT 初始化和预热。使用 CUDA Events 测量设备执行，或在主机计时区间前后同步 GPU；不能只测异步调用返回所花的时间。各候选使用相同初态，交替或随机顺序重复测量，建议至少 7 次并报告中位数与波动。把系数准备、数据传输、输入恢复、编译和建图是否计入耗时写清楚；保存 GPU/软件版本、输入规模、`BLOCK`、`num_warps`、chunk、误差及源码版本。

这些算子会原地修改输入。采用 `triton.autotune` 时，必须在不同候选之间恢复输入，例如为实际输入参数指定 `restore_value`，或使用受控的输入恢复钩子；连续测试同一份已经被修改的场会破坏比较。自动调参应在 CUDA Graph 捕获之前完成。参见 [Triton autotune 文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)。

**当前工具覆盖范围：** `compare_chunks.py` 已支持 chunk 扫描；`benchmark_operators.py` 目前只支持 `kerr`、`kerr_raman`、`linear_and_pump`，尚未支持 `disp_and_loss`、`kerr_shock`，也没有 `BLOCK`/`num_warps` 搜索入口。开展 kernel 调优前，应先补齐这两个算子的性能/精度基线和配置入口。上面的 chunk 命令使用现有 kernel 配置，不会自动搜索块大小。

完成配置选择后，可以用以下命令检查完整 NLSE 的 PyTorch/Triton 对照；其中 `4` 替换为实际选中的 chunk：

```bash
python benchmarks/compare_implementations.py \
  --samples 1024 2048 4096 --steps 1000 \
  --graph-chunk 4 --repeats 7 \
  --output results/tuning_implementations_nlse.json
```

## 代码结构

```text
src/flashlle/
├── __init__.py          轻量入口
├── operators/           独立算子及张量检查
└── solvers/
    ├── __init__.py      CPU 参数、网格和参考求解器入口
    ├── params.py        NLSEParams、时间/频率网格、线性因子
    ├── nlse_gpu_opt.py  GPU 单步组合、传播循环与 CUDA Graph
    ├── nlse_numpy_baseline.py  NumPy NLSE 基线实现
    └── _validation.py  主机输入与执行参数检查
examples/run_tod.py      初始脉冲与三阶色散示例
benchmarks/             独立算子和 NLSE 对照
tests/                 数值、收敛性和执行行为测试
```

`LLEParams` 和 `flashlle.solvers.lle` 已移除，请按上面的 NLSE 接口传入物理参数；旧 LLE 参数不能直接当作 NLSE 参数使用。

## 历史性能结果

以下数据来自保留在 `results/` 中的原始算子测量与旧 LLE 仿真。完整 LLE 的加速比不代表当前 NLSE 求解器性能；旧 JSON 和源码哈希保留原样。

在 GTX 1660 上，相同 GPU、`complex64` 精度且两边均启用 CUDA Graph 时，**Kerr 算子加速 9.16–9.35×，线性/泵浦算子加速 2.29–3.34×，Kerr–Raman 算子加速 2.16–2.73×，完整仿真加速 1.88–2.32×**。

以下为历史版本的实测结果：PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA `12.6`；预热后交替测量四种实现，取 **7 次中位数**。PyTorch 基线使用普通张量运算，未使用 `torch.compile`。

### 算子性能汇总

两边均使用 CUDA Graph，每次回放 64 次算子调用；每轮共 1,024 次调用。表中为摊销单次耗时，单位 **µs**。

| 算子 | 元素数 | PyTorch + Graph | Triton + Graph | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| Kerr | 1024 | 13.377 | 1.460 | **9.16×** |
| 线性 + 泵浦 | 1024 | 3.087 | 1.237 | **2.50×** |
| Kerr–Raman | 1024 | 30.973 | 12.441 | **2.49×** |
| Kerr | 2048 | 13.441 | 1.463 | **9.19×** |
| 线性 + 泵浦 | 2048 | 4.249 | 1.273 | **3.34×** |
| Kerr–Raman | 2048 | 49.206 | 18.002 | **2.73×** |
| Kerr | 4096 | 21.956 | 2.349 | **9.35×** |
| 线性 + 泵浦 | 4096 | 4.827 | 2.110 | **2.29×** |
| Kerr–Raman | 4096 | 44.039 | 20.347 | **2.16×** |

计时包含主机提交与 GPU 执行，排除传输、输入恢复、编译、预热和建图。Kerr–Raman 包含强度计算、RFFT/IRFFT、谱求导及相位更新，使用固定平滑周期初态，`h=0.005`、`raman=0.01`；其余两种算子使用随机复数初态。数据来自两轮独立测量，每行比较同一算子的 PyTorch/Triton 实现。

普通调用（eager）下，线性/泵浦此次未获得加速，Kerr–Raman 加速 **1.29–1.65×**。连续调用的最大复场相对 L2 误差：Kerr 与线性/泵浦为 `1.11e-4`，Kerr–Raman 为 `9.64e-4`，均通过 `1e-3` 门槛及 Graph/eager 一致性检查。[Kerr、线性/泵浦数据](results/benchmark_operators_current.json) · [曲线](results/benchmark_operators_current.png)；[Kerr–Raman 数据](results/benchmark_kerr_raman.json) · [曲线](results/benchmark_kerr_raman.png)。

### 完整 LLE 仿真

每条轨迹 **1,000 步**，Graph 每次回放 16 步；相同初态、系数与 cuFFT，单位 **ms**。

| 模态数 | PyTorch eager | Triton eager | PyTorch + Graph | Triton + Graph | 两边均用 Graph 时加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 254.804 | 211.950 | 23.992 | 10.334 | **2.32×** |
| 2048 | 260.071 | 194.593 | 37.781 | 17.058 | **2.21×** |
| 4096 | 251.844 | 215.169 | 34.543 | 18.390 | **1.88×** |

计时包含输入状态复制、完整循环、输出复制及同步，排除主机传输、系数准备、编译、预热和建图。Triton + Graph 相对 PyTorch eager 加速 **13.69–24.66×**，其中同时包含算子优化与图回放的收益。

完整循环的 PyTorch/Triton 及 Graph/eager 对照相对 L2 误差不超过 `3.03e-6`。[完整循环数据与测量配置](results/compare_implementations_current.json)记录了逐次耗时、误差及源码哈希。结果限于上述设备与规模；性能结论分别对应所链接的实测文件。
