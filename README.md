# flash-ssft

通过 **GPU 算子开发与调优加速非线性光学 LLE 仿真**，采用分步傅里叶法，结合 Triton、cuFFT 与 CUDA Graph，以数值精度和完整计算耗时评估优化效果。

## 性能结果

在 GTX 1660 上，相同 GPU、`complex64` 精度且两边均启用 CUDA Graph 时，**Kerr 算子加速 9.16–9.35×，线性/泵浦算子加速 2.29–3.34×，Kerr–Raman 算子加速 2.16–2.73×，完整仿真加速 1.88–2.32×**。

以下为当前实现的实测结果：PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA `12.6`；预热后交替测量四种实现，取 **7 次中位数**。PyTorch 基线使用普通张量运算，未使用 `torch.compile`。

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

## 优化实现

以相同算法、精度和执行方式下的完整算子或仿真耗时衡量收益。算子融合、访存优化与图回放是实现手段；是否采用由实测性能和误差共同决定。

- **Kerr 融合：**在一个 Triton kernel 内完成模平方、相位计算、sin/cos 与复数乘法，寄存器内计算并原地写回。
- **线性/泵浦融合：**将频域复数乘法和零模泵浦注入合并，传播系数在循环外预计算。
- **Kerr–Raman 谱方法：**三个 Triton kernel 分别计算强度、频谱乘 `ik` 和联合相位更新，中间调用 cuFFT 的 RFFT/IRFFT；性能对照计入整个算子。
- **CUDA Graph：**捕获多步迭代并缓存回放，减少 Python/CUDA 提交开销；FFT/IFFT 保持独立调用。

Kerr–Raman 后续调优先测量各阶段耗时以定位瓶颈，再用完整算子的耗时与数值误差验收；当前结果证明相对基线的收益，尚未证明该实现最优。

每步接收并返回时域场，采用一阶 Lie 分裂：

```text
时域场 → FFT → 线性演化 + 泵浦 → IFFT → Kerr 非线性 → 下一步时域场
```

线性因子为 `exp(λh)`，`λ = -1 - i(α + d₂μ²/2 + d₃μ³/6)`；零模泵浦增量为 `N·F·h`，Kerr 更新为 `ψ *= exp(i·h·|ψ|²)`。线性与泵浦联合推进采用一阶近似。

## 快速开始

需要 Python ≥ 3.10、NVIDIA GPU 及支持 CUDA 的 PyTorch。

```bash
python -m pip install -e '.[dev,benchmark]'
```

```python
from flashlle import LLEParams, initial_field, simulate_gpu

params = LLEParams(n_modes=1024, pump_f=2.5, detuning_alpha=5.0, h=0.005)
final = simulate_gpu(initial_field(params), params, steps=1000, graph_chunk=16)
```

输入和输出为 NumPy 时域复数数组，GPU 内部使用 `complex64`。正步数自动使用 CUDA Graph，末尾不足一块的步骤直接执行；首次调用包含编译和建图。`simulate_numpy` 提供数值参考，`clear_graph_cache()` 可释放缓存图。

### 独立 Kerr–Raman 算子

`kerr_raman_(angular, h, raman)` 在长度为 `2π` 的周期角度网格上原地计算 `ψ *= exp(i*h*(I - raman*∂θI))`，其中 `I = |ψ|²`。`raman` 是无量纲系数；为零时直接调用 `kerr_`。

```python
import torch
from flashlle.operators import kerr_raman_

angular = torch.as_tensor(initial_field(params), dtype=torch.complex64, device="cuda")
kerr_raman_(angular, h=0.005, raman=0.01)
```

输入须为非空、连续的一维 CUDA `complex64` 张量，返回原张量。强度经 RFFT → `ik` 乘法 → IRFFT 求导，偶数长度的 Nyquist 导数置零；最后融合 Kerr/Raman 相位更新。支持 CUDA Graph，目前作为独立算子提供，`simulate_gpu` 仍使用原 Kerr 算子。

## 复现与测试

```bash
python benchmarks/benchmark_operators.py --sizes 1024 2048 4096 --repeats 7 --output results/benchmark_operators_current
python benchmarks/compare_implementations.py --modes 1024 2048 4096 --steps 1000 --repeats 7 --graph-chunk 16 --output results/compare_implementations_current.json
python benchmarks/benchmark_operators.py --operators kerr_raman --sizes 1024 2048 4096 --repeats 7 --raman 0.01 --output results/benchmark_kerr_raman
python -m pytest
```

测试覆盖算子公式、原地更新、边界尺寸、完整时间步、图分块和缓存复用；Kerr–Raman 另检查解析导数、奇偶最高频率、强度保持与更换输入后的图回放。`compare_chunks.py` 用于评估图分块大小的耗时与显存；`compare.py` 用于 NumPy/GPU 主机输入到主机输出的整体对比。

## 代码结构

```text
src/flashlle/
├── operators/       Triton 融合算子及张量检查
├── gpu.py           参数、系数、单步组合与 CUDA Graph 执行
├── reference.py     NumPy 数值参考
├── model.py         初始光场
└── _validation.py   主机输入检查
benchmarks/          算子、完整循环与图分块对照
tests/              数值与执行行为测试
```
