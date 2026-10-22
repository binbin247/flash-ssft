# flash-ssft

通过 **Triton 算子融合与 CUDA Graph 加速非线性光学 LLE 仿真**，采用分步傅里叶法，FFT/IFFT 使用 cuFFT。

## 性能结果

在 GTX 1660 上，相同 GPU、`complex64` 精度且两边均启用 CUDA Graph 时，**Kerr 算子加速 9.16–9.35×，线性/泵浦算子加速 2.29–3.34×，完整仿真加速 1.88–2.32×**。

以下为当前实现的实测结果：PyTorch `2.13.0+cu126`、Triton `3.7.1`、CUDA `12.6`；预热后交替测量四种实现，取 **7 次中位数**。PyTorch 基线使用普通张量运算，未使用 `torch.compile`。

### 融合算子

两边均使用 CUDA Graph，每次回放 64 次算子调用；每轮共 1,024 次调用。表中为摊销单次耗时，单位 **µs**。

| 算子 | 元素数 | PyTorch + Graph | Triton + Graph | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| Kerr | 1024 | 13.377 | 1.460 | **9.16×** |
| 线性 + 泵浦 | 1024 | 3.087 | 1.237 | **2.50×** |
| Kerr | 2048 | 13.441 | 1.463 | **9.19×** |
| 线性 + 泵浦 | 2048 | 4.249 | 1.273 | **3.34×** |
| Kerr | 4096 | 21.956 | 2.349 | **9.35×** |
| 线性 + 泵浦 | 4096 | 4.827 | 2.110 | **2.29×** |

计时包含主机提交与 GPU 执行，排除传输、输入恢复、编译、预热和建图。普通调用（eager）下，线性/泵浦算子此次测量未获得加速；四组完整数据及逐次样本见 [算子结果](results/benchmark_operators_current.json)，[性能曲线](results/benchmark_operators_current.png)。

### 完整 LLE 仿真

每条轨迹 **1,000 步**，Graph 每次回放 16 步；相同初态、系数与 cuFFT，单位 **ms**。

| 模态数 | PyTorch eager | Triton eager | PyTorch + Graph | Triton + Graph | 两边均用 Graph 时加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 254.804 | 211.950 | 23.992 | 10.334 | **2.32×** |
| 2048 | 260.071 | 194.593 | 37.781 | 17.058 | **2.21×** |
| 4096 | 251.844 | 215.169 | 34.543 | 18.390 | **1.88×** |

计时包含输入状态复制、完整循环、输出复制及同步，排除主机传输、系数准备、编译、预热和建图。Triton + Graph 相对 PyTorch eager 加速 **13.69–24.66×**，其中同时包含算子优化与图回放的收益。

所有数值检查通过：算子连续调用相对解析公式的复场 L2 误差不超过 `1.11e-4`；完整循环的 PyTorch/Triton 及 Graph/eager 对照相对 L2 误差不超过 `3.03e-6`。[完整循环数据与测量配置](results/compare_implementations_current.json)记录了逐次耗时、误差及源码哈希。结果限于上述设备与规模；其他 `results/` 文件为历史实验，不用于本页性能结论。

## 优化实现

- **Kerr 融合：**在一个 Triton kernel 内完成模平方、相位计算、sin/cos 与复数乘法，寄存器内计算并原地写回。
- **线性/泵浦融合：**将频域复数乘法和零模泵浦注入合并，传播系数在循环外预计算。
- **CUDA Graph：**捕获多步迭代并缓存回放，减少 Python/CUDA 提交开销；FFT/IFFT 保持独立调用。

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

## 复现与测试

```bash
python benchmarks/benchmark_operators.py --sizes 1024 2048 4096 --repeats 7 --output results/benchmark_operators_current
python benchmarks/compare_implementations.py --modes 1024 2048 4096 --steps 1000 --repeats 7 --graph-chunk 16 --output results/compare_implementations_current.json
python -m pytest
```

测试覆盖算子公式、原地更新、边界尺寸、完整时间步、图分块和缓存复用。`compare_chunks.py` 用于评估图分块大小的耗时与显存；`compare.py` 用于 NumPy/GPU 主机输入到主机输出的整体对比。

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
