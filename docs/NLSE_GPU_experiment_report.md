# NLSE GPU 性能实验报告

## 1. 实验概述

本实验比较 `flashlle` 中三种 NLSE 实现的数值结果和运行性能：

1. NumPy `complex128` 参考实现，用于 CPU 端参考和收敛性检查；
2. 普通 PyTorch CUDA `complex64` 实现，作为 GPU baseline；
3. Triton + cuFFT + CUDA Graph 的优化 GPU 实现。

实验重点回答以下问题：

- Triton 算子和 CUDA Graph 相对于普通 PyTorch GPU 实现的收益；
- `BLOCK`、`num_warps` 和 `graph_chunk` 的合适配置；
- Triton 实现与 PyTorch baseline、NumPy 参考结果的一致性；
- `complex64` 长距离传播的累计误差以及步长收敛趋势。

实验在 L40S-GPU 服务器的 GPU-6 上完成。原始 JSON、CSV、Markdown 汇总和日志保存在 [`benchmarks/NLSE_bench/results/L40s/`](../benchmarks/NLSE_bench/results/L40s/)；本报告中的数值均由这些原始结果整理而来。

## 2. 结论摘要

在本次 L40S 上，四个主规模共同选择出的配置是：

```text
BLOCK = 128
num_warps = 4
graph_chunk = 256
```

在 4096 步、目标传播距离 `z=4.096` 的稳态端到端测试中，优化实现相对于普通 PyTorch GPU baseline 的加速比为 `8.52×–29.73×`，四个规模的几何平均加速比为约 `13.76×`。相对于 NumPy CPU 端到端时间，优化实现的加速比为 `8.61×–26.80×`。

CUDA Graph 的收益在 GPU 常驻 kernel 测量中更容易观察。固定 `BLOCK=128`、`num_warps=4` 和 `graph_chunk=16` 时，Graph 相对于 Triton eager 的加速比为 `7.52×–15.86×`，几何平均约 `11.28×`。将 `graph_chunk` 从 `1` 增加到 `256` 后，4096 步稳态时间下降约 `5.58%–10.94%`。

短步数、非整除长度的尾部 mask 和所有有限值检查均通过。长距离传播的严格场误差阈值没有全部通过：25 个记录的正确性用例中有 19 个通过、6 个未通过，未通过项全部是 4096 步传播。GPU 输出之间彼此接近，但相对于 NumPy `complex128` 的场相对 L2 误差达到约 `1.34e-3–2.18e-3`，超过预设的 `1e-3` 上限。这一结果应在使用优化实现时保留，不能标记为完全满足严格精度验收。

## 3. 实验环境和源码版本

### 3.1 硬件和容器

| 项目 | 设置 |
| --- | --- |
| 服务器 | `L40s-GPU` |
| GPU | 宿主机 GPU-6，NVIDIA L40S |
| GPU UUID | `GPU-67eb355c-16df-6023-b22c-6a831c2bea24` |
| 容器内设备 | `cuda:0` |
| 计算能力 | `8.9` |
| 显存 | 47,667,740,672 bytes，约 44.4 GiB |
| Docker 镜像 | `flash-ssft-nlse-bench:20261110` |
| 基础镜像 | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` |

### 3.2 软件版本

| 软件 | 版本 |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| Triton | 3.6.0 |
| NumPy | 2.4.3 |
| pytest | 9.2.0 |

远程结果文件记录了 Linux 内核平台、软件版本、GPU 型号和显存信息。实验使用的本地源码 commit 为 `9d60f65`（`Add configurable NLSE GPU benchmark harness`）；结果元数据同时保存了参与实验的源文件 SHA-256，便于在没有 Git commit 的容器环境中复核源码版本。

服务器 Docker 环境中的完整测试结果为 `190 passed`。本地 CPU 测试结果为 `53 passed, 68 skipped`；跳过项是本地没有 CUDA/Triton 运行环境的 GPU 测试。

## 4. 方程、离散和实现路径

三种实现使用相同的一阶 Lie splitting 顺序：

```text
时域 A
  -> FFT
  -> 乘以线性传播因子 L(w)
  -> IFFT
  -> Kerr 相位旋转
  -> 计算 |A|^2 A 的频域导数
  -> 显式 Euler 自陡峭更新
```

其中线性传播因子为

\[
L(w)=\exp\left[\Delta z\left(-\frac{\alpha}{2}
 + i\,D(w)\right)\right],
\]

Kerr 和自陡峭项分别由 `kerr_shock_` 完成。NumPy 参考实现使用 `complex128`；PyTorch baseline 和 Triton 优化路径使用 `complex64`。GPU 优化路径使用 PyTorch/cuFFT 执行 FFT/IFFT，使用 Triton 执行色散/损耗、Kerr 和自陡峭相关逐点算子，并使用 CUDA Graph 回放固定数量的完整 NLSE 步。

本轮代码准备包括：

- 为 Triton kernel 显式暴露 `block` 和 `num_warps`；
- 让 `nlse_step()` 和 `simulate_gpu()` 接收这两个配置；
- 将 kernel 配置加入 CUDA Graph 缓存 key；
- 为 PyTorch baseline 增加 GPU 常驻的 `pytorch_nlse_step()`；
- 恢复测试需要的 NumPy/PyTorch reference 模块；
- 增加 `benchmarks/NLSE_bench/run_experiment.py`，统一执行正确性、kernel 搜索、chunk 搜索、最终性能和收敛检查。

## 5. 固定实验参数

主性能实验的配置如下：

| 参数 | 值 |
| --- | --- |
| 模式数 `N` | `1024, 2048, 4096, 8192` |
| 步数 | `4096` |
| `dz` | `0.001` |
| 目标传播距离 | `4.096` |
| 时间窗口 | `20.0` |
| `dt` | `20.0 / N` |
| `alpha` | `0` |
| `beta2` | `-1` |
| `beta3` | `0.05` |
| `beta4` | `0` |
| `gamma0` | `1` |
| `omega0` | `20` |
| 初始场 | `sech` 脉冲，即 `1/cosh(t)` |
| 数据类型 | NumPy `complex128`；GPU `complex64` |

正确性额外覆盖了 `N=257、1025、4097`，用于检查 Triton block 尾部 mask。预设阈值为：短步数场相对 L2 误差不超过 `1e-5`，4096 步场相对 L2 误差不超过 `1e-3`。

## 6. Kernel 和 Graph 配置搜索

### 6.1 `BLOCK × num_warps`

第一阶段固定 `graph_chunk=16`，搜索：

```text
BLOCK      = 64, 128, 256, 512
num_warps = 1, 2, 4, 8
```

四个主规模共记录 64 个候选，全部完成编译、短步数正确性检查和正式计时，没有候选失败。全局选择使用四个主规模稳态时间的几何平均，结果为：

| 选择范围 | `BLOCK` | `num_warps` | 几何平均稳态时间 |
| --- | ---: | ---: | ---: |
| 四个主规模统一配置 | 128 | 4 | `0.073184 s` |

每个规模单独选择时，最佳配置有所不同：

| `N` | 单独最佳 `BLOCK` | 单独最佳 `num_warps` | 稳态时间 |
| ---: | ---: | ---: | ---: |
| 1024 | 64 | 4 | `0.051957 s` |
| 2048 | 64 | 4 | `0.065977 s` |
| 4096 | 128 | 4 | `0.076039 s` |
| 8192 | 256 | 8 | `0.109271 s` |

因此，如果应用场景固定为某一个 `N`，可以继续使用每个规模的独立最优配置；如果需要一个覆盖不同规模的统一配置，本实验选择 `128/4`，以降低配置管理和 Graph 缓存隔离的复杂度。

### 6.2 CUDA Graph 的 eager 对比

下表来自 kernel 搜索中选定的 `BLOCK=128`、`num_warps=4`、`graph_chunk=16` 候选。两列都在 GPU 常驻状态上计时，计时区间不包含主机传输；eager 逐步调用 Triton step，Graph 则回放固定长度的 CUDA Graph。

| `N` | Triton eager | Triton Graph | Graph 加速比 |
| ---: | ---: | ---: | ---: |
| 1024 | `0.825912 s` | `0.052068 s` | `15.86×` |
| 2048 | `0.827397 s` | `0.066161 s` | `12.51×` |
| 4096 | `0.824339 s` | `0.076039 s` | `10.84×` |
| 8192 | `0.823812 s` | `0.109512 s` | `7.52×` |

几何平均加速比约为 `11.28×`。这里的主要收益来自减少每个 NLSE step 的 Python/host kernel launch 开销；FFT 和 Triton kernel 本身仍在 Graph 中执行。

### 6.3 `graph_chunk` 搜索

第二阶段固定 `BLOCK=128`、`num_warps=4`，搜索：

```text
graph_chunk = 1, 2, 4, 8, 16, 32, 64, 128, 256
```

对每个主规模分别测试 `steps=4096` 和 `steps=4097`，共 72 个候选，全部成功。两个步数下四个规模的最佳 chunk 都是 `256`。与 `graph_chunk=1` 相比，4096 步稳态时间变化如下：

| `N` | `chunk=1` | `chunk=256` | 时间下降 |
| ---: | ---: | ---: | ---: |
| 1024 | `0.058072 s` | `0.051721 s` | `10.94%` |
| 2048 | `0.072828 s` | `0.066034 s` | `9.33%` |
| 4096 | `0.081598 s` | `0.075180 s` | `7.87%` |
| 8192 | `0.115600 s` | `0.109146 s` | `5.58%` |

最终配置采用 `graph_chunk=256`。当 `steps=4097` 时，Graph 执行完整的 256 步 chunk，最后 1 步走普通 step；该余数路径在 chunk 搜索中执行成功。

## 7. 最终性能结果

最终计时使用 `BLOCK=128`、`num_warps=4`、`graph_chunk=256`，每个候选预热 2 次、正式计时 7 次，表中稳态值为 7 次结果的中位数。

`NumPy` 和 `PyTorch` 时间是公开 solver 接口的端到端时间。PyTorch baseline 的时间包含从主机 NumPy 输入创建 GPU tensor 以及最终复制回主机的过程。优化实现的首次调用时间包含清空 Graph 缓存后的编译、系数准备、Graph 建图、输入传输和输出传输；稳态时间使用已经缓存的 Graph，但每次 `simulate_gpu()` 调用仍然包含输入恢复、主机到 GPU、GPU 到主机以及同步开销。

| `N` | NumPy CPU | PyTorch GPU 端到端 | 优化首次调用 | 优化稳态 | 优化 / PyTorch | 优化 / NumPy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | `0.4467 s` | `1.5418 s` | `2.0757 s` | `0.05187 s` | `29.73×` | `8.61×` |
| 2048 | `0.7883 s` | `0.8388 s` | `0.4377 s` | `0.06607 s` | `12.69×` | `11.93×` |
| 4096 | `1.4555 s` | `0.8417 s` | `0.4355 s` | `0.07539 s` | `11.16×` | `19.31×` |
| 8192 | `2.9303 s` | `0.9312 s` | `0.4835 s` | `0.10934 s` | `8.52×` | `26.80×` |

四个规模的几何平均加速比为：

- 优化稳态相对于 PyTorch GPU：`13.76×`；
- 优化稳态相对于 NumPy CPU：`15.18×`。

优化首次调用在小规模 `N=1024` 上为 `2.08 s`，高于稳态时间约 40 倍，主要反映首次编译和 Graph 建图成本。对于重复运行同一组参数的场景，稳态结果更能代表实际吞吐；只运行一次时，应单独计入首次成本。

### 7.1 Kernel 编译和 Graph 建图

在选定 kernel 配置的搜索记录中，首次 Triton 编译时间约为 `0.30–0.31 s`。在 kernel 搜索固定的 `graph_chunk=16` 下，Graph 建图约为 `0.009–0.011 s`；在最终的 `graph_chunk=256` 配置下，建图约为 `0.053–0.056 s`。这些时间没有混入稳态中位数，但包含在最终阶段的首次调用时间内。

### 7.2 显存记录

最终阶段记录到的当前 CUDA allocator 状态如下：

| `N` | 已分配显存 | allocator 已保留显存 |
| ---: | ---: | ---: |
| 1024 | `0.087 MiB` | `8 MiB` |
| 2048 | `0.173 MiB` | `10 MiB` |
| 4096 | `0.345 MiB` | `10 MiB` |
| 8192 | `0.689 MiB` | `10 MiB` |

该记录来自 PyTorch allocator 的 `memory_allocated` 和 `memory_reserved`，用于比较本次运行中的分配状态，不等同于整张 GPU 的峰值显存占用。

## 8. 正确性结果

### 8.1 用例统计

正确性阶段共记录 25 个用例：

- 主规模 `N=1024、2048、4096、8192`：测试 `steps=0、1、16、4096`；
- 尾部 mask 规模 `N=257、1025、4097`：测试 `steps=1、16、4096`；
- 所有输出检查 shape、dtype、有限值、输入保护、能量、平均功率、功率谱和场相对 L2 误差。

结果为 `19` 个通过、`6` 个未通过。短步数以及非整除规模的 `steps=1、16` 均通过，说明 Triton block 尾部 mask 和短程推进路径工作正常。

### 8.2 长距离误差

下面列出 4096 步相对于 NumPy `complex128` 参考的场相对 L2 误差：

| `N` | PyTorch `complex64` | Triton/Graph `complex64` | Triton 对 PyTorch 相对 L2 |
| ---: | ---: | ---: | ---: |
| 1024 | `1.6533e-3` | `1.6407e-3` | `2.03e-5` |
| 2048 | `1.6925e-3` | `1.6817e-3` | `1.96e-5` |
| 4096 | `1.5320e-3` | `1.5246e-3` | `1.92e-5` |
| 8192 | `1.4420e-3` | `1.4672e-3` | `5.79e-4` |
| 257 | `2.1667e-3` | `2.1785e-3` | `2.24e-5` |
| 1025 | `2.2698e-4` | `2.2984e-4` | `2.21e-5` |
| 4097 | `1.3369e-3` | `1.3357e-3` | `2.56e-5` |

对于 `N<=4096`，Triton 和 PyTorch 之间的差异约为 `2e-5`；`N=8192` 时为 `5.79e-4`，仍明显小于两者各自相对于 NumPy 的误差。这个结果说明长距离误差主要由 GPU `complex64` 在 4096 次推进中的累计误差造成，而不是 Triton 与 PyTorch 使用了不同的单步公式。

不过，`N=257、1024、2048、4096、8192、4097` 的严格场误差都超过了 `1e-3`，所以这些用例按照预先固定的验收规则必须记录为未通过。当前实验没有通过放宽阈值来修改结果。

### 8.3 非整除规模和输入保护

`N=257、1025、4097` 会产生未填满的 Triton block。它们的单步和 16 步结果均通过，输出 shape 正确，输出为 `complex64`，且没有 NaN/Inf。所有正确性用例还检查了输入初始场没有被 solver 原地修改。

## 9. 步长收敛性

收敛实验保持目标传播距离 `z=4.096` 不变，在 `N=4096` 上比较三种 `dz`。参考解取最小步长 `dz=0.00025` 的结果。

| `dz` | 步数 | 目标 `z` | 相对最小步长解的场 L2 | 能量 |
| ---: | ---: | ---: | ---: | ---: |
| `0.001` | 4096 | 4.096 | `2.7737e-4` | `2.00001403` |
| `0.0005` | 8192 | 4.096 | `9.2417e-5` | `2.00000701` |
| `0.00025` | 16384 | 4.096 | `0` | `2.00000350` |

减小 `dz` 后场误差单调下降，能量也逐渐接近最小步长结果，说明长距离误差至少包含明显的时间步长离散误差。该收敛实验当前实际只覆盖 `N=4096`；原计划中“至少一个中等规模和一个大规模”的第二个规模尚未运行，因此不能把收敛结论直接推广到所有主规模。

## 10. 结果解释和限制

1. **统一配置与逐规模配置不同。** `N=1024/2048` 更适合较小的 `BLOCK`，`N=8192` 的单独最佳配置使用 `BLOCK=256、num_warps=8`。本报告选择 `128/4` 是四个规模的统一折中，而不是每个规模的局部最优。
2. **稳态端到端时间仍包含传输。** 最终表中的优化稳态来自 `simulate_gpu()`，每次调用都会把主机输入传到 GPU，并把结果复制回主机。GPU 常驻的 kernel 级 Graph/eager 对比单独记录在第 6.2 节，但本轮最终结果没有把 PyTorch 和优化 solver 的 GPU 常驻单步接口做成独立的最终汇总表。
3. **首次调用和稳态差异很大。** 首次调用包含 Triton 编译、FFT 初始化、系数准备和 CUDA Graph 建图，不能用稳态时间代表一次性任务成本。
4. **严格精度验收未完全通过。** GPU 路径使用 `complex64`，4096 步累计误差超过 `1e-3`。如果应用要求严格复现 NumPy `complex128`，需要进一步评估更高精度实现、分段重正交/校正或更严格的步长策略；本轮没有通过修改阈值掩盖这一问题。
5. **收敛实验覆盖不足。** 当前收敛结果只在 `N=4096` 上完成，后续若要完全按原实验计划验收，应再增加一个大规模（例如 `N=8192`）的固定传播距离收敛实验。
6. **结果依赖硬件和软件环境。** `BLOCK`、`num_warps`、Graph chunk 和加速比只对本次 L40S、PyTorch 2.11、CUDA 12.8、Triton 3.6 环境有效，不能直接外推到其他 GPU。

## 11. 原始结果和复现实验

主要结果文件如下：

- [`correctness.json`](../benchmarks/NLSE_bench/results/L40s/correctness.json)：正确性原始记录；
- [`kernel_search.json`](../benchmarks/NLSE_bench/results/L40s/kernel_search.json)：64 组 `BLOCK × num_warps` 搜索；
- [`chunk_search.json`](../benchmarks/NLSE_bench/results/L40s/chunk_search.json)：72 组 `graph_chunk` 搜索；
- [`final.json`](../benchmarks/NLSE_bench/results/L40s/final.json)：四个主规模最终性能结果；
- [`convergence.json`](../benchmarks/NLSE_bench/results/L40s/convergence.json)：步长收敛结果；
- [`correctness.log`](../benchmarks/NLSE_bench/results/L40s/correctness.log) 和 [`kernel_search.log`](../benchmarks/NLSE_bench/results/L40s/kernel_search.log)：实验日志。

复现实验需要在带 CUDA 和 Triton 的环境中，从仓库根目录执行。例如最终阶段使用：

```bash
python benchmarks/NLSE_bench/run_experiment.py \
  --phase final \
  --output-dir benchmarks/NLSE_bench/results \
  --steps 4096 \
  --repeats 7 \
  --warmups 2
```

运行 kernel 搜索和 chunk 搜索前，应先完成正确性阶段，并让后续阶段读取同一结果目录中的配置选择文件。实验结果默认只保存到 `benchmarks/NLSE_bench/results/`，没有自动提交到 Git。
