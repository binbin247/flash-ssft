# NLSE 算子优化实验计划

## 1. 实验目标

验证 `src/flashlle` 中 NLSE 算子优化对完整仿真性能的影响，并确认优化前后计算结果一致。比较以下三种实现：

1. `src/flashlle/solvers/nlse_numpy_baseline.py`
   - NumPy `complex128` 参考实现。
   - 在 CPU 上执行普通 FFT、逐点 Kerr 更新和频域自陡峭更新。
   - 主要用于结果参考和收敛性检查，不作为 GPU 算子速度的直接参照。
2. `src/flashlle/solvers/nlse_pytorch_baseline.py`
   - CUDA 上的普通 PyTorch 实现，使用 `complex64`。
   - 使用 PyTorch FFT 和逐点运算，不使用 Triton 和 CUDA Graph。
   - 作为 GPU 端的未优化基线。
3. `src/flashlle/solvers/nlse_gpu_opt.py`
   - 使用 Triton 的 `disp_and_loss_` 和 `kerr_shock_`。
   - 使用 cuFFT 完成 FFT/IFFT。
   - 使用 CUDA Graph 将多个 NLSE 推进步录制为一个可回放的 chunk。

实验最终需要回答：

- Triton 算子相对于普通 PyTorch 实现的加速比是多少？
- CUDA Graph 在重复运行时带来的收益是多少？
- 最合适的 `BLOCK`、`num_warps` 和 `graph_chunk` 是否随模式数变化？
- 优化是否在所有测试规模和传播步数下保持数值正确？

## 2. 核心计算路径

三种实现必须使用相同的离散公式和推进顺序：

```text
时域 A
  → FFT
  → 乘以线性传播因子 L(w)
  → IFFT
  → Kerr: B = A * exp(i * h * |A|²)
  → cubic = |B|² * B
  → FFT(cubic)
  → 乘以 i*w/omega0
  → IFFT
  → A_new = B - h * derivative
```

其中：

- `h = gamma0 * dz`；
- `L(w) = exp(dz * (-alpha/2 + i*dispersion(w)))`；
- `w = 2*pi*fftfreq(n_samples, dt)`；
- 自陡峭使用显式 Euler 更新；
- 整体是线性部分先执行、非线性部分后执行的一阶 Lie 分裂。

`nlse_gpu_opt.py` 中的 CUDA Graph 只改变执行方式，不得改变上述单步公式。Graph 每次回放固定执行 `graph_chunk` 个完整 NLSE 步；当 `steps` 不能被 `graph_chunk` 整除时，余数由普通 `step` 调用执行。

## 3. 固定实验问题

主性能实验固定以下条件：

| 项目 | 设置 |
| --- | --- |
| 模式数 `N` | `2**10`、`2**11`、`2**12`、`2**13` |
| 推进步数 | `4096` |
| `dz` | `0.001` |
| 目标传播距离 | `z = 4096 * dz = 4.096` |
| 时间窗口 | `20.0` |
| `dt` | `20.0 / N`，随 `N` 变化 |
| `alpha` | `0.0` |
| `beta2` | `-1.0` |
| `beta3` | `0.05` |
| `beta4` | `0.0` |
| `gamma0` | `1.0` |
| `omega0` | `20.0` |
| 初始场 | `A(0,T) = sech(T)` |
| 边界条件 | 周期边界，使用未 shift 的 FFT |

参数应直接复用 `examples/run_tod.py` 中的 `example_params()` 和 `initial_field()`，避免 benchmark 与项目示例使用不同物理问题。每个 `N` 使用相同长度的时间窗口和相同的初始脉冲，只有采样间隔变化。

性能实验还需要记录：

- GPU 型号、计算能力、显存大小；
- Python、PyTorch、CUDA、Triton、NumPy 版本；
- 操作系统和驱动版本；
- 当前 Git commit 或源码版本；
- 是否包含输入传输、输出传输、系数准备、Triton 编译和 CUDA Graph 建图时间。

## 4. 先完成源代码准备

当前 `src` 中的 Triton kernel 将 `BLOCK=256` 直接写在启动位置，例如：

- `operators/disp_and_loss.py` 的 `_disp_and_loss_kernel`；
- `operators/kerr_shock.py` 的 `_kerr_cubic_kernel`；
- `operators/kerr_shock.py` 的 `_shock_derivative_kernel`；
- `operators/kerr_shock.py` 的 `_shock_update_kernel`。

在开始调参前，需要将这些固定值改为可配置的编译期参数，并让启动网格使用对应的 `triton.cdiv(N, BLOCK)`。同时把 `num_warps` 作为 Triton kernel launch 的可调参数传入。

建议先支持一组统一的算子配置：

```text
kernel_config = {
    "BLOCK": ...,
    "num_warps": ...,
}
```

如果统一配置已经足够，再进一步允许四个 kernel 分别配置。`disp_and_loss_` 和 `kerr_shock_` 的公共接口应保持兼容，未指定配置时继续使用当前默认值，避免影响已有调用。

调参配置必须进入 CUDA Graph 缓存的区分条件，或者每个候选配置使用独立进程。当前 `_GRAPH_CACHE` 的 key 包含物理参数、chunk 和设备，但不包含 `BLOCK`、`num_warps`；如果在同一进程中切换 kernel 配置而不清空缓存，可能会错误地复用旧 Graph。

## 5. 正确性检查

正确性检查在性能计时之前完成。任何候选配置出现非有限结果或超过误差阈值，都不进入后续性能排名。

### 5.1 常规规模

对四个主测试规模分别运行以下步数：

- `steps = 0`：检查输入副本和边界行为；
- `steps = 1`：检查单步公式；
- `steps = 16`：检查一个典型 Graph chunk；
- `steps = 4096`：主实验步数。

每次运行前恢复同一份初始场，不能复用已经被原地修改的 GPU tensor。检查：

- 输出 shape 为 `(N,)`；
- NumPy 输出为复数数组；
- GPU 输出为 `complex64`；
- 输出和中间结果均为有限值；
- 输入初始场没有被修改；
- 不同 `graph_chunk` 得到的结果一致；
- Graph 回放和普通 eager 执行得到的结果一致。

### 5.2 参考结果和误差

NumPy `complex128` 结果作为高精度参考。由于 PyTorch/Triton GPU 使用 `complex64`，比较时将 NumPy 结果转换为 `complex64`，并同时保存未转换的 NumPy 结果用于误差分析。

至少记录以下指标：

- 场的相对 L2 误差：

  ```text
  ||A_test - A_ref||₂ / max(||A_ref||₂, 1e-30)
  ```

- 最大绝对误差；
- 平均功率相对误差；
- 离散能量相对误差：`dt * sum(abs(A)**2)`；
- 输出功率谱的相对 L2 误差；
- 运行是否产生 NaN 或 Inf。

初始验收阈值可采用：

- 单步及短步数：相对场误差不超过 `1e-5`；
- `4096` 步长时间传播：相对场误差先以 `1e-3` 作为上限；
- Triton 与 PyTorch 的误差必须与两者分别对 NumPy 的误差相符，不能用两个错误结果互相一致来代替参考检查。

阈值应在性能搜索前固定，并记录在结果文件中。若长距离误差超过阈值，先判断是 `complex64` 累积误差、单步实现差异还是数值不稳定，不得直接通过放宽阈值解决。

### 5.3 非整除规模

主性能规模都是 `256` 的整数倍，无法覆盖 Triton kernel 的尾部 mask。另加以下正确性规模：

- `N = 257`；
- `N = 1025`；
- `N = 4097`。

至少验证 `steps = 1`、`16` 和 `4096`，确保最后一个 Triton program 的未满 block 不会漏算或越界。

### 5.4 步长收敛性

算子一致不等于数值收敛。单独做一次收敛性检查，保持目标传播距离不变：

| `dz` | `steps` | 目标 `z` |
| ---: | ---: | ---: |
| `0.001` | `4096` | `4.096` |
| `0.0005` | `8192` | `4.096` |
| `0.00025` | `16384` | `4.096` |

在至少一个中等规模和一个大规模上比较场、能量和功率谱变化。该实验用于确认误差主要来自时间步长时的收敛趋势，不计入主性能排名。

## 6. 性能测量口径

需要分开报告两种时间，避免把不同开销混在一起。

### 6.1 端到端时间

调用三个 solver 的公开接口，输入为主机 NumPy 数组，输出为主机 NumPy 数组。记录：

- NumPy CPU 总耗时；
- PyTorch GPU 总耗时，包含主机到 GPU 和 GPU 到主机传输；
- 优化 GPU 总耗时，包含传输、首次编译和首次建图的时间；
- 优化 GPU 缓存 Graph 后的总耗时。

端到端结果反映实际调用成本，但不能直接解释 kernel 加速比。

### 6.2 GPU 稳态时间

对 PyTorch baseline 和优化 GPU solver，预先完成：

- 输入 tensor 分配和传输；
- 线性传播因子和频率数组准备；
- Triton 编译；
- FFT 初始化；
- CUDA Graph 建图；
- 至少两次预热运行。

计时时只包含输入状态恢复、`4096` 步推进和独立输出复制。使用 CUDA Event 或在计时区间前后调用 `torch.cuda.synchronize()`，不能只测异步 kernel 提交耗时。

优化 GPU solver 需要额外报告：

- 首次调用时间；
- Triton 编译时间；
- CUDA Graph 建图时间；
- 缓存 Graph 的稳态时间；
- Graph 回放次数；
- 余数步骤数；
- CUDA Graph 相对于 eager 的加速比。

## 7. `BLOCK` 和 `num_warps` 联合搜索

### 7.1 候选范围

第一阶段使用统一配置，候选为：

| 参数 | 初始候选 |
| --- | --- |
| `BLOCK` | `64`、`128`、`256`、`512` |
| `num_warps` | `1`、`2`、`4`、`8` |

对每个 `N` 测试全部合法组合。若编译失败、资源不足或正确性不通过，记录失败原因并跳过该组合，不把失败当作零耗时。

### 7.2 搜索流程

1. 固定 `graph_chunk=16`，先不改变 CUDA Graph 的其他行为。
2. 对每个 `BLOCK × num_warps` 组合，在 `N=4096`、`steps=4096` 上完成正确性检查。
3. 对通过检查的组合，在四个主规模上测量 GPU 稳态时间。
4. 每个候选至少预热 2 次，正式测量至少 7 次，报告中位数、最小值、最大值和离散程度。
5. 选出每个 `N` 的最佳配置，并同时选出一个覆盖所有 `N` 的统一配置。
6. 若不同 kernel 的耗时差异明显，再分别搜索 `disp_and_loss`、`_kerr_cubic`、`_shock_derivative` 和 `_shock_update` 的配置；否则保留统一配置，避免调参复杂度超过收益。

选择标准按以下顺序：

1. 所有正确性检查通过；
2. 结果无 NaN/Inf；
3. 稳态中位数耗时最低；
4. 在相近性能下优先显存占用和运行波动更小的配置；
5. 记录是按单个 `N` 选择还是按全部规模的综合结果选择。

## 8. CUDA Graph `graph_chunk` 搜索

固定上一阶段选出的 kernel 配置，再测试：

```text
graph_chunk = 1, 2, 4, 8, 16, 32, 64, 128, 256
```

对每个 chunk 记录：

- 建图时间；
- 建图后的显存增量；
- 缓存 Graph 稳态时间；
- 总回放次数 `steps // graph_chunk`；
- 余数步骤数 `steps % graph_chunk`；
- 相对于 eager PyTorch 和 eager Triton 的加速比。

由于 `4096` 可被许多候选 chunk 整除，另外用 `steps=4097` 或 `steps=4100` 测试余数路径。至少覆盖：

- `steps < graph_chunk`；
- `steps == graph_chunk`；
- `steps == graph_chunk + 1`；
- `steps % graph_chunk != 0`。

分别报告“一次性运行”的总成本和“同一参数重复运行”的稳态成本。若应用只运行一次，不应只根据缓存 Graph 的稳态时间选择 chunk。

## 9. 重复和隔离规则

- 每个候选配置使用独立进程，或运行前清空 Graph 缓存并确认没有旧的 Triton kernel 配置残留。
- 所有实现使用同一份初始场；每次重复都从原始初始场重新复制。
- 候选配置交替执行或随机化执行顺序，避免 GPU 温度和系统负载与某个配置固定相关。
- 正式测量前至少执行 2 次预热；预热结果不纳入统计。
- 正式测量至少重复 7 次，优先报告中位数和四分位范围，不只报告一次最快时间。
- 每轮测量后同步 CUDA；需要时记录 `torch.cuda.max_memory_allocated()` 和 `torch.cuda.max_memory_reserved()`。
- 若使用多线程 NumPy，记录并固定 BLAS/OpenMP 线程数；CPU 与 GPU 的线程配置不能混用为同一项性能结论。

## 10. 结果记录格式

每一行结果至少包含：

```text
backend
n_samples
steps
dt
dz
target_z
dtype
alpha
beta2
beta3
beta4
gamma0
omega0
block
num_warps
graph_chunk
warmup_count
compile_seconds
capture_seconds
first_call_seconds
steady_median_seconds
steady_min_seconds
steady_max_seconds
steps_per_second
samples_steps_per_second
memory_allocated_bytes
memory_reserved_bytes
relative_l2_to_numpy
relative_l2_to_pytorch
max_absolute_error
energy_relative_error
finite
```

原始结果保存为 JSON，另生成一份汇总表，至少包含：

- 三种实现的端到端耗时；
- PyTorch baseline 到优化 GPU 的加速比；
- eager Triton 到 CUDA Graph 的加速比；
- 每个 `N` 的最佳 `BLOCK × num_warps`；
- 每个 `N` 的最佳 `graph_chunk`；
- 首次运行成本与稳态运行成本；
- 通过或失败的正确性检查及失败原因。

速度统一使用：

```text
steps_per_second = steps / seconds
samples_steps_per_second = N * steps / seconds
speedup = baseline_seconds / optimized_seconds
```

## 11. 实验执行顺序

1. 完成普通 PyTorch baseline，并验证它与 NumPy baseline 在单步和短步数下的一致性。
2. 为 Triton kernel 增加可配置的 `BLOCK` 和 `num_warps`，验证默认配置行为不变。
3. 在 `N=257`、`1025`、`4097` 上完成 kernel 尾部正确性测试。
4. 固定默认 kernel 配置，完成三种 solver 的基准正确性测试。
5. 完成 `BLOCK × num_warps` 联合搜索，保存每个候选的原始数据。
6. 选定 kernel 配置后完成 `graph_chunk` 搜索。
7. 用选出的配置在四个主规模、`4096` 步上完成正式端到端和 GPU 稳态实验。
8. 完成 `dz` 收敛性检查，确认性能结论没有建立在不稳定的数值结果上。
9. 汇总速度、误差、显存和首次运行成本，形成最终结论。

## 12. 最终验收条件

实验结果只有同时满足以下条件才可用于宣称优化有效：

- 四个主规模和非整除规模均通过正确性检查；
- 所有输出均为有限值，且初始输入未被修改；
- 优化 GPU 结果满足预先约定的误差阈值；
- 正式性能结果包含充分预热和 CUDA 同步；
- 首次调用成本和缓存后的稳态成本分开报告；
- `BLOCK`、`num_warps`、`graph_chunk`、dtype、物理参数和 GPU 环境均有记录；
- 加速比至少相对于普通 PyTorch GPU baseline 报告，不能只与 CPU NumPy 时间比较；
- 结果能够区分 Triton 算子收益、CUDA Graph 收益和主机数据传输收益。
