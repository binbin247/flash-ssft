做 NLSE 计算的测试。目的是为了验证算子优化后的仿真性能。

- 模式数分别取：2**10, 2**11, 2**12, 2**13
- 运行步数：4096 步
- 每一个模式数下，分别运行如下三种算法：
    - /Users/binbin/Desktop/flash-ssft/src/flashlle/solvers/nlse_numpy_baseline.py
    - /Users/binbin/Desktop/flash-ssft/src/flashlle/solvers/nlse_pytorch_baseline.py
    - /Users/binbin/Desktop/flash-ssft/src/flashlle/solvers/nlse_gpu_opt.py
    对于 gpu 优化算法，需要优化超参数 ： 优先联合搜索 BLOCK 和 num_warps， 再搜索 CUDA Graph 的 chunk