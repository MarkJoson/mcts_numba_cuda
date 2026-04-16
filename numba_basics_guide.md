# Numba 与 Numba.CUDA 基础指南

这篇文档专为了解并开发该 MCTS (蒙特卡洛树搜索) 仓库所编写。仓库中高度依赖 `numba` 的即时编译（JIT）以及 `numba.cuda` 在 GPU 上进行高并发计算。以下内容将涵盖本仓库中出现的 Numba 核心用法。

## 1. Numba 基础概念与 `@jit`

Numba 是一个开源的 JIT (Just-In-Time) 编译器，它可以将 Python 和 NumPy 代码在运行时编译为快速的机器码，性能可以逼近 C/C++。

在这个仓库的某些游戏逻辑（例如 `src/gomoku.py`, `src/c4.py`）中，你会看到对普通 CPU 运算使用的基础加速：

```python
from numba import jit, int8

# 使用 @jit 装饰器可以让 Numba 自动分析并编译函数
@jit(nopython=True)  # 或者直接写 @njit
def check_win(board):
    # 里面通常是一些计算密集的循环逻辑
    pass
```

- **`nopython=True` (亦作 `@njit`)**：强制 Numba 在编译时不使用 Python C-API。如果遇到 Numba 不支持的语法类型，它会报错而不是回退到慢速的 Python 模式。对于追求极速的 MCTS 开发，通常都会要求代码完全处于 `nopython` 模式。

## 2. Numba.CUDA 基础（编写 GPU 核函数）

MCTSNC (MCTS with Numba & CUDA) 的核心在于 GPU 并发。本仓库在 `src/mctsnc.py` 等文件中大量使用了 `@cuda.jit`。

GPU 编程有两类主要函数：**Kernel 函数（核函数）** 和 **Device 函数（设备函数）**。

### A. 全局 Kernel 函数（Global Kernel）

被 `@cuda.jit` (并在括号里带上签名，或者直接 `@cuda.jit`) 装饰的是 Host（CPU）调用并在 Device（GPU）上运行的核函数。

```python
from numba import cuda
from numba import void, int32, float32, int8

# 1. 显式类型签名：可以加速编译，确保传入的数组类型和内存连续性准确无误
# 注意 `int8[:, :]` 代表 2 维的数组 (矩阵)，`int32[:]` 代表 1 维数组
@cuda.jit(void(int8[:, :], int32[:]))
def mcts_kernel(board_state, result_array):
    # 2. 获取当前线程的全局索引
    # 这是 CUDA 开发最常用的一句话，用以确定当前线程需要处理哪一段数据
    thread_id = cuda.grid(1) 
    
    # 3. 越界检查
    if thread_id < result_array.shape[0]:
        # 具体的计算操作
        result_array[thread_id] = board_state[thread_id, 0] + 1
```

**如何调用 Kernel 函数？**
调用 GPU kernel 时需要指定**网格尺寸（Grid Size）**和**线程块尺寸（Block Size）**：

```python
# 定义每个 Block 启动多少个 Thread
threads_per_block = 256
# 定义 grid 中包含多少个 Block (利用进位除法确保覆盖所有数据点)
blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block

# 调用 kernel (注意方括号里的执行配置)
mcts_kernel[blocks_per_grid, threads_per_block](d_board_state, d_result_array)
```

### B. 设备函数（Device Functions）

在 `src/mctsnc_game_mechanics.py` 中，你会看到大量这种装饰器：
```python
@cuda.jit(device=True)
def get_valid_moves(board):
    # 逻辑代码
    ...
```
- **`device=True`**：这意味着该函数**不能从 CPU 端直接调用**。它只能由其它的 `@cuda.jit` 函数或其它的 `device=True` 函数调用。
- **作用**：用于代码逻辑解耦，将复杂的 Kernel 拆分成多个更清晰的小函数，类似 C++ 里的 `__device__` 修饰符。

## 3. 内存与数组

核函数只能操作在 GPU 显存上的数据。Numba 提供了一些专门的类型表示 GPU 数组：

- 通过 `cuda.to_device()` 把主机数据（如 NumPy array）拷贝到 GPU 成为 Device Array：
```python
import numpy as np
host_data = np.zeros(1000, dtype=np.int32)

# host -> device
device_data = cuda.to_device(host_data)

# 调用核函数
kernel[blocks, threads](device_data)

# device -> host: 将结果拷回内存
result = device_data.copy_to_host()
```
*在极致优化下，为了避免频繁的主机-设备拷贝降低性能，往往会在 GPU 上维护尽可能久的状态，这也就是为什么 MCTS 的树结构可以直接在显存中存储。*

## 4. CUDA 并行随机数生成 (Random)

传统的 Python `random` 或是 numpy `np.random` 并不适用于并行的 GPU kernel。由于同一时刻成千上万的线程在生成随机数，它们需要相互独立且线程安全的随机数生成器。

该仓库使用了 Numba 提供的 `xoroshiro128p` 并行随机数生成器（`src/mctsnc.py` 中有相关导入）：
```python
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type
```

**使用范例**：
```python
# 1. (Host端) 首先在 CPU 初始化 GPU 的随机状态数组
# 如果你打算开启 1024 个线程，建议状态数也设为 1024
rng_states = create_xoroshiro128p_states(1024, seed=42)

# 2. Kernel 函数定义，传入 rng_states (类型为 xoroshiro128p_type[:])
@cuda.jit
def random_rollout_kernel(rng_states, out_array):
    thread_id = cuda.grid(1)
    if thread_id < out_array.shape[0]:
        # 3. 在 GPU 设备端生成随机数
        # 传入状态数组和该线程专用的 state_id
        rand_val = xoroshiro128p_uniform_float32(rng_states, thread_id)
        out_array[thread_id] = rand_val
```
这种形式是本仓库 MCTS 在并发 Simulation / Rollout 阶段能够高速进行随机选择的关键基石。

## 5. 类型注解的快速解读

你在 `src/mctsnc.py` 许多函数的头部会看到非常复杂的类型参数定义，例如：
`@cuda.jit(void(int8[:, :], int32[:, :, :], boolean[:, :], int16[:, :], xoroshiro128p_type[:]))`

这里逐个拆解代表的意思：
- `void`: 返回值为空，因为 CUDA kernel 函数必须不返回任何值 (数据全部通过写入参数数组里传出)。
- `int8`, `int16`, `int32`: 8位、16位、32位整数。为了榨取极限的显存资源，避免所有数据都用64位来存储。
- `float32`: 32位单精度浮点。
- `boolean`: 布尔值数组类型。
- `[:, :]`: 数组的维度，2个冒号代表这是一个 2阶 张量/矩阵。`[:, :, :]` 就是 3维 的。
- `xoroshiro128p_type[:]`: 一维的随机数状态数组，通过上文提到的随机数初始化函数提供。

## 总结

阅读该仓库的代码时，可以按以下宏观视角理解它的 Numba 架构：
1. **环境与常量准备（CPU）**：处理用户参数，准备 numpy 数组内存，分发到显存 `to_device()`。并利用 `create_xoroshiro128p_states` 初始化随机状态。
2. **运算核心（GPU）**：所有带 `@cuda.jit` 的重头戏都在并发计算 MCTS（包括 Selection，Expansion，Simulation，Backpropagation），期间利用 `@cuda.jit(device=True)` 抽取出来的游戏特有逻辑保持运算的可维护性。
3. **结果回收（CPU）**：GPU 利用指针修改完成动作分布、胜率统计后，通过 `copy_to_host()` 把微小的数据取回。
