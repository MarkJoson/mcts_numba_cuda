# GPU PUCT MCTS 框架架构分析

> 入口点: `run_example8.py` → 追逐-逃避 (Pursuit-Evasion) 多智能体博弈

---

## 1. 项目文件结构

| 文件 | 职责 | 层级 |
|------|------|------|
| [run_example8.py](file:///home/robomaster/Research/mcts_numba_cuda/src/run_example8.py) | CLI 入口、Episode 循环、绘图/动画 | 应用层 |
| [example8_adapter.py](file:///home/robomaster/Research/mcts_numba_cuda/src/examples/example8_adapter.py) | GPU PUCT ↔ Problem 适配层 | 适配层 |
| [example8_problem.py](file:///home/robomaster/Research/mcts_numba_cuda/src/examples/example8_problem.py) | 追逐-逃避博弈逻辑 (4 机器人, MINCO 轨迹) | 问题域 |
| [puct_gpu.py](file:///home/robomaster/Research/mcts_numba_cuda/src/puct_gpu.py) | PUCTGpu 搜索引擎 (主循环 + GPU 调度) | 引擎层 |
| [puct_gpu_kernels.py](file:///home/robomaster/Research/mcts_numba_cuda/src/puct_gpu_kernels.py) | 全部 `@cuda.jit` 核函数 | CUDA 核函数层 |
| [puct_gpu_nn_bridge.py](file:///home/robomaster/Research/mcts_numba_cuda/src/puct_gpu_nn_bridge.py) | Numba ↔ PyTorch 零拷贝 GPU 共享内存桥 | 桥接层 |
| [puct_gpu_mechanics.py](file:///home/robomaster/Research/mcts_numba_cuda/src/puct_gpu_mechanics.py) | Device 函数 (双积分器 demo, 非 example8 使用) | 设备函数层 |

---

## 2. 组件关系图

```mermaid
graph TB
    subgraph "应用层 Application"
        RE8["run_example8.py<br/>CLI + Episode Runner + Plotting"]
    end

    subgraph "适配层 Adapter"
        ADAPTER["GPU_PUCT_Adapter<br/>policy() → action"]
        UPN["UniformPriorNet<br/>nn.Module → 均匀 logits"]
        VBW["ValueBridgeWrapper<br/>nn.Module → 零值"]
    end

    subgraph "问题域 Problem Domain"
        EX8["Example8<br/>4-robot pursuit-evasion<br/>MINCO quintic rollout"]
    end

    subgraph "引擎层 Engine"
        PUCT["PUCTGpu<br/>GPU tree search orchestrator<br/>3-phase sandwich loop"]
    end

    subgraph "桥接层 Bridge"
        BRIDGE["NumbaPytorchBridge<br/>零拷贝 GPU 共享缓冲区<br/>PyTorch tensor ↔ Numba DeviceNDArray"]
    end

    subgraph "CUDA 核函数层 Kernels"
        K_RESET["_reset_puct"]
        K_SELECT["_select_puct<br/>PUCT UCB tree walk"]
        K_EXTRACT["_extract_leaf_states"]
        K_PREP["_prepare_expansion_puct<br/>prior sort + action export"]
        K_COMMIT["_commit_expansion_puct_v2<br/>child alloc + edge reward"]
        K_BACKUP["_backup_with_edge_rewards_puct<br/>stepwise discounted backup"]
        K_ROT["_reduce_over_trees_puct"]
        K_ROA["_reduce_over_actions_puct"]
    end

    RE8 -->|"创建"| EX8
    RE8 -->|"创建"| ADAPTER
    RE8 -->|"run_episode()"| ADAPTER

    ADAPTER -->|"_ensure_agent()"| PUCT
    ADAPTER -->|"创建"| UPN
    ADAPTER -->|"创建"| VBW
    ADAPTER -->|"problem.step/reward"| EX8

    PUCT -->|"init_device_side_arrays()"| BRIDGE
    PUCT -->|"调度内核"| K_RESET
    PUCT -->|"调度内核"| K_SELECT
    PUCT -->|"调度内核"| K_EXTRACT
    PUCT -->|"调度内核"| K_PREP
    PUCT -->|"调度内核"| K_COMMIT
    PUCT -->|"调度内核"| K_BACKUP
    PUCT -->|"调度内核"| K_ROT
    PUCT -->|"调度内核"| K_ROA
    PUCT -->|"NN推理"| UPN
    PUCT -->|"NN推理"| VBW
    PUCT -->|"env step"| EX8

    BRIDGE -.->|"零拷贝视图"| K_EXTRACT
    BRIDGE -.->|"零拷贝视图"| K_PREP
    BRIDGE -.->|"零拷贝视图"| K_COMMIT
    BRIDGE -.->|"零拷贝视图"| K_BACKUP

    style RE8 fill:#4a90d9,color:#fff
    style ADAPTER fill:#7b68ee,color:#fff
    style PUCT fill:#e74c3c,color:#fff
    style BRIDGE fill:#f39c12,color:#fff
    style K_SELECT fill:#2ecc71,color:#fff
    style K_BACKUP fill:#2ecc71,color:#fff
    style EX8 fill:#9b59b6,color:#fff
```

---

## 3. 时序图 — 单步 `policy()` 调用

```mermaid
sequenceDiagram
    participant R as run_example8
    participant A as GPU_PUCT_Adapter
    participant P as PUCTGpu.run()
    participant GPU as CUDA Kernels
    participant BR as NumbaPytorchBridge
    participant NN as PolicyNet / ValueNet
    participant E as Example8 (CPU)

    R->>A: policy(problem, state)
    A->>A: _ensure_agent(problem) [惰性初始化]
    A->>A: 创建 UniformPriorNet, ValueBridgeWrapper

    loop 每个 turn_group [evaders, pursuers]
        A->>P: run(root_state, turn, policy_net, value_net, problem, sampler)

        Note over P: ══ Phase 0: Reset ══
        P->>GPU: _reset_puct [n_trees blocks]
        GPU-->>P: cuda.synchronize()

        loop step = 1..max_simulations (200)
            Note over P,GPU: ══ Kernel A: Selection ══
            P->>GPU: _select_puct [PUCT UCB → leaf]
            GPU-->>P: sync
            P->>GPU: _extract_leaf_states → bridge buffer
            GPU->>BR: 写入 leaf_states (零拷贝)
            GPU-->>P: sync

            Note over P,NN: ══ Host: NN Inference ══
            P->>BR: 读取 leaf_states (零拷贝)
            P->>NN: policy_model(states) → priors
            P->>NN: value_model(states) → values
            P->>BR: 写入 nn_priors, nn_values

            Note over P,GPU: ══ Kernel B1: Prepare Expansion ══
            P->>GPU: _prepare_expansion_puct
            GPU->>BR: 写入 parent_states, actions (零拷贝)
            GPU-->>P: sync

            Note over P,E: ══ Host: Environment Step ══
            P->>BR: 读取 expansion_valid, parent_states
            P->>E: problem.step(state, action, dt)
            P->>E: problem.normalized_reward(state, action)
            P->>E: problem.is_terminal(next_state)
            P->>BR: 写入 next_states, rewards, terminals

            Note over P,GPU: ══ Kernel B2+B3: Commit + Backup ══
            P->>GPU: _commit_expansion_puct_v2
            GPU-->>P: sync
            P->>GPU: _backup_with_edge_rewards_puct
            GPU-->>P: sync
        end

        Note over P,GPU: ══ Reduction ══
        P->>GPU: _reduce_over_trees_puct [sum ns/value]
        P->>GPU: _reduce_over_actions_puct [argmax]
        P-->>A: (best_action_idx, best_n, info)
    end

    A->>A: 从 root_action_cache 解析连续动作
    A-->>R: action (action_dim, 1)
```

---

## 4. 三明治迭代流程图 (核心搜索循环)

> [!IMPORTANT]
> **关键路径**: Host Environment Step 是计算瓶颈 — 每棵树的 `problem.step()` 顺序执行在 CPU 上

```mermaid
flowchart TD
    START([开始 PUCTGpu.run]) --> RESET

    subgraph RESET_PHASE["Phase 0: Reset"]
        RESET["_reset_puct<br/>每棵树初始化为单根节点<br/>复制 root_state"]
    end

    RESET --> LOOP_CHECK

    LOOP_CHECK{{"step < max_simulations<br/>且未超时?"}}
    LOOP_CHECK -->|否| REDUCE
    LOOP_CHECK -->|是| SELECT

    subgraph KERNEL_A["🟢 Kernel A — GPU Selection"]
        SELECT["_select_puct<br/>从根到叶 PUCT UCB 游走<br/>共享内存 argmax reduction"]
        EXTRACT["_extract_leaf_states<br/>叶节点状态 → Bridge 缓冲区"]
        SELECT --> EXTRACT
    end

    EXTRACT --> NN_INFER

    subgraph HOST_NN["🔵 Host — NN 推理 (PyTorch)"]
        NN_INFER["policy_model(leaf_states) → softmax priors<br/>value_model(leaf_states) → values<br/>通过零拷贝 Bridge 读写"]
    end

    NN_INFER --> PREP

    subgraph KERNEL_B1["🟢 Kernel B1 — GPU Expansion Prep"]
        PREP["_prepare_expansion_puct<br/>存储 priors, 插入排序 rank<br/>导出 parent_state + best_action"]
    end

    PREP --> ENV_STEP

    subgraph HOST_ENV["🔴 Host — Environment Step (CPU, 关键瓶颈)"]
        ENV_STEP["for each tree:<br/>  action_sampler(parent, idx) → 连续动作<br/>  problem.step(state, action, dt) → next_state<br/>  problem.normalized_reward() → reward<br/>  problem.is_terminal() → terminal"]
    end

    ENV_STEP --> COMMIT

    subgraph KERNEL_B2B3["🟢 Kernel B2+B3 — GPU Commit & Backup"]
        COMMIT["_commit_expansion_puct_v2<br/>分配子节点, 链接拓扑<br/>存储 edge_reward, 更新 PW boundary"]
        BACKUP["_backup_with_edge_rewards_puct<br/>沿路径累加折扣奖励<br/>value(d) = Σ γ^k · r_edge + γ^L · V_nn"]
        COMMIT --> BACKUP
    end

    BACKUP --> LOOP_CHECK

    subgraph REDUCE_PHASE["Phase Final: Reduction"]
        REDUCE["_reduce_over_trees_puct<br/>跨 n_trees 求和 ns/value<br/>1 block per action, 1 thread per tree"]
        ARGMAX["_reduce_over_actions_puct<br/>argmax 找最多访问的根动作<br/>1 block, 1 thread per action"]
        REDUCE --> ARGMAX
    end

    ARGMAX --> RESULT([返回 best_action, best_n])

    style KERNEL_A fill:#d4edda,stroke:#28a745
    style KERNEL_B1 fill:#d4edda,stroke:#28a745
    style KERNEL_B2B3 fill:#d4edda,stroke:#28a745
    style HOST_NN fill:#cce5ff,stroke:#004085
    style HOST_ENV fill:#f8d7da,stroke:#721c24
    style REDUCE_PHASE fill:#fff3cd,stroke:#856404
```

---

## 5. 数据流图 — GPU 内存与 CPU 交互

```mermaid
flowchart LR
    subgraph CPU_MEM["CPU Memory"]
        ROOT_STATE["root_state<br/>float32[29]"]
        ACTION_OUT["best_action: int<br/>root_action_cache: dict"]
        PROBLEM["Example8<br/>.step() .reward()<br/>.is_terminal()"]
    end

    subgraph GPU_TREES["GPU Device Arrays (per-tree)"]
        DEV_TREES["dev_trees<br/>int32[T,S,1+A]<br/>拓扑 parent+children"]
        DEV_STATES["dev_trees_states<br/>float32[T,S,D=29]"]
        DEV_NS["dev_trees_ns<br/>int32[T,S]"]
        DEV_TV["dev_trees_total_value<br/>float32[T,S,R=4]"]
        DEV_ER["dev_trees_edge_rewards<br/>float32[T,S,R=4]"]
        DEV_AP["dev_trees_action_priors<br/>float32[T,S,A=32]"]
        DEV_PR["dev_trees_prior_rank<br/>int16[T,S,A]"]
        DEV_PW["dev_trees_pw_boundary<br/>int16[T,S]"]
        DEV_RT["dev_trees_robot_turns<br/>int8[T,S]"]
    end

    subgraph BRIDGE["NumbaPytorchBridge (零拷贝)"]
        direction TB
        LS["leaf_states<br/>float32[T,D]"]
        LV["leaf_valid<br/>int32[T]"]
        NP_B["nn_priors<br/>float32[T,A]"]
        NV_B["nn_values<br/>float32[T,R]"]
        EV["expansion_valid<br/>int32[T]"]
        EPS["expanded_parent_states<br/>float32[T,D]"]
        EA["expanded_actions<br/>float32[T,Da]"]
        ENS["expanded_next_states<br/>float32[T,D]"]
        ER_B["expanded_rewards<br/>float32[T,R]"]
        ET["expanded_terminals<br/>bool[T]"]
    end

    subgraph REDUCTION["Reduction Arrays"]
        ANS["dev_actions_ns<br/>int64[A]"]
        ATV["dev_actions_total_value<br/>float32[A,R]"]
        BA["dev_best_action<br/>int32[1]"]
        BN["dev_best_n<br/>int64[1]"]
    end

    ROOT_STATE -->|"cuda.to_device"| DEV_STATES
    DEV_STATES -->|"_extract"| LS
    LS -->|"零拷贝读取"| NP_B
    LS -->|"零拷贝读取"| NV_B
    NP_B -->|"_prepare"| DEV_AP
    EPS -->|".cpu().numpy()"| PROBLEM
    PROBLEM -->|".cuda()"| ENS
    PROBLEM -->|".cuda()"| ER_B
    ENS -->|"_commit_v2"| DEV_STATES
    ER_B -->|"_commit_v2"| DEV_ER
    DEV_ER -->|"_backup"| DEV_TV
    NV_B -->|"_backup"| DEV_TV
    DEV_NS -->|"_reduce_trees"| ANS
    DEV_TV -->|"_reduce_trees"| ATV
    ANS -->|"_reduce_actions"| BA
    BA -->|"copy_to_host"| ACTION_OUT

    style BRIDGE fill:#fff3cd,stroke:#856404
    style GPU_TREES fill:#d4edda,stroke:#28a745
    style CPU_MEM fill:#f8d7da,stroke:#721c24
    style REDUCTION fill:#cce5ff,stroke:#004085
```

---

## 6. Episode 主循环流程图

```mermaid
flowchart TD
    MAIN([main]) --> PARSE["parse_args()"]
    PARSE --> CREATE_PROBLEM["problem = Example8()<br/>state_dim=29, action_dim=8<br/>num_robots=4, dt=1, tf=80"]
    CREATE_PROBLEM --> CREATE_SOLVER["solver = GPU_PUCT_Adapter(<br/>  max_actions=32, n_trees=8,<br/>  simulations=200, ...)"]
    CREATE_SOLVER --> TRIAL_LOOP

    subgraph TRIAL_LOOP["Trial 循环"]
        INIT["state = problem.initialize()<br/>随机初始位置, 验证距离约束"]
        INIT --> STEP_LOOP

        subgraph STEP_LOOP["Step 循环 (最多 80 步)"]
            POLICY["action = solver.policy(problem, state)<br/>══ 搜索核心: 2组 turn_group ══<br/>  evaders [0,1]: 1次 PUCTGpu.run<br/>  pursuers [2,3]: 1次 PUCTGpu.run"]
            REWARD["reward = problem.reward(state, action)<br/>捕获奖励 + 存活奖励 + 越界惩罚"]
            STEP["next_state = problem.step(state, action, dt)<br/>MINCO quintic rollout per robot"]
            TERMINAL{"problem.is_terminal(next_state)?<br/>越界 | 所有逃避者被捕 |<br/>所有追捕者已用 | t ≥ tf"}

            POLICY --> REWARD --> STEP --> TERMINAL
            TERMINAL -->|否| POLICY
        end

        TERMINAL -->|是| COLLECT["收集 sim_result"]
    end

    COLLECT --> PLOT["plot_sim_result() + render()"]
    PLOT --> MOVIE{"--movie?"}
    MOVIE -->|是| GIF["make_movie() → episode.gif"]
    MOVIE -->|否| SUMMARY
    GIF --> SUMMARY["打印统计摘要"]

    style POLICY fill:#e74c3c,color:#fff
    style STEP fill:#9b59b6,color:#fff
```

---

## 7. 关键架构特征总结

### 三明治 (Sandwich) 迭代模式

每轮模拟迭代是一个 **GPU → CPU → GPU** 三明治：

| 阶段 | 执行位置 | 核函数/函数 | 耗时 |
|------|---------|------------|------|
| Selection | GPU | `_select_puct` + `_extract_leaf_states` | ~μs |
| NN Inference | GPU (PyTorch) | `policy_model()` + `value_model()` | ~μs (均匀先验) |
| Expansion Prep | GPU | `_prepare_expansion_puct` | ~μs |
| **Env Step** | **CPU** | `problem.step()` × n_trees | **~ms (瓶颈)** |
| Commit+Backup | GPU | `_commit_v2` + `_backup_with_edge_rewards` | ~μs |

> [!WARNING]
> **关键瓶颈**: Environment Step 在 CPU 上顺序执行 `n_trees` 次 `problem.step()`，涉及 MINCO 矩阵求逆。这是整个搜索循环中最慢的部分。

### 零拷贝桥接 (NumbaPytorchBridge)

- PyTorch 拥有 GPU 内存，`cuda.as_cuda_array()` 创建 Numba 视图
- CUDA 核函数直接写入 → PyTorch 直接读取，**无 PCIe 往返**
- 12 个共享缓冲区覆盖：叶状态、先验、值、扩展数据

### Progressive Widening

子节点延迟扩展：`pw_boundary = ceil(C_pw × N^alpha_pw)`，最大不超过 `max_actions`。
先验排序 (insertion sort on GPU) 决定扩展优先级。

### 根并行 (Root Parallelism)

`n_trees=8` 棵独立树并行搜索，最终通过两级 reduction 合并：
1. `_reduce_over_trees`: 每个 action 跨树求和 ns/value
2. `_reduce_over_actions`: argmax 选最多访问动作

### Turn Group 机制

`turn_groups = [[0,1], [2,3]]` — evaders 共享一次搜索，pursuers 共享一次。
每个 `policy()` 调用执行 **2 次完整 PUCTGpu.run()**。
