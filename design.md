下面是当前 PyTorch 状态转移/碰撞模块的数据结构和算法说明。

| 模块/数据结构 | 位置 | 用途 | 形状/内容 |
|---|---|---|---|
| `MincoTorchTransition` | `src/examples/minco_torch_transition.py` | 单点 MINCO 动力学 | 处理单个点的 `6x2` 多项式系数状态 |
| `MincoPointEnvTransition` | `src/examples/minco_torch_transition.py` | 多点场景环境转移 | 管理多个点、时间、active mask、碰撞、done |
| `MincoPointEnvStep` | `dataclass` | 结构化 step 输出 | `coefficients / active / time / done / valid / collision masks / projected_targets` |
| flat state | `MincoPointEnvTransition.pack_state()` | MCTS tree 中保存状态 | `[time, active_0..active_N-1, coeff(agent0), ..., coeff(agentN-1)]` |
| coefficients | `(..., agents, 6, 2)` | 每个点的 MINCO 轨迹系数 | `6` 个五次多项式系数，`2` 维位置 |
| obstacle buffers | 初始化预计算 | 静态凸多边形障碍物 | `edge_points / edge_normals / edge_offsets / edge_mask` |
| collision pair buffers | 初始化预计算 | 点-点碰撞候选对 | `collision_pair_i / collision_pair_j`，按队伍过滤 |

| 算法 | 实现方式 | 每步开销特点 |
|---|---|---|
| MINCO 单点转移 | `F_stab @ coeff + G_stab @ target` | batched `einsum`，无 Python 循环 |
| target 可行域投影 | 预计算速度/加速度约束系数，step 时 `max/min + clamp` | 每步只做张量归约 |
| 时间推进 | `time += dt` | 纯 tensor 操作 |
| inactive 点处理 | inactive 点保持位置，高阶系数清零 | `torch.where` |
| 点-点碰撞 | 预计算跨队候选 pair，按距离排序做一对一最近匹配 | pair 数很小，适合当前 4 点场景 |
| 凸多边形障碍检测 | 半平面法：点到所有边外法线有符号距离，取最大值 | 核心是 `einsum("...ad,oed->...aoe")` |
| 障碍物 padding | 不同边数多边形 padded 到 `max_obstacle_vertices<=8` | 用 `edge_mask` 忽略无效边 |
| done 判断 | `time>=tf`、越界、障碍碰撞、队伍消灭 | 全部 batched boolean tensor |
| tree-pool expand 支持 | `gather_step_flat(pool, tree_ids, node_ids, target)` | 直接从 `(trees,nodes,state_dim)` gather parent state |

**初始化阶段预计算**

- MINCO 矩阵：`F/G/K/Kpp/F_stab/G_stab`
- target bound 的约束系数和除法缩放项
- 点-点碰撞候选 pair
- 障碍物每条边的单位外法线、offset、valid edge mask

**每次 rollout/expand 必算**

- gather parent state
- target 投影
- MINCO 转移
- 时间推进
- 点-点碰撞
- 障碍物半平面距离检测
- active mask / done 更新

核心设计目标是：把所有固定几何和线性系统量放到初始化，rollout/expand 阶段只保留大批量 PyTorch 张量计算。