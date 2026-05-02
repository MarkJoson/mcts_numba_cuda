# puct_gpu_v2 Select Kernel 实验总结

本文记录 `src/puct_gpu_v2.py` 中并行 PUCT selection 的功能验证、压力测试和
`all-candidate preclaim` 与 `winner-only` 虚拟损失策略的对比经验。

当前代码保留的 select kernel 只有两个：

- `preclaim`: 默认 all-candidate preclaim。
- `winner_recalc`: winner claim 发生 inflight 冲突后，排除原 winner 并重算一次。
  当前 claim 使用 counter-CAS，即 `CAS(old, old + 1)`，不是 `CAS(0, 1)` 锁。

历史实验中的 `winner_only`、`topk*`、`preclaim_dedup` 已从代码和 benchmark
variant 中移除；下文涉及这些名字的段落仅作为实验记录保留。

## 背景

`puct_gpu_v2.py` 的并行粒度是：

- 1 个 CUDA block 遍历 1 棵树。
- 1 个 warp 执行 1 次独立 selection。
- 同一个 block 内多个 warp 可能同时到达同一个节点。

因此 virtual loss 的时机很关键。若多个 warp 同时在同一节点计算 PUCT 分数，
而 virtual loss 尚未写入，那么它们会在同一份旧统计上做出相同选择，virtual
loss 无法起到分散并发路径的作用。

当前默认实现仍使用 `all-candidate preclaim`：

1. 每个 warp 在评分当前节点所有合法候选边前，对自己负责的候选边执行
   `node_inflight[child] += 1`。
2. 用 `edge_N(parent, slot) + node_inflight[child]` 计算 PUCT。
3. warp reduce 得到 winner。
4. loser child rollback，winner child 保留 inflight，直到 backup 转为真实统计。

这比 winner-only 多很多 atomic，但能让同层并发 selection 在评分阶段就感知
其它 warp 的虚拟访问。

## Counter-CAS Winner Recalc

曾经尝试把 `winner_recalc` 的 winner claim 写成 strict `CAS(0, 1)`。这会把
inflight 从计数器退化成 0/1 锁：多 warp 同时合法选择同一 child 时，只有一个
warp 能持有 virtual loss，其它 warp 会变成 invalid。这不符合 MCTS selection
语义，也会在 benchmark 中制造高 invalid rate。

当前实现改为 counter-CAS：

```text
child = edge_child_id[tree, node, eid]
old = node_inflight[tree, child]
CAS(node_inflight, old, old + 1)
```

语义：

- `old == 0`: 本 warp claim 了一个此前未 inflight 的 winner。
- `old > 0`: 本 warp 仍然成功 claim 了一份 virtual loss，但知道发生了冲突。
- 冲突时，`winner_recalc` 先释放自己的原 winner claim，再排除原 winner 重算一次。
- 若没有其它合法 child，则允许回到原 winner；same-path 不是非法状态。
- bounded CAS 重试耗尽后退化为 `atomic.add`，避免瞬时 CAS 竞争把合法路径变成
  `SELECT_INVALID`。

这个设计保留了 virtual loss 的计数容量，同时给 `winner_recalc` 一个检测和补救
重复 winner child 的信号。它比 strict CAS 更接近 CPU batch reference，也不会像
winner-only 那样完全丢失冲突信息。

## Node Inflight Refactor

当前代码已移除 `edge_inflight`，改成两张 node 级数组：

```text
node_inflight[tree, child]         # 已存在 child node 的 virtual loss
node_expand_inflight[tree, node]   # 当前 node 上正在进行的 expand ticket 数
```

选择已展开 edge 时仍然先读取 `edge_child_id[parent, slot]`，再用
`node_inflight[child]` 参与 PUCT denominator。这样不会丢失 sibling action 的
virtual loss 区分；每条 action 的区分来自它指向的 child。区别在于如果多个
parent/action 指向同一个 child，它们会共享 virtual loss，这更接近 state/node
级 transposition 语义。

expand claim 不再占用空 edge slot 的 inflight，而是：

```text
ticket = atomic.add(node_expand_inflight[tree, node], 1)
slot = cur_expanded + ticket
```

若 `slot >= allowed`，立即 rollback ticket。backup 看到 `SELECT_EXPAND` 后释放
`node_expand_inflight[tree, selected_node]`。这保留了“同一节点可并发扩展多个
不同 slot，但同一个 slot 只被一个 warp 持有”的语义。

访存影响：

- 热路径多了一次 `edge_child_id -> child -> node_inflight[child]` 的间接读取。
- 同时删除了巨大的 `edge_inflight[tree,node,action]` 数组，显存和 L2 压力下降。
- 当前 deep-wide benchmark 已修正为 sibling-unique child：同一个父节点下的
  不同 action 映射到不同 child id。这样测到的是宽节点扫描和真实 child
  inflight 的开销，而不是人为把所有 action 聚到同一个 child 上制造原子热点。
- 修正后的 24G scale benchmark 可以完整覆盖 4096/8192/16384 树、actions
  256/128/64、depth 512/1024，全部 `invalid=0`、`inflight=ok`。
- 若要评估 transposition 或多 action 指向同一 child 的情况，应单独作为
  shared-child 压力场景，而不是混入普通 deep-wide 性能结论。

## 结果编码与验证

v2 只保留一个 packed output：`out_selected_node`，没有单独的
`out_selected_kind`。

## Benchmark 计时协议

select benchmark 只把 select kernel 放进 CUDA event elapsed 中。由于每次 select
都会持有 `node_inflight/node_expand_inflight`，每次 select 后仍然必须排队一个
backup kernel 释放状态；这个 backup 不计入 select-only 时间，但会计入实际 kernel
启动次数。

当前默认值：

- `--v2-bench` 的浅层 select-only：每个数据点 `warmup=10`，计时 select
  launch `100` 次，另有 `110` 次 untimed backup launch。
- `--v2-bench` 的 deep-tree：每个数据点 `warmup=5`，depth <= 128 时计时
  select launch `20` 次，depth > 128 时计时 `8` 次，backup 次数为
  `warmup + timed_select_launches`。
- `--v2-scale-bench`：每个数据点 `warmup=5`，计时 select launch `5` 次，
  另有 `10` 次 untimed backup launch。

这些值都可以用环境变量覆盖：

```text
PUCT_V2_BENCH_WARMUP
PUCT_V2_BENCH_ITERS
PUCT_V2_DEEP_WARMUP
PUCT_V2_DEEP_ITERS_SHORT
PUCT_V2_DEEP_ITERS_LONG
PUCT_V2_SCALE_WARMUP
PUCT_V2_SCALE_ITERS
```

## CPU Sequential Baseline

新增 `--v2-cpu-bench` 用于比较单线程 CPU 顺序 select 与 GPU select kernel。

CPU baseline：

- 使用 Numba `njit(cache=True, fastmath=True)`。
- 顺序遍历所有 `trees * traversals_per_tree`。
- 不引入 virtual loss，不做 backup，只按当前 `edge_N/W/prior` 计算 PUCT。
- 与 GPU 使用相同树数量和每树遍历次数：
  `traversals_per_tree = warps_per_tree * gpu_timed_select_launches`。

GPU 侧仍然使用现有 select kernel；每次 select 后需要 untimed backup 清理 inflight，
但 CUDA event 只统计 select kernel elapsed。

默认结果节选：

```text
scenario       variant          cpu sel/s     gpu sel/s   gpu/cpu
narrow_d64     preclaim          4.74M        24.62M       5.19
narrow_d64     winner_recalc     4.74M        15.24M       3.21
wide32_d64     preclaim          0.27M        15.26M      56.80
wide128_d128   preclaim          0.04M         2.75M      72.20
wide256_d256   preclaim          9.50K       421.06K      44.31
```

解释：

- 窄深链 fanout=1 时，CPU 顺序 baseline 已经很强，GPU 主要依靠批量树和 warp
  并行取得数倍优势。
- 宽节点下 CPU 每次遍历都要顺序扫描 fanout；GPU 的 warp-lane fanout 扫描优势
  明显，差距扩大到几十倍。
- 这个 benchmark 不是 CPU/MCTS 完整语义对照；它只衡量“同等 selection 数量和
  类似 PUCT 扫描工作量”下的 select throughput。

## CPU/GPU Scale-Aligned Baseline

新增：

```bash
bash src/run_puct_tests.sh --v2-cpu-scale-bench
```

这个 benchmark 对齐 `--v2-scale-bench` 的深宽 synthetic topology：

- tree 数覆盖 4096/8192/16384。
- 默认子集覆盖 `actions=256/128/64` 与 `depth=512/1024`。
- GPU 仍使用 `make_deep_chain_case_device`，也就是和 scale benchmark 相同的
  device-side 初始化。
- CPU 侧使用 procedural deep-wide tree，避免为 4096..16384 树分配巨大 host
  数组；它保留 PUCT fanout scan 计算量，但不模拟 virtual loss、backup 或随机
  rollout。
- `traversals_per_tree = warps_per_tree * gpu_timed_select_launches`，确保 CPU/GPU
  选择次数对齐。
- 设置 `PUCT_V2_CPU_SCALE_FULL=1` 可跑完整 12 场景矩阵。

当前默认子集结果：

```text
scenario       variant          cpu sel/s     gpu sel/s   gpu/cpu
eq256_d512     preclaim          15.51K       498.67K      32.15
eq256_d512     winner_recalc     15.51K       472.67K      30.48
eq256_d1024    preclaim           7.79K       241.50K      31.02
eq256_d1024    winner_recalc      7.79K       238.73K      30.66
eq128_d1024    preclaim          16.66K       687.17K      41.24
eq128_d1024    winner_recalc     16.66K       498.04K      29.89
eq64_d1024     preclaim          32.77K         1.15M      35.04
eq64_d1024     winner_recalc     32.77K       627.34K      19.14
hot64_d1024    preclaim          49.66K         1.16M      23.39
hot64_d1024    winner_recalc     49.66K       638.54K      12.86
```

完整 GPU-only scale benchmark 当前结果仍然通过 24G budget 下的 12 场景矩阵：

```text
variant         scenario        sel/s
preclaim        eq256_d512      470.93K
preclaim        eq256_d1024     224.72K
preclaim        hot256_d512     470.38K
preclaim        hot256_d1024    229.02K
preclaim        eq128_d512        1.35M
preclaim        eq128_d1024     665.80K
preclaim        hot128_d512       1.36M
preclaim        hot128_d1024    668.23K
preclaim        eq64_d512         2.24M
preclaim        eq64_d1024        1.13M
preclaim        hot64_d512        2.25M
preclaim        hot64_d1024       1.14M
winner_recalc   eq256_d512      457.32K
winner_recalc   eq256_d1024     228.61K
winner_recalc   hot256_d512     461.50K
winner_recalc   hot256_d1024    227.86K
winner_recalc   eq128_d512      954.37K
winner_recalc   eq128_d1024     479.70K
winner_recalc   hot128_d512     968.35K
winner_recalc   hot128_d1024    483.61K
winner_recalc   eq64_d512         1.25M
winner_recalc   eq64_d1024      632.10K
winner_recalc   hot64_d512        1.26M
winner_recalc   hot64_d1024     630.63K
```

结论：

- 大规模深宽场景下，GPU 相对 CPU 顺序 procedural baseline 仍有 12.9x..41.2x
  throughput 优势。
- `preclaim` 在 `actions=128/64` 的大规模场景明显快于 `winner_recalc`；在
  `actions=256` 时两者接近。
- CPU procedural baseline 的 ns/edge 很乐观，因为它没有真实数组访存、原子、
  virtual loss 和 backup；因此它适合作为“同等 fanout/depth 扫描量”的下界参考，
  不适合作为完整 CPU MCTS baseline。

## Block-Local Tie-Break

同分散列有两类低成本做法：

- block-local deterministic tie-break：只在 `score == best_score` 时用
  `wid % cur_expanded` 旋转 edge 优先级。
- epsilon/noise：给 score 加一个很小的 per-warp/per-edge 扰动。

当前 preclaim 采用第一种。它只影响严格同分，不改变 PUCT score，也不会在“近似同分但
确有优劣”的情况下翻转排序；额外成本主要是每层一次 `wid % cur_expanded` 和同分
分支里的整数 key 比较。epsilon/noise 每条候选都要多做浮点扰动，并且需要非常小心
数值尺度，否则可能改变真实 PUCT 排序。

该 tie-break 只在一个 block 内生效。如果未来允许多个 block 同时遍历同一棵树，
不同 block 之间仍然可能选择相同 tie winner；那需要跨 block 的 claim/reselect、
tree-level scheduling，或把 block id/hash 纳入更高层的并发协议。

`winner_recalc` 不使用这个 block-local tie-break。它的 readonly 选择用固定
`eid` tie-break，然后依赖 `node_inflight` 上的 counter-CAS 冲突信号做重算；
这样不会把 winner_recalc 的语义限制在单个 block 内。当前 recalc 代码是一个
`while retry < MAX_RECALC_RETRY` 循环，默认最多两次：第一次选 winner，若 claim
发现冲突则释放并跳过该 eid 重算一次；如果跳过后没有其它合法 child，则允许回到
原 winner，因为 same path 本身不是非法状态。

实验结果：

- 去掉 tie-break 后，浅层 `preclaim wide_node` 退化为
  `unique_first=1`、`path_H=0`、`invalid=75%`。
- 恢复 block-local tie-break 后，同场景回到 `unique_first=5`、`path_H=0.835`。
- 大规模 `preclaim eq256_d1024` 约 `217k sel/s`、`max1/tree=28.69%`；
  `winner_recalc eq256_d1024` 约 `231k sel/s`、`max1/tree=86.03%`。

编码格式：

```text
kind        = raw >> 22
expand_slot = (raw >> 14) & 0xFF
node_id     = raw & 0x3FFF
```

重要约束：

- `node_id < 16384`
- `node_count[tree]` 是每棵树的有效 node 上界；合法 node 范围为
  `[0, node_count[tree])`
- `node_count[tree] <= node array capacity` 且 `node_count[tree] <= 16384`
- `max_actions <= 256`
- path edge 编码为 `(parent << 8) | slot`

`node_expanded.shape[1]` 只表示数组容量，不再作为逻辑有效节点数。select、
rollback、backup 中 child 合法性统一检查 `child < node_count[tree]`。这样动态树
可以预分配大容量，但不会把尚未分配的 node slot 当作 fresh leaf。

新增的 kernel-level 测试位于 `src/test_puct_gpu_v2.py`，运行方式：

```bash
bash src/run_puct_tests.sh --v2
bash src/run_puct_tests.sh --v2-stress
bash src/run_puct_tests.sh --v2-bench
bash src/run_puct_tests.sh --v2-scale-bench
bash src/run_puct_tests.sh --v2-cpu-ref
bash src/run_puct_tests.sh --v2-cpu-bench
bash src/run_puct_tests.sh --v2-cpu-scale-bench
```

注意：GPU 在沙箱内不可见时会 skip；需要在可访问 NVIDIA driver 的环境运行。

已覆盖 corner cases：

- fresh root expand claim/release
- terminal root
- invalid child/node rollback
- child id 超过 per-tree `node_count` 时 invalid 且 rollback
- progressive widening 多 warp slot claim
- fully-expanded select + backup roundtrip
- depth limit
- packing boundary: 256 actions 可用，257 invalid；容量可以超过 16384，但
  `node_count > 16384` invalid
- deep tree stress: depth 16/64/128/256/512/1024，窄树和宽树

`node_count` 改动后的验证：

```text
bash src/run_puct_tests.sh --v2
bash src/run_puct_tests.sh --v2-stress
bash src/run_puct_tests.sh --v2-bench
```

结果均通过。性能上，每个 warp 由 lane0 读取一次 `node_count[tree]`，再用
`shfl_sync` 广播为寄存器标量参与 child 边界判断；热路径没有为每条候选边增加
额外全局访存。

## 已删除的 Winner-only 历史实验

历史上曾加入实验 kernel：

```python
_select_kernel_winner_only(..., reselect_on_conflict)
```

两个变体：

- `winner_only`: 先评分，reduce 出 winner，再只对 winner 做 `edge_inflight += 1`。
- `winner_reselect`: 若 winner claim 时发现旧 inflight 已大于 0，则撤销该 claim，
  排除第一次 winner 后局部重选一次。

它们的优点是 atomic 数显著减少：

```text
preclaim:         每层约 2 * fanout 次 atomic
winner_only:      每层约 1 次 atomic
winner_reselect:  每层约 3 次 atomic
```

但它们无法解决“多个 warp 同时评分同一节点”的根本并发问题。虚拟损失写入发生
在 winner 确定之后，所以评分阶段仍可能出现高度一致的路径选择。

## 实测摘要

测试环境：沙箱外真实 CUDA 环境，`bash src/run_puct_tests.sh --v2-bench`。

### 浅层场景

```text
variant          scenario                    sel/s
preclaim         high_contention_equal_prior  237796
winner_only      high_contention_equal_prior  221347
winner_reselect  high_contention_equal_prior  233420

preclaim         wide_node                   1971764
winner_only      wide_node                   1863968
winner_reselect  wide_node                   1845069

preclaim         biased_prior                1718228
winner_only      biased_prior                1818446
winner_reselect  biased_prior                1598261
```

观察：

- 高竞争 equal-prior 根节点：winner-only 不占优。
- 普通 wide_node：winner-only 更慢，且 first-edge diversity 更差。
- 强偏置 prior：winner-only 有小幅收益，因为本来就会集中到 hot action。

### 深宽树场景

深树构造方式：

- `narrow`: 每层 `fanout=1`。
- `wide64`: 每层 64 个合法候选，全部指向下一层主路径节点。
- `wide256`: 每层 256 个合法候选，全部指向下一层主路径节点。

这种构造避免指数级节点爆炸，同时保留每层完整 fanout 的评分和 atomic 压力。

```text
wide256 depth=64
preclaim         5725 sel/s
winner_only      6656 sel/s
winner_reselect  6611 sel/s

wide256 depth=128
preclaim         2994 sel/s
winner_only      3256 sel/s
winner_reselect  3437 sel/s

wide256 depth=256
preclaim         1532 sel/s
winner_only      1649 sel/s
winner_reselect  1693 sel/s
```

观察：

- winner-only 在深层、宽 fanout、低并发冲突场景能获得约 7%-16% 吞吐收益。
- winner_reselect 在部分深宽场景略优，但浅层或偏置 prior 场景可能明显变慢。
- 所有 benchmark 都额外检查了 `node_inflight/node_expand_inflight` 无负数且 backup 后归零。

## 结论

默认 select kernel 不应切换到 winner-only。

建议策略：

- 默认继续使用 `all-candidate preclaim`，保证并发 selection 的 virtual loss
  语义最强。
- `winner_only` 可以作为实验 fast path，用于：
  - 深树；
  - 大 fanout，尤其 128/256；
  - warp 间低冲突；
  - prior/Q 已经自然分散，或者允许较强路径集中。
- `winner_reselect` 只适合继续实验，不建议默认启用；它能改善某些冲突指标，
  但吞吐稳定性不如 winner_only。

一个保守的自动选择规则可以是：

```text
if cur_expanded >= 128 and depth >= 64 and active_warps_per_tree <= 2:
    use winner_only
else:
    use preclaim
```

这条规则还没有接入生产路径；目前应保持显式 benchmark 对比，避免隐藏改变 MCTS
统计行为。

## Top-k Preclaim 实验

新增实验 kernel：

```python
_select_kernel_topk_preclaim(..., top_k)
```

历史上曾支持 `topk2/topk4/topk8`。实现思路是在每层先用轻量分数
`prior / (1 + N + inflight)` 选出 top-k 候选并预加 virtual loss，然后仍对全部
合法边计算完整 PUCT 分数；非 winner 的 top-k 候选 rollback，若 winner 不在 top-k
中则补一次 winner claim。

实测结果不适合作为默认路径：

- 浅层 `wide_node` 中，`topk4` 能把 invalid rate 从约 75% 降到约 48%，first-edge
  unique 从 3 提高到 5，但吞吐从约 1.90M sel/s 降到约 1.62M sel/s。
- 深宽树中 top-k 明显变慢。`wide256 depth=1024` 上，`preclaim` 约 386 sel/s，
  `winner_only` 约 456 sel/s，而 `topk4` 约 142 sel/s。

主要原因是当前 top-k 每层需要额外多轮 cheap-score 扫描，再叠加 top-k atomic。
当 depth 和 fanout 同时变大时，这个额外扫描成本被放大。top-k 已从当前代码和
benchmark variant 中移除；它只作为历史实验记录保留。

## 大规模深宽树实验

新增 runner：

```bash
bash src/run_puct_tests.sh --v2-scale-bench
```

该模式直接在 device 上构造大规模深宽链式树，避免 host 侧复制一份 10-20GiB
级别的数组。默认显存预算为 24GiB，并额外保留 0.5GiB：

```text
PUCT_V2_SCALE_BUDGET_GB=24.0
PUCT_V2_SCALE_RESERVE_GB=0.50
PUCT_V2_SCALE_ITERS=3
PUCT_V2_SCALE_WARPS=8
PUCT_V2_SCALE_VARIANTS=preclaim,winner_recalc
```

重要修正：每棵树 `1 warp` 时没有同树竞争，winner-only 的结果只能作为“少
atomic 的吞吐基线”，不能证明 virtual loss 策略正确。大规模竞争测试默认改为
每棵树 `8 warp`，并新增单树内部指标。注意，路径重复本身不是错误：多条
selection 共享一段公共前缀，甚至少数 warp 选择同一路径，都是 PUCT 下允许出现的。
我们真正关心的是重复是否“必然发生”或“以很大概率发生”。

- `max1/tree%`: 每棵树内最热门 first edge 覆盖 warp 的平均比例。
- `maxpath%`: 每棵树内最热门完整 path 覆盖 warp 的平均比例。
- `path75%`: 最热门完整 path 覆盖至少 75% warp 的树比例。
- `samepath%`: 所有有效 warp 都选择完全相同 path 的极端比例，仅作为补充。

为了避免 fresh leaf expand claim 把路径竞争表现为 invalid，scale benchmark 的
深链末端使用 terminal leaf；这样所有 warp 都能完成 selection，路径塌缩会直接
体现在 `maxpath%/path75%/samepath%` 上。

### 当前 counter-CAS 多 warp 结果

在每树 `8 warp`、terminal leaf、24GiB 预算下，当前只比较
`preclaim,winner_recalc`。节选结果：

```text
variant          scenario       trees  depth  acts   sel/s    max1/tree  maxpath  invalid  inflight
preclaim         eq256_d1024     4096   1024   256   241706     28.02%    12.50%    0.00%       ok
winner_recalc    eq256_d1024     4096   1024   256   376016     87.50%    12.50%    0.00%       ok

preclaim         eq128_d1024     8192   1024   128   410060     43.30%    12.50%    0.00%       ok
winner_recalc    eq128_d1024     8192   1024   128   641148     84.34%    12.50%    0.00%       ok

preclaim         eq64_d512      16384    512    64  1205887     66.83%    12.50%    0.00%       ok
winner_recalc    eq64_d512      16384    512    64  1632436     66.69%    12.50%    0.00%       ok
```

结论：

- counter-CAS 没有破坏 virtual-loss 计数：large-scale benchmark 中
  `invalid=0` 且 `inflight=ok`。
- `winner_recalc` 在深宽大批量场景吞吐高于 preclaim，但 first-edge 分布更集中，
  尤其 `acts=128/256` 时 `max1/tree%` 明显更高。
- 因此它仍应作为可选实验路径，而不是默认替代 preclaim；默认策略要优先保证
  同层评分前的 virtual loss 可见性。

### 单 warp 基线

在每树 `1 warp` 的无竞争基线中，以下 20GiB 级配置均跑通，`path_len` 正确且
backup 后 `inflight=ok`：

```text
variant      trees   depth  acts   estGiB   sel/s
preclaim      4096    1024   256    20.07   124320
winner_only   4096    1024   256    20.07   137869
topk4         4096    1024   256    20.07    95668

preclaim      8192    1024   128    20.11   239566
winner_only   8192    1024   128    20.11   314304
topk4         8192    1024   128    20.11   168230

preclaim     16384    1024    64    20.21   472475
winner_only  16384    1024    64    20.21   536596
topk4        16384    1024    64    20.21   227216
```

大规模结果强化了之前的判断：

- 低竞争、每树 1 warp、深宽链式树上，`winner_only` 稳定快于 preclaim；但该
  场景没有同树竞争，不应用来论证 virtual loss 方案。
- `topk4` 在大规模深宽场景仍显著慢于 preclaim，说明当前 top-k 的额外扫描成本
  没有被减少 atomic 的收益抵消。
- 这组测试只说明 select kernel 的吞吐与 virtual-loss bookkeeping 正常；策略质量
  仍需在完整 MCTS 搜索闭环里评估。

### 多 warp 竞争检查

在每树 `8 warp`、terminal leaf、equal/hot prior 混合场景下，winner-only 的逻辑
问题被直接复现：吞吐更高，但同一棵树内的 warp 高概率选择同一路径。在旧表里
用 `samepath%` 表达的是极端全塌缩；当前 benchmark 输出改为 `maxpath%` 与
`path75%`，更适合区分“允许的部分重复”和“高概率塌缩”。

节选结果：

```text
variant          scenario       trees  depth  acts  sel/s    same1(old)  samepath(old)
preclaim         eq256_d1024     4096   1024   256  250860       0.00%       0.00%
winner_only      eq256_d1024     4096   1024   256  365508     100.00%     100.00%
winner_reselect  eq256_d1024     4096   1024   256  350306       0.00%       0.00%
topk4            eq256_d1024     4096   1024   256  104302     100.00%     100.00%

preclaim         hot256_d1024    4096   1024   256  249192       0.00%       0.00%
winner_only      hot256_d1024    4096   1024   256  357462     100.00%     100.00%
winner_reselect  hot256_d1024    4096   1024   256  352535       0.00%       0.00%
topk4            hot256_d1024    4096   1024   256  101632      99.98%      83.62%

preclaim         eq64_d512      16384    512    64 1222565       3.23%       0.00%
winner_only      eq64_d512      16384    512    64 1638693     100.00%      99.96%
winner_reselect  eq64_d512      16384    512    64 1579652       0.00%       0.00%
topk4            eq64_d512      16384    512    64  465543     100.00%      99.98%
```

结论：

- `winner_only` 在多 warp 同树竞争时不应作为默认策略；它只是更快地让多个 warp
  高概率走到同一路径，virtual loss 没有在评分前产生分散作用。
- `winner_reselect` 能修复这组 synthetic benchmark 中的同路径塌缩，而且仍比
  preclaim 快；但它是“冲突后补救”，不是评分前可见的 virtual loss，仍需完整
  MCTS 质量评估。
- 当前 `topk4` 在多 warp 竞争下也会高概率塌缩，说明“每个 warp 独立 top-k”不是
  足够的同树协同方案。
- 默认策略应继续保持 preclaim，winner-only 只能保留为无竞争/低竞争吞吐实验。

## 已删除的 Shared-memory Duplicate Priority 历史实验

历史上曾基于“原始 all-candidate preclaim”新增实验 kernel：

```python
_select_kernel_preclaim_dedup(...)
```

这个 variant 保留主线语义：每个 warp 在每层仍然对所有合法候选边执行
`edge_inflight += 1`，用 virtual loss 参与 PUCT 打分。不同之处在 winner 产生后：

1. 每个 warp 将本层的 `(node, best_eid)` 写入 shared memory。
2. 若同一 block 内多个 warp 选择了相同 `(node, best_eid)`，则按 `wid` 决定优先级。
3. 低优先级 warp 会排除更高优先级 warp 已占用的 edge，在本层重新选择。
4. 最后仍然 rollback 非 winner 的 all-candidate preclaim。

实现上不能直接在原始 per-warp `while` 里插入 `syncthreads`，否则某些 warp 提前
到达 terminal/expand/depth-limit 后会造成 block barrier 死锁。因此该实验 kernel
使用统一 depth 循环：inactive warp 也参与 barrier，只有 active warp 执行本层选择。

多 warp 竞争 benchmark 结果：

```text
variant          scenario       trees  depth  acts  sel/s    same1(old)  samepath(old)
preclaim         eq256_d1024     4096   1024   256  238048       0.00%       0.00%
preclaim_dedup   eq256_d1024     4096   1024   256   94170       0.00%       0.00%
winner_only      eq256_d1024     4096   1024   256  345162     100.00%     100.00%
winner_reselect  eq256_d1024     4096   1024   256  337282       0.00%       0.00%
topk4            eq256_d1024     4096   1024   256   96118     100.00%      99.98%

preclaim         hot128_d1024    8192   1024   128  422909       0.00%       0.00%
preclaim_dedup   hot128_d1024    8192   1024   128  112390       0.00%       0.00%
winner_reselect  hot128_d1024    8192   1024   128  538758       0.00%       0.00%
```

结论：

- shared-memory duplicate priority 能消除“必然同路径”的极端塌缩，旧
  `samepath%` 与 preclaim 一样保持 0。当前应继续用 `maxpath%/path75%` 判断它是否
  只是降低了全塌缩，还是也降低了高概率重复。
- 但当前实现显著慢于原始 preclaim，主要成本来自每层 block-level barrier 和多轮
  shared-memory 重选；深度越大，这个固定同步成本越重。
- 这个版本证明了“按 warp id 做 deterministic duplicate resolution”的语义可行，
  但不是性能更优的实现形态，因此已从当前代码中移除。

下一步更值得探索的是把它改成局部、低同步的版本：

- 只在检测到高竞争节点或 high-prior 热点时启用 duplicate resolution。
- 使用一次性 shared-memory owner table，例如 `owner[edge] = min(wid)`，避免多轮
  reselect barrier。
- 对小 fanout 使用 warp ballot/match_any 风格的细粒度查重，避免 block-wide sync。
- 与 `winner_reselect` 结合：winner-only 快速路径先选 winner，只在 block 内重复
  winner 时做 shared-memory 去重和局部重选。

## CPU Parallel Reference 对照

新增无 CUDA 依赖的 CPU 对照：

```bash
bash src/run_puct_tests.sh --v2-cpu-ref
```

它参考 `src/ref.py` 的 `parallel_tree_search` 行为：一个 batch 内按 readout 顺序
执行 selection，每选到一个 leaf，就立即给选中路径加 virtual loss；等 batch 的
evaluation 完成后再 revert/backup。因此后续 readout 会看到前面 readout 留在路径
上的 virtual loss。

CPU 对照实现了三组 synthetic batch selection：

- `cpu_ref`: ref.py 风格，逐个 readout select，并立即给选中路径加 inflight。
- `preclaim`: 模拟 all-candidate preclaim，每层所有候选先整体 inflight。
- `winner_recalc`: 模拟 winner claim；若 winner 已 inflight，则排除原 winner
  重算一次。

核心指标：

- `first_TV`: first-edge 分布与 `cpu_ref` 的 total variation distance，越低越像。
- `path_match%`: 与 `cpu_ref` 对应 readout 的完整路径完全一致比例，越高越像。
- `maxpath/path75/samepath`: 路径是否高概率塌缩。

结果：

```text
scenario       variant          first_TV  path_match%  maxpath%  path75%  samepath%
equal64_d32    preclaim            0.000      100.00%    12.50%    0.00%     0.00%
equal64_d32    winner_recalc       0.000      100.00%    12.50%    0.00%     0.00%

hot64_d32      preclaim            0.750       25.00%   100.00%  100.00%   100.00%
hot64_d32      winner_recalc       0.125       87.50%    12.50%    0.00%     0.00%

hot128_d64     preclaim            0.750       25.00%   100.00%  100.00%   100.00%
hot128_d64     winner_recalc       0.125       87.50%    12.50%    0.00%     0.00%

hot256_d64     preclaim            0.750       25.00%   100.00%  100.00%   100.00%
hot256_d64     winner_recalc       0.125       87.50%    12.50%    0.00%     0.00%
```

聚合：

```text
preclaim_TV=2.250
winner_recalc_TV=0.375
preclaim_match=375.00
winner_recalc_match=562.50
```

结论：

- equal prior 场景下，两者都符合 CPU batch reference。
- hot prior 场景下，`preclaim` 因为所有候选边同时被预占，hot action 仍保持最高
  相对分数，导致 batch 内路径全塌缩；这不符合 `ref.py` 中“前一个 readout 的
  virtual loss 会影响后一个 readout”的预期。
- `winner_recalc` 更接近 CPU parallel reference：它不是严格等价于 ref.py 的逐个
  readout selection，但在 hot-prior 场景中 first-edge 分布和完整路径都明显更接近。

## 后续优化方向

1. 为深宽场景增加 adaptive selector：按 `cur_expanded`、`depth`、warp 数和
   recent collision rate 选择 preclaim 或 winner_recalc。
2. 扩展 benchmark 指标：记录每层 unique edge ratio，而不仅是 first edge。
3. 对 winner_recalc 做更严格的策略质量评估：不只看 select 吞吐，还要看最终 visit
   分布、value 收敛和 action choice 稳定性。
