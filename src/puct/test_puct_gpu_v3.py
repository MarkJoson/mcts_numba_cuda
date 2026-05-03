"""
Kernel-level tests and select-only experiments for puct_gpu_v3.py.

Usage:
    python test_puct_gpu_v3.py
    python test_puct_gpu_v3.py --stress
    python test_puct_gpu_v3.py --bench
    python test_puct_gpu_v3.py --scale-bench
    python test_puct_gpu_v3.py --cpu-bench
    python test_puct_gpu_v3.py --cpu-scale-bench
    python test_puct_gpu_v3.py --gpu-long-stress
    python test_puct_gpu_v3.py --gpu-long-stress-smoke
"""

import gc
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from numba import config
from numba import cuda
from numba import njit


PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
SKIP_MARK = "  [SKIP]"

RUN_STRESS = "--stress" in sys.argv
RUN_BENCH = "--bench" in sys.argv
RUN_SCALE_BENCH = "--scale-bench" in sys.argv
RUN_CPU_BENCH = "--cpu-bench" in sys.argv
RUN_CPU_SCALE_BENCH = "--cpu-scale-bench" in sys.argv
RUN_GPU_LONG_STRESS_SMOKE = "--gpu-long-stress-smoke" in sys.argv
RUN_GPU_LONG_STRESS = "--gpu-long-stress" in sys.argv or RUN_GPU_LONG_STRESS_SMOKE

results: list[tuple[str, bool]] = []

VARIANT_WINNER_RECALC = "winner_recalc"
VARIANT_WINNER_SOFT = "winner_soft"
SELECT_VARIANTS = [VARIANT_WINNER_RECALC, VARIANT_WINNER_SOFT]


def record(name: str, ok: bool, detail: str = ""):
    tag = PASS_MARK if ok else FAIL_MARK
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))


def skip_all(reason: str):
    print(f"{SKIP_MARK}  puct_gpu_v3 CUDA tests  ({reason})")
    sys.exit(0)


if not cuda.is_available():
    skip_all("CUDA driver/device is not available")
if getattr(config, "ENABLE_CUDASIM", False):
    skip_all("CUDA simulator does not implement the warp intrinsics used here")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import puct.puct_gpu_v3 as v3  # noqa: E402
try:
    import puct.puct_gpu_v3_cpp as v3_cpp  # noqa: E402
except Exception:
    v3_cpp = None


SELECT_BACKEND = os.environ.get("PUCT_V3_BACKEND", "numba").strip().lower()


def decode(raw: int):
    kind = (raw >> v3.PACKED_KIND_SHIFT) & v3.PACKED_KIND_MASK
    slot = (raw >> v3.PACKED_SLOT_SHIFT) & v3.PACKED_EDGE_MASK
    node = raw & v3.PACKED_NODE_MASK
    return int(kind), int(slot), int(node)


def make_case(trees=1, nodes=4, actions=4, warps=1, path_depth=4):
    edge_child = np.full((trees, nodes, actions), -1, np.int32)
    edge_prior = np.ones((trees, nodes, actions), np.float32)
    edge_w = np.zeros((trees, nodes, actions), np.float32)
    edge_n = np.zeros((trees, nodes, actions), np.int32)
    edge_inflight = np.zeros((trees, nodes, actions), np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    node_expanded = np.zeros((trees, nodes), np.int32)
    node_count = np.full((trees,), nodes, np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, max(1, path_depth)), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)
    return {
        "edge_child": edge_child,
        "edge_prior": edge_prior,
        "edge_w": edge_w,
        "edge_n": edge_n,
        "edge_inflight": edge_inflight,
        "node_expand_inflight": node_expand_inflight,
        "node_expanded": node_expanded,
        "node_count": node_count,
        "out_selected": out_selected,
        "out_path": out_path,
        "out_len": out_len,
        "warps": warps,
        "trees": trees,
    }


def make_deep_chain_case(trees=1, tree_depth=64, actions=4, warps=1, shape="narrow"):
    """
    Build a synthetic deep traversal workload.

    narrow: one legal child per node, good for pure depth traversal pressure.
    wide: every expanded node has `actions` legal children, and sibling actions
    point to distinct child node ids. The graph is recurrent across depths to
    avoid exponential allocation while preserving per-action child identity.
    """
    nodes = tree_depth + 1 if shape == "narrow" else actions + 1
    case = make_case(
        trees=trees,
        nodes=nodes,
        actions=actions,
        warps=warps,
        path_depth=max(1, tree_depth),
    )
    fanout = 1 if shape == "narrow" else actions
    if shape == "narrow":
        for depth in range(tree_depth):
            case["node_expanded"][:, depth] = fanout
            case["edge_child"][:, depth, 0] = depth + 1
            case["edge_n"][:, depth, 0] = 1
            case["edge_prior"][:, depth, 0] = 1.0
        case["node_expanded"][:, tree_depth] = 0
    else:
        case["node_expanded"][:, :] = fanout
        for node in range(nodes):
            for action in range(fanout):
                case["edge_child"][:, node, action] = action + 1
                case["edge_n"][:, node, action] = 1
                case["edge_prior"][:, node, action] = 1.0
            case["edge_prior"][:, node, 0] = 2.0
    return case


@cuda.jit
def _init_deep_chain_edges_kernel(
    tree_depth,
    actions,
    fanout,
    unique_wide,
    layered_wide,
    prior_mode,
    terminal_leaf,
    terminal_value,
    edge_child,
    edge_prior,
    edge_w,
    edge_n,
    edge_inflight,
    node_expand_inflight,
    node_expanded,
    node_count,
):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = edge_child.size
    nodes = edge_child.shape[1]
    while idx < total:
        action = idx % actions
        node = (idx // actions) % nodes
        tree = idx // (actions * nodes)
        legal = action < fanout
        if layered_wide != 0:
            level = node // actions
            pos = node - level * actions
            if node == 0:
                level = np.int32(0)
                pos = np.int32(0)
            legal = level < tree_depth and action < fanout
            child = (level + np.int32(1)) * actions + action
        elif unique_wide == 0:
            legal = node < tree_depth and action < fanout
            child = node + 1
        else:
            child = action + 1
        edge_child[tree, node, action] = child if legal else -1
        edge_prior[tree, node, action] = 2.0 if legal and action == 0 and prior_mode == 1 else 1.0
        edge_w[tree, node, action] = 0.0
        edge_n[tree, node, action] = 1 if legal else 0
        edge_inflight[tree, node, action] = 0
        if action == 0:
            node_expand_inflight[tree, node] = 0
            if node == 0:
                node_count[tree] = nodes
            if layered_wide != 0:
                level = node // actions
                node_expanded[tree, node] = fanout if level < tree_depth else terminal_value
            elif unique_wide != 0:
                node_expanded[tree, node] = fanout
            elif node < tree_depth:
                node_expanded[tree, node] = fanout
            elif terminal_leaf == 1:
                node_expanded[tree, node] = terminal_value
            else:
                node_expanded[tree, node] = 0
        idx += stride


@cuda.jit
def _init_select_outputs_kernel(packed_invalid, out_selected, out_path, out_len):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total_path = out_path.size
    while idx < total_path:
        tree = idx // (out_path.shape[1] * out_path.shape[2])
        rem = idx - tree * out_path.shape[1] * out_path.shape[2]
        warp = rem // out_path.shape[2]
        depth = rem - warp * out_path.shape[2]
        out_path[tree, warp, depth] = -1
        idx += stride

    idx = cuda.grid(1)
    total_out = out_selected.size
    while idx < total_out:
        tree = idx // out_selected.shape[1]
        warp = idx - tree * out_selected.shape[1]
        out_selected[tree, warp] = packed_invalid
        out_len[tree, warp] = 0
        idx += stride


@cuda.jit
def _check_inflight_2d_kernel(inflight, stats):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = inflight.size
    nodes = inflight.shape[1]
    while idx < total:
        node = idx % nodes
        tree = idx // nodes
        val = inflight[tree, node]
        if val < 0:
            cuda.atomic.add(stats, 0, 1)
        if val != 0:
            cuda.atomic.add(stats, 1, 1)
        idx += stride


@cuda.jit
def _check_inflight_3d_kernel(inflight, stats):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = inflight.size
    edges = inflight.shape[2]
    nodes = inflight.shape[1]
    while idx < total:
        action = idx % edges
        rem = idx // edges
        node = rem % nodes
        tree = rem // nodes
        val = inflight[tree, node, action]
        if val < 0:
            cuda.atomic.add(stats, 0, 1)
        if val != 0:
            cuda.atomic.add(stats, 1, 1)
        idx += stride


@cuda.jit
def _release_claims_kernel(
    virtual_loss,
    edge_inflight,
    node_expand_inflight,
    node_count,
    out_selected,
    out_path,
    out_len,
):
    tree = cuda.blockIdx.x
    lane = cuda.threadIdx.x & np.int32(31)
    wid = cuda.threadIdx.x >> np.int32(5)

    if tree >= out_selected.shape[0]:
        return
    if wid >= min(cuda.blockDim.x >> np.int32(5), out_selected.shape[1]):
        return

    node_limit = np.int32(0)
    if lane == np.int32(0):
        node_limit = node_count[tree]
    node_limit = cuda.shfl_sync(v3.FULL_MASK, node_limit, 0)
    if node_limit <= np.int32(0) or node_limit > edge_inflight.shape[1]:
        return

    raw_selected = out_selected[tree, wid]
    if raw_selected < np.int32(0):
        return

    kind = (raw_selected >> np.int32(v3.PACKED_KIND_SHIFT)) & np.int32(v3.PACKED_KIND_MASK)
    if kind == np.int32(v3.SELECT_INVALID) or kind == np.int32(v3.SELECT_BUSY):
        return

    plen = out_len[tree, wid]
    if plen <= np.int32(0):
        return

    edge_count = plen - np.int32(1)
    max_edges = edge_inflight.shape[2]
    d = lane
    while d < edge_count:
        encoded = out_path[tree, wid, d]
        if encoded >= np.int32(0):
            parent = encoded >> np.int32(8)
            slot = encoded & np.int32(0xFF)
            if (
                parent >= np.int32(0)
                and parent < node_limit
                and slot >= np.int32(0)
                and slot < max_edges
            ):
                cuda.atomic.sub(edge_inflight, (tree, parent, slot), virtual_loss)
        d += np.int32(v3.WARP_SIZE)

    if lane == np.int32(0) and kind == np.int32(v3.SELECT_EXPAND):
        leaf = raw_selected & np.int32(v3.PACKED_NODE_MASK)
        if leaf >= np.int32(0) and leaf < node_limit:
            cuda.atomic.sub(node_expand_inflight, (tree, leaf), np.int32(1))


def flush_cuda_deallocations():
    gc.collect()
    try:
        cuda.current_context().deallocations.clear()
    except Exception:
        pass


def estimate_deep_chain_bytes(trees, tree_depth, actions, warps):
    nodes = actions + 1
    edge_bytes = trees * nodes * actions * 20
    node_bytes = trees * nodes * 8 + trees * 4
    output_bytes = trees * warps * 8 + trees * warps * max(1, tree_depth) * 4
    return edge_bytes + node_bytes + output_bytes


def estimate_deep_case_bytes(trees, tree_depth, actions, warps, shape):
    nodes = tree_depth + 1 if shape == "narrow" else actions + 1
    if shape == "layered":
        nodes = (tree_depth + 1) * actions
    edge_bytes = trees * nodes * actions * 20
    node_bytes = trees * nodes * 8 + trees * 4
    output_bytes = trees * warps * 8 + trees * warps * max(1, tree_depth) * 4
    return edge_bytes + node_bytes + output_bytes


def make_deep_chain_case_device(
    trees=1,
    tree_depth=64,
    actions=4,
    warps=1,
    shape="wide",
    prior_mode="hot",
    terminal_leaf=False,
):
    nodes = tree_depth + 1 if shape == "narrow" else actions + 1
    if shape == "layered":
        nodes = (tree_depth + 1) * actions
    fanout = 1 if shape == "narrow" else actions
    d = {
        "edge_child": cuda.device_array((trees, nodes, actions), np.int32),
        "edge_prior": cuda.device_array((trees, nodes, actions), np.float32),
        "edge_w": cuda.device_array((trees, nodes, actions), np.float32),
        "edge_n": cuda.device_array((trees, nodes, actions), np.int32),
        "edge_inflight": cuda.device_array((trees, nodes, actions), np.int32),
        "node_expand_inflight": cuda.device_array((trees, nodes), np.int32),
        "node_expanded": cuda.device_array((trees, nodes), np.int32),
        "node_count": cuda.device_array((trees,), np.int32),
        "out_selected": cuda.device_array((trees, warps), np.int32),
        "out_path": cuda.device_array((trees, warps, max(1, tree_depth)), np.int32),
        "out_len": cuda.device_array((trees, warps), np.int32),
        "warps": warps,
        "trees": trees,
    }
    threads = 256
    edge_blocks = min(65535, max(1, (d["edge_child"].size + threads - 1) // threads))
    out_blocks = min(65535, max(1, (d["out_path"].size + threads - 1) // threads))
    _init_deep_chain_edges_kernel[edge_blocks, threads](
        np.int32(tree_depth),
        np.int32(actions),
        np.int32(fanout),
        np.int32(1 if shape != "narrow" else 0),
        np.int32(1 if shape == "layered" else 0),
        np.int32(1 if prior_mode == "hot" else 0),
        np.int32(1 if terminal_leaf else 0),
        np.int32(v3.NODE_EXPANDED_TERMINAL),
        d["edge_child"],
        d["edge_prior"],
        d["edge_w"],
        d["edge_n"],
        d["edge_inflight"],
        d["node_expand_inflight"],
        d["node_expanded"],
        d["node_count"],
    )
    _init_select_outputs_kernel[out_blocks, threads](
        np.int32(v3.PACKED_INVALID),
        d["out_selected"],
        d["out_path"],
        d["out_len"],
    )
    return d


def to_device(case):
    return {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}


def copy_back(dcase):
    return {k: v.copy_to_host() if hasattr(v, "copy_to_host") else v for k, v in dcase.items()}


def launch_select(d, cpuct=1.0, c_pw=1.0, alpha_pw=0.5, variant=VARIANT_WINNER_RECALC):
    if variant not in SELECT_VARIANTS:
        raise ValueError(f"unknown select variant for v3: {variant}")
    if SELECT_BACKEND == "cpp":
        if v3_cpp is None:
            raise RuntimeError("PUCT_V3_BACKEND=cpp requested but puct_gpu_v3_cpp could not be imported")
        v3_cpp.launch_select(
            d,
            cpuct=cpuct,
            c_pw=c_pw,
            alpha_pw=alpha_pw,
            soft_winner=1 if variant == VARIANT_WINNER_SOFT else 0,
        )
        return
    if SELECT_BACKEND != "numba":
        raise ValueError(f"unknown PUCT_V3_BACKEND: {SELECT_BACKEND}")
    v3._select_kernel_winner_recalc[d["trees"], d["warps"] * v3.WARP_SIZE](
        np.float32(cpuct),
        np.float32(c_pw),
        np.float32(alpha_pw),
        np.int32(1 if variant == VARIANT_WINNER_SOFT else 0),
        d["edge_child"],
        d["edge_prior"],
        d["edge_w"],
        d["edge_n"],
        d["edge_inflight"],
        d["node_expand_inflight"],
        d["node_expanded"],
        d["node_count"],
        d["out_selected"],
        d["out_path"],
        d["out_len"],
    )


def launch_release(dcase, trees=None, warps=None, virtual_loss=1):
    if trees is None:
        trees = dcase["trees"]
    if warps is None:
        warps = dcase["warps"]
    _release_claims_kernel[trees, warps * v3.WARP_SIZE](
        np.int32(virtual_loss),
        dcase["edge_inflight"],
        dcase["node_expand_inflight"],
        dcase["node_count"],
        dcase["out_selected"],
        dcase["out_path"],
        dcase["out_len"],
    )


def run_select(case, cpuct=1.0, c_pw=1.0, alpha_pw=0.5, variant=VARIANT_WINNER_RECALC):
    d = to_device(case)
    launch_select(d, cpuct=cpuct, c_pw=c_pw, alpha_pw=alpha_pw, variant=variant)
    cuda.synchronize()
    return d, copy_back(d)


def run_release(dcase, virtual_loss=1):
    launch_release(dcase, virtual_loss=virtual_loss)
    cuda.synchronize()
    return copy_back(dcase)


def _env_int(name, default, minimum=1):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _env_float(name, default, minimum=0.0):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _float_or_none(value):
    try:
        if value.strip().upper() == "N/A":
            return None
        return float(value)
    except Exception:
        return None


def is_valid_selection_kind(kind: int) -> bool:
    return kind in (v3.SELECT_EXPAND, v3.SELECT_TERMINAL, v3.SELECT_DEPTH_LIMIT)


def query_gpu_telemetry_once(gpu_index):
    if shutil.which("nvidia-smi") is None:
        return None
    fields = [
        "timestamp",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "utilization.memory",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "pstate",
        "memory.used",
        "memory.total",
    ]
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3.0)
    except Exception:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(fields):
        return None
    return {
        "timestamp": parts[0],
        "power_w": _float_or_none(parts[1]),
        "power_limit_w": _float_or_none(parts[2]),
        "gpu_util_pct": _float_or_none(parts[3]),
        "mem_util_pct": _float_or_none(parts[4]),
        "sm_clock_mhz": _float_or_none(parts[5]),
        "mem_clock_mhz": _float_or_none(parts[6]),
        "temp_c": _float_or_none(parts[7]),
        "pstate": parts[8],
        "mem_used_mib": _float_or_none(parts[9]),
        "mem_total_mib": _float_or_none(parts[10]),
        "t_wall": time.perf_counter(),
    }


def start_gpu_telemetry_sampler(gpu_index, interval_sec):
    samples = []
    stop_event = threading.Event()

    def sample_loop():
        while not stop_event.is_set():
            sample = query_gpu_telemetry_once(gpu_index)
            if sample is not None:
                samples.append(sample)
            stop_event.wait(interval_sec)

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    return stop_event, thread, samples


def summarize_gpu_telemetry(samples):
    def values(key):
        return [float(s[key]) for s in samples if s.get(key) is not None]

    def avg(vals):
        return sum(vals) / float(len(vals)) if vals else float("nan")

    def peak(vals):
        return max(vals) if vals else float("nan")

    gpu_util = values("gpu_util_pct")
    mem_util = values("mem_util_pct")
    power = values("power_w")
    power_limit = values("power_limit_w")
    sm_clock = values("sm_clock_mhz")
    temp = values("temp_c")
    mem_used = values("mem_used_mib")
    avg_power = avg(power)
    avg_power_limit = avg(power_limit)
    return {
        "samples": len(samples),
        "gpu_util_avg": avg(gpu_util),
        "gpu_util_max": peak(gpu_util),
        "idle_avg": 100.0 - avg(gpu_util) if gpu_util else float("nan"),
        "mem_util_avg": avg(mem_util),
        "power_avg": avg_power,
        "power_max": peak(power),
        "power_limit_avg": avg_power_limit,
        "power_limit_pct": (
            (avg_power / avg_power_limit * 100.0)
            if power and power_limit and avg_power_limit > 0.0
            else float("nan")
        ),
        "sm_clock_avg": avg(sm_clock),
        "temp_max": peak(temp),
        "mem_used_max_gib": peak(mem_used) / 1024.0 if mem_used else float("nan"),
    }


def fmt_float(value, digits=1):
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _run_select_release_once(d, trees=None, warps=None, c_pw=1.0, variant=VARIANT_WINNER_RECALC):
    launch_select(d, c_pw=c_pw, variant=variant)
    launch_release(d, trees=trees, warps=warps)


def time_select_kernel_only(d, trees, warps, c_pw, variant, warmup, iterations):
    for _ in range(warmup):
        _run_select_release_once(d, trees=trees, warps=warps, c_pw=c_pw, variant=variant)
    cuda.synchronize()

    events = []
    for _ in range(iterations):
        start = cuda.event()
        end = cuda.event()
        start.record()
        launch_select(d, c_pw=c_pw, variant=variant)
        end.record()
        launch_release(d, trees=trees, warps=warps)
        events.append((start, end))
    cuda.synchronize()
    elapsed_ms = 0.0
    for start, end in events:
        elapsed_ms += cuda.event_elapsed_time(start, end)
    return elapsed_ms


def time_select_kernel_only_chunked(
    d, trees, warps, c_pw, variant, warmup, iterations, chunk_size, progress_every=0
):
    for _ in range(warmup):
        _run_select_release_once(d, trees=trees, warps=warps, c_pw=c_pw, variant=variant)
    cuda.synchronize()

    elapsed_ms = 0.0
    completed = 0
    wall_start = time.perf_counter()
    while completed < iterations:
        cur = min(chunk_size, iterations - completed)
        events = []
        for _ in range(cur):
            start = cuda.event()
            end = cuda.event()
            start.record()
            launch_select(d, c_pw=c_pw, variant=variant)
            end.record()
            launch_release(d, trees=trees, warps=warps)
            events.append((start, end))
        cuda.synchronize()
        for start, end in events:
            elapsed_ms += cuda.event_elapsed_time(start, end)
        completed += cur
        if progress_every > 0 and (completed == iterations or completed % progress_every == 0):
            wall_s = time.perf_counter() - wall_start
            print(
                f"    progress: {completed}/{iterations} timed select launches, wall={wall_s:.1f}s",
                flush=True,
            )

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return elapsed_ms, wall_ms


def _entropy_from_counts(counts):
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    ent = 0.0
    for count in counts:
        if count > 0:
            p = float(count) / total
            ent -= p * np.log(p)
    return float(ent)


def intra_tree_diversity_metrics(out_path, out_len):
    trees = out_path.shape[0]
    warps = out_path.shape[1]
    same_first_trees = 0
    same_path_trees = 0
    dominant_first_75_trees = 0
    dominant_path_75_trees = 0
    considered_trees = 0
    first_unique_sum = 0.0
    path_unique_sum = 0.0
    dominant_first_sum = 0.0
    dominant_path_sum = 0.0
    for tree in range(trees):
        first_slots = []
        paths = []
        for wid in range(warps):
            plen = int(out_len[tree, wid]) - 1
            if plen > 0 and int(out_path[tree, wid, 0]) >= 0:
                first_slots.append(int(out_path[tree, wid, 0]) & v3.PACKED_EDGE_MASK)
                paths.append(tuple(int(out_path[tree, wid, d]) for d in range(plen)))
        valid = len(first_slots)
        if valid > 1:
            considered_trees += 1
            first_unique = len(set(first_slots))
            path_unique = len(set(paths))
            first_unique_sum += float(first_unique) / float(valid)
            path_unique_sum += float(path_unique) / float(valid)

            first_counts = {}
            for slot in first_slots:
                first_counts[slot] = first_counts.get(slot, 0) + 1
            path_counts = {}
            for path in paths:
                path_counts[path] = path_counts.get(path, 0) + 1

            dominant_first = float(max(first_counts.values())) / float(valid)
            dominant_path = float(max(path_counts.values())) / float(valid)
            dominant_first_sum += dominant_first
            dominant_path_sum += dominant_path
            if dominant_first >= 0.75:
                dominant_first_75_trees += 1
            if dominant_path >= 0.75:
                dominant_path_75_trees += 1
            if first_unique == 1:
                same_first_trees += 1
            if path_unique == 1:
                same_path_trees += 1

    denom = max(1, considered_trees)
    return {
        "tree_same_first_pct": 100.0 * float(same_first_trees) / float(denom),
        "tree_same_path_pct": 100.0 * float(same_path_trees) / float(denom),
        "tree_first_unique_pct": 100.0 * first_unique_sum / float(denom),
        "tree_path_unique_pct": 100.0 * path_unique_sum / float(denom),
        "tree_dominant_first_pct": 100.0 * dominant_first_sum / float(denom),
        "tree_dominant_path_pct": 100.0 * dominant_path_sum / float(denom),
        "tree_dominant_first_75_pct": 100.0 * float(dominant_first_75_trees) / float(denom),
        "tree_dominant_path_75_pct": 100.0 * float(dominant_path_75_trees) / float(denom),
    }


def selection_metrics(h, action_count):
    out_path = h["out_path"]
    out_len = h["out_len"]
    raws = h["out_selected"].reshape(-1)
    invalid = sum(1 for raw in raws if decode(int(raw))[0] == v3.SELECT_INVALID)
    busy = sum(1 for raw in raws if decode(int(raw))[0] == v3.SELECT_BUSY)
    invalid_pct = 100.0 * invalid / max(1, len(raws))
    busy_pct = 100.0 * busy / max(1, len(raws))

    valid_first = []
    for tree in range(out_path.shape[0]):
        for wid in range(out_path.shape[1]):
            kind = decode(int(h["out_selected"][tree, wid]))[0]
            if is_valid_selection_kind(kind) and int(out_len[tree, wid]) > 1:
                encoded = int(out_path[tree, wid, 0])
                if encoded >= 0:
                    valid_first.append(encoded)
    unique_first = len(set(int(x) & v3.PACKED_EDGE_MASK for x in valid_first))
    collision_pct = 0.0
    if len(valid_first) > 0:
        collision_pct = 100.0 * (1.0 - float(unique_first) / float(len(valid_first)))

    slot_counts = [0 for _ in range(max(1, action_count))]
    for encoded in valid_first:
        slot = int(encoded) & v3.PACKED_EDGE_MASK
        if 0 <= slot < len(slot_counts):
            slot_counts[slot] += 1
    root_entropy = _entropy_from_counts(slot_counts)
    root_kl_uniform = np.log(float(max(1, action_count))) - root_entropy

    ratios = []
    max_edges = out_path.shape[2]
    for d in range(max_edges):
        vals = []
        for tree in range(out_path.shape[0]):
            for wid in range(out_path.shape[1]):
                kind = decode(int(h["out_selected"][tree, wid]))[0]
                if is_valid_selection_kind(kind) and out_len[tree, wid] > d + 1:
                    encoded = int(out_path[tree, wid, d])
                    if encoded >= 0:
                        vals.append(encoded)
        if vals:
            ratios.append(float(len(set(vals))) / float(len(vals)))
    mean_depth_unique_pct = 100.0 * float(np.mean(ratios)) if ratios else 0.0

    path_counts = {}
    for tree in range(out_path.shape[0]):
        for wid in range(out_path.shape[1]):
            kind = decode(int(h["out_selected"][tree, wid]))[0]
            plen = int(out_len[tree, wid]) - 1
            if is_valid_selection_kind(kind) and plen > 0:
                path = tuple(int(out_path[tree, wid, d]) for d in range(plen))
                path_counts[path] = path_counts.get(path, 0) + 1
    path_entropy = _entropy_from_counts(path_counts.values())
    path_entropy_norm = 0.0
    if len(path_counts) > 1:
        path_entropy_norm = path_entropy / np.log(float(len(path_counts)))

    intra = intra_tree_diversity_metrics(out_path, out_len)
    metrics = {
        "invalid_pct": invalid_pct,
        "busy_pct": busy_pct,
        "collision_pct": collision_pct,
        "unique_first": unique_first,
        "root_kl": float(root_kl_uniform),
        "depth_unique_pct": mean_depth_unique_pct,
        "path_entropy_norm": float(path_entropy_norm),
    }
    metrics.update(intra)
    return metrics


def selection_metrics_scale(h, action_count, max_path_samples=2048, max_depth_samples=64):
    out_path = h["out_path"]
    out_len = h["out_len"]
    raws = h["out_selected"].reshape(-1)
    invalid = sum(1 for raw in raws if decode(int(raw))[0] == v3.SELECT_INVALID)
    busy = sum(1 for raw in raws if decode(int(raw))[0] == v3.SELECT_BUSY)
    invalid_pct = 100.0 * invalid / max(1, len(raws))
    busy_pct = 100.0 * busy / max(1, len(raws))

    valid_first = []
    for tree in range(out_path.shape[0]):
        for wid in range(out_path.shape[1]):
            kind = decode(int(h["out_selected"][tree, wid]))[0]
            if is_valid_selection_kind(kind) and int(out_len[tree, wid]) > 1:
                encoded = int(out_path[tree, wid, 0])
                if encoded >= 0:
                    valid_first.append(encoded)
    unique_first = len(set(int(x) & v3.PACKED_EDGE_MASK for x in valid_first))
    collision_pct = 0.0
    if len(valid_first) > 0:
        collision_pct = 100.0 * (1.0 - float(unique_first) / float(len(valid_first)))

    slot_counts = [0 for _ in range(max(1, action_count))]
    for encoded in valid_first:
        slot = int(encoded) & v3.PACKED_EDGE_MASK
        if 0 <= slot < len(slot_counts):
            slot_counts[slot] += 1
    root_entropy = _entropy_from_counts(slot_counts)
    root_kl_uniform = np.log(float(max(1, action_count))) - root_entropy

    trees = out_path.shape[0]
    warps = out_path.shape[1]
    max_edges = out_path.shape[2]
    depth_count = min(max_depth_samples, max_edges)
    depth_stride = max(1, max_edges // max(1, depth_count))
    ratios = []
    for d in range(0, max_edges, depth_stride):
        vals = []
        for tree in range(trees):
            for wid in range(warps):
                kind = decode(int(h["out_selected"][tree, wid]))[0]
                if is_valid_selection_kind(kind) and out_len[tree, wid] > d + 1:
                    encoded = int(out_path[tree, wid, d])
                    if encoded >= 0:
                        vals.append(encoded)
        if vals:
            ratios.append(float(len(set(vals))) / float(len(vals)))
        if len(ratios) >= depth_count:
            break
    mean_depth_unique_pct = 100.0 * float(np.mean(ratios)) if ratios else 0.0

    sample_count = min(max_path_samples, trees * warps)
    step = max(1, (trees * warps) // max(1, sample_count))
    path_counts = {}
    flat_index = 0
    used = 0
    while flat_index < trees * warps and used < sample_count:
        tree = flat_index // warps
        wid = flat_index - tree * warps
        kind = decode(int(h["out_selected"][tree, wid]))[0]
        plen = int(out_len[tree, wid]) - 1
        if is_valid_selection_kind(kind) and plen > 0:
            path = tuple(int(out_path[tree, wid, d]) for d in range(plen))
            path_counts[path] = path_counts.get(path, 0) + 1
            used += 1
        flat_index += step
    path_entropy = _entropy_from_counts(path_counts.values())
    path_entropy_norm = 0.0
    if len(path_counts) > 1:
        path_entropy_norm = path_entropy / np.log(float(len(path_counts)))

    intra = intra_tree_diversity_metrics(out_path, out_len)
    metrics = {
        "invalid_pct": invalid_pct,
        "busy_pct": busy_pct,
        "collision_pct": collision_pct,
        "unique_first": unique_first,
        "root_kl": float(root_kl_uniform),
        "depth_unique_pct": mean_depth_unique_pct,
        "path_entropy_norm": float(path_entropy_norm),
    }
    metrics.update(intra)
    return metrics


def copy_selection_outputs(dcase):
    return {
        "out_selected": dcase["out_selected"].copy_to_host(),
        "out_path": dcase["out_path"].copy_to_host(),
        "out_len": dcase["out_len"].copy_to_host(),
    }


def check_inflight_empty(dcase):
    stats = cuda.to_device(np.zeros(2, np.int32))
    threads = 256

    edge_blocks = min(65535, max(1, (dcase["edge_inflight"].size + threads - 1) // threads))
    _check_inflight_3d_kernel[edge_blocks, threads](dcase["edge_inflight"], stats)

    node_blocks = min(65535, max(1, (dcase["node_expand_inflight"].size + threads - 1) // threads))
    _check_inflight_2d_kernel[node_blocks, threads](dcase["node_expand_inflight"], stats)

    cuda.synchronize()
    h = stats.copy_to_host()
    return int(h[0]) == 0 and int(h[1]) == 0, int(h[0]), int(h[1])


@njit(cache=True, fastmath=True)
def _cpu_select_sequential_no_vloss(
    cpuct,
    edge_child,
    edge_prior,
    edge_w,
    edge_n,
    node_expanded,
    traversals_per_tree,
    max_depth,
):
    trees = edge_child.shape[0]
    n_nodes = node_expanded.shape[1]
    checksum = np.int64(0)
    total_path_len = np.int64(0)
    valid = np.int64(0)

    for tree in range(trees):
        for _ in range(traversals_per_tree):
            node = 0
            depth = 0
            while True:
                if node < 0 or node >= n_nodes:
                    break
                if depth >= max_depth:
                    total_path_len += depth + 1
                    valid += 1
                    checksum += node + depth
                    break

                node_info = node_expanded[tree, node]
                if node_info == v3.NODE_EXPANDED_TERMINAL:
                    total_path_len += depth + 1
                    valid += 1
                    checksum += node + 17
                    break
                if node_info <= 0:
                    total_path_len += depth + 1
                    valid += 1
                    checksum += node + 31
                    break

                parent_n = 0
                for eid in range(node_info):
                    child = edge_child[tree, node, eid]
                    if child >= 0 and child < n_nodes:
                        parent_n += edge_n[tree, node, eid]
                if parent_n < 1:
                    parent_n = 1
                sqrt_parent = np.sqrt(np.float32(parent_n))

                best_score = np.float32(-3.4028234663852886e38)
                best_eid = -1
                best_child = -1
                for eid in range(node_info):
                    child = edge_child[tree, node, eid]
                    if child >= 0 and child < n_nodes:
                        n_edge = edge_n[tree, node, eid]
                        q = np.float32(0.0)
                        if n_edge > 0:
                            q = edge_w[tree, node, eid] / np.float32(n_edge)
                        score = q + cpuct * edge_prior[tree, node, eid] * sqrt_parent / np.float32(n_edge + 1)
                        if score > best_score or (score == best_score and eid < best_eid):
                            best_score = score
                            best_eid = eid
                            best_child = child

                if best_eid < 0:
                    break
                checksum += (node << 8) | best_eid
                node = best_child
                depth += 1

    return checksum, total_path_len, valid


def time_cpu_select_no_vloss(case, traversals_per_tree, warmup, repeats, cpuct=1.0):
    max_depth = case["out_path"].shape[2]
    best_ms = None
    checksum = np.int64(0)
    total_path_len = np.int64(0)
    valid = np.int64(0)
    for _ in range(warmup):
        checksum, total_path_len, valid = _cpu_select_sequential_no_vloss(
            np.float32(cpuct),
            case["edge_child"],
            case["edge_prior"],
            case["edge_w"],
            case["edge_n"],
            case["node_expanded"],
            np.int32(traversals_per_tree),
            np.int32(max_depth),
        )
    for _ in range(repeats):
        start = time.perf_counter()
        checksum, total_path_len, valid = _cpu_select_sequential_no_vloss(
            np.float32(cpuct),
            case["edge_child"],
            case["edge_prior"],
            case["edge_w"],
            case["edge_n"],
            case["node_expanded"],
            np.int32(traversals_per_tree),
            np.int32(max_depth),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if best_ms is None or elapsed_ms < best_ms:
            best_ms = elapsed_ms
    return float(best_ms if best_ms is not None else 0.0), int(checksum), int(total_path_len), int(valid)


@njit(cache=True, fastmath=True)
def _cpu_select_deep_wide_procedural(
    cpuct,
    trees,
    actions,
    depth_limit,
    traversals_per_tree,
    prior_hot,
):
    checksum = np.int64(0)
    total_path_len = np.int64(0)
    valid = np.int64(0)

    for tree in range(trees):
        for traversal in range(traversals_per_tree):
            node = 0
            depth = 0
            while depth < depth_limit:
                parent_n = 0
                for _ in range(actions):
                    parent_n += 1
                sqrt_parent = np.sqrt(np.float32(parent_n))

                best_score = np.float32(-3.4028234663852886e38)
                best_eid = -1
                for eid in range(actions):
                    prior = np.float32(1.0)
                    if prior_hot != 0 and eid == 0:
                        prior = np.float32(2.0)
                    score = cpuct * prior * sqrt_parent * np.float32(0.5)
                    if score > best_score or (score == best_score and eid < best_eid):
                        best_score = score
                        best_eid = eid

                if best_eid < 0:
                    break
                checksum += (
                    np.int64((node << 8) | best_eid)
                    + np.int64(tree & 7)
                    + np.int64(traversal & 3)
                )
                node = best_eid + 1
                depth += 1

            total_path_len += np.int64(depth + 1)
            valid += 1

    return checksum, total_path_len, valid


def time_cpu_deep_wide_procedural(
    trees,
    actions,
    depth,
    traversals_per_tree,
    prior_mode,
    warmup,
    repeats,
    cpuct=1.0,
):
    best_ms = None
    checksum = np.int64(0)
    total_path_len = np.int64(0)
    valid = np.int64(0)
    prior_hot = np.int32(1 if prior_mode == "hot" else 0)
    for _ in range(warmup):
        checksum, total_path_len, valid = _cpu_select_deep_wide_procedural(
            np.float32(cpuct),
            np.int32(trees),
            np.int32(actions),
            np.int32(depth),
            np.int32(traversals_per_tree),
            prior_hot,
        )
    for _ in range(repeats):
        start = time.perf_counter()
        checksum, total_path_len, valid = _cpu_select_deep_wide_procedural(
            np.float32(cpuct),
            np.int32(trees),
            np.int32(actions),
            np.int32(depth),
            np.int32(traversals_per_tree),
            prior_hot,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if best_ms is None or elapsed_ms < best_ms:
            best_ms = elapsed_ms
    return float(best_ms if best_ms is not None else 0.0), int(checksum), int(total_path_len), int(valid)


def host_inflight_sum(h):
    return int(h["edge_inflight"].sum()) + int(h["node_expand_inflight"].sum())


def host_inflight_nonnegative(h):
    return (
        not np.any(h["edge_inflight"] < 0)
        and not np.any(h["node_expand_inflight"] < 0)
    )


def test_fresh_root_expand_roundtrip():
    case = make_case(nodes=2, actions=4, path_depth=2)
    d, h = run_select(case)
    kind, slot, node = decode(int(h["out_selected"][0, 0]))
    inflight_after_select = int(h["node_expand_inflight"][0, 0])
    h2 = run_release(d)
    ok = (
        kind == v3.SELECT_EXPAND
        and slot == 0
        and node == 0
        and int(h["out_len"][0, 0]) == 1
        and inflight_after_select == 1
        and host_inflight_sum(h2) == 0
    )
    record("fresh root expands with packed slot and releases claim", ok)


def test_terminal_root():
    case = make_case(nodes=2, actions=4, path_depth=2)
    case["node_expanded"][0, 0] = v3.NODE_EXPANDED_TERMINAL
    _, h = run_select(case)
    kind, _, node = decode(int(h["out_selected"][0, 0]))
    ok = (
        kind == v3.SELECT_TERMINAL
        and node == 0
        and int(h["out_len"][0, 0]) == 1
        and host_inflight_sum(h) == 0
    )
    record("terminal root returns terminal without virtual loss", ok)


def test_invalid_child_rolls_back_path():
    case = make_case(nodes=2, actions=4, path_depth=4)
    case["node_expanded"][0, 0] = 1
    case["node_expanded"][0, 1] = -2
    case["edge_child"][0, 0, 0] = 1
    case["edge_prior"][0, 0, 0] = 1.0
    case["edge_n"][0, 0, 0] = 1
    _, h = run_select(case)
    kind, _, _ = decode(int(h["out_selected"][0, 0]))
    ok = kind == v3.SELECT_INVALID and host_inflight_sum(h) == 0
    record("invalid child node rolls back selected path virtual loss", ok)


def test_child_beyond_node_count_invalid():
    case = make_case(nodes=4, actions=4, path_depth=3)
    case["node_count"][0] = 2
    case["node_expanded"][0, 0] = 1
    case["edge_child"][0, 0, 0] = 3
    case["edge_n"][0, 0, 0] = 1
    _, h = run_select(case)
    kind, _, _ = decode(int(h["out_selected"][0, 0]))
    ok = kind == v3.SELECT_INVALID and host_inflight_sum(h) == 0
    record("child id beyond per-tree node_count is invalid", ok)


def test_progressive_widening_multi_warp_claims_unique_slots():
    case = make_case(nodes=3, actions=4, warps=4, path_depth=2)
    case["node_expanded"][0, 0] = 1
    case["edge_child"][0, 0, 0] = 1
    case["edge_n"][0, 0, 0] = 16
    d, h = run_select(case, c_pw=1.0, alpha_pw=0.5)

    slots = []
    root_slots = []
    child_expands = 0
    for wid in range(4):
        kind, slot, node = decode(int(h["out_selected"][0, wid]))
        if kind == v3.SELECT_EXPAND:
            slots.append(slot)
            if node == 0:
                root_slots.append(slot)
            elif node == 1:
                child_expands += 1
            ok_node = node == 0 or node == 1
        else:
            ok_node = kind == v3.SELECT_BUSY
        if not ok_node:
            record("progressive widening multi-warp claims unique slots", False, f"bad wid={wid}")
            return

    h2 = run_release(d)
    ok = (
        sorted(root_slots) == [1, 2, 3]
        and child_expands <= 1
        and int(h["node_expand_inflight"][0, 0]) == 3
        and host_inflight_sum(h2) == 0
    )
    record(
        "progressive widening multi-warp claims unique slots",
        ok,
        f"root_slots={root_slots}, child_expands={child_expands}, slots={slots}",
    )


def test_single_edge_contention_allows_singleton_claims():
    case = make_case(nodes=2, actions=4, warps=8, path_depth=2)
    case["node_expanded"][0, 0] = 1
    case["edge_child"][0, 0, 0] = 1
    case["edge_prior"][0, 0, 0] = 1.0
    case["edge_n"][0, 0, 0] = 1
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    d, h = run_select(case, c_pw=0.0)

    kinds = [decode(int(h["out_selected"][0, wid]))[0] for wid in range(case["warps"])]
    valid_count = sum(1 for kind in kinds if kind == v3.SELECT_TERMINAL)
    busy_count = sum(1 for kind in kinds if kind == v3.SELECT_BUSY)
    h2 = run_release(d)
    ok = (
        valid_count > 0
        and valid_count + busy_count == case["warps"]
        and int(h["edge_inflight"][0, 0, 0]) == valid_count
        and host_inflight_sum(h2) == 0
    )
    record("single-edge contention allows singleton claims", ok, f"kinds={kinds}")


def test_pw_ticket_full_falls_through_to_existing_child():
    case = make_case(nodes=3, actions=4, warps=2, path_depth=2)
    case["node_expanded"][0, 0] = 1
    case["node_expand_inflight"][0, 0] = 3
    case["edge_child"][0, 0, 0] = 1
    case["edge_prior"][0, 0, 0] = 1.0
    case["edge_n"][0, 0, 0] = 16
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    d, h = run_select(case, c_pw=1.0, alpha_pw=0.5)

    kinds = [decode(int(h["out_selected"][0, wid]))[0] for wid in range(case["warps"])]
    selected_nodes = [decode(int(h["out_selected"][0, wid]))[2] for wid in range(case["warps"])]
    valid_count = sum(1 for kind in kinds if kind == v3.SELECT_TERMINAL)
    busy_count = sum(1 for kind in kinds if kind == v3.SELECT_BUSY)
    h2 = run_release(d)
    ok = (
        valid_count > 0
        and valid_count + busy_count == case["warps"]
        and all(node in (0, 1) for node in selected_nodes)
        and int(h["edge_inflight"][0, 0, 0]) == valid_count
        and int(h["node_expand_inflight"][0, 0]) == 3
        and int(h2["node_expand_inflight"][0, 0]) == 3
        and int(h2["edge_inflight"][0, 0, 0]) == 0
    )
    record("full PW ticket quota falls through to existing child", ok, f"kinds={kinds}, nodes={selected_nodes}")


def test_winner_soft_is_explicit_fake_claim_option():
    case = make_case(nodes=4, actions=4, warps=8, path_depth=2)
    case["node_expanded"][0, 0] = 2
    for slot, child in enumerate((1, 2)):
        case["edge_child"][0, 0, slot] = child
        case["edge_prior"][0, 0, slot] = 1.0
        case["edge_n"][0, 0, slot] = 1
        case["node_expanded"][0, child] = v3.NODE_EXPANDED_TERMINAL

    d_strict, h_strict = run_select(case, c_pw=0.0, variant=VARIANT_WINNER_RECALC)
    strict_kinds = [decode(int(h_strict["out_selected"][0, wid]))[0] for wid in range(case["warps"])]
    strict_valid = sum(1 for kind in strict_kinds if kind == v3.SELECT_TERMINAL)
    strict_busy = sum(1 for kind in strict_kinds if kind == v3.SELECT_BUSY)
    h_strict_released = run_release(d_strict)

    d_soft, h_soft = run_select(case, c_pw=0.0, variant=VARIANT_WINNER_SOFT)
    soft_kinds = [decode(int(h_soft["out_selected"][0, wid]))[0] for wid in range(case["warps"])]
    soft_valid = sum(1 for kind in soft_kinds if kind == v3.SELECT_TERMINAL)
    soft_busy = sum(1 for kind in soft_kinds if kind == v3.SELECT_BUSY)
    h_soft_released = run_release(d_soft)

    ok = (
        strict_valid > 0
        and strict_busy > 0
        and soft_valid == case["warps"]
        and soft_busy == 0
        and host_inflight_sum(h_strict_released) == 0
        and host_inflight_sum(h_soft_released) == 0
    )
    record(
        "winner_soft explicitly enables fake winner claims",
        ok,
        f"strict={strict_kinds}, soft={soft_kinds}",
    )


def test_fully_expanded_select_and_release_roundtrip():
    case = make_case(nodes=3, actions=4, path_depth=4)
    case["node_expanded"][0, 0] = 2
    case["edge_child"][0, 0, 0] = 1
    case["edge_child"][0, 0, 1] = 2
    case["edge_prior"][0, 0, 0] = 0.1
    case["edge_prior"][0, 0, 1] = 0.9
    case["edge_n"][0, 0, 0] = 1
    case["edge_n"][0, 0, 1] = 1
    d, h = run_select(case)
    kind, slot, node = decode(int(h["out_selected"][0, 0]))
    h2 = run_release(d)
    ok = (
        kind == v3.SELECT_EXPAND
        and node == 2
        and slot == 0
        and int(h["out_path"][0, 0, 0]) == 1
        and int(h["out_len"][0, 0]) == 2
        and int(h["edge_inflight"][0, 0, 1]) == 1
        and int(h["node_expand_inflight"][0, 2]) == 1
        and host_inflight_sum(h2) == 0
    )
    record("fully expanded select picks high PUCT edge and release roundtrips", ok)


def test_depth_limit():
    case = make_case(nodes=3, actions=4, path_depth=1)
    case["node_expanded"][0, 0] = 1
    case["node_expanded"][0, 1] = 1
    case["edge_child"][0, 0, 0] = 1
    case["edge_child"][0, 1, 0] = 2
    case["edge_n"][0, 0, 0] = 1
    case["edge_n"][0, 1, 0] = 1
    d, h = run_select(case)
    kind, _, node = decode(int(h["out_selected"][0, 0]))
    h2 = run_release(d)
    ok = (
        kind == v3.SELECT_DEPTH_LIMIT
        and node == 1
        and int(h["out_len"][0, 0]) == 2
        and int(h["edge_inflight"][0, 0, 0]) == 1
        and host_inflight_sum(h2) == 0
    )
    record("depth limit returns packed depth-limit leaf", ok)


def test_n_zero_edge_is_still_selectable():
    case = make_case(nodes=3, actions=4, path_depth=3)
    case["node_expanded"][0, 0] = 1
    case["edge_child"][0, 0, 0] = 1
    case["edge_prior"][0, 0, 0] = 1.0
    case["edge_n"][0, 0, 0] = 0
    _, h = run_select(case)
    kind, _, node = decode(int(h["out_selected"][0, 0]))
    ok = kind == v3.SELECT_EXPAND and node == 1 and int(h["out_len"][0, 0]) == 2
    record("zero-N edge remains selectable (q=0 fallback)", ok)


def test_encoding_boundaries():
    case_256 = make_case(nodes=2, actions=256, path_depth=2)
    _, h_256 = run_select(case_256)
    kind_256, _, _ = decode(int(h_256["out_selected"][0, 0]))

    case_257 = make_case(nodes=2, actions=257, path_depth=2)
    _, h_257 = run_select(case_257)
    kind_257, _, _ = decode(int(h_257["out_selected"][0, 0]))

    case_nodes = make_case(nodes=v3.PACKED_NODE_LIMIT + 1, actions=4, path_depth=2)
    _, h_nodes = run_select(case_nodes)
    kind_nodes, _, _ = decode(int(h_nodes["out_selected"][0, 0]))

    case_capacity = make_case(nodes=v3.PACKED_NODE_LIMIT + 1, actions=4, path_depth=2)
    case_capacity["node_count"][0] = v3.PACKED_NODE_LIMIT
    _, h_capacity = run_select(case_capacity)
    kind_capacity, _, _ = decode(int(h_capacity["out_selected"][0, 0]))

    ok = (
        kind_256 == v3.SELECT_EXPAND
        and kind_257 == v3.SELECT_INVALID
        and kind_nodes == v3.SELECT_INVALID
        and kind_capacity == v3.SELECT_EXPAND
    )
    record("packing boundaries: 256 actions ok, node_count gates packed node ids", ok)


def test_v3_stress():
    configs = []
    for trees in [1, 16, 128]:
        for warps in [1, 2, 4, 8]:
            for actions in [4, 32, 128, 256]:
                for depth in [1, 16, 64]:
                    configs.append((trees, warps, actions, depth))

    all_ok = True
    failed = []
    for trees, warps, actions, depth in configs:
        try:
            case = make_case(trees=trees, nodes=4, actions=actions, warps=warps, path_depth=depth)
            case["node_expanded"][:, 0] = 1
            case["edge_child"][:, 0, 0] = 1
            case["edge_n"][:, 0, 0] = 1
            d = to_device(case)
            valid = 0
            loops = 10000 if (trees, warps, actions, depth) == (1, 8, 256, 64) else 10
            for _ in range(loops):
                launch_select(d, c_pw=1.0, variant=VARIANT_WINNER_RECALC)
                launch_release(d, trees=trees, warps=warps)
            cuda.synchronize()
            h = copy_back(d)
            raws = h["out_selected"].reshape(-1)
            for raw in raws:
                if is_valid_selection_kind(decode(int(raw))[0]):
                    valid += 1
            if not host_inflight_nonnegative(h) or host_inflight_sum(h) != 0 or valid == 0:
                all_ok = False
                failed.append((trees, warps, actions, depth))
        except Exception as exc:
            all_ok = False
            failed.append((trees, warps, actions, depth, repr(exc)))
            break
    record("stress matrix keeps inflight non-negative and produces valid selections", all_ok, f"failed={failed[:3]}")


def _run_deep_select_release(d, trees, warps, c_pw=0.0, variant=VARIANT_WINNER_RECALC):
    _run_select_release_once(d, trees=trees, warps=warps, c_pw=c_pw, variant=variant)


def test_v3_deep_tree_stress():
    depths = [16, 64, 128, 256, 512, 1024]
    configs = []
    for depth in depths:
        configs.append(("narrow", 16, 4, 4, depth))
        configs.append(("wide64", 4, 2, 64, depth))
        configs.append(("wide256", 1, 2, 256, depth))

    print("\n  V3 deep-tree stress")
    print("  " + "-" * 82)
    print(
        f"  {'shape':>8} | {'depth':>5} | {'trees':>5} | {'warps':>5} | {'acts':>5} | {'loops':>5} | {'max_len':>7} | result"
    )
    print("  " + "-" * 82)

    all_ok = True
    failed = []
    for shape_name, trees, warps, actions, depth in configs:
        shape = "narrow" if shape_name == "narrow" else "wide"
        loops = 5 if depth <= 128 else 2
        try:
            case = make_deep_chain_case(
                trees=trees,
                tree_depth=depth,
                actions=actions,
                warps=warps,
                shape=shape,
            )
            d = to_device(case)
            for _ in range(loops):
                _run_deep_select_release(d, trees, warps, c_pw=0.0)
            cuda.synchronize()
            h = copy_back(d)
            kinds = [decode(int(raw))[0] for raw in h["out_selected"].reshape(-1)]
            valid = sum(1 for kind in kinds if is_valid_selection_kind(kind))
            max_len = int(h["out_len"].max())
            ok = (
                valid > 0
                and max_len == depth + 1
                and host_inflight_nonnegative(h)
                and host_inflight_sum(h) == 0
            )
            if not ok:
                all_ok = False
                failed.append((shape_name, depth, "valid", valid, "max_len", max_len))
            print(
                f"  {shape_name:>8} | {depth:5d} | {trees:5d} | {warps:5d} | "
                f"{actions:5d} | {loops:5d} | {max_len:7d} | {'PASS' if ok else 'FAIL'}"
            )
        except Exception as exc:
            all_ok = False
            failed.append((shape_name, depth, repr(exc)))
            print(
                f"  {shape_name:>8} | {depth:5d} | {trees:5d} | {warps:5d} | "
                f"{actions:5d} | {'-':>5} | {'-':>7} | FAIL: {exc}"
            )
            break

    record("deep-tree stress across depth 16..1024 and narrow/wide shapes", all_ok, f"failed={failed[:3]}")


def bench_select_only():
    warmup = _env_int("PUCT_V3_BENCH_WARMUP", 10)
    iterations = _env_int("PUCT_V3_BENCH_ITERS", 100)
    scenarios = [
        ("high_contention_equal_prior", 1, 8, 256, 2),
        ("low_contention_few_warps", 128, 1, 128, 16),
        ("wide_node", 16, 4, 256, 16),
        ("biased_prior", 16, 4, 128, 16),
    ]
    print("\n  V3 select-only benchmark")
    print(
        f"  timing: warmup={warmup} timed_select_launches={iterations} untimed_release_launches={warmup + iterations}"
    )
    print("  " + "-" * 230)
    print(
        f"  {'variant':>15} | {'scenario':>28} | {'trees':>5} | {'warps':>5} | {'acts':>5} | "
        f"{'sel/s':>12} | {'ns/edge':>10} | {'atomics/sel':>11} | "
        f"{'collision%':>10} | {'unique_first':>12} | {'root_kl':>8} | "
        f"{'depth_unique%':>13} | {'path_H':>7} | {'max1/tree%':>10} | "
        f"{'maxpath%':>8} | {'path75%':>7} | {'invalid%':>8} | {'busy%':>7}"
    )
    print("  " + "-" * 230)

    all_ok = True
    for variant in SELECT_VARIANTS:
        for name, trees, warps, actions, depth in scenarios:
            case = make_case(trees=trees, nodes=4, actions=actions, warps=warps, path_depth=depth)
            expanded = min(actions, 32)
            case["node_expanded"][:, 0] = expanded
            case["edge_n"][:, 0, :expanded] = 1
            for t in range(trees):
                for a in range(expanded):
                    case["edge_child"][t, 0, a] = 1 + (a % 3)
                    if name == "biased_prior":
                        case["edge_prior"][t, 0, a] = 100.0 if a == 0 else 1.0
            d = to_device(case)

            elapsed_ms = time_select_kernel_only(
                d,
                trees=trees,
                warps=warps,
                c_pw=1.0,
                variant=variant,
                warmup=warmup,
                iterations=iterations,
            )

            h = copy_back(d)
            inflight_ok = host_inflight_nonnegative(h) and host_inflight_sum(h) == 0
            metrics = selection_metrics(h, expanded)
            selections = iterations * trees * warps
            sps = selections / max(elapsed_ms / 1000.0, 1e-9)
            ns_per_edge = elapsed_ms * 1.0e6 / max(1, selections * expanded)
            atomics_per_selection = 3
            print(
                f"  {variant:>15} | {name:>28} | {trees:5d} | {warps:5d} | {actions:5d} | "
                f"{sps:12.1f} | {ns_per_edge:10.2f} | {atomics_per_selection:11d} | "
                f"{metrics['collision_pct']:9.2f}% | {metrics['unique_first']:12d} | "
                f"{metrics['root_kl']:8.3f} | {metrics['depth_unique_pct']:12.2f}% | "
                f"{metrics['path_entropy_norm']:7.3f} | {metrics['tree_dominant_first_pct']:9.2f}% | "
                f"{metrics['tree_dominant_path_pct']:7.2f}% | "
                f"{metrics['tree_dominant_path_75_pct']:6.2f}% | "
                f"{metrics['invalid_pct']:7.2f}% | {metrics['busy_pct']:6.2f}%"
            )
            all_ok = all_ok and metrics["invalid_pct"] < 100.0 and inflight_ok

    record("select-only benchmark completed", all_ok, "winner_recalc")


def bench_deep_tree_depths():
    warmup = _env_int("PUCT_V3_DEEP_WARMUP", 5)
    short_iterations = _env_int("PUCT_V3_DEEP_ITERS_SHORT", 20)
    long_iterations = _env_int("PUCT_V3_DEEP_ITERS_LONG", 8)
    depths = [16, 64, 128, 256, 512, 1024]
    configs = []
    for depth in depths:
        configs.append(("narrow", 16, 4, 4, depth))
        configs.append(("wide64", 4, 2, 64, depth))
        configs.append(("wide256", 1, 2, 256, depth))

    print("\n  V3 deep-tree select benchmark")
    print(
        f"  timing: warmup={warmup} timed_select_launches={short_iterations} for depth<=128, "
        f"{long_iterations} for depth>128; untimed release after every select"
    )
    print("  " + "-" * 220)
    print(
        f"  {'variant':>15} | {'shape':>8} | {'depth':>5} | {'trees':>5} | {'warps':>5} | {'acts':>5} | "
        f"{'path_len':>8} | {'sel/s':>12} | {'ns/edge':>10} | {'atomics/sel':>11} | "
        f"{'root_kl':>8} | {'depth_unique%':>13} | {'path_H':>7} | "
        f"{'max1/tree%':>10} | {'maxpath%':>8} | {'path75%':>7} | {'invalid%':>8} | {'busy%':>7}"
    )
    print("  " + "-" * 220)

    all_ok = True
    for variant in SELECT_VARIANTS:
        for shape_name, trees, warps, actions, depth in configs:
            shape = "narrow" if shape_name == "narrow" else "wide"
            case = make_deep_chain_case(
                trees=trees,
                tree_depth=depth,
                actions=actions,
                warps=warps,
                shape=shape,
            )
            d = to_device(case)
            iterations = short_iterations if depth <= 128 else long_iterations
            elapsed_ms = time_select_kernel_only(
                d,
                trees=trees,
                warps=warps,
                c_pw=0.0,
                variant=variant,
                warmup=warmup,
                iterations=iterations,
            )

            h = copy_back(d)
            inflight_ok = host_inflight_nonnegative(h) and host_inflight_sum(h) == 0
            path_len = int(h["out_len"].max())
            fanout = 1 if shape == "narrow" else actions
            metrics = selection_metrics(h, fanout)
            selections = iterations * trees * warps
            traversed_edges = max(1, path_len - 1)
            sps = selections / max(elapsed_ms / 1000.0, 1e-9)
            ns_per_edge = elapsed_ms * 1.0e6 / max(1, selections * traversed_edges * fanout)
            atomics_per_selection = traversed_edges * 3
            all_ok = all_ok and metrics["invalid_pct"] < 100.0 and path_len == depth + 1 and inflight_ok
            print(
                f"  {variant:>15} | {shape_name:>8} | {depth:5d} | {trees:5d} | {warps:5d} | {actions:5d} | "
                f"{path_len:8d} | {sps:12.1f} | {ns_per_edge:10.2f} | "
                f"{atomics_per_selection:11d} | {metrics['root_kl']:8.3f} | "
                f"{metrics['depth_unique_pct']:12.2f}% | {metrics['path_entropy_norm']:7.3f} | "
                f"{metrics['tree_dominant_first_pct']:9.2f}% | "
                f"{metrics['tree_dominant_path_pct']:7.2f}% | "
                f"{metrics['tree_dominant_path_75_pct']:6.2f}% | "
                f"{metrics['invalid_pct']:7.2f}% | {metrics['busy_pct']:6.2f}%"
            )

    record("deep-tree select benchmark completed", all_ok, "winner_recalc")


def bench_cpu_vs_gpu_sequential():
    warmup = _env_int("PUCT_V3_CPU_BENCH_WARMUP", 2)
    cpu_repeats = _env_int("PUCT_V3_CPU_BENCH_REPEATS", 3)
    gpu_iterations_default = _env_int("PUCT_V3_CPU_BENCH_GPU_ITERS", 10)
    warps = _env_int("PUCT_V3_CPU_BENCH_WARPS", 8)
    configs = [
        ("narrow_d64", 1024, warps, 4, 64, "narrow", gpu_iterations_default),
        ("wide32_d64", 512, warps, 32, 64, "wide", gpu_iterations_default),
        ("wide128_d128", 128, warps, 128, 128, "wide", max(2, gpu_iterations_default // 2)),
        ("wide256_d256", 64, warps, 256, 256, "wide", max(1, gpu_iterations_default // 2)),
    ]

    print("\n  V3 CPU sequential no-vloss vs GPU select benchmark")
    print(
        f"  cpu: numba njit(cache=True, fastmath=True), sequential, no virtual loss; "
        f"warmup={warmup} repeats={cpu_repeats}"
    )
    print(
        f"  gpu: select-only CUDA event, release launch after each select is untimed; "
        f"default_timed_select_launches={gpu_iterations_default}"
    )
    print("  " + "-" * 170)
    print(
        f"  {'scenario':>14} | {'variant':>13} | {'trees':>6} | {'warps':>5} | {'trav/tree':>9} | "
        f"{'depth':>5} | {'acts':>5} | {'shape':>6} | {'cpu sel/s':>12} | {'gpu sel/s':>12} | "
        f"{'gpu/cpu':>8} | {'cpu ns/edge':>11} | {'gpu ns/edge':>11} | {'valid':>8}"
    )
    print("  " + "-" * 170)

    all_ok = True
    for name, trees, cfg_warps, actions, depth, shape, gpu_iterations in configs:
        case = make_deep_chain_case(
            trees=trees,
            tree_depth=depth,
            actions=actions,
            warps=cfg_warps,
            shape=shape,
        )
        traversals_per_tree = cfg_warps * gpu_iterations
        total_selects = trees * traversals_per_tree
        fanout = 1 if shape == "narrow" else actions
        visited_edges = max(1, total_selects * depth * fanout)

        cpu_ms, checksum, total_path_len, valid = time_cpu_select_no_vloss(
            case,
            traversals_per_tree=traversals_per_tree,
            warmup=warmup,
            repeats=cpu_repeats,
        )
        cpu_sps = float(total_selects) / max(cpu_ms / 1000.0, 1e-9)
        cpu_ns_edge = cpu_ms * 1.0e6 / float(visited_edges)
        ok = valid == total_selects and total_path_len >= total_selects and checksum != 0

        for variant in SELECT_VARIANTS:
            d = to_device(case)
            elapsed_ms = time_select_kernel_only(
                d,
                trees=trees,
                warps=cfg_warps,
                c_pw=0.0,
                variant=variant,
                warmup=warmup,
                iterations=gpu_iterations,
            )
            inflight_ok, _, _ = check_inflight_empty(d)
            gpu_sps = float(total_selects) / max(elapsed_ms / 1000.0, 1e-9)
            gpu_ns_edge = elapsed_ms * 1.0e6 / float(visited_edges)
            speedup = gpu_sps / max(cpu_sps, 1e-9)
            all_ok = all_ok and ok and inflight_ok
            print(
                f"  {name:>14} | {variant:>13} | {trees:6d} | {cfg_warps:5d} | {traversals_per_tree:9d} | "
                f"{depth:5d} | {actions:5d} | {shape:>6} | {cpu_sps:12.1f} | {gpu_sps:12.1f} | "
                f"{speedup:8.2f} | {cpu_ns_edge:11.2f} | {gpu_ns_edge:11.2f} | "
                f"{valid:8d}"
            )
            del d
            flush_cuda_deallocations()

    record("CPU sequential no-vloss vs GPU select benchmark completed", all_ok, "same trees and traversals/tree")


def cpu_scale_configs(warps, full_matrix):
    configs = [
        ("eq256_d512", 4096, warps, 256, 512, "equal"),
        ("eq256_d1024", 4096, warps, 256, 1024, "equal"),
        ("hot256_d512", 4096, warps, 256, 512, "hot"),
        ("hot256_d1024", 4096, warps, 256, 1024, "hot"),
        ("eq128_d512", 8192, warps, 128, 512, "equal"),
        ("eq128_d1024", 8192, warps, 128, 1024, "equal"),
        ("hot128_d512", 8192, warps, 128, 512, "hot"),
        ("hot128_d1024", 8192, warps, 128, 1024, "hot"),
        ("eq64_d512", 16384, warps, 64, 512, "equal"),
        ("eq64_d1024", 16384, warps, 64, 1024, "equal"),
        ("hot64_d512", 16384, warps, 64, 512, "hot"),
        ("hot64_d1024", 16384, warps, 64, 1024, "hot"),
    ]
    if full_matrix:
        return configs
    return [
        configs[0],
        configs[1],
        configs[5],
        configs[9],
        configs[11],
    ]


def bench_cpu_vs_gpu_scale_aligned():
    budget_gib = float(os.environ.get("PUCT_V3_CPU_SCALE_BUDGET_GB", "24.0"))
    budget_bytes = int(budget_gib * (1024**3))
    try:
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception:
        free_bytes, total_bytes = budget_bytes, budget_bytes
    reserve_gib = float(os.environ.get("PUCT_V3_CPU_SCALE_RESERVE_GB", "0.50"))
    reserve_bytes = int(reserve_gib * (1024**3))
    usable_bytes = min(budget_bytes, max(0, free_bytes - reserve_bytes))
    warps = _env_int("PUCT_V3_CPU_SCALE_WARPS", 4)
    cpu_warmup = _env_int("PUCT_V3_CPU_SCALE_CPU_WARMUP", 1)
    cpu_repeats = _env_int("PUCT_V3_CPU_SCALE_CPU_REPEATS", 1)
    gpu_warmup = _env_int("PUCT_V3_CPU_SCALE_GPU_WARMUP", 2)
    gpu_iterations = _env_int("PUCT_V3_CPU_SCALE_GPU_ITERS", 1)
    full_matrix = os.environ.get("PUCT_V3_CPU_SCALE_FULL", "0") == "1"
    variants = parse_scale_variants()
    configs = cpu_scale_configs(warps, full_matrix)

    print("\n  V3 CPU sequential no-vloss vs GPU large-scale aligned benchmark")
    print(
        f"  memory: free={free_bytes / (1024**3):.2f}GiB total={total_bytes / (1024**3):.2f}GiB "
        f"budget={budget_gib:.2f}GiB reserve={reserve_gib:.2f}GiB usable={usable_bytes / (1024**3):.2f}GiB"
    )
    print(
        f"  cpu: procedural deep-wide tree, numba njit(cache=True, fastmath=True), sequential, no virtual loss; "
        f"warmup={cpu_warmup} repeats={cpu_repeats}"
    )
    print(
        f"  gpu: same device tree generator as --scale-bench; variants={','.join(variants)} "
        f"warmup={gpu_warmup} timed_select_launches={gpu_iterations} warps_per_tree={warps} "
        f"matrix={'full' if full_matrix else 'default-subset'}"
    )
    print("  " + "-" * 190)
    print(
        f"  {'scenario':>14} | {'variant':>13} | {'trees':>6} | {'warps':>5} | {'trav/tree':>9} | "
        f"{'depth':>5} | {'acts':>5} | {'prior':>5} | {'estGiB':>7} | {'cpu sel/s':>12} | "
        f"{'gpu sel/s':>12} | {'gpu/cpu':>8} | {'cpu ns/edge':>11} | {'gpu ns/edge':>11} | {'inflight':>10}"
    )
    print("  " + "-" * 190)

    any_ran = False
    all_ok = True
    for name, trees, cfg_warps, actions, depth, prior_mode in configs:
        estimated = estimate_deep_chain_bytes(trees, depth, actions, cfg_warps)
        estimated_gib = estimated / float(1024**3)
        traversals_per_tree = cfg_warps * gpu_iterations
        total_selects = trees * traversals_per_tree
        visited_edges = max(1, total_selects * depth * actions)

        if estimated > usable_bytes:
            for variant in variants:
                print(
                    f"  {name:>14} | {variant:>13} | {trees:6d} | {cfg_warps:5d} | {traversals_per_tree:9d} | "
                    f"{depth:5d} | {actions:5d} | {prior_mode:>5} | {estimated_gib:7.2f} | "
                    f"{'SKIP budget':>12} | {'-':>12} | {'-':>8} | {'-':>11} | {'-':>11} | {'-':>10}"
                )
            continue

        cpu_ms, checksum, total_path_len, valid = time_cpu_deep_wide_procedural(
            trees=trees,
            actions=actions,
            depth=depth,
            traversals_per_tree=traversals_per_tree,
            prior_mode=prior_mode,
            warmup=cpu_warmup,
            repeats=cpu_repeats,
        )
        cpu_sps = float(total_selects) / max(cpu_ms / 1000.0, 1e-9)
        cpu_ns_edge = cpu_ms * 1.0e6 / float(visited_edges)
        cpu_ok = valid == total_selects and total_path_len == total_selects * (depth + 1) and checksum != 0

        for variant in variants:
            d = None
            try:
                any_ran = True
                d = make_deep_chain_case_device(
                    trees=trees,
                    tree_depth=depth,
                    actions=actions,
                    warps=cfg_warps,
                    shape="wide",
                    prior_mode=prior_mode,
                    terminal_leaf=True,
                )
                cuda.synchronize()
                elapsed_ms = time_select_kernel_only(
                    d,
                    trees=trees,
                    warps=cfg_warps,
                    c_pw=0.0,
                    variant=variant,
                    warmup=gpu_warmup,
                    iterations=gpu_iterations,
                )
                inflight_ok, neg_count, nonzero_count = check_inflight_empty(d)
                gpu_sps = float(total_selects) / max(elapsed_ms / 1000.0, 1e-9)
                gpu_ns_edge = elapsed_ms * 1.0e6 / float(visited_edges)
                speedup = gpu_sps / max(cpu_sps, 1e-9)
                all_ok = all_ok and cpu_ok and inflight_ok
                inflight_label = "ok" if inflight_ok else f"bad:{neg_count}/{nonzero_count}"
                print(
                    f"  {name:>14} | {variant:>13} | {trees:6d} | {cfg_warps:5d} | {traversals_per_tree:9d} | "
                    f"{depth:5d} | {actions:5d} | {prior_mode:>5} | {estimated_gib:7.2f} | "
                    f"{cpu_sps:12.1f} | {gpu_sps:12.1f} | {speedup:8.2f} | "
                    f"{cpu_ns_edge:11.2f} | {gpu_ns_edge:11.2f} | {inflight_label:>10}"
                )
            except Exception as exc:
                all_ok = False
                print(
                    f"  {name:>14} | {variant:>13} | {trees:6d} | {cfg_warps:5d} | {traversals_per_tree:9d} | "
                    f"{depth:5d} | {actions:5d} | {prior_mode:>5} | {estimated_gib:7.2f} | "
                    f"{cpu_sps:12.1f} | {'FAIL':>12} | {'-':>8} | "
                    f"{cpu_ns_edge:11.2f} | {'-':>11} | {type(exc).__name__:>10}: {exc}"
                )
            finally:
                if d is not None:
                    del d
                flush_cuda_deallocations()

    record(
        "CPU sequential no-vloss vs GPU large-scale aligned benchmark completed",
        any_ran and all_ok,
        "same scale configs and traversals/tree as GPU subset",
    )


def parse_scale_variants():
    raw = os.environ.get("PUCT_V3_SCALE_VARIANTS", "winner_recalc")
    valid = set(SELECT_VARIANTS)
    variants = []
    for name in raw.split(","):
        name = name.strip()
        if name in valid and name not in variants:
            variants.append(name)
    return variants if variants else [VARIANT_WINNER_RECALC]


def select_named_scale_configs(configs, env_name, default_names):
    by_name = {cfg[0]: cfg for cfg in configs}
    raw = os.environ.get(env_name, default_names)
    if raw.strip().lower() == "full":
        return configs
    selected = []
    for name in raw.split(","):
        name = name.strip()
        if name in by_name and name not in [cfg[0] for cfg in selected]:
            selected.append(by_name[name])
    return selected if selected else [by_name[default_names.split(",")[0].strip()]]


def bench_gpu_long_stress():
    smoke = RUN_GPU_LONG_STRESS_SMOKE
    budget_gib = float(os.environ.get("PUCT_V3_LONG_STRESS_BUDGET_GB", "24.0"))
    budget_bytes = int(budget_gib * (1024**3))
    try:
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception:
        free_bytes, total_bytes = budget_bytes, budget_bytes
    reserve_gib = float(os.environ.get("PUCT_V3_LONG_STRESS_RESERVE_GB", "0.50"))
    reserve_bytes = int(reserve_gib * (1024**3))
    usable_bytes = min(budget_bytes, max(0, free_bytes - reserve_bytes))
    gpu_index = _env_int("PUCT_V3_LONG_STRESS_GPU", 0, minimum=0)
    warps = _env_int("PUCT_V3_LONG_STRESS_WARPS", 4)
    warmup = 1 if smoke else _env_int("PUCT_V3_LONG_STRESS_WARMUP", 3)
    iterations = 4 if smoke else _env_int("PUCT_V3_LONG_STRESS_ITERS", 128)
    chunk_size = 2 if smoke else _env_int("PUCT_V3_LONG_STRESS_CHUNK", 8)
    progress_every = 2 if smoke else _env_int("PUCT_V3_LONG_STRESS_PROGRESS", 16)
    sample_interval = _env_float("PUCT_V3_LONG_STRESS_SAMPLE_SEC", 0.5, minimum=0.1)
    variants = parse_scale_variants()
    configs = select_named_scale_configs(
        cpu_scale_configs(warps, True),
        "PUCT_V3_LONG_STRESS_SCENARIOS",
        "eq64_d512" if smoke else "eq256_d512",
    )

    print("\n  V3 GPU long-run large-scale stress benchmark")
    print(
        f"  memory: free={free_bytes / (1024**3):.2f}GiB total={total_bytes / (1024**3):.2f}GiB "
        f"budget={budget_gib:.2f}GiB reserve={reserve_gib:.2f}GiB usable={usable_bytes / (1024**3):.2f}GiB"
    )
    print(
        f"  variants={','.join(variants)} scenarios={','.join(cfg[0] for cfg in configs)} "
        f"warmup={warmup} timed_select_launches={iterations} chunk={chunk_size} "
        f"warps_per_tree={warps} telemetry_interval={sample_interval:.2f}s"
    )
    print("  note: gpu_util/idle are nvidia-smi utilization counters, not Nsight Compute theoretical occupancy.")
    print("  " + "-" * 210)
    print(
        f"  {'variant':>13} | {'scenario':>12} | {'trees':>6} | {'depth':>5} | {'acts':>5} | {'estGiB':>7} | "
        f"{'iters':>6} | {'sel/s':>12} | {'wall sel/s':>12} | {'ns/edge':>9} | "
        f"{'util avg/max':>12} | {'idle avg':>8} | {'memutil':>7} | {'pwr avg/max':>12} | "
        f"{'pwr%':>6} | {'smclk':>7} | {'temp':>5} | {'memGiB':>7} | {'samples':>7} | {'inflight':>9}"
    )
    print("  " + "-" * 210)

    any_ran = False
    all_ok = True
    for variant in variants:
        for name, trees, cfg_warps, actions, depth, prior_mode in configs:
            estimated = estimate_deep_chain_bytes(trees, depth, actions, cfg_warps)
            estimated_gib = estimated / float(1024**3)
            if estimated > usable_bytes:
                print(
                    f"  {variant:>13} | {name:>12} | {trees:6d} | {depth:5d} | {actions:5d} | {estimated_gib:7.2f} | "
                    f"{iterations:6d} | {'SKIP budget':>12} | {'-':>12} | {'-':>9} | "
                    f"{'-':>12} | {'-':>8} | {'-':>7} | {'-':>12} | {'-':>6} | "
                    f"{'-':>7} | {'-':>5} | {'-':>7} | {'-':>7} | {'-':>9}"
                )
                continue

            d = None
            try:
                print(
                    f"    starting: variant={variant} scenario={name} trees={trees} "
                    f"depth={depth} actions={actions} iters={iterations} chunk={chunk_size}",
                    flush=True,
                )
                d = make_deep_chain_case_device(
                    trees=trees,
                    tree_depth=depth,
                    actions=actions,
                    warps=cfg_warps,
                    shape="wide",
                    prior_mode=prior_mode,
                    terminal_leaf=True,
                )
                cuda.synchronize()

                stop_event, thread, samples = start_gpu_telemetry_sampler(gpu_index, sample_interval)
                try:
                    elapsed_ms, wall_ms = time_select_kernel_only_chunked(
                        d,
                        trees=trees,
                        warps=cfg_warps,
                        c_pw=0.0,
                        variant=variant,
                        warmup=warmup,
                        iterations=iterations,
                        chunk_size=chunk_size,
                        progress_every=progress_every,
                    )
                finally:
                    stop_event.set()
                    thread.join(timeout=max(1.0, sample_interval * 2.0))

                any_ran = True
                inflight_ok, neg_count, nonzero_count = check_inflight_empty(d)
                selections = iterations * trees * cfg_warps
                traversed_edges = depth
                sps = selections / max(elapsed_ms / 1000.0, 1e-9)
                wall_sps = selections / max(wall_ms / 1000.0, 1e-9)
                ns_per_edge = elapsed_ms * 1.0e6 / max(1, selections * traversed_edges * actions)
                telemetry = summarize_gpu_telemetry(samples)
                all_ok = all_ok and inflight_ok and telemetry["samples"] > 0
                inflight_label = "ok" if inflight_ok else f"bad:{neg_count}/{nonzero_count}"
                print(
                    f"  {variant:>13} | {name:>12} | {trees:6d} | {depth:5d} | {actions:5d} | {estimated_gib:7.2f} | "
                    f"{iterations:6d} | {sps:12.1f} | {wall_sps:12.1f} | {ns_per_edge:9.3f} | "
                    f"{fmt_float(telemetry['gpu_util_avg'], 1)}/{fmt_float(telemetry['gpu_util_max'], 0):>4} | "
                    f"{fmt_float(telemetry['idle_avg'], 1):>8} | {fmt_float(telemetry['mem_util_avg'], 1):>7} | "
                    f"{fmt_float(telemetry['power_avg'], 1)}/{fmt_float(telemetry['power_max'], 0):>4} | "
                    f"{fmt_float(telemetry['power_limit_pct'], 1):>6} | {fmt_float(telemetry['sm_clock_avg'], 0):>7} | "
                    f"{fmt_float(telemetry['temp_max'], 0):>5} | {fmt_float(telemetry['mem_used_max_gib'], 2):>7} | "
                    f"{telemetry['samples']:7d} | {inflight_label:>9}"
                )
            except Exception as exc:
                all_ok = False
                print(
                    f"  {variant:>13} | {name:>12} | {trees:6d} | {depth:5d} | {actions:5d} | {estimated_gib:7.2f} | "
                    f"{iterations:6d} | {'FAIL':>12} | {'-':>12} | {'-':>9} | "
                    f"{'-':>12} | {'-':>8} | {'-':>7} | {'-':>12} | {'-':>6} | "
                    f"{'-':>7} | {'-':>5} | {'-':>7} | {'-':>7} | {type(exc).__name__:>9}: {exc}"
                )
            finally:
                if d is not None:
                    del d
                flush_cuda_deallocations()

    record(
        "GPU long-run large-scale stress benchmark completed",
        any_ran and all_ok,
        "nvidia-smi telemetry sampled during timed launches",
    )


def bench_large_scale_deep_wide():
    budget_gib = float(os.environ.get("PUCT_V3_SCALE_BUDGET_GB", "24.0"))
    budget_bytes = int(budget_gib * (1024**3))
    try:
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception:
        free_bytes, total_bytes = budget_bytes, budget_bytes
    reserve_gib = float(os.environ.get("PUCT_V3_SCALE_RESERVE_GB", "0.50"))
    reserve_bytes = int(reserve_gib * (1024**3))
    usable_bytes = min(budget_bytes, max(0, free_bytes - reserve_bytes))
    warps = int(os.environ.get("PUCT_V3_SCALE_WARPS", "4"))
    warmup = _env_int("PUCT_V3_SCALE_WARMUP", 5)
    iterations = _env_int("PUCT_V3_SCALE_ITERS", 5)
    variants = parse_scale_variants()

    configs = [
        ("eq256_d512", 4096, warps, 256, 512, "equal"),
        ("eq256_d1024", 4096, warps, 256, 1024, "equal"),
        ("hot256_d512", 4096, warps, 256, 512, "hot"),
        ("hot256_d1024", 4096, warps, 256, 1024, "hot"),
        ("eq128_d512", 8192, warps, 128, 512, "equal"),
        ("eq128_d1024", 8192, warps, 128, 1024, "equal"),
        ("hot128_d512", 8192, warps, 128, 512, "hot"),
        ("hot128_d1024", 8192, warps, 128, 1024, "hot"),
        ("eq64_d512", 16384, warps, 64, 512, "equal"),
        ("eq64_d1024", 16384, warps, 64, 1024, "equal"),
        ("hot64_d512", 16384, warps, 64, 512, "hot"),
        ("hot64_d1024", 16384, warps, 64, 1024, "hot"),
    ]

    print("\n  V3 large-scale deep-wide select benchmark")
    print(
        f"  memory: free={free_bytes / (1024**3):.2f}GiB total={total_bytes / (1024**3):.2f}GiB "
        f"budget={budget_gib:.2f}GiB reserve={reserve_gib:.2f}GiB usable={usable_bytes / (1024**3):.2f}GiB"
    )
    print(
        f"  variants={','.join(variants)} warmup={warmup} timed_select_launches={iterations} "
        f"untimed_release_launches={warmup + iterations} warps_per_tree={warps}"
    )
    print("  " + "-" * 190)
    print(
        f"  {'variant':>15} | {'scenario':>14} | {'trees':>6} | {'warps':>5} | {'depth':>5} | {'acts':>5} | "
        f"{'estGiB':>7} | {'path_len':>8} | {'sel/s':>12} | {'ns/edge':>10} | "
        f"{'root_kl':>8} | {'max1/tree%':>10} | {'maxpath%':>8} | "
        f"{'path75%':>7} | {'samepath%':>9} | {'invalid%':>8} | {'busy%':>7} | {'inflight':>10}"
    )
    print("  " + "-" * 190)

    any_ran = False
    all_ok = True
    for variant in variants:
        for name, trees, cfg_warps, actions, depth, prior_mode in configs:
            estimated = estimate_deep_chain_bytes(trees, depth, actions, cfg_warps)
            estimated_gib = estimated / float(1024**3)
            if estimated > usable_bytes:
                print(
                    f"  {variant:>15} | {name:>14} | {trees:6d} | {cfg_warps:5d} | {depth:5d} | {actions:5d} | "
                    f"{estimated_gib:7.2f} | {'-':>8} | {'SKIP budget':>12} | {'-':>10} | "
                    f"{'-':>8} | {'-':>10} | {'-':>8} | {'-':>7} | {'-':>9} | {'-':>8} | {'-':>10}"
                )
                continue

            d = None
            try:
                any_ran = True
                d = make_deep_chain_case_device(
                    trees=trees,
                    tree_depth=depth,
                    actions=actions,
                    warps=cfg_warps,
                    shape="wide",
                    prior_mode=prior_mode,
                    terminal_leaf=True,
                )
                cuda.synchronize()
                elapsed_ms = time_select_kernel_only(
                    d,
                    trees=trees,
                    warps=cfg_warps,
                    c_pw=0.0,
                    variant=variant,
                    warmup=warmup,
                    iterations=iterations,
                )

                h = copy_selection_outputs(d)
                inflight_ok, neg_count, nonzero_count = check_inflight_empty(d)
                metrics = selection_metrics_scale(h, actions)
                path_len = int(h["out_len"].max())
                selections = iterations * trees * cfg_warps
                traversed_edges = max(1, path_len - 1)
                sps = selections / max(elapsed_ms / 1000.0, 1e-9)
                ns_per_edge = elapsed_ms * 1.0e6 / max(1, selections * traversed_edges * actions)
                ok = metrics["invalid_pct"] < 100.0 and path_len == depth + 1 and inflight_ok
                all_ok = all_ok and ok
                inflight_label = "ok" if inflight_ok else f"bad:{neg_count}/{nonzero_count}"
                print(
                    f"  {variant:>15} | {name:>14} | {trees:6d} | {cfg_warps:5d} | {depth:5d} | {actions:5d} | "
                    f"{estimated_gib:7.2f} | {path_len:8d} | {sps:12.1f} | {ns_per_edge:10.2f} | "
                    f"{metrics['root_kl']:8.3f} | {metrics['tree_dominant_first_pct']:9.2f}% | "
                    f"{metrics['tree_dominant_path_pct']:7.2f}% | "
                    f"{metrics['tree_dominant_path_75_pct']:6.2f}% | "
                    f"{metrics['tree_same_path_pct']:8.2f}% | "
                    f"{metrics['invalid_pct']:7.2f}% | {metrics['busy_pct']:6.2f}% | "
                    f"{inflight_label:>10}"
                )
            except Exception as exc:
                all_ok = False
                print(
                    f"  {variant:>15} | {name:>14} | {trees:6d} | {cfg_warps:5d} | {depth:5d} | {actions:5d} | "
                    f"{estimated_gib:7.2f} | {'-':>8} | {'FAIL':>12} | {'-':>10} | "
                    f"{'-':>8} | {'-':>10} | {'-':>8} | {'-':>7} | {'-':>9} | "
                    f"{'-':>8} | {type(exc).__name__:>10}: {exc}"
                )
            finally:
                if d is not None:
                    del d
                flush_cuda_deallocations()

    record("large-scale deep-wide benchmark completed", any_ran and all_ok, "4096..16384 trees within memory budget")


def summarize_results_and_exit():
    failed = [name for name, ok in results if not ok]
    print("\n================================================================")
    if failed:
        print(f"  FAILED: {len(failed)} test(s)")
        for name in failed:
            print(f"    - {name}")
        sys.exit(1)
    print(f"  ALL PASSED: {len(results)} test(s)")
    print("================================================================")


def main():
    if (
        RUN_CPU_BENCH
        and not RUN_CPU_SCALE_BENCH
        and not RUN_GPU_LONG_STRESS
        and not RUN_SCALE_BENCH
        and not RUN_STRESS
        and not RUN_BENCH
    ):
        bench_cpu_vs_gpu_sequential()
        summarize_results_and_exit()
        return

    if (
        RUN_CPU_SCALE_BENCH
        and not RUN_CPU_BENCH
        and not RUN_GPU_LONG_STRESS
        and not RUN_SCALE_BENCH
        and not RUN_STRESS
        and not RUN_BENCH
    ):
        bench_cpu_vs_gpu_scale_aligned()
        summarize_results_and_exit()
        return

    if (
        RUN_GPU_LONG_STRESS
        and not RUN_CPU_BENCH
        and not RUN_CPU_SCALE_BENCH
        and not RUN_SCALE_BENCH
        and not RUN_STRESS
        and not RUN_BENCH
    ):
        bench_gpu_long_stress()
        summarize_results_and_exit()
        return

    if (
        RUN_SCALE_BENCH
        and not RUN_CPU_BENCH
        and not RUN_CPU_SCALE_BENCH
        and not RUN_GPU_LONG_STRESS
        and not RUN_STRESS
        and not RUN_BENCH
    ):
        bench_large_scale_deep_wide()
        summarize_results_and_exit()
        return

    test_fresh_root_expand_roundtrip()
    test_terminal_root()
    test_invalid_child_rolls_back_path()
    test_child_beyond_node_count_invalid()
    test_progressive_widening_multi_warp_claims_unique_slots()
    test_single_edge_contention_allows_singleton_claims()
    test_pw_ticket_full_falls_through_to_existing_child()
    test_winner_soft_is_explicit_fake_claim_option()
    test_fully_expanded_select_and_release_roundtrip()
    test_depth_limit()
    test_n_zero_edge_is_still_selectable()
    test_encoding_boundaries()
    if RUN_STRESS:
        test_v3_stress()
        test_v3_deep_tree_stress()
    if RUN_BENCH:
        bench_select_only()
        bench_deep_tree_depths()
    if RUN_SCALE_BENCH:
        bench_large_scale_deep_wide()
    if RUN_CPU_BENCH:
        bench_cpu_vs_gpu_sequential()
    if RUN_CPU_SCALE_BENCH:
        bench_cpu_vs_gpu_scale_aligned()
    if RUN_GPU_LONG_STRESS:
        bench_gpu_long_stress()

    summarize_results_and_exit()


if __name__ == "__main__":
    main()
