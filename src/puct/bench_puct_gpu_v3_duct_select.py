"""Small DUCT select throughput benchmark for puct_gpu_v3_duct.py.

This is intentionally focused on select/release throughput so it can be
compared with batched environment rollout throughput.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import numpy as np
from numba import cuda

import puct.puct_gpu_v3 as v3
import puct.puct_gpu_v3_duct as duct
from puct.test_puct_gpu_v3_duct import launch_release_duct, launch_select_duct


@dataclass(frozen=True)
class DuctBenchResult:
    scenario: str
    trees: int
    warps: int
    actions: int
    depth: int
    warmup: int
    repeats: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    selections_per_s: float
    traversals_per_s: float
    edge_checks_per_s: float
    peak_mem_mb: float
    path_len: int


def _make_deep_duct_case(trees: int, warps: int, actions: int, depth: int) -> dict:
    nodes = depth + 1
    edge_child = np.full((trees, nodes, duct.DUCT_JOINT_ACTIONS), duct.DUCT_EDGE_UNEXPANDED, np.int32)
    action_w = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.float32)
    action_n = np.ones((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_inflight = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_counts = np.full((trees, nodes, duct.DUCT_PLAYERS), actions, np.int32)
    node_n = np.full((trees, nodes), 64, np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    node_count = np.full((trees,), nodes, np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, depth + 1), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)

    for d in range(depth):
        edge_child[:, d, 0] = d + 1
        action_w[:, d, :, 0] = 100.0
    edge_child[:, depth - 1, 0] = v3.NODE_EXPANDED_TERMINAL
    return {
        "edge_child": edge_child,
        "action_w": action_w,
        "action_n": action_n,
        "action_inflight": action_inflight,
        "action_counts": action_counts,
        "node_n": node_n,
        "node_expand_inflight": node_expand_inflight,
        "node_count": node_count,
        "out_selected": out_selected,
        "out_path": out_path,
        "out_len": out_len,
        "trees": trees,
        "warps": warps,
    }


def _to_device(case: dict) -> dict:
    return {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}


def _copy_path_len(dcase: dict) -> int:
    out_len = dcase["out_len"].copy_to_host()
    return int(out_len.max())


def _bench_one(
    *,
    scenario: str,
    trees: int,
    warps: int,
    actions: int,
    depth: int,
    warmup: int,
    repeats: int,
) -> DuctBenchResult:
    case = _make_deep_duct_case(trees, warps, actions, depth)
    dcase = _to_device(case)
    cuda.synchronize()
    cuda.current_context().deallocations.clear()
    cuda.synchronize()

    for _ in range(warmup):
        launch_select_duct(dcase, c_uct=0.0, c_pw=0.0)
        launch_release_duct(dcase)
    cuda.synchronize()

    samples = []
    cuda.current_context().deallocations.clear()
    for _ in range(repeats):
        start = time.perf_counter()
        launch_select_duct(dcase, c_uct=0.0, c_pw=0.0)
        launch_release_duct(dcase)
        cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    path_len = _copy_path_len(dcase)
    mean_ms = statistics.fmean(samples)
    selections = trees * warps
    traversed_edges = max(1, path_len - 1)
    return DuctBenchResult(
        scenario=scenario,
        trees=trees,
        warps=warps,
        actions=actions,
        depth=depth,
        warmup=warmup,
        repeats=repeats,
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        p95_ms=sorted(samples)[max(0, int(0.95 * len(samples)) - 1)],
        selections_per_s=selections / max(mean_ms / 1000.0, 1e-12),
        traversals_per_s=selections * traversed_edges / max(mean_ms / 1000.0, 1e-12),
        edge_checks_per_s=selections * traversed_edges * actions * duct.DUCT_PLAYERS / max(mean_ms / 1000.0, 1e-12),
        peak_mem_mb=0.0,
        path_len=path_len,
    )


def _print_table(results: list[DuctBenchResult]) -> None:
    print(
        f"{'scenario':>14} | {'trees':>6} | {'warps':>5} | {'acts':>5} | {'depth':>5} | "
        f"{'ms':>8} | {'sel/s':>12} | {'edge checks/s':>15} | {'path':>5} | {'p95':>8}"
    )
    print("-" * 116)
    for row in results:
        print(
            f"{row.scenario:>14} | {row.trees:6d} | {row.warps:5d} | {row.actions:5d} | {row.depth:5d} | "
            f"{row.mean_ms:8.3f} | {row.selections_per_s:12.0f} | {row.edge_checks_per_s:15.0f} | "
            f"{row.path_len:5d} | {row.p95_ms:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/puct_duct_select_throughput")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not cuda.is_available():
        raise RuntimeError("CUDA is required for DUCT select benchmark")

    configs = [
        ("d64_a16", 4096, 4, 16, 64),
        ("d64_a16", 8192, 4, 16, 64),
        ("d64_a16", 16384, 4, 16, 64),
        ("d128_a16", 4096, 4, 16, 128),
        ("d128_a16", 8192, 4, 16, 128),
    ]
    results = [
        _bench_one(
            scenario=name,
            trees=trees,
            warps=warps,
            actions=actions,
            depth=depth,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for name, trees, warps, actions, depth in configs
    ]
    _print_table(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "puct_duct_select_throughput.json"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2), encoding="utf-8")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
