"""
Visualize per-warp selection paths for puct_gpu_v3.

Examples:
  # default sparse case: tree=1, actions=4, depth=4, warps=8
  conda run -n py312_numba python src/puct/plot_puct_v3_warp_paths.py

  # full tree
  conda run -n py312_numba python src/puct/plot_puct_v3_warp_paths.py --mode full

  # custom output prefix and formats
  conda run -n py312_numba python src/puct/plot_puct_v3_warp_paths.py \
      --output-prefix /tmp/my_case --formats png,svg
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from numba import config
from numba import cuda


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import puct.puct_gpu_v3 as v3  # noqa: E402


def decode(raw: int):
    kind = (raw >> v3.PACKED_KIND_SHIFT) & v3.PACKED_KIND_MASK
    slot = (raw >> v3.PACKED_SLOT_SHIFT) & v3.PACKED_EDGE_MASK
    node = raw & v3.PACKED_NODE_MASK
    return int(kind), int(slot), int(node)


def decode_reason(raw: int):
    return int((raw >> v3.PACKED_REASON_SHIFT) & v3.PACKED_REASON_MASK)


def kind_name(kind: int) -> str:
    return {
        v3.SELECT_INVALID: "SELECT_INVALID",
        v3.SELECT_EXPAND: "SELECT_EXPAND",
        v3.SELECT_TERMINAL: "SELECT_TERMINAL",
        v3.SELECT_DEPTH_LIMIT: "SELECT_DEPTH_LIMIT",
        v3.SELECT_BUSY: "SELECT_BUSY",
    }.get(kind, f"UNKNOWN({kind})")


def reason_name(reason: int) -> str:
    return {
        v3.REASON_UNSET: "REASON_UNSET",
        v3.REASON_OK_EXPAND: "OK_EXPAND",
        v3.REASON_OK_TERMINAL: "OK_TERMINAL",
        v3.REASON_OK_DEPTH_LIMIT: "OK_DEPTH_LIMIT",
        v3.REASON_BUSY_EXPAND_INFLIGHT: "BUSY_EXPAND_INFLIGHT",
        v3.REASON_INVALID_SHAPE: "INVALID_SHAPE",
        v3.REASON_INVALID_NODE_INFO: "INVALID_NODE_INFO",
        v3.REASON_INVALID_EXPAND_TICKET: "INVALID_EXPAND_TICKET",
        v3.REASON_INVALID_WINNER_CAS: "INVALID_WINNER_CAS",
        v3.REASON_INVALID_NO_VALID_EDGE: "INVALID_NO_VALID_EDGE",
        v3.REASON_INVALID_CHILD_OOB: "INVALID_CHILD_OOB",
        v3.REASON_INVALID_UNKNOWN: "INVALID_UNKNOWN",
    }.get(reason, f"REASON_{reason}")


def _require_cuda():
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if getattr(config, "ENABLE_CUDASIM", False):
        raise RuntimeError("CUDA simulator is not supported for this script.")


def build_full_spec(actions: int, depth: int):
    level_offsets = [0]
    total = 1
    for d in range(1, depth + 1):
        total += actions**d
        level_offsets.append(total - actions**d)

    levels: dict[int, list[int]] = {}
    for lvl in range(depth + 1):
        start = level_offsets[lvl]
        width = actions**lvl
        levels[lvl] = list(range(start, start + width))

    expanded_edges: dict[int, list[tuple[int, int]]] = {}
    for lvl in range(depth):
        start = level_offsets[lvl]
        width = actions**lvl
        child_start = level_offsets[lvl + 1]
        for i in range(width):
            parent = start + i
            cur = []
            for a in range(actions):
                child = child_start + i * actions + a
                cur.append((a, child))
            expanded_edges[parent] = cur

    terminal_nodes = set(levels[depth])
    main_path = []
    node = 0
    main_path.append(node)
    for _ in range(depth):
        # in BFS indexing, action 0 always points to first child.
        child = expanded_edges[node][0][1]
        main_path.append(child)
        node = child

    return {
        "nodes": total,
        "levels": levels,
        "expanded_edges": expanded_edges,
        "candidate_edges": {},
        "terminal_nodes": terminal_nodes,
        "main_path": main_path,
        "expand_hint_node": None,
    }


def build_sparse_spec(
    actions: int,
    depth: int,
    extras_min: int,
    extras_max: int,
    root_extras: int | None = None,
):
    if actions < 2:
        raise ValueError("sparse mode requires actions >= 2")

    next_node = 0
    main_path = [next_node]
    next_node += 1
    for _ in range(depth):
        main_path.append(next_node)
        next_node += 1

    levels: dict[int, list[int]] = defaultdict(list)
    for lvl, nid in enumerate(main_path):
        levels[lvl].append(nid)

    expanded_edges: dict[int, list[tuple[int, int]]] = {}
    candidate_edges: dict[int, list[tuple[int, int]]] = {}
    terminal_nodes: set[int] = set()

    # Pick a mid node where we keep extra branches as "candidates" (not expanded yet),
    # so SELECT_EXPAND is likely to occur there instead of hitting depth limit.
    expand_hint_level = max(1, depth // 2)
    expand_hint_node = main_path[expand_hint_level]

    for lvl in range(depth):
        parent = main_path[lvl]
        main_child = main_path[lvl + 1]

        # Main path edge always in slot 0.
        cur_expanded = [(0, main_child)]

        max_extra = max(0, actions - 1)
        if lvl == 0 and root_extras is not None:
            desired_extra = root_extras
        else:
            desired_extra = extras_min if (lvl % 2 == 0) else extras_max
        extra_count = min(max_extra, max(0, desired_extra))

        # Allocate extra side branches at the next visual level.
        extra_children = []
        for _ in range(extra_count):
            child = next_node
            next_node += 1
            extra_children.append(child)
            levels[lvl + 1].append(child)
            terminal_nodes.add(child)

        if lvl == expand_hint_level:
            # Keep extras as unexpanded candidates to trigger SELECT_EXPAND.
            cand = []
            for i, child in enumerate(extra_children):
                cand.append((1 + i, child))
            candidate_edges[parent] = cand
        else:
            # Fully expanded on non-hint levels.
            for i, child in enumerate(extra_children):
                cur_expanded.append((1 + i, child))

        expanded_edges[parent] = cur_expanded

    terminal_nodes.add(main_path[-1])

    return {
        "nodes": next_node,
        "levels": dict(levels),
        "expanded_edges": expanded_edges,
        "candidate_edges": candidate_edges,
        "terminal_nodes": terminal_nodes,
        "main_path": main_path,
        "expand_hint_node": expand_hint_node,
    }


def make_case_from_spec(
    spec: dict,
    actions: int,
    warps: int,
    path_depth: int,
    main_prior: float,
    side_prior: float,
    main_n: int,
    side_n: int,
):
    trees = 1
    nodes = int(spec["nodes"])

    edge_child = np.full((trees, nodes, actions), -1, np.int32)
    edge_prior = np.zeros((trees, nodes, actions), np.float32)
    edge_w = np.zeros((trees, nodes, actions), np.float32)
    edge_n = np.zeros((trees, nodes, actions), np.int32)
    edge_inflight = np.zeros((trees, nodes, actions), np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    node_expanded = np.zeros((trees, nodes), np.int32)
    node_count = np.full((trees,), nodes, np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, path_depth), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)

    terminal_nodes = spec["terminal_nodes"]

    for parent, slot_children in spec["expanded_edges"].items():
        node_expanded[0, parent] = len(slot_children)
        for slot, child in slot_children:
            edge_child[0, parent, slot] = child
            edge_prior[0, parent, slot] = np.float32(main_prior if slot == 0 else side_prior)
            edge_n[0, parent, slot] = np.int32(main_n if slot == 0 else side_n)

    for nid in terminal_nodes:
        # Don't overwrite non-leaf expanded nodes.
        if node_expanded[0, nid] == 0:
            node_expanded[0, nid] = v3.NODE_EXPANDED_TERMINAL

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
        "trees": trees,
        "warps": warps,
        "actions": actions,
        "path_depth": path_depth,
        "spec": spec,
    }


def compute_expected_expand(case: dict, host: dict, c_pw: float, alpha_pw: float):
    """
    Host-side mirror of kernel logic for per-node expected expansion count.
    For each expanded node:
      N_eff(node) = sum_a (edge_N + edge_inflight) over current expanded edges
      allowed(node) = ceil(c_pw * N_eff^alpha_pw), clamped to [1, actions]
    """
    actions = int(case["actions"])
    nodes = host["node_expanded"].shape[1]
    allowed = np.zeros(nodes, dtype=np.int32)
    n_eff = np.zeros(nodes, dtype=np.int32)

    for node in range(nodes):
        cur_expanded = int(host["node_expanded"][0, node])
        if cur_expanded <= 0 or cur_expanded == v3.NODE_EXPANDED_TERMINAL:
            continue

        total = 0
        for eid in range(cur_expanded):
            total += int(host["edge_n"][0, node, eid]) + int(host["edge_inflight"][0, node, eid])
        total = max(total, 1)
        n_eff[node] = total

        if abs(alpha_pw - 0.5) < 1e-8:
            val = c_pw * math.sqrt(float(total))
        else:
            val = c_pw * math.pow(float(total), alpha_pw)
        a = int(math.ceil(val))
        a = max(1, min(actions, a))
        allowed[node] = a
    return {"allowed": allowed, "n_eff": n_eff}


def run_select(case: dict, cpuct: float, c_pw: float, alpha_pw: float):
    d = {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}
    v3._select_kernel_winner_recalc[d["trees"], d["warps"] * v3.WARP_SIZE](
        np.float32(cpuct),
        np.float32(c_pw),
        np.float32(alpha_pw),
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
    cuda.synchronize()
    return {k: v.copy_to_host() if hasattr(v, "copy_to_host") else v for k, v in d.items()}


def collect_warp_infos(case: dict, host: dict):
    infos = []
    for wid in range(case["warps"]):
        raw = int(host["out_selected"][0, wid])
        kind, slot, selected_node = decode(raw)
        reason_code = decode_reason(raw)
        out_len = int(host["out_len"][0, wid])

        edges = []
        path_nodes = [0]
        for d in range(case["path_depth"]):
            enc = int(host["out_path"][0, wid, d])
            if enc < 0:
                break
            parent = enc >> 8
            eid = enc & 0xFF
            child = int(host["edge_child"][0, parent, eid]) if 0 <= eid < case["actions"] else -1
            edges.append((parent, eid, child))
            if child >= 0:
                path_nodes.append(child)

        exit_node = selected_node if kind != v3.SELECT_INVALID else (path_nodes[-1] if path_nodes else 0)
        temp = {
            "wid": wid,
            "kind": kind,
            "kind_name": kind_name(kind),
            "slot": slot,
            "selected_node": selected_node,
            "out_len": out_len,
            "edges": edges,
            "path_nodes": path_nodes,
            "exit_node": exit_node,
        }
        temp["exit_reason"] = reason_name(reason_code)
        temp["exit_reason_code"] = reason_code
        infos.append(
            temp
        )
    return infos


def _node_fill_color(nid: int, spec: dict):
    if nid in spec["main_path"]:
        return "#e9f6ec"
    if nid in spec["terminal_nodes"]:
        return "#fdf7e3"
    return "#eef6ff"


def make_dot(case: dict, host: dict, infos: list[dict], title: str, expected: dict):
    spec = case["spec"]
    levels = spec["levels"]
    allowed = expected["allowed"]
    n_eff = expected["n_eff"]

    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#17becf",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    lines = []
    lines.append("digraph WarpPaths {")
    lines.append('  graph [rankdir=TB, bgcolor="white", pad="0.30", nodesep="0.35", ranksep="0.60", labelloc="t", fontsize=14, fontname="Helvetica", label="' + title + '"];')
    lines.append('  node [shape=circle, style="filled", color="#9a9a9a", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [color="#bdbdbd", penwidth=1.2, arrowsize=0.7, fontname="Helvetica", fontsize=9];')

    # Nodes + rank constraints
    for lvl in sorted(levels.keys()):
        rank_nodes = []
        for nid in levels[lvl]:
            nexp = int(host["node_expanded"][0, nid])
            state = "T" if nexp == v3.NODE_EXPANDED_TERMINAL else str(nexp)
            fill = _node_fill_color(nid, spec)
            if nexp > 0 and nexp != v3.NODE_EXPANDED_TERMINAL:
                lines.append(
                    f'  n{nid} [label="n{nid}\\nexp={state} allow={int(allowed[nid])}\\nNeff={int(n_eff[nid])}", fillcolor="{fill}"];'
                )
            else:
                lines.append(f'  n{nid} [label="n{nid}\\nexp={state}", fillcolor="{fill}"];')
            rank_nodes.append(f"n{nid}")
        lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")

    # Draw expanded edges (actual tree)
    for parent, slot_children in spec["expanded_edges"].items():
        for slot, child in slot_children:
            prior = float(host["edge_prior"][0, parent, slot])
            nval = int(host["edge_n"][0, parent, slot])
            lines.append(f'  n{parent} -> n{child} [label="a{slot} p={prior:.1f} n={nval}"];')

    # Draw candidate (not yet expanded) edges as dashed hints
    for parent, slot_children in spec["candidate_edges"].items():
        for slot, child in slot_children:
            lines.append(f'  n{parent} -> n{child} [style=dashed, color="#d9a400", label="a{slot} candidate"];')

    # Overlay warp paths
    edge_use = defaultdict(list)
    for info in infos:
        for parent, eid, child in info["edges"]:
            if child >= 0:
                edge_use[(parent, child)].append(info["wid"])

    for info in infos:
        wid = info["wid"]
        color = colors[wid % len(colors)]
        for parent, eid, child in info["edges"]:
            if child < 0:
                continue
            lines.append(
                f'  n{parent} -> n{child} [color="{color}", penwidth=3.6, arrowsize=0.9, fontcolor="{color}", label="w{wid}:a{eid}"];'
            )

    # Event callouts
    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#f8f8f8", color="#6f6f6f", fontsize=10];')
    for info in infos:
        wid = info["wid"]
        color = colors[wid % len(colors)]
        label = f"w{wid}: {info['kind_name']}\\nexit=n{info['exit_node']}, out_len={info['out_len']}"
        label += f"\\nreason={info['exit_reason']}"
        if info["kind"] == v3.SELECT_EXPAND:
            label += f"\\nexpand_slot={info['slot']}"
        if info["kind"] == v3.SELECT_INVALID:
            lines.append(f'  ebox{wid} [label="{label}", color="{color}", fillcolor="#ffeef0"];')
            lines.append(f'  n{info["exit_node"]} -> ebox{wid} [style=dashed, color="{color}", arrowhead=none];')
        else:
            lines.append(f'  ebox{wid} [label="{label}", color="{color}", fillcolor="#eef8ff"];')
            lines.append(f'  n{info["exit_node"]} -> ebox{wid} [style=dashed, color="{color}", arrowhead=none];')

    # Inflight summary
    root_inf = [int(x) for x in host["edge_inflight"][0, 0].tolist()]
    expand_nonzero = np.argwhere(host["node_expand_inflight"][0] > 0).reshape(-1)
    expand_txt = ", ".join([f"n{int(n)}:{int(host['node_expand_inflight'][0, n])}" for n in expand_nonzero]) or "none"
    inflight_label = (
        "after select\\n"
        f"edge_inflight[n0]={{a0:{root_inf[0]},a1:{root_inf[1]},a2:{root_inf[2]},a3:{root_inf[3]}}}\\n"
        f"node_expand_inflight={{{expand_txt}}}"
    )
    lines.append('  inflight [shape=note, style="filled", fillcolor="#f0fff0", color="#2ca02c", label="' + inflight_label + '"];')
    lines.append("  ebox0 -> inflight [style=dashed, color=\"#2ca02c\", arrowhead=none];")

    # Shared hot edges
    hot_lines = []
    for (parent, child), wids in sorted(edge_use.items(), key=lambda x: (-len(x[1]), x[0][0], x[0][1])):
        if len(wids) > 1:
            hot_lines.append(f"n{parent}->n{child}: " + ",".join([f"w{w}" for w in wids]))
    hot_txt = "\\n".join(hot_lines[:10]) if hot_lines else "none"
    lines.append('  hot [shape=note, style="filled", fillcolor="#fff9e8", color="#c47f00", label="shared edges\\n' + hot_txt + '"];')
    lines.append("  inflight -> hot [style=dashed, color=\"#c47f00\", arrowhead=none];")

    if spec["expand_hint_node"] is not None:
        n = int(spec["expand_hint_node"])
        lines.append('  hint [shape=note, style="filled", fillcolor="#fff8e1", color="#d9a400", label="expand-hint node\\nkeep 1~2 branches as candidate"];')
        lines.append(f"  n{n} -> hint [style=dashed, color=\"#d9a400\", arrowhead=none];")

    lines.append("}")
    return "\n".join(lines)


def write_summary(path: str, infos: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for info in infos:
            f.write(
                f"warp={info['wid']} kind={info['kind_name']} slot={info['slot']} "
                f"selected_node={info['selected_node']} out_len={info['out_len']} "
                f"reason={info['exit_reason']} path_nodes={info['path_nodes']}\n"
            )


def render_dot(dot_path: str, out_path: str, fmt: str):
    cmd = ["dot", f"-T{fmt}", dot_path, "-o", out_path]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Plot per-warp selection paths for puct_gpu_v3.")
    p.add_argument("--mode", choices=["sparse", "full"], default="sparse", help="Tree construction mode.")
    p.add_argument("--actions", type=int, default=4, help="Branching factor.")
    p.add_argument("--depth", type=int, default=4, help="Main path depth in edges.")
    p.add_argument("--warps", type=int, default=8, help="Warps per tree/block.")
    p.add_argument("--path-depth", type=int, default=None, help="Kernel path capacity. Default: depth+2.")
    p.add_argument("--extras-min", type=int, default=1, help="Sparse mode: min extra branches per level.")
    p.add_argument("--extras-max", type=int, default=2, help="Sparse mode: max extra branches per level.")
    p.add_argument(
        "--root-extras",
        type=int,
        default=None,
        help="Sparse mode: override root extra branches (0..actions-1).",
    )
    p.add_argument("--cpuct", type=float, default=1.0, help="PUCT cpuct.")
    p.add_argument("--c-pw", type=float, default=1.0, help="Progressive widening c_pw.")
    p.add_argument("--alpha-pw", type=float, default=0.5, help="Progressive widening alpha_pw.")
    p.add_argument("--main-prior", type=float, default=4.0, help="Prior on main-path edges.")
    p.add_argument("--side-prior", type=float, default=1.0, help="Prior on side edges.")
    p.add_argument("--main-n", type=int, default=4, help="N on main-path expanded edges.")
    p.add_argument("--side-n", type=int, default=1, help="N on side expanded edges.")
    p.add_argument("--output-prefix", type=str, default="outputs/puct_v3_warp_paths", help="Output file prefix.")
    p.add_argument("--formats", type=str, default="png,svg", help="Comma-separated formats rendered by dot.")
    p.add_argument("--no-render", action="store_true", help="Only write .dot/.txt, skip rendering.")
    return p.parse_args()


def main():
    args = parse_args()
    _require_cuda()

    if args.actions < 1:
        raise ValueError("--actions must be >= 1")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.warps < 1:
        raise ValueError("--warps must be >= 1")

    if args.path_depth is None:
        path_depth = args.depth + 2
    else:
        path_depth = args.path_depth
    if path_depth < 1:
        raise ValueError("--path-depth must be >= 1")

    if args.mode == "full":
        spec = build_full_spec(actions=args.actions, depth=args.depth)
    else:
        spec = build_sparse_spec(
            actions=args.actions,
            depth=args.depth,
            extras_min=args.extras_min,
            extras_max=args.extras_max,
            root_extras=args.root_extras,
        )

    case = make_case_from_spec(
        spec=spec,
        actions=args.actions,
        warps=args.warps,
        path_depth=path_depth,
        main_prior=args.main_prior,
        side_prior=args.side_prior,
        main_n=args.main_n,
        side_n=args.side_n,
    )
    host = run_select(case, cpuct=args.cpuct, c_pw=args.c_pw, alpha_pw=args.alpha_pw)
    expected = compute_expected_expand(case, host, c_pw=args.c_pw, alpha_pw=args.alpha_pw)
    infos = collect_warp_infos(case, host)

    title = f"PUCT v3 warp paths | mode={args.mode} tree=1 actions={args.actions} depth={args.depth} warps={args.warps}"
    dot_text = make_dot(case, host, infos, title=title, expected=expected)

    prefix = args.output_prefix
    dot_path = prefix + ".dot"
    txt_path = prefix + ".txt"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_text)
    write_summary(txt_path, infos)

    print(f"[OK] dot: {dot_path}")
    print(f"[OK] summary: {txt_path}")

    if not args.no_render:
        fmts = [x.strip() for x in args.formats.split(",") if x.strip()]
        for fmt in fmts:
            out_path = prefix + "." + fmt
            render_dot(dot_path, out_path, fmt)
            print(f"[OK] render: {out_path}")


if __name__ == "__main__":
    main()
