"""
Visualize per-warp DUCT selection paths for puct_gpu_v3_duct.

Examples:
  # default sparse case: one main chain plus an unexpanded joint-action hint
  conda run -n py312_numba python src/puct/plot_puct_v3_duct_warp_paths.py

  # smaller full joint tree; full mode grows as (actions * actions) ** depth
  conda run -n py312_numba python src/puct/plot_puct_v3_duct_warp_paths.py \
      --mode full --actions 2 --depth 4

  # custom output prefix and formats
  conda run -n py312_numba python src/puct/plot_puct_v3_duct_warp_paths.py \
      --output-prefix /tmp/my_duct_case --formats png,svg
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
import puct.puct_gpu_v3_duct as duct  # noqa: E402


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
        v3.REASON_BUSY_WINNER_RECALC: "BUSY_WINNER_RECALC",
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


def joint_slot(action0: int, action1: int) -> int:
    return (action0 << duct.DUCT_ACTION_BITS) | action1


def slot_action0(slot: int) -> int:
    return slot >> duct.DUCT_ACTION_BITS


def slot_action1(slot: int) -> int:
    return slot & duct.DUCT_ACTION_MASK


def _candidate_pairs(actions: int):
    pairs = []
    for a0 in range(actions):
        for a1 in range(actions):
            if a0 == 0 and a1 == 0:
                continue
            pairs.append((a0, a1))
    return pairs


def build_full_spec(actions: int, depth: int):
    branch = actions * actions
    level_offsets = [0]
    total = 1
    for d in range(1, depth + 1):
        width = branch**d
        total += width
        level_offsets.append(total - width)

    if total > v3.PACKED_NODE_LIMIT:
        raise ValueError(
            f"full mode would create {total} nodes, above PACKED_NODE_LIMIT="
            f"{v3.PACKED_NODE_LIMIT}. Lower --actions or --depth."
        )

    levels: dict[int, list[int]] = {}
    for lvl in range(depth + 1):
        start = level_offsets[lvl]
        width = branch**lvl
        levels[lvl] = list(range(start, start + width))

    expanded_edges: dict[int, list[tuple[int, int, int]]] = {}
    pairs = [(a0, a1) for a0 in range(actions) for a1 in range(actions)]
    for lvl in range(depth):
        start = level_offsets[lvl]
        width = branch**lvl
        child_start = level_offsets[lvl + 1]
        for i in range(width):
            parent = start + i
            cur = []
            for pair_idx, (a0, a1) in enumerate(pairs):
                child = child_start + i * branch + pair_idx
                cur.append((a0, a1, child))
            expanded_edges[parent] = cur

    terminal_nodes = set(levels[depth])
    main_path = [0]
    node = 0
    for _ in range(depth):
        child = expanded_edges[node][0][2]
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
        raise ValueError("sparse mode requires --actions >= 2")

    next_node = 0
    main_path = [next_node]
    next_node += 1
    for _ in range(depth):
        main_path.append(next_node)
        next_node += 1

    levels: dict[int, list[int]] = defaultdict(list)
    for lvl, nid in enumerate(main_path):
        levels[lvl].append(nid)

    expanded_edges: dict[int, list[tuple[int, int, int]]] = {}
    candidate_edges: dict[int, list[tuple[int, int, int]]] = {}
    terminal_nodes: set[int] = set()
    pairs = _candidate_pairs(actions)
    expand_hint_level = max(1, depth // 2)
    expand_hint_node = main_path[expand_hint_level]

    for lvl in range(depth):
        parent = main_path[lvl]
        main_child = main_path[lvl + 1]
        cur_expanded = [(0, 0, main_child)]

        max_extra = min(len(pairs), actions - 1)
        if lvl == 0 and root_extras is not None:
            desired_extra = root_extras
        else:
            desired_extra = extras_min if (lvl % 2 == 0) else extras_max
        extra_count = min(max_extra, max(0, desired_extra))

        extra_children = []
        for _ in range(extra_count):
            child = next_node
            next_node += 1
            extra_children.append(child)
            levels[lvl + 1].append(child)
            terminal_nodes.add(child)

        if lvl == expand_hint_level:
            cand = []
            for i, child in enumerate(extra_children):
                a0, a1 = pairs[i]
                cand.append((a0, a1, child))
            candidate_edges[parent] = cand
        else:
            for i, child in enumerate(extra_children):
                a0, a1 = pairs[i]
                cur_expanded.append((a0, a1, child))

        expanded_edges[parent] = cur_expanded

    terminal_nodes.add(main_path[-1])
    if next_node > v3.PACKED_NODE_LIMIT:
        raise ValueError(f"sparse spec created {next_node} nodes, above packed node limit.")

    return {
        "nodes": next_node,
        "levels": dict(levels),
        "expanded_edges": expanded_edges,
        "candidate_edges": candidate_edges,
        "terminal_nodes": terminal_nodes,
        "main_path": main_path,
        "expand_hint_node": expand_hint_node,
    }


def _preferred_pair_for_node(spec: dict, node: int):
    if node == spec["expand_hint_node"] and spec["candidate_edges"].get(node):
        a0, a1, _ = spec["candidate_edges"][node][0]
        return a0, a1
    return 0, 0


def make_case_from_spec(
    spec: dict,
    actions: int,
    warps: int,
    path_depth: int,
    node_n: int,
    best_w: float,
    side_w: float,
    action_n: int,
):
    trees = 1
    nodes = int(spec["nodes"])

    edge_child = np.full((trees, nodes, duct.DUCT_JOINT_ACTIONS), duct.DUCT_EDGE_UNEXPANDED, np.int32)
    edge_actions = np.full(
        (trees, nodes, duct.DUCT_JOINT_ACTIONS, duct.DUCT_PLAYERS),
        -1,
        np.int32,
    )
    action_w = np.full((trees, nodes, duct.DUCT_PLAYERS, actions), side_w, np.float32)
    action_n_arr = np.full((trees, nodes, duct.DUCT_PLAYERS, actions), action_n, np.int32)
    action_inflight = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_counts = np.full((trees, nodes, duct.DUCT_PLAYERS), actions, np.int32)
    node_n_arr = np.full((trees, nodes), node_n, np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    node_count = np.full((trees,), nodes, np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, path_depth), -1, np.int32)
    out_path_actions = np.full((trees, warps, path_depth, duct.DUCT_PLAYERS), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)

    for parent, children in spec["expanded_edges"].items():
        for a0, a1, child in children:
            slot = joint_slot(a0, a1)
            edge_child[0, parent, slot] = child
            edge_actions[0, parent, slot, 0] = a0
            edge_actions[0, parent, slot, 1] = a1

    for parent, children in spec["expanded_edges"].items():
        for a0, a1, child in children:
            if child in spec["terminal_nodes"]:
                slot = joint_slot(a0, a1)
                edge_child[0, parent, slot] = v3.NODE_EXPANDED_TERMINAL

    for nid in range(nodes):
        if nid in spec["terminal_nodes"]:
            continue
        a0, a1 = _preferred_pair_for_node(spec, nid)
        action_w[0, nid, 0, a0] = np.float32(best_w)
        action_w[0, nid, 1, a1] = np.float32(best_w)

    return {
        "edge_child": edge_child,
        "edge_actions": edge_actions,
        "action_w": action_w,
        "action_n": action_n_arr,
        "action_inflight": action_inflight,
        "action_counts": action_counts,
        "node_n": node_n_arr,
        "node_expand_inflight": node_expand_inflight,
        "node_count": node_count,
        "out_selected": out_selected,
        "out_path": out_path,
        "out_path_actions": out_path_actions,
        "out_len": out_len,
        "trees": trees,
        "warps": warps,
        "actions": actions,
        "path_depth": path_depth,
        "spec": spec,
    }


def compute_expected_pw(case: dict, host: dict, c_pw: float, alpha_pw: float):
    actions = int(case["actions"])
    nodes = host["node_n"].shape[1]
    allowed = np.zeros((nodes, duct.DUCT_PLAYERS), dtype=np.int32)
    n_eff = np.zeros((nodes, duct.DUCT_PLAYERS), dtype=np.int32)

    for node in range(nodes):
        if node in case["spec"]["terminal_nodes"]:
            continue

        for player in range(duct.DUCT_PLAYERS):
            count = int(host["action_counts"][0, node, player])
            if count <= 0:
                continue
            total = int(host["node_n"][0, node])
            for action in range(count):
                total += int(host["action_inflight"][0, node, player, action])
            total = max(total, 1)
            n_eff[node, player] = total

            if abs(alpha_pw - 0.5) < 1e-8:
                val = c_pw * math.sqrt(float(total))
            else:
                val = c_pw * math.pow(float(total), alpha_pw)
            a = int(math.ceil(val))
            allowed[node, player] = max(1, min(actions, a))
    return {"allowed": allowed, "n_eff": n_eff}


def run_select(case: dict, c_uct: float, c_pw: float, alpha_pw: float):
    d = {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}
    duct._select_kernel_duct_winner_recalc[d["trees"], d["warps"] * v3.WARP_SIZE](
        np.float32(c_uct),
        np.float32(c_pw),
        np.float32(alpha_pw),
        d["edge_child"],
        d["edge_actions"],
        d["action_w"],
        d["action_n"],
        d["action_inflight"],
        d["action_counts"],
        d["node_n"],
        d["node_expand_inflight"],
        d["node_count"],
        d["out_selected"],
        d["out_path"],
        d["out_path_actions"],
        d["out_len"],
    )
    cuda.synchronize()
    return {k: v.copy_to_host() if hasattr(v, "copy_to_host") else v for k, v in d.items()}


def _candidate_child(spec: dict, parent: int, slot: int) -> int:
    for a0, a1, child in spec["candidate_edges"].get(parent, []):
        if joint_slot(a0, a1) == slot:
            return child
    return -1


def collect_warp_infos(case: dict, host: dict):
    spec = case["spec"]
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
            joint = enc & 0xFF
            a0 = int(host["out_path_actions"][0, wid, d, 0])
            a1 = int(host["out_path_actions"][0, wid, d, 1])
            child = int(host["edge_child"][0, parent, joint])
            visual_child = child if child >= 0 else _candidate_child(spec, parent, joint)
            edges.append(
                {
                    "parent": parent,
                    "slot": joint,
                    "a0": a0,
                    "a1": a1,
                    "child": child,
                    "visual_child": visual_child,
                }
            )
            if child >= 0:
                path_nodes.append(child)

        exit_node = selected_node if kind != v3.SELECT_INVALID else (path_nodes[-1] if path_nodes else 0)
        infos.append(
            {
                "wid": wid,
                "kind": kind,
                "kind_name": kind_name(kind),
                "slot": slot,
                "slot_a0": slot_action0(slot) if 0 <= slot < duct.DUCT_JOINT_ACTIONS else -1,
                "slot_a1": slot_action1(slot) if 0 <= slot < duct.DUCT_JOINT_ACTIONS else -1,
                "selected_node": selected_node,
                "out_len": out_len,
                "edges": edges,
                "path_nodes": path_nodes,
                "exit_node": exit_node,
                "exit_reason": reason_name(reason_code),
                "exit_reason_code": reason_code,
            }
        )
    return infos


def _node_fill_color(nid: int, spec: dict):
    if nid in spec["main_path"]:
        return "#e9f6ec"
    if nid in spec["terminal_nodes"]:
        return "#fdf7e3"
    return "#eef6ff"


def _edge_style_for_slot(slot: int) -> tuple[str, float]:
    if slot == 0:
        return "#9a9a9a", 1.4
    return "#c6c6c6", 1.1


def _nonzero_action_inflight(host: dict, node: int, player: int):
    vals = []
    row = host["action_inflight"][0, node, player]
    for action, val in enumerate(row.tolist()):
        if int(val) != 0:
            vals.append(f"a{action}:{int(val)}")
    return ",".join(vals) if vals else "none"


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
    lines.append("digraph DuctWarpPaths {")
    lines.append(
        '  graph [rankdir=TB, bgcolor="white", pad="0.30", nodesep="0.35", '
        'ranksep="0.60", labelloc="t", fontsize=14, fontname="Helvetica", label="'
        + title
        + '"];'
    )
    lines.append('  node [shape=circle, style="filled", color="#9a9a9a", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [color="#bdbdbd", penwidth=1.2, arrowsize=0.7, fontname="Helvetica", fontsize=9];')

    for lvl in sorted(levels.keys()):
        rank_nodes = []
        for nid in levels[lvl]:
            fill = _node_fill_color(nid, spec)
            if nid in spec["terminal_nodes"]:
                label = f"n{nid}\\nterminal"
            else:
                label = (
                    f"n{nid}\\nallow=({int(allowed[nid, 0])},{int(allowed[nid, 1])})"
                    f"\\nNeff=({int(n_eff[nid, 0])},{int(n_eff[nid, 1])})"
                )
            lines.append(f'  n{nid} [label="{label}", fillcolor="{fill}"];')
            rank_nodes.append(f"n{nid}")
        lines.append("  { rank=same; " + "; ".join(rank_nodes) + "; }")

    for parent, children in spec["expanded_edges"].items():
        for a0, a1, child in children:
            slot = joint_slot(a0, a1)
            color, width = _edge_style_for_slot(slot)
            n0 = int(host["action_n"][0, parent, 0, a0])
            n1 = int(host["action_n"][0, parent, 1, a1])
            q0 = float(host["action_w"][0, parent, 0, a0]) / max(1, n0)
            q1 = float(host["action_w"][0, parent, 1, a1]) / max(1, n1)
            lines.append(
                f'  n{parent} -> n{child} [color="{color}", penwidth={width:.1f}, '
                f'label="s{slot} ({a0},{a1})\\nq=({q0:.1f},{q1:.1f})"];'
            )

    for parent, children in spec["candidate_edges"].items():
        for a0, a1, child in children:
            slot = joint_slot(a0, a1)
            lines.append(
                f'  n{parent} -> n{child} [style=dashed, color="#d9a400", '
                f'label="s{slot} ({a0},{a1}) candidate"];'
            )

    edge_use = defaultdict(list)
    for info in infos:
        for edge in info["edges"]:
            key = (edge["parent"], edge["visual_child"], edge["slot"])
            edge_use[key].append(info["wid"])

    for info in infos:
        wid = info["wid"]
        color = colors[wid % len(colors)]
        for edge in info["edges"]:
            parent = edge["parent"]
            child = edge["visual_child"]
            slot = edge["slot"]
            a0 = edge["a0"]
            a1 = edge["a1"]
            if child >= 0:
                style = "solid" if edge["child"] >= 0 else "dashed"
                lines.append(
                    f'  n{parent} -> n{child} [style={style}, color="{color}", '
                    f'penwidth=3.6, arrowsize=0.9, fontcolor="{color}", '
                    f'label="w{wid}:s{slot} ({a0},{a1})"];'
                )

    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#f8f8f8", color="#6f6f6f", fontsize=10];')
    for info in infos:
        wid = info["wid"]
        color = colors[wid % len(colors)]
        label = f"w{wid}: {info['kind_name']}\\nexit=n{info['exit_node']}, out_len={info['out_len']}"
        label += f"\\nreason={info['exit_reason']}"
        if 0 <= info["slot"] < duct.DUCT_JOINT_ACTIONS:
            label += f"\\nslot=s{info['slot']} ({info['slot_a0']},{info['slot_a1']})"
        fill = "#ffeef0" if info["kind"] == v3.SELECT_INVALID else "#eef8ff"
        lines.append(f'  ebox{wid} [label="{label}", color="{color}", fillcolor="{fill}"];')
        lines.append(f'  n{info["exit_node"]} -> ebox{wid} [style=dashed, color="{color}", arrowhead=none];')

    expand_nonzero = np.argwhere(host["node_expand_inflight"][0] > 0).reshape(-1)
    expand_txt = ", ".join(
        [f"n{int(n)}:{int(host['node_expand_inflight'][0, n])}" for n in expand_nonzero]
    ) or "none"
    expanding_slots = []
    where_expanding = np.argwhere(host["edge_child"][0] == duct.DUCT_EDGE_EXPANDING)
    for node, slot in where_expanding[:12]:
        expanding_slots.append(f"n{int(node)}:s{int(slot)}")
    expanding_txt = ", ".join(expanding_slots) if expanding_slots else "none"
    inflight_label = (
        "after select\\n"
        f"root p0={_nonzero_action_inflight(host, 0, 0)}\\n"
        f"root p1={_nonzero_action_inflight(host, 0, 1)}\\n"
        f"node_expand_inflight={{{expand_txt}}}\\n"
        f"edge_expanding={{{expanding_txt}}}"
    )
    lines.append(
        '  inflight [shape=note, style="filled", fillcolor="#f0fff0", '
        'color="#2ca02c", label="' + inflight_label + '"];'
    )
    lines.append('  ebox0 -> inflight [style=dashed, color="#2ca02c", arrowhead=none];')

    hot_lines = []
    for (parent, child, slot), wids in sorted(edge_use.items(), key=lambda x: (-len(x[1]), x[0][0], x[0][2])):
        if len(wids) > 1:
            hot_lines.append(f"n{parent}->n{child}:s{slot} " + ",".join([f"w{w}" for w in wids]))
    hot_txt = "\\n".join(hot_lines[:10]) if hot_lines else "none"
    lines.append(
        '  hot [shape=note, style="filled", fillcolor="#fff9e8", '
        'color="#c47f00", label="shared joint edges\\n' + hot_txt + '"];'
    )
    lines.append('  inflight -> hot [style=dashed, color="#c47f00", arrowhead=none];')

    if spec["expand_hint_node"] is not None:
        n = int(spec["expand_hint_node"])
        lines.append(
            '  hint [shape=note, style="filled", fillcolor="#fff8e1", '
            'color="#d9a400", label="expand-hint node\\npreferred pair is left unexpanded"];'
        )
        lines.append(f'  n{n} -> hint [style=dashed, color="#d9a400", arrowhead=none];')

    lines.append("}")
    return "\n".join(lines)


def write_summary(path: str, infos: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for info in infos:
            edge_txt = []
            for edge in info["edges"]:
                visual = ""
                if edge["visual_child"] >= 0 and edge["visual_child"] != edge["child"]:
                    visual = f",visual=n{edge['visual_child']}"
                edge_txt.append(
                    f"n{edge['parent']}:s{edge['slot']}({edge['a0']},{edge['a1']})->{edge['child']}{visual}"
                )
            f.write(
                f"warp={info['wid']} kind={info['kind_name']} slot={info['slot']} "
                f"pair=({info['slot_a0']},{info['slot_a1']}) "
                f"selected_node={info['selected_node']} out_len={info['out_len']} "
                f"reason={info['exit_reason']} path_nodes={info['path_nodes']} "
                f"edges=[{'; '.join(edge_txt)}]\n"
            )


def render_dot(dot_path: str, out_path: str, fmt: str):
    cmd = ["dot", f"-T{fmt}", dot_path, "-o", out_path]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Plot per-warp selection paths for puct_gpu_v3_duct.")
    p.add_argument("--mode", choices=["sparse", "full"], default="sparse", help="Tree construction mode.")
    p.add_argument("--actions", type=int, default=4, help="Marginal actions per player, max 16.")
    p.add_argument("--depth", type=int, default=4, help="Main path depth in edges.")
    p.add_argument("--warps", type=int, default=8, help="Warps per tree/block.")
    p.add_argument("--path-depth", type=int, default=None, help="Kernel path capacity. Default: depth+2.")
    p.add_argument("--extras-min", type=int, default=1, help="Sparse mode: min extra joint branches per level.")
    p.add_argument("--extras-max", type=int, default=2, help="Sparse mode: max extra joint branches per level.")
    p.add_argument(
        "--root-extras",
        type=int,
        default=None,
        help="Sparse mode: override root extra joint branches.",
    )
    p.add_argument("--c-uct", type=float, default=0.0, help="DUCT UCB exploration coefficient.")
    p.add_argument("--c-pw", type=float, default=1.0, help="Progressive widening c_pw.")
    p.add_argument("--alpha-pw", type=float, default=0.5, help="Progressive widening alpha_pw.")
    p.add_argument("--node-n", type=int, default=16, help="Host node visit count used by PW/UCB.")
    p.add_argument("--best-w", type=float, default=8.0, help="W value for preferred marginal actions.")
    p.add_argument("--side-w", type=float, default=0.0, help="W value for other marginal actions.")
    p.add_argument("--action-n", type=int, default=1, help="N value for all marginal actions.")
    p.add_argument(
        "--soft-winner",
        action="store_true",
        help="Require PUCT_DUCT_SOFT_WINNER=1; soft modes are compile-time macros.",
    )
    p.add_argument("--output-prefix", type=str, default="outputs/puct_v3_duct_warp_paths", help="Output file prefix.")
    p.add_argument("--formats", type=str, default="png,svg", help="Comma-separated formats rendered by dot.")
    p.add_argument("--no-render", action="store_true", help="Only write .dot/.txt, skip rendering.")
    return p.parse_args()


def main():
    args = parse_args()
    _require_cuda()

    if args.soft_winner and duct.DUCT_SOFT_WINNER == 0:
        raise RuntimeError(
            "--soft-winner requires running with PUCT_DUCT_SOFT_WINNER=1 before module import."
        )
    if args.soft_winner and duct.DUCT_SOFT_EXPAND == 0:
        raise RuntimeError(
            "--soft-winner requires PUCT_DUCT_SOFT_EXPAND=1 or the default soft-expand macro."
        )

    if args.actions < 1 or args.actions > duct.DUCT_MARGINAL_ACTIONS:
        raise ValueError(f"--actions must be in [1, {duct.DUCT_MARGINAL_ACTIONS}]")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.warps < 1:
        raise ValueError("--warps must be >= 1")

    path_depth = args.depth + 2 if args.path_depth is None else args.path_depth
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
        node_n=args.node_n,
        best_w=args.best_w,
        side_w=args.side_w,
        action_n=args.action_n,
    )
    host = run_select(
        case,
        c_uct=args.c_uct,
        c_pw=args.c_pw,
        alpha_pw=args.alpha_pw,
    )
    expected = compute_expected_pw(case, host, c_pw=args.c_pw, alpha_pw=args.alpha_pw)
    infos = collect_warp_infos(case, host)

    title = (
        f"DUCT warp paths | mode={args.mode} tree=1 actions={args.actions} "
        f"joint=256 depth={args.depth} warps={args.warps} "
        f"soft_winner={duct.DUCT_SOFT_WINNER} soft_expand={duct.DUCT_SOFT_EXPAND}"
    )
    dot_text = make_dot(case, host, infos, title=title, expected=expected)

    prefix = args.output_prefix
    dot_path = prefix + ".dot"
    txt_path = prefix + ".txt"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
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
