"""
DUCT select tests for puct_gpu_v3_duct.py.

Usage:
    conda run -n py312_numba python src/puct/test_puct_gpu_v3_duct.py
"""

import os
import sys

import numpy as np
from numba import cuda
from numba import config


PASS_MARK = "  [PASS]"
FAIL_MARK = "  [FAIL]"
SKIP_MARK = "  [SKIP]"

results: list[tuple[str, bool]] = []


def record(name: str, ok: bool, detail: str = ""):
    tag = PASS_MARK if ok else FAIL_MARK
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, ok))


def skip_all(reason: str):
    print(f"{SKIP_MARK}  DUCT CUDA tests  ({reason})")
    sys.exit(0)


if not cuda.is_available():
    skip_all("CUDA driver/device is not available")
if getattr(config, "ENABLE_CUDASIM", False):
    skip_all("CUDA simulator is not used for these warp-level tests")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import puct.puct_gpu_v3 as v3  # noqa: E402
import puct.puct_gpu_v3_duct as duct  # noqa: E402


def decode(raw: int):
    kind = (raw >> v3.PACKED_KIND_SHIFT) & v3.PACKED_KIND_MASK
    slot = (raw >> v3.PACKED_SLOT_SHIFT) & v3.PACKED_EDGE_MASK
    node = raw & v3.PACKED_NODE_MASK
    return int(kind), int(slot), int(node)


def decode_reason(raw: int) -> int:
    return int((raw >> v3.PACKED_REASON_SHIFT) & v3.PACKED_REASON_MASK)


def joint_slot(action0: int, action1: int) -> int:
    return (action0 << duct.DUCT_ACTION_BITS) | action1


def set_edge(case, parent: int, action0: int, action1: int, child: int):
    slot = joint_slot(action0, action1)
    case["edge_child"][0, parent, slot] = child
    case["edge_actions"][0, parent, slot] = (action0, action1)
    return slot


def make_duct_case(trees=1, nodes=4, actions=16, warps=1, path_depth=4):
    edge_child = np.full((trees, nodes, duct.DUCT_JOINT_ACTIONS), -1, np.int32)
    edge_actions = np.full(
        (trees, nodes, duct.DUCT_JOINT_ACTIONS, duct.DUCT_PLAYERS),
        -1,
        np.int32,
    )
    action_w = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.float32)
    action_n = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_inflight = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_counts = np.full((trees, nodes, duct.DUCT_PLAYERS), actions, np.int32)
    node_n = np.zeros((trees, nodes), np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    node_expanded = np.zeros((trees, nodes), np.int32)
    node_count = np.full((trees,), nodes, np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, max(1, path_depth)), -1, np.int32)
    out_path_actions = np.full(
        (trees, warps, max(1, path_depth), duct.DUCT_PLAYERS),
        -1,
        np.int32,
    )
    out_len = np.zeros((trees, warps), np.int32)
    return {
        "edge_child": edge_child,
        "edge_actions": edge_actions,
        "action_w": action_w,
        "action_n": action_n,
        "action_inflight": action_inflight,
        "action_counts": action_counts,
        "node_n": node_n,
        "node_expand_inflight": node_expand_inflight,
        "node_expanded": node_expanded,
        "node_count": node_count,
        "out_selected": out_selected,
        "out_path": out_path,
        "out_path_actions": out_path_actions,
        "out_len": out_len,
        "warps": warps,
        "trees": trees,
    }


def to_device(case):
    return {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}


def copy_back(dcase):
    return {k: v.copy_to_host() if hasattr(v, "copy_to_host") else v for k, v in dcase.items()}


@cuda.jit
def _release_duct_claims_kernel(
    virtual_loss,
    edge_child,
    action_inflight,
    node_expand_inflight,
    node_count,
    out_selected,
    out_path,
    out_path_actions,
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
    if node_limit <= np.int32(0) or node_limit > action_inflight.shape[1]:
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

    action_steps = plen - np.int32(1)
    if kind == np.int32(v3.SELECT_EXPAND):
        action_steps = plen

    action_capacity = action_inflight.shape[3]
    d = lane
    while d < action_steps:
        encoded = out_path[tree, wid, d]
        if encoded >= np.int32(0):
            parent = encoded >> np.int32(8)
            a0 = out_path_actions[tree, wid, d, np.int32(0)]
            a1 = out_path_actions[tree, wid, d, np.int32(1)]
            if (
                parent >= np.int32(0)
                and parent < node_limit
                and a0 >= np.int32(0)
                and a0 < action_capacity
                and a1 >= np.int32(0)
                and a1 < action_capacity
            ):
                cuda.atomic.sub(action_inflight, (tree, parent, np.int32(0), a0), virtual_loss)
                cuda.atomic.sub(action_inflight, (tree, parent, np.int32(1), a1), virtual_loss)
        d += np.int32(v3.WARP_SIZE)

    if lane == np.int32(0) and kind == np.int32(v3.SELECT_EXPAND):
        leaf = raw_selected & np.int32(v3.PACKED_NODE_MASK)
        slot = (raw_selected >> np.int32(v3.PACKED_SLOT_SHIFT)) & np.int32(v3.PACKED_EDGE_MASK)
        if leaf >= np.int32(0) and leaf < node_limit:
            if slot >= np.int32(0) and slot < edge_child.shape[2]:
                cuda.atomic.cas(
                    edge_child,
                    (tree, leaf, slot),
                    np.int32(duct.DUCT_EDGE_EXPANDING),
                    np.int32(duct.DUCT_EDGE_UNEXPANDED),
                )
            cuda.atomic.sub(node_expand_inflight, (tree, leaf), np.int32(1))


def launch_select_duct(d, c_uct=1.0, c_pw=1.0, alpha_pw=0.5):
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
        d["node_expanded"],
        d["node_count"],
        d["out_selected"],
        d["out_path"],
        d["out_path_actions"],
        d["out_len"],
    )


def launch_release_duct(dcase, trees=None, warps=None, virtual_loss=1):
    if trees is None:
        trees = dcase["trees"]
    if warps is None:
        warps = dcase["warps"]
    _release_duct_claims_kernel[trees, warps * v3.WARP_SIZE](
        np.int32(virtual_loss),
        dcase["edge_child"],
        dcase["action_inflight"],
        dcase["node_expand_inflight"],
        dcase["node_count"],
        dcase["out_selected"],
        dcase["out_path"],
        dcase["out_path_actions"],
        dcase["out_len"],
    )


def run_select_duct(case, c_uct=1.0, c_pw=1.0, alpha_pw=0.5):
    d = to_device(case)
    launch_select_duct(d, c_uct=c_uct, c_pw=c_pw, alpha_pw=alpha_pw)
    cuda.synchronize()
    return d, copy_back(d)


def run_release_duct(dcase, virtual_loss=1):
    launch_release_duct(dcase, virtual_loss=virtual_loss)
    cuda.synchronize()
    return copy_back(dcase)


@cuda.jit
def _claim_two_actions_partial_cas_kernel(action_inflight, out_held):
    lane = cuda.threadIdx.x & np.int32(31)
    scenario = cuda.blockIdx.x

    held = np.int32(0)
    if scenario == np.int32(0):
        held = duct._duct_claim_two_actions(
            action_inflight,
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            lane,
        )
    elif scenario == np.int32(1):
        held = duct._duct_claim_two_actions(
            action_inflight,
            np.int32(0),
            np.int32(1),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            lane,
        )

    if lane == np.int32(0):
        out_held[scenario] = held


def host_duct_inflight_sum(h):
    return int(h["action_inflight"].sum()) + int(h["node_expand_inflight"].sum())


def host_duct_inflight_nonnegative(h):
    return (
        not np.any(h["action_inflight"] < 0)
        and not np.any(h["node_expand_inflight"] < 0)
    )


def test_fresh_root_expand_roundtrip():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    case["node_n"][0, 0] = 1
    d, h = run_select_duct(case)
    kind, slot, node = decode(int(h["out_selected"][0, 0]))
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_EXPAND
        and node == 0
        and slot == 0
        and int(h["out_len"][0, 0]) == 1
        and int(h["out_path"][0, 0, 0]) == 0
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (0, 0)
        and int(h["action_inflight"][0, 0, 0, 0]) == 1
        and int(h["action_inflight"][0, 0, 1, 0]) == 1
        and int(h["node_expand_inflight"][0, 0]) == 1
        and host_duct_inflight_sum(h2) == 0
        and int(h2["edge_child"][0, 0, 0]) == duct.DUCT_EDGE_UNEXPANDED
    )
    record("fresh root expands joint action and releases claims", ok)


def test_independent_ucb_existing_joint_edge():
    case = make_duct_case(nodes=3, actions=16, path_depth=3)
    slot_01 = joint_slot(0, 1)
    slot_23 = joint_slot(2, 3)
    case["edge_child"][0, 0, slot_01] = 1
    case["edge_child"][0, 0, slot_23] = 2
    case["edge_actions"][0, 0, slot_01] = (0, 1)
    case["edge_actions"][0, 0, slot_23] = (2, 3)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_expanded"][0, 2] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 10
    case["action_n"][0, 0, :, :] = 1
    case["action_w"][0, 0, 0, 0] = 0.1
    case["action_w"][0, 0, 0, 2] = 4.0
    case["action_w"][0, 0, 1, 1] = 0.2
    case["action_w"][0, 0, 1, 3] = 3.5
    d, h = run_select_duct(case, c_pw=100.0, c_uct=0.0)
    kind, _, node = decode(int(h["out_selected"][0, 0]))
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_TERMINAL
        and node == 2
        and int(h["out_len"][0, 0]) == 2
        and int(h["out_path"][0, 0, 0]) == slot_23
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (2, 3)
        and int(h["action_inflight"][0, 0, 0, 2]) == 1
        and int(h["action_inflight"][0, 0, 1, 3]) == 1
        and host_duct_inflight_sum(h2) == 0
    )
    record("existing edge uses independent per-player UCB", ok)


def test_selected_unexpanded_joint_action_expands():
    case = make_duct_case(nodes=3, actions=16, path_depth=3)
    slot_01 = joint_slot(0, 1)
    slot_23 = joint_slot(2, 3)
    case["edge_child"][0, 0, slot_01] = 1
    case["edge_actions"][0, 0, slot_01] = (0, 1)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 1
    case["action_n"][0, 0, :, :] = 1
    case["action_w"][0, 0, 0, 2] = 9.0
    case["action_w"][0, 0, 1, 3] = 8.0
    d, h = run_select_duct(case, c_pw=100.0, c_uct=0.0)
    kind, slot, node = decode(int(h["out_selected"][0, 0]))
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_EXPAND
        and node == 0
        and slot == slot_23
        and int(h["out_path"][0, 0, 0]) == slot_23
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (2, 3)
        and int(h["edge_child"][0, 0, slot_23]) == duct.DUCT_EDGE_EXPANDING
        and int(h["action_inflight"][0, 0, 0, 2]) == 1
        and int(h["action_inflight"][0, 0, 1, 3]) == 1
        and int(h["node_expand_inflight"][0, 0]) == 1
        and host_duct_inflight_sum(h2) == 0
        and int(h2["edge_child"][0, 0, slot_23]) == duct.DUCT_EDGE_UNEXPANDED
    )
    record("selected unexpanded joint action expands", ok)


def test_multi_warp_claims_unique_pw_slots():
    case = make_duct_case(nodes=2, actions=16, warps=4, path_depth=2)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 16
    d, h = run_select_duct(case, c_pw=1.0, alpha_pw=0.5)
    slots = []
    expand_count = 0
    terminal_count = 0
    for wid in range(case["warps"]):
        kind, slot, node = decode(int(h["out_selected"][0, wid]))
        if kind == v3.SELECT_EXPAND:
            expand_count += 1
            slots.append(slot)
            if node != 0:
                record("multi-warp PW claims unique slots", False, f"bad node={node}")
                return
        elif kind == v3.SELECT_TERMINAL:
            terminal_count += 1
    h2 = run_release_duct(d)
    ok = (
        sorted(slots) == [joint_slot(0, 1), joint_slot(1, 2), joint_slot(2, 3), joint_slot(3, 0)]
        and expand_count == 4
        and terminal_count == 0
        and int(h["node_expand_inflight"][0, 0]) == 4
        and host_duct_inflight_nonnegative(h)
        and host_duct_inflight_sum(h2) == 0
    )
    record("multi-warp PW claims unique joint slots", ok, f"slots={slots}")


def test_terminal_root_returns_without_claims():
    case = make_duct_case(nodes=1, actions=16, path_depth=2)
    case["node_expanded"][0, 0] = v3.NODE_EXPANDED_TERMINAL
    _, h = run_select_duct(case)
    raw = int(h["out_selected"][0, 0])
    kind, _, node = decode(raw)
    ok = (
        kind == v3.SELECT_TERMINAL
        and node == 0
        and decode_reason(raw) == v3.REASON_OK_TERMINAL
        and int(h["out_len"][0, 0]) == 1
        and host_duct_inflight_sum(h) == 0
    )
    record("terminal root returns without claims", ok)


def test_expanding_joint_edge_returns_busy_and_rolls_back():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    slot = joint_slot(0, 0)
    case["edge_child"][0, 0, slot] = duct.DUCT_EDGE_EXPANDING
    case["node_n"][0, 0] = 1
    d, h = run_select_duct(case, c_pw=1.0, c_uct=0.0)
    raw = int(h["out_selected"][0, 0])
    kind, _, node = decode(raw)
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_BUSY
        and node == 0
        and decode_reason(raw) == v3.REASON_BUSY_EXPAND_INFLIGHT
        and int(h["out_len"][0, 0]) == 0
        and int(h["edge_child"][0, 0, slot]) == duct.DUCT_EDGE_EXPANDING
        and host_duct_inflight_sum(h) == 0
        and host_duct_inflight_sum(h2) == 0
    )
    record("expanding joint edge returns busy and rolls back", ok)


def test_invalid_child_oob_rolls_back_actions():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    slot = set_edge(case, parent=0, action0=0, action1=0, child=3)
    case["node_n"][0, 0] = 1
    _, h = run_select_duct(case, c_pw=1.0, c_uct=0.0)
    raw = int(h["out_selected"][0, 0])
    kind, _, _ = decode(raw)
    ok = (
        kind == v3.SELECT_INVALID
        and decode_reason(raw) == v3.REASON_INVALID_CHILD_OOB
        and int(h["edge_child"][0, 0, slot]) == 3
        and host_duct_inflight_sum(h) == 0
    )
    record("invalid child OOB rolls back action claims", ok)


def test_depth_limit_after_existing_edge():
    case = make_duct_case(nodes=2, actions=16, path_depth=1)
    slot = set_edge(case, parent=0, action0=0, action1=0, child=1)
    case["node_n"][0, 0] = 1
    d, h = run_select_duct(case, c_pw=1.0, c_uct=0.0)
    raw = int(h["out_selected"][0, 0])
    kind, _, node = decode(raw)
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_DEPTH_LIMIT
        and node == 1
        and decode_reason(raw) == v3.REASON_OK_DEPTH_LIMIT
        and int(h["out_len"][0, 0]) == 2
        and int(h["out_path"][0, 0, 0]) == slot
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (0, 0)
        and int(h["action_inflight"][0, 0, 0, 0]) == 1
        and int(h["action_inflight"][0, 0, 1, 0]) == 1
        and host_duct_inflight_sum(h2) == 0
    )
    record("depth limit after existing edge releases path claims", ok)


def test_invalid_action_count_returns_invalid():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    case["action_counts"][0, 0, 0] = 17
    _, h = run_select_duct(case)
    raw = int(h["out_selected"][0, 0])
    kind, _, _ = decode(raw)
    ok = (
        kind == v3.SELECT_INVALID
        and decode_reason(raw) == v3.REASON_INVALID_NODE_INFO
        and host_duct_inflight_sum(h) == 0
    )
    record("invalid action count returns invalid without claims", ok)


def test_invalid_shape_rejects_non_256_joint_edges():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    case["edge_child"] = np.full((1, 2, 128), -1, np.int32)
    case["edge_actions"] = np.full((1, 2, 128, duct.DUCT_PLAYERS), -1, np.int32)
    _, h = run_select_duct(case)
    raw = int(h["out_selected"][0, 0])
    kind, _, _ = decode(raw)
    ok = (
        kind == v3.SELECT_INVALID
        and decode_reason(raw) == v3.REASON_INVALID_SHAPE
        and host_duct_inflight_sum(h) == 0
    )
    record("invalid shape rejects non-256 joint edge table", ok)


def test_pw_window_hides_high_value_actions():
    case = make_duct_case(nodes=3, actions=16, path_depth=2)
    slot_00 = set_edge(case, parent=0, action0=0, action1=0, child=1)
    slot_23 = set_edge(case, parent=0, action0=2, action1=3, child=2)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_expanded"][0, 2] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 1
    case["action_n"][0, 0, :, :] = 1
    case["action_w"][0, 0, 0, 2] = 100.0
    case["action_w"][0, 0, 1, 3] = 100.0
    d, h = run_select_duct(case, c_pw=1.0, alpha_pw=0.5, c_uct=0.0)
    raw = int(h["out_selected"][0, 0])
    kind, _, node = decode(raw)
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_TERMINAL
        and node == 1
        and int(h["out_path"][0, 0, 0]) == slot_00
        and int(h["out_path"][0, 0, 0]) != slot_23
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (0, 0)
        and int(h["action_inflight"][0, 0, 0, 2]) == 0
        and int(h["action_inflight"][0, 0, 1, 3]) == 0
        and host_duct_inflight_sum(h2) == 0
    )
    record("PW window hides high-value actions outside prefix", ok)


def test_variable_action_counts_selects_valid_prefix_actions():
    case = make_duct_case(nodes=2, actions=16, path_depth=2)
    slot = set_edge(case, parent=0, action0=2, action1=4, child=1)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 10
    case["action_counts"][0, 0, 0] = 3
    case["action_counts"][0, 0, 1] = 5
    case["action_n"][0, 0, 0, :3] = 1
    case["action_n"][0, 0, 1, :5] = 1
    case["action_w"][0, 0, 0, 2] = 7.0
    case["action_w"][0, 0, 1, 4] = 8.0
    d, h = run_select_duct(case, c_pw=100.0, c_uct=0.0)
    raw = int(h["out_selected"][0, 0])
    kind, _, node = decode(raw)
    h2 = run_release_duct(d)
    ok = (
        kind == v3.SELECT_TERMINAL
        and node == 1
        and int(h["out_path"][0, 0, 0]) == slot
        and tuple(int(x) for x in h["out_path_actions"][0, 0, 0]) == (2, 4)
        and int(h["action_inflight"][0, 0, 0, 2]) == 1
        and int(h["action_inflight"][0, 0, 1, 4]) == 1
        and host_duct_inflight_sum(h2) == 0
    )
    record("variable action counts select valid per-player prefixes", ok)


def test_claim_two_actions_rolls_back_partial_cas():
    case = make_duct_case(nodes=2, actions=16, path_depth=1)
    case["action_inflight"][0, 0, 1, 0] = 1
    case["action_inflight"][0, 1, 0, 0] = 1
    d = to_device(case)
    out_held = cuda.to_device(np.full((2,), -1, np.int32))

    _claim_two_actions_partial_cas_kernel[2, v3.WARP_SIZE](
        d["action_inflight"],
        out_held,
    )
    cuda.synchronize()

    h = copy_back(d)
    held = out_held.copy_to_host()
    ok = (
        tuple(int(x) for x in held) == (0, 0)
        and int(h["action_inflight"][0, 0, 0, 0]) == 0
        and int(h["action_inflight"][0, 0, 1, 0]) == 1
        and int(h["action_inflight"][0, 1, 0, 0]) == 1
        and int(h["action_inflight"][0, 1, 1, 0]) == 0
        and host_duct_inflight_nonnegative(h)
        and host_duct_inflight_sum(h) == 2
    )
    record("two-action claim rolls back either partial CAS", ok, f"held={held.tolist()}")


def test_soft_winner_keeps_busy_claim():
    if duct.DUCT_SOFT_WINNER == 0:
        record("soft winner keeps busy duplicate claim", True, "skipped: PUCT_DUCT_SOFT_WINNER=0")
        return

    case = make_duct_case(nodes=2, actions=16, warps=2, path_depth=2)
    slot = set_edge(case, parent=0, action0=0, action1=0, child=1)
    case["node_expanded"][0, 1] = v3.NODE_EXPANDED_TERMINAL
    case["node_n"][0, 0] = 1
    d, h = run_select_duct(case, c_pw=1.0, c_uct=0.0)
    kinds = [decode(int(h["out_selected"][0, wid]))[0] for wid in range(case["warps"])]
    h2 = run_release_duct(d)
    ok = (
        kinds.count(v3.SELECT_TERMINAL) == 2
        and int(h["out_path"][0, 0, 0]) == slot
        and int(h["out_path"][0, 1, 0]) == slot
        and int(h["action_inflight"][0, 0, 0, 0]) == 2
        and int(h["action_inflight"][0, 0, 1, 0]) == 2
        and host_duct_inflight_sum(h2) == 0
    )
    record("soft winner keeps busy duplicate claim", ok, f"kinds={kinds}")


def summarize_results_and_exit():
    failed = [name for name, ok in results if not ok]
    print("================================================================")
    if failed:
        print(f"  FAILED: {len(failed)} test(s)")
        for name in failed:
            print(f"    - {name}")
        sys.exit(1)
    print(f"  ALL PASSED: {len(results)} test(s)")
    print("================================================================")


def main():
    test_fresh_root_expand_roundtrip()
    test_independent_ucb_existing_joint_edge()
    test_selected_unexpanded_joint_action_expands()
    test_multi_warp_claims_unique_pw_slots()
    test_terminal_root_returns_without_claims()
    test_expanding_joint_edge_returns_busy_and_rolls_back()
    test_invalid_child_oob_rolls_back_actions()
    test_depth_limit_after_existing_edge()
    test_invalid_action_count_returns_invalid()
    test_invalid_shape_rejects_non_256_joint_edges()
    test_pw_window_hides_high_value_actions()
    test_variable_action_counts_selects_valid_prefix_actions()
    test_claim_two_actions_rolls_back_partial_cas()
    test_soft_winner_keeps_busy_claim()
    summarize_results_and_exit()


if __name__ == "__main__":
    main()
