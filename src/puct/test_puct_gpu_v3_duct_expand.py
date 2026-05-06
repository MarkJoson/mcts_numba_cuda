"""
DUCT expand tests for puct_gpu_v3_duct.py.

Usage:
    conda run -n py312_numba python src/puct/test_puct_gpu_v3_duct_expand.py
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
    print(f"{SKIP_MARK}  DUCT expand CUDA tests  ({reason})")
    sys.exit(0)


if not cuda.is_available():
    skip_all("CUDA driver/device is not available")
if getattr(config, "ENABLE_CUDASIM", False):
    skip_all("CUDA simulator is not used for these warp-level tests")


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import puct.puct_gpu_v3 as v3  # noqa: E402
import puct.puct_gpu_v3_duct as duct  # noqa: E402

try:
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402

    TORCH_CUDA = torch.cuda.is_available()
    if TORCH_CUDA:
        _torch_cuda_sentinel = torch.zeros(1, device="cuda")
except Exception:
    torch = None
    nn = object
    TORCH_CUDA = False


def joint_slot(action0: int, action1: int) -> int:
    return (action0 << duct.DUCT_ACTION_BITS) | action1


def pack_expand(parent: int, slot: int) -> np.int32:
    return np.int32(
        (v3.REASON_OK_EXPAND << v3.PACKED_REASON_SHIFT)
        | (v3.SELECT_EXPAND << v3.PACKED_KIND_SHIFT)
        | (slot << v3.PACKED_SLOT_SHIFT)
        | parent
    )


class NumpyExpandBridge:
    names = (
        "node_states",
        "node_action_targets",
        "node_action_probs",
        "expand_valid",
        "expand_tree",
        "expand_parent",
        "expand_slot",
        "expand_child",
        "expand_next_states",
        "expand_done",
        "expand_child_action_targets",
        "expand_child_action_probs",
    )

    def __init__(self, trees: int, nodes: int, state_dim: int,
                 num_agents: int, warps: int):
        self.n_trees = trees
        self.node_capacity = nodes
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.max_warps = warps
        self.node_states = np.zeros((trees, nodes, state_dim), np.float32)
        self.node_action_targets = np.zeros((trees, nodes, 2, 16, num_agents, 2), np.float32)
        self.node_action_probs = np.zeros((trees, nodes, 2, 16), np.float32)
        self.expand_valid = np.zeros((trees, warps), np.int32)
        self.expand_tree = np.full((trees, warps), -1, np.int32)
        self.expand_parent = np.full((trees, warps), -1, np.int32)
        self.expand_slot = np.full((trees, warps), -1, np.int32)
        self.expand_child = np.full((trees, warps), -1, np.int32)
        self.expand_next_states = np.zeros((trees, warps, state_dim), np.float32)
        self.expand_done = np.zeros((trees, warps), np.int32)
        self.expand_child_action_targets = np.zeros((trees, warps, 2, 16, num_agents, 2), np.float32)
        self.expand_child_action_probs = np.zeros((trees, warps, 2, 16), np.float32)

    def to_device(self):
        return DeviceExpandBridge(self)


class DeviceExpandBridge:
    def __init__(self, host: NumpyExpandBridge):
        self.n_trees = host.n_trees
        self.node_capacity = host.node_capacity
        self.state_dim = host.state_dim
        self.num_agents = host.num_agents
        self.max_warps = host.max_warps
        self._host = host
        for name in host.names:
            setattr(self, name, getattr(host, name).copy())
            setattr(self, f"dev_{name}", cuda.to_device(getattr(host, name)))

    def copy_field(self, name: str):
        return getattr(self, f"dev_{name}").copy_to_host()

    def copy_payload_to_device(self):
        for name in (
            "expand_next_states",
            "expand_done",
            "expand_child_action_targets",
            "expand_child_action_probs",
        ):
            getattr(self, f"dev_{name}").copy_to_device(getattr(self, name))

    def copy_status_to_device(self):
        self.dev_expand_valid.copy_to_device(self.expand_valid)


def make_case(trees=1, nodes=8, warps=1, state_dim=8, num_agents=4, actions=16):
    edge_child = np.full((trees, nodes, duct.DUCT_JOINT_ACTIONS), duct.DUCT_EDGE_UNEXPANDED, np.int32)
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
    tree_nodes = np.ones((trees,), np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, 2), -1, np.int32)
    out_path_actions = np.full((trees, warps, 2, duct.DUCT_PLAYERS), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)
    bridge = NumpyExpandBridge(trees, nodes, state_dim, num_agents, warps)
    return {
        "edge_child": edge_child,
        "edge_actions": edge_actions,
        "action_w": action_w,
        "action_n": action_n,
        "action_inflight": action_inflight,
        "action_counts": action_counts,
        "node_n": node_n,
        "node_expand_inflight": node_expand_inflight,
        "tree_nodes": tree_nodes,
        "out_selected": out_selected,
        "out_path": out_path,
        "out_path_actions": out_path_actions,
        "out_len": out_len,
        "bridge": bridge,
        "trees": trees,
        "warps": warps,
        "nodes": nodes,
        "state_dim": state_dim,
        "num_agents": num_agents,
    }


def to_device(case):
    out = {}
    for k, v in case.items():
        if isinstance(v, np.ndarray):
            out[k] = cuda.to_device(v)
        elif isinstance(v, NumpyExpandBridge):
            out[k] = v.to_device()
        else:
            out[k] = v
    return out


def copy_back(dcase):
    return {k: v.copy_to_host() if hasattr(v, "copy_to_host") else v for k, v in dcase.items()}


def numpy_case_to_torch_stage1(case):
    if not TORCH_CUDA:
        raise RuntimeError("PyTorch CUDA is required for DUCT expand stage1")
    return {
        "edge_child": torch.as_tensor(case["edge_child"], dtype=torch.int32, device="cuda"),
        "tree_nodes": torch.as_tensor(case["tree_nodes"], dtype=torch.int32, device="cuda"),
        "out_selected": torch.as_tensor(case["out_selected"], dtype=torch.int32, device="cuda"),
        "bridge": duct.DuctExpandBridge(
            case["trees"],
            case["nodes"],
            case["state_dim"],
            case["num_agents"],
            case["warps"],
        ),
    }


def seed_root_slots(bridge):
    state = np.arange(bridge.state_dim, dtype=np.float32)
    bridge.node_states[0, 0] = state
    for p in range(duct.DUCT_PLAYERS):
        for a in range(duct.DUCT_MARGINAL_ACTIONS):
            bridge.node_action_probs[0, 0, p, a] = float(a + 1) / 100.0
            for ag in range(bridge.num_agents):
                bridge.node_action_targets[0, 0, p, a, ag, 0] = 1000.0 * p + 100.0 * a + 10.0 * ag
                bridge.node_action_targets[0, 0, p, a, ag, 1] = 1000.0 * p + 100.0 * a + 10.0 * ag + 1.0


def seed_child_payload(bridge, done: int = 0):
    for wid in range(bridge.max_warps):
        bridge.expand_next_states[0, wid] = np.arange(
            bridge.state_dim, dtype=np.float32,
        ) + 10.0 * (wid + 1)
        bridge.expand_done[0, wid] = done
        for p in range(duct.DUCT_PLAYERS):
            for a in range(duct.DUCT_MARGINAL_ACTIONS):
                bridge.expand_child_action_probs[0, wid, p, a] = 0.01 * (a + 1 + p)
                bridge.expand_child_action_targets[0, wid, p, a] = (
                    float(100 * wid + 10 * p + a)
                )
    if isinstance(bridge, DeviceExpandBridge):
        bridge.copy_payload_to_device()


def mark_expand_job(case, wid: int, action0: int, action1: int):
    slot = joint_slot(action0, action1)
    case["out_selected"][0, wid] = pack_expand(0, slot)
    case["out_path"][0, wid, 0] = slot
    case["out_path_actions"][0, wid, 0] = (action0, action1)
    case["out_len"][0, wid] = 1
    case["edge_child"][0, 0, slot] = duct.DUCT_EDGE_EXPANDING
    case["edge_actions"][0, 0, slot] = (action0, action1)
    case["node_expand_inflight"][0, 0] += 1
    case["action_inflight"][0, 0, 0, action0] += 1
    case["action_inflight"][0, 0, 1, action1] += 1
    return slot


def launch_prepare_torch(case):
    return duct.prepare_duct_expand_stage1_torch(
        case["bridge"],
        case["edge_child"],
        case["tree_nodes"],
        case["out_selected"],
    )


def launch_commit(dcase):
    b = dcase["bridge"]
    duct._commit_expand_stage2_duct[dcase["trees"], dcase["warps"] * v3.WARP_SIZE](
        dcase["edge_child"],
        dcase["edge_actions"],
        dcase["action_w"],
        dcase["action_n"],
        dcase["action_inflight"],
        dcase["action_counts"],
        dcase["node_n"],
        dcase["node_expand_inflight"],
        dcase["tree_nodes"],
        dcase["out_path"],
        dcase["out_path_actions"],
        dcase["out_len"],
        b.dev_node_states,
        b.dev_node_action_targets,
        b.dev_node_action_probs,
        b.dev_expand_valid,
        b.dev_expand_parent,
        b.dev_expand_slot,
        b.dev_expand_child,
        b.dev_expand_next_states,
        b.dev_expand_done,
        b.dev_expand_child_action_targets,
        b.dev_expand_child_action_probs,
    )


def prepare_case_for_commit(case):
    torch_case = numpy_case_to_torch_stage1(case)
    launch_prepare_torch(torch_case)
    cuda_case = to_device(case)
    for name in (
        "expand_valid",
        "expand_tree",
        "expand_parent",
        "expand_slot",
        "expand_child",
    ):
        getattr(cuda_case["bridge"], f"dev_{name}").copy_to_device(
            getattr(torch_case["bridge"], name).detach().cpu().numpy()
        )
    cuda_case["tree_nodes"].copy_to_device(torch_case["tree_nodes"].detach().cpu().numpy())
    return cuda_case


def test_fresh_expand_commits_child():
    case = make_case(nodes=6, warps=1, state_dim=8, num_agents=3)
    seed_root_slots(case["bridge"])
    slot = mark_expand_job(case, 0, 2, 3)
    d = prepare_case_for_commit(case)
    seed_child_payload(d["bridge"])
    launch_commit(d)
    cuda.synchronize()
    h = copy_back(d)
    b = case["bridge"]
    b_node_states = d["bridge"].copy_field("node_states")
    b_node_action_targets = d["bridge"].copy_field("node_action_targets")
    b_expand_valid = d["bridge"].copy_field("expand_valid")
    ok = (
        int(h["edge_child"][0, 0, slot]) == 1
        and int(h["tree_nodes"][0]) == 2
        and int(h["node_expand_inflight"][0, 0]) == 0
        and tuple(int(x) for x in h["action_counts"][0, 1]) == (16, 16)
        and int(h["action_inflight"][0, 0, 0, 2]) == 1
        and int(h["action_inflight"][0, 0, 1, 3]) == 1
        and np.allclose(b_node_states[0, 1], np.arange(8, dtype=np.float32) + 10.0)
        and float(b_node_action_targets[0, 1, 1, 4, 0, 0]) == 14.0
        and int(b_expand_valid[0, 0]) == duct.DUCT_EXPAND_JOB_EMPTY
    )
    record("fresh expand commits child and publishes edge", ok)


def test_multi_warp_expand_allocates_unique_children():
    case = make_case(nodes=8, warps=4, state_dim=4, num_agents=2)
    seed_root_slots(case["bridge"])
    slots = [mark_expand_job(case, wid, wid, (wid + 1) % 4) for wid in range(4)]
    d = prepare_case_for_commit(case)
    seed_child_payload(d["bridge"])
    launch_commit(d)
    cuda.synchronize()
    h = copy_back(d)
    children = [int(h["edge_child"][0, 0, slot]) for slot in slots]
    ok = (
        sorted(children) == [1, 2, 3, 4]
        and int(h["tree_nodes"][0]) == 5
        and int(h["node_expand_inflight"][0, 0]) == 0
        and len(set(children)) == 4
    )
    record("multi-warp expand allocates unique child ids", ok, f"children={children}")


def test_non_expand_and_stale_jobs_are_ignored():
    case = make_case(nodes=4, warps=2, state_dim=4, num_agents=2)
    seed_root_slots(case["bridge"])
    stale_slot = joint_slot(1, 1)
    case["out_selected"][0, 0] = v3.PACKED_INVALID
    case["out_selected"][0, 1] = pack_expand(0, stale_slot)
    case["edge_child"][0, 0, stale_slot] = 2
    torch_case = numpy_case_to_torch_stage1(case)
    info = launch_prepare_torch(torch_case)
    ok = (
        info == {"num_jobs": 0, "num_ready": 0, "num_dropped": 0}
        and int(case["tree_nodes"][0].item()) == 1
        and int(case["edge_child"][0, 0, stale_slot].item()) == 2
        and int(case["bridge"].expand_valid[0, 0].item()) == duct.DUCT_EXPAND_JOB_EMPTY
        and int(case["bridge"].expand_valid[0, 1].item()) == duct.DUCT_EXPAND_JOB_EMPTY
    )
    record("non-SELECT_EXPAND and stale jobs are ignored", ok)


def test_failed_commit_discards_job_without_rollback():
    case = make_case(nodes=4, warps=1, state_dim=4, num_agents=2)
    seed_root_slots(case["bridge"])
    slot = mark_expand_job(case, 0, 4, 5)
    d = prepare_case_for_commit(case)
    d["bridge"].expand_valid[0, 0] = duct.DUCT_EXPAND_JOB_FAILED
    d["bridge"].copy_status_to_device()
    launch_commit(d)
    cuda.synchronize()
    h = copy_back(d)
    ok = (
        int(h["edge_child"][0, 0, slot]) == duct.DUCT_EDGE_EXPANDING
        and int(h["node_expand_inflight"][0, 0]) == 1
        and int(h["action_inflight"][0, 0, 0, 4]) == 1
        and int(h["action_inflight"][0, 0, 1, 5]) == 1
        and int(d["bridge"].copy_field("expand_valid")[0, 0]) == duct.DUCT_EXPAND_JOB_EMPTY
    )
    record("failed commit discards job without rollback", ok)


def test_terminal_commit_publishes_terminal_sentinel():
    case = make_case(nodes=4, warps=1, state_dim=4, num_agents=2)
    seed_root_slots(case["bridge"])
    slot = mark_expand_job(case, 0, 1, 2)
    d = prepare_case_for_commit(case)
    seed_child_payload(d["bridge"], done=1)
    launch_commit(d)
    cuda.synchronize()
    h = copy_back(d)
    ok = (
        int(h["edge_child"][0, 0, slot]) == v3.NODE_EXPANDED_TERMINAL
        and int(h["node_expand_inflight"][0, 0]) == 0
        and tuple(int(x) for x in h["action_counts"][0, 1]) == (1, 1)
        and int(d["bridge"].copy_field("expand_valid")[0, 0]) == duct.DUCT_EXPAND_JOB_EMPTY
    )
    record("terminal commit publishes NODE_EXPANDED_TERMINAL sentinel", ok)


def make_torch_stage1_tensors(nodes=4, warps=2):
    device = "cuda"
    edge_child = torch.full(
        (1, nodes, duct.DUCT_JOINT_ACTIONS),
        duct.DUCT_EDGE_UNEXPANDED,
        dtype=torch.int32,
        device=device,
    )
    tree_nodes = torch.ones((1,), dtype=torch.int32, device=device)
    out_selected = torch.full((1, warps), v3.PACKED_INVALID, dtype=torch.int32, device=device)
    bridge = duct.DuctExpandBridge(1, nodes, 5, 2, warps)
    return {
        "edge_child": edge_child,
        "tree_nodes": tree_nodes,
        "out_selected": out_selected,
        "bridge": bridge,
    }


def mark_torch_stage1_job(case, wid: int, action0: int, action1: int):
    slot = joint_slot(action0, action1)
    case["out_selected"][0, wid] = int(pack_expand(0, slot))
    case["edge_child"][0, 0, slot] = duct.DUCT_EDGE_EXPANDING
    return slot


def test_torch_stage1_allocates_ready_jobs_without_kernel():
    if not TORCH_CUDA:
        record("torch stage1 allocates READY jobs without CUDA prepare kernel", True, "skipped: PyTorch CUDA unavailable")
        return
    case = make_torch_stage1_tensors(nodes=6, warps=3)
    mark_torch_stage1_job(case, 0, 2, 3)
    stale_slot = joint_slot(4, 5)
    case["out_selected"][0, 1] = int(pack_expand(0, stale_slot))
    case["edge_child"][0, 0, stale_slot] = 9
    mark_torch_stage1_job(case, 2, 6, 7)
    info = duct.prepare_duct_expand_stage1_torch(
        case["bridge"],
        case["edge_child"],
        case["tree_nodes"],
        case["out_selected"],
    )
    torch.cuda.synchronize()
    ready = case["bridge"].expand_valid.cpu().numpy().tolist()[0]
    children = case["bridge"].expand_child.cpu().numpy().tolist()[0]
    ok = (
        info == {"num_jobs": 2, "num_ready": 2, "num_dropped": 0}
        and int(case["tree_nodes"][0].item()) == 3
        and ready == [duct.DUCT_EXPAND_JOB_READY, duct.DUCT_EXPAND_JOB_EMPTY, duct.DUCT_EXPAND_JOB_READY]
        and children == [1, -1, 2]
    )
    record("torch stage1 allocates READY jobs without CUDA prepare kernel", ok, f"children={children}")


def test_torch_stage1_drops_overflow_jobs_without_rollback():
    if not TORCH_CUDA:
        record("torch stage1 drops overflow jobs without rollback", True, "skipped: PyTorch CUDA unavailable")
        return
    case = make_torch_stage1_tensors(nodes=1, warps=1)
    slot = mark_torch_stage1_job(case, 0, 0, 1)
    info = duct.prepare_duct_expand_stage1_torch(
        case["bridge"],
        case["edge_child"],
        case["tree_nodes"],
        case["out_selected"],
    )
    torch.cuda.synchronize()
    ok = (
        info == {"num_jobs": 1, "num_ready": 0, "num_dropped": 1}
        and int(case["tree_nodes"][0].item()) == 1
        and int(case["edge_child"][0, 0, slot].item()) == duct.DUCT_EDGE_EXPANDING
        and int(case["bridge"].expand_valid[0, 0].item()) == duct.DUCT_EXPAND_JOB_EMPTY
    )
    record("torch stage1 drops overflow jobs without rollback", ok)


class FixedPolicy(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.to(x.device).unsqueeze(0).expand(x.shape[0], -1)


def make_candidate_targets(num_agents=2):
    table = torch.zeros((2, 16, num_agents, 2), dtype=torch.float32, device="cuda")
    for p in range(2):
        for a in range(16):
            table[p, a, :, :] = float(100 * p + a)
    return table


def test_policy_sampling_prefers_policy_first_slot():
    if not TORCH_CUDA:
        record("policy sampling fills policy-biased slot permutations", True, "skipped: PyTorch CUDA unavailable")
        return
    bridge = duct.DuctExpandBridge(1, 2, 5, 2, 1)
    states = torch.zeros((1, 5), dtype=torch.float32, device="cuda")
    logits0 = [-20.0] * 16
    logits1 = [-20.0] * 16
    logits0[7] = 20.0
    logits1[3] = 20.0
    gen = torch.Generator(device="cuda")
    gen.manual_seed(123)
    targets, probs = duct.fill_duct_node_action_slots(
        bridge,
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
        states,
        make_candidate_targets(),
        FixedPolicy(logits0).cuda(),
        FixedPolicy(logits1).cuda(),
        uniform_sample_prob=0.0,
        generator=gen,
    )
    ok = (
        targets.shape == (1, 2, 16, 2, 2)
        and float(probs[0, 0, 0]) > 0.99
        and float(probs[0, 1, 0]) > 0.99
    )
    record("policy sampling fills policy-biased slot permutations", ok)


def test_uniform_sampling_uses_uniform_probabilities():
    if not TORCH_CUDA:
        record("uniform_sample_prob=1 uses uniform probabilities", True, "skipped: PyTorch CUDA unavailable")
        return
    bridge = duct.DuctExpandBridge(1, 2, 5, 2, 1)
    states = torch.zeros((1, 5), dtype=torch.float32, device="cuda")
    logits = [100.0] + [-100.0] * 15
    gen = torch.Generator(device="cuda")
    gen.manual_seed(321)
    targets, probs = duct.fill_duct_node_action_slots(
        bridge,
        [0],
        [0],
        states,
        make_candidate_targets(),
        FixedPolicy(logits).cuda(),
        FixedPolicy(logits).cuda(),
        uniform_sample_prob=1.0,
        generator=gen,
    )
    expected = torch.full((16,), 1.0 / 16.0, device="cuda")
    ok = (
        targets.shape == (1, 2, 16, 2, 2)
        and torch.allclose(probs[0, 0], expected)
        and torch.allclose(probs[0, 1], expected)
    )
    record("uniform_sample_prob=1 uses uniform probabilities", ok)


def test_nan_inf_policy_falls_back_to_uniform():
    if not TORCH_CUDA:
        record("NaN/Inf policy logits fall back to uniform", True, "skipped: PyTorch CUDA unavailable")
        return
    bridge = duct.DuctExpandBridge(1, 2, 5, 2, 1)
    states = torch.zeros((1, 5), dtype=torch.float32, device="cuda")
    bad0 = [float("nan")] * 16
    bad1 = [float("inf")] + [0.0] * 15
    gen = torch.Generator(device="cuda")
    gen.manual_seed(111)
    _, probs = duct.fill_duct_node_action_slots(
        bridge,
        [0],
        [0],
        states,
        make_candidate_targets(),
        FixedPolicy(bad0).cuda(),
        FixedPolicy(bad1).cuda(),
        uniform_sample_prob=0.0,
        generator=gen,
    )
    expected = torch.full((16,), 1.0 / 16.0, device="cuda")
    ok = torch.allclose(probs[0, 0], expected) and torch.allclose(probs[0, 1], expected)
    record("NaN/Inf policy logits fall back to uniform", ok)


def test_minco_smoke_select_prepare_project_commit():
    if not TORCH_CUDA:
        record("MINCO smoke prepares, projects, steps, and commits", True, "skipped: PyTorch CUDA unavailable")
        return
    from examples.minco_torch.env import MincoPointEnvTransition

    env = MincoPointEnvTransition(
        num_agents=2,
        piece_t=0.1,
        n_checkpoints=4,
        obstacle_vertices=[[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]],
        obstacle_target_projection=True,
        obstacle_projection_iters=4,
        obstacle_projection_topk=4,
        device="cuda",
    )
    root = env.initial_flat_state(torch.tensor([[-3.0, 0.0], [3.0, 0.0]], device="cuda")).view(1, -1)
    case = make_case(nodes=4, warps=1, state_dim=env.state_dim, num_agents=2)
    case["bridge"] = duct.DuctExpandBridge(1, 4, env.state_dim, 2, 1)
    case["bridge"].node_states[0, 0] = root[0]
    case["bridge"].node_action_targets.zero_()
    case["bridge"].node_action_probs[0, 0] = 1.0 / 16.0
    candidate_targets = torch.zeros((2, 16, 2, 2), dtype=torch.float32, device="cuda")
    slot = mark_expand_job(case, 0, 0, 0)
    d = prepare_case_for_commit(case)
    info = duct.run_duct_expand_stage2_minco(
        case["bridge"],
        env,
        candidate_targets,
        FixedPolicy([0.0] * 16).cuda(),
        FixedPolicy([0.0] * 16).cuda(),
        uniform_sample_prob=1.0,
    )
    launch_commit(d)
    cuda.synchronize()
    h = copy_back(d)
    projected_changed = bool(
        torch.max(torch.abs(info["projected_targets"] - info["full_targets"])).item() > 1e-5
    )
    ok = (
        info["num_jobs"] == 1
        and info["num_valid"] == 1
        and projected_changed
        and int(h["edge_child"][0, 0, slot]) == 1
        and int(h["node_expand_inflight"][0, 0]) == 0
        and torch.isfinite(case["bridge"].node_states[0, 1]).all().item()
    )
    record("MINCO smoke prepares, projects, steps, and commits", ok)


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
    test_fresh_expand_commits_child()
    test_multi_warp_expand_allocates_unique_children()
    test_non_expand_and_stale_jobs_are_ignored()
    test_failed_commit_discards_job_without_rollback()
    test_terminal_commit_publishes_terminal_sentinel()
    test_torch_stage1_allocates_ready_jobs_without_kernel()
    test_torch_stage1_drops_overflow_jobs_without_rollback()
    test_policy_sampling_prefers_policy_first_slot()
    test_uniform_sampling_uses_uniform_probabilities()
    test_nan_inf_policy_falls_back_to_uniform()
    test_minco_smoke_select_prepare_project_commit()
    summarize_results_and_exit()


if __name__ == "__main__":
    main()
