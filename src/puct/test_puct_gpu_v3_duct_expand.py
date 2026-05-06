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


def make_case(trees=1, nodes=8, warps=1, state_dim=8, num_agents=4, actions=16):
    edge_child = np.full((trees, nodes, duct.DUCT_JOINT_ACTIONS), duct.DUCT_EDGE_UNEXPANDED, np.int32)
    action_w = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.float32)
    action_n = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_inflight = np.zeros((trees, nodes, duct.DUCT_PLAYERS, actions), np.int32)
    action_counts = np.full((trees, nodes, duct.DUCT_PLAYERS), actions, np.int32)
    node_n = np.zeros((trees, nodes), np.int32)
    node_expand_inflight = np.zeros((trees, nodes), np.int32)
    tree_nodes = np.ones((trees,), np.int32)
    out_selected = np.full((trees, warps), v3.PACKED_INVALID, np.int32)
    out_path = np.full((trees, warps, 2), -1, np.int32)
    out_len = np.zeros((trees, warps), np.int32)
    bridge = None
    return {
        "edge_child": edge_child,
        "action_w": action_w,
        "action_n": action_n,
        "action_inflight": action_inflight,
        "action_counts": action_counts,
        "node_n": node_n,
        "node_expand_inflight": node_expand_inflight,
        "tree_nodes": tree_nodes,
        "out_selected": out_selected,
        "out_path": out_path,
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
        else:
            out[k] = v
    return out


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


def ensure_torch_bridge(case):
    if not TORCH_CUDA:
        raise RuntimeError("PyTorch CUDA is required for DUCT expand bridge tests")
    if not isinstance(case.get("bridge"), duct.DuctExpandBridge):
        case["bridge"] = duct.DuctExpandBridge(
            case["trees"],
            case["nodes"],
            case["state_dim"],
            case["num_agents"],
            case["warps"],
        )
    return case["bridge"]


def copy_stage1_to_case_bridge(case, torch_bridge):
    bridge = ensure_torch_bridge(case)
    bridge.expand_count.copy_(torch_bridge.expand_count)
    bridge.expand_tree.copy_(torch_bridge.expand_tree)
    bridge.expand_parent.copy_(torch_bridge.expand_parent)
    bridge.expand_slot.copy_(torch_bridge.expand_slot)
    bridge.expand_child.copy_(torch_bridge.expand_child)
    return bridge


def seed_root_slots(bridge):
    state = np.arange(bridge.state_dim, dtype=np.float32)
    if hasattr(bridge.node_states, "detach"):
        bridge.node_states[0, 0] = torch.as_tensor(state, device=bridge.device)
    else:
        bridge.node_states[0, 0] = state
    for p in range(duct.DUCT_PLAYERS):
        for a in range(duct.DUCT_MARGINAL_ACTIONS):
            bridge.node_action_probs[0, 0, p, a] = float(a + 1) / 100.0
            for ag in range(bridge.num_agents):
                bridge.node_action_targets[0, 0, p, a, ag, 0] = 1000.0 * p + 100.0 * a + 10.0 * ag
                bridge.node_action_targets[0, 0, p, a, ag, 1] = 1000.0 * p + 100.0 * a + 10.0 * ag + 1.0


def seed_child_payload(bridge, done: int = 0):
    ready_children = []
    if hasattr(bridge.expand_count, "detach"):
        job_count = int(bridge.expand_count.detach().cpu()[0].item())
    else:
        job_count = int(np.asarray(bridge.expand_count)[0])
    if job_count <= 0:
        job_count = bridge.max_warps
    for job in range(job_count):
        child = job + 1
        if hasattr(bridge, "expand_child"):
            try:
                staged_child = int(bridge.expand_child[job])
                if staged_child >= 0:
                    child = staged_child
            except TypeError:
                pass
        ready_children.append(child)
        next_state = np.arange(bridge.state_dim, dtype=np.float32) + 10.0 * (job + 1)
        if hasattr(bridge.expand_next_states, "detach"):
            next_state_t = torch.as_tensor(next_state, device=bridge.device)
            bridge.expand_next_states[job] = next_state_t
            bridge.node_states[0, child] = next_state_t
        else:
            bridge.expand_next_states[job] = next_state
            bridge.node_states[0, child] = next_state
        bridge.expand_done[job] = done
        for p in range(duct.DUCT_PLAYERS):
            for a in range(duct.DUCT_MARGINAL_ACTIONS):
                bridge.node_action_probs[0, child, p, a] = 0.01 * (a + 1 + p)
                bridge.node_action_targets[0, child, p, a] = (
                    float(100 * job + 10 * p + a)
                )
    return ready_children


def mark_expand_job(case, wid: int, action0: int, action1: int):
    slot = joint_slot(action0, action1)
    case["out_selected"][0, wid] = pack_expand(0, slot)
    case["out_path"][0, wid, 0] = slot
    case["out_len"][0, wid] = 1
    case["edge_child"][0, 0, slot] = duct.DUCT_EDGE_EXPANDING
    case["node_expand_inflight"][0, 0] += 1
    case["action_inflight"][0, 0, 0, action0] += 1
    case["action_inflight"][0, 0, 1, action1] += 1
    return slot


def torchify_case(case):
    out = {}
    for k, v in case.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.as_tensor(v, device="cuda")
        else:
            out[k] = v
    out["bridge"] = ensure_torch_bridge(case)
    return out


def launch_prepare_torch(case):
    return duct.prepare_duct_expand_stage1_torch(
        case["bridge"],
        case["edge_child"],
        case["tree_nodes"],
        case["out_selected"],
    )


def prepare_case_for_publish(case):
    torch_case = numpy_case_to_torch_stage1(case)
    launch_prepare_torch(torch_case)
    copy_stage1_to_case_bridge(case, torch_case["bridge"])
    tcase = torchify_case(case)
    tcase["tree_nodes"].copy_(torch_case["tree_nodes"])
    return tcase


def publish_seeded_jobs(case):
    return duct.publish_expand_edges(
        case["bridge"],
        case["edge_child"],
    )


def test_fresh_expand_publishes_child():
    case = make_case(nodes=6, warps=1, state_dim=8, num_agents=3)
    ensure_torch_bridge(case)
    seed_root_slots(case["bridge"])
    slot = mark_expand_job(case, 0, 2, 3)
    d = prepare_case_for_publish(case)
    seed_child_payload(d["bridge"])
    child = int(d["bridge"].expand_child[0].item())
    d["edge_child"][0, child, :] = duct.DUCT_EDGE_UNEXPANDED
    d["action_w"][0, child, :, :] = 0.0
    d["action_n"][0, child, :, :] = 0
    d["action_inflight"][0, child, :, :] = 0
    d["node_n"][0, child] = 0
    d["action_counts"][0, child, :] = duct.DUCT_MARGINAL_ACTIONS
    publish_seeded_jobs(d)
    torch.cuda.synchronize()
    ok = (
        int(d["edge_child"][0, 0, slot].item()) == 1
        and int(d["tree_nodes"][0].item()) == 2
        and tuple(int(x) for x in d["action_counts"][0, 1].detach().cpu().tolist()) == (16, 16)
        and int(d["action_inflight"][0, 0, 0, 2].item()) == 1
        and int(d["action_inflight"][0, 0, 1, 3].item()) == 1
        and torch.allclose(d["bridge"].node_states[0, 1], torch.arange(8, device="cuda", dtype=torch.float32) + 10.0)
        and float(d["bridge"].node_action_targets[0, 1, 1, 4, 0, 0].item()) == 14.0
    )
    record("fresh expand publishes child edge", ok)


def test_multi_warp_expand_allocates_unique_children():
    case = make_case(nodes=8, warps=4, state_dim=4, num_agents=2)
    ensure_torch_bridge(case)
    seed_root_slots(case["bridge"])
    slots = [mark_expand_job(case, wid, wid, (wid + 1) % 4) for wid in range(4)]
    d = prepare_case_for_publish(case)
    children = seed_child_payload(d["bridge"])
    for child in children:
        d["edge_child"][0, child, :] = duct.DUCT_EDGE_UNEXPANDED
        d["action_w"][0, child, :, :] = 0.0
        d["action_n"][0, child, :, :] = 0
        d["action_inflight"][0, child, :, :] = 0
        d["node_n"][0, child] = 0
        d["action_counts"][0, child, :] = duct.DUCT_MARGINAL_ACTIONS
    publish_seeded_jobs(d)
    torch.cuda.synchronize()
    edge_children = [int(d["edge_child"][0, 0, slot].item()) for slot in slots]
    ok = (
        sorted(edge_children) == [1, 2, 3, 4]
        and int(d["tree_nodes"][0].item()) == 5
        and len(set(edge_children)) == 4
    )
    record("multi-warp expand allocates unique child ids", ok, f"children={edge_children}")


def test_terminal_expand_publishes_terminal_sentinel():
    case = make_case(nodes=4, warps=1, state_dim=4, num_agents=2)
    ensure_torch_bridge(case)
    seed_root_slots(case["bridge"])
    slot = mark_expand_job(case, 0, 1, 2)
    d = prepare_case_for_publish(case)
    children = seed_child_payload(d["bridge"], done=1)
    for child in children:
        d["edge_child"][0, child, :] = duct.DUCT_EDGE_UNEXPANDED
        d["action_w"][0, child, :, :] = 0.0
        d["action_n"][0, child, :, :] = 0
        d["action_inflight"][0, child, :, :] = 0
        d["node_n"][0, child] = 0
        d["action_counts"][0, child, :] = 1
    publish_seeded_jobs(d)
    torch.cuda.synchronize()
    ok = (
        int(d["edge_child"][0, 0, slot].item()) == v3.NODE_EXPANDED_TERMINAL
        and tuple(int(x) for x in d["action_counts"][0, 1].detach().cpu().tolist()) == (1, 1)
    )
    record("terminal expand publishes NODE_EXPANDED_TERMINAL sentinel", ok)


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
    mark_torch_stage1_job(case, 1, 4, 5)
    mark_torch_stage1_job(case, 2, 6, 7)
    info = duct.prepare_duct_expand_stage1_torch(
        case["bridge"],
        case["edge_child"],
        case["tree_nodes"],
        case["out_selected"],
    )
    torch.cuda.synchronize()
    count = int(case["bridge"].expand_count[0].item())
    trees = case["bridge"].expand_tree[:count].cpu().numpy().tolist()
    children = case["bridge"].expand_child[:count].cpu().numpy().tolist()
    slots = case["bridge"].expand_slot[:count].cpu().numpy().tolist()
    ok = (
        info == {"num_jobs": 3, "num_ready": 3, "num_dropped": 0}
        and int(case["tree_nodes"][0].item()) == 4
        and count == 3
        and trees == [0, 0, 0]
        and children == [1, 2, 3]
        and slots == [joint_slot(2, 3), joint_slot(4, 5), joint_slot(6, 7)]
    )
    record("torch stage1 compacts READY jobs without CUDA prepare kernel", ok, f"children={children}")


class FixedGaussianPolicy(nn.Module):
    def __init__(self, mean, log_std=-12.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("log_std", torch.full_like(self.mean, float(log_std)))

    def forward(self, x):
        mean = self.mean.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        log_std = self.log_std.to(x.device).unsqueeze(0).expand_as(mean)
        return mean, log_std


def make_candidate_targets(num_agents=2):
    table = torch.zeros((2, 16, num_agents, 2), dtype=torch.float32, device="cuda")
    for p in range(2):
        for a in range(16):
            table[p, a, :, :] = float(100 * p + a)
    return table


def test_gaussian_policy_sampling_fills_continuous_slots():
    if not TORCH_CUDA:
        record("Gaussian policy sampling fills continuous target slots", True, "skipped: PyTorch CUDA unavailable")
        return
    bridge = duct.DuctExpandBridge(1, 2, 5, 2, 1)
    states = torch.zeros((1, 5), dtype=torch.float32, device="cuda")
    mean0 = torch.tensor([[-2.0, 1.0], [-1.0, 2.0]], dtype=torch.float32, device="cuda")
    mean1 = torch.tensor([[2.0, -1.0], [1.0, -2.0]], dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(123)
    targets, probs = duct.fill_duct_node_action_slots(
        bridge,
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
        states,
        make_candidate_targets(),
        FixedGaussianPolicy(mean0).cuda(),
        FixedGaussianPolicy(mean1).cuda(),
        uniform_sample_prob=0.0,
        generator=gen,
    )
    ok = (
        targets.shape == (1, 2, 16, 2, 2)
        and torch.allclose(targets[0, 0].mean(dim=0), mean0, atol=5e-2)
        and torch.allclose(targets[0, 1].mean(dim=0), mean1, atol=5e-2)
        and torch.all(probs[0, 0, :-1] >= probs[0, 0, 1:]).item()
        and torch.all(probs[0, 1, :-1] >= probs[0, 1, 1:]).item()
    )
    record("Gaussian policy sampling fills continuous target slots", ok)


def test_minco_smoke_select_prepare_project_publish():
    if not TORCH_CUDA:
        record("MINCO smoke prepares, projects, steps, and publishes", True, "skipped: PyTorch CUDA unavailable")
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
    d = prepare_case_for_publish(case)
    info = duct.run_duct_expand_stage2_minco(
        case["bridge"],
        env,
        candidate_targets,
        d["edge_child"],
        d["action_w"],
        d["action_n"],
        d["action_inflight"],
        d["action_counts"],
        d["node_n"],
        d["tree_nodes"],
        FixedGaussianPolicy(torch.zeros((2, 2), dtype=torch.float32, device="cuda")).cuda(),
        FixedGaussianPolicy(torch.zeros((2, 2), dtype=torch.float32, device="cuda")).cuda(),
        uniform_sample_prob=0.0,
    )
    cuda.synchronize()
    projected_changed = bool(
        torch.max(torch.abs(info["projected_targets"] - info["full_targets"])).item() > 1e-5
    )
    ok = (
        info["num_jobs"] == 1
        and info["num_valid"] == 1
        and info["num_published"] == 1
        and projected_changed
        and int(d["edge_child"][0, 0, slot].item()) == 1
        and torch.isfinite(case["bridge"].node_states[0, 1]).all().item()
    )
    record("MINCO smoke prepares, projects, steps, and publishes", ok)


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
    test_fresh_expand_publishes_child()
    test_multi_warp_expand_allocates_unique_children()
    test_terminal_expand_publishes_terminal_sentinel()
    test_torch_stage1_allocates_ready_jobs_without_kernel()
    test_gaussian_policy_sampling_fills_continuous_slots()
    test_minco_smoke_select_prepare_project_publish()
    summarize_results_and_exit()


if __name__ == "__main__":
    main()
