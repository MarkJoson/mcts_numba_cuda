"""
example8_adapter.py
-------------------
Self-contained GPU PUCT adapter for the example8 pursuit-evasion problem.

This is a standalone port of ``decision_making/code/solvers/gpu_puct_adapter.py``
with ALL external dependencies removed. It only depends on:
  - mcts_numba_cuda/src/puct_gpu.py (PUCTGpu engine)
  - numpy, torch (standard)
"""

import torch
if torch.cuda.is_available():
    _torch_cuda_sentinel = torch.zeros(1, device="cuda")

# Patch Numba's CTK_SUPPORTED for CUDA toolkit >= 12.5
try:
    from numba.cuda.cudadrv import nvvm as _nvvm_mod
    if hasattr(_nvvm_mod, "CTK_SUPPORTED"):
        _ctk = _nvvm_mod.CTK_SUPPORTED
        for _v in [(12, 5), (12, 6), (12, 7), (12, 8), (12, 9)]:
            if _v not in _ctk:
                _ctk[_v] = ((5, 0), (9, 0))
except Exception:
    pass

import numpy as np
import torch.nn as nn
from puct.puct_gpu import PUCTGpu


# ════════════════════════════════════════════════════════════════════════════
# Neural-network helpers
# ════════════════════════════════════════════════════════════════════════════

class UniformPriorNet(nn.Module):
    """
    Returns uniform logits so softmax → 1/max_actions for every slot.
    """

    def __init__(self, state_dim: int, max_actions: int):
        super().__init__()
        self.max_actions = max_actions
        # Dummy parameter so CUDA .to() / .cuda() works correctly
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.ones(B, self.max_actions, device=x.device)


class ValueBridgeWrapper(nn.Module):
    """
    Wraps the original value oracle for the ``NumbaPytorchBridge``.

    Original signature:
        ``oracle.eval(problem, state_col)  →  np.ndarray (num_robots, 1)``

    Bridge expects:
        ``model(batch_cuda_tensor)  →  torch.Tensor (B, num_robots)``
    """

    def __init__(self, value_oracle, problem):
        super().__init__()
        self.oracle = value_oracle
        self.problem = problem
        self._dummy = nn.Linear(problem.state_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        R = self.problem.num_robots
        if self.oracle is None:
            return torch.zeros(B, R, device=x.device)
        results = []
        for i in range(B):
            state_col = x[i].cpu().numpy().reshape(-1, 1)
            val = self.oracle.eval(self.problem, state_col)
            results.append(
                torch.tensor(val.flatten(), dtype=torch.float32)
            )
        return torch.stack(results, dim=0).to(x.device)


# ════════════════════════════════════════════════════════════════════════════
# GPU_PUCT_Adapter
# ════════════════════════════════════════════════════════════════════════════

class GPU_PUCT_Adapter:
    """
    Drop-in solver backed by GPU PUCT.

    Parameters
    ----------
    policy_oracle : list of oracle or [None]
    value_oracle : oracle or None
    search_depth, number_simulations, C_pw, alpha_pw, C_exp, alpha_exp,
    beta_policy, beta_value, vis_on : solver params (same as PUCT_V1)
    max_actions, n_trees, device_memory : GPU-specific params
    """

    def __init__(
        self,
        policy_oracle=None,
        value_oracle=None,
        search_depth=10,
        number_simulations=1000,
        C_pw=2.0,
        alpha_pw=0.5,
        C_exp=1.0,
        alpha_exp=0.25,
        beta_policy=0.0,
        beta_value=0.0,
        vis_on=False,
        max_actions=32,
        n_trees=8,
        device_memory=1.0,
    ):
        self.policy_oracle = policy_oracle if policy_oracle is not None else [None]
        self.value_oracle = value_oracle
        self.search_depth = search_depth
        self.number_simulations = number_simulations
        self.C_pw = C_pw
        self.alpha_pw = alpha_pw
        self.C_exp = C_exp
        self.alpha_exp = alpha_exp
        self.beta_policy = beta_policy
        self.beta_value = beta_value
        self.vis_on = vis_on
        self.max_actions = max_actions
        self.n_trees = n_trees
        self.device_memory = device_memory
        self.solver_name = "GPU_PUCT"

        self._agent = None

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _ensure_agent(self, problem):
        """Create and initialise PUCTGpu on first call (needs problem dims)."""
        if self._agent is not None:
            return
        self._agent = PUCTGpu(
            state_dim=problem.state_dim,
            action_dim=problem.action_dim,
            max_actions=self.max_actions,
            num_robots=problem.num_robots,
            n_trees=self.n_trees,
            max_simulations=self.number_simulations,
            C_exp=self.C_exp,
            alpha_exp=self.alpha_exp,
            C_pw=self.C_pw,
            alpha_pw=self.alpha_pw,
            gamma=problem.gamma,
            device_memory=self.device_memory,
            verbose_info=False,
        )
        self._agent.init_device_side_arrays()

    # ── Solver interface ─────────────────────────────────────────────────────
    # !
    def multiturn_policy(self, problem, root_state):
        """
        Compute a joint action for all robots.

        Supports two modes:
        1. ``turn_groups`` (e.g. example8): search once per group, assign
           the result to all robots in the group.  Matches C_PUCT behavior.
        2. Per-robot search (default): one PUCT search per robot.
        """
        

        self._ensure_agent(problem)         #! 惰性初始化 保证已经实例化 GpuPUCT求解器

        if problem.is_terminal(root_state):
            return np.zeros((problem.action_dim, 1))

        policy_net = UniformPriorNet(
            problem.state_dim, self.max_actions
        ).cuda()
        value_net = ValueBridgeWrapper(self.value_oracle, problem).cuda()

        action = np.zeros((problem.action_dim, 1))

        #? Multiturn group 配置，分为追捕者组和逃跑者组
        groups = getattr(problem, 'turn_groups', None)
        if groups is not None:                 
            search_units = [
                (int(group[0]), list(group))
                for group in groups
            ]
        else:
            search_units = [
                (robot, [robot])
                for robot in range(problem.num_robots)
            ]

        #! 轮流调用不同 group 的 robot, 双层循环，外层循环对应 group, 内层循环对应 robot
        for turn, robot_list in search_units:
            # cache 用于存储浮点数动作向量
            # TODO. 把这个root_action_cache搬到GPU上
            root_action_cache = {}
            representative_robot = robot_list[0]
            robot_action_idxs = problem.action_idxs[representative_robot]
            
            # 动作生成回调函数，用于提供给MCTS，生成若干个候选动作
            def _make_sampler(robot_idx, oracle_list, beta_pol):
                def action_sampler(parent_state_flat, action_idx, tree_idx):
                    if (not all(x is None for x in oracle_list)         # 如果oracle网络已经加载，就调用 oracle 网络（仅 robot_idx）
                            and np.random.uniform() < beta_pol          # 其他机器人使用随机动作
                            and oracle_list[robot_idx] is not None):
                        state_col = parent_state_flat.reshape(-1, 1)
                        a = oracle_list[robot_idx].eval(
                            problem, state_col, robot_idx
                        )
                        full_action = problem.sample_action()
                        full_action[robot_action_idxs, 0] = a.flatten()
                        return full_action
                    else:
                        return problem.sample_action()
                return action_sampler

            sampler = _make_sampler(
                representative_robot, self.policy_oracle, self.beta_policy
            )
            
            # TODO. 在搜索时也没有做死亡屏蔽？
            best_idx, best_n, info = self._agent.run(
                root_state=root_state.flatten().astype(np.float32),
                root_turn=turn,
                policy_model=policy_net,
                value_model=value_net,
                problem=problem,
                action_sampler=sampler,
                root_action_cache=root_action_cache,
            )

            for robot in robot_list:
                r_action_idxs = problem.action_idxs[robot]
                if best_idx in root_action_cache:                           # 优先级 1：应用 MCTS 的搜索结果
                    action[r_action_idxs, 0] = (
                        root_action_cache[best_idx][r_action_idxs, 0]
                    )
                elif (not all(x is None for x in self.policy_oracle)        # 优先级 2：回退至专家策略
                        and self.policy_oracle[robot] is not None):
                    a_fallback = self.policy_oracle[robot].eval(
                        problem, root_state, robot
                    )
                    action[r_action_idxs, 0] = a_fallback.flatten()
                else:
                    a_fallback = problem.sample_action()                    # 优先级 3：回退至随机采样
                    action[r_action_idxs, 0] = a_fallback[r_action_idxs, 0]

        # 对于死亡的机器人，将动作屏蔽
        if hasattr(problem, 'is_active'):
            for robot in range(problem.num_robots):
                if not problem.is_active(root_state, robot):
                    action[problem.action_idxs[robot], 0] = 0.0

        return action

    def __str__(self):
        return (
            f"GPU_PUCT_Adapter("
            f"max_actions={self.max_actions}, n_trees={self.n_trees}, "
            f"max_simulations={self.number_simulations}, "
            f"C_exp={self.C_exp}, alpha_exp={self.alpha_exp}, "
            f"C_pw={self.C_pw}, alpha_pw={self.alpha_pw})"
        )

    def __repr__(self):
        return str(self)
