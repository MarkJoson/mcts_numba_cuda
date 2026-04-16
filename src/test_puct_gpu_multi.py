"""
test_puct_gpu_multi.py
----------------------
Multi-dimensional stress test and benchmarking for the GPU-accelerated PUCT implementation.

Varies key parameters systematically to ensure correct behavior, memory stability,
and performance tracking across different scaling requirements:
    - n_trees: Number of parallel trees
    - max_simulations: Search depth / simulations per step
    - max_actions: State transition branching factor
    - state_dim: Dimensionality of the game state
    - num_robots: Number of players/agents

Usage:
    python test_puct_gpu_multi.py
"""

import sys
import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from numba import cuda

# Ensure local path is accessible
sys.path.insert(0, os.path.dirname(__file__))

# Import the compatibility shim if present
try:
    import numba_torch_compat
except ImportError:
    pass

from puct_gpu import PUCTGpu

class UniformPolicyNet(nn.Module):
    """Fallback uniform policy net for benchmark."""
    def __init__(self, state_dim, max_actions):
        super().__init__()
        self.max_actions = max_actions
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        return torch.ones(B, self.max_actions, device=x.device) / self.max_actions

class ConstantValueNet(nn.Module):
    """Fallback constant value net for benchmark."""
    def __init__(self, state_dim, num_robots, constant=0.5):
        super().__init__()
        self.num_robots = num_robots
        self.constant = constant
        self._dummy = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        return torch.full((B, self.num_robots), self.constant, dtype=torch.float32, device=x.device)

def run_multi_dimensional_test():
    # Define test grid
    n_trees_list = [1, 16, 128]
    max_simulations_list = [10, 50, 100]
    max_actions_list = [8, 32]
    state_dim_list = [4, 64]
    num_robots_list = [1, 2]

    combinations = list(itertools.product(
        n_trees_list, max_simulations_list, max_actions_list, state_dim_list, num_robots_list
    ))
    
    total = len(combinations)
    print(f"Starting Multi-Dimensional Stress Test over {total} configurations")
    print("=" * 85)
    print(f"{'Trees':>6} | {'Sims':>5} | {'Acts':>4} | {'Dim':>4} | {'Robots':>6} | {'Time (s)':>8} | Result")
    print("-" * 85)

    passed = 0
    failed = 0

    for idx, config in enumerate(combinations):
        n_trees, max_sims, max_acts, state_dim, num_robots = config
        
        # Instantiate networks
        policy = UniformPolicyNet(state_dim, max_acts).cuda()
        value = ConstantValueNet(state_dim, num_robots, 0.5).cuda()
        
        try:
            agent = PUCTGpu(
                state_dim=state_dim,
                action_dim=1,
                max_actions=max_acts,
                num_robots=num_robots,
                n_trees=n_trees,
                max_simulations=max_sims,
                C_pw=1.0,
                alpha_pw=0.5,
                gamma=0.99,
                device_memory=0.05,  # 5% per instantiation to avoid sum OOM
                verbose_info=False
            )
            agent.init_device_side_arrays()
            
            root_state = np.zeros(state_dim, dtype=np.float32)
            
            # Warm up & sync
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Run PUCT iteration
            best_action, best_n, action_infos = agent.run(root_state, root_turn=0, policy_model=policy, value_model=value, problem=None)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Validate output ranges
            if 0 <= best_action < max_acts and best_n > 0:
                print(f"{n_trees:6d} | {max_sims:5d} | {max_acts:4d} | {state_dim:4d} | {num_robots:6d} | {elapsed:8.4f} | PASS")
                passed += 1
            else:
                print(f"{n_trees:6d} | {max_sims:5d} | {max_acts:4d} | {state_dim:4d} | {num_robots:6d} | {elapsed:8.4f} | FAIL (Invalid return)")
                failed += 1
                
        except Exception as e:
            print(f"{n_trees:6d} | {max_sims:5d} | {max_acts:4d} | {state_dim:4d} | {num_robots:6d} | {'-':>8} | FAIL: {e}")
            failed += 1
            break # Break completely to avoid cascading CUDA errors or tracebacks
            
    print("=" * 85)
    print(f"Summary: {passed} passed, {failed} failed out of {idx + 1} run(s).")
    return failed == 0

if __name__ == '__main__':
    import warnings
    # Filter NumbaPerformanceWarning spam during loops
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
    
    ok = run_multi_dimensional_test()
    sys.exit(0 if ok else 1)
