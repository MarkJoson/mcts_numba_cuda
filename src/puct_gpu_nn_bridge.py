"""
puct_gpu_nn_bridge.py
---------------------
Zero-copy shared GPU buffers between Numba CUDA and PyTorch.

The ``NumbaPytorchBridge`` class owns PyTorch tensors and exposes Numba
``DeviceNDArray`` views of the *same* GPU memory. This avoids any PCIe
round-trip when leaf states must be sent to the neural network and the
resulting priors / values must be written back into the PUCT data structure.

Notes
-----
- PyTorch tensors **must** remain alive (as class attributes) for the entire
  lifetime of the Numba views.  Letting them be garbage-collected while the
  Numba wrappers still exist is undefined behaviour.
- Tensors must be **contiguous** before wrapping.  Use ``tensor.contiguous()``
  if in doubt.
"""

import torch
import numpy as np

# IMPORTANT: PyTorch must initialise the CUDA primary context BEFORE Numba
# imports its runtime.  Failure to do so causes a segfault on some driver
# versions because both libraries attempt to create incompatible contexts.
if torch.cuda.is_available():
    _torch_init_sentinel = torch.zeros(1, device="cuda")

from numba import cuda


class NumbaPytorchBridge:
    """
    Zero-copy shared GPU buffers between Numba CUDA and PyTorch.

    Parameters
    ----------
    n_trees : int
        Number of independent PUCT trees (= batch size for NN calls).
    state_dim : int
        Dimensionality of the continuous state vector.
    action_dim : int
        Dimensionality of the continuous action vector (stored per node).
    max_actions : int
        Maximum number of discrete actions / prior slots per node.
    num_robots : int
        Number of agents (robots); determines width of the value vector.
    """

    def __init__(self, n_trees: int, state_dim: int, action_dim: int,
                 max_actions: int, num_robots: int):
        self.n_trees = n_trees
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_actions = max_actions
        self.num_robots = num_robots

        # ── PyTorch owns the GPU memory ──────────────────────────────────────
        # leaf_states  : states extracted by _extract_leaf_states kernel, fed
        #                directly into the policy / value network.
        self.leaf_states: torch.Tensor = torch.zeros(
            (n_trees, state_dim), dtype=torch.float32, device="cuda"
        )
        # nn_priors    : policy-network output written back here; read by
        #                _expand_and_backup_puct kernel.
        self.nn_priors: torch.Tensor = torch.zeros(
            (n_trees, max_actions), dtype=torch.float32, device="cuda"
        )
        # nn_values    : value-network output written back here; read by
        #                _expand_and_backup_puct kernel.
        self.nn_values: torch.Tensor = torch.zeros(
            (n_trees, num_robots), dtype=torch.float32, device="cuda"
        )
        # leaf_valid   : 1 if this tree reached a non-terminal leaf that
        #                needs NN evaluation, 0 otherwise.
        self.leaf_valid: torch.Tensor = torch.zeros(
            (n_trees,), dtype=torch.int32, device="cuda"
        )

        # ── Numba views of the *same* memory (zero-copy) ────────────────────
        self.dev_leaf_states = cuda.as_cuda_array(self.leaf_states)
        self.dev_nn_priors = cuda.as_cuda_array(self.nn_priors)
        self.dev_nn_values = cuda.as_cuda_array(self.nn_values)
        self.dev_leaf_valid = cuda.as_cuda_array(self.leaf_valid)

    # ── Diagnostic helpers ───────────────────────────────────────────────────

    def verify_zero_copy(self) -> bool:
        """
        Quick sanity check: write a sentinel via PyTorch, read it via Numba.

        Returns
        -------
        bool
            ``True`` if the memory is truly shared.
        """
        sentinel = 42.0
        self.leaf_states[0, 0] = sentinel
        host_val = float(self.dev_leaf_states[0, 0])
        ok = abs(host_val - sentinel) < 1e-5
        self.leaf_states[0, 0] = 0.0
        return ok

    def reset(self):
        """Zero-fill all bridge tensors (useful between runs)."""
        self.leaf_states.zero_()
        self.nn_priors.zero_()
        self.nn_values.zero_()
        self.leaf_valid.zero_()

    def __repr__(self) -> str:
        return (
            f"NumbaPytorchBridge("
            f"n_trees={self.n_trees}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, max_actions={self.max_actions}, "
            f"num_robots={self.num_robots})"
        )
