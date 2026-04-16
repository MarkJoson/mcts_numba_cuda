"""
numba_torch_compat.py
---------------------
Compatibility shim for Numba CUDA 0.61 + CUDA 12.9 on RTX 4090 (sm_89).

Import FIRST, before any numba or torch import:

    import numba_torch_compat
    from numba import cuda
    import torch

Problems fixed
--------------
1. ``_ActiveContext.devnum`` returns a garbage pointer (not a device ordinal)
   under CUDA 12.9 due to a Numba ctypes API mismatch.  Fixed by patching
   ``_ActiveContext.__bool__ = lambda self: False`` so Numba always takes
   the safe ``_activate_context_for(0)`` path.

2. ``find_closest_arch`` raises ``NvvmSupportError("No supported GPU compute
   capabilities found")`` because ``CTK_SUPPORTED`` in nvvm.py does not
   include CUDA 12.5-12.9.  Fixed by patching both ``CTK_SUPPORTED`` and by
   resetting the ``NVVM`` singleton's ``_supported_ccs`` after the patch.

3. Zero-copy memory sharing between Numba and PyTorch requires both to use
   the same CUDA device, which the above two fixes enable.
"""

import math
import logging
_log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Patch 1 — ctypes active-context devnum bug
# ════════════════════════════════════════════════════════════════════════════
try:
    from numba.cuda.cudadrv.driver import _ActiveContext

    def _always_false(self):
        """
        Force 'no active context' so _get_or_create_context_uncached always
        calls _activate_context_for(0) instead of trying to look up
        ac.devnum which can be a garbage pointer value under CUDA 12.9.
        """
        return False

    _ActiveContext.__bool__ = _always_false
    _log.debug("numba_torch_compat: _ActiveContext.__bool__ patched")
except Exception as _e:
    _log.warning("numba_torch_compat: _ActiveContext patch failed: %s", _e)

# ════════════════════════════════════════════════════════════════════════════
# Patch 2 — CTK_SUPPORTED + NVVM singleton._supported_ccs
# ════════════════════════════════════════════════════════════════════════════
_EXTRA_CUDA_VERSIONS = [(12, 5), (12, 6), (12, 7), (12, 8), (12, 9)]
_SM89_RANGE = ((5, 0), (9, 0))

try:
    from numba.cuda.cudadrv import nvvm as _nvvm_mod
    from numba.cuda.cudadrv.nvvm import (
        CTK_SUPPORTED, COMPUTE_CAPABILITIES, NvvmSupportError
    )

    # 2a. Extend CTK_SUPPORTED dict in-place
    for _v in _EXTRA_CUDA_VERSIONS:
        if _v not in CTK_SUPPORTED:
            CTK_SUPPORTED[_v] = _SM89_RANGE

    # 2b. Reset singleton's cached _supported_ccs if it was already created
    try:
        nvvm_inst = _nvvm_mod.NVVM()   # returns the singleton
        # Re-compute with the updated table
        nvvm_inst._supported_ccs = _nvvm_mod.get_supported_ccs()
        if not nvvm_inst._supported_ccs:
            # Fallback: include all COMPUTE_CAPABILITIES from sm_50 up
            nvvm_inst._supported_ccs = tuple(
                cc for cc in COMPUTE_CAPABILITIES if cc >= (5, 0)
            )
        _log.debug(
            "numba_torch_compat: NVVM supported_ccs refreshed: %s",
            nvvm_inst._supported_ccs
        )
    except Exception as _e2:
        _log.warning("numba_torch_compat: NVVM singleton refresh failed: %s", _e2)

    # 2c. Patch find_closest_arch to use a fallback if supported_ccs still empty
    _orig_find_closest = _nvvm_mod.find_closest_arch

    def _robust_find_closest_arch(mycc):
        try:
            return _orig_find_closest(mycc)
        except NvvmSupportError:
            # Fallback: pick the highest CC ≤ mycc from COMPUTE_CAPABILITIES
            candidates = [cc for cc in COMPUTE_CAPABILITIES if cc <= mycc]
            if candidates:
                return candidates[-1]
            return COMPUTE_CAPABILITIES[-1]

    _nvvm_mod.find_closest_arch = _robust_find_closest_arch
    _log.debug("numba_torch_compat: find_closest_arch patched with fallback")

except Exception as _e:
    _log.warning("numba_torch_compat: NVVM patches failed: %s", _e)

# ════════════════════════════════════════════════════════════════════════════
# Verify by importing numba.cuda
# ════════════════════════════════════════════════════════════════════════════
from numba import cuda as _numba_cuda   # noqa: F401 — side-effect import

_log.debug("numba_torch_compat: setup complete")
