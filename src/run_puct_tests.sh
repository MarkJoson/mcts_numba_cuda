#!/usr/bin/env bash
# run_puct_tests.sh — Run PUCT GPU tests in the correct environment
#
# Usage:
#   bash run_puct_tests.sh [--verbose]
#
# Sets:
#   - LD_LIBRARY_PATH to include libnvvm.so from the uv venv
#   - NUMBA_DISABLE_PERFORMANCE_WARNINGS=1 (suppress low-occupancy warnings)
#   - NUMBA_CACHE_DIR to avoid recompilation costs
#   - Python from the project's uv venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"

# Find libnvvm.so in the uv venv nvidia packages
NVVM_LIB=$(find "$PROJECT_ROOT/.venv" -name "libnvvm.so" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)

if [ -z "$NVVM_LIB" ]; then
    echo "[run_puct_tests.sh] WARNING: libnvvm.so not found in .venv"
    echo "  Install with: cd $PROJECT_ROOT && uv pip install nvidia-cuda-nvcc-cu12"
fi

# Also look for libcudart.so in torch's libs (needed for Numba's runtime)
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
CUDART_LIB="$TORCH_LIB/lib"

export LD_LIBRARY_PATH="${NVVM_LIB}:${CUDART_LIB}:${LD_LIBRARY_PATH}"
export NUMBA_DISABLE_PERFORMANCE_WARNINGS=1
export NUMBA_CACHE_DIR="$PROJECT_ROOT/.numba_cache"

mkdir -p "$NUMBA_CACHE_DIR"

echo "================================================================"
echo "  PUCT GPU — Test Runner"
echo "================================================================"
echo "  Python : $VENV_PY"
echo "  NVVM   : ${NVVM_LIB:-'(system)'}"
echo "  TORCH  : $CUDART_LIB"
echo "================================================================"

exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu.py" "$@"
