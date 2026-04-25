#!/usr/bin/env bash
# run_decision_making_gpu.sh
# ---------------------------
# Run the GPU_PUCT_Adapter example runner against decision_making/code examples.
#
# Usage:
#   # Generic runner (any example):
#   bash run_decision_making_gpu.sh                          # example1, smoke test
#   bash run_decision_making_gpu.sh --example 8              # example8, quick test
#   bash run_decision_making_gpu.sh --example 8 --steps 40   # full episode
#
#   # Full example8 pipeline (with plots, movie, diagnostics):
#   bash run_decision_making_gpu.sh example8                 # run_example8.py
#   bash run_decision_making_gpu.sh example8 --movie --trials 5
#
# Environment:
#   Python   : py311_numba conda env (Numba 0.65 + torch 2.5.1 + CUDA 12.4)
#   Must run from: /home/robomaster/Research/mcts_numba_cuda/src
#   (so that puct_gpu.py, puct_gpu_kernels.py etc. are importable by name)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCTS_SRC="${SCRIPT_DIR}/src"
DECISION_CODE="/home/robomaster/Research/decision_making/code"

VENV_PY="/home/robomaster/anaconda3/envs/py311_numba/bin/python"
TORCH_LIB="/home/robomaster/anaconda3/envs/py311_numba/lib/python3.11/site-packages/torch/lib"

export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
export NUMBA_DISABLE_PERFORMANCE_WARNINGS=1
export NUMBA_CACHE_DIR="${SCRIPT_DIR}/.numba_cache"

echo ""
echo "========================================================"
echo "  GPU PUCT — decision_making/code example runner"
echo "========================================================"
echo "  Python  : ${VENV_PY}"
echo "  MCTS src: ${MCTS_SRC}"
echo "  DM code : ${DECISION_CODE}"
echo "  Args    : $*"
echo "========================================================"
echo ""

cd "${MCTS_SRC}"

# If first argument is "example8", use the dedicated full-pipeline runner
if [ "${1:-}" = "example8" ]; then
    shift
    exec "${VENV_PY}" "${MCTS_SRC}/run_example8.py" "$@"
else
    exec "${VENV_PY}" "${DECISION_CODE}/run_gpu_example.py" "$@"
fi
