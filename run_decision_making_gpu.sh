#!/usr/bin/env bash
# run_decision_making_gpu.sh
# ---------------------------
# Run the GPU_PUCT_Adapter example runner against decision_making/code examples.
#
# Usage:
#   bash run_decision_making_gpu.sh                          # example1, smoke test
#   bash run_decision_making_gpu.sh --example 8              # example8, 2-robot pursuit-evasion
#   bash run_decision_making_gpu.sh --example 8 --steps 40   # full episode
#   bash run_decision_making_gpu.sh --example 8 \
#       --model-dir /path/to/saved/example8                  # with pretrained oracles
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
exec "${VENV_PY}" "${DECISION_CODE}/run_gpu_example.py" "$@"
