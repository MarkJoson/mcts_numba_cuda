#!/usr/bin/env bash
# run_puct_tests.sh — Run PUCT GPU tests in the correct environment
#
# Usage:
#   bash run_puct_tests.sh                  # unit tests (Tests 1-10)
#   bash run_puct_tests.sh --examples       # example-style tests (Tests 11-23)
#   bash run_puct_tests.sh --multi          # multi-dimensional stress tests (72 configs)
#   bash run_puct_tests.sh --diversity      # input diversity + invariant tests (Groups A-R)
#   bash run_puct_tests.sh --diversity --quick   # quick subset of diversity groups
#   bash run_puct_tests.sh --perf           # performance benchmarks + stress tests
#   bash run_puct_tests.sh --perf --bench-only   # benchmarks only (no stress)
#   bash run_puct_tests.sh --perf --stress-only  # stress only (no benchmarks)
#   bash run_puct_tests.sh --all            # ALL suites in sequence
#
# Environment:
#   - Python   : py311_numba conda env (Numba 0.65 + torch 2.5.1 + CUDA 12.4)
#   - LD_LIBRARY_PATH : includes torch/lib for libcuda sharing
#   - NUMBA_DISABLE_PERFORMANCE_WARNINGS=1
#   - NUMBA_CACHE_DIR : avoids repeated JIT recompilation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="/home/robomaster/anaconda3/envs/py311_numba/bin/python"

# Use torch lib from py311_numba env so Numba shares the same libcuda
TORCH_LIB="/home/robomaster/anaconda3/envs/py311_numba/lib/python3.11/site-packages/torch/lib"

export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
export NUMBA_DISABLE_PERFORMANCE_WARNINGS=1
export NUMBA_CACHE_DIR="$PROJECT_ROOT/.numba_cache"

mkdir -p "$NUMBA_CACHE_DIR"

echo "================================================================"
echo "  PUCT GPU — Test Runner"
echo "================================================================"
echo "  Python : $VENV_PY"
echo "  Torch  : $TORCH_LIB"
echo "  Cache  : $NUMBA_CACHE_DIR"
echo "================================================================"

# ── Mode dispatch ─────────────────────────────────────────────────────────

if [ "$1" = "--examples" ]; then
    echo "  Mode   : example-style integration tests (Tests 11-23)"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_examples.py"

elif [ "$1" = "--multi" ]; then
    echo "  Mode   : multi-dimensional stress tests (72 configs)"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_multi.py"

elif [ "$1" = "--diversity" ]; then
    echo "  Mode   : input diversity tests (Groups A-R)"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_diversity.py" "${@:2}"

elif [ "$1" = "--perf" ]; then
    # Detect sub-flags passed after --perf
    PERF_ARGS="${@:2}"
    if echo "$PERF_ARGS" | grep -q "\-\-bench-only"; then
        echo "  Mode   : benchmarks only (isolated process)"
        echo "================================================================"
        exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_perf.py" --bench-only
    elif echo "$PERF_ARGS" | grep -q "\-\-stress-only"; then
        echo "  Mode   : stress tests only (isolated process)"
        echo "================================================================"
        exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_perf.py" --stress-only
    else
        echo "  Mode   : benchmarks + stress tests (two isolated processes)"
        echo "================================================================"
        echo ""
        echo "  ── [1/2] Performance Benchmarks ──────────────────────────────"
        "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_perf.py" --bench-only
        BENCH_EC=$?

        echo ""
        echo "  ── [2/2] Stress Tests (fresh CUDA context) ───────────────────"
        "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_perf.py" --stress-only
        STRESS_EC=$?

        echo ""
        if [ $BENCH_EC -eq 0 ] && [ $STRESS_EC -eq 0 ]; then
            echo "  [DONE] Both bench and stress PASSED ✓"
        else
            echo "  [DONE] bench_exit=$BENCH_EC  stress_exit=$STRESS_EC"
            exit 1
        fi
    fi

elif [ "$1" = "--all" ]; then
    echo "  Mode   : ALL test suites"
    echo "================================================================"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [1/5] Unit Tests (Tests 1-10)"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [2/5] Example Tests (Tests 11-23)"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_examples.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [3/5] Multi-dimensional Stress Tests (72 configs)"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_multi.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [4/5] Input Diversity Tests (Groups A-R)"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_diversity.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [5/5] Performance Benchmarks & Stress Tests"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_perf.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ALL SUITES COMPLETE"
    echo "════════════════════════════════════════════════════════════════"

else
    echo "  Mode   : unit tests (Tests 1-10)"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu.py" "$@"
fi
