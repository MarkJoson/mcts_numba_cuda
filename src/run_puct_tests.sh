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
#   bash run_puct_tests.sh --v2             # puct_gpu_v2 kernel corner cases
#   bash run_puct_tests.sh --v2-stress      # puct_gpu_v2 stress matrix
#   bash run_puct_tests.sh --v2-bench       # puct_gpu_v2 select experiment metrics
#   bash run_puct_tests.sh --v2-scale-bench # puct_gpu_v2 4096..16384 tree deep-wide benchmark
#   bash run_puct_tests.sh --v2-cpu-ref     # puct_gpu_v2 CPU parallel reference comparison
#   bash run_puct_tests.sh --v2-cpu-bench   # puct_gpu_v2 CPU sequential vs GPU select benchmark
#   bash run_puct_tests.sh --v2-cpu-scale-bench # CPU sequential vs GPU 4096..16384 aligned subset
#   bash run_puct_tests.sh --v2-gpu-long-stress # GPU long-run scale stress + nvidia-smi telemetry
#   bash run_puct_tests.sh --v2-gpu-long-stress-smoke # short validation for long stress path
#   bash run_puct_tests.sh --all            # ALL suites in sequence
#
# Environment:
#   - Python   : set PUCT_TEST_PY to override; otherwise use py312_numba or PATH python
#   - LD_LIBRARY_PATH : includes torch/lib for libcuda sharing
#   - NUMBA_DISABLE_PERFORMANCE_WARNINGS=1
#   - NUMBA_CACHE_DIR : avoids repeated JIT recompilation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_PY="/home/robomaster/anaconda3/envs/py312_numba/bin/python"
if [ -n "${PUCT_TEST_PY:-}" ]; then
    VENV_PY="$PUCT_TEST_PY"
elif [ -x "$DEFAULT_PY" ]; then
    VENV_PY="$DEFAULT_PY"
else
    VENV_PY="$(command -v python)"
fi

# Use torch lib from the selected conda env when present so Numba shares libcuda.
PY_PREFIX="$("$VENV_PY" -c 'import sys; print(sys.prefix)')"
TORCH_LIB="$PY_PREFIX/lib/python$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/torch/lib"

if [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"
fi
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

elif [ "$1" = "--v2" ]; then
    echo "  Mode   : puct_gpu_v2 kernel corner cases"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py"

elif [ "$1" = "--v2-stress" ]; then
    echo "  Mode   : puct_gpu_v2 stress matrix"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --stress

elif [ "$1" = "--v2-bench" ]; then
    echo "  Mode   : puct_gpu_v2 select experiment metrics"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --bench

elif [ "$1" = "--v2-scale-bench" ]; then
    echo "  Mode   : puct_gpu_v2 large-scale deep-wide benchmark"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --scale-bench

elif [ "$1" = "--v2-cpu-ref" ]; then
    echo "  Mode   : puct_gpu_v2 CPU parallel reference comparison"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2_cpu_ref.py"

elif [ "$1" = "--v2-cpu-bench" ]; then
    echo "  Mode   : puct_gpu_v2 CPU sequential vs GPU select benchmark"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --cpu-bench

elif [ "$1" = "--v2-cpu-scale-bench" ]; then
    echo "  Mode   : puct_gpu_v2 CPU sequential vs GPU large-scale aligned benchmark"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --cpu-scale-bench

elif [ "$1" = "--v2-gpu-long-stress" ]; then
    echo "  Mode   : puct_gpu_v2 GPU long-run scale stress + nvidia-smi telemetry"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --gpu-long-stress

elif [ "$1" = "--v2-gpu-long-stress-smoke" ]; then
    echo "  Mode   : puct_gpu_v2 GPU long-run telemetry smoke"
    echo "================================================================"
    exec "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py" --gpu-long-stress-smoke

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
    echo "  [5/6] PUCT GPU v2 Kernel Tests"
    echo "════════════════════════════════════════════════════════════════"
    "$VENV_PY" "$SCRIPT_DIR/test_puct_gpu_v2.py"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [6/6] Performance Benchmarks & Stress Tests"
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
