#!/usr/bin/env bash
# Check SPOCK's CUDA tests for invalid accesses and unreleased device memory.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${SPOCK_BUILD_DIR:-$repo_root/build}"
case "$build_dir" in
    /*) ;;
    *) build_dir="$repo_root/$build_dir" ;;
esac

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$build_dir/venv/bin/python" ]]; then
    python_bin="$build_dir/venv/bin/python"
else
    python_bin=python3
fi
python_bin="$("$python_bin" -c 'import sys; print(sys.executable)')"

sanitizer="${SPOCK_COMPUTE_SANITIZER:-}"
if [[ -z "$sanitizer" ]]; then
    if command -v compute-sanitizer >/dev/null 2>&1; then
        sanitizer="$(command -v compute-sanitizer)"
    elif [[ -x /usr/local/cuda/bin/compute-sanitizer ]]; then
        sanitizer=/usr/local/cuda/bin/compute-sanitizer
    else
        echo "compute-sanitizer was not found; set SPOCK_COMPUTE_SANITIZER" >&2
        exit 2
    fi
fi
if [[ ! -x "$sanitizer" ]]; then
    echo "compute-sanitizer is not executable: $sanitizer" >&2
    exit 2
fi

report_dir="${SPOCK_SANITIZER_LOG_DIR:-$build_dir/sanitizer}"
case "$report_dir" in
    /*) ;;
    *) report_dir="$repo_root/$report_dir" ;;
esac
mkdir -p "$report_dir"

cmake -S "$repo_root" -B "$build_dir" \
    -DSPOCK_BUILD_TEST=ON -DPython3_EXECUTABLE="$python_bin"
cmake --build "$build_dir" --parallel "${SPOCK_BUILD_JOBS:-2}" \
    --target spock_tests check_anderson_qr check_qr_workspace
ctest --test-dir "$build_dir" --output-on-failure \
    -R '^spock_generate_fixtures$'

run_memcheck() {
    local name="$1"
    shift
    local log="$report_dir/$name.log"
    echo "Checking $name (log: $log)"
    if ! "$sanitizer" --tool memcheck --leak-check full \
        --error-exitcode 99 --check-exit-code yes "$@" >"$log" 2>&1; then
        tail -n 80 "$log" >&2
        echo "Compute Sanitizer failed: $name" >&2
        return 1
    fi
    # Treat a missing summary as incomplete instrumentation, even if the
    # program itself returned successfully.
    if ! grep -q 'ERROR SUMMARY: 0 errors' "$log" ||
       ! grep -q 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' "$log"; then
        tail -n 80 "$log" >&2
        echo "Compute Sanitizer did not report a clean run: $name" >&2
        return 1
    fi
    grep -E 'LEAK SUMMARY:|ERROR SUMMARY:' "$log"
}

# One process for all GoogleTests detects leaks at process exit and avoids
# starting a fresh CUDA context for every individual test.
run_memcheck spock_tests "$build_dir/tests/spock_tests"
# A second full pass removes legacy default-stream ordering from the solver.
SPOCK_TEST_NONBLOCKING_STREAM=1 run_memcheck spock_nonblocking_streams "$build_dir/tests/spock_tests"
run_memcheck qr_workspace "$build_dir/examples/networkedControlMpc/check_qr_workspace"
# More than 64 history changes also exercises the scheduled QR rebuild.
run_memcheck rolling_qr "$build_dir/examples/networkedControlMpc/check_anderson_qr" \
    --benchmark 256 72

echo "CUDA memory checks passed; reports are in $report_dir"
