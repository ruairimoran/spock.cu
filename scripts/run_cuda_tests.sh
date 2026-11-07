#!/usr/bin/env bash
# Build and run every CUDA-backed test registered by SPOCK's CTest project.
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

# CUDA fixtures and GPU adapter checks use the same Python environment as
# the compiled tests. CTest includes the fixture setup automatically.
cmake -S "$repo_root" -B "$build_dir" \
    -DSPOCK_BUILD_TEST=ON -DPython3_EXECUTABLE="$python_bin"
cmake --build "$build_dir" --parallel "${SPOCK_BUILD_JOBS:-2}"

# The two excluded entries are pure Python suites. GPU tests implemented in
# Python remain here because CTest runs them with the CUDA executables.
ctest --test-dir "$build_dir" --output-on-failure \
    -E '^(spock_python|spock_worker_responses)$'
