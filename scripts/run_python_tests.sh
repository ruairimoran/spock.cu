#!/usr/bin/env bash
# Run SPOCK's Python unit and reference tests without rebuilding CUDA code.
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

export PYTHONPATH="$repo_root/api/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$build_dir/_deps/gputils-src/python" ]]; then
    export PYTHONPATH="$build_dir/_deps/gputils-src/python:$PYTHONPATH"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

# Factory tests write fixture files relative to the working directory.
mkdir -p "$build_dir/tests/python-fixtures"
cd "$build_dir/tests/python-fixtures"
"$python_bin" -m unittest discover -s "$repo_root/tests/python" -p 'test*.py' -v
"$python_bin" -m unittest discover -s "$repo_root/examples/networkedControlMpc" -p 'test_*.py' -v
