#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    python main.py --dt="d"
    cmake -S $path -B ./build -Wno-dev
    cmake --build ./build
    ./build/examples/precondition/precondition
}

main
