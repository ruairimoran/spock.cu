#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
#    mkdir -p ./build
#    python main.py --dt="d"
    julia ../../tests/julia/julia.jl
#    exit_code=$?
#    if [ $exit_code -eq 0 ]; then
#        cmake -S $path -B ./build -Wno-dev
#        cmake --build ./build
#        ./build/examples/random/random
#    else
#        printf "Julia error!"
#    fi
}

main
