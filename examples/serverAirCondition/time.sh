#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    for n in {4..16}; do
        python main.py --dt="d" --h="$n"
        julia ../../tests/julia/julia.jl
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S $path -B ./build -Wno-dev
            cmake --build ./build
            ./build/examples/serverAirCondition/server
        else
            printf "Skipping bad problem."
        fi
    done
}

main
