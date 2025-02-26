#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    for i in {1..100}; do
        printf '\n\n---------------------- Problem #%s ----------------------\n\n' "$i"
        python main.py --dt="d"
        julia ../../tests/julia/test.jl
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S $path -B ./build -Wno-dev
            cmake --build ./build
            ./build/examples/random/random
        else
            printf "Skipping bad problem."
        fi
    done
}

main
