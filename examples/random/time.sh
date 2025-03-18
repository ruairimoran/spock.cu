#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    start=100000
    width=10000
    n=1
    for interval in {0..80}; do
        lo=$((start + width * interval))
        hi=$((start + width * (interval + 1)))
        for i in $(seq 1 $n); do
            printf '\n\n---------------------- Problem #%s ----------------------\n\n' "$((i + n * interval))"
            python main.py --dt="d" --lo="$lo" --hi="$hi"
            julia ../../tests/julia/julia.jl
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                cmake -S $path -B ./build -Wno-dev
                cmake --build ./build
                ./build/examples/random/random
            else
                printf "Skipping bad problem."
            fi
        done
    done
}

main
