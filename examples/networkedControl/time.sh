#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    n=10
    ch=2
    branching=0
    for ch in $(seq 2 $n); do
        for branching in {0..1}; do
            python main.py --dt="d" --br="$branching" --ch="$ch" --tree=1
            julia ../../tests/julia/julia.jl |& tee log/julia.txt
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                cmake -S $path -B ./build -Wno-dev
                cmake --build ./build
                ./build/examples/networkedControl/networked |& tee log/spock.txt
            else
                printf "Julia error!"
            fi
        done
    done
}

main
