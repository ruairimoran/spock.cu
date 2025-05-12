#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    n=5
    for ch_half in $(seq 1 $n); do
        ch=$((ch_half * 2));
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
