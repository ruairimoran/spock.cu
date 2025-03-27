#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build

    n=10
    start=10000
    width=10000

    for i in $(seq 1 $n); do
        printf '\n\n---------------------- Problem #%s ----------------------\n\n' "$i"
        python main.py --dt="d" --lo=1000 --hi="$start"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S $path -B ./build -Wno-dev
            cmake --build ./build
            ./build/examples/andersonBuffer/buffer
        else
            printf "Skipping bad problem."
        fi
    done

    for interval in {0..8}; do
        lo=$((start + width * interval))
        hi=$((start + width * (interval + 1)))
        for i in $(seq 1 $n); do
            printf '\n\n---------------------- Problem #%s ----------------------\n\n' "$((i + n * (interval+1)))"
            python main.py --dt="d" --lo="$lo" --hi="$hi"
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                cmake -S $path -B ./build -Wno-dev
                cmake --build ./build
                ./build/examples/andersonBuffer/buffer
            else
                printf "Skipping bad problem."
            fi
        done
    done
}

main
