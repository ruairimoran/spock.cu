#!/bin/bash
set -ux pipefail


main() {
    path="../../"
    export PYTHONPATH="$path"
    # shellcheck disable=SC1090
    source "${path}.venv/bin/activate"
    mkdir -p ./build
    for n in {3..15}; do
        python main.py --dt="d" --h="$n"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S $path -B ./build -Wno-dev
            cmake --build ./build
            ./build/examples/admmServerAC/admm
        else
            printf "Python error!"
        fi
    done
}

main
