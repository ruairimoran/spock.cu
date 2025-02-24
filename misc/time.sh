#!/bin/bash
set -ux pipefail  # removed e, o


main() {
    export PYTHONPATH=.
    source venv/bin/activate
    for i in {1..1000}; do
        printf "\n\nProblem #$i.\n\n"
        python main.py --dt="d"
        julia tests/julia/test.jl
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S . -B ./build -Wno-dev
            cmake --build ./build
            ./build/spock_main
        else
            printf "Skipping bad problem."
        fi
    done
}

main
