#!/bin/bash
set -ux pipefail  # removed e, o


main() {
    export PYTHONPATH=.
    source venv/bin/activate
    for _ in {1..2}; do
        python main.py --dt="d"
        julia tests/julia/test.jl
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            cmake -S . -B ./build -Wno-dev
            cmake --build ./build
            ./build/spock_main
        else
            echo "Skipping bad problem."
        fi
    done
}

main
