#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    source venv@3.10/bin/activate
    firstN=11
    lastN=14
    for s in 10 100; do
        for n in $(seq $firstN $lastN); do
            python main.py --dt="d" --nEvents=2 --nStages="$n" --stop="$((n-1))" --nStates="$s"
            cmake -S . -B ./build -Wno-dev
            cmake --build ./build
            ./build/spock_main
        done
    done
}

main
