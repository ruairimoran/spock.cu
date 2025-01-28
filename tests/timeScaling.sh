#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    source venv@3.10/bin/activate
    firstE=3
    lastE=3
    firstN=5
    lastN=10
    stop=4
    states=100
    for e in $(seq $firstE $lastE); do
        for n in $(seq $firstN $lastN); do
            python main.py --dt="d" --nEvents="$e" --nStages="$n" --stop="$stop" --nStates="$states"
            cmake -S . -B ./build -Wno-dev
            cmake --build ./build
            ./build/spock_main
        done
    done
}

main
