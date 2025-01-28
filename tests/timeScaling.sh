#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    pip install virtualenv
    virtualenv -p python3.10 venv@3.10
    source venv@3.10/bin/activate
    pip install --upgrade pip
    pip install .

    e=2
    first=3
    last=8
    python tests/timeCvxpy.py --dt="d" --nEvents="$e" --nStates="5" --first="$first" --last="$last"
    for n in $(seq $first $last); do
        python main.py --dt="d" --nStages="$n" --nEvents="$e" --nStates="5"
        cmake -S . -B ./build -Wno-dev
        cmake --build ./build
        ./build/spock_main
    done
}

main
