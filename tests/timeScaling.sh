#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    pip install virtualenv
    virtualenv -p python3.10 venv@3.10
    source venv@3.10/bin/activate
    pip install --upgrade pip
    pip install .

    for n in {3..16}; do
        python main.py --dt="d" --nStages="$n" --nStates="50"
        cmake -S . -B ./build -Wno-dev
        cmake --build ./build
        ./build/spock_main
    done
}

main
