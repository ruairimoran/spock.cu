#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    source venv@3.10/bin/activate
    for _ in {1..100}; do
        python main.py --dt="d"
        cmake -S . -B ./build -Wno-dev
        cmake --build ./build
        ./build/spock_main
    done
}

main
