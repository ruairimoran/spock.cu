#!/bin/bash
set -ux pipefail  # removed e, o


main() {
    export PYTHONPATH=.
    source venv/bin/activate
    for _ in {1..10}; do
        python main.py --dt="d"
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
