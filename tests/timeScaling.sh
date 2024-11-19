#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    pip install virtualenv
    virtualenv -p python3.10 venv@3.10
    source venv@3.10/bin/activate
    pip install --upgrade pip
    pip install .

    for N in 1 {5..10..5}
    do
        for nx in 50
        do
            python main.py --N="$N" --nx="$nx"
            cmake -S . -B ./build -Wno-dev
            cmake --build ./build
            ./build/spock_main
        done
    done
}

main
