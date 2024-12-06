#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    pip install virtualenv
    virtualenv -p python3.10 venv@3.10
    source venv@3.10/bin/activate
    pip install --upgrade pip
    pip install .

    for N in {3..5}
    do
        python main.py --dt="d" --N="$N" --nx="50"
        cmake -S . -B ./build -Wno-dev
        cmake --build ./build
        ./build/spock_main
    done
}

main
