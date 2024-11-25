#!/bin/bash
set -euxo pipefail


main() {
    export PYTHONPATH=.
    pip install virtualenv
    virtualenv -p python3.10 venv@3.10
    source venv@3.10/bin/activate
    pip install --upgrade pip
    pip install .

    for N in {3..17}
    do
        nx=50
        python main.py --N="$N" --nx="$nx"
        cmake -S . -B ./build -Wno-dev
        cmake --build ./build
        ./build/spock_main
    done
}

main
