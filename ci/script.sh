#!/bin/bash
set -euxo pipefail

test_py() {
    # Run Python tests
    # ------------------------------------

    # -- create virtual environment
    export PYTHONPATH=.

    # -- install virtualenv
    pip install virtualenv

    # -- create virtualenv
    virtualenv -p python3.11 venv

    # -- activate venv
    source venv/bin/activate

    # -- upgrade pip within venv
    pip install --upgrade pip

    # -- install dependencies
    pip install ./tests

    # -- run the python tests
    python -W ignore tests/test_tree_factories.py -v
}

test_cpp() {
    # Run C++ gtests using cmake
    # ------------------------------------

    # -- change into test directory
    cd tests

    # -- download gtest and create build files in tests/build
    cmake -S . -B build -Wno-dev

    # -- build files in build folder
    cmake --build build

    # -- change into build directory
    cd build

    # -- run tests
    ctest
}


main() {
    test_py
    test_cpp
}

main
