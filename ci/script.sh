#!/bin/bash
set -euxo pipefail

test_all() {
    # Run Python tests
    # ------------------------------------

    # -- create virtual environment
    export PYTHONPATH=.

    # -- install virtualenv
    pip install virtualenv

    # -- create virtualenv
    virtualenv -p python3.10 venv

    # -- activate venv
    source venv/bin/activate

    # -- upgrade pip within venv
    pip install --upgrade pip

    # -- install dependencies
    pip install .

    # -- run the python tests
    python -W ignore tests/test_tree_factories.py -v


    # Run C++ gtests using cmake
    # ------------------------------------

    # -- generate simple tree for testing
    python tests/test_tree_main.py

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

    # -- run compute sanitizer
    /usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full ./spock_tests
}


main() {
    test_all
}

main
