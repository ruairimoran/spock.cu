#!/bin/bash
set -euxo pipefail

# ------------------------------------
# Run Python tests
# ------------------------------------
test_python() {
    # -- create virtual environment
    export PYTHONPATH=.

    # -- install virtualenv
    pip install virtualenv

    # -- create virtualenv
    virtualenv -p python3.10 venv@3.10

    # -- activate venv
    source venv@3.10/bin/activate

    # -- upgrade pip within venv
    pip install --upgrade pip

    # -- install dependencies
    pip install '.[test]'

    # -- run the python tests
    python -W ignore tests/testTreeFactory.py -v
    python -W ignore tests/testProblemFactory.py -v

    # -- run the test file to create data for c++ testing
    python tests/test.py
}

# ------------------------------------
# Run C++ tests
# ------------------------------------
test_cpp() {
    # -- create build files
    cmake -S . -B ./build -Wno-dev -DSPOCK_BUILD_TESTING=ON

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/tests --output-on-failure

    # -- run compute sanitizer
    cd ./build/tests
    /usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full ./spock_tests
}

main() {
    test_python
    test_cpp
}

main
