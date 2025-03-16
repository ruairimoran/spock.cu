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
    virtualenv -p python3.10 .venv

    # -- activate venv
    source .venv/bin/activate

    # -- upgrade pip within venv
    pip install --upgrade pip

    # -- install dependencies
    pip install 'api/python[test]'

    # -- run the python tests
    python -W ignore tests/python/testTreeFactory.py -v
    python -W ignore tests/python/testProblemFactory.py -v

    # -- run the test file to create float- and double-type data for C++ testing
    python tests/cuda/test.py --dt='f'
    python tests/cuda/test.py --dt='d'
}

# ------------------------------------
# Run C++ tests
# ------------------------------------
test_cpp() {
    # -- create build files
    cmake -S . -B ./build -Wno-dev -DSPOCK_BUILD_TEST=ON

    # -- build files in build folder
    cmake --build ./build

    # -- run tests
    ctest --test-dir ./build/tests --output-on-failure

    # -- run compute sanitizer
    cd ./build/tests
    /usr/local/cuda-12.3/bin/compute-sanitizer --tool memcheck --leak-check=full --print-limit=3 ./spock_tests
}

main() {
    test_python
    test_cpp
}

main
