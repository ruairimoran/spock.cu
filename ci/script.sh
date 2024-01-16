#!/bin/bash
set -euxo pipefail

regular_test() {
    # Run Python tests
    # ------------------------------------

    # --- create virtual environment
    export PYTHONPATH=.

    # --- install virtualenv
    pip install virtualenv

    # --- create virtualenv
    virtualenv -p python3.11 venv

    # --- activate venv
    source venv/bin/activate

    # --- upgrade pip within venv
    pip install --upgrade pip

    # --- change into test directory
    cd ../tests

    # --- install dependencies
    pip install .

    # --- run the tests
    python -W ignore test_tree_factories.py -v
}


main() {
    regular_test
}

main
