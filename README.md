# spock.cu

A `CUDA C++` implementation of the SPOCK algorithm (available [here](https://arxiv.org/pdf/2212.01110)) for multistage risk-averse optimal control problems (RAOCPs).

This solver handles RAOCPs with:

- Affine dynamics
- Quadratic(-plus-linear) stage and terminal costs
- Coherent risk measures
- Convex state-input constraints

## Hints and tips
- Once you have created and activated a python(-tk) virtual environment, you can install all package dependencies with `pip install '.[draw, test]'`. You will need python-tk for drawing the scenario trees.
- Make sure your CUDA executable is added in `CMakeLists.txt` at the top level.
- For testing, add flag `-DSPOCK_BUILD_TEST=ON` to CMake options.
