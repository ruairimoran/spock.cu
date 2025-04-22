# spock.cu

A `CUDA C++` implementation of the SPOCK algorithm (available [here](https://arxiv.org/pdf/2212.01110)) for multistage risk-averse optimal control problems (RAOCPs).

The solver handles RAOCPs with:
- affine dynamics,
- quadratic and/or linear stage and terminal costs,
- coherent risk measures,
- convex state-input constraints.

The solver is based on scenario trees, which it can build from:
- Markov chains,
- i.i.d. processes,
- data samples.

## Hints and tips
- Make sure your CUDA executable is added in `CMakeLists.txt` at the top level.
- For `CUDA C++` testing, add flag `-DSPOCK_BUILD_TEST=ON` to CMake options.
