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

## Dependencies, build and tests

GPUtils is pinned to tag **v2.0.0** with an archive SHA-256 check in CMake
and the Python API dependency. This release includes the QR-workspace fix;
the old local GPUtils patch is no longer required. Clear any previous
`FETCHCONTENT_SOURCE_DIR_GPUTILS` override to fetch the pinned release.

For the QUB Blackwell machine (use the appropriate architecture for other GPUs):

```bash
python3 -m venv build/venv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSM_ARCH=120 \
  -UFETCHCONTENT_SOURCE_DIR_GPUTILS -DSPOCK_BUILD_TEST=ON \
  -DPython3_EXECUTABLE="$PWD/build/venv/bin/python"
build/venv/bin/python -m pip install './api/python[test]'
build/venv/bin/python -m pip install --force-reinstall --no-deps \
  build/_deps/gputils-src/python
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The explicit GPUtils Python reinstall also upgrades environments that already
contain its unchanged Python package version number, `1.7.2`; the installed
code must come from the v2.0.0 release archive. The C++ tag and Python package
metadata use different version numbers.

CTest generates float/double fixtures in the build directory and covers the
CUDA solver, Python factories, dense cost operators, Anderson fits, QR
updates and the large QR-workspace regression. GPU tests share a resource
lock so `ctest -j` does not run them concurrently. The Python tests need the
API's `test` dependencies, including CVXPY.

The MPC integration test can additionally be run in a fresh output directory:

```bash
build/venv/bin/python examples/networkedControlMpc/check_anderson_mpc.py \
  --worker build/examples/networkedControlMpc/networked_mpc_persistent_qr \
  --output build/mpc_integration_check
```
