GitHub Actions uses the self-hosted A40 runner with CUDA 12.3, architecture
`86`, and Python 3.10 (including its `venv` module). The compiler and Compute
Sanitizer paths are selected in `.github/workflows/ci.yml`.

Each job creates a separate build tree and Python environment under
`runner.temp`. It then calls these repository scripts directly:

```bash
bash ./scripts/run_python_tests.sh
bash ./scripts/run_cuda_tests.sh
bash ./scripts/run_compute_sanitizer.sh
```

`SPOCK_BUILD_DIR` and `PYTHON` make all three scripts use the same job-owned
build tree and interpreter. The workflow configures the compiler before
running them. It limits configuration to five minutes and the job to 45
minutes, and uploads CMake, CTest and sanitizer logs even when a step fails.

SPOCK fetches GPUtils as headers without configuring its standalone CMake
project. That project overrides `CMAKE_CUDA_COMPILER`, which can conflict
with SPOCK's selected compiler and cause repeated CMake cache resets.

For local testing, run the scripts above from the checkout. They default to
`build/` and use `build/venv/bin/python` when present. The old `ci/script.sh`
is no longer used by the workflow.
