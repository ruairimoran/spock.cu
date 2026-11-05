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
minutes. Bash's `pipefail` preserves failure exit codes when output is piped
through `tee` to save logs.

Logs are retained on the runner at
`$HOME/spock-ci-logs/<run-id>-<attempt>/`, outside the temporary build tree.
On the A40 machine this is `/home/biggirl/spock-ci-logs/`. Each run includes
Python environment setup, configure/build/test console output, CMake and
CTest reports, GPU validation reports, sanitizer reports and `run-info.txt`.
The final `scripts/save_ci_logs.sh` step copies available reports even if
earlier steps failed. The job summary displays the persistent directory.
Logs are not uploaded as GitHub artifacts and are retained until removed
on the runner.

For example, inspect a run directly on the A40 runner with:

```bash
ls ~/spock-ci-logs/42079491984-1/
tail -n 100 ~/spock-ci-logs/42079491984-1/cuda-tests.log
```

This example path is created only if that run uses the updated workflow;
previous runs stored their reports in the temporary build directory.

SPOCK fetches GPUtils as headers without configuring its standalone CMake
project. That project overrides `CMAKE_CUDA_COMPILER`, which can conflict
with SPOCK's selected compiler and cause repeated CMake cache resets.

For local testing, run the scripts above from the checkout. They default to
`build/` and use `build/venv/bin/python` when present. The old `ci/script.sh`
is no longer used by the workflow.
