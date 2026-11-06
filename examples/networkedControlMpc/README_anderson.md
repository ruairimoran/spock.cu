# Regularised Anderson acceleration

Incremental QR is the default through `--aa-qr update`; see
[README_fast_qr.md](README_fast_qr.md) for factor maintenance, safeguards,
options and validation. Use `--aa-qr full` to reproduce the full-QR method
described below in the main executable.

The default `regularized` mode improves the numerical treatment of the
type-II Anderson direction. It keeps the existing three-pair history,
SuperMann acceptance rules, primal-dual warm start, preconditioning and
stopping test. The ridge term applies only to the acceleration fit; it
does not change the optimal control objective.

## Numerical method

Let `Y` contain residual differences, newest first, and let the matching
columns of `C` be iterate differences minus residual differences. The
direction remains `d = -r - C gamma`.

The implementation compresses the history using GPUtils QR factors. The
reduced `R` and `Q' r` stay on the device for filtering and ridge fitting.
Rolling QR maintains its small `R` on the CPU and uploads it once per fit;
the full-QR path extracts `R` directly into the device fit buffer.

Define `s` as the largest history-column Euclidean norm. Two-pass modified
Gram-Schmidt on `R/s`, in newest-first order, chooses independent physical
columns `J`. A column is used only when its innovation norm exceeds the
relative rank threshold. This is filtering of columns, not truncation of
singular modes. Raw history pairs still age out normally; coefficients of
filtered columns are exactly zero.

The selected fit solves

    min || (Y[:,J]/s) gamma_J - r/s ||_2^2 + regularization * ||gamma_J||_2^2.

It uses GPUtils `Svd<T>` (cuSOLVER `gesvd`) for the small selected `R/s`
matrix, padded with zero columns to retain a fixed workspace size. The object
and its device workspace are constructed once; each fit refills its input.
Device kernels consume GPUtils' U, singular values and V-transpose directly,
form coefficients in double intermediate precision, and check the coefficient
and direction limits. It avoids normal equations
and never explicitly squares the potentially extreme history scale.
The coefficient vector is mapped back to the original physical columns
before multiplying by `C`.

If the fit has no independent columns, fails numerically, produces excessive
coefficients, or exceeds the direction/residual norm ratio, the solver uses
`d = -r` and clears the valid-history count. Three genuine new secant pairs
are then required before another accelerated fit. Iteration zero is not a
genuine secant pair. A nonfinite current residual produces an explicit
failure instead of attempting to continue with an invalid residual.

Device buffers, registered host buffers and QR/SVD workspaces are allocated once.
The reduced fit reads the SVD status on the device and safely handles failed
preparation with a finite zero dummy SVD input. One compact final download
provides the status, selected columns, coefficients and scalar diagnostics used
by host bookkeeping. A scaled device norm reduction handles direction checks
without squaring extreme raw values. Input-history and residual norm checks,
QR rebuild status checks, and rolling-QR maintenance retain their existing
host completion boundaries. This is a device reduced fit, not an entirely
device-controlled Anderson iteration.

GPUtils v2.2.0 supplies QR, SVD, matrix products, copies, registered transfers,
and Givens rotations. It does not provide the ordered Anderson filter, ridge
coefficient assembly, or a norm operation with a device scalar output. These
operations are implemented in `src/anderson_reduced.cuh` without new direct
cuBLAS/cuSOLVER calls. Existing rolling-QR projection/update code uses its
existing transpose GEMV operations and fused Givens/deletion kernel; GPUtils
does not expose an allocation-free transposed matrix-vector product. The CPU
reference in `tests/cuda/referenceAndersonCpu.cuh` is retained for regression
comparison and is not selected by the production solver.

Both modes in the main executable use safe, back-to-front column copies.
The previous implementation shifted overlapping device-memory regions,
whose copy behaviour was not guaranteed. Using the same safe history update
in both modes isolates the regularisation/filtering comparison. Running
processes keep their existing executable and are not changed mid-study.

## Options and defaults

The driver defaults to regularised Anderson with QR updates. Select
`--anderson legacy` to use the original acceleration method.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--aa-regularization` | `1e-8` | Positive ridge parameter in the normalised fit |
| `--aa-rank-tolerance` | `1e-8` | Global relative innovation cutoff |
| `--aa-max-coefficient-norm` | `1e6` | Maximum accepted coefficient Euclidean norm |
| `--aa-max-direction-ratio` | `1e4` | Maximum accepted `||d||_2 / ||r||_2` |

The generic C++ helper uses rank threshold `1e-4` for float when its rank
option is zero (automatic); the MPC worker uses double precision. A fixed
positive ridge does not itself establish a superlinear convergence rate.
The existing SuperMann safeguards are retained. The Euclidean direction
bound also gives a bound in its fixed positive-definite metric by norm
equivalence.

## Build and run on the QUB machine

```bash
cd ~/Documents/Development/spock.cu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSM_ARCH=120 \
  -UFETCHCONTENT_SOURCE_DIR_GPUTILS
make -C build -j2 networked_mpc_persistent check_anderson

nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_regularized_full_qr \
  --trajectories 15 --steps 20 --seed 20261103 --branching 8 8 \
  --anderson regularized --aa-qr full \
  > build/mpc_tree64_regularized_full_qr.log 2>&1 < /dev/null &
echo $! > build/mpc_tree64_regularized_full_qr.pid
```

Use a fresh output directory and avoid overlapping timing experiments on
the GPU. To compare with the original unregularised fit, use
`--anderson legacy` in the main executable and another output directory.

## Diagnostics and validation

The CSV records fits, accepted directions, filtered columns, restarts and
rejection reasons, initial/final residuals, the effective stopping tolerance,
CP-operator evaluations and backtracking counts. Settings and the worker
binary hash are stored in `parameters.json`.

The independent oracle uses a direct full-history SVD rather than the
production QR-compression path. It checks exact/near dependence, physical
column alignment, scale extremes, zero history, regularisation and safety
fallbacks. Run both precisions:

```bash
build/venv/bin/python examples/networkedControlMpc/test_anderson_reference.py \
  --gpu-executable build/examples/networkedControlMpc/check_anderson \
  --include-nonfinite --report build/anderson_double_validation.json
build/venv/bin/python examples/networkedControlMpc/test_anderson_reference.py \
  --gpu-executable build/examples/networkedControlMpc/check_anderson \
  --gpu-precision float --include-nonfinite \
  --report build/anderson_float_validation.json
```

`check_anderson_mpc.py` checks cold/warm resets, forced coefficient rejection,
rebootstrap, consistent controls, solver counters and persistent GPU storage
on a small scenario tree. `benchmark_anderson.py` replays fixed Case Study 2
snapshots with cold starts in both modes, verifying matching initial
residuals and stopping thresholds. Short validation runs do not support
claims about average or 95th-percentile closed-loop speedup.

The initial validation on the QUB GPU passed 18 CPU reference tests, 21
double-precision and 22 single-precision GPU fixtures, and ten persistent
small-tree requests, including forced rejection and rebootstrap. CUDA
Compute Sanitizer reported no errors on the double-precision fixtures.
A captured warm Case Study 2 request made no CUDA allocation/free calls.

Two matching cold Case Study 2 snapshots gave the following GPU intervals:

| Snapshot step | Legacy | Regularised | Outer iterations, both modes |
| --- | ---: | ---: | ---: |
| 0 | 14.010 s | 14.507 s | 450 |
| 2 | 15.758 s | 16.560 s | 500 |

These are individual diagnostic solves, not a statistical benchmark. They
show a 3.6–5.1% overhead without an iteration reduction on these two problems.
The rejection safeguards are exercised by the stress tests; their presence
does not establish faster convergence on the manuscript's MPC problems.


## CPU reduced fit

Use `--aa-fit cpu` with `run_mpc_persistent.py` to keep tall compression and
direction construction on the GPU while performing ordered filtering, the
small SVD, coefficient formation and coefficient checks on the CPU.
`--aa-fit gpu` remains the default. Both backends use the same regularisation,
rank thresholds, QR update policy, GPU direction norm and rejection checks.
The worker records `aa_fit_backend` in its initialisation response.

GPUtils does not provide a CPU SVD. The CPU backend uses Eigen 3.4.0's
preallocated Jacobi SVD, compiled separately from the CUDA translation units.
CMake fetches the versioned, SHA-256-pinned archive. Eigen threading is disabled;
all host scratch is retained and a Release-active allocation-guard regression
verifies that repeated SVD calls do not allocate Eigen storage. The backend
preserves the double intermediate arithmetic of the existing column filter and
coefficient assembly, with float/double SVD and output precision as requested.

With rolling QR, R is already maintained on the CPU: only Q-transpose times
residual is downloaded into registered storage. With full QR, the small R and
projected RHS are downloaded on one stream with one completion wait.
CPU results and status are uploaded in one registered transfer; GPU matrix
products, direction checks and the final diagnostic download follow on the
same stream. Borrowed RHS/coefficient aliases remain supported. No new
low-level BLAS calls are used.

For example, append `--aa-fit cpu` to an existing regularised MPC command:

```bash
build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_cpu_fit --trajectories 15 --steps 20 \
  --seed 20261103 --branching 8 8 --diagonal-costs \
  --anderson regularized --aa-qr update --aa-fit cpu \
  --projections sequential --cuda-graphs cp
```
