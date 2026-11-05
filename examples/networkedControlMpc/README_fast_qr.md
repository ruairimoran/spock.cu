# Incremental QR for regularised Anderson acceleration

The regularised solver can retain a thin QR factorisation across Anderson
fits. This reduces the cost of processing tall, three-column histories.
The ridge parameter, newest-first history filtering, safety bounds,
SuperMann rules, warm starts and stopping test are unchanged.

## Factor maintenance

`src/anderson_qr.cuh` maintains the raw residual history in oldest-first
factor order, while the caller keeps its original newest-first physical
columns. At each history change:

1. Remove the oldest column of the small `R`.
2. Restore triangular form with Givens rotations. One CUDA kernel applies
   all rotations to the thin `Q`, with one thread per row.
3. Insert the new residual difference using two-pass classical
   Gram--Schmidt, implemented with GEMV operations against the retained
   basis. Normalise and append its innovation.
4. Compute `Q' r` with GEMV. Reverse the small `R` columns back to physical
   history order before the existing filtering and ridge fit.

The transformed small matrix need not be triangular in physical order.
Filtering and regularisation depend on its column geometry, which the
orthonormal change of basis preserves. Matching physical `C` columns and
coefficient indices are unchanged.

History version counters prevent an update after unobserved history
changes. A repeated fit of the same history reuses the factors. Every MPC
solve resets factor validity; an Anderson rejection invalidates it too.

A fresh Householder QR and explicit thin `Q` are formed initially, after
64 successful updates, when a history version is skipped, or when an
update becomes unreliable. Columns whose diagonal innovation is at most
the history rank threshold times the largest column norm trigger a full
rebuild. Deficient histories remain usable by the existing filtered fit,
but are not incrementally advanced until suitable again.

Every 16 successful updates, the code checks the maximum absolute entry of
`Q'Q-I`. A failed check triggers a full rebuild before the fit is used.
Default tolerances are `1e-10` for double and `1e-4` for float. All device
buffers, QR/ORGQR workspaces and small host arrays are allocated at setup.

The deletion/insertion construction follows
[Walker's implementation notes, pp. 2–3](https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf),
with reorthogonalisation and rebuild safeguards added here. The Givens
rotations change the stored basis; they do not change the optimisation
objective or add a new convergence guarantee.

## Build and run

On the QUB machine:

```bash
cd ~/Documents/Development/spock.cu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSM_ARCH=120 \
  -UFETCHCONTENT_SOURCE_DIR_GPUTILS
make -C build -j2 networked_mpc_persistent check_anderson_qr check_anderson

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_qr_main \
  --trajectories 15 --steps 20 --seed 20261103 --branching 8 8 \
  --anderson regularized
```

The main solver and MPC driver use regularised Anderson with QR updates by
default. The worker JSON can omit `qr_mode`; regularised mode then uses QR
updates. Select `--aa-qr full` for a full QR at every fit, or
`--anderson legacy` for the original acceleration method.

Use a fresh output directory for each run. Previously completed results are
retained.

Options specific to the new path:

| Option | Default |
| --- | ---: |
| `--aa-qr` | `update` in regularised mode; `full` in legacy mode |
| `--aa-qr-rebuild-interval` | `64` |
| `--aa-qr-check-interval` | `16` |
| `--aa-qr-orthogonality-tolerance` | `0` (automatic) |

The CSV adds `aa_qr_rebuilds`, `aa_qr_updates`, `aa_qr_reuses`,
`aa_qr_update_fallbacks`, `aa_qr_orthogonality_checks` and
`aa_qr_last_orthogonality_error`. The last error is the most recent sampled
check, not a maximum over every fit. QR rebuilds are numerical maintenance
and are distinct from Anderson history restarts.

## Validation and timing

`test_anderson_qr.py` compares every frame of stateful sequences with an
independent direct tall-matrix SVD. Both precisions cover repeated history
updates, one/eight-column histories, exact/near dependence, scale changes,
periodic rebuilds, reset/reuse, missed versions, rejected fits and repair of
a deliberately damaged basis.

`check_anderson_qr --benchmark 1888055 100` compares complete Anderson fits
on identical GPU-resident histories and residuals at the Case Study 2 row
count. It alternates execution order, excludes ten warmup fits and includes
periodic checks/rebuilds. Generation and numerical comparisons are outside
the timed intervals. This measures fit cost without a changed convergence
trajectory affecting the comparison.

`benchmark_anderson.py --qr-comparison` compares complete cold-start MPC
solves on fixed snapshots with identical regularisation and stopping
thresholds. `check_anderson_mpc.py` additionally exercises cold/warm resets
and forced rejection with the new path.
