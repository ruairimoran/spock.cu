# Persistent GPU MPC

For the current QR-update executable and regularised fitting options, see
[README_fast_qr.md](README_fast_qr.md).

`run_mpc_persistent.py` uses `networked_mpc_persistent_qr`, which constructs the
scenario tree, problem data, solver cache, Cholesky objects and QR workspace
once for the whole simulation. Warm starts retain the complete accepted
primal-dual iterate on the GPU. Only the root control is downloaded.
Anderson history and scratch buffers are reset in place for each new solve;
the first problem of each trajectory receives a cold start.

Static costs, constraints, risk projections, topology and preconditioning
are generated with the original factory at startup. Subsequently,
`online_model.py` computes only the changing delay-dependent dynamics and
their Euclidean projection factors. The arrays are streamed as binary
float64 data directly to existing GPU buffers, without writing new model
files. The analytic discretisation currently requires the diagonal `Ac`
used by Case Study 2; unsupported plants are rejected explicitly.

The stored prediction horizon remains 47 transitions / 48 stages. The
stored Beta law, costs, constraints, AVaR parameter and tolerances are
preserved. At startup all eight online dynamics/projection tensors are
compared with the original factory output before simulation begins.

The original `run_mpc.py` and `networked_mpc` remain available. A process
already running those continues using the original implementation.

## Build and run

GPUtils v2.0.0 is pinned by the project and includes the QR-workspace fix
needed for larger trees. The historical `gputils_qr_workspace.patch` must
not be applied to this release.

On the QUB machine, from the repository root (the configure step registers
the new target and clears the previous dependency-source override):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSM_ARCH=120 \
  -UFETCHCONTENT_SOURCE_DIR_GPUTILS
make -C build -j2 networked_mpc_persistent_qr

nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_case_study_2_tree64_persistent \
  --trajectories 15 --steps 20 --seed 20261103 \
  --tolerance 1e-3 --max-time 300 --branching 8 8 \
  > build/mpc_case_study_2_tree64_persistent.log 2>&1 < /dev/null &
echo $! > build/mpc_case_study_2_tree64_persistent.pid
```

Use a fresh output directory and run timing experiments sequentially on
the GPU. For a 256-scenario pilot, use `--branching 16 16`,
`--trajectories 1 --steps 2`, and a new output directory.

```bash
tail --pid="$(cat build/mpc_case_study_2_tree64_persistent.pid)" \
  -f build/mpc_case_study_2_tree64_persistent.log

# After completion:
build/venv/bin/python examples/networkedControlMpc/check_persistent.py \
  build/mpc_case_study_2_tree64_persistent
build/venv/bin/python examples/networkedControlMpc/plot_mpc.py \
  build/mpc_case_study_2_tree64_persistent
```

## Timing fields

- `startup_generation_wall_s` in `parameters.json`: one-time factory
  generation and initial online/reference comparison.
- `setup_wall_s`: one-time GPU/cache construction, recorded in the first
  CSV row and zero thereafter.
- `generation_wall_s`: online CPU numerical update; zero for the first
  problem, whose coefficients were prepared at startup.
- `update_wall_s`: receiving binary coefficient arrays and uploading them
  into existing buffers, including synchronisation.
- `gpu_time_s`: CUDA-event elapsed interval around SPOCK, including
  in-place scratch reset, initial-state upload and host-paced gaps. This
  is an elapsed solve interval, not a sum of active kernel durations.
- `worker_wall_s`: request handling, coefficient transfer, solve and
  root-control extraction.
- `step_wall_s`: online CPU generation, worker exchange, and plant
  simulation, excluding one-time startup and CSV output.

The runtime plot continues to use `gpu_time_s` only. GPU memory should
remain allocated between steps, while utilisation can drop during CPU
preparation. The one-time setup still includes the original factory's
known warnings for dummy zero cost matrices.

The optional regularised Anderson fit is documented in
`README_anderson.md`. The driver defaults to `legacy`; use
`--anderson regularized` to enable the new fit explicitly.

## Validation performed

- Original 4-scenario and 64-scenario factory tensors matched by the online
  updater within numerical comparison tolerances.
- Three solves with 64 scenarios and four solves over two trajectories
  with 4 scenarios all converged and passed independent rollout checks.
- Solution pointers and tracked tensor storage remained constant across
  each simulation; GPU setup occurred only once.
- Nsight Systems captured the second 64-scenario request, including
  coefficient transfer, reset, warm solve and control extraction. The
  CUDA API trace contained no device allocation/free calls. The largest
  device-to-host transfer was 400 bytes, the 50-element root control.
- The four small-tree controls differed from the corresponding original
  study controls by at most 8.8e-5; repeated cold-start controls differed
  by at most 7.5e-5 (solver tolerance 1e-3).

In the two 64-scenario online updates, CPU preparation took about 2.57 s
and transfer/upload about 0.87 s. These are validation measurements; one
warm solve was profiled, so these short runs are not a new paper benchmark.
