# Persistent GPU MPC

For the current QR-update executable and regularised fitting options, see
[README_fast_qr.md](README_fast_qr.md).

`run_mpc_persistent.py` uses `networked_mpc_persistent`, which constructs the
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
preserved. At startup the six transmitted factors and the CPU Riccati `P`
are compared with the original factory output before simulation begins.

The `linear-v2` update protocol sends only `BTr`, `AB`, `K`, `KTr`, `ABKTr`
and `lowCholesky`, in that order. The linear GPU projection does not read
`P` or `APB`: `P` remains CPU workspace for the Riccati recursion, while the
unused `APB` product is no longer generated online. For the 64-scenario
case, this reduces each update from 2,530,620,000 to 1,821,900,000 bytes
(28.0%). Existing static problem files remain supported. Rebuild the
worker when updating the driver; the startup handshake checks the protocol,
field shapes and payload size before sending any coefficients.

The original `run_mpc.py` and `networked_mpc` remain available. A process
already running those continues using the original implementation.

## Build and run

GPUtils v2.2.0 is pinned by the project and includes the QR-workspace fix
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
- `dynamics_update_bytes`: coefficient payload sent for this solve; zero
  when no update is requested. The worker's startup metadata records the
  protocol and full per-update payload size.
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

The regularised Anderson fit is documented in `README_anderson.md`. The
driver defaults to `regularized` with QR updates; use `--anderson legacy`
to select the original fit.

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

Add `--projections concurrent` to `run_mpc_persistent.py` to overlap the S1/S2
projections. The worker creates two GPUtils stream/handle pairs before any tensors,
and joins the streams with CUDA events before using the complete primal result.
Sequential execution remains the default because it was faster on the current
case study/GPU; `--projections sequential` explicitly selects it. The mode is recorded
in `parameters.json` and in worker responses. The numerical solver settings and
warm-start policy are the same in both modes.


## Optional CUDA graphs (sequential projections)

Use `--cuda-graphs cp` to replay the complete CP operator, or `--cuda-graphs s1`
to capture only the dynamics projection. The default is `--cuda-graphs none`.
Both graph modes use sequential projections; combining them with
`--projections concurrent` is rejected. For example, after rebuilding the worker:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_graph_cp \
  --trajectories 15 --steps 20 --seed 20261103 --branching 8 8 \
  --anderson regularized --aa-qr update \
  --projections sequential --cuda-graphs cp
```

The graph is constructed once per persistent worker and reused for all cold and
warm solves, including in-place dynamics updates. Anderson, adaptive line search
and termination remain outside the graph. `gpu_time_s` excludes construction;
`graph_setup_wall_s` reports the extra one-time graph setup in the first CSV row.
The effective graph mode, node count and per-solve replay count are recorded.
Use fresh output directories and run comparisons sequentially on the GPU.
See [stream execution](../../docs/stream_execution.md) for the cache lifetime and
fixed-address requirements.


## Compact diagonal costs

Dense cost storage remains the default. Add `--diagonal-costs` to select the
existing compact kernels for Case Study 2's diagonal Q, R and terminal Q:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_diagonal \
  --trajectories 15 --steps 20 --seed 20261103 --branching 8 8 \
  --anderson regularized --aa-qr update \
  --projections sequential --cuda-graphs cp --diagonal-costs
```

Use a fresh output directory. The option rejects matrices with any nonzero
coupling off the diagonal, rather than changing their costs. General dense
weights remain supported when the option is omitted. The console prints the
actual factor structures, and `parameters.json` records `diagonal_costs` and
`cost_factor_structures`; all three factors should report `diagonal`.

The stored reference problem is checked in its original dense representation
before generating the requested tree and cost format. Plant parameters used for
rollout and saved in `plant_parameters.npz` retain their original dense arrays.
Warm starts, graph selection, dynamics updates and GPU timing retain their
existing behaviour. The cost format is selected when generating the problem;
the existing CUDA worker reads it from that problem's metadata.


## Fused backward contributions

The dynamics projection computes both child contributions, `BTr * y` and
`ABKTr * y`, in one tiled CUDA kernel per stage. It reads the existing factor
buffers directly and uses the existing input/state workspaces. For affine
dynamics, the kernel loads `y = q + P c`; linear dynamics use `y = q`.
A second kernel sums both contribution blocks for each parent and forms the
unsolved input right-hand side. Each parent/output component has one writer,
with children accumulated in their stored order and no atomics. The root
computes only the input block because its state contribution is unused.

This is the default implementation and needs no additional driver option.
The all-node Cholesky solve, forward reconstruction, binary MPC updates and
warm-start policy retain their existing interfaces. The kernels support dense
factors, float/double precision and captured CUDA graphs, and allocate no
additional device workspace.

For a fixed-input microbenchmark of the current implementation:

```bash
make -C build -j2 benchmark_dynamics
build/examples/networkedControlMpc/benchmark_dynamics \
  build/mpc_tree64_diagonal/initial_problem/data \
  build/s1_benchmark_output s1 graph 100
```

Replace the data directory with an existing MPC run. Use `cp` instead of `s1`
to measure the full CP operator, and `eager` instead of `graph` for uncaptured
execution. Standard output is JSON containing each CUDA-event sample, host
wall time, graph node count and tracked device bytes; the output prefix saves
the resulting tensor for numerical comparison. Setup, graph construction and
input preparation are excluded from the recorded intervals.


## Experimental partial lifting

On branch `f/partial-lifting`, use `--partial-lifting B` with `B` equal to
`2`, `3`, `4`, `5` or `8` to select blocked dynamics projections. The default `0`
keeps the existing fused stage sweeps. The diagnostic value `1` selects a
reordered stage baseline with whole-tree known-term/control products and affine
state reconstruction, but without block matrices.

```bash
make -C build -j2 networked_mpc_persistent benchmark_dynamics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
build/venv/bin/python -u examples/networkedControlMpc/run_mpc_persistent.py \
  --output build/mpc_tree64_partial4 \
  --trajectories 15 --steps 20 --seed 20261103 --branching 8 8 \
  --anderson regularized --aa-qr update --diagonal-costs \
  --projections sequential --cuda-graphs cp --partial-lifting 4
```

Use a fresh output directory for each schedule. Both `s1` and `cp` graph modes
are supported. The fixed-input benchmark accepts an optional final block-size
argument, for example `... DATA output.bt s1 graph 100 8`; use an output name
ending in `.bt` to save binary tensor data.

The implementation finds the common single-child suffix after the last
branching stage. It keeps the branching prefix on stage sweeps, divides each
remaining chain into short blocks, and performs independent local
summarisation, coarse boundary sweeps, and independent interior reconstruction
in both directions. Short final blocks are handled explicitly. If no unary
suffix exists, the original stage implementation is used; startup metadata
reports whether the requested schedule is active.

One transposed block matrix is shared by the forward and backward recurrences.
GPUtils composes these matrices on the GPU. Two chain-sized matrix batches are
retained as preparation scratch rather than allocating scratch for every block.
State vectors are held in block-local shared memory during short recurrences;
this requires two state vectors to fit the GPU's default shared-memory limit.
Unsupported state dimensions are rejected during configuration. All data and
workspaces remain allocated throughout the MPC simulation.

Configure the C++ interface with
`problem.dynamics()->configurePartialLifting(4)` before constructing the cache.
Reconfiguration after the first projection or graph capture is rejected because
graphs retain fixed addresses. After changing projection factors in place, call
`problem.dynamics()->refreshProjectionFactors()` on their execution stream
before the next projection. This enqueues preparation without allocating new
storage. The persistent worker does this automatically after each linear MPC
update, including when replaying an already captured graph. The existing
six-field host-to-GPU update payload is unchanged.

Startup metadata records the selected block size, suffix depth/length, block
count, summary-matrix bytes, preparation-scratch bytes and exact additional
GPUtils-tracked device bytes. For the stored 64-scenario problem, the 45-edge
unary tail gives the following block counts and approximate extra allocations,
including preparation scratch, vector/index storage and pointer metadata:

| Block size | Blocks | Extra GPU storage (MB) |
| --- | ---: | ---: |
| 2 | 1,472 | 289.9 |
| 3 | 960 | 197.1 |
| 4 | 768 | 162.3 |
| 5 | 576 | 127.5 |
| 8 | 384 | 92.7 |

Both B=3 and B=5 divide the tail exactly; short final blocks remain supported
for other horizons. The initial preparation is part of `setup_wall_s`; subsequent
`partial_lifting_update_wall_s` values record summary recomposition and are also
included in `update_wall_s`. CUDA-event `gpu_time_s` continues to measure only
the resident solver interval.
