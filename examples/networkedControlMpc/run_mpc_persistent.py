"""Case Study 2 MPC with persistent GPU storage and dynamics-only updates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from gputils_api import read_array_from_gputils_binary_file as read_bt

from online_model import OnlineModel
from run_mpc import generate_problem, read_parameters


def read_worker_response(worker, phase, *, exit_wait_seconds=1.0):
    """Read a JSON reply, allowing an exiting child to publish its return code.

    EOF can precede process exit by a short interval, so an immediate poll()
    can misleadingly report None after a worker failure.
    """
    line = worker.stdout.readline()
    if line:
        return json.loads(line)
    try:
        returncode = worker.wait(timeout=exit_wait_seconds)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Persistent GPU worker closed stdout while waiting for {phase} response "
                           f"but is still running (PID {worker.pid})") from None
    detail = f"return code {returncode}"
    if returncode < 0:
        try:
            detail += f" ({signal.Signals(-returncode).name})"
        except ValueError:
            detail += f" (signal {-returncode})"
    raise RuntimeError(f"Persistent GPU worker exited with {detail} before {phase} response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=15)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--branching", type=int, nargs="+")
    parser.add_argument("--diagonal-costs", action="store_true",
                        help="Store exactly diagonal Q, R and terminal Q compactly (default: dense)")
    parser.add_argument("--projections", choices=("sequential", "concurrent"), default="sequential",
                        help="Execution of the S1 and S2 projections (default: sequential)")
    parser.add_argument("--anderson", choices=("legacy", "regularized"), default="regularized",
                        help="Acceleration mode (default: regularized with QR updates)")
    parser.add_argument("--cuda-graphs", choices=("none", "s1", "cp"), default="none",
                        help="Capture sequential S1 or the complete CP operator (default: none)")
    parser.add_argument("--aa-fit", choices=("gpu", "cpu"), default="gpu",
                        help="Reduced filtering/SVD/coefficient backend (default: gpu)")
    parser.add_argument("--aa-regularization", type=float, default=1e-8,
                        help="Ridge parameter after normalising history columns")
    parser.add_argument("--aa-rank-tolerance", type=float, default=1e-8)
    parser.add_argument("--aa-max-coefficient-norm", type=float, default=1e6)
    parser.add_argument("--aa-max-direction-ratio", type=float, default=1e4)
    parser.add_argument("--aa-qr", choices=("full", "update"),
                        help="QR implementation for regularised Anderson (default: update)")
    parser.add_argument("--aa-qr-rebuild-interval", type=int, default=64)
    parser.add_argument("--aa-qr-check-interval", type=int, default=16)
    parser.add_argument("--aa-qr-orthogonality-tolerance", type=float, default=0.0,
                        help="Zero selects the precision-dependent default")
    args = parser.parse_args()
    if args.cuda_graphs != "none" and args.projections != "sequential":
        parser.error("CUDA graphs currently require --projections sequential")
    if args.anderson == "legacy" and args.aa_qr == "update":
        parser.error("QR updates require --anderson regularized")
    if args.anderson == "legacy" and args.aa_fit == "cpu":
        parser.error("CPU reduced fitting requires --anderson regularized")
    args.aa_qr = args.aa_qr or ("update" if args.anderson == "regularized" else "full")
    if args.trajectories < 1 or args.steps < 1 or args.tolerance <= 0 or args.max_time < 0:
        parser.error("Invalid simulation parameters")
    anderson = {"mode": args.anderson, "fit_backend": args.aa_fit, "regularization": args.aa_regularization,
                "rank_tolerance": args.aa_rank_tolerance,
                "max_coefficient_norm": args.aa_max_coefficient_norm,
                "max_direction_ratio": args.aa_max_direction_ratio,
                "qr_mode": args.aa_qr,
                "qr_rebuild_interval": args.aa_qr_rebuild_interval,
                "qr_check_interval": args.aa_qr_check_interval,
                "qr_orthogonality_tolerance": args.aa_qr_orthogonality_tolerance}
    if (not all(np.isfinite(value) for value in anderson.values() if isinstance(value, (int, float)))
            or args.aa_regularization <= 0 or not 0 <= args.aa_rank_tolerance < 1
            or args.aa_max_coefficient_norm <= 0 or args.aa_max_direction_ratio <= 0
            or args.aa_qr_rebuild_interval < 1 or args.aa_qr_check_interval < 1
            or not 0 <= args.aa_qr_orthogonality_tolerance < 1):
        parser.error("Invalid Anderson parameters")
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True)  # A fresh directory protects existing results.
    workspace = output / "initial_problem"
    workspace.mkdir()
    p = read_parameters(repo)
    original_branching = p["branching"].copy()
    if args.branching and (len(args.branching) > len(original_branching) or min(args.branching) < 1):
        parser.error("Branching factors must be positive and fit the stored prediction horizon")
    print("Generating and checking the stored reference problem once...", flush=True)
    startup_begin = time.perf_counter()
    # Check the stored dense reference before selecting a representation or tree.
    generate_problem(p, p["phi0"], p["gam"], workspace)
    comparisons = {}
    for name in ("dynamics_AB", "scaling", "nonleafCost_sqrtQ", "nonleafCost_sqrtR", "leafCost_sqrtQ"):
        reference = read_bt(str(repo / "examples/networkedControl/data" / f"{name}_d.bt"))
        generated = read_bt(str(workspace / "data" / f"{name}_d.bt"))
        comparisons[name] = float(np.max(np.abs(reference - generated)))
        np.testing.assert_allclose(generated, reference, rtol=1e-10, atol=1e-12)
    if args.branching:
        p["branching"] = args.branching + [1] * (len(original_branching) - len(args.branching))
    counts = [1]
    for factor in p["branching"]:
        counts.append(counts[-1] * factor)
    p["expected_nodes"] = sum(counts)
    if args.branching or args.diagonal_costs:
        storage = "diagonal" if args.diagonal_costs else "dense"
        print(f"Generating fixed {counts[-1]}-scenario problem with {storage} costs...", flush=True)
        generate_problem(p, p["phi0"], p["gam"], workspace,
                         diagonal_costs=args.diagonal_costs)
    problem_metadata = json.loads((workspace / "data/data.json").read_text())
    cost_structures = {key: problem_metadata["cost"].get(key, "dense")
                       for key in ("nonleafQ", "nonleafR", "leafQ")}
    if args.diagonal_costs and any(value != "diagonal" for value in cost_structures.values()):
        raise ValueError("Generated cost factors did not retain the requested diagonal structure")
    print(f"Quadratic cost factor storage: {cost_structures}", flush=True)
    online = OnlineModel(p, workspace / "data")
    online.update(p["phi0"], p["gam"])
    online_errors = online.compare_reference()
    startup_generation = time.perf_counter() - startup_begin
    executable = (args.executable or repo / "build/examples/networkedControlMpc/networked_mpc_persistent").resolve()
    metadata = {
        "implementation": "persistent GPU storage; binary dynamics-only updates",
        "anderson": anderson, "projection_mode": args.projections, "cuda_graphs": args.cuda_graphs,
        "diagonal_costs": args.diagonal_costs, "cost_factor_structures": cost_structures,
        "trajectories": args.trajectories, "steps": args.steps, "seed": args.seed,
        "physical_states": p["num_states"], "augmented_states": p["state_size"], "inputs": p["num_inputs"],
        "prediction_transitions": len(p["branching"]), "branching": p["branching"],
        "original_branching": original_branching, "tree_nodes": sum(counts),
        "tree_nonleaf_nodes": sum(counts[:-1]), "tree_scenarios": counts[-1],
        "sampling_time_hours": p["max_delay"], "phi0": p["phi0"],
        "gamma0": p["gam"], "gamma_growth": p["trick"],
        "absolute_tolerance": args.tolerance, "relative_tolerance": args.tolerance,
        "max_solver_time_s": args.max_time, "stored_problem": p["stored"],
        "source_sha256": p["source_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "executable": str(executable),
        "worker_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "online_model_sha256": hashlib.sha256(Path(__file__).with_name("online_model.py").read_bytes()).hexdigest(),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "startup_generation_wall_s": startup_generation,
        "stored_reference_tensor_max_errors": comparisons,
        "initial_online_tensor_max_errors": online_errors,
        "warm_start": "Previous complete unshifted primal-dual iterate retained on device; cold at each trajectory's step zero. Anderson history and scratch buffers reset in place.",
        "timing": "CUDA-event elapsed interval around resident SPOCK solve, including in-place scratch reset and initial-state upload, and host-paced gaps. Numerical generation, binary payload transfer/upload and one-time allocation excluded; separately recorded.",
        "generation": "Full factory called only at startup. Thereafter only changing dynamics and Euclidean dynamics-projection factors are recomputed; static tree, risk, costs, constraints and scaling are reused.",
        "prediction": "Stored Beta law gamma/(1-phi), gamma/phi, with gamma growing by the stored factor at each absolute MPC step; horizon unchanged.",
        "successful_solves": 0, "complete": False,
    }
    np.savez(output / "plant_parameters.npz", Ac=p["Ac"], Bc=p["Bc"], Q=p["Q"], R=p["R"],
             Q_leaf=p["Q_leaf"], initial_state=p["x0"])
    (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    states = np.full((args.trajectories, args.steps + 1, p["state_size"]), np.nan)
    controls = np.full((args.trajectories, args.steps, p["num_inputs"]), np.nan)
    delays = np.full((args.trajectories, args.steps + 1), np.nan)
    worker = subprocess.Popen([str(executable), str(workspace / "data"), str(args.tolerance), str(args.max_time),
                               json.dumps(anderson), "--projections", args.projections, "--cuda-graphs", args.cuda_graphs],
                              cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    columns = ["trajectory", "step", "seed", "warm_start", "converged", "status", "gpu_time_s", "solver_wall_s",
               "setup_wall_s", "generation_wall_s", "update_wall_s", "worker_wall_s", "step_wall_s", "iterations",
               "primal_size", "dual_size", "device_allocated_bytes", "solution_device_pointer", "phi", "next_phi",
               "state_norm_inf", "next_state_norm_inf", "input_norm_inf", "state_violation", "input_violation",
               "aa_fits", "aa_accepted", "aa_filtered_columns", "aa_restarts", "aa_bootstrap_steps",
               "aa_coefficient_rejections", "aa_direction_rejections", "aa_numerical_rejections",
               "aa_last_rank", "aa_last_scale", "aa_last_coefficient_norm", "initial_residual", "final_residual",
               "effective_tolerance", "cp_operator_evaluations", "backtracks"]
    columns += ["cuda_graphs", "graph_nodes", "graph_launches", "graph_setup_wall_s", "dynamics_update_bytes"]
    columns += ["aa_qr_rebuilds", "aa_qr_updates", "aa_qr_reuses", "aa_qr_update_fallbacks",
                "aa_qr_orthogonality_checks", "aa_qr_last_orthogonality_error"]
    try:
        ready = read_worker_response(worker, "initialisation")
        if not ready.get("ready"):
            raise RuntimeError("Persistent GPU worker did not initialise")
        if ready.get("dynamics_update_protocol") != online.protocol:
            raise ValueError("GPU dynamics update protocol differs from the online model; rebuild the worker")
        if ready.get("dynamics_update_bytes") != online.payload_bytes:
            raise ValueError("GPU dynamics payload byte count differs from the online model")
        if ready.get("cuda_graphs") != args.cuda_graphs:
            raise RuntimeError("GPU worker does not support the requested CUDA graph mode; rebuild it")
        if ready.get("projection_mode") != args.projections:
            raise RuntimeError("GPU worker does not support the requested projection mode; rebuild it")
        if args.anderson == "regularized" and ready.get("aa_qr_mode", "full") != args.aa_qr:
            raise ValueError("GPU worker does not support the requested QR mode; rebuild the main executable")
        expected = [(name, list(array.shape[::-1])) for name, array in online.fields.items()]
        if [(entry["name"], entry["shape"]) for entry in ready["fields"]] != expected:
            raise ValueError("GPU dynamics payload layout differs from the online model")
        metadata["worker_initialisation"] = ready
        print(f"GPU storage allocated once: {ready['device_allocated_bytes']/2**30:.3f} GiB of tracked tensors", flush=True)
        with (output / "mpc_gpu_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for path in range(args.trajectories):
                seed = args.seed + path
                rng = np.random.default_rng(seed)
                state, phi = p["x0"].copy(), p["phi0"]
                states[path, 0], delays[path, 0] = state, phi
                for step in range(args.steps):
                    step_begin = time.perf_counter()
                    gamma = p["gam"] * p["trick"] ** step
                    update = bool(path or step)
                    generation = online.update(phi, gamma) if update else 0.0
                    request = {"state": state.tolist(), "warm_start": step > 0, "update": update,
                               "sizes": [array.size for array in online.fields.values()]}
                    worker.stdin.write((json.dumps(request) + "\n").encode())
                    worker.stdin.flush()
                    if update:
                        online.write_payload(worker.stdin)
                    result = read_worker_response(worker, f"trajectory {path}, step {step}")
                    row = {key: result[key] for key in columns if key in result}
                    row.update(trajectory=path, step=step, seed=seed, phi=phi, generation_wall_s=generation,
                               state_norm_inf=float(np.max(np.abs(state[:p["num_states"]]))))
                    if result["converged"]:
                        control = np.asarray(result["control"])
                        next_phi = float(rng.beta(gamma / (1 - phi), gamma / phi)) if 0 < phi < 1 else phi
                        A, B = p["compute_Ai_Bi"](p["Ac"], p["Bc"], p["max_delay"], next_phi * p["max_delay"])
                        next_state = A @ state + B @ control
                        if not np.isfinite(next_state).all():
                            raise RuntimeError("Nonfinite simulated plant state")
                        row.update(next_phi=next_phi, next_state_norm_inf=float(np.max(np.abs(next_state[:p["num_states"]]))),
                                   input_norm_inf=float(np.max(np.abs(control))),
                                   state_violation=float(np.maximum(np.abs(next_state[:p["num_states"]])-p["states_ub"], 0).max()),
                                   input_violation=float(np.maximum(np.abs(control)-p["inputs_ub"], 0).max()))
                        controls[path, step], states[path, step + 1], delays[path, step + 1] = control, next_state, next_phi
                        state, phi = next_state, next_phi
                        metadata["successful_solves"] += 1
                    row["step_wall_s"] = time.perf_counter() - step_begin
                    writer.writerow(row)
                    stream.flush()
                    print(f"trajectory={path:02d} step={step:02d} warm={step > 0} GPU={result['gpu_time_s']:.6f}s "
                          f"generation={generation:.3f}s update={result['update_wall_s']:.3f}s "
                          f"iterations={result['iterations']} converged={result['converged']}", flush=True)
                    if not result["converged"]:
                        raise RuntimeError(f"Nonconverged MPC solve at trajectory {path}, step {step}")
        metadata["complete"] = metadata["successful_solves"] == args.trajectories * args.steps
    finally:
        try:
            worker.stdin.close()
        except BrokenPipeError:
            pass
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=30)
        finally:
            metadata["worker_returncode"] = worker.poll()
            np.savez_compressed(output / "mpc_trajectories.npz", states=states, controls=controls, phi=delays)
            (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if worker.returncode:
        raise RuntimeError(f"Persistent worker exited with code {worker.returncode}")
    print(f"Completed {metadata['successful_solves']} MPC solves; output: {output}", flush=True)


if __name__ == "__main__":
    main()
