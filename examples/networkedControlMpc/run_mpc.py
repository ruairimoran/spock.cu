"""Case Study 2 MPC, retaining the existing example's parameters and topology."""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
from scipy import stats
from scipy.linalg import expm, inv
import spock as s
from gputils_api import read_array_from_gputils_binary_file as read_bt


def read_parameters(repo):
    source_path = repo / "examples/networkedControl/main.py"
    source = source_path.read_text()
    syntax = ast.parse(source)
    namespace = {"np": np, "stats": stats, "expm": expm, "inv": inv}
    # Execute the original helper functions and numerical parameter assignments,
    # without running its file-generating top-level benchmark.
    definitions = [n for n in syntax.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"), namespace)
    wanted = {
        "max_delay", "rng", "num_states", "num_inputs", "state_size", "Ac", "Bc",
        "zero", "Q", "R", "Q_leaf", "states_ub", "states_lb", "inputs_ub", "inputs_lb", "alpha",
    }
    for node in syntax.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    for node in syntax.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "make_tree":
            for assignment in node.body:
                if isinstance(assignment, ast.Assign):
                    for target in assignment.targets:
                        if isinstance(target, ast.Name) and target.id in {"gam", "s0"}:
                            namespace[target.id] = ast.literal_eval(assignment.value)
    calls = [n for n in ast.walk(syntax) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "make_beta_tree"]
    namespace["trick"] = next(ast.literal_eval(k.value) for k in calls[0].keywords if k.arg == "trick_")
    data = repo / "examples/networkedControl/data"
    stored = json.loads((data / "data.json").read_text())
    stages = read_bt(str(data / "stages_u.bt"), dt="uint64").ravel().astype(int)
    children = read_bt(str(data / "numChildren_u.bt"), dt="uint64").ravel().astype(int)
    branching = []
    for stage in range(stored["numStages"] - 1):
        counts = np.unique(children[np.flatnonzero(stages[:len(children)] == stage)])
        if len(counts) != 1:
            raise ValueError("This runner expects the example's stage-wise regular branching")
        branching.append(int(counts[0]))
    with (repo / "examples/networkedControl/tree.pkl").open("rb") as stream:
        saved_tree = pickle.load(stream)
    namespace["phi0"] = float(saved_tree.data_values[0][0] / namespace["max_delay"])
    namespace["x0"] = read_bt(str(data / "initialState_d.bt")).ravel()
    namespace["branching"] = branching
    namespace["stored"] = stored
    namespace["source_path"] = str(source_path)
    namespace["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    if namespace["state_size"] != stored["numStates"] or namespace["num_inputs"] != stored["numInputs"]:
        raise ValueError("Stored dimensions disagree with the example source")
    if not np.isclose(namespace["alpha"], stored["risk"]["alpha"]):
        raise ValueError("Stored risk parameter disagrees with the example source")
    return namespace


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def generate_problem(p, phi, gamma, workspace):
    begin = time.perf_counter()
    with working_directory(workspace), contextlib.redirect_stdout(io.StringIO()):
        stages, ancestors, probabilities, values = p["make_beta_tree"](
            gamma, phi, p["branching"], trick_=p["trick"], plot=False)
        delays = np.asarray(values) * p["max_delay"]
        tree = s.tree.FromStructure(stages, ancestors, probabilities, delays).build()
        dynamics = [None]
        costs = [None]
        for node in range(1, tree.num_nodes):
            A, B = p["compute_Ai_Bi"](p["Ac"], p["Bc"], p["max_delay"], float(delays[node]))
            dynamics.append(s.build.Dynamics(A, B))
            costs.append(s.build.CostQuadratic(p["Q"], p["R"]))
        nonleaf = s.build.Rectangle(
            np.hstack((p["states_lb"], p["inputs_lb"], p["inputs_lb"])),
            np.hstack((p["states_ub"], p["inputs_ub"], p["inputs_ub"])))
        leaf = s.build.Rectangle(np.hstack((p["states_lb"], p["inputs_lb"])),
                                 np.hstack((p["states_ub"], p["inputs_ub"])))
        problem = (s.problem.Factory(tree, p["state_size"], p["num_inputs"])
                   .with_dynamics_list(dynamics).with_cost_nonleaf_list(costs)
                   .with_cost_leaf(s.build.CostQuadratic(p["Q_leaf"], leaf=True))
                   .with_constraint_nonleaf(nonleaf).with_constraint_leaf(leaf)
                   .with_risk(s.build.AVaR(p["alpha"]))
                   .with_preconditioning(p["stored"]["preconditioned"]).generate_problem())
        if tree.num_nodes != p["stored"]["numNodes"]:
            raise ValueError("Tree topology changed")
        del problem
    return time.perf_counter() - begin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=15)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-time", type=float, default=300.0)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "mpc_gpu_timings.csv").exists():
        raise FileExistsError("Choose a fresh output directory to preserve existing experiment data")
    workspace = output / "active_problem"
    workspace.mkdir()
    p = read_parameters(repo)
    base_generation = generate_problem(p, p["phi0"], p["gam"], workspace)
    # Check that the new step-zero model reproduces the saved GPU problem.
    comparisons = {}
    for name in ("dynamics_AB", "scaling", "nonleafCost_sqrtQ", "nonleafCost_sqrtR", "leafCost_sqrtQ"):
        reference = read_bt(str(repo / "examples/networkedControl/data" / (name + "_d.bt")))
        generated = read_bt(str(workspace / "data" / (name + "_d.bt")))
        error = float(np.max(np.abs(reference - generated)))
        comparisons[name] = error
        np.testing.assert_allclose(generated, reference, rtol=1e-10, atol=1e-12)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    metadata = {
        "trajectories": args.trajectories, "steps": args.steps, "seed": args.seed,
        "gpu": gpu, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_path": p["source_path"], "source_sha256": p["source_sha256"],
        "physical_states": p["num_states"], "augmented_states": p["state_size"], "inputs": p["num_inputs"],
        "prediction_transitions": len(p["branching"]), "branching": p["branching"], "stored_problem": p["stored"],
        "sampling_time_hours": p["max_delay"], "phi0": p["phi0"], "gamma0": p["gam"], "gamma_growth": p["trick"],
        "absolute_tolerance": args.tolerance, "relative_tolerance": args.tolerance,
        "max_solver_time_s": args.max_time, "step_zero_tensor_max_errors": comparisons,
        "warm_start": "Previous complete unshifted primal-dual iterate; cold at step zero of every trajectory.",
        "timing": "CUDA-event elapsed interval around runSpock, including its GPU transfers and host-paced gaps; not a sum of kernel durations. CPU generation, problem/cache setup, extraction and plant simulation excluded.",
        "prediction": "Tree reconditioned on current phi at every MPC step, preserving stored topology and Beta parameters. Gamma at absolute step k is gamma0 * gamma_growth**k.",
        "plant": "Sample next phi from Beta(gamma_k/(1-phi_k), gamma_k/phi_k), then propagate the stored edge-based A(T*phi_next), B(T*phi_next) with the root control.",
        "retained_source_details": "Existing br=0 stored tree has 47 transitions (48 stages), although source variable horizon=48. No horizon, risk, cost, or distribution corrections applied.",
    }
    (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez(output / "plant_parameters.npz", Ac=p["Ac"], Bc=p["Bc"], Q=p["Q"], R=p["R"],
             Q_leaf=p["Q_leaf"], initial_state=p["x0"])
    executable = repo / "build/examples/networkedControlMpc/networked_mpc"
    worker = subprocess.Popen([str(executable), str(workspace / "data")], cwd=repo,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    columns = ["trajectory", "step", "seed", "warm_start", "converged", "status", "gpu_time_s", "solver_wall_s",
               "setup_wall_s", "generation_wall_s", "iterations", "primal_size", "dual_size", "phi", "next_phi",
               "state_norm_inf", "next_state_norm_inf", "input_norm_inf", "state_violation", "input_violation"]
    state_history = np.full((args.trajectories, args.steps + 1, p["state_size"]), np.nan)
    input_history = np.full((args.trajectories, args.steps, p["num_inputs"]), np.nan)
    delay_history = np.full((args.trajectories, args.steps + 1), np.nan)
    successful = 0
    try:
        with (output / "mpc_gpu_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            stream.flush()
            for trajectory in range(args.trajectories):
                seed = args.seed + trajectory
                rng = np.random.default_rng(seed)
                state = p["x0"].copy()
                phi = p["phi0"]
                state_history[trajectory, 0] = state
                delay_history[trajectory, 0] = phi
                for step in range(args.steps):
                    gamma = p["gam"] * p["trick"] ** step
                    generation = generate_problem(p, phi, gamma, workspace) if trajectory or step else base_generation
                    request = {"state": state.tolist(), "warm_start": step > 0,
                               "tolerance": args.tolerance, "max_time_s": args.max_time}
                    worker.stdin.write(json.dumps(request) + "\n")
                    worker.stdin.flush()
                    line = worker.stdout.readline()
                    if not line:
                        raise RuntimeError(f"GPU worker exited with code {worker.poll()}")
                    result = json.loads(line)
                    row = {key: result[key] for key in columns if key in result}
                    row.update(trajectory=trajectory, step=step, seed=seed, phi=phi,
                               generation_wall_s=generation, state_norm_inf=float(np.max(np.abs(state[:p["num_states"]]))))
                    if result["converged"]:
                        control = np.asarray(result["control"])
                        next_phi = float(rng.beta(gamma / (1 - phi), gamma / phi)) if 0 < phi < 1 else phi
                        A, B = p["compute_Ai_Bi"](p["Ac"], p["Bc"], p["max_delay"], next_phi * p["max_delay"])
                        next_state = A @ state + B @ control
                        if not np.isfinite(next_state).all():
                            raise RuntimeError("Nonfinite simulated plant state")
                        row.update(next_phi=next_phi, next_state_norm_inf=float(np.max(np.abs(next_state[:p["num_states"]]))),
                                   input_norm_inf=float(np.max(np.abs(control))),
                                   state_violation=float(np.maximum(np.abs(next_state[:p["num_states"]]) - p["states_ub"], 0).max()),
                                   input_violation=float(np.maximum(np.abs(control) - p["inputs_ub"], 0).max()))
                        input_history[trajectory, step] = control
                        state_history[trajectory, step + 1] = next_state
                        delay_history[trajectory, step + 1] = next_phi
                        state, phi = next_state, next_phi
                        successful += 1
                    writer.writerow(row)
                    stream.flush()
                    print(f"trajectory={trajectory:02d} step={step:02d} warm={step > 0} "
                          f"GPU={result['gpu_time_s']:.6f}s iterations={result['iterations']} "
                          f"converged={result['converged']}", flush=True)
                    if not result["converged"]:
                        raise RuntimeError(f"Nonconverged MPC solve at trajectory {trajectory}, step {step}")
    finally:
        worker.stdin.close()
        worker.wait(timeout=30)
        np.savez_compressed(output / "mpc_trajectories.npz", states=state_history,
                            controls=input_history, phi=delay_history)
    metadata["successful_solves"] = successful
    metadata["complete"] = successful == args.trajectories * args.steps
    (output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {successful} MPC solves; output: {output}", flush=True)


if __name__ == "__main__":
    main()
