"""Exercise regularised Anderson in a small persistent scenario-tree solver.

The requests use prescribed states, not a simulated plant trajectory.  This
keeps legacy, regularised and forced-rejection runs on matching problems and
tests cold/warm resets independently of feedback divergence.  GPU execution
must be scheduled separately from any timing experiment using the same GPU.
"""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import spock as s

from test_anderson_reference import check_mpc_counters


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def generate_problem(directory):
    """Create a stable two-state plant augmented with its previous scalar input."""
    directory.mkdir()
    branching = [2, 2, 1, 1]
    stages, ancestors, probabilities, values = [0], [-1], [1.0], [0.0]
    current = [0]
    for stage, factor in enumerate(branching):
        children = []
        for parent in current:
            for branch in range(factor):
                children.append(len(stages))
                stages.append(stage + 1)
                ancestors.append(parent)
                probabilities.append(probabilities[parent] / factor)
                values.append(0.05 * (branch + 1))
        current = children
    with working_directory(directory), contextlib.redirect_stdout(io.StringIO()):
        tree = s.tree.FromStructure(stages, ancestors, probabilities, np.asarray(values)).build()
        dynamics, costs = [None], [None]
        Q, R = np.diag([1.0, 0.7, 0.02]), np.array([[0.15]])
        for value in values[1:]:
            A = np.array([[0.85, 0.0, 0.03 * value],
                          [0.0, 0.91, 0.02 * value],
                          [0.0, 0.0, 0.0]])
            B = np.array([[0.15 * (1 - value)], [0.11 * (1 - value)], [1.0]])
            dynamics.append(s.build.Dynamics(A, B))
            costs.append(s.build.CostQuadratic(Q, R))
        nonleaf = s.build.Rectangle(np.array([-2.0, -2.0, -0.8, -0.8]),
                                    np.array([2.0, 2.0, 0.8, 0.8]))
        leaf = s.build.Rectangle(np.array([-2.0, -2.0, -0.8]), np.array([2.0, 2.0, 0.8]))
        problem = (s.problem.Factory(tree, 3, 1).with_dynamics_list(dynamics)
                   .with_cost_nonleaf_list(costs)
                   .with_cost_leaf(s.build.CostQuadratic(np.diag([3.0, 2.0, 0.02]), leaf=True))
                   .with_constraint_nonleaf(nonleaf).with_constraint_leaf(leaf)
                   .with_risk(s.build.AVaR(0.95)).with_preconditioning(True).generate_problem())
        del problem
    return directory / "data"


def run_requests(worker_path, data, settings, states, tolerance, max_time):
    """Use one process and one allocation for the whole request sequence."""
    command = [str(worker_path), str(data), str(tolerance), str(max_time), json.dumps(settings)]
    worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    responses = []
    try:
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("Persistent worker exited before its ready response")
        ready = json.loads(line)
        if not ready.get("ready"):
            raise RuntimeError(f"Unexpected worker startup response: {ready}")
        for index, state in enumerate(states):
            request = {"state": state, "warm_start": index % 2 == 1, "update": False, "sizes": []}
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(f"Worker stopped before response {index}")
            response = json.loads(line)
            if not response["converged"]:
                raise AssertionError(f"{settings['mode']} request {index} did not converge: {response}")
            if response["warm_start"] != request["warm_start"]:
                raise AssertionError("Worker used the wrong warm-start mode")
            control = np.asarray(response["control"])
            if control.shape != (1,) or not np.isfinite(control).all() or np.max(np.abs(control)) > 0.8001:
                raise AssertionError("Worker returned an invalid root control")
            responses.append(response)
    finally:
        worker.stdin.close()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=10)
    if worker.returncode:
        raise RuntimeError(f"Persistent worker exited with code {worker.returncode}")
    return ready, responses


def save_responses(path, responses):
    # Scalar diagnostics form a CSV readable by the independent counter check.
    columns = [key for key, value in responses[0].items() if not isinstance(value, (list, dict))]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request"] + columns)
        writer.writeheader()
        for index, response in enumerate(responses):
            writer.writerow({"request": index, **{key: response[key] for key in columns}})


def check_storage(ready, responses):
    assert len({row["solution_device_pointer"] for row in responses}) == 1, "Solution storage moved"
    assert {row["device_allocated_bytes"] for row in responses} == {ready["device_allocated_bytes"]}
    assert responses[0]["setup_wall_s"] > 0
    assert all(row["setup_wall_s"] == 0 for row in responses[1:]), "GPU objects were rebuilt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help="Persistent CUDA worker built with Anderson diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--generate-only", action="store_true", help="Generate the CPU test problem without using a GPU")
    args = parser.parse_args()
    if args.tolerance <= 0 or args.max_time <= 0:
        parser.error("Tolerance and time limit must be positive")
    if not args.generate_only and args.worker is None:
        parser.error("--worker is required unless --generate-only is selected")
    output = args.output.resolve()
    output.mkdir(parents=True)  # Require a fresh directory.
    data = generate_problem(output / "problem")
    if args.generate_only:
        print(json.dumps({"data": str(data), "gpu_used": False}, indent=2))
        return

    worker = args.worker.resolve()
    x0, x1 = [0.65, -0.35, 0.0], [0.45, -0.25, -0.1]
    sequence = [x0, x1, x0, x1]
    regularized = {"mode": "regularized", "regularization": 1e-8, "rank_tolerance": 1e-8,
                   "max_coefficient_norm": 1e6, "max_direction_ratio": 1e4}
    reports, controls = {}, {}
    for name, settings, states in (
            ("legacy", {"mode": "legacy"}, sequence),
            ("regularized", regularized, sequence),
            ("qr_updates", {**regularized, "qr_mode": "update"}, sequence),
            ("forced_rejection", {**regularized, "qr_mode": "update", "max_coefficient_norm": 1e-12}, sequence[:2])):
        ready, responses = run_requests(worker, data, settings, states, args.tolerance, args.max_time)
        check_storage(ready, responses)
        csv_path = output / f"{name}.csv"
        save_responses(csv_path, responses)
        counter_report = check_mpc_counters(csv_path) if name != "legacy" else None
        if name == "forced_rejection":
            assert sum(row["aa_coefficient_rejections"] for row in responses) > 0, "Safety rejection was not exercised"
            assert sum(row["aa_restarts"] for row in responses) > 0
        control = np.asarray([row["control"] for row in responses])
        if len(responses) == 4:
            np.testing.assert_allclose(control[:2], control[2:], rtol=0, atol=5e-5,
                                       err_msg=f"Repeated cold/warm sequence changed controls in {name}")
        controls[name] = control
        reports[name] = {"settings": settings, "ready": ready, "responses": responses,
                         "counter_checks": counter_report, "constant_storage": True}
        print(f"{name}: {len(responses)} converged requests, storage and counter checks passed", flush=True)
    normal_difference = float(np.max(np.abs(controls["regularized"] - controls["legacy"])))
    forced_difference = float(np.max(np.abs(controls["forced_rejection"] - controls["regularized"][:2])))
    qr_difference = float(np.max(np.abs(controls["qr_updates"] - controls["regularized"])))
    np.testing.assert_allclose(controls["regularized"], controls["legacy"], rtol=0, atol=5e-5)
    np.testing.assert_allclose(controls["forced_rejection"], controls["regularized"][:2], rtol=0, atol=5e-5)
    np.testing.assert_allclose(controls["qr_updates"], controls["regularized"], rtol=0, atol=5e-5)
    report = {"all_checks_passed": True, "requests_checked": 14,
              "problem": {"states": 3, "inputs": 1, "branching": [2, 2, 1, 1]},
              "sequence": "prescribed cold x0, warm x1, cold x0, warm x1; not a plant rollout",
              "normal_control_max_difference": normal_difference,
              "updated_qr_control_max_difference": qr_difference,
              "forced_rejection_control_max_difference": forced_difference,
              "modes": reports}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "modes"}, indent=2))


if __name__ == "__main__":
    main()
