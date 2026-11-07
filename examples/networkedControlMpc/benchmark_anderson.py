"""Replay identical Case Study 2 snapshots with legacy and regularised AA.

Each replay starts cold, so both modes have the same initial vector and
stopping threshold. This isolates the direction algorithm from warm-start
and closed-loop trajectory differences. It is a short diagnostic benchmark.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from online_model import OnlineModel
from run_mpc import read_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--qr-comparison", action="store_true",
                        help="Compare full and updated QR with identical regularisation")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    repo, snapshots = args.repo.resolve(), args.snapshots.resolve()
    metadata = json.loads((snapshots / "parameters.json").read_text())
    p = read_parameters(repo)
    p["branching"] = metadata["branching"]
    data = snapshots / "initial_problem/data"
    online = OnlineModel(p, data)
    paths = np.load(snapshots / "mpc_trajectories.npz")
    workers, results = {}, []
    try:
        variants = ("full", "update") if args.qr_comparison else ("legacy", "regularized")
        for mode in variants:
            settings = {"mode": "regularized" if args.qr_comparison else mode,
                        "qr_mode": mode if args.qr_comparison else "full", "regularization": args.regularization,
                        "rank_tolerance": 1e-8, "max_coefficient_norm": 1e6, "max_direction_ratio": 1e4}
            worker = subprocess.Popen([str(args.worker.resolve()), str(data), "0.001", "90", json.dumps(settings)],
                                      cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            workers[mode] = worker
            ready = json.loads(worker.stdout.readline())
            if not ready.get("ready"):
                raise RuntimeError(f"{mode} worker did not initialise")
        for index, step in enumerate(args.steps):
            state = paths["states"][0, step]
            phi = float(paths["phi"][0, step])
            online.update(phi, p["gam"] * p["trick"] ** step)
            # Alternate order to reduce systematic first-run effects.
            modes = variants if index % 2 == 0 else tuple(reversed(variants))
            pair = []
            for mode in modes:
                worker = workers[mode]
                request = {"state": state.tolist(), "warm_start": False, "update": True,
                           "sizes": [array.size for array in online.fields.values()]}
                worker.stdin.write((json.dumps(request) + "\n").encode())
                worker.stdin.flush()
                online.write_payload(worker.stdin)
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError(f"{mode} worker stopped during replay")
                result = json.loads(line)
                result.update(mode=mode, step=step)
                results.append(result)
                pair.append(result)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"step={step} mode={mode} GPU={result['gpu_time_s']:.6f}s "
                      f"iterations={result['iterations']} CP={result['cp_operator_evaluations']} "
                      f"restarts={result['aa_restarts']} converged={result['converged']}", flush=True)
            np.testing.assert_allclose(pair[0]["initial_residual"], pair[1]["initial_residual"], rtol=1e-12)
            np.testing.assert_allclose(pair[0]["effective_tolerance"], pair[1]["effective_tolerance"], rtol=1e-12)
            for row in pair:
                if row["converged"]:
                    assert row["final_residual"] <= row["effective_tolerance"]
        columns = ["step", "mode", "converged", "gpu_time_s", "iterations", "cp_operator_evaluations", "backtracks",
                   "initial_residual", "final_residual", "effective_tolerance", "aa_fits", "aa_filtered_columns", "aa_restarts",
                   "aa_qr_rebuilds", "aa_qr_updates", "aa_qr_update_fallbacks"]
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    finally:
        for worker in workers.values():
            try:
                worker.stdin.close()
            except BrokenPipeError:
                pass
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=10)


if __name__ == "__main__":
    main()
