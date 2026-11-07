"""Check rollout bookkeeping and independently reconstruct the delayed plant."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    metadata = json.loads((directory / "parameters.json").read_text())
    with (directory / "mpc_gpu_timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    n, steps = metadata["trajectories"], metadata["steps"]
    assert len(rows) == n * steps and metadata["complete"]
    assert len({(int(r["trajectory"]), int(r["step"])) for r in rows}) == n * steps
    assert all(r["converged"] == "True" for r in rows)
    assert sum(r["warm_start"] == "False" for r in rows) == n
    trajectory = np.load(directory / "mpc_trajectories.npz")
    plant = np.load(directory / "plant_parameters.npz")
    x, u, phi = trajectory["states"], trajectory["controls"], trajectory["phi"]
    assert np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(phi).all()
    nx = metadata["physical_states"]
    np.testing.assert_allclose(x[:, 0], np.broadcast_to(plant["initial_state"], (n, x.shape[2])))
    np.testing.assert_allclose(x[:, 1:, nx:], u, rtol=0, atol=1e-12)
    Ac, Bc = plant["Ac"], plant["Bc"]
    a = np.diag(Ac)
    np.testing.assert_array_equal(Ac, np.diag(a))
    T = metadata["sampling_time_hours"]
    maximum = 0.0
    for path in range(n):
        rng = np.random.default_rng(metadata["seed"] + path)
        for k in range(steps):
            current = phi[path, k]
            gamma = metadata["gamma0"] * metadata["gamma_growth"] ** k
            sampled = rng.beta(gamma / (1 - current), gamma / current) if 0 < current < 1 else current
            np.testing.assert_allclose(sampled, phi[path, k + 1], rtol=0, atol=1e-15)
            sigma = T * phi[path, k + 1]
            # Independent exact integration: old input until sigma, new input afterwards.
            old_part = (np.exp(a * T) - np.exp(a * (T - sigma))) / a
            new_part = np.expm1(a * (T - sigma)) / a
            predicted = (np.exp(a * T) * x[path, k, :nx]
                         + old_part * (Bc @ x[path, k, nx:])
                         + new_part * (Bc @ u[path, k]))
            maximum = max(maximum, float(np.max(np.abs(predicted - x[path, k + 1, :nx]))))
    assert maximum < 1e-10, maximum
    assert all(float(row["gpu_time_s"]) > 0 for row in rows)
    report = {"checked_solves": len(rows), "cold_solves": n, "warm_solves": n * (steps - 1),
              "all_converged": True, "independent_dynamics_max_error": maximum,
              "all_checks_passed": True}
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
