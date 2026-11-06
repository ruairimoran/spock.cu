"""Plot the recorded CUDA-event SPOCK times; no CPU solver comparisons."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    with (directory / "mpc_gpu_timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    params = json.loads((directory / "parameters.json").read_text())
    if len(rows) != params["trajectories"] * params["steps"]:
        raise ValueError("Incomplete experiment: do not plot a partial study as a complete one")
    if any(row["converged"] != "True" for row in rows):
        raise ValueError("The experiment contains nonconverged solves")
    times = np.full((params["trajectories"], params["steps"]), np.nan)
    iterations = np.zeros_like(times)
    for row in rows:
        trajectory, step = int(row["trajectory"]), int(row["step"])
        if (row["warm_start"] == "True") != (step > 0):
            raise ValueError("Incorrect warm-start schedule")
        times[trajectory, step] = float(row["gpu_time_s"])
        iterations[trajectory, step] = int(row["iterations"])
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite timing samples")
    steps = np.arange(params["steps"])
    mean = times.mean(axis=0)
    q95 = np.quantile(times, 0.95, axis=0)
    with (directory / "mpc_gpu_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "samples", "mean_gpu_s", "p95_gpu_s", "min_gpu_s", "max_gpu_s", "mean_iterations"])
        for k in steps:
            writer.writerow([k, params["trajectories"], mean[k], q95[k], times[:, k].min(), times[:, k].max(), iterations[:, k].mean()])
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    ax.plot(steps, mean, color="#1764ab", marker="o", markersize=3.5,
            linewidth=1.8, label="SPOCK: mean")
    ax.plot(steps, q95, color="#d77a13", linestyle="--", linewidth=1.6,
            label="SPOCK: 95th percentile")
    ax.axvspan(-0.35, 0.35, color="0.88", alpha=0.65, zorder=-2)
    ax.set(xlabel="MPC time step", ylabel="GPU SPOCK solve time [s]", xlim=(-0.35, params["steps"] - 0.65))
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0, params["steps"], 2))
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=10)
    ax.set_title(f"Case Study 2: {params['trajectories']} closed-loop trajectories", fontsize=12)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(directory / f"spock_mpc_runtime.{suffix}", dpi=240)
    plt.close(fig)
    aggregate = {
        "solves": len(rows), "trajectories": params["trajectories"], "steps": params["steps"],
        "cold_mean_s": float(times[:, 0].mean()), "cold_p95_s": float(np.quantile(times[:, 0], 0.95)),
        "warm_mean_s": float(times[:, 1:].mean()), "warm_p95_s": float(np.quantile(times[:, 1:], 0.95)),
        "max_state_violation": max(float(row["state_violation"]) for row in rows),
        "max_input_violation": max(float(row["input_violation"]) for row in rows),
    }
    (directory / "statistics.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
