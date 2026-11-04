"""Create a clearly labelled snapshot from complete trajectories only."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--trajectories", type=int, required=True)
args = parser.parse_args()
metadata = json.loads((args.source / "parameters.json").read_text())
with (args.source / "mpc_gpu_timings.csv").open() as stream:
    reader = csv.DictReader(stream)
    columns = reader.fieldnames
    rows = [row for row in reader if int(row["trajectory"]) < args.trajectories]
assert len(rows) == args.trajectories * metadata["steps"]
assert all(row["converged"] == "True" for row in rows)
args.destination.mkdir(parents=True, exist_ok=True)
metadata["preview_of_planned_trajectories"] = metadata["trajectories"]
metadata["trajectories"] = args.trajectories
metadata["successful_solves"] = len(rows)
metadata["complete"] = True
(args.destination / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
with (args.destination / "mpc_gpu_timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
