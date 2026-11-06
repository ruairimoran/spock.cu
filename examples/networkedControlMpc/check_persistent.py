"""Validate rollout bookkeeping and persistent GPU storage across MPC steps."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name("check_mpc.py")), str(args.directory)], check=True)
    rows = list(csv.DictReader((args.directory / "mpc_gpu_timings.csv").open()))
    sizes = {int(row["device_allocated_bytes"]) for row in rows}
    pointers = {int(row["solution_device_pointer"]) for row in rows}
    assert len(sizes) == 1 and next(iter(sizes)) > 0, "GPU tensor storage size changed"
    assert len(pointers) == 1 and next(iter(pointers)) != 0, "Solution storage moved"
    assert float(rows[0]["setup_wall_s"]) > 0
    assert all(float(row["setup_wall_s"]) == 0 for row in rows[1:]), "GPU objects were rebuilt"
    report = {"checked_solves": len(rows), "constant_solution_pointer": True,
              "constant_tracked_device_bytes": next(iter(sizes)), "one_time_gpu_setup": True,
              "all_checks_passed": True,
              "scope": "Checks stable storage across steps. CUDA API tracing is needed to detect transient allocator calls."}
    (args.directory / "validation_persistent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
