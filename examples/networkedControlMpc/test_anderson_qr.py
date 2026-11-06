"""Check rolling QR against a direct tall-matrix SVD at every history change.

Sequences exercise repeated deletion/insertion, periodic rebuilds, missing
history versions, reset/reuse, deficient histories, scaling and repair of a
deliberately damaged basis. No production QR algebra is used by the oracle.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from test_anderson_reference import Options, reference_step, scaled_norm


def sequences(single):
    rng = np.random.default_rng(20261104)
    dtype = np.float32 if single else np.float64
    rank_tolerance = 1e-4 if single else 1e-8
    cases = []
    specifications = [("rolling", 3, 140, 1.0), ("one_column", 1, 30, 1.0),
                      ("wide_history", 8, 75, 1.0), ("tiny", 3, 20, 1e-12 if single else 1e-120),
                      ("large", 3, 20, 1e12 if single else 1e120), ("dependent", 3, 25, 1.0),
                      ("correlated", 3, 75, 1.0), ("reset_reuse_gap", 3, 30, 1.0),
                      ("almost_dependent", 3, 20, 1.0), ("scale_change", 3, 20, 1.0),
                      ("ill_conditioned", 3, 40, 1.0),
                      ("repair_basis", 3, 12, 1.0), ("rejection", 3, 12, 1.0)]
    for name, memory, steps, scale in specifications:
        rows = 67
        Y = rng.normal(size=(rows, memory))
        frames = []
        for step in range(steps):
            reuse = name == "reset_reuse_gap" and step in (5, 12)
            if not reuse:
                newest = rng.normal(size=rows)
                if name == "correlated":
                    newest = 0.98 * Y[:, 0] + 0.02 * newest
                if name == "ill_conditioned":
                    newest = Y[:, 0] + (2e-3 if single else 1e-7) * newest
                if name == "dependent" and 3 <= step <= 6:
                    newest = Y[:, 0].copy()
                if name == "dependent" and 12 <= step <= 16:
                    newest.fill(0)
                if name == "almost_dependent" and 2 <= step <= 6:
                    newest = Y[:, 0] + (1e-6 if single else 1e-10) * newest
                if name == "scale_change":
                    extreme = 12 if single else 120
                    newest *= 10.0 ** ([0, -extreme, extreme, 0][(step // 5) % 4])
                Y = np.column_stack((newest, Y[:, :memory - 1]))
            y = np.asarray(Y * scale, dtype=dtype).astype(float)
            noise_scale = float(np.max(np.abs(Y))) if name == "scale_change" else 1.0
            c = np.asarray((0.2 * Y + 0.01 * noise_scale * rng.normal(size=Y.shape)) * scale, dtype=dtype).astype(float)
            residual = np.asarray((0.7 * Y[:, 0] + 0.3 * noise_scale * rng.normal(size=rows)) * scale, dtype=dtype).astype(float)
            frame = {"Y": y.tolist(), "C": c.tolist(), "residual": residual.tolist()}
            if reuse:
                frame["history_changes"] = 0
            if name == "reset_reuse_gap" and step == 18:
                frame["history_changes"] = 3
            if name == "reset_reuse_gap" and step == 24:
                frame["reset"] = True
            if name == "repair_basis" and step == 5:
                frame["damage_basis"] = True
            frames.append(frame)
        case = {"name": name, "frames": frames, "rank_tolerance": rank_tolerance,
                "rebuild_interval": 64, "check_interval": 1 if name == "repair_basis" else 16}
        if name == "rejection":
            case["max_coefficient_norm"] = 1e-12
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--float", dest="single", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = sequences(args.single)
    payload = "".join(json.dumps(case, allow_nan=False) + "\n" for case in cases)
    command = [str(args.executable.resolve())] + (["--float"] if args.single else [])
    completed = subprocess.run(command, input=payload, text=True, capture_output=True, check=True)
    actual = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(actual) == len(cases)
    checks = []
    for case, output in zip(cases, actual):
        assert case["name"] == output["name"] and len(case["frames"]) == len(output["frames"])
        maximum_direction_error = maximum_orthogonality_error = maximum_coefficient_error = 0.
        for index, (frame, result) in enumerate(zip(case["frames"], output["frames"])):
            options = Options(rank_tolerance=case["rank_tolerance"],
                              max_coefficient_norm=case.get("max_coefficient_norm", 1e6))
            Y, C, residual = np.asarray(frame["Y"]), np.asarray(frame["C"]), np.asarray(frame["residual"])
            expected = reference_step(C, Y, residual, options)
            context = f"{case['name']} frame {index}"
            assert result["status"] == expected["status"], (context, result["status"], expected["status"])
            assert result["accepted_indices"] == expected["accepted_indices"], context
            gamma = np.asarray(result["gamma"])
            coefficient_error = scaled_norm(gamma - expected["gamma"]) / max(1., scaled_norm(expected["gamma"]))
            maximum_coefficient_error = max(maximum_coefficient_error, coefficient_error)
            if args.single:
                # Nearly dependent columns produce cancelling coefficients.
                # Their vector error remains meaningful when a particular
                # coefficient is small relative to the full solution norm.
                assert coefficient_error < 5e-4, (context, coefficient_error)
            else:
                np.testing.assert_allclose(gamma, expected["gamma"], rtol=2e-7, atol=2e-9, err_msg=context)
            omitted = sorted(set(range(Y.shape[1])) - set(expected["accepted_indices"]))
            np.testing.assert_array_equal(gamma[omitted], 0)
            error = scaled_norm(np.asarray(result["direction"]) - expected["direction"])
            error /= max(scaled_norm(expected["direction"]), scaled_norm(residual))
            maximum_direction_error = max(maximum_direction_error, error)
            assert error < (5e-4 if args.single else 2e-8), (context, error)
            Q = np.asarray(result["Q_columns"]).reshape(Y.shape, order="F")
            orthogonality = float(np.max(np.abs(Q.T @ Q - np.eye(Y.shape[1]))))
            maximum_orthogonality_error = max(maximum_orthogonality_error, orthogonality)
            assert orthogonality < (3e-4 if args.single else 5e-10), (context, orthogonality)
        stats = output["frames"][-1]["qr"]
        if case["name"] in ("rolling", "wide_history"):
            assert stats["updates"] > 60 and stats["rebuilds"] >= 2, stats
        if case["name"] == "repair_basis":
            assert stats["update_fallbacks"] > 0 and stats["orthogonality_checks"] > 0, stats
        if case["name"] == "rejection":
            assert stats["rebuilds"] == len(case["frames"]) and stats["updates"] == 0, stats
        if case["name"] == "reset_reuse_gap":
            frames = output["frames"]
            assert frames[12]["qr"]["reuses"] == 2
            assert frames[18]["qr"]["rebuilds"] == frames[17]["qr"]["rebuilds"] + 1
            assert frames[24]["qr"]["rebuilds"] == 1 and frames[24]["qr"]["updates"] == 0
        checks.append({"name": case["name"], "frames": len(case["frames"]), "qr": stats,
                       "max_relative_direction_error": maximum_direction_error,
                       "max_relative_coefficient_error": maximum_coefficient_error,
                       "max_orthogonality_error": maximum_orthogonality_error})
    report = {"all_checks_passed": True, "precision": "float" if args.single else "double",
              "sequence_count": len(cases), "frames_checked": sum(len(c["frames"]) for c in cases),
              "sequences": checks}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
