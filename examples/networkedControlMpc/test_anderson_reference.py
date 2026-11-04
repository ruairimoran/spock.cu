"""Independent reference and fixtures for regularised type-II Anderson steps.

The reference works directly with the tall residual history Y.  Production
code may compress it using QR, but this oracle deliberately does not share
that path.  Columns are ordered newest first, and C contains the matching
step-minus-residual differences, so the proposed direction is -r - C @ gamma.

Run this script with the project's NumPy-enabled Python interpreter.  The
optional CUDA adapter accepts one fixture per JSON line and returns one
diagnostic object per line.  It must exercise the production implementation.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np


@dataclass(frozen=True)
class Options:
    regularization: float = 1e-8
    rank_tolerance: float = 1e-8
    max_coefficient_norm: float = 1e6
    max_direction_ratio: float = 1e4

    def validate(self):
        values = tuple(asdict(self).values())
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Anderson options must be finite")
        if self.regularization <= 0:
            raise ValueError("Enabled regularisation must be positive")
        if not 0 <= self.rank_tolerance < 1:
            raise ValueError("Rank tolerance must lie in [0, 1)")
        if self.max_coefficient_norm <= 0 or self.max_direction_ratio <= 0:
            raise ValueError("Safety limits must be positive")


def scaled_norm(vector):
    """Compute a Euclidean norm without squaring very large or small entries."""
    vector = np.asarray(vector)
    if vector.size == 0:
        return 0.0
    largest = float(np.max(np.abs(vector)))
    if largest == 0 or not np.isfinite(largest):
        return largest
    return largest * float(np.linalg.norm(vector / largest))


def reference_step(C, Y, residual, options=Options()):
    """Return a filtered ridge fit and the resulting safeguarded direction.

    Scale is the largest history-column norm.  Filtering uses two passes of
    modified Gram--Schmidt on Y/scale, in newest-first order.  The ridge
    parameter belongs to this normalised problem; dropped coefficients are
    exactly zero.  A failed safety check zeros all applied coefficients.
    Rank, accepted indices and coefficient norm retain fit diagnostics even
    when the proposed direction is rejected.
    """
    options.validate()
    C = np.asarray(C, dtype=float)
    Y = np.asarray(Y, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if C.ndim != 2 or C.shape != Y.shape or residual.shape != (Y.shape[0],):
        raise ValueError("Incompatible paired histories or residual shape")
    if not np.isfinite(residual).all():
        raise ValueError("The current residual must be finite")

    count = Y.shape[1]
    result = {
        "direction": -residual.copy(),
        "gamma": np.zeros(count),
        "accepted_indices": [],
        "rank": 0,
        "scale": 0.0,
        "regularization": options.regularization,
        "coefficient_norm": 0.0,
        "status": "empty_history",
    }
    if count == 0:
        return result
    if not np.isfinite(C).all() or not np.isfinite(Y).all():
        result["status"] = "nonfinite_history"
        return result

    scale = max(scaled_norm(Y[:, j]) for j in range(count))
    if not np.isfinite(scale):
        result["status"] = "nonfinite_history"
        return result
    result["scale"] = scale
    if scale == 0:
        result["status"] = "rank_zero"
        return result

    normalised = Y / scale
    basis, accepted = [], []
    for j in range(count):
        innovation = normalised[:, j].copy()
        # The second pass protects the rank decision when columns nearly align.
        for _ in range(2):
            for q in basis:
                innovation -= np.dot(q, innovation) * q
        length = scaled_norm(innovation)
        if length > options.rank_tolerance:
            accepted.append(j)
            basis.append(innovation / length)
    result["accepted_indices"] = accepted
    result["rank"] = len(accepted)
    if not accepted:
        result["status"] = "rank_zero"
        return result

    # Full tall-matrix SVD is the independent oracle for the compressed solve.
    U, singular_values, Vh = np.linalg.svd(normalised[:, accepted], full_matrices=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rhs = U.T @ (residual / scale)
        gains = singular_values / (singular_values**2 + options.regularization)
        selected_gamma = Vh.T @ (gains * rhs)
    if not np.isfinite(selected_gamma).all():
        result["status"] = "nonfinite_solution"
        return result

    coefficient_norm = scaled_norm(selected_gamma)
    result["coefficient_norm"] = coefficient_norm
    if coefficient_norm > options.max_coefficient_norm:
        result["status"] = "coefficient_limit"
        return result
    gamma = np.zeros(count)
    gamma[accepted] = selected_gamma
    with np.errstate(over="ignore", invalid="ignore"):
        direction = -residual - C @ gamma
    if not np.isfinite(direction).all():
        result["status"] = "nonfinite_solution"
        return result
    # Compare a ratio rather than multiplying a possibly very large norm.
    residual_norm, direction_norm = scaled_norm(residual), scaled_norm(direction)
    ratio = direction_norm / residual_norm if residual_norm else (0.0 if direction_norm == 0 else np.inf)
    if ratio > options.max_direction_ratio:
        result["status"] = "direction_limit"
        return result
    result.update(direction=direction, gamma=gamma, status="accepted")
    return result


@dataclass
class Fixture:
    name: str
    C: np.ndarray
    Y: np.ndarray
    residual: np.ndarray
    options: Options = Options()
    inject_nonfinite: dict | None = None

    def json_object(self):
        result = {"name": self.name, "C": self.C.tolist(), "Y": self.Y.tolist(),
                  "residual": self.residual.tolist(), "options": asdict(self.options)}
        if self.inject_nonfinite:
            # Keep JSON numeric fields finite; the adapter injects this value
            # after parsing, immediately before exercising production code.
            result["inject_nonfinite"] = self.inject_nonfinite
        return result

    def reference_result(self):
        if not self.inject_nonfinite:
            return reference_step(self.C, self.Y, self.residual, self.options)
        C, Y = self.C.copy(), self.Y.copy()
        marker = self.inject_nonfinite
        matrix = C if marker["target"] == "C" else Y
        matrix[marker["row"], marker["column"]] = {"nan": np.nan, "inf": np.inf}[marker["value"]]
        return reference_step(C, Y, self.residual, self.options)


def fixtures():
    """Small deterministic cases, including histories with distinct C columns."""
    rng = np.random.default_rng(20481209)
    rows = 32
    Q, _ = np.linalg.qr(rng.normal(size=(rows, 6)))
    u, v, w = Q[:, :3].T
    Y = np.column_stack((2 * u, 0.7 * v, 0.2 * w))
    C = 0.1 * rng.normal(size=Y.shape)
    residual = Q[:, :4] @ np.array([0.8, -0.4, 0.2, 0.3])
    base = Fixture("full_rank", C, Y, residual)
    cases = [
        base,
        Fixture("one_column", C[:, :1], Y[:, :1], residual),
        Fixture("two_columns", C[:, :2], Y[:, :2], residual),
        Fixture("empty_history", C[:, :0], Y[:, :0], residual),
        Fixture("zero_history", C, np.zeros_like(Y), residual),
        Fixture("exact_dependence", C, np.column_stack((u, 2 * u, v)), residual),
        Fixture("rank_one_history", C, np.column_stack((u, 2 * u, -u)), residual),
        Fixture("span_dependence", C, np.column_stack((u, v, u + v)), residual),
        Fixture("near_dependence", C, np.column_stack((u, u + 1e-12 * v, w)), residual),
        Fixture("weak_first_column", C, np.column_stack((1e-12 * u, v, w)), residual),
        Fixture("independent_small_innovation", C, np.column_stack((u, u + 1e-5 * v, w)), residual),
        Fixture("zero_residual", C, Y, np.zeros(rows)),
        Fixture("coefficient_rejection", C, Y, 100 * residual,
                replace(Options(), max_coefficient_norm=0.1)),
        Fixture("direction_rejection", C, Y, residual,
                replace(Options(), max_direction_ratio=0.1)),
        Fixture("strong_regularization", C, Y, residual,
                replace(Options(), regularization=1e4)),
    ]
    permutation = np.array([2, 0, 1])
    cases.append(Fixture("paired_column_permutation", C[:, permutation], Y[:, permutation], residual))
    # Equal residual columns but distinct C columns expose incorrect column pairing.
    cases.append(Fixture("dependent_newest_preferred", C, np.column_stack((u, u, v)), residual))
    for factor, name in ((1e-120, "tiny_scale"), (1e120, "large_scale")):
        cases.append(Fixture(name, factor * C, factor * Y, factor * residual))
    return cases


def fixtures_for_precision(precision="double", include_nonfinite=False):
    """Quantise float inputs before evaluating their independent double oracle.

    Float fixtures use the production single-precision rank cutoff and scales
    within float's exponent range.  The 1e±120 double cases remain unchanged.
    """
    cases = fixtures()
    if precision == "float":
        base = cases[0]
        converted = []
        for f in cases:
            C, Y, residual = f.C, f.Y, f.residual
            if f.name in ("tiny_scale", "large_scale"):
                factor = 1e-12 if f.name == "tiny_scale" else 1e12
                C, Y, residual = factor * base.C, factor * base.Y, factor * base.residual
            options = replace(f.options, rank_tolerance=1e-4)
            converted.append(Fixture(f.name, *(np.asarray(a, dtype=np.float32).astype(float)
                                               for a in (C, Y, residual)), options))
        # Above the float cutoff, but still noticeably less well conditioned
        # than the full-rank baseline, to exercise single-precision ridge fits.
        u = base.Y[:, 0] / 2
        v = base.Y[:, 1] / 0.7
        w = base.Y[:, 2] / 0.2
        float_case = Fixture("float_independent_innovation", base.C,
                             np.column_stack((u, u + 1e-2 * v, w)), base.residual,
                             replace(Options(), rank_tolerance=1e-4))
        converted.append(Fixture(float_case.name,
                                 *(np.asarray(a, dtype=np.float32).astype(float)
                                   for a in (float_case.C, float_case.Y, float_case.residual)),
                                 float_case.options))
        cases = converted
    elif precision != "double":
        raise ValueError("Unknown GPU precision")
    if include_nonfinite:
        base = cases[0]
        for target, value in (("C", "nan"), ("Y", "inf")):
            cases.append(replace(base, name=f"nonfinite_{target}",
                                 inject_nonfinite={"target": target, "row": 0,
                                                   "column": 1, "value": value}))
    return cases


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.name: case for case in fixtures()}

    def result(self, name):
        f = self.cases[name]
        return reference_step(f.C, f.Y, f.residual, f.options)

    def test_full_rank_matches_augmented_least_squares(self):
        f = self.cases["full_rank"]
        actual = self.result(f.name)
        scale = actual["scale"]
        augmented = np.vstack((f.Y / scale, np.sqrt(f.options.regularization) * np.eye(3)))
        rhs = np.concatenate((f.residual / scale, np.zeros(3)))
        gamma, *_ = np.linalg.lstsq(augmented, rhs, rcond=None)
        np.testing.assert_allclose(actual["gamma"], gamma, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(actual["direction"], -f.residual - f.C @ gamma, rtol=2e-13, atol=2e-14)

    def test_partial_histories(self):
        for name, rank in (("one_column", 1), ("two_columns", 2)):
            with self.subTest(name=name):
                result = self.result(name)
                self.assertEqual(result["rank"], rank)
                self.assertEqual(result["status"], "accepted")

    def test_physical_column_filtering(self):
        for name, accepted in (("exact_dependence", [0, 2]),
                               ("near_dependence", [0, 2]),
                               ("weak_first_column", [1, 2]),
                               ("dependent_newest_preferred", [0, 2]),
                               ("rank_one_history", [0]),
                               ("span_dependence", [0, 1])):
            with self.subTest(name=name):
                result = self.result(name)
                self.assertEqual(result["accepted_indices"], accepted)
                self.assertEqual(result["rank"], len(accepted))
                rejected = sorted(set(range(3)) - set(accepted))
                np.testing.assert_array_equal(result["gamma"][rejected], 0)

    def test_independent_innovation_is_retained(self):
        self.assertEqual(self.result("independent_small_innovation")["rank"], 3)

    def test_empty_and_zero_histories_fall_back(self):
        for name, status in (("empty_history", "empty_history"), ("zero_history", "rank_zero")):
            with self.subTest(name=name):
                f, result = self.cases[name], self.result(name)
                self.assertEqual(result["status"], status)
                np.testing.assert_array_equal(result["direction"], -f.residual)
                np.testing.assert_array_equal(result["gamma"], 0)

    def test_safety_limits_fall_back_and_clear_applied_coefficients(self):
        for name, status in (("coefficient_rejection", "coefficient_limit"),
                             ("direction_rejection", "direction_limit")):
            with self.subTest(name=name):
                f, result = self.cases[name], self.result(name)
                self.assertEqual(result["status"], status)
                np.testing.assert_array_equal(result["direction"], -f.residual)
                np.testing.assert_array_equal(result["gamma"], 0)
                self.assertEqual(result["rank"], 3)  # Retain the rejected fit's diagnostics.

    def test_scaling_invariance(self):
        original = self.result("full_rank")
        for name, factor in (("tiny_scale", 1e-120), ("large_scale", 1e120)):
            with self.subTest(name=name):
                result = self.result(name)
                self.assertEqual(result["status"], "accepted")
                np.testing.assert_allclose(result["gamma"], original["gamma"], rtol=2e-13, atol=2e-14)
                np.testing.assert_allclose(result["direction"] / factor, original["direction"],
                                           rtol=2e-13, atol=2e-14)

    def test_paired_permutation_and_alignment(self):
        original, permuted = self.result("full_rank"), self.result("paired_column_permutation")
        np.testing.assert_allclose(permuted["direction"], original["direction"], rtol=2e-13, atol=2e-14)
        f = self.cases["full_rank"]
        misaligned = reference_step(f.C[:, [2, 0, 1]], f.Y, f.residual, f.options)
        self.assertGreater(scaled_norm(misaligned["direction"] - original["direction"]), 0.01)

    def test_zero_residual(self):
        result = self.result("zero_residual")
        np.testing.assert_array_equal(result["gamma"], 0)
        np.testing.assert_array_equal(result["direction"], 0)

    def test_nonfinite_history_falls_back(self):
        f = self.cases["full_rank"]
        for target in ("C", "Y"):
            with self.subTest(target=target):
                C, Y = f.C.copy(), f.Y.copy()
                (C if target == "C" else Y)[0, 1] = np.nan
                result = reference_step(C, Y, f.residual, f.options)
                self.assertEqual(result["status"], "nonfinite_history")
                np.testing.assert_array_equal(result["direction"], -f.residual)

    def test_nonfinite_current_residual_is_an_error(self):
        f = self.cases["full_rank"]
        bad = f.residual.copy()
        bad[0] = np.inf
        with self.assertRaises(ValueError):
            reference_step(f.C, f.Y, bad, f.options)

    def test_nonfinite_proposed_direction_falls_back(self):
        f = self.cases["full_rank"]
        C = np.full_like(f.C, np.finfo(float).max)
        # A permissive coefficient limit ensures this checks direction arithmetic.
        result = reference_step(C, f.Y, 100 * f.residual,
                                replace(f.options, max_coefficient_norm=1e300))
        self.assertEqual(result["status"], "nonfinite_solution")
        np.testing.assert_array_equal(result["direction"], -100 * f.residual)

    def test_ridge_reduces_coefficient_size(self):
        self.assertLess(self.result("strong_regularization")["coefficient_norm"],
                        self.result("full_rank")["coefficient_norm"])

    def test_accepted_fits_satisfy_ridge_stationarity_and_improve_objective(self):
        for f in self.cases.values():
            result = reference_step(f.C, f.Y, f.residual, f.options)
            if result["status"] != "accepted":
                continue
            with self.subTest(name=f.name):
                selected = result["accepted_indices"]
                matrix = f.Y[:, selected] / result["scale"]
                rhs = f.residual / result["scale"]
                gamma = result["gamma"][selected]
                error = matrix @ gamma - rhs
                gradient = matrix.T @ error + f.options.regularization * gamma
                scale = max(1.0, scaled_norm(matrix.T @ rhs))
                self.assertLess(scaled_norm(gradient), 2e-12 * scale)
                objective = np.dot(error, error) + f.options.regularization * np.dot(gamma, gamma)
                self.assertLessEqual(objective, np.dot(rhs, rhs) + 2e-12 * scale)

    def test_invalid_options_are_rejected(self):
        for options in (replace(Options(), regularization=0),
                        replace(Options(), rank_tolerance=-1),
                        replace(Options(), max_coefficient_norm=np.inf)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                options.validate()


class FloatFixtureTests(unittest.TestCase):
    def test_inputs_are_exactly_float_quantised(self):
        for f in fixtures_for_precision("float"):
            for array in (f.C, f.Y, f.residual):
                np.testing.assert_array_equal(array, array.astype(np.float32).astype(float))
            self.assertEqual(f.options.rank_tolerance, 1e-4)

    def test_float_rank_decisions_have_clear_margins(self):
        cases = {case.name: case for case in fixtures_for_precision("float")}
        for name, rank in (("exact_dependence", 2), ("near_dependence", 2),
                           ("rank_one_history", 1), ("span_dependence", 2),
                           ("independent_small_innovation", 2),
                           ("float_independent_innovation", 3)):
            with self.subTest(name=name):
                self.assertEqual(cases[name].reference_result()["rank"], rank)

    def test_float_scale_cases_remain_equivalent_after_quantisation(self):
        cases = {case.name: case for case in fixtures_for_precision("float")}
        expected = cases["full_rank"].reference_result()
        for name, factor in (("tiny_scale", 1e-12), ("large_scale", 1e12)):
            with self.subTest(name=name):
                result = cases[name].reference_result()
                np.testing.assert_allclose(result["gamma"], expected["gamma"], rtol=5e-6, atol=2e-7)
                np.testing.assert_allclose(result["direction"] / factor, expected["direction"],
                                           rtol=5e-6, atol=2e-7)


def compare_adapter(executable, cases, precision="double", qr_updates=False):
    """Compare production adapter directions, filtering and applied coefficients."""
    payload = "".join(json.dumps(case.json_object(), allow_nan=False) + "\n" for case in cases)
    command = [str(executable)] + (["--float"] if precision == "float" else [])
    if qr_updates:
        command.append("--qr-updates")
    process = subprocess.run(command, input=payload, text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(outputs) != len(cases):
        raise AssertionError(f"Adapter returned {len(outputs)} cases; expected {len(cases)}. stderr: {process.stderr}")
    checks = []
    single = precision == "float"
    scale_rtol = 3e-6 if single else 2e-12
    coefficient_rtol, coefficient_atol = (5e-4, 5e-6) if single else (2e-7, 2e-9)
    direction_rtol = 3e-4 if single else 2e-8
    for case, actual in zip(cases, outputs):
        expected = case.reference_result()
        if actual.get("name", case.name) != case.name:
            raise AssertionError("CUDA adapter changed fixture ordering")
        if actual["rank"] != expected["rank"] or actual["accepted_indices"] != expected["accepted_indices"]:
            raise AssertionError(f"Incorrect filtering for {case.name}: {actual}")
        if (actual["status"] == "accepted") != (expected["status"] == "accepted"):
            raise AssertionError(f"Incorrect acceptance for {case.name}: {actual['status']} versus {expected['status']}")
        direction = np.asarray(actual["direction"])
        gamma = np.asarray(actual["gamma"])
        if direction.shape != case.residual.shape or gamma.shape != (case.Y.shape[1],):
            raise AssertionError(f"Incorrect output dimensions for {case.name}")
        if not np.isfinite(direction).all() or not np.isfinite(gamma).all():
            raise AssertionError(f"Nonfinite output for {case.name}")
        np.testing.assert_allclose(actual["scale"], expected["scale"], rtol=scale_rtol, atol=0,
                                   err_msg=f"History scale mismatch for {case.name}")
        np.testing.assert_allclose(actual["regularization"], expected["regularization"], rtol=0, atol=0,
                                   err_msg=f"Normalised regularisation mismatch for {case.name}")
        np.testing.assert_allclose(actual["coefficient_norm"], expected["coefficient_norm"],
                                   rtol=coefficient_rtol, atol=coefficient_atol,
                                   err_msg=f"Pre-rejection coefficient norm mismatch for {case.name}")
        # Compare relative to problem scale, including the 1e±120 cases.
        direction_scale = max(scaled_norm(expected["direction"]), scaled_norm(case.residual))
        error = scaled_norm(direction - expected["direction"])
        if error > direction_rtol * direction_scale + 1e-300:
            raise AssertionError(f"Direction mismatch for {case.name}: {error / direction_scale}")
        np.testing.assert_allclose(gamma, expected["gamma"], rtol=coefficient_rtol, atol=coefficient_atol,
                                   err_msg=f"Coefficient mismatch for {case.name}")
        rejected = sorted(set(range(case.Y.shape[1])) - set(expected["accepted_indices"]))
        np.testing.assert_array_equal(gamma[rejected], 0)
        if expected["status"] != "accepted":
            np.testing.assert_array_equal(gamma, 0)
        checks.append({"name": case.name, "rank": actual["rank"], "status": actual["status"],
                       "relative_direction_error": error / direction_scale if direction_scale else error})
    return checks


def check_mpc_counters(path, memory=3):
    """Check enabled-AA accounting without launching a new simulation.

    The terminating outer iteration does not request a direction.  Initial
    bootstrap therefore counts m calls (including iteration zero), while a
    completed restart contributes m-1 subsequent bootstrap calls: the failed
    fit itself already returned the safe splitting direction.
    """
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise AssertionError("No MPC solve rows to check")
    for row in rows:
        iterations = int(row["iterations"])
        fits, accepted = int(row["aa_fits"]), int(row["aa_accepted"])
        restarts, bootstrap = int(row["aa_restarts"]), int(row["aa_bootstrap_steps"])
        rejection_count = sum(int(row[key]) for key in
                              ("aa_coefficient_rejections", "aa_direction_rejections", "aa_numerical_rejections"))
        assert fits + bootstrap == iterations, "Direction-call accounting differs from outer iterations"
        assert accepted + restarts == fits, "A fit was neither accepted nor restarted"
        assert rejection_count == restarts, "Restart reasons do not add up"
        if restarts == 0:
            assert bootstrap == min(memory, iterations), "Bootstrap was not reset for this solve"
        else:
            lower = memory + (memory - 1) * (restarts - 1)
            upper = memory + (memory - 1) * restarts
            assert lower <= bootstrap <= upper, "A restart reused stale history or bootstrapped too long"
        assert 0 <= int(row["aa_last_rank"]) <= memory
        assert 0 <= int(row["aa_filtered_columns"]) <= memory * fits
        if "aa_qr_rebuilds" in row:
            qr_calls = sum(int(row[key]) for key in ("aa_qr_rebuilds", "aa_qr_updates", "aa_qr_reuses"))
            assert qr_calls <= fits, "QR was maintained more often than a direction was fitted"
            assert int(row["aa_qr_update_fallbacks"]) <= int(row["aa_qr_rebuilds"])
        assert int(row["cp_operator_evaluations"]) >= iterations + 1 + int(row["backtracks"])
        if row["converged"] == "True":
            final, tolerance = float(row["final_residual"]), float(row["effective_tolerance"])
            assert np.isfinite(final) and np.isfinite(tolerance)
            assert final <= tolerance * (1 + 1e-12) + 1e-300
    return {"rows_checked": len(rows), "memory": memory, "all_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixtures", type=Path, help="Write finite test cases as JSON lines")
    parser.add_argument("--gpu-executable", type=Path, help="Compare a CUDA adapter with this independent oracle")
    parser.add_argument("--gpu-precision", choices=("double", "float"), default="double",
                        help="Float mode invokes the adapter with --float and quantises fixture inputs")
    parser.add_argument("--include-nonfinite", action="store_true",
                        help="Include standard-JSON markers requesting nonfinite history injection")
    parser.add_argument("--qr-updates", action="store_true", help="Test incremental QR initialisation")
    parser.add_argument("--mpc-csv", type=Path, help="Check enabled-AA bootstrap/restart accounting in an existing CSV")
    parser.add_argument("--report", type=Path, help="Save a machine-readable validation report")
    args = parser.parse_args()
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(FloatFixtureTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    cases = fixtures_for_precision(args.gpu_precision, args.include_nonfinite)
    if args.write_fixtures:
        args.write_fixtures.write_text("".join(json.dumps(case.json_object(), allow_nan=False) + "\n" for case in cases))
    report = {"cpu_tests": result.testsRun, "cpu_tests_passed": True,
              "fixture_precision": args.gpu_precision, "fixture_count": len(cases)}
    if args.gpu_executable:
        report["gpu_cases"] = compare_adapter(args.gpu_executable.resolve(), cases, args.gpu_precision, args.qr_updates)
        report["gpu_cases_passed"] = True
    if args.mpc_csv:
        report["mpc_counter_checks"] = check_mpc_counters(args.mpc_csv)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
