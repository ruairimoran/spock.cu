import unittest
from unittest.mock import patch
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from spock import build
from spock.weights import Diagonal, diagonal_values, stack_weights


class DiagonalWeightTests(unittest.TestCase):
    def test_validation_and_owned_storage(self):
        for values in ([], [[1, 2]], [-1, 2], [np.nan], [np.inf], [1 + 1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Diagonal(values)
        values = np.array([0., 2., 3.])
        weight = Diagonal(values)
        values[1] = 99
        np.testing.assert_array_equal(weight.values, [0, 2, 3])
        np.testing.assert_allclose(weight @ np.ones((3, 2)), weight.to_dense() @ np.ones((3, 2)))

    def test_compact_stack_and_mixed_node_fallback(self):
        weights = [Diagonal([1, 2]), Diagonal([3, 4])]
        self.assertEqual(stack_weights(weights).shape, (2, 1, 2))
        mixed = stack_weights([weights[0], np.array([[3., .2], [.2, 4.]])])
        self.assertEqual(mixed.shape, (2, 2, 2))
        np.testing.assert_array_equal(mixed[:, :, 0], weights[0].to_dense())

    def test_quadratic_zero_weights_and_linear_invertibility(self):
        build.CostQuadratic(Diagonal([0, 1]), Diagonal([0]))
        with self.assertRaises(ValueError):
            build.CostQuadraticPlusLinear(Diagonal([0, 1]), np.ones(2), Diagonal([1]), np.ones(1))

    def test_reference_operators_scaling_and_adjoint(self):
        rng = np.random.default_rng(91)
        nx, nu, nodes = 3, 2, 5
        ancestors = [-1, 0, 0, 1, 1]
        qd, rd = np.array([1., 2., 4.]), np.array([3., 5.])
        sx, su = np.diag([.5, .3, .2]), np.diag([.4, .6])
        for linear in (False, True):
            for leaf in (False, True):
                with self.subTest(linear=linear, leaf=leaf):
                    def make(diagonal):
                        Q = Diagonal(qd) if diagonal else np.diag(qd)
                        R = None if leaf else (Diagonal(rd) if diagonal else np.diag(rd))
                        if linear:
                            cost = build.CostQuadraticPlusLinear(Q, np.array([.1, -.2, .3]), R,
                                                                None if leaf else np.array([.2, -.1]), leaf)
                        else:
                            cost = build.CostQuadratic(Q, R, leaf)
                        from copy import deepcopy
                        costs = [deepcopy(cost) for _ in range(nodes)]
                        if not leaf:
                            costs[0].node_zero()
                        return cost.get_class()(costs)
                    dense = make(False)
                    # Diagonal setup and preconditioning must avoid the dense square root.
                    with patch('spock.weights.sqrtm', side_effect=AssertionError('dense square root')):
                        compact = make(True)
                        compact.condition(sx, su)
                    dense.condition(sx, su)
                    np.testing.assert_allclose(compact.translation, dense.translation, atol=1e-14)
                    self.assertIsInstance(compact.Q[0], Diagonal)
                    np.testing.assert_allclose(diagonal_values(compact.Q[-1]), qd * np.diag(sx)**2)
                    scales = compact.get_scaling_states(np.ones(nx))
                    np.testing.assert_allclose(scales, dense.get_scaling_states(np.ones(nx)))
                    x = [rng.normal(size=(nx, 1)) for _ in range(nodes)]
                    u = [rng.normal(size=(nu, 1)) for _ in range(nodes)]
                    scalar = list(rng.normal(size=nodes))
                    dual = [rng.normal(size=(compact.dim_per_node, 1)) for _ in range(nodes)]
                    if not leaf:
                        dual[0].fill(0)
                    def forward(cost):
                        return cost.op_leaf(x, scalar) if leaf else cost.op_nonleaf(x, u, scalar, ancestors)
                    def adjoint(cost):
                        return cost.adj_leaf(dual, nx) if leaf else cost.adj_nonleaf(dual, nx, nu, ancestors)
                    np.testing.assert_allclose(forward(compact), forward(dense), atol=1e-13)
                    for actual, expected in zip(adjoint(compact), adjoint(dense)):
                        for actual_node, expected_node in zip(actual, expected):
                            np.testing.assert_allclose(actual_node, expected_node, atol=1e-13)
                    lhs = sum(float((a.T @ b).item()) for a, b in zip(forward(compact), dual))
                    result = adjoint(compact)
                    rhs = sum(float((a.T @ b).item()) for a, b in zip(x, result[0]))
                    if not leaf:
                        rhs += sum(float((a.T @ b).item()) for a, b in zip(u, result[1]))
                    rhs += sum(float(a * np.asarray(b).item()) for a, b in zip(scalar, result[-1]))
                    self.assertAlmostEqual(lhs, rhs, places=11)

    def test_cvxpy_diagonal_form(self):
        from spock.model import quadratic_form
        import cvxpy as cp
        x = cp.Variable(3)
        x.value = np.array([1., 2., 3.])
        self.assertAlmostEqual(quadratic_form(x, Diagonal([0, 2, 4])).value, 44.)

    def test_factory_export_and_reference_solve(self):
        import spock as s
        objectives = []
        for compact in (False, True):
            previous = Path.cwd()
            with tempfile.TemporaryDirectory() as directory:
                try:
                    os.chdir(directory)
                    with contextlib.redirect_stdout(io.StringIO()):
                        tree = s.tree.MarkovChain(np.full((2, 2), .5), np.full(2, .5), 2, 1).build()
                        Q = Diagonal([1, 2]) if compact else np.diag([1., 2.])
                        R = Diagonal([3]) if compact else np.diag([3.])
                        factory = (s.problem.Factory(tree, 2, 1)
                                   .with_dynamics(build.Dynamics(.7 * np.eye(2), np.full((2, 1), .1))))
                        # Any accidental expansion in the compact generation path is a regression.
                        with patch.object(Diagonal, 'to_dense', side_effect=AssertionError('dense expansion')):
                            problem = (factory.with_cost_nonleaf(build.CostQuadratic(Q, R))
                                       .with_cost_leaf(build.CostQuadratic(Q, leaf=True))
                                       .with_risk(build.AVaR(.95)).with_preconditioning(True).generate_problem())
                        metadata = json.loads(Path('data/data.json').read_text())
                        self.assertEqual(metadata['cost']['nonleafQ'], 'diagonal' if compact else 'dense')
                        for model_class in (s.model.Model, s.model.ModelWithPrecondition):
                            model = model_class(tree, problem)
                            objectives.append(model.solve(np.array([.2, -.1]), tol=1e-7))
                            self.assertEqual(model.status, 'optimal')
                finally:
                    os.chdir(previous)
        for objective in objectives[1:]:
            self.assertAlmostEqual(objectives[0], objective, places=6)


if __name__ == '__main__':
    unittest.main()
