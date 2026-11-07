"""Regress the MPC generator's explicit dense/diagonal representation choice."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from gputils_api import read_array_from_gputils_binary_file as read_bt
from spock.weights import Diagonal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/networkedControlMpc'))
from run_mpc import diagonal_weight, generate_problem


class MpcCostFormatTests(unittest.TestCase):
    def test_diagonal_conversion_owns_its_values(self):
        matrix = np.diag([0., 2., 3.])
        weight = diagonal_weight(matrix)
        matrix[1, 1] = 99.
        np.testing.assert_array_equal(weight.values, [0., 2., 3.])
        copied = diagonal_weight(weight)
        copied.values[1] = 7.
        np.testing.assert_array_equal(weight.values, [0., 2., 3.])

    def test_invalid_weights_are_rejected_without_discarding_couplings(self):
        for matrix in (np.ones((2, 3)), np.array([1., 2.]),
                       np.array([[1., 1e-14], [0., 2.]]),
                       np.diag([-1., 2.]), np.diag([np.inf, 2.]),
                       np.diag([1. + 1j, 2.])):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                diagonal_weight(matrix)

    @staticmethod
    def parameters(precondition):
        def make_tree(gamma, phi, branching, **kwargs):
            return [0, 1, 1], [-1, 0, 0], [1., .5, .5], [phi, phi, .2]

        def dynamics(*args):
            return np.array([[.7, .1], [0., 0.]]), np.array([[.2], [1.]])

        return {
            'make_beta_tree': make_tree, 'compute_Ai_Bi': dynamics,
            'branching': [2], 'trick': 2., 'max_delay': .1,
            'Ac': np.array([[-1.]]), 'Bc': np.array([[1.]]),
            'state_size': 2, 'num_inputs': 1,
            'Q': np.diag([2., 3.]), 'R': np.diag([4.]),
            'Q_leaf': np.diag([5., 6.]),
            'states_lb': np.array([-10.]), 'states_ub': np.array([10.]),
            'inputs_lb': np.array([-2.]), 'inputs_ub': np.array([2.]),
            'alpha': .95, 'stored': {'numNodes': 3, 'preconditioned': precondition},
        }

    def test_default_dense_and_compact_exports_are_equivalent(self):
        for precondition in (False, True):
            with self.subTest(precondition=precondition), tempfile.TemporaryDirectory() as directory:
                parameters = self.parameters(precondition)
                originals = {key: parameters[key].copy() for key in ('Q', 'R', 'Q_leaf')}
                folders = {}
                for compact in (False, True):
                    folder = Path(directory) / ('diagonal' if compact else 'dense')
                    folder.mkdir()
                    if compact:
                        generate_problem(parameters, .1, 1., folder, diagonal_costs=True)
                    else:
                        generate_problem(parameters, .1, 1., folder)  # Default remains dense.
                    folders[compact] = folder / 'data'
                for part, component in [('nonleaf', 'Q'), ('nonleaf', 'R'), ('leaf', 'Q')]:
                    filename = f'{part}Cost_sqrt{component}_d.bt'
                    dense = read_bt(str(folders[False] / filename))
                    compact = read_bt(str(folders[True] / filename))
                    self.assertEqual(compact.shape[1], 1)
                    expanded = np.dstack([np.diag(compact[:, 0, node])
                                          for node in range(compact.shape[2])])
                    np.testing.assert_allclose(expanded, dense, rtol=1e-12, atol=1e-13)
                    for flag, expected in [(False, 'dense'), (True, 'diagonal')]:
                        config = json.loads((folders[flag] / 'data.json').read_text())
                        self.assertEqual(config['cost'][part + component], expected)
                files = ['dynamics_AB_d.bt']
                if precondition:
                    files.append('scaling_d.bt')
                for filename in files:
                    np.testing.assert_allclose(read_bt(str(folders[False] / filename)),
                                               read_bt(str(folders[True] / filename)),
                                               rtol=1e-12, atol=1e-13)
                for key, original in originals.items():
                    np.testing.assert_array_equal(parameters[key], original)


if __name__ == '__main__':
    unittest.main()
