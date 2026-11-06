"""Generate equivalent compact and dense cost problems for CUDA regressions."""
import os
import json
from pathlib import Path

import numpy as np
import spock as s


def generate(destination, kind, diagonal_q, diagonal_r, precondition, scalar=False):
    nx, nu = (1, 1) if scalar else (5, 3)
    qd, rd = np.linspace(1, 4, nx), np.linspace(2, 5, nu)
    Q, R = np.diag(qd), np.diag(rd)
    if not diagonal_q:
        Q += .1 * np.ones((nx, nx))
    if not diagonal_r:
        R += .1 * np.ones((nu, nu))
    for representation in ('compact', 'dense'):
        folder = destination / representation
        folder.mkdir(parents=True, exist_ok=True)
        os.chdir(folder)
        for precision in ('f', 'd'):
            tree = s.tree.FromStructure([0, 1, 1, 1, 2, 2, 2, 2], [-1, 0, 0, 0, 1, 1, 2, 3],
                                       [1, .4, .3, .3, .2, .2, .3, .3], np.zeros(8), dt=precision).build()
            def make_cost(scale, leaf=False):
                state = s.build.Diagonal(qd * scale) if representation == 'compact' and diagonal_q else Q * scale
                control = s.build.Diagonal(rd * scale) if representation == 'compact' and diagonal_r else R * scale
                if kind == 'quadratic':
                    return s.build.CostQuadratic(state, None if leaf else control, leaf)
                return s.build.CostQuadraticPlusLinear(state, np.linspace(-.1, .2, nx),
                                                      None if leaf else control,
                                                      None if leaf else np.linspace(.1, .3, nu), leaf)
            stage = [None] + [make_cost(1 + node / 10) for node in range(1, 8)]
            (s.problem.Factory(tree, nx, nu)
             .with_dynamics(s.build.Dynamics(.7 * np.eye(nx), np.full((nx, nu), .1)))
             .with_cost_nonleaf_list(stage).with_cost_leaf(make_cost(2, leaf=True))
             .with_risk(s.build.AVaR(.95)).with_preconditioning(precondition).generate_problem())
            if representation == 'dense':
                # Exercise legacy files, which do not declare factor structure.
                metadata = Path('data/data.json')
                content = json.loads(metadata.read_text())
                for key in ('nonleafQ', 'nonleafR', 'leafQ'):
                    content['cost'].pop(key, None)
                metadata.write_text(json.dumps(content))
    os.chdir(destination.parent)


def main():
    destination = Path.cwd()
    for kind in ('quadratic', 'quadratic_plus_linear'):
        for mode, dq, dr in (('diagonal', True, True), ('mixed_q', True, False), ('mixed_r', False, True)):
            generate(destination / (kind + '_' + mode), kind, dq, dr, precondition=mode != 'diagonal')
    generate(destination / 'scalar_quadratic', 'quadratic', True, True, precondition=True, scalar=True)
    os.chdir(destination)


if __name__ == '__main__':
    main()
