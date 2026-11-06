"""Generate dense costs for operator and adjoint regression checks."""
import os
from pathlib import Path

import numpy as np
import spock as s


def main():
    destination = Path.cwd()
    nx, nu = 65, 35
    a, b = np.linspace(-1, 1, nx), np.linspace(0.2, 1, nu)
    Q, R = np.eye(nx) + np.outer(a, a) / nx, np.eye(nu) + np.outer(b, b) / nu
    for kind in ("quadratic", "quadratic_plus_linear"):
        folder = destination / kind
        folder.mkdir(exist_ok=True)
        os.chdir(folder)
        for precision in ("f", "d"):
            tree = s.tree.MarkovChain(np.full((2, 2), 0.5), np.full(2, 0.5),
                                      horizon=2, stopping_stage=1, dt=precision).build()
            if kind == "quadratic":
                stage = s.build.CostQuadratic(Q.copy(), R.copy())
                leaf = s.build.CostQuadratic(Q.copy(), leaf=True)
            else:
                stage = s.build.CostQuadraticPlusLinear(Q.copy(), 0.1 * a, R.copy(), 0.1 * b)
                leaf = s.build.CostQuadraticPlusLinear(Q.copy(), 0.1 * a, leaf=True)
            (s.problem.Factory(tree, nx, nu)
             .with_dynamics(s.build.Dynamics(0.8 * np.eye(nx), np.full((nx, nu), 0.01)))
             .with_cost_nonleaf(stage).with_cost_leaf(leaf)
             .with_risk(s.build.AVaR(0.95)).with_preconditioning(False).generate_problem())
    os.chdir(destination)


if __name__ == "__main__":
    main()
