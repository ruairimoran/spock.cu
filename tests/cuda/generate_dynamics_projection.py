"""Small dynamics fixtures with independent dense Euclidean projections."""
import json
import os
from pathlib import Path

import numpy as np
import spock as s
from scipy.linalg import cho_solve


def read_tensor(path, precision):
    with path.open("rb") as stream:
        rows, cols, count = map(int, np.fromfile(stream, dtype="<u8", count=3))
        return np.fromfile(stream, dtype="<f4" if precision == "f" else "<f8").reshape(
            count, cols, rows).swapaxes(1, 2).astype(float)


def write_tensor(path, values, precision):
    """Write (node, row, column) data in the device's column-major layout."""
    count, rows, cols = values.shape
    with path.open("wb") as stream:
        np.array([rows, cols, count], dtype="<u8").tofile(stream)
        values.swapaxes(1, 2).astype("<f4" if precision == "f" else "<f8").tofile(stream)


def projection_factors(A, B, c, ancestors, internal):
    """Prepare Algorithm 1's factors; the online reference below uses dense QR."""
    nodes, nx, nu = B.shape
    P = np.repeat(np.eye(nx)[None], nodes, axis=0)
    K, lower = np.zeros((internal, nu, nx)), np.zeros((internal, nu, nu))
    F, G = np.zeros_like(A), np.zeros((nodes, nx, nu))
    for parent in reversed(range(internal)):
        children = np.flatnonzero(np.asarray(ancestors) == parent)
        H, cross = np.eye(nu), np.zeros((nu, nx))
        for child in children:
            H += B[child].T @ P[child] @ B[child]
            cross += B[child].T @ P[child] @ A[child]
        lower[parent] = np.linalg.cholesky(H)
        K[parent] = cho_solve((lower[parent], True), -cross)
        P[parent] = np.eye(nx) + K[parent].T @ K[parent]
        for child in children:
            F[child] = A[child] + B[child] @ K[parent]
            G[child] = F[child].T @ P[child] @ B[child]
            P[parent] += F[child].T @ P[child] @ F[child]
    return {"BTr": B.swapaxes(1, 2), "AB": np.concatenate((A, B), axis=2),
            "K": K, "KTr": K.swapaxes(1, 2), "ABKTr": F.swapaxes(1, 2),
            "P": P, "APB": G, "lowCholesky": lower, "c": c[:, :, None]}


def generate(destination, ancestors, nx, nu, affine=False, scaled=False,
             rank_deficient=False, seed=527):
    destination.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(destination)
    try:
        stages, probabilities = [0], [1.]
        for node, parent in enumerate(ancestors[1:], 1):
            stages.append(stages[parent] + 1)
            probabilities.append(probabilities[parent] / ancestors.count(parent))
        nodes = len(ancestors)
        internal = next(i for i, stage in enumerate(stages) if stage == max(stages))
        rng = np.random.default_rng(seed)
        A, B, c = np.zeros((nodes, nx, nx)), np.zeros((nodes, nx, nu)), np.zeros((nodes, nx))
        for node in range(1, nodes):
            A[node] = .6 * np.eye(nx) + rng.normal(scale=.08, size=(nx, nx))
            B[node] = rng.normal(scale=.25, size=(nx, nu))
            if rank_deficient:
                B[node, :, 1:] = B[node, :, :1]
            if affine:
                c[node] = rng.normal(scale=.15, size=nx)
        if scaled:
            sx, su = np.geomspace(.3, 3, nx), np.geomspace(.5, 2, nu)
            A = sx[None, :, None] * A / sx[None, None, :]
            B = sx[None, :, None] * B / su[None, None, :]
            c *= sx
        factors = projection_factors(A, B, c, ancestors, internal)
        for precision in ("f", "d"):
            tree = s.tree.FromStructure(stages, ancestors, probabilities,
                                        np.zeros(nodes), dt=precision).build()
            tree.generate_tree_files()
            data = Path("data")
            (data / "data.json").write_text(json.dumps({
                "numEvents": tree.num_events, "numStages": max(stages) + 1,
                "numNodes": nodes, "numNonleafNodes": internal,
                "numStates": nx, "numInputs": nu}))
            for name, values in factors.items():
                write_tensor(data / f"dynamics_{name}_{precision}.bt", values, precision)
            AB = read_tensor(data / f"dynamics_AB_{precision}.bt", precision)
            offset = read_tensor(data / f"dynamics_c_{precision}.bt", precision).reshape(nodes, nx)

            # Constraints C z = b describe the affine dynamics and fixed root.
            # A thin QR computes the projection independently of the Riccati
            # recursion, without forming normal equations or an explicit inverse.
            C = np.zeros((nodes * nx, nodes * nx + internal * nu))
            C[:nx, :nx] = np.eye(nx)
            for node in range(1, nodes):
                parent = ancestors[node]
                C[node*nx:(node+1)*nx, node*nx:(node+1)*nx] = np.eye(nx)
                C[node*nx:(node+1)*nx, parent*nx:(parent+1)*nx] = -AB[node, :, :nx]
                start = nodes * nx + parent * nu
                C[node*nx:(node+1)*nx, start:start+nu] = -AB[node, :, nx:]
            Q, R = np.linalg.qr(C.T, mode="reduced")
            trial_rng = np.random.default_rng(seed + 1)
            for trial in range(3):
                dtype = np.float32 if precision == "f" else np.float64
                zbar = trial_rng.normal(size=C.shape[1]).astype(dtype).astype(float)
                root = trial_rng.normal(size=nx).astype(dtype).astype(float)
                b = offset.reshape(-1).copy()
                b[:nx] = root
                projected = zbar - Q @ np.linalg.solve(R.T, C @ zbar - b)
                np.testing.assert_allclose(C @ projected, b, atol=1e-11, rtol=1e-11)
                for name, values in {
                    "xbar": zbar[:nodes*nx].reshape(nodes, nx, 1),
                    "ubar": zbar[nodes*nx:].reshape(internal, nu, 1),
                    "root": root.reshape(1, nx, 1),
                    "x": projected[:nodes*nx].reshape(nodes, nx, 1),
                    "u": projected[nodes*nx:].reshape(internal, nu, 1),
                }.items():
                    write_tensor(data / f"projection_{trial}_{name}_{precision}.bt", values, precision)
    finally:
        os.chdir(previous)


def main():
    destination = Path.cwd() / "dynamics_projection"
    irregular = [-1, 0, 0, 0, 1, 1, 2, 3]
    generate(destination / "linear", irregular, 5, 3)
    generate(destination / "affine", irregular, 5, 3, affine=True)
    generate(destination / "chain", [-1] + list(range(12)), 5, 3)
    generate(destination / "scalar", [-1, 0], 1, 1, affine=True)
    generate(destination / "wide", [-1, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6],
             7, 50, rank_deficient=True)
    generate(destination / "scaled", irregular, 5, 3, affine=True, scaled=True)
    generate(destination / "linear_update", irregular, 5, 3, seed=928)
    # Cross the 128-column vector tile and the 32-row output tile boundaries.
    generate(destination / "tiled", [-1, 0, 0, 1, 2], 133, 37, affine=True, seed=803)


if __name__ == "__main__":
    main()
