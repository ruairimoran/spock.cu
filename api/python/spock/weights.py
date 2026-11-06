"""Compact quadratic weights shared by the factory and reference solvers."""
import numpy as np
from scipy.linalg import sqrtm


class Diagonal:
    """A diagonal quadratic weight, specified by its nonnegative diagonal."""

    def __init__(self, values):
        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0 or not np.isrealobj(values):
            raise ValueError("Diagonal weights require a nonempty real vector")
        self.values = np.array(values, dtype=float, copy=True)
        self.validate()

    def validate(self):
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("Diagonal weights must be finite and nonnegative")

    @property
    def shape(self):
        return (self.values.size, self.values.size)

    def __matmul__(self, other):
        other = np.asarray(other)
        if other.ndim not in (1, 2) or other.shape[0] != self.values.size:
            raise ValueError("Incompatible diagonal product dimensions")
        return self.values * other if other.ndim == 1 else self.values[:, None] * other

    def to_dense(self):
        """Expand explicitly for consumers that require a dense matrix."""
        return np.diag(self.values)


def diagonal_values(weight):
    return weight.values if isinstance(weight, Diagonal) else np.diag(weight)


def sqrt_weight(weight):
    return Diagonal(np.sqrt(weight.values)) if isinstance(weight, Diagonal) else sqrtm(weight)


def scale_weight(weight, scaling):
    if isinstance(weight, Diagonal):
        diagonal = np.diag(scaling)
        if np.array_equal(scaling, np.diag(diagonal)):
            return Diagonal(weight.values * diagonal**2)
        # A general change of coordinates can destroy diagonal structure.
        return scaling.T @ (weight @ scaling)
    return scaling.T @ weight @ scaling


def zero_weight(weight):
    if isinstance(weight, Diagonal):
        weight.values.fill(0)
    else:
        weight[:] = 0


def dense_weight(weight):
    return weight.to_dense() if isinstance(weight, Diagonal) else weight


def factor_structure(weights):
    return "diagonal" if all(isinstance(weight, Diagonal) for weight in weights) else "dense"


def stack_weights(weights):
    # Different nodes can select different representations. If one node is
    # dense, use the existing dense stack for that component only.
    if factor_structure(weights) == "diagonal":
        return np.dstack([weight.values[:, None] for weight in weights])
    return np.dstack([dense_weight(weight) for weight in weights])
