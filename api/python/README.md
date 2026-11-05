# SPOCK API
For generating files that SPOCK can use to solve RAOCPs.

## Hints and tips
- If running `pip` at top level, replace `.` with `api/python`.
- First, create and activate a python(-tk) virtual environment.
- Then, install the basic dependencies with `pip install .`.
- For developing this api, include edit flag `pip install -e .`.
- For drawing scenario trees, you will need python-tk and run `pip install '.[draw]'`.
- For running examples and tests, use `pip install '.[all]'`.

## Diagonal quadratic weights

Use `spock.build.Diagonal` to supply only the diagonal entries of a weight
matrix. Entries are weights, rather than square-root factors:

```python
import numpy as np
import spock as s

stage = s.build.CostQuadratic(
    Q=s.build.Diagonal([1., 2., 4.]),
    R=s.build.Diagonal([.5, .8]),
)
terminal = s.build.CostQuadratic(s.build.Diagonal([3., 6., 12.]), leaf=True)

# Dense and mixed matrices are supported through the same interface.
mixed = s.build.CostQuadratic(s.build.Diagonal([1., 2., 4.]), np.eye(2))
```

The vector must be nonempty, finite, nonnegative, and match the state or input
dimension. Quadratic-plus-linear costs use the same wrapper but require
strictly positive entries because their translation uses inverse weights.
Diagonal preconditioning and square roots keep the compact representation.
The CPU reference models also accept these weights.

Generate problem files again with this interface to use compact GPU storage.
Each diagonal factor is stored as an `n × 1 × nodes` tensor, with explicit
`nonleafQ`, `nonleafR`, and `leafQ` structure fields in `data.json` under
`cost`. The CUDA reader treats missing fields in existing files as dense.
The diagonal CUDA path gathers and multiplies directly, and its adjoint sums
child contributions without atomics. It avoids dense factors and their
matrix-product work buffers.

Q and R can select different structures. If a component contains a dense
matrix at any scenario-tree node, that component is exported as a dense
stack across nodes. The optional legacy Julia export also uses dense
matrices. `Diagonal.to_dense()` provides an explicit conversion for other
consumers that require one.
