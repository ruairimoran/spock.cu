# spock.cu

A `CUDA C++` implementation of the SPOCK algorithm (available [here](https://arxiv.org/pdf/2212.01110)) for multistage risk-averse optimal control problems (RAOCPs).

This solver handles RAOCPs with:
- affine dynamics,
- quadratic(-plus-linear) stage and terminal costs,
- coherent risk measures,
- convex state-input constraints.

This solver is based on scenario trees, which the solver can build from:
- Markov chains,
- i.i.d. processes,
- data samples.
