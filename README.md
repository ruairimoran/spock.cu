# spock.cu

A `CUDA C++` implementation of the SPOCK algorithm (available [here](https://arxiv.org/pdf/2212.01110)) for multistage risk-averse optimal control problems (RAOCPs).

This solver handles RAOCPs with:

- Affine dynamics
- Quadratic(-plus-linear) stage and terminal costs
- Coherent risk measures
- Convex state-input constraints
