# spock.cu

A `CUDA C++` implementation of the SPOCK algorithm (available [here](https://arxiv.org/pdf/2212.01110)) for multistage risk-averse optimal control problems (RAOCPs).

This solver handles RAOCPs with:

- Linear dynamics
- Quadratic stage and terminal costs
- Coherent risk measures
- Convex input-state constraints
