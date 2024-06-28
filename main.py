import py
import numpy as np


# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])

v = np.array([0.8, 0.2])

(horizon, stopping_stage) = (3, 2)
tree = py.treeFactory.TreeFactoryMarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=horizon,
    stopping_stage=stopping_stage
).generate_tree()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------

# Sizes
num_states = 3
num_inputs = 2

# State dynamics
A = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
As = [1.5 * A, A, -1.5 * A]

# Input dynamics
B = np.array([[3, 0], [1, 0], [0, 2]])
Bs = [-1.5 * B, B, 1.5 * B]

# State cost
Q = np.eye(num_states)
Qs = [2 * Q, 2 * Q, 2 * Q]

# Input cost
R = np.eye(num_inputs)
Rs = [1 * R, 1 * R, 1 * R]

# Terminal state cost
T = 100 * np.eye(num_states)

# State constraint
state_lim = 6
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
state_constraint = py.build.Rectangle(state_lb, state_ub)

# Input constraint
input_lim = 0.3
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
input_constraint = py.build.Rectangle(state_lb, state_ub)

# Risk
alpha = .95
risk = py.build.AVaR(alpha)

# Generate problem data
problem = (
    py.problemFactory.ProblemFactory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_markovian_dynamics(As, Bs)
    .with_markovian_nonleaf_costs(Qs, Rs)
    .with_all_leaf_costs(T)
    .with_all_constraints(state_constraint, input_constraint)
    .with_all_risks(risk)
    .generate_problem()
)
