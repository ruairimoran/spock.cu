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
As = [1.5 * A, A]

# Input dynamics
B = np.array([[3, 0], [1, 0], [0, 2]])
Bs = [-1.5 * B, B]

# State cost
Q = np.eye(num_states)
Qs = [2 * Q, 2 * Q]

# Input cost
R = np.eye(num_inputs)
Rs = [1 * R, 1 * R]

# Terminal state cost
T = 100 * np.eye(num_states)

# State-input constraint
state_lim = 6
input_lim = 0.3
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
si_lb = np.vstack((state_lb, input_lb))
si_ub = np.vstack((state_ub, input_ub))
state_input_constraint = py.build.Rectangle(si_lb, si_ub)

# Terminal constraint
leaf_state_lim = 0.1
leaf_state_lb = -leaf_state_lim * np.ones((num_states, 1))
leaf_state_ub = leaf_state_lim * np.ones((num_states, 1))
leaf_state_constraint = py.build.Rectangle(leaf_state_lb, leaf_state_ub)

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
    .with_leaf_cost(T)
    .with_nonleaf_constraint(state_input_constraint)
    .with_leaf_constraint(leaf_state_constraint)
    .with_risk(risk)
    .generate_problem()
)