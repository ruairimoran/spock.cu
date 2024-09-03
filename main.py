import py
import numpy as np


# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])

v = np.array([0.3, 0.7])

(horizon, stopping_stage) = (2, 1)
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
num_states = 1
num_inputs = 1
num_events = 2

# State dynamics
A = 0.1 * np.eye(num_states)
As = [A, A]

# Input dynamics
B = 0.5 * np.eye(num_inputs)
Bs = [B, B]

# State cost
Q = 2 * np.eye(num_states)
Qs = [Q, Q]

# Input cost
R = 10 * np.eye(num_inputs)
Rs = [R, R]

# Terminal state cost
T = 50 * np.eye(num_states)

# State-input constraint
state_lim = 100.
input_lim = 100.
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
nonleaf_lb = np.vstack((state_lb, input_lb))
nonleaf_ub = np.vstack((state_ub, input_ub))
nonleaf_constraint = py.build.Rectangle(nonleaf_lb, nonleaf_ub)

# Terminal constraint
leaf_state_lim = 10.
leaf_lb = -leaf_state_lim * np.ones((num_states, 1))
leaf_ub = leaf_state_lim * np.ones((num_states, 1))
leaf_constraint = py.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = .99
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
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .generate_problem()
)
