import build

import py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Problem data generator.')
parser.add_argument("--nStages", type=int, default=5)
parser.add_argument("--nStates", type=int, default=4)
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])

v = np.array([0.3, 0.7])

num_stages = args.nStages
(final_stage, stop_branching_stage) = (num_stages - 1, num_stages - 1)
tree = py.treeFactory.MarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage,
    dt=dt
).generate_tree()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------

# Sizes
num_states = args.nStates
num_inputs = 3
num_events = 2

# Dynamics
off_diag = 0.01 * np.ones((1, num_states - 1))[0]
As = [None] * num_events
for w in range(num_events):
    As[w] = np.diag(off_diag, -1) + np.diag(off_diag, 1)
    for j in range(num_states):
        diag = 1 + (1 + j / num_states) * w / num_events
        As[w][j][j] = diag

B = 1. * np.ones((num_states, num_inputs))

dynamics = [py.build.LinearDynamics(As[0], B), py.build.LinearDynamics(As[1], B)]

# Nonleaf costs
Q = 1e-1 * np.eye(num_states)
Qs = [Q, Q]
q = 1e-1 * np.ones(num_states)
qs = [q, q]

R = 1. * np.eye(num_inputs)
Rs = [R, R]
r = 1. * np.ones(num_inputs)
rs = [r, r]

nonleaf_costs = [py.build.NonleafCost(Q, R, q, r), py.build.NonleafCost(Q, .9*R, q, r)]

# Leaf costs
T = 1e-1 * np.eye(num_states)
t = 1e-1 * np.ones(num_states)

leaf_costs = py.build.LeafCost(Q, q)

# State-input constraint
nl_state_lim = 1.
nl_input_lim = 1.5
nl_r_ub = np.vstack((nl_state_lim * np.ones((num_states, 1)), nl_input_lim * np.ones((num_inputs, 1))))
nl_r_lb = -nl_r_ub
nonleaf_constraint = py.build.Rectangle(nl_r_lb, nl_r_ub)

# Terminal constraint
l_state_lim = 1.
l_r_ub = np.vstack((l_state_lim * np.ones((num_states, 1))))
l_r_lb = -l_r_ub
leaf_constraint = py.build.Rectangle(l_r_lb, l_r_ub)

# Risk
alpha = .95
risk = py.build.AVaR(alpha)

# Generate problem data
problem = (
    py.problemFactory.ProblemFactory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_markovian_dynamics(dynamics)
    .with_markovian_nonleaf_costs(nonleaf_costs)
    .with_leaf_cost(leaf_costs)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .generate_problem()
)
