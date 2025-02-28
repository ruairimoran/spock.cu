import factories as f
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test data generator.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt


# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.3, 0.7], [0.3, 0.7]])

v = np.array([0.3, 0.7])

(horizon, stopping_stage) = (3, 2)
tree = f.tree.MarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=horizon,
    stopping_stage=stopping_stage,
    dt=dt
).build()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------

# Sizes
num_states = 3
num_inputs = 2

# State dynamics
A = np.eye(num_states)

# Input dynamics
B = 0.5 * np.ones((num_states, num_inputs))

dynamics = [f.build.LinearDynamics(1.5 * A, 1.5 * B), f.build.LinearDynamics(A, B)]

# State cost
Q = np.eye(num_states)

# Input cost
R = np.eye(num_inputs)

nonleaf_costs = [f.build.NonleafCost(2 * Q, R), f.build.NonleafCost(2 * Q, R)]

# Terminal state cost
T = 3 * np.eye(num_states)

leaf_cost = f.build.LeafCost(T)

# State-input constraint
state_lim = 1.
input_lim = 5.
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
si_lb = np.vstack((state_lb, input_lb))
si_ub = np.vstack((state_ub, input_ub))
nonleaf_constraint = f.build.Rectangle(si_lb, si_ub)

# Terminal constraint
leaf_state_lim = .1
leaf_state_lb = -leaf_state_lim * np.ones((num_states, 1))
leaf_state_ub = leaf_state_lim * np.ones((num_states, 1))
leaf_constraint = f.build.Rectangle(leaf_state_lb, leaf_state_ub)

# Risk
alpha = .95
risk = f.build.AVaR(alpha)

# Generate problem data
problem = (
    f.problem.Factory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_stochastic_dynamics(dynamics)
    .with_stochastic_nonleaf_costs(nonleaf_costs)
    .with_leaf_cost(leaf_cost)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .with_tests()
    .generate_problem()
)
