import numpy as np
import argparse
import spock as s

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
tree = s.tree.MarkovChain(
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

dynamics = [s.build.Dynamics(1.5 * A, 1.5 * B), s.build.Dynamics(A, B)]

# State cost
q = np.eye(num_states)

# Input cost
r = np.eye(num_inputs)

nonleaf_costs = []
for i in range(1, 3):
    q *= i
    nonleaf_costs += [s.build.CostQuadratic(q, r)]

# Terminal state cost
T = 3 * np.eye(num_states)

leaf_cost = s.build.CostQuadratic(T, leaf=True)

# State-input constraint
state_lim = 1.
input_lim = 5.
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
si_lb = np.vstack((state_lb, input_lb))
si_ub = np.vstack((state_ub, input_ub))
nonleaf_constraint = s.build.Rectangle(si_lb, si_ub)

# Terminal constraint
leaf_state_lim = .1
leaf_state_lb = -leaf_state_lim * np.ones((num_states, 1))
leaf_state_ub = leaf_state_lim * np.ones((num_states, 1))
leaf_constraint = s.build.Rectangle(leaf_state_lb, leaf_state_ub)

# Risk
alpha = .95
risk = s.build.AVaR(alpha)

# Generate problem data
problem = (
    s.problem.Factory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_dynamics_events(dynamics)
    .with_cost_nonleaf_events(nonleaf_costs)
    .with_cost_leaf(leaf_cost)
    .with_constraint_nonleaf(nonleaf_constraint)
    .with_constraint_leaf(leaf_constraint)
    .with_risk(risk)
    .with_preconditioning()
    .with_tests()
    .generate_problem()
)
print(problem)
