import numpy as np
import argparse
import spock as s


parser = argparse.ArgumentParser(description='Example: power distribution: generate problem data.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# --------------------------------------------------------
# Read tree files
# --------------------------------------------------------
tree = s.tree.FromFiles("tree0000")

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
num_states = 10
num_inputs = 10

# Dynamics
A = np.eye(num_states)
B = np.ones(0., 1., size=(num_states, num_inputs))
dynamics = s.build.LinearDynamics(A, B)

# Costs
Q = np.eye(num_states)
R = np.eye(num_inputs)
nonleaf_costs = s.build.NonleafCost(Q, R)
T = Q
leaf_cost = s.build.LeafCost(T)

# Constraints
nonleaf_state_ub = np.ones(num_states)
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = np.ones(num_inputs)
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = s.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = np.ones(num_states)
leaf_lb = -leaf_ub
leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = .95
risk = s.build.AVaR(alpha)

# Generate problem data
problem = (
    s.problem.Factory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_dynamics(dynamics)
    .with_nonleaf_costs(nonleaf_costs)
    .with_leaf_cost(leaf_cost)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .with_julia()
    .generate_problem()
)

# Initial state
x0 = np.zeros(num_states)
tree.write_to_file_fp("initialState", x0)
