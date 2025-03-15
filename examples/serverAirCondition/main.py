import numpy as np
import argparse
import spock as s


parser = argparse.ArgumentParser(description='Example: server air conditioning.')
parser.add_argument("--dt", type=str, default='d')
parser.add_argument("--h", type=int, default=5)
args = parser.parse_args()
dt = args.dt

# Sizes
horizon = args.h
stopping = horizon
num_events = 2
num_inputs = 50
num_states = num_inputs

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])
v = np.array([0.3, 0.7])
(final_stage, stop_branching_stage) = (horizon, stopping)
tree = s.tree.MarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage,
    dt=dt,
).build()
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
# Dynamics
dynamics = []
off_diag = np.ones((1, num_states - 1))[0] * .01
As = [None] * num_events
B = np.eye(num_inputs) * 1.
for w in range(num_events):
    As[w] = np.diag(off_diag, -1) + np.diag(off_diag, 1)
    for j in range(num_states):
        diag = 1. + (1. + j / num_states) * w / num_events
        As[w][j][j] = diag
    dynamics += [s.build.LinearDynamics(As[w], B)]

# Costs
nonleaf_costs = []
for i in range(num_events):
    Q = np.eye(num_states) * .1
    R = np.eye(num_inputs) * 1.
    nonleaf_costs += [s.build.NonleafCost(Q, R)]

T = np.eye(num_states) * .1
leaf_cost = s.build.LeafCost(T)

# Constraints (the states saturate)
nonleaf_state_ub = np.ones(num_states) * 1.
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = np.ones(num_inputs) * 1.5
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = s.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_constraint = s.build.Rectangle(nonleaf_state_lb, nonleaf_state_ub)

# Risk
alpha = .95
risk = s.build.AVaR(alpha)

# Generate problem data
problem = (
    s.problem.Factory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_stochastic_dynamics(dynamics)
    .with_stochastic_nonleaf_costs(nonleaf_costs)
    .with_leaf_cost(leaf_cost)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .with_preconditioning()
    .with_julia()
    .generate_problem()
)
print(problem)

# Initial state
x0 = np.ones(num_states) * .1
tree.write_to_file_fp("initialState", x0)
