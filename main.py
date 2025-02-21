import py.treeFactory as treeFactory
import py.build as build
import py.problemFactory as problemFactory
import numpy as np
import argparse


def enforce_contraction(A_):
    if np.linalg.eigvals(A_).all() > 0:  # Check if A is positive definite
        # print("Matrix is already firmly nonexpansive.")
        return A_
    else:
        # print("Modifying A to ensure firmly nonexpansive...")
        rho_ = max(abs(np.linalg.eigvals(A_)))  # Compute spectral radius
        A_ = A_ / (rho_ + 1e-1)  # Scale A to ensure contraction
        return A_


parser = argparse.ArgumentParser(description='Time cvxpy solvers.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# Sizes::random
num_events = 0
num_stages = 0
stopping = 0
num_inputs = 0
num_states = 0
rng = np.random.default_rng()
num_nodes = np.inf
while num_nodes > 1e4:
    num_events = 2  # rng.integers(2, 10, endpoint=True)
    num_stages = 3  # rng.integers(3, 10, endpoint=True)
    stopping = 1  # rng.integers(1, min(num_stages, 3))
    num_inputs = 2  # rng.integers(5, 20, endpoint=True)
    num_states = 3  # num_inputs * 2
    v = 1 / num_events * np.ones(num_events)
    (final_stage, stop_branching_stage) = (num_stages - 1, stopping)
    tree_test = treeFactory.IidProcess(
        distribution=v,
        horizon=final_stage,
        stopping_stage=stop_branching_stage
    ).generate_tree(files=False)
    num_nodes = tree_test.num_nodes

print(
    "\n",
    "Events:", num_events, "\n",
    "Stages:", num_stages, "\n",
    "Stop branch:", stopping, "\n",
    "States:", num_states, "\n",
    "Inputs:", num_inputs, "\n",
)

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

r = rng.uniform(size=num_events)
v = r / sum(r)

(final_stage, stop_branching_stage) = (num_stages - 1, stopping)
tree = treeFactory.IidProcess(
    distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage
).generate_tree()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
# Dynamics
dynamics = []
A_base = np.eye(num_states)
B_base = rng.normal(0., 1., size=(num_states, num_inputs))
for i in range(num_events):
    A = A_base + rng.normal(0., .1, size=(num_states, num_states))
    A = enforce_contraction(A)
    B = B_base + rng.normal(0., .1, size=(num_states, num_inputs))
    dynamics += [build.LinearDynamics(A, B)]

# Costs
nonleaf_costs = []
for i in range(num_events):
    Q_flat = rng.uniform(0., 10., num_states)
    R_flat = rng.uniform(0., .1, num_inputs)
    Q = np.diagflat(Q_flat)
    R = np.diagflat(R_flat)
    nonleaf_costs += [build.NonleafCost(Q, R)]
flat_T = rng.uniform(0., 10., num_states)
T = np.diagflat(flat_T)
leaf_cost = build.LeafCost(T)

# Constraints
nonleaf_state_ub = rng.uniform(1., 2., num_states)
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = rng.uniform(0., .1, num_inputs)
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = rng.uniform(1., 2., num_states)
leaf_lb = -leaf_ub
leaf_constraint = build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = rng.uniform(0., 1.)
risk = build.AVaR(alpha)

# Generate problem data
problem = (
    problemFactory.ProblemFactory(
        scenario_tree=tree,
        num_states=num_states,
        num_inputs=num_inputs)
    .with_stochastic_dynamics(dynamics)
    .with_stochastic_nonleaf_costs(nonleaf_costs)
    .with_leaf_cost(leaf_cost)
    .with_nonleaf_constraint(nonleaf_constraint)
    .with_leaf_constraint(leaf_constraint)
    .with_risk(risk)
    .with_julia()
    .generate_problem()
)

# Initial state
x0 = np.zeros(num_states)
for k in range(num_states):
    con = .5 * nonleaf_state_ub[k]
    x0[k] = rng.uniform(-con, con)
tree.write_to_file_fp("initialState", x0)
