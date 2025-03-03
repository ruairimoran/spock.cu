import numpy as np
import argparse
import factories as f


def check_spd(mat, name):
    eigs = np.linalg.eigvals(mat)
    is_positive_definite = eigs.all() > 0
    is_symmetric = np.allclose(mat, mat.T)
    print("Is " + name + " symmetric positive-definite?", is_positive_definite and is_symmetric,
          " with norm (", np.linalg.norm(mat), ").")


parser = argparse.ArgumentParser(description='Example: random.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# Sizes::random
horizon, num_events, stopping, num_inputs, num_states = 0, 0, 0, 0, 0
rng = np.random.default_rng()
num_vars = np.inf
while num_vars > 4e3:
    horizon = rng.integers(5, 15, endpoint=True)
    stopping = rng.integers(1, 3, endpoint=True)
    num_events = rng.integers(2, 10, endpoint=True)
    num_inputs = rng.integers(10, 100, endpoint=True)
    num_states = num_inputs * 2
    num_nodes = ((num_events**(stopping + 1) - 1) / (num_events - 1)) + ((num_events**stopping) * (horizon - stopping))
    num_vars = num_nodes * (num_states + num_inputs)

print(
    "\n",
    "Events:", num_events, "\n",
    "Horizon:", horizon, "\n",
    "Stop branch:", stopping, "\n",
    "States:", num_states, "\n",
    "Inputs:", num_inputs, "\n",
    "Variables:", num_vars, "\n",
)

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

r = rng.uniform(size=num_events)
v = r / sum(r)

(final_stage, stop_branching_stage) = (horizon, stopping)
tree = f.tree.IidProcess(
    distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage
).build()

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
    A = A_base + rng.normal(0., .01, size=(num_states, num_states))
    # A = enforce_contraction(A)
    B = B_base + rng.normal(0., .01, size=(num_states, num_inputs))
    dynamics += [f.build.LinearDynamics(A, B)]

# Costs
nonleaf_costs = []
Q_base = rng.normal(0., .02, size=(num_states, num_states))
R_base = rng.normal(0., .01, size=(num_inputs, num_inputs))
for i in range(num_events):
    Q_w = Q_base + rng.normal(0., .01, size=(num_states, num_states))
    R_w = R_base + rng.normal(0., .01, size=(num_inputs, num_inputs))
    Q = Q_w @ Q_w.T
    R = R_w @ R_w.T
    # check_spd(Q, "Q")
    # check_spd(R, "R")
    nonleaf_costs += [f.build.NonleafCost(Q, R)]

T = Q_base @ Q_base.T
# check_spd(T, "T")
leaf_cost = f.build.LeafCost(T)

# Constraints
nonleaf_state_ub = rng.uniform(1., 2., num_states)
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = rng.uniform(0., .1, num_inputs)
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = f.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = rng.uniform(1., 2., num_states)
leaf_lb = -leaf_ub
leaf_constraint = f.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = rng.uniform(0., 1.)
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
    .with_preconditioning(True)
    .with_julia()
    .generate_problem()
)

# Initial state
x0 = np.zeros(num_states)
for k in range(num_states):
    con = .5 * nonleaf_state_ub[k]
    x0[k] = rng.uniform(-con, con)
tree.write_to_file_fp("initialState", x0)
