import numpy as np
import argparse
import factories as f


def check_spd(mat, name):
    eigs = np.linalg.eigvals(mat)
    is_positive_definite = eigs.all() > 0
    is_symmetric = np.allclose(mat, mat.T)
    print("Is " + name + " symmetric positive-definite?", is_positive_definite and is_symmetric,
          " with norm (", np.linalg.norm(mat), ").")


parser = argparse.ArgumentParser(description='Example: preconditioning.')
parser.add_argument("--dt", type=str, default='d')
parser.add_argument("--precondition", type=int, default=True)
args = parser.parse_args()
dt = args.dt
precondition = bool(args.precondition)

# Sizes::random
horizon = 3
stopping = 0
num_events = 1
num_inputs = 1
num_states = 1

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

r = np.ones(num_events)
v = r / sum(r)
(final_stage, stop_branching_stage) = (horizon, stopping)
tree = f.tree.IidProcess(
    distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage
).build()
print(tree)
tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
# Dynamics
dynamics = []
A_base = np.eye(num_states) * 1.
B_base = np.ones((num_states, num_inputs)) * .5
for i in range(1, num_events + 1):
    A = A_base * i
    B = B_base * i
    dynamics += [f.build.LinearDynamics(A, B)]

# Costs
nonleaf_costs = []
Q_base = np.eye(num_states) * 1.0
R_base = np.eye(num_inputs) * 0.1
for i in range(1, num_events + 1):
    Q = Q_base * i
    R = R_base * i
    check_spd(Q, "Q")
    check_spd(R, "R")
    nonleaf_costs += [f.build.NonleafCost(Q, R)]

T = Q_base
check_spd(T, "T")
leaf_cost = f.build.LeafCost(T)

# Constraints
nonleaf_state_ub = np.ones(num_states) * 10000.
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = np.ones(num_inputs) * 10000.
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = f.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = np.ones(num_states) * 10000.
leaf_lb = -leaf_ub
leaf_constraint = f.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = 1.
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
    .with_preconditioning(precondition)
    .generate_problem()
)
print(problem)

# Initial state
x0 = np.ones(num_states) * 5.
tree.write_to_file_fp("initialState", x0)

# Check mosek
if precondition:
    print("\n---- Normal problem ----")
    model = f.model.Model(tree, problem)
    model.solve(x0, tol=1e-6)
    print("IPOPT normal status: ", model.status)
    print("States:\n", model.states, "\nInputs:\n", model.inputs, "\n")

    print("---- Preconditioned problem ----")
    model = f.model.ModelWithPrecondition(tree, problem)
    model.solve(x0, tol=1e-6)
    print("IPOPT preconditioned status: ", model.status)
    if model.status == "optimal":
        print("States:\n", model.states, "\nInputs:\n", model.inputs, "\n")
