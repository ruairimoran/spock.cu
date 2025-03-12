import numpy as np
import argparse
import spock as s
import cvxpy as cp


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
stopping = 1
num_events = 2
num_inputs = 1
num_states = 2

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

r = np.ones(num_events)
v = r / sum(r)
(final_stage, stop_branching_stage) = (horizon, stopping)
tree = s.tree.IidProcess(
    distribution=v,
    horizon=final_stage,
    stopping_stage=stop_branching_stage
).build()
print(tree)
# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')  # requires python-tk@3.x installation

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
    dynamics += [s.build.LinearDynamics(A, B)]

# Costs
nonleaf_costs = []
Q_base = np.eye(num_states) * 10.
R_base = np.eye(num_inputs) * 1.
for i in range(1, num_events + 1):
    Q = Q_base * i
    R = R_base * i
    check_spd(Q, "Q")
    check_spd(R, "R")
    nonleaf_costs += [s.build.NonleafCost(Q, R)]

T = Q_base
check_spd(T, "T")
leaf_cost = s.build.LeafCost(T)

# Constraints
nonleaf_state_ub = np.ones(num_states) * 10.
nonleaf_state_lb = -nonleaf_state_ub
nonleaf_input_ub = np.ones(num_inputs) * 7.
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = s.build.Rectangle(nonleaf_lb, nonleaf_ub)
leaf_ub = np.ones(num_states) * 1.
leaf_lb = -leaf_ub
leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = 1.
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
    .with_preconditioning(precondition)
    .generate_problem()
)
print(problem)

# Initial state
x0 = np.ones(num_states) * 5.
tree.write_to_file_fp("initialState", x0)


# Compare solutions
def run_unconditioned(sol):
    model = s.model.Model(tree, problem)
    model.solve(x0, sol, tol=1e-8)
    print("\n---- Normal problem ----")
    print(sol.__str__(), "normal status: ", model.status)
    print("States:\n", model.states, "\nInputs:\n", model.inputs, "\n")
    return model.states, model.inputs


def run_conditioned(sol):
    model = s.model.ModelWithPrecondition(tree, problem)
    model.solve(x0, sol, tol=1e-8)
    print("---- Preconditioned problem ----")
    print(sol.__str__(), "preconditioned status: ", model.status)
    return model.states, model.inputs, model.status


try:
    solver = cp.MOSEK
    states, inputs = run_unconditioned(solver)
except:
    solver = cp.SCS
    states, inputs = run_unconditioned(solver)
if problem.preconditioned:
    states_, inputs_, status = run_conditioned(solver)
    if status != "infeasible":
        tol = 1e-2
        print("States:\n", states_, "\nInputs:\n", inputs_, "\n")
        print("Equal: ",
              np.allclose(states_, states, tol) and
              np.allclose(inputs_, inputs, tol),
              "\n")
