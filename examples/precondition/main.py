import numpy as np
import argparse
import spock as s
import cvxpy as cp


parser = argparse.ArgumentParser(description='Example: preconditioning.')
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# Sizes
horizon = 2
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
A_base = np.eye(num_states)
B_base = np.ones((num_states, num_inputs)) * .5
for i in range(1, num_events + 1):
    A = A_base * 1/i
    B = B_base * 1/i
    dynamics += [s.build.Dynamics(A, B)]

# Costs
rng = np.random.default_rng()
nonleaf_costs = []
zero = 1e-1
Q_base = np.eye(num_states) * zero
R_base = np.eye(num_inputs) * zero
for i in range(1, num_events + 1):
    Q = Q_base
    R = R_base
    q = np.ones(num_states) * zero
    r = np.ones(num_inputs) * i * 3
    nonleaf_costs += [s.build.CostQuadraticPlusLinear(Q, q, R, r)]

T = Q_base
t = np.ones(num_states) * zero
leaf_cost = s.build.CostQuadraticPlusLinear(T, t, leaf=True)

# Constraints
nonleaf_state_ub = np.ones(num_states) * 5.
nonleaf_state_lb = np.zeros(num_states)
nonleaf_input_ub = np.ones(num_inputs) * 4.
nonleaf_input_lb = -nonleaf_input_ub
nonleaf_lb = np.hstack((nonleaf_state_lb, nonleaf_input_lb))
nonleaf_ub = np.hstack((nonleaf_state_ub, nonleaf_input_ub))
nonleaf_constraint = s.build.Polyhedron(np.eye(num_states + num_inputs), nonleaf_lb, nonleaf_ub)
leaf_ub = np.ones(num_states) * 1.
leaf_lb = np.zeros(num_states)
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
    .with_dynamics_events(dynamics)
    .with_cost_nonleaf_events(nonleaf_costs)
    .with_cost_leaf(leaf_cost)
    .with_constraint_nonleaf(nonleaf_constraint)
    .with_constraint_leaf(leaf_constraint)
    .with_risk(risk)
    .with_preconditioning(False)
    .with_julia()
    .generate_problem()
)
print("Unconditioned: \n", problem)

problem_cond = (
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
    .with_preconditioning(True)
    .generate_problem()
)
print("Preconditioned: \n", problem_cond)

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
    return model.states, model.inputs, model.primal


def run_conditioned(sol):
    model = s.model.ModelWithPrecondition(tree, problem_cond)
    model.solve(x0, sol, tol=1e-8)
    print("---- Preconditioned problem ----")
    print(sol.__str__(), "preconditioned status: ", model.status)
    return model.states, model.inputs, model.status


np.set_printoptions(suppress=True)
try:
    solver = cp.MOSEK
    states, inputs, primal = run_unconditioned(solver)
except:
    solver = cp.SCS
    states, inputs, primal = run_unconditioned(solver)
states_, inputs_, status = run_conditioned(solver)
if status != "infeasible":
    tol = 1e-2
    print("States:\n", states_, "\nInputs:\n", inputs_, "\n")
    print("Equal: ",
          np.allclose(states_, states, tol) and
          np.allclose(inputs_, inputs, tol),
          "\n")

# Primal solution
tree.write_to_file_fp("primal", primal)
