import py.treeFactory as treeFactory
import py.build as build
import py.problemFactory as problemFactory
import py.modelFactory as modelFactory
import numpy as np
import scipy as sp
import cvxpy as cp
import argparse


def enforce_contraction(A_):
    Q_ = np.eye(A_.shape[0])  # Choose Q = I (identity matrix)
    P_ = sp.linalg.solve_discrete_lyapunov(A_.T, -Q_)  # Solve Atr @ P @ A - P = -Q
    if np.all(np.linalg.eigvals(P_) > 0):  # Check if P is positive definite
        print("Matrix is already firmly nonexpansive.")
        return A_
    else:
        print("Modifying A to ensure firmly nonexpansive...")
        rho_ = max(abs(np.linalg.eigvals(A_)))  # Compute spectral radius
        return A_ / (rho_ + 1e-3)  # Scale A to ensure contraction


parser = argparse.ArgumentParser(description='Time cvxpy solvers.')
# parser.add_argument("--nEvents", type=int, default=2)
# parser.add_argument("--nStages", type=int, default=3)
# parser.add_argument("--stop", type=int, default=2)
# parser.add_argument("--nStates", type=int, default=10)
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# Sizes
# num_stages = args.nStages
# num_events = args.nEvents
# stopping = args.stop
# num_states = args.nStates
# num_inputs = num_states

# Sizes::random
rng = np.random.default_rng(1)
num_events = np.random.randint(2, 5)
num_stages = np.random.randint(3, 4)
stopping = np.random.randint(1, num_stages - 1)
num_inputs = np.random.randint(2, 10)
num_states = num_inputs

print(num_events, num_stages, stopping, num_inputs, num_states)

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

r = np.random.rand(num_events)
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
for i in range(num_events):
    A = np.eye(num_states) - (np.random.rand(num_states, num_states) * .01)
    A = enforce_contraction(A)
    B = np.random.rand(num_states, num_inputs)
    dynamics += [build.LinearDynamics(A, B)]

# Costs
nonleaf_costs = []
for i in range(num_events):
    flat_Q = rng.uniform(0., 10., num_states)
    flat_R = rng.uniform(0., 1, num_inputs)
    Q = np.diagflat(flat_Q)
    R = np.diagflat(flat_R)
    nonleaf_costs += [build.NonleafCost(Q, R)]

flat_T = rng.uniform(10., 100., num_states)
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
leaf_ub = rng.uniform(.01, .1, num_states)
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
    .generate_problem()
)

# Cache solvers
solvers = [cp.MOSEK]
s = len(solvers) + 1
cache = [0. for _ in range(s)]
cache[0] = tree.num_nodes
for i in range(1, s):
    times = []
    print("Solving...")
    for j in range(3):
        model = modelFactory.Model(tree, problem)
        # Initial state
        x0 = np.zeros(num_states)
        for k in range(num_states):
            con = .5 * nonleaf_state_ub[k]
            x0[k] = rng.uniform(-con, con)
        try:
            model.solve(x0=x0, solver=solvers[i - 1], tol=1e-3, warm_start=False)
            time = model.solve_time
        except:
            time = 0.
        if j != 0:
            times += [time]
    cache[i] = sum(times) / len(times)
    print(cache[i], " s")

# Save to csv
with open('misc/timeCvxpy.csv', "a") as f:
    np.savetxt(fname=f, X=np.array(cache).reshape(1, -1), fmt='%.5f', delimiter=', ', newline=', ')
