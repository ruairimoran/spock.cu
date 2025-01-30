import py.treeFactory as treeFactory
import py.build as build
import py.problemFactory as problemFactory
import py.modelFactory as modelFactory
import numpy as np
import cvxpy as cp
import argparse

parser = argparse.ArgumentParser(description='Time cvxpy solvers.')
parser.add_argument("--nEvents", type=int, default=10)
parser.add_argument("--nStages", type=int, default=10)
parser.add_argument("--stop", type=int, default=2)
parser.add_argument("--nStates", type=int, default=2)
parser.add_argument("--dt", type=str, default='d')
args = parser.parse_args()
dt = args.dt

# Sizes
num_stages = args.nStages
num_events = args.nEvents
stopping = args.stop
num_states = args.nStates
num_inputs = num_states

# Solvers
x0 = [.1 for _ in range(num_states)]
solvers = [cp.MOSEK]
s = len(solvers) + 1
cache = [0. for _ in range(s)]

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

v = np.ones(num_events) * 1 / num_events

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
# State dynamics
A = .1 * np.eye(num_states)

# Input dynamics
B = 1. * np.eye(num_inputs)

dynamics = []
for i in range(num_events):
    dynamics += [build.LinearDynamics(A * (i + 1), B)]

# State cost
Q = .1 * np.eye(num_states)

# Input cost
R = 1. * np.eye(num_inputs)

nonleaf_costs = []
for i in range(num_events):
    nonleaf_costs += [build.NonleafCost(Q, R)]

# Terminal state cost
T = .1 * np.eye(num_states)

leaf_cost = build.LeafCost(T)

# State-input constraint
state_lim = 4.
input_lim = .5
state_lb = -state_lim * np.ones((num_states, 1))
state_ub = state_lim * np.ones((num_states, 1))
input_lb = -input_lim * np.ones((num_inputs, 1))
input_ub = input_lim * np.ones((num_inputs, 1))
nonleaf_lb = np.vstack((state_lb, input_lb))
nonleaf_ub = np.vstack((state_ub, input_ub))
nonleaf_constraint = build.Rectangle(nonleaf_lb, nonleaf_ub)

# Terminal constraint
leaf_state_lim = .1
leaf_lb = -leaf_state_lim * np.ones((num_states, 1))
leaf_ub = leaf_state_lim * np.ones((num_states, 1))
leaf_constraint = build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = .95
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

# # Cache
# cache[0] = tree.num_nodes
# for i in range(1, s):
#     model = modelFactory.Model(tree, problem)
#     print("Solving...")
#     model.solve(x0=x0, solver=solvers[i - 1], tol=1e-3)
#     cache[i] = model.solve_time * 1e3
#     print(cache[i], " ms")
#
# # Save to csv
# with open('misc/timeCvxpy.csv', "a") as f:
#     np.savetxt(fname=f, X=np.array(cache).reshape(1, -1), fmt='%.5f', delimiter=', ', newline='\n')
