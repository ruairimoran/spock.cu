import numpy as np
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import spock as s


parser = argparse.ArgumentParser(description='Example: network communication.')
parser.add_argument("--dt", type=str, default='d')
parser.add_argument("--br", type=int, default=0)
parser.add_argument("--ch", type=int, default=2)
parser.add_argument("--tree", type=int, default=1)
args = parser.parse_args()
dt = args.dt
br = args.br
ch = args.ch
make_tree = args.tree

if make_tree:
    # --------------------------------------------------------
    # Create data
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Create tree
    # --------------------------------------------------------
    horizon = 20
    branching = np.ones(horizon, dtype=np.int32)
    match br:
        case 0:
            branching[0:2] = [ch, ch]
        case 1:
            branching[0] = np.power(ch, 2)
    data = err_samples
    tree = s.tree.FromData(data, branching).build()
    with open('tree.pkl', 'wb') as f:
        pickle.dump(tree, f)
else:
    with open('tree.pkl', 'rb') as f:
        tree = pickle.load(f)
print(tree)

# --------------------------------------------------------
# Plot scenarios
# --------------------------------------------------------
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
# start = 0  # 1 hour periods
# times = np.arange(start, start + max_time_steps) * 100
# titles = ["demand", "renewables", "price"]
# axes[1].set_ylabel("Error (1. = 100% of forecast value)")
# for i, ax in enumerate(axes):
#     ax.set_xlabel("Time (24 hr)")
#     ax.set_title(f"Multiplier vs. Time ({titles[i]})")
#     ax.grid(True)
#     plot_scenario_values(ax, tree, times, i)
#     set_ticks(ax, times)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------
# Create forecast for problem
# --------------------------------------------------------


# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------
num_states = 10
num_inputs = 5

# Dynamics
dynamics = [None]
for node in range(1, tree.num_nodes):
    dynamics += [s.build.Dynamics()]

# Costs
nonleaf_costs = [None]
for node in range(1, tree.num_nodes):
    nonleaf_costs += [s.build.CostQuadratic(Q, R)]
leaf_cost = s.build.CostQuadratic(Q, leaf=True)

# Constraints
nonleaf_constraint = s.build.PolyhedronWithIdentity(nonleaf_rect, nonleaf_poly)
leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)

# Risk
alpha = .95
risk = s.build.AVaR(alpha)

# Generate
problem = (
    s.problem.Factory(scenario_tree=tree, num_states=num_states, num_inputs=num_inputs)
    .with_dynamics_list(dynamics)
    .with_cost_nonleaf_list(nonleaf_costs)
    .with_cost_leaf(leaf_cost)
    .with_constraint_nonleaf(nonleaf_constraint)
    .with_constraint_leaf(leaf_constraint)
    .with_risk(risk)
    .with_julia()
    .with_preconditioning(True)
    .generate_problem()
)
print(problem)

# --------------------------------------------------------
# Initial state
# --------------------------------------------------------
if br == 0:
    x0 = np.zeros(num_states)
    for k in range(num_states):
        x0[k] = .5 * (leaf_lb[k] + leaf_ub[k])
    tree.write_to_file_fp("initialState", x0)
