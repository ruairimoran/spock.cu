import numpy as np
import argparse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats
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


def approx_beta_dist(n_points, alp, bet):
    """
    Approximates a beta distribution with `n_points` sample points.
    Parameters:
        n_points (int): Number of sample points to generate.
        alp (float): Alpha (shape1) parameter of the Beta distribution.
        bet (float): Beta (shape2) parameter of the Beta distribution.
    Returns:
        x (array-like): Sample values from the beta distribution.
    """
    if n_points == 1:
        return [alp / (alp + bet)]
    p = np.linspace(0, 1, n_points + 2)
    p = p[1:-1]  # Remove first and last points (0 and 1 to prevent numerical issues)
    x = stats.beta.ppf(p, alp, bet)  # Inverse Beta CDF
    return x


def make_beta_tree(gam, s0, bf, plot=False):
    """
    Builds a tree of nodes with values from a Beta distribution.
    Each stage builds upon the nodes of the previous stage, generating children according to a
    Beta distribution that depends on the parent's value.
    Parameters:
        gam (float): Scaling factor used to derive alpha and beta parameters.
        s0 (float): Initial root value at stage 0.
        bf (list[int]): A list where bf[i] is the number of points to generate at stage i.
    Returns:
        nodes (list[dict]): All nodes in a list (1-based index).
            Each node is a dictionary with:
            - 'index': Node index (int)
            - 'parent': Index of the parent node (int)
            - 'value': Value of the node (float)
            - 'stage': Tree stage (int)
        anc_ (list[int]): List of parent indices for each node in 1-based indexing.
    """
    num_stages = len(bf)
    # 1-based index is used consistently (like MATLAB)
    nodes = [None]  # Index 0 is unused, nodes[1] is the root
    nodes.append({'index': 1, 'parent': 0, 'value': s0, 'stage': 0})
    current_indices = [1]
    current_stage = 0

    while current_stage < num_stages:
        n_points = bf[current_stage]
        next_indices = []

        for idx in current_indices:
            node = nodes[idx]
            alp = gam * node['value']
            bet = gam * (1 - node['value'])
            children = approx_beta_dist(n_points, alp, bet)

            for child_val in children:
                new_idx = len(nodes)
                new_node = {
                    'index': new_idx,
                    'parent': node['index'],
                    'value': child_val,
                    'stage': current_stage + 1
                }
                nodes.append(new_node)
                next_indices.append(new_idx)

        current_indices = next_indices
        current_stage += 1

    # Extract value, parent (anc_), and stage for each node
    data_ = [node['value'] for node in nodes if node is not None]
    anc_ = [node['parent'] - 1 for node in nodes if node is not None]  # 0 based indexing of nodes inside anc_
    stages_ = [node['stage'] for node in nodes if node is not None]

    # Assign equiprobable probabilities
    stages_array = np.array(stages_)
    probs_ = [1.]
    for stage in range(1, num_stages + 1):
        indices = np.where(stages_array == stage)
        n = indices[0].size
        if n == 0:
            raise Exception(f"[make_beta_tree] No points at stage ({stage})!")
        else:
            probs_ += [1 / n] * n

    # Plot the tree
    if plot:
        plt.figure()
        for j in range(1, len(nodes)):
            if j == 1:
                continue  # Skip the root (has no parent)
            st_j = stages_[j-1]
            parent_index = anc_[j-1]
            v_parent = data_[parent_index - 1]
            v_current = data_[j-1]
            plt.plot([st_j - 1, st_j], [v_parent, v_current], '-o',
                     color=(0.3, 0.3, 0.3, 0.5))  # Gray with transparency

        plt.xlabel('Stage')
        plt.ylabel('Value')
        plt.title(f'γ={gam}, s0={s0}')
        plt.tight_layout()
        plt.show()

    return stages_, anc_, probs_, data_


# --------------------------------------------------------
# Create tree
# --------------------------------------------------------
if make_tree:
    horizon = 3
    gam = 10
    s0 = 0.1
    max_delay = 20  # milliseconds
    branching = np.ones(horizon, dtype=np.int32).tolist()
    match br:
        case 0:
            branching[0:2] = [ch, ch]
        case 1:
            branching[0] = np.power(ch, 2)

    stages, anc, probs, data = make_beta_tree(gam, s0, branching, plot=False)
    data = np.array([d * max_delay for d in data])
    tree = s.tree.FromStructure(stages, anc, probs, data).build()
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


# # --------------------------------------------------------
# # Generate problem data
# # --------------------------------------------------------
# num_states = 10
# num_inputs = 5
#
# # Dynamics
# dynamics = [None]
# for node in range(1, tree.num_nodes):
#     dynamics += [s.build.Dynamics()]
#
# # Costs
# nonleaf_costs = [None]
# for node in range(1, tree.num_nodes):
#     nonleaf_costs += [s.build.CostQuadratic(Q, R)]
# leaf_cost = s.build.CostQuadratic(Q, leaf=True)
#
# # Constraints
# nonleaf_constraint = s.build.PolyhedronWithIdentity(nonleaf_rect, nonleaf_poly)
# leaf_constraint = s.build.Rectangle(leaf_lb, leaf_ub)
#
# # Risk
# alpha = .95
# risk = s.build.AVaR(alpha)
#
# # Generate
# problem = (
#     s.problem.Factory(scenario_tree=tree, num_states=num_states, num_inputs=num_inputs)
#     .with_dynamics_list(dynamics)
#     .with_cost_nonleaf_list(nonleaf_costs)
#     .with_cost_leaf(leaf_cost)
#     .with_constraint_nonleaf(nonleaf_constraint)
#     .with_constraint_leaf(leaf_constraint)
#     .with_risk(risk)
#     .with_julia()
#     .with_preconditioning(True)
#     .generate_problem()
# )
# print(problem)
#
# # --------------------------------------------------------
# # Initial state
# # --------------------------------------------------------
# if br == 0:
#     x0 = np.zeros(num_states)
#     for k in range(num_states):
#         x0[k] = .5 * (leaf_lb[k] + leaf_ub[k])
#     tree.write_to_file_fp("initialState", x0)
