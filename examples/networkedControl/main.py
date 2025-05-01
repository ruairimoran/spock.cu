import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm
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


def make_beta_tree(gam_, s0_, bf_, plot=False):
    """
    Builds a tree of nodes with values from a Beta distribution.
    Each stage builds upon the nodes of the previous stage, generating children according to a
    Beta distribution that depends on the parent's value.
    Parameters:
        gam_ (float): Scaling factor used to derive alpha and beta parameters.
        s0_ (float): Initial root value at stage 0.
        bf_ (list[int]): A list where bf[i] is the number of points to generate at stage i.
        plot (bool): whether to plot values
    Returns:
        nodes (list[dict]): All nodes in a list (1-based index).
            Each node is a dictionary with:
            - 'index': Node index (int)
            - 'parent': Index of the parent node (int)
            - 'value': Value of the node (float)
            - 'stage': Tree stage (int)
        anc_ (list[int]): List of parent indices for each node in 1-based indexing.
    """
    num_stages = len(bf_)
    # 1-based index is used consistently (like MATLAB)
    nodes = [None, {'index': 1, 'parent': 0, 'value': s0_, 'stage': 0}]  # Index 0 is unused, nodes[1] is the root
    current_indices = [1]
    current_stage = 0

    while current_stage < num_stages:
        n_points = bf_[current_stage]
        next_indices = []

        for idx in current_indices:
            node = nodes[idx]
            alp = gam_ * node['value']
            bet = gam_ * (1 - node['value'])
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
    anc_ = [node['parent'] for node in nodes if node is not None]  # remove 1-based naming inside
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
            st_j = stages_[j - 1]
            parent_index = anc_[j - 1]
            v_parent = data_[parent_index - 1]
            v_current = data_[j - 1]
            plt.plot([st_j - 1, st_j], [v_parent, v_current], '-o',
                     color=(0.3, 0.3, 0.3, 0.5))  # Gray with transparency

        plt.xlabel('Stage')
        plt.ylabel('Value')
        plt.title(f'γ={gam_}, s0={s0_}')
        plt.tight_layout()
        plt.show()

    anc_ = [a - 1 for a in anc_]  # remove 1-based naming inside

    return stages_, anc_, probs_, data_


def integrate_matrix_function(f_, a_, b_, num_points=20):
    """Integrate a matrix-valued function f(s) over [a, b] using Gauss–Legendre quadrature."""
    if a_ == b_:
        return np.zeros_like(f_((a_ + b_) / 2))  # Return zero matrix if interval is empty

    # Gauss–Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(num_points)

    # Map to [a, b]
    s_vals = 0.5 * (b_ - a_) * x + 0.5 * (a_ + b_)
    weights = 0.5 * (b_ - a_) * w

    # Evaluate integrand at all points and sum weighted values
    result = np.zeros_like(f_(s_vals[0]))
    for s, weight in zip(s_vals, weights):
        result += weight * f_(s)
    return result


def make_matrix_integrand(Ac_, Bc_):
    """Returns a matrix-valued function f(s) = expm(Ac * s) @ Bc."""
    def integrand(s):
        return expm(Ac_ * s) @ Bc_
    return integrand


def compute_Ai_Bi(Ac_, Bc_, tau_, h_, num_quad_points=20):
    n, m = Ac_.shape[0], Bc_.shape[1]
    integrand = make_matrix_integrand(Ac_, Bc_)

    # Ai components
    int_Ai = integrate_matrix_function(integrand, h_ - tau_, h_, num_quad_points)
    top_Ai = np.hstack((expm(Ac_ * h_), int_Ai))
    bot_Ai = np.hstack((np.zeros((m, n)), np.zeros((m, m))))
    Ai = np.vstack((top_Ai, bot_Ai))

    # Bi components
    int_Bi = integrate_matrix_function(integrand, 0, h_ - tau_, num_quad_points)
    Bi = np.vstack((int_Bi, np.eye(m)))

    return Ai, Bi


# --------------------------------------------------------
# Create tree
# --------------------------------------------------------
max_delay = 16  # milliseconds
if make_tree:
    horizon = 3
    gam = 100
    s0 = 0.2
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
# Generate problem data
# state = [x(k), u(k-1)]
# input = [u(k)]
# --------------------------------------------------------
num_states = 2
num_inputs = 1
state_size = num_states + num_inputs

# Dynamics
dynamics = [None]
Ac = np.array([[0., 1.],
               [0., 0.]])
Bc = np.array([[0.],
               [126.7]])
for node in range(1, tree.num_nodes):
    t = tree.data_values[node][0]
    A, B = compute_Ai_Bi(Ac, Bc, t, max_delay)
    dynamics += [s.build.Dynamics(A, B)]

# Costs
zero = 1e-6
nonleaf_costs = [None]
Q = np.diag([10. for _ in range(num_states)] + [zero for _ in range(num_inputs)])
R = np.diag([1. for _ in range(num_inputs)])
for node in range(1, tree.num_nodes):
    nonleaf_costs += [s.build.CostQuadratic(Q, R)]
leaf_cost = s.build.CostQuadratic(Q, leaf=True)

# Constraints
states_ub = np.ones(num_states) * 10.
inputs_ub = np.ones(num_inputs) * 2.
states_lb = -states_ub
inputs_lb = -inputs_ub
nonleaf_constraint = s.build.Rectangle(np.hstack((states_lb, inputs_lb, inputs_lb)),
                                       np.hstack((states_ub, inputs_ub, inputs_ub)))
leaf_constraint = s.build.Rectangle(np.hstack((states_lb, inputs_lb)),
                                    np.hstack((states_ub, inputs_ub)))

# Risk
alpha = .1
risk = s.build.AVaR(alpha)

# Generate
problem = (
    s.problem.Factory(scenario_tree=tree, num_states=state_size, num_inputs=num_inputs)
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
    x0 = np.zeros(state_size)
    for k in range(num_states):
        x0[k] = np.random.uniform(.5 * states_lb[k], .5 * states_ub[k])
    tree.write_to_file_fp("initialState", x0)
