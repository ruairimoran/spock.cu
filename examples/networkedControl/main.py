import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm, inv
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


def compute_exp_integral(Ac, Bc, t):
    """
    Computes e^{Ac t} and Phi(t) = integral_0^t e^{Ac s} ds Bc
    using the augmented matrix exponential method.

    Args:
        Ac (np.ndarray): Square matrix (n x n).
        Bc (np.ndarray): Matrix (n x m).
        t (float): Time duration.

    Returns:
        tuple: (exp_Ac_t, phi_t) where
               exp_Ac_t is e^{Ac t} (n x n)
               phi_t is integral_0^t e^{Ac s} ds Bc (n x m)
    """
    if not isinstance(Ac, np.ndarray) or Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
        raise ValueError("Ac must be a square numpy array.")
    if not isinstance(Bc, np.ndarray) or Bc.ndim != 2:
        raise ValueError("Bc must be a 2D numpy array.")
    if Ac.shape[0] != Bc.shape[0]:
        raise ValueError(f"Incompatible shapes: Ac ({Ac.shape}) and Bc ({Bc.shape})")
    if not isinstance(t, (int, float)):
        raise ValueError("t must be a scalar number.")

    n = Ac.shape[0]
    m = Bc.shape[1]

    if t == 0:
        return np.eye(n), np.zeros((n, m))
    if t < 0:
        raise Exception("[compute_exp_integral] Require (tau_i >= 0.).")

    # Construct the augmented matrix M
    M = np.zeros((n + m, n + m), dtype=float) # Use float for broader compatibility
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    # Lower blocks are already zeros

    # Compute the exponential of M*t
    exp_M_t = expm(M * t)

    # Extract the relevant blocks
    exp_Ac_t = exp_M_t[:n, :n]
    phi_t = exp_M_t[:n, n:]

    return exp_Ac_t, phi_t


def compute_Ai_Bi(Ac, Bc, h, tau_i):
    """
    Computes the matrices Ai and Bi based on the provided formulas.

    Ai = [ e^{Ac h}   integral_{h-tau_i}^{h} e^{Ac s} ds Bc ]
         [   0            0                               ]

    Bi = [ integral_{0}^{h-tau_i} e^{Ac s} ds Bc ]
         [          I                           ]

    Args:
        Ac (np.ndarray): Square matrix (n x n).
        Bc (np.ndarray): Matrix (n x m).
        h (float): Positive scalar time duration.
        tau_i (float): Scalar delay, expected 0 <= tau_i <= h.

    Returns:
        tuple: (Ai, Bi) numpy arrays.
    """
    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("[compute_Ai_Bi] h must be a positive scalar number.")
    if not isinstance(tau_i, (int, float)) or tau_i < 0:
        raise ValueError("[compute_Ai_Bi] tau_i must be a non-negative scalar number.")
    if tau_i > h:
        raise ValueError(f"[compute_Ai_Bi] tau_i ({tau_i}) > h ({h}). Integral limits might reverse.")

    n = Ac.shape[0]
    m = Bc.shape[1]  # Assuming Bc is n x m, then I should be m x m

    # Calculate terms at t = h
    exp_Ac_h, phi_h = compute_exp_integral(Ac, Bc, h)

    # Calculate terms at t = h - tau_i
    t_lower = h - tau_i
    _, phi_h_minus_tau = compute_exp_integral(Ac, Bc, t_lower)

    # Calculate the definite integrals needed
    # Integral for Ai: integral_{h-tau_i}^{h} (...) = phi(h) - phi(h-tau_i)
    integral_Ai = phi_h - phi_h_minus_tau

    # Integral for Bi: integral_{0}^{h-tau_i} (...) = phi(h-tau_i)
    integral_Bi = phi_h_minus_tau

    # Assemble the block matrices
    # Ai: top-left=exp_Ac_h, top-right=integral_Ai, bottom-left=0(m x n), bottom-right=0(m x m)
    zero_mn = np.zeros((m, n))
    zero_mm = np.zeros((m, m))
    Ai = np.block([
        [exp_Ac_h, integral_Ai],
        [zero_mn,  zero_mm    ]
    ])

    # Bi: top=integral_Bi, bottom=I(m x m)
    identity_m = np.eye(m)
    Bi = np.block([
        [integral_Bi],
        [identity_m ]
    ])

    return Ai, Bi


# --- Tests ---

# Test Case 1: Ac = 0
def test_Ac_zero():
    n, m = 2, 1
    Ac = np.zeros((n, n))
    Bc = np.array([[1.0], [2.0]])
    h = 1.0
    tau_i = 0.3

    Ai_expected = np.array([
        [1.0, 0.0, 0.3 * 1.0],  # [ I   tau_i * Bc ]
        [0.0, 1.0, 0.3 * 2.0],
        [0.0, 0.0, 0.0      ],  # [ 0      0       ]
        [0.0, 0.0, 0.0      ]
    ])
    # Reshape Ai_expected to match block structure (n+m) x (n+m)
    Ai_expected = np.block([
        [np.eye(n), tau_i * Bc],
        [np.zeros((m,n)), np.zeros((m,m))]
    ])

    Bi_expected = np.array([
        [(h - tau_i) * 1.0],  # [ (h-tau_i)*Bc ]
        [(h - tau_i) * 2.0],
        [1.0              ]   # [      I       ]
    ])
    # Reshape Bi_expected to match block structure (n+m) x m
    Bi_expected = np.block([
        [(h - tau_i) * Bc],
        [np.eye(m)]
    ])

    Ai_calc, Bi_calc = compute_Ai_Bi(Ac, Bc, h, tau_i)

    print("\nTest Ac = 0:")
    print("Ai expected:\n", Ai_expected)
    print("Ai calculated:\n", Ai_calc)
    print("Bi expected:\n", Bi_expected)
    print("Bi calculated:\n", Bi_calc)

    assert np.allclose(Ai_calc, Ai_expected), "Test Ac=0 Failed for Ai"
    assert np.allclose(Bi_calc, Bi_expected), "Test Ac=0 Failed for Bi"


# Test Case 2: Scalar case (n=1, m=1)
def test_scalar():
    Ac = np.array([[-0.5]])  # a = -0.5
    Bc = np.array([[2.0]])   # b = 2.0
    a = Ac[0, 0]
    b = Bc[0, 0]
    h = 2.0
    tau_i = 0.5

    # Analytical formulas for scalar case (a != 0)
    e_ah = np.exp(a * h)
    e_a_h_tau = np.exp(a * (h - tau_i))

    int_Ai_analytic = (b / a) * (e_ah - e_a_h_tau)
    int_Bi_analytic = (b / a) * (e_a_h_tau - 1.0)

    Ai_expected = np.block([
        [e_ah,          int_Ai_analytic],
        [0.0,           0.0            ]
    ])

    Bi_expected = np.block([
        [int_Bi_analytic],
        [1.0            ]
    ])

    Ai_calc, Bi_calc = compute_Ai_Bi(Ac, Bc, h, tau_i)

    print("\nTest Scalar:")
    print("Ai expected:\n", Ai_expected)
    print("Ai calculated:\n", Ai_calc)
    print("Bi expected:\n", Bi_expected)
    print("Bi calculated:\n", Bi_calc)

    assert np.allclose(Ai_calc, Ai_expected), "Scalar Test Failed for Ai"
    assert np.allclose(Bi_calc, Bi_expected), "Scalar Test Failed for Bi"


# Test Case 3: Invertible Ac - Compare integral computation
def test_invertible_Ac_integral():
    Ac = np.array([[1.0, 1.0], [0.0, -0.5]])
    Bc = np.array([[1.0], [0.5]])
    h = 1.0
    tau_i = 0.2
    t_lower = h - tau_i
    t_upper = h

    # Compute using augmented matrix method
    exp_Ac_upper, phi_upper = compute_exp_integral(Ac, Bc, t_upper)
    exp_Ac_lower, phi_lower = compute_exp_integral(Ac, Bc, t_lower)

    integral_Ai_augmented = phi_upper - phi_lower
    integral_Bi_augmented = phi_lower

    # Compute using Ac inverse method: integral_a^b = Ac^{-1} (e^{Ac b} - e^{Ac a}) Bc
    try:
        Ac_inv = inv(Ac)
        integral_Ai_inv = Ac_inv @ (exp_Ac_upper - exp_Ac_lower) @ Bc
        integral_Bi_inv = Ac_inv @ (exp_Ac_lower - np.eye(Ac.shape[0])) @ Bc # Integral from 0 to t_lower

        print("\nTest Invertible Ac Integral:")
        print("Integral Ai (Augmented):\n", integral_Ai_augmented)
        print("Integral Ai (Inverse):\n", integral_Ai_inv)
        print("Integral Bi (Augmented):\n", integral_Bi_augmented)
        print("Integral Bi (Inverse):\n", integral_Bi_inv)

        assert np.allclose(integral_Ai_augmented, integral_Ai_inv), "Invertible test failed for Ai integral"
        assert np.allclose(integral_Bi_augmented, integral_Bi_inv), "Invertible test failed for Bi integral"

    except np.linalg.LinAlgError:
        print("Skipping invertible test as Ac is singular.")


# Test Case 4: Edge case tau_i = 0
def test_tau_i_zero():
    n, m = 2, 1
    Ac = np.array([[0.1, 0.0], [0.2, -0.1]])
    Bc = np.array([[1.0], [0.0]])
    h = 1.5
    tau_i = 0.0

    Ai_calc, Bi_calc = compute_Ai_Bi(Ac, Bc, h, tau_i)

    # Expected results when tau_i = 0:
    # Integral Ai = integral_h^h = 0
    # Integral Bi = integral_0^h = Phi(h)
    exp_Ac_h, phi_h = compute_exp_integral(Ac, Bc, h)

    Ai_expected = np.block([
        [exp_Ac_h, np.zeros((n,m))],
        [np.zeros((m,n)), np.zeros((m,m))]
    ])

    Bi_expected = np.block([
        [phi_h],
        [np.eye(m)]
    ])

    print("\nTest tau_i = 0:")
    print("Ai calculated:\n", Ai_calc)
    print("Bi calculated:\n", Bi_calc)

    assert np.allclose(Ai_calc, Ai_expected), "Test tau_i=0 Failed for Ai"
    assert np.allclose(Bi_calc, Bi_expected), "Test tau_i=0 Failed for Bi"


# Test Case 5: Edge case tau_i = h
def test_tau_i_h():
    n, m = 2, 1
    Ac = np.array([[0.1, 0.0], [0.2, -0.1]])
    Bc = np.array([[1.0], [0.0]])
    h = 1.5
    tau_i = h # tau_i = 1.5

    Ai_calc, Bi_calc = compute_Ai_Bi(Ac, Bc, h, tau_i)

    # Expected results when tau_i = h:
    # Integral Ai = integral_0^h = Phi(h)
    # Integral Bi = integral_0^0 = 0
    exp_Ac_h, phi_h = compute_exp_integral(Ac, Bc, h)

    Ai_expected = np.block([
        [exp_Ac_h, phi_h],
        [np.zeros((m,n)), np.zeros((m,m))]
    ])

    Bi_expected = np.block([
        [np.zeros((n,m))],
        [np.eye(m)]
    ])

    print("\nTest tau_i = h:")
    print("Ai calculated:\n", Ai_calc)
    print("Bi calculated:\n", Bi_calc)

    assert np.allclose(Ai_calc, Ai_expected), "Test tau_i=h Failed for Ai"
    assert np.allclose(Bi_calc, Bi_expected), "Test tau_i=h Failed for Bi"


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
"""
Test integrals
"""
# print("Running tests...")
# test_Ac_zero()
# test_scalar()
# test_invertible_Ac_integral()
# test_tau_i_zero()
# test_tau_i_h()
""""""
dynamics = [None]
Ac = np.array([[0., 1.],
               [0., 0.]])
Bc = np.array([[0.],
               [1.]])
for node in range(1, tree.num_nodes):
    t = tree.data_values[node][0]
    A, B = compute_Ai_Bi(Ac, Bc, max_delay, t)
    dynamics += [s.build.Dynamics(A, B)]

# Costs
zero = 1e-6
nonleaf_costs = [None]
Q = np.diag([1. for _ in range(num_states)] + [zero for _ in range(num_inputs)])
R = np.diag([10. for _ in range(num_inputs)])
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
alpha = .95
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
    .generate_problem()
)
print(problem)

# --------------------------------------------------------
# Initial state
# --------------------------------------------------------
if br == 0:
    x0 = np.zeros(state_size)
    for k in range(num_states):
        x0[k] = 2.
    tree.write_to_file_fp("initialState", x0)
