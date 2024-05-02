import py.treeFactory as factories
import numpy as np

# --------------------------------------------------------
# Generate scenario tree
# --------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])

v = np.array([0.6, 0.4])

(horizon, stopping_stage) = (2, 2)
tree = factories.TreeFactoryMarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=horizon,
    stopping_stage=stopping_stage
).generate_tree()

# tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
print(sum(tree.probability_of_node(tree.nodes_of_stage(2))))
print(tree)

# --------------------------------------------------------
# Generate problem data
# --------------------------------------------------------

(num_states, num_inputs) = 3, 2
factor = 0.1

A = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
B = np.array([[3, 0], [1, 0], [0, 2]])
As = [1.5 * A, A, -1.5 * A]
Bs = [-1.5 * B, B, 1.5 * B]

Q = np.eye(num_states)
R = np.eye(num_inputs)
Qs = [2 * Q, 2 * Q, 2 * Q]
Rs = [1 * R, 1 * R, 1 * R]
T = np.eye(num_states)

nonleaf_size = num_states + num_inputs
leaf_size = num_states
x_lim = 6
u_lim = 0.3
nl_min = np.vstack((-x_lim * np.ones((num_states, 1)),
                    -u_lim * np.ones((num_inputs, 1))))
nl_max = np.vstack((x_lim * np.ones((num_states, 1)),
                    u_lim * np.ones((num_inputs, 1))))
l_min = -x_lim * np.ones((leaf_size, 1))
l_max = x_lim * np.ones((leaf_size, 1))
nl_rect = rectangle.Rectangle(nl, nl_min, nl_max)
l_rect = rectangle.Rectangle(l, l_min, l_max)

alpha = .95
risk_1 = risks.AVaR(0.1)
risk_5 = risks.AVaR(0.5)
risk_9 = risks.AVaR(0.9)

problem_1 = r.core.RAOCP(scenario_tree=tree) \
    .with_markovian_dynamics(mark_dynamics) \
    .with_markovian_nonleaf_costs(mark_nl_costs) \
    .with_all_leaf_costs(leaf_cost) \
    .with_all_risks(risk_1) \
    .with_all_nonleaf_constraints(nl_rect) \
    .with_all_leaf_constraints(l_rect)
