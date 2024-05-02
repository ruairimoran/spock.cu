import py.treeFactory as factories
import numpy as np


# ScenarioTree generation ----------------------------------------------------------------

p = np.array([[0.5, 0.5], [0.5, 0.5]])

v = np.array([0.6, 0.4])

(horizon, stopping_stage) = (2, 2)
tree = factories.ScenarioTreeFactoryMarkovChain(
    transition_prob=p,
    initial_distribution=v,
    horizon=horizon,
    stopping_stage=stopping_stage
).generate_tree()

tree.bulls_eye_plot(dot_size=6, radius=300, filename='scenario-tree.eps')
print(sum(tree.probability_of_node(tree.nodes_of_stage(2))))
print(tree)
