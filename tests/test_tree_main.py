import sys
sys.path.append(sys.path[0] + '/..')
import py.tree_factories as factories
import numpy as np


# Generate scenario tree from stopped Markov chain for testing C++ tree functionality
transition_mat = np.array([[0.5, 0.5], [0.5, 0.5]])
initial_dist = np.array([0.6, 0.4])
(horizon, stopping_stage) = (2, 2)
tree = factories.ScenarioTreeFactoryMarkovChain(
    transition_prob=transition_mat, 
    initial_distribution=initial_dist, 
    horizon=horizon, 
    stopping_stage=stopping_stage
).generate_tree()
