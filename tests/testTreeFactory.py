import sys
sys.path.append(sys.path[0] + '/..')
import py.treeFactory as factories
import unittest
import numpy as np


class TestScenarioTree(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None

    @staticmethod
    def __construct_tree_from_markov():
        if TestScenarioTree.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0.0])
            (N, tau) = (4, 3)
            TestScenarioTree.__tree_from_markov = \
                factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=True)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestScenarioTree.__construct_tree_from_markov()
    
    def test_markov_num_nodes(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(32, tree.num_nodes)

    def test_markov_num_nonleaf_nodes(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(20, tree.num_nonleaf_nodes)

    def test_markov_num_events(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(3, tree.num_events)

    def test_markov_ancestor_of_node(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(0, tree.ancestor_of_node(1))
        self.assertEqual(0, tree.ancestor_of_node(2))
        self.assertEqual(1, tree.ancestor_of_node(3))
        self.assertEqual(1, tree.ancestor_of_node(4))
        self.assertEqual(1, tree.ancestor_of_node(5))
        self.assertEqual(2, tree.ancestor_of_node(6))
        self.assertEqual(2, tree.ancestor_of_node(7))
        self.assertEqual(3, tree.ancestor_of_node(8))
        self.assertEqual(3, tree.ancestor_of_node(9))
        self.assertEqual(3, tree.ancestor_of_node(10))
        self.assertEqual(5, tree.ancestor_of_node(13))
        for i in range(12):
            self.assertEqual(8 + i, tree.ancestor_of_node(20 + i))

    def test_markov_children_of_node(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(2, len(tree.children_of_node(0)))
        self.assertEqual(3, len(tree.children_of_node(1)))
        self.assertEqual(2, len(tree.children_of_node(2)))
        self.assertEqual(2, len(tree.children_of_node(5)))
        self.assertEqual(3, len(tree.children_of_node(6)))
        for idx in range(8, 20):
            self.assertEqual(1, len(tree.children_of_node(idx)))

    def test_markov_children_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_markov
        with self.assertRaises(IndexError):
            _ = tree.children_of_node(20)

    def test_markov_stage_of_node(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(0, tree.stage_of_node(0))
        self.assertEqual(1, tree.stage_of_node(1))
        self.assertEqual(1, tree.stage_of_node(2))
        for idx in range(3, 8):
            self.assertEqual(2, tree.stage_of_node(idx))
        for idx in range(8, 20):
            self.assertEqual(3, tree.stage_of_node(idx))
        for idx in range(20, 32):
            self.assertEqual(4, tree.stage_of_node(idx))

    def test_markov_stage_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_markov
        with self.assertRaises(ValueError):
            _ = tree.stage_of_node(-1)
        with self.assertRaises(IndexError):
            _ = tree.stage_of_node(32)

    def test_markov_num_stages(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(5, tree.num_stages)

    def test_markov_nodes_of_stage(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(1, len(tree.nodes_of_stage(0)))
        self.assertEqual(2, len(tree.nodes_of_stage(1)))
        self.assertEqual(5, len(tree.nodes_of_stage(2)))
        self.assertEqual(12, len(tree.nodes_of_stage(3)))
        self.assertEqual(12, len(tree.nodes_of_stage(4)))
        self.assertTrue(([1, 2] == tree.nodes_of_stage(1)).all())
        self.assertTrue((range(3, 8) == tree.nodes_of_stage(2)).all())
        self.assertTrue((range(8, 20) == tree.nodes_of_stage(3)).all())
        self.assertTrue((range(20, 32) == tree.nodes_of_stage(4)).all())

    def test_markov_probability_of_node(self):
        tol = 1e-10
        tree = TestScenarioTree.__tree_from_markov
        self.assertAlmostEqual(1, tree.probability_of_node(0), delta=tol)
        self.assertAlmostEqual(0.5, tree.probability_of_node(1), delta=tol)
        self.assertAlmostEqual(0.5, tree.probability_of_node(2), delta=tol)
        self.assertAlmostEqual(0.05, tree.probability_of_node(3), delta=tol)
        self.assertAlmostEqual(0.4, tree.probability_of_node(4), delta=tol)
        self.assertAlmostEqual(0.05, tree.probability_of_node(5), delta=tol)
        self.assertAlmostEqual(0.2, tree.probability_of_node(6), delta=tol)
        self.assertAlmostEqual(0.3, tree.probability_of_node(7), delta=tol)
        self.assertAlmostEqual(0.005, tree.probability_of_node(8), delta=tol)
        self.assertAlmostEqual(0.005, tree.probability_of_node(20), delta=tol)
        self.assertAlmostEqual(0.5 * 0.4 * 0.1, tree.probability_of_node(29), delta=tol)

    def test_markov_siblings_of_node(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertEqual(1, len(tree.siblings_of_node(0)))
        self.assertEqual(2, len(tree.siblings_of_node(1)))
        self.assertEqual(2, len(tree.siblings_of_node(2)))
        self.assertEqual(3, len(tree.siblings_of_node(3)))
        self.assertEqual(2, len(tree.siblings_of_node(7)))
        self.assertEqual(2, len(tree.siblings_of_node(11)))
        for i in range(20, 32):
            self.assertEqual(1, len(tree.siblings_of_node(i)))

    def test_markov_values(self):
        tree = TestScenarioTree.__tree_from_markov
        self.assertTrue(([0, 1] == tree.event_of_node(range(1, 3))).all())
        self.assertTrue(([0, 1, 2, 0, 1] == tree.event_of_node(range(3, 8))).all())
        self.assertTrue(([0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1] == tree.event_of_node(range(8, 20))).all())
        self.assertTrue((tree.event_of_node(range(8, 20)) == tree.event_of_node(range(20, 32))).all())

    def test_markov_cond_prob_of_children_of_node(self):
        tol = 1e-5
        tree = TestScenarioTree.__tree_from_markov
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_markov_cond_prob_of_children_of_node_large_tree(self):
        n, tol = 4, 1e-10
        p = np.random.rand(n, n)
        for i in range(n):
            p[i, :] /= sum(p[i, :])
        v = np.random.rand(n, )
        v /= sum(v)
        (N, tau) = (20, 5)
        tree = factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=True)
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_stopping_stage_failure(self):
        n = 3
        p = np.random.rand(n, n)
        for i in range(n):
            p[i, :] /= sum(p[i, :])
        v = np.random.rand(n, )
        v /= sum(v)
        (N, tau) = (4, 5)
        with self.assertRaises(ValueError):
            _ = factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=True)

    def test_stage_and_stop_is_one(self):
        p = np.array([[0.1, 0.8, 0.1],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7]])
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (1, 1)
        _ = factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=True)

    def test_stop_is_one(self):
        p = np.array([[0.1, 0.8, 0.1],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7]])
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (3, 1)
        _ = factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=True)

    def test_generate_not_test_tree(self):
        p = np.array([[0.1, 0.8, 0.1],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7]])
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (3, 1)
        _ = factories.TreeFactoryMarkovChain(p, v, N, tau).generate_tree(testing=False)


if __name__ == '__main__':
    unittest.main()
