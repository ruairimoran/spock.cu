import unittest
import numpy as np
import spock as s


class TestScenarioTree(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __tree_from_data = None
    __tree_from_structure = None

    @staticmethod
    def __construct_tree_from_markov():
        if TestScenarioTree.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.5, 0.0])
            (N, tau) = (4, 3)
            TestScenarioTree.__tree_from_markov = s.tree.MarkovChain(p, v, N, tau).build()

    @staticmethod
    def __construct_tree_from_iid():
        if TestScenarioTree.__tree_from_iid is None:
            v = np.array([0.1, 0.2, 0.7])
            (N, tau) = (4, 2)
            TestScenarioTree.__tree_from_iid = s.tree.IidProcess(v, N, tau).build()

    @staticmethod
    def __construct_tree_from_data():
        if TestScenarioTree.__tree_from_data is None:
            col = np.ones((10, 1))
            for i in range(col.size):
                col[i] = .1 * i
            data = np.dstack((.1 * col, .2 * col, .3 * col, .4 * col))  # samples x dim x time
            branching = [3, 2, 1, 1]
            TestScenarioTree.__tree_from_data = s.tree.FromData(data, branching).build()

    @staticmethod
    def __construct_tree_from_structure():
        if TestScenarioTree.__tree_from_structure is None:
            stages = [
                0,
                1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
            ]
            anc = [
                -1,
                0, 0,
                1, 1, 1,
                2, 2,
                3, 3, 3,
                4, 4,
                5, 5,
                6, 6, 6,
                7, 7,
                8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
            ]
            probs = [
                1.0,
                0.5, 0.5,
                0.05, 0.4, 0.05, 0.2, 0.3,
                0.005, 0.04, 0.005, 0.16, 0.24, 0.015, 0.035, 0.02, 0.16, 0.02, 0.12, 0.18,
                0.005, 0.04, 0.005, 0.16, 0.24, 0.015, 0.035, 0.02, 0.16, 0.02, 0.12, 0.18]
            data = np.array([i for i in range(len(stages))])
            TestScenarioTree.__tree_from_structure = s.tree.FromStructure(stages, anc, probs, data).build()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestScenarioTree.__construct_tree_from_markov()
        TestScenarioTree.__construct_tree_from_iid()
        TestScenarioTree.__construct_tree_from_data()
        TestScenarioTree.__construct_tree_from_structure()

    """
    Markov tests
    """
    
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
        tree = s.tree.MarkovChain(p, v, N, tau).build()
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_markov_stopping_stage_failure(self):
        n = 3
        p = np.random.rand(n, n)
        for i in range(n):
            p[i, :] /= sum(p[i, :])
        v = np.random.rand(n, )
        v /= sum(v)
        (N, tau) = (4, 5)
        with self.assertRaises(ValueError):
            _ = s.tree.MarkovChain(p, v, N, tau).build()

    def test_markov_stage_and_stop_is_one(self):
        p = np.array([[0.1, 0.8, 0.1],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7]])
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (1, 1)
        _ = s.tree.MarkovChain(p, v, N, tau).build()

    def test_markov_stop_is_one(self):
        p = np.array([[0.1, 0.8, 0.1],
                      [0.4, 0.6, 0],
                      [0, 0.3, 0.7]])
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (3, 1)
        _ = s.tree.MarkovChain(p, v, N, tau).build()

    """
    Iid tests
    """

    def test_iid_num_nodes(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(31, tree.num_nodes)

    def test_iid_num_nonleaf_nodes(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(22, tree.num_nonleaf_nodes)

    def test_iid_num_events(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(3, tree.num_events)

    def test_iid_ancestor_of_node(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(0, tree.ancestor_of_node(1))
        self.assertEqual(0, tree.ancestor_of_node(2))
        self.assertEqual(0, tree.ancestor_of_node(3))
        self.assertEqual(1, tree.ancestor_of_node(4))
        self.assertEqual(1, tree.ancestor_of_node(5))
        self.assertEqual(1, tree.ancestor_of_node(6))
        self.assertEqual(2, tree.ancestor_of_node(7))
        self.assertEqual(2, tree.ancestor_of_node(8))
        self.assertEqual(2, tree.ancestor_of_node(9))
        self.assertEqual(3, tree.ancestor_of_node(10))
        self.assertEqual(3, tree.ancestor_of_node(11))
        self.assertEqual(3, tree.ancestor_of_node(12))
        for i in range(13, tree.num_nodes):
            self.assertEqual(i - 9, tree.ancestor_of_node(i))

    def test_iid_children_of_node(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(3, len(tree.children_of_node(0)))
        self.assertEqual(3, len(tree.children_of_node(1)))
        self.assertEqual(3, len(tree.children_of_node(2)))
        self.assertEqual(3, len(tree.children_of_node(3)))
        for idx in range(4, tree.num_nonleaf_nodes):
            self.assertEqual(1, len(tree.children_of_node(idx)))

    def test_iid_children_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_iid
        with self.assertRaises(IndexError):
            _ = tree.children_of_node(22)

    def test_iid_stage_of_node(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(0, tree.stage_of_node(0))
        for idx in range(1, 4):
            self.assertEqual(1, tree.stage_of_node(idx))
        for idx in range(4, 13):
            self.assertEqual(2, tree.stage_of_node(idx))
        for idx in range(13, 22):
            self.assertEqual(3, tree.stage_of_node(idx))
        for idx in range(22, 31):
            self.assertEqual(4, tree.stage_of_node(idx))

    def test_iid_stage_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_iid
        with self.assertRaises(ValueError):
            _ = tree.stage_of_node(-1)
        with self.assertRaises(IndexError):
            _ = tree.stage_of_node(31)

    def test_iid_num_stages(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(5, tree.num_stages)

    def test_iid_nodes_of_stage(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(1, len(tree.nodes_of_stage(0)))
        self.assertEqual(3, len(tree.nodes_of_stage(1)))
        self.assertEqual(9, len(tree.nodes_of_stage(2)))
        self.assertEqual(9, len(tree.nodes_of_stage(3)))
        self.assertEqual(9, len(tree.nodes_of_stage(4)))
        self.assertTrue(([1, 2, 3] == tree.nodes_of_stage(1)).all())
        self.assertTrue((range(4, 13) == tree.nodes_of_stage(2)).all())
        self.assertTrue((range(13, 22) == tree.nodes_of_stage(3)).all())
        self.assertTrue((range(22, 31) == tree.nodes_of_stage(4)).all())

    def test_iid_probability_of_node(self):
        tol = 1e-10
        tree = TestScenarioTree.__tree_from_iid
        self.assertAlmostEqual(1., tree.probability_of_node(0), delta=tol)
        self.assertAlmostEqual(.1, tree.probability_of_node(1), delta=tol)
        self.assertAlmostEqual(.2, tree.probability_of_node(2), delta=tol)
        self.assertAlmostEqual(.7, tree.probability_of_node(3), delta=tol)
        self.assertAlmostEqual(.01, tree.probability_of_node(4), delta=tol)
        self.assertAlmostEqual(.02, tree.probability_of_node(5), delta=tol)
        self.assertAlmostEqual(.07, tree.probability_of_node(6), delta=tol)
        self.assertAlmostEqual(.02, tree.probability_of_node(7), delta=tol)
        self.assertAlmostEqual(.04, tree.probability_of_node(8), delta=tol)
        self.assertAlmostEqual(.14, tree.probability_of_node(9), delta=tol)
        self.assertAlmostEqual(.07, tree.probability_of_node(10), delta=tol)
        self.assertAlmostEqual(.14, tree.probability_of_node(11), delta=tol)
        self.assertAlmostEqual(.49, tree.probability_of_node(12), delta=tol)
        for idx in range(13, tree.num_nodes):
            anc = tree.ancestor_of_node(idx)
            self.assertAlmostEqual(tree.probability_of_node(anc), tree.probability_of_node(idx), delta=tol)

    def test_iid_siblings_of_node(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertEqual(1, len(tree.siblings_of_node(0)))
        for idx in range(1, 13):
            self.assertEqual(3, len(tree.siblings_of_node(idx)))
        for idx in range(13, tree.num_nodes):
            self.assertEqual(1, len(tree.siblings_of_node(idx)))

    def test_iid_values(self):
        tree = TestScenarioTree.__tree_from_iid
        self.assertTrue(([0, 1, 2] == tree.event_of_node(range(1, 4))).all())
        self.assertTrue(([0, 1, 2, 0, 1, 2, 0, 1, 2] == tree.event_of_node(range(4, 13))).all())
        self.assertTrue((tree.event_of_node(range(4, 13)) == tree.event_of_node(range(13, 22))).all())
        self.assertTrue((tree.event_of_node(range(4, 13)) == tree.event_of_node(range(22, 31))).all())

    def test_iid_cond_prob_of_children_of_node(self):
        tol = 1e-5
        tree = TestScenarioTree.__tree_from_iid
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_iid_cond_prob_of_children_of_node_large_tree(self):
        n, tol = 4, 1e-10
        v = np.random.rand(n,)
        v /= sum(v)
        (N, tau) = (20, 5)
        tree = s.tree.IidProcess(v, N, tau).build()
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_iid_stopping_stage_failure(self):
        n = 3
        v = np.random.rand(n,)
        v /= sum(v)
        (N, tau) = (4, 5)
        with self.assertRaises(ValueError):
            _ = s.tree.IidProcess(v, N, tau).build()

    def test_iid_stage_and_stop_is_one(self):
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (1, 1)
        _ = s.tree.IidProcess(v, N, tau).build()

    def test_iid_stop_is_one(self):
        v = np.array([0.5, 0.4, 0.1])
        (N, tau) = (3, 1)
        _ = s.tree.IidProcess(v, N, tau).build()

    """
    Data tests
    """

    def test_data_num_nodes(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(22, tree.num_nodes)

    def test_data_num_nonleaf_nodes(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(16, tree.num_nonleaf_nodes)

    def test_data_num_events(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(len(tree.children_of_node(0)), tree.num_events)

    def test_data_ancestor_of_node(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(0, tree.ancestor_of_node(1))
        self.assertEqual(0, tree.ancestor_of_node(2))
        self.assertEqual(0, tree.ancestor_of_node(3))
        self.assertEqual(1, tree.ancestor_of_node(4))
        self.assertEqual(1, tree.ancestor_of_node(5))
        self.assertEqual(2, tree.ancestor_of_node(6))
        self.assertEqual(2, tree.ancestor_of_node(7))
        self.assertEqual(3, tree.ancestor_of_node(8))
        self.assertEqual(3, tree.ancestor_of_node(9))
        for i in range(10, tree.num_nodes):
            self.assertEqual(i - 6, tree.ancestor_of_node(i))

    def test_data_children_of_node(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(3, len(tree.children_of_node(0)))
        self.assertEqual(2, len(tree.children_of_node(1)))
        self.assertEqual(2, len(tree.children_of_node(2)))
        self.assertEqual(2, len(tree.children_of_node(3)))
        for idx in range(4, tree.num_nonleaf_nodes):
            self.assertEqual(1, len(tree.children_of_node(idx)))

    def test_data_children_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_data
        with self.assertRaises(IndexError):
            _ = tree.children_of_node(20)

    def test_data_stage_of_node(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(0, tree.stage_of_node(0))
        for idx in range(1, 4):
            self.assertEqual(1, tree.stage_of_node(idx))
        for idx in range(4, 10):
            self.assertEqual(2, tree.stage_of_node(idx))
        for idx in range(10, 16):
            self.assertEqual(3, tree.stage_of_node(idx))
        for idx in range(16, 22):
            self.assertEqual(4, tree.stage_of_node(idx))

    def test_data_stage_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_data
        with self.assertRaises(ValueError):
            _ = tree.stage_of_node(-1)
        with self.assertRaises(IndexError):
            _ = tree.stage_of_node(22)

    def test_data_num_stages(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(5, tree.num_stages)

    def test_data_nodes_of_stage(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(1, len(tree.nodes_of_stage(0)))
        self.assertEqual(3, len(tree.nodes_of_stage(1)))
        self.assertEqual(6, len(tree.nodes_of_stage(2)))
        self.assertEqual(6, len(tree.nodes_of_stage(3)))
        self.assertEqual(6, len(tree.nodes_of_stage(4)))
        self.assertTrue((range(1, 4) == tree.nodes_of_stage(1)).all())
        self.assertTrue((range(4, 10) == tree.nodes_of_stage(2)).all())
        self.assertTrue((range(10, 16) == tree.nodes_of_stage(3)).all())
        self.assertTrue((range(16, 22) == tree.nodes_of_stage(4)).all())

    def test_data_probability_of_node(self):
        tol = 1e-10
        tree = TestScenarioTree.__tree_from_data
        self.assertAlmostEqual(1., tree.probability_of_node(0), delta=tol)
        self.assertAlmostEqual(.4, tree.probability_of_node(1), delta=tol)
        self.assertAlmostEqual(.3, tree.probability_of_node(2), delta=tol)
        self.assertAlmostEqual(.3, tree.probability_of_node(3), delta=tol)
        self.assertAlmostEqual(.3, tree.probability_of_node(4), delta=tol)
        self.assertAlmostEqual(.1, tree.probability_of_node(5), delta=tol)
        self.assertAlmostEqual(.2, tree.probability_of_node(6), delta=tol)
        self.assertAlmostEqual(.1, tree.probability_of_node(7), delta=tol)
        self.assertAlmostEqual(.2, tree.probability_of_node(8), delta=tol)
        self.assertAlmostEqual(.1, tree.probability_of_node(9), delta=tol)
        for idx in range(10, tree.num_nodes):
            anc = tree.ancestor_of_node(idx)
            self.assertAlmostEqual(tree.probability_of_node(anc), tree.probability_of_node(idx), delta=tol)

    def test_data_siblings_of_node(self):
        tree = TestScenarioTree.__tree_from_data
        self.assertEqual(1, len(tree.siblings_of_node(0)))
        for idx in range(1, 4):
            self.assertEqual(3, len(tree.siblings_of_node(idx)))
        for idx in range(4, 10):
            self.assertEqual(2, len(tree.siblings_of_node(idx)))
        for idx in range(10, tree.num_nodes):
            self.assertEqual(1, len(tree.siblings_of_node(idx)))

    def test_data_values(self):
        tree = TestScenarioTree.__tree_from_data
        for node in range(1, tree.num_nodes):
            self.assertEqual(0, tree.event_of_node(node))

    def test_data_cond_prob_of_children_of_node(self):
        tol = 1e-5
        tree = TestScenarioTree.__tree_from_data
        for stage in range(tree.num_stages - 1):
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_data_cond_prob_of_children_of_node_large_tree(self):
        tol = 1e-10
        N = 20
        data = np.random.random((10, 1, N))  # samples x dim x time
        branching = np.ones(N)
        branching[:5] *= 4
        tree = s.tree.FromData(data, branching).build()
        for stage in range(tree.num_stages - 1):
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    """
    Structure tests
    """

    def test_structure_num_nodes(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(32, tree.num_nodes)

    def test_structure_num_nonleaf_nodes(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(20, tree.num_nonleaf_nodes)

    def test_structure_num_events(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(3, tree.num_events)

    def test_structure_ancestor_of_node(self):
        tree = TestScenarioTree.__tree_from_structure
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

    def test_structure_children_of_node(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(2, len(tree.children_of_node(0)))
        self.assertEqual(3, len(tree.children_of_node(1)))
        self.assertEqual(2, len(tree.children_of_node(2)))
        self.assertEqual(2, len(tree.children_of_node(5)))
        self.assertEqual(3, len(tree.children_of_node(6)))
        for idx in range(8, 20):
            self.assertEqual(1, len(tree.children_of_node(idx)))

    def test_structure_children_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_structure
        with self.assertRaises(IndexError):
            _ = tree.children_of_node(20)

    def test_structure_stage_of_node(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(0, tree.stage_of_node(0))
        self.assertEqual(1, tree.stage_of_node(1))
        self.assertEqual(1, tree.stage_of_node(2))
        for idx in range(3, 8):
            self.assertEqual(2, tree.stage_of_node(idx))
        for idx in range(8, 20):
            self.assertEqual(3, tree.stage_of_node(idx))
        for idx in range(20, 32):
            self.assertEqual(4, tree.stage_of_node(idx))

    def test_structure_stage_of_node_failure(self):
        tree = TestScenarioTree.__tree_from_structure
        with self.assertRaises(ValueError):
            _ = tree.stage_of_node(-1)
        with self.assertRaises(IndexError):
            _ = tree.stage_of_node(32)

    def test_structure_num_stages(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(5, tree.num_stages)

    def test_structure_nodes_of_stage(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(1, len(tree.nodes_of_stage(0)))
        self.assertEqual(2, len(tree.nodes_of_stage(1)))
        self.assertEqual(5, len(tree.nodes_of_stage(2)))
        self.assertEqual(12, len(tree.nodes_of_stage(3)))
        self.assertEqual(12, len(tree.nodes_of_stage(4)))
        self.assertTrue(([1, 2] == tree.nodes_of_stage(1)).all())
        self.assertTrue((range(3, 8) == tree.nodes_of_stage(2)).all())
        self.assertTrue((range(8, 20) == tree.nodes_of_stage(3)).all())
        self.assertTrue((range(20, 32) == tree.nodes_of_stage(4)).all())

    def test_structure_probability_of_node(self):
        tol = 1e-10
        tree = TestScenarioTree.__tree_from_structure
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

    def test_structure_siblings_of_node(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertEqual(1, len(tree.siblings_of_node(0)))
        self.assertEqual(2, len(tree.siblings_of_node(1)))
        self.assertEqual(2, len(tree.siblings_of_node(2)))
        self.assertEqual(3, len(tree.siblings_of_node(3)))
        self.assertEqual(2, len(tree.siblings_of_node(7)))
        self.assertEqual(2, len(tree.siblings_of_node(11)))
        for i in range(20, 32):
            self.assertEqual(1, len(tree.siblings_of_node(i)))

    def test_structure_values(self):
        tree = TestScenarioTree.__tree_from_structure
        self.assertTrue((0 == tree.event_of_node(range(1, 32))).all())

    def test_structure_cond_prob_of_children_of_node(self):
        tol = 1e-5
        tree = TestScenarioTree.__tree_from_structure
        for stage in range(tree.num_stages - 1):  # 0, 1, ..., N-1
            for node_idx in tree.nodes_of_stage(stage):
                prob_child = tree.cond_prob_of_children_of_node(node_idx)
                sum_prob = sum(prob_child)
                self.assertAlmostEqual(1.0, sum_prob, delta=tol)

    def test_structure_stages_failure(self):
        stages = [0, 1, 1, 2, 2, 2, 2]
        anc = [-1, 0, 0, 1, 1, 2, 2]
        probs = [1., .4, .6, .2, .2, .3, .3]
        stages[0] = 1
        with self.assertRaises(Exception):
            _ = s.tree.FromStructure(stages, anc, probs).build()

    def test_structure_ancestors_failure(self):
        stages = [0, 1, 1, 2, 2, 2, 2]
        anc = [-1, 0, 0, 1, 1, 2, 2]
        probs = [1., .4, .6, .2, .2, .3, .3]
        anc[0] = 0
        with self.assertRaises(Exception):
            _ = s.tree.FromStructure(stages, anc, probs).build()

    def test_structure_probabilities_failure(self):
        stages = [0, 1, 1, 2, 2, 2, 2]
        anc = [-1, 0, 0, 1, 1, 2, 2]
        probs = [1., .4, .6, .2, .2, .3, .3]
        probs[1] = .3
        with self.assertRaises(Exception):
            _ = s.tree.FromStructure(stages, anc, probs).build()

    def test_structure_data_failure(self):
        stages = [0, 1, 1, 2, 2, 2, 2]
        anc = [-1, 0, 0, 1, 1, 2, 2]
        probs = [1., .4, .6, .2, .2, .3, .3]
        data = [i for i in range(len(stages))]
        with self.assertRaises(Exception):
            _ = s.tree.FromStructure(stages, anc, probs, data).build()


if __name__ == '__main__':
    unittest.main()
