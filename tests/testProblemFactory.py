import unittest
import numpy as np
import py
import py.build as b


class TestProblem(unittest.TestCase):
    __tree_from_markov = None
    __tree_from_iid = None
    __problem_from_markov = None
    __problem_from_markov_with_markov = None
    __problem_from_iid = None
    __num_states = 3
    __num_inputs = 2

    @staticmethod
    def _construct_tree_from_markov():
        if TestProblem.__tree_from_markov is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.4, 0.1])
            (N, tau) = (4, 3)
            TestProblem.__tree_from_markov = \
                py.treeFactory.TreeFactoryMarkovChain(p, v, N, tau).generate_tree()

    @staticmethod
    def _construct_problem_from_markov():
        if TestProblem.__problem_from_markov is None:
            tree = TestProblem.__tree_from_markov

            # construct markovian set of system and control dynamics
            system = np.random.randn(TestProblem.__num_states * TestProblem.__num_states).reshape(
                TestProblem.__num_states, TestProblem.__num_states)
            control = np.random.randn(TestProblem.__num_states * TestProblem.__num_inputs).reshape(
                TestProblem.__num_states, TestProblem.__num_inputs)
            dynamics = [b.LinearDynamics(system, control),
                        b.LinearDynamics(2 * system, 2 * control),
                        b.LinearDynamics(3 * system, 3 * control)]

            # construct cost weight matrices
            nonleaf_state_weight = 10 * np.eye(TestProblem.__num_states)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(TestProblem.__num_inputs)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            nonleaf_costs = [b.NonleafCost(nonleaf_state_weights[0], control_weights[0]),
                             b.NonleafCost(nonleaf_state_weights[1], control_weights[1]),
                             b.NonleafCost(nonleaf_state_weights[2], control_weights[2])]
            leaf_cost = b.LeafCost(5 * np.eye(TestProblem.__num_states))

            # State-input constraint
            state_lim = 6
            input_lim = 0.3
            state_lb = -state_lim * np.ones((TestProblem.__num_states, 1))
            state_ub = state_lim * np.ones((TestProblem.__num_states, 1))
            input_lb = -input_lim * np.ones((TestProblem.__num_inputs, 1))
            input_ub = input_lim * np.ones((TestProblem.__num_inputs, 1))
            si_lb = np.vstack((state_lb, input_lb))
            si_ub = np.vstack((state_ub, input_ub))
            state_input_constraint = b.Rectangle(si_lb, si_ub)

            # Terminal constraint
            leaf_state_lim = 0.1
            leaf_state_lb = -leaf_state_lim * np.ones((TestProblem.__num_states, 1))
            leaf_state_ub = leaf_state_lim * np.ones((TestProblem.__num_states, 1))
            leaf_state_constraint = b.Rectangle(leaf_state_lb, leaf_state_ub)

            # define risks
            alpha = 0.5
            risks = b.AVaR(alpha)

            # create problem
            TestProblem.__problem_from_markov = (
                py.problemFactory.ProblemFactory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_markovian_dynamics(dynamics)
                .with_nonleaf_cost(nonleaf_costs[0])
                .with_leaf_cost(leaf_cost)
                .with_nonleaf_constraint(state_input_constraint)
                .with_leaf_constraint(leaf_state_constraint)
                .with_risk(risks)
                .with_tests()
            ).generate_problem()

            TestProblem.__problem_from_markov_with_markov = (
                py.problemFactory.ProblemFactory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_markovian_dynamics(dynamics)
                .with_markovian_nonleaf_costs(nonleaf_costs)
                .with_leaf_cost(leaf_cost)
                .with_nonleaf_constraint(state_input_constraint)
                .with_leaf_constraint(leaf_state_constraint)
                .with_risk(risks)
                .with_tests()
            ).generate_problem()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestProblem._construct_tree_from_markov()
        TestProblem._construct_problem_from_markov()

    def test_markovian_dynamics_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov_with_markov
        for i in range(0, tree.num_nodes):
            self.assertTrue(problem.dynamics_at_node(i) is not None)

    def test_markovian_nonleaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov_with_markov
        for i in range(tree.num_nodes):
            self.assertTrue(problem.nonleaf_cost_at_node(i) is not None)

    def test_all_nonleaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        for i in range(tree.num_nodes):
            self.assertTrue(problem.nonleaf_cost_at_node(i) is not None)

    def test_leaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        for i in range(tree.num_leaf_nodes):
            self.assertTrue(problem.leaf_cost_at_node(i) is not None)

    def test_no_constraints_loaded(self):
        problem = TestProblem.__problem_from_markov_with_markov
        self.assertTrue(problem.nonleaf_constraint() is not None)
        self.assertTrue(problem.leaf_constraint() is not None)

    def test_risks_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(problem.risk_at_node(i) is not None)


if __name__ == '__main__':
    unittest.main()
