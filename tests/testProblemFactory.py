import unittest
import numpy as np
import py


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
            set_system = [system, 2 * system, 3 * system]  # n x n matrices
            control = np.random.randn(TestProblem.__num_states * TestProblem.__num_inputs).reshape(
                TestProblem.__num_states, TestProblem.__num_inputs)
            set_control = [control, 2 * control, 3 * control]  # n x u matrices

            # construct cost weight matrices
            nonleaf_state_weight = 10 * np.eye(2)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(2)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            leaf_state_weight = 5 * np.eye(2)  # n x n matrix

            # construct constraint min and max
            state_lim = 2
            state_min = -state_lim * np.ones((TestProblem.__num_states, 1))
            state_max = state_lim * np.ones((TestProblem.__num_states, 1))
            input_lim = 1
            input_min = -input_lim * np.ones((TestProblem.__num_inputs, 1))
            input_max = input_lim * np.ones((TestProblem.__num_inputs, 1))
            state_rect = py.build.Rectangle(state_min, state_max)
            input_rect = py.build.Rectangle(input_min, input_max)

            # define risks
            alpha = 0.5
            risks = py.build.AVaR(alpha)

            # create problem
            TestProblem.__problem_from_markov = (
                py.problemFactory.ProblemFactory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_markovian_dynamics(set_system, set_control)
                .with_nonleaf_cost(nonleaf_state_weight, control_weight)
                .with_leaf_cost(leaf_state_weight)
                .with_risk(risks)
            ).generate_problem()

            TestProblem.__problem_from_markov_with_markov = (
                py.problemFactory.ProblemFactory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_markovian_dynamics(set_system, set_control)
                .with_markovian_nonleaf_costs(nonleaf_state_weights, control_weights)
                .with_leaf_cost(leaf_state_weight)
                .with_constraint(state_rect, input_rect)
                .with_risk(risks)
            ).generate_problem()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestProblem._construct_tree_from_markov()
        TestProblem._construct_problem_from_markov()

    def test_markovian_dynamics_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov_with_markov
        self.assertTrue(problem.state_dynamics_at_node(0) is None)
        self.assertTrue(problem.input_dynamics_at_node(0) is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(problem.state_dynamics_at_node(i) is not None)
            self.assertTrue(problem.input_dynamics_at_node(i) is not None)

    def test_markovian_nonleaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov_with_markov
        self.assertTrue(problem.nonleaf_state_cost_at_node(0) is None)
        self.assertTrue(problem.nonleaf_input_cost_at_node(0) is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(problem.nonleaf_state_cost_at_node(i) is not None)
            self.assertTrue(problem.nonleaf_input_cost_at_node(i) is not None)

    def test_all_nonleaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        self.assertTrue(problem.nonleaf_state_cost_at_node(0) is None)
        self.assertTrue(problem.nonleaf_input_cost_at_node(0) is None)
        for i in range(1, tree.num_nodes):
            self.assertTrue(problem.nonleaf_state_cost_at_node(i) is not None)
            self.assertTrue(problem.nonleaf_input_cost_at_node(i) is not None)

    def test_leaf_costs_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(problem.leaf_state_cost_at_node(i) is None)
            else:
                self.assertTrue(problem.leaf_state_cost_at_node(i) is not None)

    def test_no_constraints_loaded(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov_with_markov
        for i in range(tree.num_nodes):
            if i < tree.num_nonleaf_nodes:
                self.assertTrue(problem.state_constraint_at_node(i) is not None)
                self.assertTrue(problem.input_constraint_at_node(i) is not None)
            else:
                self.assertTrue(problem.state_constraint_at_node(i) is not None)

    def test_risks_list(self):
        tree = TestProblem.__tree_from_markov
        problem = TestProblem.__problem_from_markov
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(problem.risk_at_node(i) is not None)


if __name__ == '__main__':
    unittest.main()
