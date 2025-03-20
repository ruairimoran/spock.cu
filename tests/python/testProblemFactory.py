import unittest
import numpy as np
import spock as s


class TestProblem(unittest.TestCase):
    __tree = None
    __problem = None
    __num_states = 3
    __num_inputs = 2

    @staticmethod
    def _construct_tree():
        if TestProblem.__tree is None:
            p = np.array([[0.1, 0.8, 0.1],
                          [0.4, 0.6, 0.],
                          [0, 0.3, 0.7]])
            v = np.array([0.5, 0.4, 0.1])
            (N, tau) = (4, 3)
            TestProblem.__tree = \
                s.tree.MarkovChain(p, v, N, tau).build()

    @staticmethod
    def _construct_problem():
        if TestProblem.__problem is None:
            tree = TestProblem.__tree

            # construct stochastic set of system and control dynamics
            system = np.random.randn(TestProblem.__num_states, TestProblem.__num_states)
            systems = [system, 2 * system, 3 * system]
            control = np.random.randn(TestProblem.__num_states, TestProblem.__num_inputs)
            controls = [control, 2 * control, 3 * control]
            dynamics = [s.build.LinearDynamics(systems[0], controls[0]),
                        s.build.LinearDynamics(systems[1], controls[1]),
                        s.build.LinearDynamics(systems[2], controls[2])]

            # construct cost weight matrices
            nonleaf_state_weight = 10 * np.eye(TestProblem.__num_states)  # n x n matrix
            nonleaf_state_weights = [nonleaf_state_weight, 2 * nonleaf_state_weight, 3 * nonleaf_state_weight]
            control_weight = np.eye(TestProblem.__num_inputs)  # u x u matrix OR scalar
            control_weights = [control_weight, 2 * control_weight, 3 * control_weight]
            nonleaf_costs = [s.build.NonleafCost(nonleaf_state_weights[0], control_weights[0]),
                             s.build.NonleafCost(nonleaf_state_weights[1], control_weights[1]),
                             s.build.NonleafCost(nonleaf_state_weights[2], control_weights[2])]
            leaf_cost = s.build.LeafCost(5 * np.eye(TestProblem.__num_states))

            # state-input constraint
            state_lim = 6
            input_lim = 0.3
            state_lb = -state_lim * np.ones((TestProblem.__num_states, 1))
            state_ub = state_lim * np.ones((TestProblem.__num_states, 1))
            input_lb = -input_lim * np.ones((TestProblem.__num_inputs, 1))
            input_ub = input_lim * np.ones((TestProblem.__num_inputs, 1))
            si_lb = np.vstack((state_lb, input_lb))
            si_ub = np.vstack((state_ub, input_ub))
            state_input_constraint = s.build.Rectangle(si_lb, si_ub)

            # terminal constraint
            leaf_state_lim = 0.1
            leaf_state_lb = -leaf_state_lim * np.ones((TestProblem.__num_states, 1))
            leaf_state_ub = leaf_state_lim * np.ones((TestProblem.__num_states, 1))
            leaf_state_constraint = s.build.Rectangle(leaf_state_lb, leaf_state_ub)

            # define risks
            alpha = 0.95
            risks = s.build.AVaR(alpha)

            TestProblem.__problem = (
                s.problem.Factory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_dynamics_events(dynamics)
                .with_cost_nonleaf_events(nonleaf_costs)
                .with_cost_leaf(leaf_cost)
                .with_constraint_nonleaf(state_input_constraint)
                .with_constraint_leaf(leaf_state_constraint)
                .with_risk(risks)
                .with_tests()
            ).generate_problem()

            # test if deterministic problem data can be generated
            _ = (
                s.problem.Factory(tree, TestProblem.__num_states, TestProblem.__num_inputs)
                .with_dynamics(dynamics[0])
                .with_cost_nonleaf(nonleaf_costs[0])
                .with_cost_leaf(leaf_cost)
                .with_constraint_nonleaf(state_input_constraint)
                .with_constraint_leaf(leaf_state_constraint)
                .with_risk(risks)
                .with_tests()
            ).generate_problem()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        TestProblem._construct_tree()
        TestProblem._construct_problem()

    def test_stochastic_dynamics_list(self):
        tree = TestProblem.__tree
        problem = TestProblem.__problem
        for i in range(0, tree.num_nodes):
            self.assertTrue(problem.dynamics_at_node(i) is not None)

    def test_stochastic_nonleaf_costs_list(self):
        tree = TestProblem.__tree
        problem = TestProblem.__problem
        for i in range(tree.num_nodes):
            self.assertTrue(problem.nonleaf_cost_at_node(i) is not None)

    def test_stochastic_all_nonleaf_costs_list(self):
        tree = TestProblem.__tree
        problem = TestProblem.__problem
        for i in range(tree.num_nodes):
            self.assertTrue(problem.nonleaf_cost_at_node(i) is not None)

    def test_stochastic_leaf_costs_list(self):
        tree = TestProblem.__tree
        problem = TestProblem.__problem
        for i in range(tree.num_nonleaf_nodes, tree.num_nodes):
            self.assertTrue(problem.leaf_cost_at_node(i) is not None)

    def test_stochastic_no_constraints_loaded(self):
        problem = TestProblem.__problem
        self.assertTrue(problem.nonleaf_constraint() is not None)
        self.assertTrue(problem.leaf_constraint() is not None)

    def test_stochastic_risks_list(self):
        tree = TestProblem.__tree
        problem = TestProblem.__problem
        for i in range(tree.num_nonleaf_nodes):
            self.assertTrue(problem.risk_at_node(i) is not None)

    def test_stochastic_operator(self):
        problem = TestProblem.__problem
        m = 100
        prim = m * np.random.randn(problem.size_prim, 1)
        dual = m * np.random.randn(problem.size_dual, 1)
        # y'Lx
        Lx = np.array(problem._Problem__op(prim)).reshape(-1, 1)
        uno = (dual.T @ Lx)[0, 0]
        # (L'y)'x
        Ly = np.array(problem._Problem__adj(dual)).reshape(-1, 1)
        dos = (Ly.T @ prim)[0, 0]
        # Compare results
        self.assertAlmostEqual(uno, dos)


if __name__ == '__main__':
    unittest.main()
