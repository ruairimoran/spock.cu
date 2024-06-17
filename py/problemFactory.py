import jinja2 as j2
import os
import numpy as np
import scipy as sp
import cvxpy as cvx
from copy import deepcopy
from . import treeFactory
from . import build


class Problem:
    """
    Risk-averse optimal control problem and offline cache storage
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs, state_dyn, input_dyn, state_cost,
                 input_cost, terminal_cost, state_constraint, input_constraint, risk):
        """
        :param scenario_tree: instance of ScenarioTree
        :param num_states: number of system states
        :param num_inputs: number of system inputs
        :param state_dyn: list of state dynamics matrices (size: num_nodes, used: 1 to num_nodes)
        :param input_dyn: list of input dynamics matrices (size: num_nodes, used: 1 to num_nodes)
        :param state_cost: list of state cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param input_cost: list of input cost matrices (size: num_nodes, used: 1 to num_nodes)
        :param terminal_cost: list of terminal cost matrices (size: num_nodes, used: num_nonleaf_nodes to num_nodes)
        :param state_constraint: list of state constraint classes (size: num_nodes, used: 1 to num_nodes)
        :param input_constraint: list of input constraint classes (size: num_nodes, used: 0 to num_nonleaf_nodes)
        :param risk: list of risk classes (size: num_nonleaf_nodes, used: all)

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_state_dynamics = state_dyn
        self.__list_of_input_dynamics = input_dyn
        self.__list_of_nonleaf_state_costs = state_cost
        self.__list_of_nonleaf_input_costs = input_cost
        self.__list_of_leaf_state_costs = terminal_cost
        self.__list_of_state_constraints = state_constraint
        self.__list_of_input_constraints = input_constraint
        self.__list_of_risks = risk
        # Dynamics projection
        self.__P = [np.zeros((self.__num_states, self.__num_states))] * self.__tree.num_nodes
        self.__q = [np.zeros((self.__num_states, 1))] * self.__tree.num_nodes
        self.__K = [np.zeros((self.__num_states, self.__num_states))] * self.__tree.num_nonleaf_nodes
        self.__d = [np.zeros((self.__num_states, 1))] * self.__tree.num_nonleaf_nodes
        self.__lower = True  # Do not change!
        self.__cholesky_lower = [None] * self.__tree.num_nonleaf_nodes
        self.__sum_of_dynamics_tr = [np.zeros((0, 0))] * self.__tree.num_nodes  # A+B@K
        self.__At_P_B = [np.zeros((0, 0))] * self.__tree.num_nodes  # At@P@B
        self.__dp_test_init_state = None
        self.__dp_test_states = None
        self.__dp_test_inputs = None
        self.__dp_projected_states = None
        self.__dp_projected_inputs = None
        # Kernel projection
        self.__kernel_constraint_matrix = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        self.__nullspace_projection_matrix = [np.zeros((0, 0))] * self.__tree.num_nonleaf_nodes
        self.__max_nullspace_dim = self.__tree.num_events * 4 + 1

    # GETTERS
    def state_dynamics_at_node(self, idx):
        return self.__list_of_state_dynamics[idx]

    def input_dynamics_at_node(self, idx):
        return self.__list_of_input_dynamics[idx]

    def nonleaf_state_cost_at_node(self, idx):
        return self.__list_of_nonleaf_state_costs[idx]

    def nonleaf_input_cost_at_node(self, idx):
        return self.__list_of_nonleaf_input_costs[idx]

    def leaf_state_cost_at_node(self, idx):
        return self.__list_of_leaf_state_costs[idx]

    def state_constraint_at_node(self, idx):
        return self.__list_of_state_constraints[idx]

    def input_constraint_at_node(self, idx):
        return self.__list_of_input_constraints[idx]

    def risk_at_node(self, idx):
        return self.__list_of_risks[idx]

    def generate_problem_json(self):
        # Setup jinja environment
        file_loader = j2.FileSystemLoader(searchpath=["py/"])
        env = j2.Environment(loader=file_loader,
                             trim_blocks=True,
                             lstrip_blocks=True,
                             block_start_string="\% ",
                             block_end_string="%\\",
                             variable_start_string="\~",
                             variable_end_string="~\\",
                             comment_start_string="\#",
                             comment_end_string="#\\")
        # Generate "problemData.json" from template "problemTemplate.json.jinja2"
        template = env.get_template("problemTemplate.json.jinja2")
        output = template.render(num_states=self.__num_states,
                                 num_inputs=self.__num_inputs,
                                 state_dyn=self.__list_of_state_dynamics,
                                 input_dyn=self.__list_of_input_dynamics,
                                 state_cost=self.__list_of_nonleaf_state_costs,
                                 input_cost=self.__list_of_nonleaf_input_costs,
                                 terminal_cost=self.__list_of_leaf_state_costs,
                                 state_constraint=self.__list_of_state_constraints,
                                 input_constraint=self.__list_of_input_constraints,
                                 risk=self.__list_of_risks,
                                 P=self.__P,
                                 K=self.__K,
                                 low_chol=self.__cholesky_lower,
                                 dyn_tr=self.__sum_of_dynamics_tr,
                                 APB=self.__At_P_B,
                                 dpTestStates=self.__dp_test_states,
                                 dpTestInputs=self.__dp_test_inputs,
                                 dpProjectedStates=self.__dp_projected_states,
                                 dpProjectedInputs=self.__dp_projected_inputs,
                                 null_dim=self.__max_nullspace_dim,
                                 null=self.__nullspace_projection_matrix)
        path = os.path.join(os.getcwd(), self.__tree.folder)
        os.makedirs(path, exist_ok=True)
        output_file = os.path.join(path, "problemData.json")
        fh = open(output_file, "w")
        fh.write(output)
        fh.close()

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def generate_offline_cache(self):
        self.__offline_projection_dynamics()
        self.__offline_projection_kernel()
        self.__test_dynamic_programming()

    def __offline_projection_dynamics(self):
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__P[i] = np.eye(self.__num_states)
        for i in reversed(range(self.__tree.num_nonleaf_nodes)):
            children_of_i = self.__tree.children_of_node(i)
            sum_for_r = 0
            sum_for_k = 0
            for j in children_of_i:
                sum_for_r = sum_for_r \
                            + self.__list_of_input_dynamics[j].T @ self.__P[j] @ self.__list_of_input_dynamics[j]
                sum_for_k = sum_for_k \
                            + self.__list_of_input_dynamics[j].T @ self.__P[j] @ self.__list_of_state_dynamics[j]
            r_tilde = np.eye(self.__num_inputs) + sum_for_r
            self.__cholesky_lower[i] = sp.linalg.cholesky(r_tilde, lower=self.__lower, check_finite=True)
            self.__K[i] = sp.linalg.cho_solve((self.__cholesky_lower[i], self.__lower), -sum_for_k)
            sum_for_p = 0
            for j in children_of_i:
                sum_of_dynamics = self.__list_of_state_dynamics[j] + self.__list_of_input_dynamics[j] @ self.__K[i]
                self.__sum_of_dynamics_tr[j] = sum_of_dynamics.T
                sum_for_p = sum_for_p + self.__sum_of_dynamics_tr[j] @ self.__P[j] @ sum_of_dynamics
                self.__At_P_B[j] = self.__sum_of_dynamics_tr[j] @ self.__P[j] @ self.__list_of_input_dynamics[j]
            self.__P[i] = np.eye(self.__num_states) + self.__K[i].T @ self.__K[i] + sum_for_p

    def __offline_projection_kernel(self):
        for i in range(self.__tree.num_nonleaf_nodes):
            eye = np.eye(len(self.__tree.children_of_node(i)))
            zeros = np.zeros((self.__list_of_risks[i].f.shape[1], eye.shape[0]))
            row1 = np.hstack((self.__list_of_risks[i].e.T, -eye, -eye))
            row2 = np.hstack((self.__list_of_risks[i].f.T, zeros, zeros))
            self.__kernel_constraint_matrix[i] = np.vstack((row1, row2))
            n = sp.linalg.null_space(self.__kernel_constraint_matrix[i])
            projection_matrix = n @ n.T
            self.__nullspace_projection_matrix[i] = self.__pad(projection_matrix)

    def __pad(self, nullspace):
        row_pad = self.__max_nullspace_dim - nullspace.shape[0]
        col_pad = self.__max_nullspace_dim - nullspace.shape[1]
        padded_nullspace = np.pad(nullspace, [(0, row_pad), (0, col_pad)], mode="constant", constant_values=0.)
        return padded_nullspace

    def __test_dynamic_programming(self):
        # Solve with cvxpy
        x_bar = 10 * np.random.randn(self.__num_states, self.__tree.num_nodes)
        u_bar = 10 * np.random.randn(self.__num_inputs, self.__tree.num_nonleaf_nodes)
        x = cvx.Variable((self.__num_states, self.__tree.num_nodes))
        u = cvx.Variable((self.__num_inputs, self.__tree.num_nonleaf_nodes))
        self.__dp_test_init_state = x_bar[:, 0]
        # Sum problem objectives and concatenate constraints
        cost = 0
        constraints = [x[:, 0] == x_bar[:, 0]]
        # Nonleaf nodes
        for node in range(self.__tree.num_nonleaf_nodes):
            cost += cvx.sum_squares(x[:, node] - x_bar[:, node]) + cvx.sum_squares(u[:, node] - u_bar[:, node])
            for ch in self.__tree.children_of_node(node):
                constraints += [x[:, ch] ==
                                self.__list_of_state_dynamics[ch] @ x[:, node] +
                                self.__list_of_input_dynamics[ch] @ u[:, node]]

        # Leaf nodes
        for node in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            cost += cvx.sum_squares(x[:, node] - x_bar[:, node])

        problem = cvx.Problem(cvx.Minimize(cost), constraints)
        problem.solve()
        self.__dp_test_states = x_bar.T
        self.__dp_test_inputs = u_bar.T
        self.__dp_projected_states = x.value.T
        self.__dp_projected_inputs = u.value.T


class ProblemFactory:
    """
    Risk-averse optimal control problem builder
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__num_states = num_states
        self.__num_inputs = num_inputs
        self.__list_of_state_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_input_dynamics = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_state_costs = [None] * self.__tree.num_nodes
        self.__list_of_nonleaf_input_costs = [None] * self.__tree.num_nodes
        self.__list_of_leaf_state_costs = [None] * self.__tree.num_nodes
        self.__list_of_state_constraints = [None] * self.__tree.num_nodes
        self.__list_of_input_constraints = [None] * self.__tree.num_nonleaf_nodes
        self.__list_of_risks = [None] * self.__tree.num_nonleaf_nodes
        self.__preload_constraints()

    def __check_markovian(self, section):
        if not self.__tree.is_markovian:
            raise ValueError("Markovian " + section + " provided but scenario tree is not Markovian")

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def with_markovian_dynamics(self, state_dynamics, input_dynamics):
        self.__check_markovian("dynamics")
        num_events = len(state_dynamics)
        # check equal number of dynamics given
        if num_events != len(input_dynamics):
            raise ValueError("Different number of Markovian state and input dynamics given")
        for i in range(1, num_events):
            # check all state dynamics have same shape
            if state_dynamics[i].shape != state_dynamics[0].shape:
                raise ValueError("Markovian state dynamics matrices are different shapes")
            # check all control dynamics have same shape
            if input_dynamics[i].shape != input_dynamics[0].shape:
                raise ValueError("Markovian input dynamics matrices are different shapes")
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_state_dynamics[i] = deepcopy(state_dynamics[event])
            self.__list_of_input_dynamics[i] = deepcopy(input_dynamics[event])
        return self

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------
    def with_markovian_nonleaf_costs(self, state_costs, input_costs):
        self.__check_markovian("costs")
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_nonleaf_state_costs[i] = deepcopy(state_costs[event])
            self.__list_of_nonleaf_input_costs[i] = deepcopy(input_costs[event])
        return self

    def with_all_nonleaf_costs(self, state_cost, input_cost):
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_nonleaf_state_costs[i] = deepcopy(state_cost)
            self.__list_of_nonleaf_input_costs[i] = deepcopy(input_cost)
        return self

    def with_all_leaf_costs(self, state_cost):
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__list_of_leaf_state_costs[i] = deepcopy(state_cost)
        return self

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    def __preload_constraints(self):
        # load "No constraint" into constraints list
        for i in range(self.__tree.num_nodes):
            if i < self.__tree.num_nonleaf_nodes:
                self.__list_of_state_constraints[i] = build.No()
                self.__list_of_input_constraints[i] = build.No()
            else:
                self.__list_of_state_constraints[i] = build.No()

    def with_all_constraints(self, state_constraint, input_constraint):
        for i in range(self.__tree.num_nodes):
            if i == 0:
                self.__list_of_input_constraints[i] = deepcopy(input_constraint)
            elif i < self.__tree.num_nonleaf_nodes:
                self.__list_of_state_constraints[i] = deepcopy(state_constraint)
                self.__list_of_input_constraints[i] = deepcopy(input_constraint)
            else:
                self.__list_of_state_constraints[i] = deepcopy(state_constraint)
        return self

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------
    def with_all_risks(self, risk):
        for i in range(self.__tree.num_nonleaf_nodes):
            risk_i = deepcopy(risk)
            self.__list_of_risks[i] = risk_i.make_risk(self.__tree.cond_prob_of_children_of_node(i))
        return self

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------
    def generate_problem(self):
        """
        Generates problem data from the given build
        """
        problem = Problem(self.__tree,
                          self.__num_states,
                          self.__num_inputs,
                          self.__list_of_state_dynamics,
                          self.__list_of_input_dynamics,
                          self.__list_of_nonleaf_state_costs,
                          self.__list_of_nonleaf_input_costs,
                          self.__list_of_leaf_state_costs,
                          self.__list_of_state_constraints,
                          self.__list_of_input_constraints,
                          self.__list_of_risks)
        problem.generate_offline_cache()
        problem.generate_problem_json()
        return problem
