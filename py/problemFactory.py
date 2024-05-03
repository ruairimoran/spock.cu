from . import treeFactory
from . import build


class Problem:
    """
    Risk-averse optimal control problem storage
    """

    def __init__(self, scenario_tree: treeFactory.Tree):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__num_nodes = self.__tree.num_nodes
        self.__num_nonleaf_nodes = self.__tree.num_nonleaf_nodes
        self.__num_events = len(self.__tree.children_of_node(0))
        self.__list_of_state_dynamics = [None] * self.__num_nodes
        self.__list_of_input_dynamics = [None] * self.__num_nodes
        self.__list_of_nonleaf_state_costs = [None] * self.__num_nodes
        self.__list_of_nonleaf_input_costs = [None] * self.__num_nodes
        self.__list_of_leaf_state_costs = [None] * self.__num_nodes
        self.__list_of_state_constraints = [None] * self.__num_nodes
        self.__list_of_input_constraints = [None] * self.__num_nonleaf_nodes
        self.__list_of_risks = [None] * self.__num_nonleaf_nodes
        self._load_constraints()

    # GETTERS
    @property
    def tree(self):
        return self.__tree

    @property
    def list_of_dynamics(self):
        return self.__list_of_dynamics

    @property
    def list_of_nonleaf_costs(self):
        return self.__list_of_nonleaf_costs

    @property
    def list_of_leaf_costs(self):
        return self.__list_of_leaf_costs

    @property
    def list_of_nonleaf_constraints(self):
        return self.__list_of_nonleaf_constraints

    @property
    def list_of_leaf_constraints(self):
        return self.__list_of_leaf_constraints

    @property
    def list_of_risks(self):
        return self.__list_of_risks

    def state_dynamics_at_node(self, idx):
        return self.list_of_dynamics[idx].state_dynamics

    def control_dynamics_at_node(self, idx):
        return self.list_of_dynamics[idx].control_dynamics

    def nonleaf_cost_at_node(self, idx):
        return self.list_of_nonleaf_costs[idx]

    def leaf_cost_at_node(self, idx):
        return self.list_of_leaf_costs[idx]

    def nonleaf_constraint_at_node(self, idx):
        return self.list_of_nonleaf_constraints[idx]

    def leaf_constraint_at_node(self, idx):
        return self.list_of_leaf_constraints[idx]

    def risk_at_node(self, idx):
        return self.list_of_risks[idx]

    def _is_dynamics_given(self):
        for i in range(1, len(self.__list_of_dynamics)):
            if self.__list_of_dynamics[i] is None:
                return False
            else:
                return True

    def _check_dynamics_before_constraints(self):
        # check dynamics are already given
        if not self._is_dynamics_given():
            raise Exception("Constraints provided before dynamics - dynamics must be provided first")

    def _load_constraints(self):
        # load "No Constraint" into constraints list
        for i in range(self.__num_nodes):
            if i < self.__num_nonleaf_nodes:
                self.__list_of_nonleaf_constraints[i] = build.No()
            else:
                self.__list_of_leaf_constraints[i] = build.No()


class ProblemFactory:
    """
    Risk-averse optimal control problem builder
    """

    def __init__(self, scenario_tree: treeFactory.Tree, state_dyn, input_dyn, state_cost, input_cost, terminal_cost,
                 state_constraint, input_constraint, risk):
        """
        :param scenario_tree: instance of ScenarioTree
        """
        self.__tree = scenario_tree
        self.__list_of_state_dynamics = state_dyn
        self.__list_of_input_dynamics = input_dyn
        self.__list_of_nonleaf_state_costs = state_cost
        self.__list_of_nonleaf_input_costs = input_cost
        self.__list_of_leaf_state_costs = terminal_cost
        self.__list_of_state_constraints = state_constraint
        self.__list_of_input_constraints = input_constraint
        self.__list_of_risks = risk

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def with_markovian_dynamics(self, state_dynamics, input_dynamics):
        numEvents = len(state_dynamics)
        # check equal number of dynamics given
        if numEvents != len(input_dynamics):
            raise ValueError("Different number of Markovian state and input dynamics given")
        for i in range(1, numEvents):
            # check all state dynamics have same shape
            if state_dynamics[i].shape != state_dynamics[0].shape:
                raise ValueError("Markovian state dynamics matrices are different shapes")
            # check all control dynamics have same shape
            if input_dynamics[i].shape != input_dynamics[0].shape:
                raise ValueError("Markovian input dynamics matrices are different shapes")
        # check that scenario tree provided is Markovian
        if not self.__tree.isMarkovian:
            raise ValueError("Markovian dynamics provided but scenario tree is not Markovian")
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_state_dynamics[i] = ordered_list_of_dynamics[self.__tree.value_at_node(i)]
        return self

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------
    def with_markovian_nonleaf_costs(self, ordered_list_of_costs):
        # check costs are nonleaf
        for costs in ordered_list_of_costs:
            if not costs.node_type.is_nonleaf:
                raise Exception("Markovian costs provided are not nonleaf")

        # check that scenario tree is Markovian
        if self.__tree.isMarkovian:
            for i in range(1, self.__num_nodes):
                self.__list_of_nonleaf_costs[i] = ordered_list_of_costs[self.__tree.value_at_node(i)]

            return self
        else:
            raise TypeError("costs provided as Markovian, scenario tree provided is not Markovian")

    def with_all_nonleaf_costs(self, cost):
        # check cost are nonleaf
        if not cost.node_type.is_nonleaf:
            raise Exception("Nonleaf cost provided is not nonleaf")
        for i in range(1, self.__num_nodes):
            self.__list_of_nonleaf_costs[i] = cost

        return self

    def with_all_leaf_costs(self, cost):
        # check cost are leaf
        if not cost.node_type.is_leaf:
            raise Exception("Leaf cost provided is not leaf")
        for i in range(self.__num_nonleaf_nodes, self.__num_nodes):
            self.__list_of_leaf_costs[i] = cost

        return self

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    def with_all_nonleaf_constraints(self, nonleaf_constraint):
        self._check_dynamics_before_constraints()
        # check constraints are nonleaf
        if not nonleaf_constraint.node_type.is_nonleaf:
            raise Exception("Nonleaf constraint provided is not nonleaf")
        nonleaf_constraint.state_size = self.__list_of_dynamics[-1].state_dynamics.shape[1]
        nonleaf_constraint.control_size = self.__list_of_dynamics[-1].control_dynamics.shape[1]
        for i in range(self.__tree.num_nonleaf_nodes):
            self.__list_of_nonleaf_constraints[i] = nonleaf_constraint

        return self

    def with_all_leaf_constraints(self, leaf_constraint):
        self._check_dynamics_before_constraints()
        if not leaf_constraint.node_type.is_leaf:
            raise Exception("Leaf constraint provided is not leaf")
        leaf_constraint.state_size = self.__list_of_dynamics[-1].state_dynamics.shape[1]
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__list_of_leaf_constraints[i] = leaf_constraint

        return self

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------
    def with_all_risks(self, risk):
        # check risk type
        if not risk.is_risk:
            raise Exception("Risk provided is not of risk type")
        for i in range(self.__tree.num_nonleaf_nodes):
            risk_i = risk
            risk_i.probs = self.__tree.conditional_probabilities_of_children(i)
            self.__list_of_risks[i] = risk_i

        return self

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------
    def generate_problem(self):
        """
        Generates problem data from the given build
        """
        problem = Problem(self.__tree)
        problem.generate_json()
        return problem
