import datetime
import jinja2 as j2
import os
from . import treeFactory
from . import build


class Problem:
    """
    Risk-averse optimal control problem storage
    """

    def __init__(self, scenario_tree: treeFactory.Tree, num_states, num_inputs, state_dyn, input_dyn, state_cost,
                 input_cost, terminal_cost, state_constraint, input_constraint, risk):
        """
        :param scenario_tree: instance of ScenarioTree
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
                                 risk=self.__list_of_risks)
        path = os.path.join(os.getcwd(), "json")
        os.makedirs(path, exist_ok=True)
        output_file = os.path.join(path, "problemData.json")
        fh = open(output_file, "w")
        fh.write(output)
        fh.close()

    def run_offline_cache(self):
        pass

    def generate_cache_json(self):
        pass


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
        if not self.__tree.isMarkovian:
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
            self.__list_of_state_dynamics[i] = state_dynamics[event]
            self.__list_of_input_dynamics[i] = input_dynamics[event]
        return self

    # --------------------------------------------------------
    # Costs
    # --------------------------------------------------------
    def with_markovian_nonleaf_costs(self, state_costs, input_costs):
        self.__check_markovian("costs")
        for i in range(1, self.__tree.num_nodes):
            event = self.__tree.event_of_node(i)
            self.__list_of_nonleaf_state_costs[i] = state_costs[event]
            self.__list_of_nonleaf_input_costs[i] = input_costs[event]
        return self

    def with_all_nonleaf_costs(self, state_cost, input_cost):
        for i in range(1, self.__tree.num_nodes):
            self.__list_of_nonleaf_state_costs[i] = state_cost
            self.__list_of_nonleaf_input_costs[i] = input_cost
        return self

    def with_all_leaf_costs(self, state_cost):
        for i in range(self.__tree.num_nonleaf_nodes, self.__tree.num_nodes):
            self.__list_of_leaf_state_costs[i] = state_cost
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
                self.__list_of_input_constraints[i] = input_constraint
            elif i < self.__tree.num_nonleaf_nodes:
                self.__list_of_state_constraints[i] = state_constraint
                self.__list_of_input_constraints[i] = input_constraint
            else:
                self.__list_of_state_constraints[i] = state_constraint
        return self

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------
    def with_all_risks(self, risk):
        for i in range(self.__tree.num_nonleaf_nodes):
            risk.make_risk(self.__tree.cond_prob_of_children_of_node(i))
            self.__list_of_risks[i] = risk.make_risk(self.__tree.cond_prob_of_children_of_node(i))
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
        problem.generate_problem_json()
        problem.run_offline_cache()
        problem.generate_cache_json()
        return problem
