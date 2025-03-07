import os
import numpy as np
import turtle
import gputils_api as ga


class Tree:
    """
    Scenario tree creation and visualisation
    """

    def __init__(self, dt, stages, ancestors, probability, w_values=None, children=None):
        """
        :param dt: data type
        :param stages: array where `array position=node number` and `value at position=stage at node`
        :param ancestors: array where `array position=node number` and `value at position=node ancestor`
        :param probability: array where `array position=node number` and `value at position=probability node occurs`
        :param w_values: (optional) array where `array position=node number` and `value at position=value of w`
        :param children: (optional) array where `array position=node number` and `value at position=children of node`

        Note: avoid using this constructor directly; use a factory instead.
        """
        self.__folder = "data"
        self.__path = os.path.join(os.getcwd(), self.__folder)
        os.makedirs(self.__path, exist_ok=True)
        self.__file_ext = ".bt"
        self.__dt = dt
        self.__is_eventful = w_values is not None
        self.__stages = stages
        self.__ancestors = ancestors
        self.__probability = probability
        self.__conditional_probability = None  # this will be updated later (the user doesn't need to provide it)
        self.__w_idx = w_values  # ^
        self.__children = children  # ^
        self.__num_children = None  # ^
        self.__nodes_of_stage = None  # ^
        self.__update()

    def __update(self):
        # Update events
        if self.__w_idx is None:
            self.__w_idx = np.zeros(self.num_nodes, np.intp)
            self.__w_idx[0] = -1
        # Update children
        if self.__children is None:
            self.__children = []
            for i in range(self.num_nonleaf_nodes):
                children_of_i = np.where(self.__ancestors == i)
                self.__children += children_of_i
        # Update conditional probabilities
        cond_prob = [-1]
        for i in range(1, self.num_nodes):
            anc = self.ancestor_of_node(i)
            prob_anc = self.__probability[anc]
            prob_ch = self.__probability[i]
            cond_prob += [prob_ch / prob_anc]
        self.__conditional_probability = np.asarray(cond_prob)
        # Update number children
        self.__num_children = []
        for i in range(self.num_nonleaf_nodes):
            self.__num_children += self.children_of_node(i).shape
        # Update stage from
        self.__nodes_of_stage = []
        for i in range(self.num_stages):
            self.__nodes_of_stage += [self.nodes_of_stage(i)]

    @property
    def folder(self):
        return self.__folder

    @property
    def dtype(self):
        return self.__dt

    @property
    def is_eventful(self):
        return self.__is_eventful

    @property
    def num_nonleaf_nodes(self):
        return np.sum(self.__stages < (self.num_stages - 1))

    @property
    def num_events(self):
        """
        :return: total number of events
        """
        return max(self.__w_idx) + 1

    @property
    def num_nodes(self):
        """
        :return: total number of nodes of the tree
        """
        return len(self.__ancestors)

    @property
    def num_leaf_nodes(self):
        """
        :return: number of leaf nodes of the tree
        """
        return self.num_nodes - self.num_nonleaf_nodes

    @property
    def num_stages(self):
        """
        :return: number of stages including zero stage
        """
        return self.__stages[-1] + 1

    def ancestor_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: index of ancestor node
        """
        return self.__ancestors[node_idx]

    def children_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: list of children of given node
        """
        return self.__children[node_idx]

    def stage_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: stage of given node
        """
        if node_idx < 0:
            raise ValueError("node_idx cannot be <0")
        return self.__stages[node_idx]

    def event_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: value of the disturbance (`w`) at the given node (if any)
        """
        return self.__w_idx[node_idx]

    def nodes_of_stage(self, stage_idx):
        """
        :param stage_idx: index of stage
        :return: array of node indices at given stage
        """
        return np.where(self.__stages == stage_idx)[0]

    def probability_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: probability to visit the given node
        """
        return self.__probability[node_idx]

    def siblings_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: array of siblings of given node (including the given node)
        """
        if node_idx == 0:
            return [0]
        return self.children_of_node(self.ancestor_of_node(node_idx))

    def cond_prob_of_children_of_node(self, node_idx):
        """
        :param node_idx: node index
        :return: array of conditional probabilities of the children of a given node
        """
        children = self.__children[node_idx]
        return self.__conditional_probability[children]

    def __str__(self):
        return f"Scenario Tree\n+ Nodes: {self.num_nodes}\n+ Stages: {self.num_stages}\n" \
               f"+ Scenarios: {len(self.nodes_of_stage(self.num_stages - 1))}"

    def __repr__(self):
        return f"Scenario tree with {self.num_nodes} nodes, {self.num_stages} stages " \
               f"and {len(self.nodes_of_stage(self.num_stages - 1))} scenarios"

    @property
    def get_folder_path(self):
        return self.__path

    def get_file_path(self, name, dt):
        return os.path.join(self.__path, name + "_" + dt + self.__file_ext)

    def write_to_file_uint(self, name, vector):
        ga.write_array_to_gputils_binary_file(np.array(vector, dtype=np.uintp), self.get_file_path(name, 'u'))

    def write_to_file_fp(self, name, tensor):
        ga.write_array_to_gputils_binary_file(tensor.astype(self.__dt), self.get_file_path(name, self.__dt))

    def generate_tree_files(self):
        # Generate extra lists
        child_from = [arr[0] for arr in self.__children]
        child_to = [arr[-1] for arr in self.__children]
        stage_from = [arr[0] for arr in self.__nodes_of_stage]
        stage_to = [arr[-1] for arr in self.__nodes_of_stage]
        # Create tensor dict
        vectors_int = {
            "stages": self.__stages,
            "ancestors": self.__ancestors,
            "events": self.__w_idx,
            "childrenFrom": child_from,
            "childrenTo": child_to,
            "numChildren": self.__num_children,
            "stageFrom": stage_from,
            "stageTo": stage_to
        }
        vectors_fp = {
            "probabilities": np.array(self.__probability),
            "conditionalProbabilities": np.array(self.__conditional_probability)
        }
        # Generate files
        for name, vector in vectors_int.items():
            self.write_to_file_uint(name, vector)
        for name, vector in vectors_fp.items():
            self.write_to_file_fp(name, vector)

    # Visualisation

    @staticmethod
    def __circle_coord(rad, arc):
        return rad * np.cos(np.deg2rad(arc)), rad * np.sin(np.deg2rad(arc))

    @staticmethod
    def __goto_circle_coord(trt, rad, arc):
        trt.penup()
        trt.goto(Tree.__circle_coord(rad, arc))
        trt.pendown()

    @staticmethod
    def __draw_circle(trt, rad):
        trt.penup()
        trt.home()
        trt.goto(0, -rad)
        trt.pendown()
        trt.circle(rad)

    def __draw_leaf_nodes_on_circle(self, trt, radius, dot_size=6):
        trt.pencolor('gray')
        Tree.__draw_circle(trt, radius)
        leaf_nodes = self.nodes_of_stage(self.num_stages - 1)
        num_leaf_nodes = len(leaf_nodes)
        dv = 360 / num_leaf_nodes
        arcs = np.zeros(self.num_nodes)
        for i in range(num_leaf_nodes):
            Tree.__goto_circle_coord(trt, radius, i * dv)
            trt.pencolor('black')
            trt.dot(dot_size)
            trt.pencolor('gray')
            arcs[leaf_nodes[i]] = i * dv

        trt.pencolor('black')
        return arcs

    def __draw_nonleaf_nodes_on_circle(self, trt, radius, larger_radius, stage, arcs, dot_size=6):
        trt.pencolor('gray')
        Tree.__draw_circle(trt, radius)
        nodes = self.nodes_of_stage(stage)
        for n in nodes:
            mean_arc = np.mean(arcs[self.children_of_node(n)])
            arcs[n] = mean_arc
            Tree.__goto_circle_coord(trt, radius, mean_arc)
            trt.pencolor('black')
            trt.dot(dot_size)
            for nc in self.children_of_node(n):
                current_pos = trt.pos()
                trt.goto(Tree.__circle_coord(larger_radius, arcs[nc]))
                trt.goto(current_pos)
            trt.pencolor('gray')
        return arcs

    def bulls_eye_plot(self, dot_size=5, radius=300, filename=None):
        """
        Bull's eye plot of scenario tree

        :param dot_size: size of node [default: 5]
        :param radius: radius of largest circle [default: 300]
        :param filename: name of file, with .eps extension, to save the plot [default: None]
        """
        wn = turtle.Screen()
        wn.tracer(0)
        t = turtle.Turtle(visible=False)
        t.speed(0)

        arcs = self.__draw_leaf_nodes_on_circle(t, radius, dot_size)
        radius_step = radius / (self.num_stages - 1)
        for n in range(self.num_stages - 2, -1, -1):
            radius -= radius_step
            arcs = self.__draw_nonleaf_nodes_on_circle(t, radius, radius + radius_step, n, arcs, dot_size)

        wn.update()

        root_window = wn.getcanvas().winfo_toplevel()
        root_window.call('wm', 'attributes', '.', '-topmost', '1')  # Launch window on top

        if filename is not None:
            wn.getcanvas().postscript(file=filename)
        wn.mainloop()


class MarkovChain:
    """
    Factory class to construct scenario trees from stopped Markov chains
    """

    def __init__(self, transition_prob, initial_distribution, horizon, stopping_stage=None, dt='d'):
        """
        :param transition_prob: transition matrix of the Markov chain
        :param initial_distribution: initial distribution of `w`
        :param horizon: horizon of the scenario tree (N) or number of final stage
        :param stopping_stage: stopping stage, which must be no larger than the number of stages [default: None]
        :param dt: data type
        """
        self.__dt = dt
        if stopping_stage is None:
            stopping_stage = horizon
        else:
            self.__check_stopping_stage(horizon, stopping_stage)
        self.__transition_prob = transition_prob
        self.__initial_distribution = initial_distribution
        self.__num_stages = horizon
        self.__stopping_stage = stopping_stage
        # check correctness of `transition_prob` and `initial_distribution`
        for pi in transition_prob:
            self.__check_probability_vector(pi)
        self.__check_probability_vector(initial_distribution)
        self.__num_nonzero_init_distr = len(list(filter(lambda x: (x > 0), self.__initial_distribution)))

    @staticmethod
    def __check_stopping_stage(n, t):
        if t > n:
            raise ValueError("stopping time greater than number of stages")
        return True

    @staticmethod
    def __check_probability_vector(p):
        if abs(sum(p) - 1) >= 1e-10:
            raise ValueError("probability vector does not sum up to 1")
        if any(pi <= -1e-16 for pi in p):
            raise ValueError("probability vector contains negative entries")
        return True

    def __cover(self, i):
        pi = self.__transition_prob[i, :]
        return np.flatnonzero(pi)

    def __make_ancestors_values_stages(self):
        """
        :return: ancestors, values of w and stages
        """
        # Initialise `ancestors`
        ancestors = np.zeros((self.__num_nonzero_init_distr + 1,), dtype=int)
        ancestors[0] = -1  # node 0 does not have an ancestor
        # Initialise `values`
        values = np.zeros((self.__num_nonzero_init_distr + 1,), dtype=int)
        values[0] = -1
        values[1:] = np.flatnonzero(self.__initial_distribution)
        # Initialise `stages`
        stages = np.ones((self.__num_nonzero_init_distr + 1,), dtype=int)
        stages[0] = 0

        cursor = 1
        num_nodes_of_stage = self.__num_nonzero_init_distr
        for stage_idx in range(1, self.__stopping_stage):
            nodes_added_at_stage = 0
            cursor_new = cursor + num_nodes_of_stage
            for i in range(num_nodes_of_stage):
                node_id = cursor + i
                cover = self.__cover(values[node_id])
                length_cover = len(cover)
                ones = np.ones((length_cover,), dtype=int)
                ancestors = np.concatenate((ancestors, node_id * ones))
                nodes_added_at_stage += length_cover
                values = np.concatenate((values, cover))

            num_nodes_of_stage = nodes_added_at_stage
            cursor = cursor_new
            ones = np.ones(nodes_added_at_stage, dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))

        for stage_idx in range(self.__stopping_stage, self.__num_stages):
            ancestors = np.concatenate((ancestors, range(cursor, cursor + num_nodes_of_stage)))
            cursor += num_nodes_of_stage
            ones = np.ones((num_nodes_of_stage,), dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))
            values = np.concatenate((values, values[-num_nodes_of_stage::]))

        return ancestors, values, stages

    def __make_probability_values(self, ancestors, values, stages):
        """
        :return: probability
        """
        # Initialise `probs`
        probs = np.zeros((self.__num_nonzero_init_distr + 1,))
        probs[0] = 1
        probs[1:] = self.__initial_distribution[np.flatnonzero(self.__initial_distribution)]
        num_nodes = len(values)
        index = 0
        for i in range(self.__num_nonzero_init_distr + 1, num_nodes):
            if stages[i] == self.__stopping_stage + 1:
                index = i
                break
            probs_new = probs[ancestors[i]] * self.__transition_prob[values[ancestors[i]], values[i]]
            probs = np.concatenate((probs, [probs_new]))
            index = i

        if index != num_nodes - 1:
            for j in range(index, num_nodes):
                probs_new = probs[ancestors[j]]
                probs = np.concatenate((probs, [probs_new]))

        return probs

    def build(self):
        """
        Generates a scenario tree from the given Markov chain
        """
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values(ancestors, values, stages)
        stochastic = max(self.__transition_prob.shape) > 1
        tree = Tree(self.__dt,
                    stages,
                    ancestors,
                    probs,
                    w_values=values,
                    )
        print("Generating tree files...")
        tree.generate_tree_files()
        return tree


class IidProcess:
    """
    Factory class to construct n-ary scenario trees from i.i.d. processes
    """

    def __init__(self, distribution, horizon, stopping_stage=None, dt='d'):
        """
        :param distribution: distribution of `w`, size = number of events, `w`, per node before stopping stage
        :param horizon: horizon of the scenario tree (N) or number of final stage
        :param stopping_stage: stopping stage, which must be no larger than the number of stages [default: None]
        :param dt: data type
        """
        self.__dt = dt
        if stopping_stage is None:
            stopping_stage = horizon
        else:
            self.__check_stopping_stage(horizon, stopping_stage)
        self.__horizon = horizon
        self.__stopping_stage = stopping_stage
        self.__distribution = np.array(distribution).reshape(-1, )
        self.__n = self.__distribution.size
        self.__check_probability_vector(self.__distribution)
        self.__num_nonzero_distr = len(list(filter(lambda x: (x > 0), self.__distribution)))

    @staticmethod
    def __check_stopping_stage(n, t):
        if t > n:
            raise ValueError("stopping time greater than number of stages")
        return True

    @staticmethod
    def __check_probability_vector(p):
        if abs(sum(p) - 1) >= 1e-10:
            raise ValueError("probability vector does not sum up to 1")
        if any(pi <= -1e-16 for pi in p):
            raise ValueError("probability vector contains negative entries")
        if any(pi <= 1e-16 for pi in p):
            raise ValueError("probability vector contains zero entries")
        return True

    def __make_ancestors_values_stages(self):
        """
        :return: ancestors, values of w, and stages
        """
        # Initialise `ancestors`
        ancestors = np.zeros((self.__num_nonzero_distr + 1,), dtype=int)
        ancestors[0] = -1  # node 0 does not have an ancestor
        # Initialise `values`
        values = np.zeros((self.__num_nonzero_distr + 1,), dtype=int)
        values[0] = -1
        values[1:] = np.flatnonzero(self.__distribution)
        # Initialise `stages`
        stages = np.ones((self.__num_nonzero_distr + 1,), dtype=int)
        stages[0] = 0

        cursor = 1
        num_nodes_of_stage = self.__num_nonzero_distr
        cover = np.flatnonzero(self.__distribution)
        length_cover = len(cover)
        for stage_idx in range(1, self.__stopping_stage):
            nodes_added_at_stage = 0
            cursor_new = cursor + num_nodes_of_stage
            for i in range(num_nodes_of_stage):
                node_id = cursor + i
                ones = np.ones((length_cover,), dtype=int)
                ancestors = np.concatenate((ancestors, node_id * ones))
                nodes_added_at_stage += length_cover
                values = np.concatenate((values, cover))

            num_nodes_of_stage = nodes_added_at_stage
            cursor = cursor_new
            ones = np.ones(nodes_added_at_stage, dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))

        for stage_idx in range(self.__stopping_stage, self.__horizon):
            ancestors = np.concatenate((ancestors, range(cursor, cursor + num_nodes_of_stage)))
            cursor += num_nodes_of_stage
            ones = np.ones((num_nodes_of_stage,), dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))
            values = np.concatenate((values, values[-num_nodes_of_stage::]))

        return ancestors, values, stages

    def __make_probability_values(self, ancestors, values, stages):
        """
        :return: probability
        """
        # Initialise `probs`
        probs = np.zeros((self.__num_nonzero_distr + 1,))
        probs[0] = 1.
        probs[1:] = self.__distribution[np.flatnonzero(self.__distribution)]
        num_nodes = len(values)
        index = 0
        for i in range(self.__num_nonzero_distr + 1, num_nodes):
            if stages[i] == self.__stopping_stage + 1:
                index = i
                break
            probs_new = probs[ancestors[i]] * self.__distribution[values[i]]
            probs = np.concatenate((probs, [probs_new]))
            index = i

        if index != num_nodes - 1:
            for j in range(index, num_nodes):
                probs_new = probs[ancestors[j]]
                probs = np.concatenate((probs, [probs_new]))

        return probs

    def build(self):
        """
        Generates a scenario tree from the given n-ary distribution
        """
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values(ancestors, values, stages)
        tree = Tree(self.__dt,
                    stages,
                    ancestors,
                    probs,
                    w_values=values,
                    )
        print("Generating tree files...")
        tree.generate_tree_files()
        return tree


class FromData:
    """
    Factory class to construct scenario trees from data
    """

    def __init__(self, samples_dim_time, branching_per_stage, dt='d', sw_scaling=False, original_problem=None):
        """
        :param samples_dim_time: tensor of tree data, [samples x dimension x time]
        :param branching_per_stage: maximum number of children per node at index stage
        :param dt: data type
        """
        if len(samples_dim_time.shape) != 3:
            raise Exception("[FromData] Data provided is not a 3D array.")
        self.__data = samples_dim_time
        self.__branching = np.array(branching_per_stage, dtype=np.int32)
        self.__dt = dt
        self.__sw_scaling = sw_scaling
        self.__original_problem = original_problem
        self.__dict = None

    @staticmethod
    def __find_min_dist_to_j(og_scenarios, elim_scenarios, probs, temp_kept_scenarios, j):
        """
        Finds minimum distance to node j and return it.

        Args:
            og_scenarios (array_like): Array of original scenarios
            elim_scenarios (array_like): Array of scenarios that will be eliminated
            probs (array_like): Probability of original scenario
            temp_kept_scenarios (array_like): Array of current kept scenarios
            j (int): Current eliminated node

        Returns:
            float: Updated probability based on current eliminated node
        """
        dist = 0
        minDistToKeptScen = np.inf
        for h, _ in enumerate(temp_kept_scenarios):
            distToKeptScen = np.linalg.norm(
                og_scenarios[elim_scenarios[j], :] - og_scenarios[temp_kept_scenarios[h], :],
                ord=2,
            )
            if distToKeptScen < minDistToKeptScen:
                minDistToKeptScen = distToKeptScen
        dist += probs[elim_scenarios[j]] * minDistToKeptScen
        return dist

    @staticmethod
    def __redistribute_probabilities(probs, kept_scenarios, closest_nodes):
        """
        Apply redistribution by adding probabilities of deleted scenarios
        to the closest kept scenario.

        Args:
            probs (array_like): Probabilities of original scenarios
            kept_scenarios (array_like): Kept scenarios
            closest_nodes (array_like): Closest kept node for each original node
        """
        resultingProbabilities = np.zeros(probs.shape)
        for keptNode in kept_scenarios:
            resultingProbabilities[keptNode] = sum(
                probs[np.argwhere(closest_nodes == keptNode)]
            ).item()
        return resultingProbabilities

    @staticmethod
    def __compute_closest_nodes(og_scenarios, kept_scenarios):
        """
        Compute the closest node that is kept for each original node.

        Args:
            og_scenarios (array_like): Scenarios
            kept_scenarios (array_like): Desired scenarios
        """
        closestNodes = np.zeros(len(og_scenarios))
        for j, _ in enumerate(og_scenarios):
            distMin = np.inf
            for keptNode in kept_scenarios:
                dist = np.linalg.norm(
                    og_scenarios[j, :] - og_scenarios[keptNode, :], ord=2
                )
                if dist < distMin:
                    distMin = dist
                    curClosestNode = keptNode
            closestNodes[j] = curClosestNode
        return closestNodes

    def __single_stage(self, og_scenarios, num_desired_scenarios, probs=None):
        """
        Optimal scenario reduction via forward recursion - single stage
        Reduce scenarios to a desired number by minimizing
        Wasserstein-Kantorovitch Lr metric.

        Input arguments:
        origScenarios: random variables
        numDesiredScenarios: number of desired scenarios
        probs: probabilities

        Output arguments:
        resulting_probabilities: probabilities of reduced scenarios
        kept_scenarios: indices of kept scenarios
        closest_nodes: closestNodes[j] == i iff scenario i in allScenarios is the closest to scenario j

        See Algorithm 2 at page 28 of E-PRICE D3.1
        """

        if num_desired_scenarios > len(og_scenarios):
            raise ValueError("[FromData] Number of desired scenarios too large.")
        if probs is None:
            probs = np.full(og_scenarios.shape[0], 1 / og_scenarios.shape[0])

        # Reduce scenarios
        numScenarios = len(og_scenarios)  # no. of scenarios
        allScenarios = np.arange(numScenarios)
        elimScenarios = np.arange(numScenarios)
        kept_scenarios = np.empty(num_desired_scenarios, dtype=np.int32)
        for i, _ in enumerate(kept_scenarios):
            distMin = np.inf
            for s, _ in enumerate(elimScenarios):
                dist = 0
                tempKeptScenarios = np.array(
                    list(set(allScenarios) - (set(elimScenarios) - set([s])))
                )
                for j, _ in enumerate(elimScenarios):
                    if j != s:
                        dist += self.__find_min_dist_to_j(
                            og_scenarios, elimScenarios, probs, tempKeptScenarios, j
                        )
                if dist < distMin:
                    distMin = dist
                    sMin = s
            kept_scenarios[i] = elimScenarios[sMin]
            elimScenarios = np.delete(elimScenarios, sMin)

        closest_nodes = self.__compute_closest_nodes(og_scenarios, kept_scenarios)
        resulting_probabilities = self.__redistribute_probabilities(probs, kept_scenarios, closest_nodes)
        return resulting_probabilities, kept_scenarios, closest_nodes

    def __reduce_scenarios(self):
        if self.__original_problem is None:
            originalProb = np.full(
                (self.__data.shape[0], self.__data.shape[2]),
                1 / (self.__data.shape[0] * self.__data.shape[2]),
            )

        nScen = self.__data.shape[0]
        nUncertainVars = self.__data.shape[1]

        desiredPredictionHorizon = self.__branching.size
        originalPredictionHorizon = self.__data.shape[2]
        predictionHorizon = min(desiredPredictionHorizon, originalPredictionHorizon)

        if self.__sw_scaling:
            scaled_data = np.zeros(self.__data.shape)
            for varIndex in range(nUncertainVars):
                uncertainVars = self.__data[:, varIndex, :]
                uncertainVarsZeroMean = uncertainVars - np.mean(uncertainVars)
                scaled_data[:, varIndex, :] = uncertainVarsZeroMean / np.std(uncertainVars)
        else:
            scaled_data = self.__data

        self.__dict = {
            'stage': np.array([0], dtype=np.int32),
            'value': np.zeros((1, nUncertainVars)),
            'prob': np.array([1]),
            'ancestor': np.array([-1], dtype=np.int32),
            'children': np.array([], dtype=object),
            'leaves': np.array([0], dtype=np.int32),
            'num_nodes': np.int32,
            'num_leaf': np.int32,
            'num_nonleaf': np.int32,
            'num_stages': np.int32,
        }
        cluster = [np.arange(nScen)]

        while any(self.__dict['stage'][self.__dict['leaves']] < predictionHorizon):
            nLeaves = (self.__dict['leaves']).size

            for leaf in range(nLeaves):
                stage = self.__dict['stage'][self.__dict['leaves'][leaf]]

                if stage < predictionHorizon:
                    nNodes = (self.__dict['ancestor']).size

                    n = self.__dict['leaves'][leaf]  # this might not increase linearly

                    self.__dict['leaves'] = np.delete(self.__dict['leaves'], leaf)
                    clusterValues = self.__data[cluster[n], :, stage]
                    scaledClusterValues = scaled_data[cluster[n], :, stage]
                    nScenInCluster = clusterValues.shape[0]
                    clusterProb = originalProb[cluster[n], stage]
                    clusterProb = clusterProb / np.sum(clusterProb)

                    if nScenInCluster > self.__branching[stage]:
                        [reducedProbs, keptNodes, closestNodes] = self.__single_stage(
                            scaledClusterValues, self.__branching[stage], clusterProb
                        )
                    else:
                        reducedProbs = clusterProb
                        keptNodes = np.arange(nScenInCluster)
                        closestNodes = keptNodes
                    newNodes = keptNodes.size

                    for i in range(newNodes):
                        if nNodes + i >= len(cluster):
                            cluster.append(cluster[n][closestNodes == keptNodes[i]])
                        else:
                            cluster[nNodes + i] = cluster[n][closestNodes == keptNodes[i]]

                    self.__dict['leaves'] = np.append(self.__dict['leaves'], nNodes + np.arange(newNodes))
                    self.__dict['stage'] = np.append(
                        self.__dict['stage'], np.ones(newNodes, dtype=np.int32) * (stage + 1)
                    )
                    self.__dict['ancestor'] = np.append(
                        self.__dict['ancestor'], np.ones(newNodes, dtype=np.int32) * n
                    )
                    self.__dict['value'] = np.concatenate(
                        (self.__dict['value'], clusterValues[keptNodes, :])
                    )
                    self.__dict['prob'] = np.append(
                        self.__dict['prob'], self.__dict['prob'][n] * reducedProbs[keptNodes]
                    )

                    # The children need to be arrays...
                    nMissingEntries = max(0, n + 1 - self.__dict['children'].size)
                    for _ in range(nMissingEntries):
                        self.__dict['children'] = np.append(self.__dict['children'], np.array([None]))
                        self.__dict['children'][-1] = np.array([])
                    if newNodes == 1:
                        self.__dict['children'][n] = np.array([nNodes])
                    else:
                        self.__dict['children'][n] = np.array(
                            nNodes + np.arange(newNodes), dtype=np.int32
                        )

        # Compute sizes
        self.__dict['num_nodes'] = self.__dict['stage'].size
        self.__dict['num_leaf'] = self.__dict['leaves'].size
        self.__dict['num_nonleaf'] = self.__dict['num_nodes'] - self.__dict['num_leaf']
        self.__dict['num_stages'] = max(self.__dict['stage']) + 1

        self.__sort_and_relabel()

    def __get_scenarios(self):
        """
        Return list of scenarios as arrays.
        """
        index = []
        maxStage = max(self.__dict["stage"])
        for scenario in range(self.__dict["leaves"].size):
            index.append(np.zeros(maxStage + 1, dtype=np.int32))
            index[scenario][-1] = self.__dict["leaves"][scenario]
            for k in reversed(range(maxStage)):
                index[scenario][k] = self.__dict["ancestor"][index[scenario][k + 1]]
        return index

    def __sort_and_relabel(self):
        """
        Sort arrays into node number = index number.
        """
        # Sort scenarios lexicographically backwards
        scenarios = self.__get_scenarios()
        stacked = np.vstack(scenarios)
        idx_leaves = np.lexsort(stacked.T[::-1])
        sorted_scenarios = [scenarios[idx] for idx in idx_leaves]
        self.__dict['leaves'] = self.__dict['leaves'][idx_leaves]
        # Sort scenarios column-wise into a list with no duplicates
        idx_sort = []
        seen = set()
        for ele in range(self.__dict['num_stages']):  # Iterate over elements
            for scenario in range(self.__dict['num_leaf']):  # Iterate over scenarios
                val = sorted_scenarios[scenario][ele]
                if val not in seen:
                    idx_sort.append(val)
                    seen.add(val)
        self.__dict['stage'] = self.__dict['stage'][idx_sort]
        self.__dict['value'] = self.__dict['value'][idx_sort]
        self.__dict['prob'] = self.__dict['prob'][idx_sort]
        self.__dict['ancestor'] = self.__dict['ancestor'][idx_sort]
        self.__dict['children'] = self.__dict['children'][idx_sort[:self.__dict['num_nonleaf']]]
        # Relabel nodes in ancestor and children
        mapping = {num: new_idx for new_idx, num in enumerate(idx_sort)}
        self.__dict['ancestor'][1:] = np.array([self.__map(ele, mapping) for ele in self.__dict['ancestor'][1:]])
        self.__dict['children'] = [np.array([self.__map(ele, mapping) for ele in ch]) for ch in self.__dict['children']]

    @staticmethod
    def __map(num, mapping):
        if num not in mapping:
            raise ValueError(f"[FromData] Number {num} not found in key mapping!")
        return mapping[num]

    def build(self):
        """
        Generates a scenario tree from the given data
        """
        # check input data
        self.__reduce_scenarios()
        stochastic = True if self.__branching.any() != 1 else False
        tree = Tree(self.__dt,
                    self.__dict['stage'],
                    self.__dict['ancestor'],
                    self.__dict['prob'],
                    children=self.__dict['children'],
                    )
        print("Generating tree files...")
        tree.generate_tree_files()
        return tree
