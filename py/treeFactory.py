import os
import numpy as np
import turtle
import jinja2 as j2


class Tree:
    """
    Scenario tree creation and visualisation
    """

    def __init__(self, stages, ancestors, probability, w_values=None, is_markovian=False, is_iid=False):
        """
        :param stages: array where `array position=node number` and `value at position=stage at node`
        :param ancestors: array where `array position=node number` and `value at position=node ancestor`
        :param probability: array where `array position=node number` and `value at position=probability node occurs`
        :param w_values: array where `array position=node number` and `value at position=value of w`

        Note: avoid using this constructor directly; use a factory instead
        """
        self.__folder = "data"
        self.__is_markovian = is_markovian
        self.__is_iid = is_iid
        self.__stages = stages
        self.__ancestors = ancestors
        self.__probability = probability
        self.__conditional_probability = None  # this will be updated later (the user doesn't need to provide it)
        self.__w_idx = w_values
        self.__children = None  # ^
        self.__num_children = None  # ^
        self.__nodes_of_stage = None  # ^
        self.__update()

    def __update(self):
        # Update children
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
    def is_markovian(self):
        return self.__is_markovian

    @property
    def is_iid(self):
        return self.__is_iid

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

    def generate_tree_files(self):
        # Generate extra lists
        child_from = [arr[0] for arr in self.__children]
        child_to = [arr[-1] for arr in self.__children]
        stage_from = [arr[0] for arr in self.__nodes_of_stage]
        stage_to = [arr[-1] for arr in self.__nodes_of_stage]
        # Create tensor dict
        tensors = {
            "stages": self.__stages,
            "ancestors": self.__ancestors,
            "probabilities": self.__probability,
            "conditionalProbabilities": self.__conditional_probability,
            "events": self.__w_idx,
            "childrenFrom": child_from,
            "childrenTo": child_to,
            "numChildren": self.__num_children,
            "stageFrom": stage_from,
            "stageTo": stage_to
        }
        # Generate tensor files
        for name, tensor in tensors.items():
            path = os.path.join(os.getcwd(), self.__folder)
            os.makedirs(path, exist_ok=True)
            output_file = os.path.join(path, name)
            np.savetxt(output_file,
                       X=tensor,
                       fmt='%-.15f',
                       delimiter='\n',
                       newline='\n',
                       header=f"{len(tensor)}\n"
                              f"{1}\n"
                              f"{1}",
                       comments='')


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

    if filename is not None:
        wn.getcanvas().postscript(file=filename)
    wn.mainloop()


class TreeFactoryMarkovChain:
    """
    Factory class to construct scenario trees from stopped Markov chains
    """

    def __init__(self, transition_prob, initial_distribution, horizon, stopping_stage=None):
        """
        :param transition_prob: transition matrix of the Markov chain
        :param initial_distribution: initial distribution of `w`
        :param horizon: horizon of the scenario tree (N) or number of final stage
        :param stopping_stage: stopping stage, which must be no larger than the number of stages [default: None]
        """
        # self.__factory_type = "MarkovChain"
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
        num_nonzero_init_distr = len(list(filter(lambda x: (x > 0), self.__initial_distribution)))
        # Initialise `ancestors`
        ancestors = np.zeros((num_nonzero_init_distr + 1,), dtype=int)
        ancestors[0] = -1  # node 0 does not have an ancestor
        # Initialise `values`
        values = np.zeros((num_nonzero_init_distr + 1,), dtype=int)
        values[0] = -1
        values[1:] = np.flatnonzero(self.__initial_distribution)
        # Initialise `stages`
        stages = np.ones((num_nonzero_init_distr + 1,), dtype=int)
        stages[0] = 0

        cursor = 1
        num_nodes_of_stage = num_nonzero_init_distr
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
        num_nonzero_init_distr = len(list(filter(lambda x: (x > 0), self.__initial_distribution)))
        # Initialise `probs`
        probs = np.zeros((num_nonzero_init_distr + 1,))
        probs[0] = 1
        probs[1:] = self.__initial_distribution[np.flatnonzero(self.__initial_distribution)]
        num_nodes = len(values)
        index = 0
        for i in range(num_nonzero_init_distr + 1, num_nodes):
            if stages[i] == self.__stopping_stage + 1:
                index = i
                break
            probs_new = probs[ancestors[i]] * \
                        self.__transition_prob[values[ancestors[i]], values[i]]
            probs = np.concatenate((probs, [probs_new]))
            index = i

        if index != num_nodes - 1:
            for j in range(index, num_nodes):
                probs_new = probs[ancestors[j]]
                probs = np.concatenate((probs, [probs_new]))

        return probs

    def generate_tree(self):
        """
        Generates a scenario tree from the given Markov chain
        """
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values(ancestors, values, stages)
        tree = Tree(stages, ancestors, probs, values, is_markovian=True)
        print("Generating tree files...")
        tree.generate_tree_files()
        return tree
