#include <stdexcept>
#define standard_type double


template <typename type, size_t len>
bool _check_probability_vector(type (&p)[len]) {
    if (abs(sum(p) - 1) >= 1e-10) {
        throw std::runtime_error("probability vector does not sum up to 1");
    }
    if (any(pi <= -1e-16 for pi in p)) {
        throw std::runtime_error("probability vector contains negative entries");
    }
    return true;
}


bool _check_stopping_time(unsigned int n, unsigned int t) {
    if (t > n) {
        throw std::runtime_error("stopping time greater than number of stages");
    }
    return true;
}


class ScenarioTree {
    private:
        int __stages;
        int __ancestors[];
        double __probability[];
        int __w_values[]=NULL; 
        bool __is_markovian=false;
        int __children[][] = None[][];  // array of arrays of the children at each node
        void __update_children();
        ScenarioTree(int stages, int ancestors[], double probability[], int w_values[]=NULL, bool is_markovian=false);
    
    public:
        bool is_markovian() {
            return __is_markovian;
        }

        unsigned int num_nonleaf_nodes() {
            return np.sum(self.__stages < (self.num_stages - 1));
        }

        unsigned int num_nodes() {
            /** @return total number of nodes of the tree */
            return len(self.__ancestors);
        }

        unsigned int num_stages() {
            /** @return number of stages including zero stage */
            return self.__stages[-1] + 1;
        }

        unsigned int get_ancestor_of(node_idx) {
            /**
            @param node_idx node index
            @return index of ancestor node 
            */
            return self.__ancestors[node_idx];
        }

        unsigned int[] get_children_of(node_idx) {
            /**
            @param node_idx node index
            @return list of children of given node
            */
            return self.__children[node_idx];
        }

        unsigned int get_stage_of(node_idx) {
            /**
            @param node_idx node index
            @return stage of given node
            */
            if (node_idx < 0) {
                throw std::runtime_error("node_idx cannot be <0")
            }
            return self.__stages[node_idx];
        }

        unsigned int get_value_at_node(node_idx):
            /**
            @param node_idx node index
            @return value of the disturbance (`w`) at the given node (if any)
            */
            return self.__w_idx[node_idx]

        unsigned int get_nodes_at_stage(stage_idx) {
            /**
            @param stage_idx index of stage
            @return array of node indices at given stage
            */
            return np.where(self.__stages == stage_idx)[0];
        }

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
            return self.children_of(self.ancestor_of(node_idx))

        def conditional_probabilities_of_children(self, node_idx):
            """
            :param node_idx: node index
            :return: array of conditional probabilities of the children of a given node
            """
            prob_node_idx = self.probability_of_node(node_idx)
            children = self.children_of(node_idx)
            prob_children = self.__probability[children]
            return prob_children / prob_node_idx
}


ScenarioTree::ScenarioTree(int stages, int ancestors[], double probability[], int w_values[]=NULL, bool is_markovian=false) {
    /** @Scenario tree constructor
    @param stages: integer number of total tree stages (N+1)
    @param ancestors: array where `array position=node number` and `value at position=node ancestor`
    @param probability: array where `array position=node number` and `value at position=probability node occurs`
    @param w_values: array where `array position=node number` and `value at position=value of w`

    Note: avoid using this constructor directly; use a factory instead
    */
    bool __is_markovian = is_markovian;
    int __stages = stages;
    int __ancestors[] = ancestors[];
    double __probability = probability[];
    int __w_idx = w_values[];
}

void __update_children() {
    this.__children = [];
    for (i=0; i<this.num_nonleaf_nodes; i++) {
        children_of_i = np.where(self.__ancestors == i)
        self.__children += children_of_i
    }
}

def __str__(self):
    return f"Scenario Tree\n+ Nodes: {self.num_nodes}\n+ Stages: {self.num_stages}\n" \
            f"+ Scenarios: {len(self.nodes_at_stage(self.num_stages - 1))}\n" \
            f"+ Data: {self.__data is not None}"

def __repr__(self):
    return f"Scenario tree with {self.num_nodes} nodes, {self.num_stages} stages " \
            f"and {len(self.nodes_at_stage(self.num_stages - 1))} scenarios"


class MarkovChainScenarioTreeFactory {
    """
    Factory class to construct scenario trees from stopped Markov chains
    """

    def __init__(self, transition_prob, initial_distribution, num_stages, stopping_time=None):
        """
        :param transition_prob: transition matrix of the Markov chain
        :param initial_distribution: initial distribution of `w`
        :param num_stages: total number of stages or horizon of the scenario tree
        :param stopping_time: stopping time, which must be no larger than the number of stages [default: None]
        """
        self.__factory_type = "MarkovChain"
        if stopping_time is None:
            stopping_time = num_stages
        else:
            _check_stopping_time(num_stages, stopping_time)
        self.__transition_prob = transition_prob
        self.__initial_distribution = initial_distribution
        self.__num_stages = num_stages
        self.__stopping_time = stopping_time
        # --- check correctness of `transition_prob` and `initial_distribution`
        for pi in transition_prob:
            _check_probability_vector(pi)
        _check_probability_vector(initial_distribution)
}

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
        num_nodes_at_stage = num_nonzero_init_distr
        for stage_idx in range(1, self.__stopping_time):
            nodes_added_at_stage = 0
            cursor_new = cursor + num_nodes_at_stage
            for i in range(num_nodes_at_stage):
                node_id = cursor + i
                cover = self.__cover(values[node_id])
                length_cover = len(cover)
                ones = np.ones((length_cover,), dtype=int)
                ancestors = np.concatenate((ancestors, node_id * ones))
                nodes_added_at_stage += length_cover
                values = np.concatenate((values, cover))

            num_nodes_at_stage = nodes_added_at_stage
            cursor = cursor_new
            ones = np.ones(nodes_added_at_stage, dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))

        for stage_idx in range(self.__stopping_time, self.__num_stages):
            ancestors = np.concatenate((ancestors, range(cursor, cursor + num_nodes_at_stage)))
            cursor += num_nodes_at_stage
            ones = np.ones((num_nodes_at_stage,), dtype=int)
            stages = np.concatenate((stages, (1 + stage_idx) * ones))
            values = np.concatenate((values, values[-num_nodes_at_stage::]))

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
            if stages[i] == self.__stopping_time + 1:
                index = i
                break
            probs_new = probs[ancestors[i]] * \
                self.__transition_prob[values[ancestors[i]], values[i]]
            probs = np.concatenate((probs, [probs_new]))
            index = i

        for j in range(index, num_nodes):
            probs_new = probs[ancestors[j]]
            probs = np.concatenate((probs, [probs_new]))

        return probs

    def create(self):
        """
        Creates a scenario tree from the given Markov chain
        """
        # check input data
        ancestors, values, stages = self.__make_ancestors_values_stages()
        probs = self.__make_probability_values(ancestors, values, stages)
        tree = ScenarioTree(stages, ancestors, probs, values, is_markovian=True)
        return tree
