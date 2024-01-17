#include <vector>
using namespace std;


class ScenarioTree {
    private:
        #include "tree_attributes.h"

    public:
        const bool is_markovian() {
            /** @return whether tree is a stopped Markovian process */
            return _is_markovian;
        }

        const bool is_iid() {
            /** @return whether tree is an independent and identically distributed sequence */
            return _is_iid;
        }

        const int num_nonleaf_nodes() {
            /** @return total number of nonleaf nodes */
            return _num_nonleaf_nodes;
        }

        const int num_nodes() {
            /** @return total number of nodes */
            return _num_nodes;
        }

        const int num_stages() {
            /** @return total number of stages */
            return _num_stages;
        }

        int get_stage_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return stage of given node
            */
            if (node_idx < 0 || node_idx >= _num_nodes) {
                throw invalid_argument("node_idx error: get_stage_of()");
            }
            return stages[node_idx];
        }

        int get_ancestor_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return ancestor of given node 
            */
            if (node_idx <= 0 || node_idx >= _num_nodes) {
                throw invalid_argument("node_idx error: get_ancestor_of()");
            }
            return ancestors[node_idx];
        }

        double get_probability_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return probability to visit the given node
            */
            if (node_idx < 0 || node_idx >= _num_nodes) {
                throw invalid_argument("node_idx error: get_probability_of_node()");
            } 
            return probability[node_idx];
        }

        int get_event_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return event of the disturbance (`w`) at the given node (if any)
            */
            if (node_idx <= 0 || node_idx >= _num_nodes) {
                throw invalid_argument("node_idx error: get_event_at_node()");
            }
            return events[node_idx];
        }

        vector<const int> get_children_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return last child of given node
            */
            if (node_idx < 0 || node_idx >= _num_nonleaf_nodes) {
                throw invalid_argument("node_idx error: get_last_child_of()");
            }
            return children[node_idx];
        }

        /** NOT IMPLEMENTED - CURRENTLY NOT USED ANYWHERE
        void get_siblings_of_node(node_idx){
            --
            @param node_idx node index
            @return vector of siblings of given node (including the given node)
            --
            if (node_idx == 0) {
                return [0];
            }
            return self.children_of(self.ancestor_of(node_idx));
        }
        */

        vector<double> get_cond_prob_of_children_of_node(int node_idx) {
            /**
            @param node_idx node index
            @return vector of conditional probabilities of the children of a given node
            */
            if (node_idx < 0 || node_idx >= _num_nonleaf_nodes) {
                throw invalid_argument("node_idx error: get_conditional_probabilities_of_children()");
            }
            double prob_parent_ = get_probability_of_node(node_idx);
            vector<const int> children_ = get_children_of_node(node_idx);
            int size_of_ch = children_.size();
            vector<double> cond_prob_(size_of_ch);
            for (int i=0; i<size_of_ch; i++) {
                cond_prob_[i] = get_probability_of_node(children_[i]) / prob_parent_;
            }
            return cond_prob_;
        }

        vector<int> get_nodes_of_stage(int stage_idx) {
            /**
            @param stage_idx stage index
            @return vector of node indices at given stage
            */
            if (stage_idx < 0 || stage_idx >= _num_stages) {
                throw invalid_argument("stage_idx error: get_nodes_at_stage()");
            }
            vector<int> nodes;
            for (int i=0; i<_num_nodes; i++) {
                if (stages[i] == stage_idx) {
                    nodes.insert(nodes.end(), i);
                }
            }
            return nodes;
        }
};
