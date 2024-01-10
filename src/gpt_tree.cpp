#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class ScenarioTree {
private:
    bool __is_markovian;
    int __stages;
    std::vector<int> __ancestors;
    std::vector<double> __probability;
    std::vector<int> __w_idx;
    std::vector<std::vector<int>> __children;

public:
    ScenarioTree(int stages, std::vector<int> ancestors, std::vector<double> probability, std::vector<int> w_values = {}, bool is_markovian = false)
        : __is_markovian(is_markovian), __stages(stages), __ancestors(ancestors), __probability(probability), __w_idx(w_values) {
        __update_children();
    }

    int num_nonleaf_nodes() const {
        return std::count_if(__stages.begin(), __stages.end(), [this](int stage) { return stage < (__stages.back() - 1); });
    }

    int num_nodes() const {
        return __ancestors.size();
    }

    int num_stages() const {
        return __stages.back() + 1;
    }

    int ancestor_of(int node_idx) const {
        return __ancestors[node_idx];
    }

    const std::vector<int>& children_of(int node_idx) const {
        return __children[node_idx];
    }

    int stage_of(int node_idx) const {
        if (node_idx < 0) {
            throw std::invalid_argument("node_idx cannot be < 0");
        }
        return __stages[node_idx];
    }

    int value_at_node(int node_idx) const {
        return __w_idx[node_idx];
    }

    std::vector<int> nodes_at_stage(int stage_idx) const {
        std::vector<int> result;
        for (int i = 0; i < __stages.size(); ++i) {
            if (__stages[i] == stage_idx) {
                result.push_back(i);
            }
        }
        return result;
    }

    double probability_of_node(int node_idx) const {
        return __probability[node_idx];
    }

    std::vector<int> siblings_of_node(int node_idx) const {
        if (node_idx == 0) {
            return {0};
        }
        return children_of(ancestor_of(node_idx));
    }

    std::vector<double> conditional_probabilities_of_children(int node_idx) const {
        double prob_node_idx = probability_of_node(node_idx);
        const std::vector<int>& children = children_of(node_idx);
        std::vector<double> prob_children(children.size());
        for (int i = 0; i < children.size(); ++i) {
            prob_children[i] = __probability[children[i]] / prob_node_idx;
        }
        return prob_children;
    }

    std::string to_string() const {
        return "Scenario Tree\n+ Nodes: " + std::to_string(num_nodes()) + "\n+ Stages: " + std::to_string(num_stages())
            + "\n+ Scenarios: " + std::to_string(nodes_at_stage(num_stages() - 1).size()) + "\n+ Data: " + (__data.empty() ? "false" : "true");
    }

    std::string to_repr() const {
        return "Scenario tree with " + std::to_string(num_nodes()) + " nodes, " + std::to_string(num_stages()) +
            " stages, and " + std::to_string(nodes_at_stage(num_stages() - 1).size()) + " scenarios";
    }

private:
    void __update_children() {
        __children.resize(num_nonleaf_nodes());
        for (int i = 0; i < num_nonleaf_nodes(); ++i) {
            for (int j = 0; j < __ancestors.size(); ++j) {
                if (__ancestors[j] == i) {
                    __children[i].push_back(j);
                }
            }
        }
    }
};

class MarkovChainScenarioTreeFactory {
private:
    std::string __factory_type;
    std::vector<std::vector<double>> __transition_prob;
    std::vector<double> __initial_distribution;
    int __num_stages;
    int __stopping_time;

public:
    MarkovChainScenarioTreeFactory(const std::vector<std::vector<double>>& transition_prob, const std::vector<double>& initial_distribution, int num_stages, int stopping_time = -1)
        : __factory_type("MarkovChain"), __transition_prob(transition_prob), __initial_distribution(initial_distribution), __num_stages(num_stages) {
        if (stopping_time == -1) {
            stopping_time = num_stages;
        } else {
            _check_stopping_time(num_stages, stopping_time);
        }
        // Check correctness of `transition_prob` and `initial_distribution`
        for (const auto& pi : transition_prob) {
            _check_probability_vector(pi);
        }
        _check_probability_vector(initial_distribution);
    }

    ScenarioTree create() {
        // Check input data
        std::vector<int> ancestors, values, stages;
        std::tie(ancestors, values, stages) = __make_ancestors_values_stages();
        std::vector<double> probs = __make_probability_values(ancestors, values, stages);
        return ScenarioTree(stages, ancestors, probs, values, true);
    }

private:
    void _check_stopping_time(int n, int t) const {
        if (t > n) {
            throw std::invalid_argument("stopping time greater than the number of stages");
        }
    }

    void _check_probability_vector(const std::vector<double>& p) const {
        if (std::abs(std::accumulate(p.begin(), p.end(), 0.0) - 1) >= 1e-10) {
            throw std::invalid_argument("probability vector does not sum up to 1");
        }
        if (std::any_of(p.begin(), p.end(), [](double pi) { return pi <= -1e-16; })) {
            throw std::invalid_argument("probability vector contains negative entries");
        }
    }

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> __make_ancestors_values_stages() const {
        int num_nonzero_init_distr = std::count_if(__initial_distribution.begin(), __initial_distribution.end(),
            [](double x) { return x > 0; });

        std::vector<int> ancestors(num_nonzero_init_distr + 1, 0);
        ancestors[0] = -1;
        std::vector<int> values(num_nonzero_init_distr + 1, 0);
        values[0] = -1;
        for (int i = 1; i <= num_nonzero_init_distr; ++i) {
            values[i] = i - 1;
        }
        std::vector<int> stages(num_nonzero_init_distr + 1, 0);
        stages[0] = 0;

        int cursor = 1;
        int num_nodes_at_stage = num_nonzero_init_distr;

        for (int stage_idx = 1; stage_idx < __stopping_time; ++stage_idx) {
            int nodes_added_at_stage = 0;
            int cursor_new = cursor + num_nodes_at_stage;

            for (int i = 0; i < num_nodes_at_stage; ++i) {
                int node_id = cursor + i;
                std::vector<int> cover = __cover(values[node_id]);
                int length_cover = cover.size();
                std::vector<int> ones(length_cover, 1);
                ancestors.insert(ancestors.end(), ones.begin(), ones.end());
                nodes_added_at_stage += length_cover;
                values.insert(values.end(), cover.begin(), cover.end());
            }

            num_nodes_at_stage = nodes_added_at_stage;
            cursor = cursor_new;
            std::vector<int> ones(num_nodes_at_stage, 1);
            stages.insert(stages.end(), ones.begin(), ones.end());
        }

        for (int stage_idx = __stopping_time; stage_idx < __num_stages; ++stage_idx) {
            std::vector<int> nodes(cursor, cursor + num_nodes_at_stage);
            ancestors.insert(ancestors.end(), nodes.begin(), nodes.end());
            cursor += num_nodes_at_stage;
            std::vector<int> ones(num_nodes_at_stage, 1);
            stages.insert(stages.end(), ones.begin(), ones.end());
            values.insert(values.end(), values.end() - num_nodes_at_stage, values.end());
        }

        return std::make_tuple(ancestors, values, stages);
    }

    std::vector<double> __make_probability_values(const std::vector<int>& ancestors, const std::vector<int>& values, const std::vector<int>& stages) const {
        int num_nonzero_init_distr = std::count_if(__initial_distribution.begin(), __initial_distribution.end(),
            [](double x) { return x > 0; });

        std::vector<double> probs(num_nonzero_init_distr + 1, 0.0);
        probs[0] = 1.0;
        for (int i = 1; i <= num_nonzero_init_distr; ++i) {
            probs[i] = __initial_distribution[values[i]];
        }

        int num_nodes = values.size();
        int index = 0;

        for (int i = num_nonzero_init_distr + 1; i < num_nodes; ++i) {
            if (stages[i] == __stopping_time + 1) {
                index = i;
                break;
            }
            double probs_new = probs[ancestors[i]] * __transition_prob[values[ancestors[i]]][values[i]];
            probs.push_back(probs_new);
            index = i;
        }

        for (int j = index; j < num_nodes; ++j) {
            double probs_new = probs[ancestors[j]];
            probs.push_back(probs_new);
        }

        return probs;
    }

    std::vector<int> __cover(int i) const {
        std::vector<double> pi = __transition_prob[i];
        std::vector<int> result;
        for (int j = 0; j < pi.size(); ++j) {
            if (pi[j] > 0.0) {
                result.push_back(j);
            }
        }
        return result;
    }
};
