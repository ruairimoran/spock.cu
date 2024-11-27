#ifndef TREE_CUH
#define TREE_CUH

#include "../include/gpu.cuh"
#include <fstream>
#include <utility>


/**
 * Store scenario tree data
 * - from JSON file
 *
 * Note: `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class ScenarioTree {

private:
    std::string m_pathToDataFolder;
    std::string m_jsonFile = "data.json";
    /* Host data */
    size_t m_numEvents = 0;  ///< Total number of possible events
    size_t m_numNodes = 0;  ///< Total number of nodes (incl. root)
    size_t m_numNonleafNodes = 0;  ///< Total number of nonleaf nodes (incl. root)
    size_t m_numStages = 0;  ///< Total number of stages (incl. root)
    std::vector<size_t> m_childFrom;
    std::vector<size_t> m_childTo;
    std::vector<size_t> m_numChildren;
    std::vector<size_t> m_stageFrom;
    std::vector<size_t> m_stageTo;
    std::vector<size_t> m_maxCh;  ///< Max number of children of any node of stage at index
    std::vector<size_t> m_ancestors;
    /* Device data */
    std::unique_ptr<DTensor<size_t>> m_d_stages = nullptr;  ///< Ptr to stage of node at index
    std::unique_ptr<DTensor<size_t>> m_d_ancestors = nullptr;  ///< Ptr to ancestor of node at index
    std::unique_ptr<DTensor<T>> m_d_probabilities = nullptr;  ///< Ptr to probability of visiting node at index
    std::unique_ptr<DTensor<T>> m_d_conditionalProbabilities = nullptr;  ///< Ptr to conditional probability of visiting node at index
    std::unique_ptr<DTensor<size_t>> m_d_events = nullptr;  ///< Ptr to event occurred that led to node at index
    std::unique_ptr<DTensor<size_t>> m_d_childFrom = nullptr;  ///< Ptr to first child of node at index
    std::unique_ptr<DTensor<size_t>> m_d_childTo = nullptr;  ///< Ptr to last child of node at index
    std::unique_ptr<DTensor<size_t>> m_d_numChildren = nullptr;  ///< Ptr to number of children of node at index
    std::unique_ptr<DTensor<size_t>> m_d_stageFrom = nullptr;  ///< Ptr to first node of stage at index
    std::unique_ptr<DTensor<size_t>> m_d_stageTo = nullptr;  ///< Ptr to last node of stage at index

    std::ostream &print(std::ostream &out) const {
        out << "Number of events: " << m_numEvents << "\n";
        out << "Number of nonleaf nodes: " << m_numNonleafNodes << "\n";
        out << "Number of nodes: " << m_numNodes << "\n";
        out << "Number of stages: " << m_numStages << "\n";
        printIfTensor(out, "Stages (from device): ", m_d_stages);
        printIfTensor(out, "Ancestors (from device): ", m_d_ancestors);
        printIfTensor(out, "Probabilities (from device): ", m_d_probabilities);
        printIfTensor(out, "Conditional probabilities (from device): ", m_d_conditionalProbabilities);
        printIfTensor(out, "Events (from device): ", m_d_events);
        printIfTensor(out, "Children::from (from device): ", m_d_childFrom);
        printIfTensor(out, "Children::to (from device): ", m_d_childTo);
        printIfTensor(out, "Number children (from device): ", m_d_numChildren);
        printIfTensor(out, "Stage::from (from device): ", m_d_stageFrom);
        printIfTensor(out, "Stage::to (from device): ", m_d_stageTo);
        return out;
    }

public:
    /**
     * Constructor from JSON file stream
     */
    ScenarioTree(std::string pathToDataFolder = "./data/") : m_pathToDataFolder(std::move(pathToDataFolder)) {
        std::ifstream file(m_pathToDataFolder + m_jsonFile);
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());

        if (doc.HasParseError()) {
            err << "[Tree] Error parsing tree JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument(err.str());
        }

        /* Store single element data from JSON in host memory */
        m_numEvents = doc["numEvents"].GetInt();
        m_numNonleafNodes = doc["numNonleafNodes"].GetInt();
        m_numNodes = doc["numNodes"].GetInt();
        m_numStages = doc["numStages"].GetInt();

        /* Allocate memory on host for JSON data */
        std::vector<size_t> hostStages(m_numNodes);
        std::vector<T> hostProbabilities(m_numNodes);
        std::vector<T> hostConditionalProbabilities(m_numNodes);
        std::vector<size_t> hostEvents(m_numNodes);
        m_childFrom = std::vector<size_t>(m_numNonleafNodes);
        m_childTo = std::vector<size_t>(m_numNonleafNodes);
        m_numChildren = std::vector<size_t>(m_numNonleafNodes);
        m_stageFrom = std::vector<size_t>(m_numStages);
        m_stageTo = std::vector<size_t>(m_numStages);
        m_maxCh = std::vector<size_t>(m_numStages);
        m_ancestors = std::vector<size_t>(m_numNodes);

        /* Allocate memory on device */
        m_d_stages = std::make_unique<DTensor<size_t>>(m_numNodes);
        m_d_ancestors = std::make_unique<DTensor<size_t>>(m_numNodes);
        m_d_probabilities = std::make_unique<DTensor<T>>(m_numNodes);
        m_d_conditionalProbabilities = std::make_unique<DTensor<T>>(m_numNodes);
        m_d_events = std::make_unique<DTensor<size_t>>(m_numNodes);
        m_d_childFrom = std::make_unique<DTensor<size_t>>(m_numNonleafNodes);
        m_d_childTo = std::make_unique<DTensor<size_t>>(m_numNonleafNodes);
        m_d_numChildren = std::make_unique<DTensor<size_t>>(m_numNonleafNodes);
        m_d_stageFrom = std::make_unique<DTensor<size_t>>(m_numStages);
        m_d_stageTo = std::make_unique<DTensor<size_t>>(m_numStages);

        /* Store array data from JSON in host memory */
        ///< Note that ancestors[0] and events[0] will be max(size_t) on device because they are -1 on host
        for (rapidjson::SizeType i = 0; i < m_numNodes; i++) {
            if (i < m_numNonleafNodes) {
                m_childFrom[i] = doc["childrenFrom"][i].GetInt();
                m_childTo[i] = doc["childrenTo"][i].GetInt();
                m_numChildren[i] = doc["numChildren"][i].GetInt();
            }
            hostStages[i] = doc["stages"][i].GetInt();
            m_ancestors[i] = doc["ancestors"][i].GetInt();
            hostProbabilities[i] = doc["probabilities"][i].GetDouble();
            hostConditionalProbabilities[i] = doc["conditionalProbabilities"][i].GetDouble();
            hostEvents[i] = doc["events"][i].GetInt();
        }
        for (rapidjson::SizeType i = 0; i < m_numStages; i++) {
            m_stageFrom[i] = doc["stageFrom"][i].GetInt();
            m_stageTo[i] = doc["stageTo"][i].GetInt();
        }

        /* Transfer JSON array data to device */
        m_d_stages->upload(hostStages);
        m_d_ancestors->upload(m_ancestors);
        m_d_probabilities->upload(hostProbabilities);
        m_d_conditionalProbabilities->upload(hostConditionalProbabilities);
        m_d_events->upload(hostEvents);
        m_d_childFrom->upload(m_childFrom);
        m_d_childTo->upload(m_childTo);
        m_d_numChildren->upload(m_numChildren);
        m_d_stageFrom->upload(m_stageFrom);
        m_d_stageTo->upload(m_stageTo);

        /* Update remaining fields */
        for (size_t stage = 0; stage < m_numStages - 1; stage++) {
            size_t stageFr = m_stageFrom[stage];
            size_t stageTo = m_stageTo[stage];
            size_t maxCh = 0;
            for (size_t node = stageFr; node <= stageTo; node++) { maxCh = std::max(maxCh, m_numChildren[node]); }
            m_maxCh[stage] = maxCh;
        }
    }

    /**
     * Destructor
     */
    ~ScenarioTree() {}

    /**
     * Getters
     */
    std::string path() { return m_pathToDataFolder; }

    std::string json() { return m_jsonFile; }

    size_t numEvents() { return m_numEvents; }

    size_t numNonleafNodes() { return m_numNonleafNodes; }

    size_t numLeafNodes() { return m_numNodes - m_numNonleafNodes; }

    size_t numNodes() { return m_numNodes; }

    size_t numStages() { return m_numStages; }

    std::vector<size_t> &childFrom() { return m_childFrom; }

    std::vector<size_t> &childTo() { return m_childTo; }

    std::vector<size_t> &numChildren() { return m_numChildren; }

    std::vector<size_t> &childMax() { return m_maxCh; }

    std::vector<size_t> &stageFrom() { return m_stageFrom; }

    std::vector<size_t> &stageTo() { return m_stageTo; }

    std::vector<size_t> &ancestors() { return m_ancestors; }

    DTensor<size_t> &d_stages() { return *m_d_stages; }

    DTensor<size_t> &d_ancestors() { return *m_d_ancestors; }

    DTensor<T> &d_probabilities() { return *m_d_probabilities; }

    DTensor<T> &d_conditionalProbabilities() { return *m_d_conditionalProbabilities; }

    DTensor<size_t> &d_events() { return *m_d_events; }

    DTensor<size_t> &d_childFrom() { return *m_d_childFrom; }

    DTensor<size_t> &d_childTo() { return *m_d_childTo; }

    DTensor<size_t> &d_numChildren() { return *m_d_numChildren; }

    DTensor<size_t> &d_stageFrom() { return *m_d_stageFrom; }

    DTensor<size_t> &d_stageTo() { return *m_d_stageTo; }

    friend std::ostream &operator<<(std::ostream &out, const ScenarioTree<T> &data) { return data.print(out); }
};

#endif
