#ifndef TREE_CUH
#define TREE_CUH

#include "../include/gpu.cuh"
#include <fstream>
#include <utility>


TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyNode2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyAnc2Node(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t *);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyLeaf2Zero(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyZero2Leaf(T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

TEMPLATE_WITH_TYPE_T
__global__ void k_memCpyCh2Node(T *, T *, size_t, size_t, size_t, size_t, const size_t *, const size_t *, bool);

/**
 * Parts of tree
 */
enum TreePart {
    nonleaf,
    leaf
};


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
    std::string m_jsonFileName = "data.json";
    std::string m_fp;
    std::string m_fileExt = ".bt";
    /* Host data */
    size_t m_numEvents = 0;  ///< Total number of possible events
    size_t m_numNodes = 0;  ///< Total number of nodes (incl. root)
    size_t m_numNonleafNodes = 0;  ///< Total number of nonleaf nodes (incl. root)
    size_t m_numStages = 0;  ///< Total number of stages (incl. root)
    size_t m_numLeafNodes = 0;
    size_t m_numNodesMinus1 = 0;
    size_t m_numNonleafNodesMinus1 = 0;
    size_t m_numLeafNodesMinus1 = 0;
    size_t m_numStagesMinus1 = 0;  ///< horizon
    size_t m_numStates = 0;  ///< Total number system states
    size_t m_numInputs = 0;  ///< Total number control inputs
    size_t m_numStatesAndInputs = 0;
    std::vector<size_t> m_childFrom;
    std::vector<size_t> m_childTo;
    std::vector<size_t> m_numChildren;
    std::vector<size_t> m_stageFrom;
    std::vector<size_t> m_stageTo;
    std::vector<size_t> m_maxCh;  ///< Max number of children of any node of stage at index
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
        out << "Number of states: " << m_numStates << "\n";
        out << "Number of inputs: " << m_numInputs << "\n";
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
    explicit ScenarioTree(std::string pathToDataFolder = "./data/") : m_pathToDataFolder(std::move(pathToDataFolder)) {
        std::ifstream file(m_pathToDataFolder + m_jsonFileName);
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());

        if (doc.HasParseError()) {
            err << "[Tree] Error parsing tree JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw ERR;
        }

        /* Store single element data from JSON in host memory */
        m_numEvents = doc["numEvents"].GetInt();
        m_numNonleafNodes = doc["numNonleafNodes"].GetInt();
        m_numNodes = doc["numNodes"].GetInt();
        m_numStages = doc["numStages"].GetInt();
        m_numStates = doc["numStates"].GetInt();
        m_numInputs = doc["numInputs"].GetInt();
        m_numStatesAndInputs = m_numStates + m_numInputs;

        /* Assign file extensions */
        if constexpr (std::is_same_v<T, float>) { m_fp = "_f" + m_fileExt; }
        else if constexpr (std::is_same_v<T, double>) { m_fp = "_d" + m_fileExt; }
        std::string ui = "_u" + m_fileExt;

        /* Read tensors onto device */
        // Note that ancestors[0] and events[0] will be max(size_t) on device because they are -1 on host
        m_d_stages = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "stages" + ui));
        m_d_ancestors = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "ancestors" + ui));
        m_d_probabilities = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_pathToDataFolder + "probabilities" + m_fp));
        m_d_conditionalProbabilities = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_pathToDataFolder + "conditionalProbabilities" + m_fp));
        m_d_events = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "events" + ui));
        m_d_childFrom = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "childrenFrom" + ui));
        m_d_childTo = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "childrenTo" + ui));
        m_d_numChildren = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "numChildren" + ui));
        m_d_stageFrom = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "stageFrom" + ui));
        m_d_stageTo = std::make_unique<DTensor<size_t>>(
            DTensor<size_t>::parseFromFile(m_pathToDataFolder + "stageTo" + ui));

        /* Allocate memory on host for data */
        m_childFrom = std::vector<size_t>(m_numNonleafNodes);
        m_childTo = std::vector<size_t>(m_numNonleafNodes);
        m_numChildren = std::vector<size_t>(m_numNonleafNodes);
        m_stageFrom = std::vector<size_t>(m_numStages);
        m_stageTo = std::vector<size_t>(m_numStages);
        m_maxCh = std::vector<size_t>(m_numStages);

        /* Download data to host */
        m_d_childFrom->download(m_childFrom);
        m_d_childTo->download(m_childTo);
        m_d_numChildren->download(m_numChildren);
        m_d_stageFrom->download(m_stageFrom);
        m_d_stageTo->download(m_stageTo);

        /* Update remaining fields */
        m_numLeafNodes = m_numNodes - m_numNonleafNodes;
        m_numNodesMinus1 = m_numNodes - 1;
        m_numNonleafNodesMinus1 = m_numNonleafNodes - 1;
        m_numLeafNodesMinus1 = m_numLeafNodes - 1;
        m_numStagesMinus1 = m_numStages - 1;
        for (size_t stage = 0; stage < m_numStagesMinus1; stage++) {
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
    ~ScenarioTree() = default;

    /**
     * Getters
     */
    std::string strOfPart(TreePart part) {
        if (part == nonleaf) return "nonleaf";
        else if (part == leaf) return "leaf";
        else { err << "[tree::strOfPart] tree part not valid\n"; throw ERR; }
    }

    size_t numNodesOfPart(TreePart part) {
        if (part == nonleaf) return m_numNonleafNodes;
        else if (part == leaf) return m_numLeafNodes;
        else { err << "[tree::numNodesOfPart] tree part not valid\n"; throw ERR; }
    }

    std::string path() { return m_pathToDataFolder; }

    std::string json() { return m_jsonFileName; }

    std::string fpFileExt() { return m_fp; }

    size_t numEvents() { return m_numEvents; }

    size_t numNodes() { return m_numNodes; }

    size_t numNonleafNodes() { return m_numNonleafNodes; }

    size_t numLeafNodes() { return m_numLeafNodes; }

    size_t numNodesMinus1() { return m_numNodesMinus1; }

    size_t numNonleafNodesMinus1() { return m_numNonleafNodesMinus1; }

    size_t numLeafNodesMinus1() { return m_numLeafNodesMinus1; }

    size_t numStages() { return m_numStages; }

    size_t numStagesMinus1() { return m_numStagesMinus1; }

    size_t numStates() { return m_numStates; }

    size_t numInputs() { return m_numInputs; }

    size_t numStatesAndInputs() { return m_numStatesAndInputs; }

    std::vector<size_t> &childFrom() { return m_childFrom; }

    std::vector<size_t> &childTo() { return m_childTo; }

    std::vector<size_t> &numChildren() { return m_numChildren; }

    std::vector<size_t> &childMax() { return m_maxCh; }

    std::vector<size_t> &stageFrom() { return m_stageFrom; }

    std::vector<size_t> &stageTo() { return m_stageTo; }

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

    void memCpyAnc2Node(DTensor<T> &, DTensor<T> &, size_t, size_t, size_t, size_t = 0, size_t = 0);

    /**
     * Caution! Dst must not be src.
     */
    void memCpyCh2Node(DTensor<T> &dst, DTensor<T> &src, size_t, size_t, size_t, bool = false);

    /**
     * For leaf transfers, you must transfer all leaf nodes! So `nodeFrom` == numNonleafNodes.
     * The `nodeFrom/To` requires the actual node numbers (not zero-indexed).
     */
    void memCpyLeaf2Zero(DTensor<T> &, DTensor<T> &, size_t, size_t = 0, size_t = 0);

    void memCpyZero2Leaf(DTensor<T> &, DTensor<T> &, size_t, size_t = 0, size_t = 0);
};

static void setKernelDimensions(dim3 &blocks, dim3 &threads, size_t nX, size_t nY) {
    size_t threadsPerBlockPerAxis = 32;  // Do not change! Max number of threads per block is 32x32=1024
    threads = dim3(std::min(nX, threadsPerBlockPerAxis), std::min(nY, threadsPerBlockPerAxis));
    blocks = dim3(numBlocks(nX, threads.x), numBlocks(nY, threads.y));
}

template<typename T>
static void memCpyNode2Node(DTensor<T> &dst, DTensor<T> &src,
                            size_t nodeFrom, size_t nodeTo, size_t numEl,
                            size_t elFromDst = 0, size_t elFromSrc = 0) {
    size_t nodeSizeDst = dst.numRows();
    size_t nodeSizeSrc = src.numRows();
    if (dst.numCols() != 1 || src.numCols() != 1)
        throw std::invalid_argument("[scenarioTree::memCpy::node2Node] numCols must be 1.");
    dim3 blocks, threads;
    setKernelDimensions(blocks, threads, nodeTo - nodeFrom + 1, numEl);
    k_memCpyNode2Node<<<blocks, threads>>>(dst.raw(), src.raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                           nodeSizeSrc, elFromDst, elFromSrc);
}

template<typename T>
void ScenarioTree<T>::memCpyAnc2Node(DTensor<T> &dst, DTensor<T> &src,
                                     size_t nodeFrom, size_t nodeTo, size_t numEl,
                                     size_t elFromDst, size_t elFromSrc) {
    size_t nodeSizeDst = dst.numRows();
    size_t nodeSizeSrc = src.numRows();
    if (dst.numCols() != 1 || src.numCols() != 1)
        throw std::invalid_argument("[scenarioTree::memCpy::anc2Node] numCols must be 1.");
    if (nodeFrom < 1)
        throw std::invalid_argument("[scenarioTree::memCpy::anc2Node] Root node has no ancestor.");
    dim3 blocks, threads;
    setKernelDimensions(blocks, threads, nodeTo - nodeFrom + 1, numEl);
    k_memCpyAnc2Node<<<blocks, threads>>>(dst.raw(), src.raw(), nodeFrom, nodeTo, numEl, nodeSizeDst,
                                          nodeSizeSrc, elFromDst, elFromSrc, m_d_ancestors->raw());
}

template<typename T>
void ScenarioTree<T>::memCpyCh2Node(DTensor<T> &dst, DTensor<T> &src,
                                    size_t nodeFrom, size_t nodeTo, size_t chIdx, bool add) {
    size_t numEl = src.numRows();
    if (dst.numCols() != 1 || src.numCols() != 1)
        throw std::invalid_argument("[scenarioTree::memCpy::ch2Node] numCols must be 1.");
    if (dst.numRows() != numEl)
        throw std::invalid_argument("[scenarioTree::memCpy::ch2Node] Source and destination dimensions mismatch.");
    dim3 blocks, threads;
    setKernelDimensions(blocks, threads, nodeTo - nodeFrom + 1, numEl);
    k_memCpyCh2Node<<<blocks, threads>>>(dst.raw(), src.raw(),
                                         nodeFrom, nodeTo, numEl, chIdx,
                                         m_d_childFrom->raw(), m_d_numChildren->raw(), add);
}

template<typename T>
void ScenarioTree<T>::memCpyLeaf2Zero(DTensor<T> &dst, DTensor<T> &src,
                                      size_t numEl, size_t elFromDst, size_t elFromSrc) {
    size_t nodeSizeDst = dst.numRows();
    size_t nodeSizeSrc = src.numRows();
    if (dst.numCols() != 1 || src.numCols() != 1)
        throw std::invalid_argument("[scenarioTree::memCpy::leaf2Zero] numCols must be 1.");
    dim3 blocks, threads;
    setKernelDimensions(blocks, threads, m_numLeafNodes, numEl);
    k_memCpyLeaf2Zero<<<blocks, threads>>>(dst.raw(), src.raw(), m_numNonleafNodes, m_numNodesMinus1, numEl,
                                           nodeSizeDst, nodeSizeSrc, elFromDst, elFromSrc);
}

template<typename T>
void ScenarioTree<T>::memCpyZero2Leaf(DTensor<T> &dst, DTensor<T> &src,
                                      size_t numEl, size_t elFromDst, size_t elFromSrc) {
    size_t nodeSizeDst = dst.numRows();
    size_t nodeSizeSrc = src.numRows();
    if (dst.numCols() != 1 || src.numCols() != 1)
        throw std::invalid_argument("[scenarioTree::memCpy::zero2Leaf] numCols must be 1.");
    dim3 blocks, threads;
    setKernelDimensions(blocks, threads, m_numLeafNodes, numEl);
    k_memCpyZero2Leaf<<<blocks, threads>>>(dst.raw(), src.raw(), m_numNonleafNodes, m_numNodesMinus1, numEl,
                                           nodeSizeDst, nodeSizeSrc, elFromDst, elFromSrc);
}

#endif
