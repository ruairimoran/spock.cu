#ifndef PROBLEM_CUH
#define PROBLEM_CUH

#include "../include/gpu.cuh"
#include "tree.cuh"
#include "dynamics.cuh"
#include "constraints.cuh"
#include "risks.cuh"


/**
 * Store problem data:
 * - from file
 *
 * Notes:
 * - use column-major storage
 * - `d_` indicates a device pointer
 */
TEMPLATE_WITH_TYPE_T
class ProblemData {

private:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree of problem
    size_t m_numY = 0;  ///< Size of primal vector 'y'
    T m_stepSize = 0;  ///< Step size of CP operator T
    T m_stepSizeRecip = 0;  ///< Reciprocal of step size of CP operator T
    std::unique_ptr<DTensor<T>> m_d_stepSize = nullptr;  ///< Step size of CP operator T (on device)
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtInputWeight = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtStateWeightLeaf = nullptr;
    std::unique_ptr<Dynamics<T>> m_dynamics = nullptr;
    std::unique_ptr<Constraint<T>> m_nonleafConstraint = nullptr;
    std::unique_ptr<Constraint<T>> m_leafConstraint = nullptr;
    std::unique_ptr<CoherentRisk<T>> m_risk = nullptr;
    /* Kernel projection */
    size_t m_nullDim = 0;
    std::unique_ptr<DTensor<T>> m_d_nullspaceProj = nullptr;

    void parseDynamics(const rapidjson::Value &value) {
        std::string typeStr = value["type"].GetString();
        if (typeStr == std::string("linear")) {
            m_dynamics = std::make_unique<Linear<T>>(m_tree);
        } else if (typeStr == std::string("affine")) {
            m_dynamics = std::make_unique<Affine<T>>(m_tree);
        } else {
            err << "[parseDynamics] Dynamics type " << typeStr
                << " is not supported. Supported types include: linear, affine" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

    void parseConstraint(const rapidjson::Value &value, std::unique_ptr<Constraint<T>> &constraint,
                         ConstraintMode mode) {
        std::string modeStr;
        size_t numNodes;
        if (mode == nonleaf) {
            modeStr = "nonleaf";
            numNodes = m_tree.numNonleafNodes();
        } else if (mode == leaf) {
            modeStr = "leaf";
            numNodes = m_tree.numLeafNodes();
        }
        std::string typeStr = value[modeStr.c_str()].GetString();
        std::string filePrefix = m_tree.path() + modeStr + "Constraint";
        if (typeStr == std::string("no")) {
            constraint = std::make_unique<NoConstraint<T>>();
        } else if (typeStr == std::string("rectangle")) {
            constraint = std::make_unique<Rectangle<T>>(filePrefix, m_tree.fpFileExt(), numNodes,
                                                        m_tree.numStates(), m_tree.numInputs());
        } else if (typeStr == std::string("polyhedron")) {
            constraint = std::make_unique<Polyhedron<T>>(filePrefix, m_tree.fpFileExt(), numNodes,
                                                         m_tree.numStates(), m_tree.numInputs(), mode);
        } else if (typeStr == std::string("polyhedronWithIdentity")) {
            constraint = std::make_unique<PolyhedronWithIdentity<T>>(filePrefix, m_tree.fpFileExt(), numNodes,
                                                                     m_tree.numStates(), m_tree.numInputs());
        } else {
            err << "[parseConstraint] Constraint type " << typeStr
                << " is not supported. Supported types include: none, rectangle, polyhedron, polyhedronWithIdentity\n";
            throw std::invalid_argument(err.str());
        }
    }

    void parseRisk(const rapidjson::Value &value) {
        std::string typeStr = value["type"].GetString();
        if (typeStr == std::string("avar")) {
            m_risk = std::make_unique<AVaR<T>>(m_tree.path(), m_tree.fpFileExt(), m_tree.numChildren());
        } else {
            err << "[parseRisk] Risk type " << typeStr
                << " is not supported. Supported types include: avar" << "\n";
            throw std::invalid_argument(err.str());
        }
    }

    std::ostream &print(std::ostream &out) const {
        out << "Nonleaf constraint: " << *m_nonleafConstraint;
        out << "Leaf constraint: " << *m_leafConstraint;
        out << "Risk: " << *m_risk;
        return out;
    }

public:
    /**
     * Constructor from JSON file stream
     */
    ProblemData(ScenarioTree<T> &tree) : m_tree(tree) {
        std::ifstream file(tree.path() + tree.json());
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        if (doc.HasParseError()) {
            err << "[Problem] Cannot parse problem data JSON file: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument(err.str());
        }

        /* Store single element data from JSON in host memory */
        m_nullDim = doc["rowsNNtr"].GetInt();
        m_numY = m_nullDim - (m_tree.numEvents() * 2);
        m_stepSize = doc["stepSize"].GetDouble();
        m_stepSizeRecip = 1. / m_stepSize;

        /* Allocate memory on device */
        std::string ext = m_tree.fpFileExt();
        m_d_stepSize = std::make_unique<DTensor<T>>(std::vector(1, m_stepSize), 1);
        m_d_sqrtStateWeight = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtStateCost" + ext));
        m_d_sqrtInputWeight = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtInputCost" + ext));
        m_d_sqrtStateWeightLeaf = std::make_unique<DTensor<T>>(
            DTensor<T>::parseFromFile(m_tree.path() + "sqrtTerminalCost" + ext));

        /* Parse dynamics, constraints, and risks */
        parseDynamics(doc["dynamics"]);
        parseConstraint(doc["constraint"], m_nonleafConstraint, nonleaf);
        parseConstraint(doc["constraint"], m_leafConstraint, leaf);
        parseRisk(doc["risk"]);
    }

    /**
     * Destructor
     */
    ~ProblemData() = default;

    /**
     * Getters
     */
    T stepSize() { return m_stepSize; }

    T stepSizeRecip() { return m_stepSizeRecip; }

    size_t nullDim() { return m_nullDim; }

    size_t yDim() { return m_numY; }

    DTensor<T> &d_stepSize() { return *m_d_stepSize; }

    DTensor<T> &sqrtStateWeight() { return *m_d_sqrtStateWeight; }

    DTensor<T> &sqrtInputWeight() { return *m_d_sqrtInputWeight; }

    DTensor<T> &sqrtStateWeightLeaf() { return *m_d_sqrtStateWeightLeaf; }

    std::unique_ptr<Dynamics<T>> &dynamics() { return m_dynamics; }

    std::unique_ptr<Constraint<T>> &nonleafConstraint() { return m_nonleafConstraint; }

    std::unique_ptr<Constraint<T>> &leafConstraint() { return m_leafConstraint; }

    std::unique_ptr<CoherentRisk<T>> &risk() { return m_risk; }

    friend std::ostream &operator<<(std::ostream &out, const ProblemData<T> &data) { return data.print(out); }
};


#endif
