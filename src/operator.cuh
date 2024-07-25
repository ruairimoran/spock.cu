#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include "../include/gpu.cuh"



/**
 * Linear operator 'L' and its adjoint
 */
TEMPLATE_WITH_TYPE_T
class LinearOperator {
protected:
    ScenarioTree<T> &m_tree;  ///< Previously created scenario tree
    ProblemData<T> &m_data;  ///< Previously created problem
    std::unique_ptr<DTensor<T>> m_d_b = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaU = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaX = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtR = nullptr;
    std::unique_ptr<DTensor<T>> m_d_sqrtQ = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaUTr = nullptr;
    std::unique_ptr<DTensor<T>> m_d_gammaXTr = nullptr;

public:
    /**
     * Constructor
     */
    LinearOperator(ScenarioTree<T> &tree, ProblemData<T> &data) :
        m_tree(tree), m_data(data) {
        std::string json((std::istreambuf_iterator<char>(m_data.file())),
                         std::istreambuf_iterator<char>());
        rapidjson::Document doc;
        doc.Parse(json.c_str());

        if (doc.HasParseError()) {
            std::cerr << "Error parsing problem data JSON: " << GetParseError_En(doc.GetParseError()) << "\n";
            throw std::invalid_argument("Cannot parse problem data JSON file");
        }

        /** Allocate space on device */
        m_d_b = std::make_unique<DTensor<T>>(m_data.numY(), 1, m_tree.numNonleafNodes(), true);

        /** Upload to device */
        const char *nodeString = nullptr;
        for (size_t i = 0; i < m_tree.numNonleafNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["b"][nodeString], m_d_b);
            parseMatrix(i, doc["gammaU"][nodeString], m_d_gammaU);
        }
        for (size_t i = 0; i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["gammaX"][nodeString], m_d_gammaX);
        }
        for (size_t i = 1; i < m_tree.numNodes(); i++) {
            nodeString = std::to_string(i).c_str();
            parseMatrix(i, doc["sqrtQ"][nodeString], m_d_sqrtR);
            parseMatrix(i, doc["sqrtQ"][nodeString], m_d_sqrtQ);
        }

        /* Update remaining fields */
        DTensor<T> gammaUTr = m_d_gammaU->tr();
        gammaUTr.deviceCopyTo(*m_d_gammaUTr);
        DTensor<T> gammaXTr = m_d_gammaX->tr();
        gammaXTr.deviceCopyTo(*m_d_gammaXTr);
    }

    ~LinearOperator() {}

    /**
     * Public methods
     */
    void op(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &,
            DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &);

    void adj(DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &,
             DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &, DTensor<T> &);
};

template<typename T>
void LinearOperator<T>::op(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                           DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                           DTensor<T> &v, DTensor<T> &vi) {
    /* I */
    y.deviceCopyTo(i);
    /* II */
    y.deviceCopyTo(ii);
}

template<typename T>
void LinearOperator<T>::adj(DTensor<T> &u, DTensor<T> &x, DTensor<T> &y, DTensor<T> &t, DTensor<T> &s,
                            DTensor<T> &i, DTensor<T> &ii, DTensor<T> &iii, DTensor<T> &iv,
                            DTensor<T> &v, DTensor<T> &vi) {

}


#endif
