#include <gtest/gtest.h>
#include "../include/gpu.cuh"


class OrderMajorTest : public testing::Test {

protected:
    size_t m_rows = 4;
    size_t m_cols = 3;
    std::vector<real_t> m_expectedRowOrder = {1, 2, 3,
                                              4, 5, 6,
                                              7, 8, 9,
                                              10, 11, 12};
    std::vector<real_t> m_expectedColOrder = {1, 4, 7, 10,
                                              2, 5, 8, 11,
                                              3, 6, 9, 12};

    OrderMajorTest() {};

    virtual ~OrderMajorTest() {}
};

TEST_F(OrderMajorTest, Row2Col) {
    std::vector<real_t> Arow(m_expectedRowOrder);
    std::vector<real_t> Acol;
    row2col(Acol, Arow, m_rows, m_cols);
    ASSERT_EQ(Acol, m_expectedColOrder);
}

TEST_F(OrderMajorTest, Col2Row) {
    std::vector<real_t> Acol(m_expectedColOrder);
    std::vector<real_t> Arow;
    col2row(Arow, Acol, m_rows, m_cols);
    ASSERT_EQ(Arow, m_expectedRowOrder);
}

TEST_F(OrderMajorTest, Row2ColSameStorage) {
    std::vector<real_t> A(m_expectedRowOrder);
    row2col(A, A, m_rows, m_cols);
    ASSERT_EQ(A, m_expectedColOrder);
}

TEST_F(OrderMajorTest, Col2RowSameStorage) {
    std::vector<real_t> A(m_expectedColOrder);
    col2row(A, A, m_rows, m_cols);
    ASSERT_EQ(A, m_expectedRowOrder);
}
