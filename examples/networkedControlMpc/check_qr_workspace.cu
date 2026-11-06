#include <tensor.cuh>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

template<typename T>
void check(size_t rows, T tolerance) {
    constexpr size_t cols = 3;
    std::vector<T> a(rows * cols), b(rows);
    for (size_t i = 0; i < rows; ++i) {
        a[i] = T(1) + T(i % 7) * T(0.01);
        a[i + rows] = T(i % 11) * T(0.1) - T(0.5);
        a[i + 2 * rows] = T(i % 13) * T(0.1) - T(0.6);
        b[i] = T(0.5) * a[i] - T(0.25) * a[i + rows] + T(2) * a[i + 2 * rows];
    }
    DTensor<T> matrix(rows, cols, 1, true), rhs(rows, 1, 1, true);
    matrix.upload(a);
    rhs.upload(b);
    QRFactoriser<T> qr(matrix);
    qr.factorise();
    qr.leastSquares(rhs);
    rhs.download(b);
    const T expected[] = {T(0.5), T(-0.25), T(2)};
    T error = 0;
    for (size_t j = 0; j < cols; ++j) error = std::max(error, std::abs(b[j] - expected[j]));
    if (error > tolerance || !std::isfinite(error)) throw std::runtime_error("Incorrect QR least-squares solution");
    std::cout << "rows=" << rows << " sizeof(T)=" << sizeof(T) << " max_coefficient_error=" << error << '\n';
}

int main() {
    check<double>(115169, 1e-9);
    check<double>(2000000, 1e-9);
    check<float>(115169, 1e-4f);
    check<float>(2000000, 1e-4f);
}
