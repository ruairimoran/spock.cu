#include <tensor.cuh>

#define T double


int main() {
    std::vector<T> a(9, 3.);
    DTensor<T> A(a, 3, 3, 1);
    std::vector<T> b(3, 4.);
    DTensor<T> B(b, 3, 1, 1);

    B.addAB(A, B);
    std::cout << B;

    return 0;
}