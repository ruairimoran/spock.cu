#include <tensor.cuh>

#define T double


int main() {
    DTensor<T> C(3, 3, 10, true);
    std::vector<T> a(9, 3.);
    DTensor<T> A(a, 3, 3, 1);
    std::vector<T> b(9, 4.);
    DTensor<T> B(b, 3, 3, 1);
    DTensor<T> sliceC(C, 2, 0, 0);

    /* Fails */
    sliceC = A * B;
    std::cout << sliceC;

    /* Succeeds */
    DTensor<T> AB = A * B;
    AB.deviceCopyTo(sliceC);
    std::cout << sliceC;

    return 0;
}