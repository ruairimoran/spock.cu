#include <tensor.cuh>
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include "../src/cones.cuh"
#include "../src/projections.cuh"

#define real_t double


template<typename T>
void projectionSerial(size_t dim, std::vector<T> &vec) {
    std::vector<T> vecFirstPart(vec.begin(), vec.end() - 1);
    std::vector<T> squares(dim - 1);
    T sum = 0;
    for (size_t i=0;i<dim-2;i++) {
        T temp = vecFirstPart[i];
        squares[i] = temp * temp;
        sum += squares[i];
    }
    T nrm = sqrt(sum);
    float vecLastElement = vec[dim - 1];
    if (nrm <= vecLastElement) {
        return;  // Do nothing!
    } else if (nrm <= -vecLastElement) {
        for (size_t i = 0; i < dim; i++) { vec[i] = 0.; }
    } else {
        T scaling = (nrm + vecLastElement) / (2. * nrm);
        for (size_t i = 0; i < dim - 1; i++) { vec[i] *= scaling; }
        vec[dim - 1] = scaling * nrm;
    }
}


int main() {
    /* Create data */
    size_t coneDim = 100;
    size_t numCones = 1000;
    std::vector<real_t> numbers(coneDim * numCones);
    std::iota(std::begin(numbers), std::end(numbers), 0);

    /* Test parallel projection */
    DTensor<real_t> d_dataB(numbers, coneDim, numCones);
    SocProjection socPara(d_dataB);
    const auto tickB = std::chrono::high_resolution_clock::now();
    socPara.project(d_dataB);
    const auto tockB = std::chrono::high_resolution_clock::now();
    double durationMilliB = std::chrono::duration<double, std::milli>(tockB - tickB).count();
    std::cout << "parallel timer stopped:  " << durationMilliB << " ms" << "\n";

    /* Test serial projection */
    std::vector<std::vector<real_t>> split(numCones);
    for (size_t i = 0; i < numCones; i++) { split[i] = std::vector<real_t>(numbers.begin() + coneDim * i,
                                                                           numbers.begin() + coneDim * (i + 1)); }
    const auto tickC = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numCones; i++) { projectionSerial(coneDim, split[i]); }
    const auto tockC = std::chrono::high_resolution_clock::now();
    double durationMilliC = std::chrono::duration<double, std::milli>(tockC - tickC).count();
    std::cout << "serial timer stopped:  " << durationMilliC << " ms" << "\n";

    cudaDeviceReset();
    return 0;
}