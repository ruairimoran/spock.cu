#include <tensor.cuh>
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include "../src/cones.cuh"
#include "../src/projections.cuh"

#define real_t double


template<typename T>
T avgTime(std::vector<T> &times) {
    size_t num = times.size();
    T sum = 0;
    for (size_t i = 0; i < num; i++) {
        sum += times[i];
    }
    T avg = sum / num;
    return avg;
}


template<typename T>
void projectionSerial(size_t dim, std::vector<T> &vec) {
    std::vector<T> vecFirstPart(vec.begin(), vec.end() - 1);
    std::vector<T> squares(dim - 1);
    T sum = 0;
    for (size_t i = 0; i < dim - 2; i++) {
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
    size_t runs = 10;
    size_t coneDim = 100;
    size_t numCones = 1000;
    std::vector<real_t> numbers(coneDim * numCones);
    std::iota(std::begin(numbers), std::end(numbers), 0);

    /* Test parallel projection */
    std::vector<real_t> timesB(runs);
    std::vector<DTensor<real_t>> dataB(runs);
    for (size_t i = 0; i < runs; i++) {
        dataB[i] = DTensor<real_t>(numbers, coneDim, numCones);
        SocProjection socPara(dataB[i]);
        const auto tickB = std::chrono::high_resolution_clock::now();
        socPara.project(dataB[i]);
        const auto tockB = std::chrono::high_resolution_clock::now();
        timesB[i] = std::chrono::duration<real_t, std::milli>(tockB - tickB).count();
    }
    real_t avgB = avgTime(timesB);
    std::cout << "parallel: " << avgB << " ms" << "\n";

    /* Test serial projection */
    std::vector<real_t> timesC(runs);
    std::vector<std::vector<std::vector<real_t>>> dataC(runs);
    for (size_t i = 0; i < runs; i++) {
        dataC[i].resize(numCones);
        for (size_t j = 0; j < numCones; j++) {
            dataC[i][j] = std::vector<real_t>(numbers.begin() + coneDim * j,
                                              numbers.begin() + coneDim * (j + 1));
        }
        const auto tickC = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < numCones; j++) { projectionSerial(coneDim, dataC[i][j]); }
        const auto tockC = std::chrono::high_resolution_clock::now();
        timesC[i] = std::chrono::duration<double, std::milli>(tockC - tickC).count();
    }
    real_t avgC = avgTime(timesC);
    std::cout << "serial: " << avgC << " ms" << "\n";

    cudaDeviceReset();
    return 0;
}