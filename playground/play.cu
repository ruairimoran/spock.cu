#include <tensor.cuh>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <filesystem>
#include "../src/projections.cuh"


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


template<typename T>
T testParallel(size_t runs, size_t numCones, size_t coneDim, bool print = false) {
    /* Create data */
    std::vector<T> numbers(coneDim * numCones);
    std::iota(std::begin(numbers), std::end(numbers), 0);
    /* Test parallel projection */
    std::vector<T> times(runs);
    std::vector<DTensor<T>> data(runs);
    for (size_t i = 0; i < runs; i++) {
        data[i] = DTensor<T>(numbers, coneDim, numCones);
        SocProjection socPara(data[i]);
        const auto tick = std::chrono::high_resolution_clock::now();
        socPara.project(data[i]);
        const auto tock = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<T, std::milli>(tock - tick).count();
    }
    T avg = avgTime(times);
    if (print) std::cout << "parallel: " << avg << " ms @ " << coneDim << " dimension and " << numCones << " cones\n";
    return avg;
}


template<typename T>
T testSerial(size_t runs, size_t numCones, size_t coneDim, bool print = false) {
    /* Create data */
    std::vector<T> numbers(coneDim * numCones);
    std::iota(std::begin(numbers), std::end(numbers), 0);
    /* Test serial projection */
    std::vector<T> times(runs);
    std::vector<std::vector<std::vector<T>>> data(runs);
    for (size_t i = 0; i < runs; i++) {
        data[i].resize(numCones);
        for (size_t j = 0; j < numCones; j++) {
            data[i][j] = std::vector<T>(numbers.begin() + coneDim * j, numbers.begin() + coneDim * (j + 1));
        }
        const auto tick = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < numCones; j++) { projectionSerial(coneDim, data[i][j]); }
        const auto tock = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(tock - tick).count();
    }
    T avg = avgTime(times);
    if (print) std::cout << "serial:   " << avg << " ms @ " << coneDim << " dimension and " << numCones << " cones\n";
    return avg;
}


template<typename T>
void printTimes(size_t runs, size_t maxNumCones, size_t maxConeDim, std::vector<T> &times) {
    times.resize(maxNumCones * maxConeDim);
    for (size_t cone = 1; cone < maxNumCones; cone++) {
        for (size_t dim = 2; dim < maxConeDim; dim++) {
            times[cone * maxConeDim + dim] = testParallel<T>(runs, cone, dim);
        }
        std::cout << cone << "\n";
    }
}


template<typename T>
void addArrayToJson(rapidjson::Document &doc, rapidjson::GenericStringRef<char> name, std::vector<T> &vec) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < vec.size(); i++) {
        array.PushBack(vec[vec.size() - i], doc.GetAllocator());
    }
    doc.AddMember(name, array, doc.GetAllocator());
}


int main() {
    size_t runs = 10;
    size_t coneDim = 100;
    size_t numCones = 100;
    std::vector<double> times;
    printTimes(runs, numCones, coneDim, times);

    char text[65536];
    rapidjson::MemoryPoolAllocator<> allocator(text, sizeof(text));
    rapidjson::Document doc(&allocator, 256);
    doc.SetObject();
    doc.AddMember("runs", runs, doc.GetAllocator());
    doc.AddMember("coneDim", coneDim, doc.GetAllocator());
    doc.AddMember("numCones", numCones, doc.GetAllocator());
    rapidjson::GenericStringRef<char> timesName = "times";
    addArrayToJson(doc, timesName, times);

    typedef rapidjson::GenericStringBuffer<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>> StringBuffer;
    StringBuffer buffer(&allocator);
    rapidjson::Writer<StringBuffer> writer(buffer, reinterpret_cast<rapidjson::CrtAllocator *>(&allocator));
    doc.Accept(writer);
    std::string json(buffer.GetString(), buffer.GetSize());

    std::ofstream of("/home/biggirl/Documents/remote_host/raocp-parallel/playground/times.json");
    of << json;
    if (!of.good()) throw std::runtime_error("Can't write the JSON string to the file!");

    return 0;
}