#include "../include/gpu.cuh"


class thisOne {
private:
    DTensor<real_t> *d_mat = nullptr;

public:
    thisOne() {
        size_t n = 3;
        std::vector<real_t> mat{1, 1, 1,
                                1, 1, 1,
                                1, 1, 1};
        d_mat = new DTensor<real_t>(mat, n*n);
    }

    ~thisOne() {}

    template<typename T>
    void printIf(DTensor<T> *data, std::string description) const {
        if (data) {
            std::cout << description << *data;
        } else {
            std::cout << description << "HAS NO DATA TO PRINT.";
        }
    }

    void print() {
        printIf(d_mat, "here it is: ");
    }
};

int main() {
    thisOne here;
    here.print();

    return 0;
}
