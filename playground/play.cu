#include "../include/gpu.cuh"


class thisOne {
private:
    DTensor<real_t> mats;

public:
    thisOne() {
        size_t n = 3;
        std::vector<real_t> mat{1, 1, 1,
                                1, 1, 1,
                                1, 1, 1};
        mats(DTensor<real_t>{mat, n});
        std::cout << mats << "\n";
    }

    ~thisOne() {}

    void print() {
        std::cout << mats << "\n";
    }
};

int main() {
    thisOne here;

    return 0;
}
