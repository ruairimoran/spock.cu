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
        d_mat = new DTensor<real_t>(mat, n, n, 1);
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

    void uploadSomeData(){
        std::cout << *d_mat;
        DTensor<real_t> slice(*d_mat, 1, 0, 0);
        std::vector<real_t> values = {3., 4., 5.};
        slice.upload(values);
        std::cout << *d_mat;
    }
};

int main() {
    thisOne here;
    here.uploadSomeData();
//    here.print();

    std::vector<real_t> myvec(3);
    myvec[0] = 1;


    return 0;
}
