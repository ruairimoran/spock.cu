#include "../include/gpu.cuh"


TEMPLATE_WITH_TYPE_T
class thisOne {
private:
    DTensor<T> *d_mat = nullptr;

public:
    thisOne() {
        size_t n = 3;
        std::vector<T> mat{1, 1, 1,
                                1, 1, 1,
                                1, 1, 1};
        d_mat = new DTensor<T>(mat, n, n, 1);
    }

    ~thisOne() {}

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
        DTensor<T> slice(*d_mat, 1, 0, 0);
        std::vector<T> values = {3., 4., 5.};
        slice.upload(values);
        std::cout << *d_mat;
    }
};

int main() {
    thisOne here;
    here.uploadSomeData();
//    here.print();

    std::vector<double> myvec(3);
    myvec[0] = 1;


    return 0;
}
