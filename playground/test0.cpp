#include <iostream>
#include <ctime>
using namespace std;
#define N 1e9

void add_by_pointer(int* x) { 
    *x += 1;
};

void add_by_reference(int& loc_of_x) {
    loc_of_x += 1;
};


int main() {
    int x = 1;
    int* px = &x;
    clock_t time0 = clock();
    for (int i=0; i<N; i++) {
        add_by_pointer(px);
    }
    clock_t time1 = clock();
    x = 1;
    clock_t time2 = clock();
    for (int i=0; i<N; i++) {
        add_by_reference(x);
    }
    clock_t time3 = clock();
    float ptr = (float) (time1 - time0) / CLOCKS_PER_SEC;
    float ref = (float) (time3 - time2) / CLOCKS_PER_SEC;
    cout << "N = " << N << 
    "\nby_ptr: " << ptr << 
    "\nby_ref: " << ref << endl;
    return 0;
}
