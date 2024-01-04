#include <iostream>
#include <ctime>
using namespace std;
#define N 1e9

void inline inline_add(int *p) { *p += 1; }

void outline_add(int *p) {
    *p += 1;
}


int main() {
    int x = 1;
    int *px = &x;
    clock_t time0 = clock();
    for (int i=0; i<N; i++) {
        inline_add(px);
    }
    clock_t time1 = clock();
    x = 1;
    clock_t time2 = clock();
    for (int i=0; i<N; i++) {
        outline_add(px);
    }
    clock_t time3 = clock();
    float in = (float) (time1 - time0) / CLOCKS_PER_SEC;
    float out = (float) (time3 - time2) / CLOCKS_PER_SEC;
    cout << "N = " << N << "\ninline: " << in << "\noutline: " << out << "\n";
    // cout << "\n" << time0 << "\n" << time1 << "\n" << time2 << "\n" << time3;
    return 0;
}
