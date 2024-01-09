#include <iostream>
#include <ctime>
#include <typeinfo>
using namespace std;
#define number_type double

number_type a1[] = {1, 2, 3, 4, 0};
number_type a2[] = {9, 8, 7};
number_type* A[] = {a1, a2};

template <typename t, size_t n>
void print_array(t (&array)[n]) {
    for (int i : array) {
        cout << i << "\n";
    }
}

int main() {
    double x = A[0][1] * A[1][2]; 
    cout << x;
    cout << "\n";
    print_array(a1);
    cout << "\n";
    print_array(a2);
    cout << "\n";
    cout << typeid(a1).name() << " --- " << typeid(A[0]).name() << "\n";

    cout <<
    "\nA[0][0] : " << A[0][0] << "\n";
    return 0;
}
