#include "include/stdgpu.h"
#include "src/cones.cuh"
#include <iostream>
#include <math.h>


int main() {
    Context context; /* Create one context only */

    std::vector<real_t> xHost{4., -5., 6., 9., 8., 5., 9., -10., 9., 11.};

    /* PosOrth(3)*/
    NonnegativeOrthantCone orthant(context, 3);

    /* SOC(4) */
    SecondOrderCone soc(context, 4);

    /* Cartesian product: X = PosOrth(3) x SOC(4) x PosOrth(3) */
    Cartesian cartesian(context);
    cartesian.addCone(orthant);
    cartesian.addCone(soc);
    cartesian.addCone(orthant);

    DeviceVector<real_t> x(xHost);
    cartesian.projectOnCone(x);

    x.download(xHost);
    printf("\nVector x after projection:\n");
    for (real_t xi: xHost) {
        printf("xi = %g\n", xi);
    }

    return 0;
}
