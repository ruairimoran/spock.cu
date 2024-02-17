/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/stdgpu.h"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cones.cuh"


int main() {
    /** SCENARIO TREE */
    std::ifstream fileTree("tests/default_tree_data.json"); 
    ScenarioTree tree(fileTree);
  	tree.print();

    /** PROBLEM DATA */
    std::ifstream fileProblem("tests/default_problem_data.json"); 
    ProblemData problem(tree, fileProblem);
  	problem.print();

    /** CONES */
    Context context; /* Create one context only */
    // std::vector<real_t> xHost{4., -5., 6., 9., 8., 5., 9., -10., 9., 11.};
    // NonnegativeOrthantCone orthant(context, 3);  ///< PosOrth(3)
    // SecondOrderCone soc(context, 4);  ///< SOC(4)
    // Cartesian cartesian(context);  ///< Cartesian product: x = PosOrth(3) x SOC(4) x PosOrth(3)
    // cartesian.addCone(orthant);
    // cartesian.addCone(soc);
    // cartesian.addCone(orthant);
    // DeviceVector<real_t> x(xHost);
    // cartesian.projectOnCone(x);
    // x.download(xHost);
    // printf("\nVector x after projection:\n");
    // for (real_t xi: xHost) {
    //     printf("xi = %g\n", xi);
    // }


    return 0;
}
