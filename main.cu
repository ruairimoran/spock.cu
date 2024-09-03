/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/gpu.cuh"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cache.cuh"



int main() {
    /** SCENARIO TREE */
    std::ifstream fileTree("json/treeData.json");
    ScenarioTree tree(fileTree);
//  	tree.print();

    /** PROBLEM DATA */
    std::ifstream fileProblem("json/problemData.json");
    ProblemData problem(tree, fileProblem);
//  	problem.print();

    /** CACHE */
    double tol = 1e-3;
    size_t maxIters = 3000;
    bool detectInfeasibility = false;
    Cache cache(tree, problem, tol, maxIters, detectInfeasibility);
//    cache.print();

    /** VANILLA CP */
    std::vector<double> initState(problem.numStates(), .1);
    size_t exit_status = cache.cpTime(initState);
    std::cout << "cp exit status: " << exit_status << std::endl;

    return 0;
}
