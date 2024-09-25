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
//  	std::cout << tree;

    /** PROBLEM DATA */
    std::ifstream fileProblem("json/problemData.json");
    ProblemData problem(tree, fileProblem);
//  	std::cout << problem;

    /** CACHE */
    double tol = 1e-5;
    size_t maxOuterIters = 500;
    size_t maxInnerIters = 8;
    bool detectInfeas = false;
    Cache cacheA(tree, problem, detectInfeas, tol, maxOuterIters);
    Cache cacheB(tree, problem, detectInfeas, tol, maxOuterIters, maxInnerIters);

    /** ALGORITHM */
    size_t exit_status;
    std::vector<double> initState(problem.numStates(), .1);
//    exit_status = cacheA.cpTime(initState);
//    std::cout << "cp exit status: " << exit_status << std::endl;
    exit_status = cacheB.spTime(initState);
    std::cout << "spock exit status: " << exit_status << std::endl;

    return 0;
}
