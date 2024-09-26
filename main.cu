/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/gpu.cuh"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cache.cuh"

#define T float


int main() {
    /** SCENARIO TREE */
    std::ifstream fileTree("json/treeData.json");
    ScenarioTree<T> tree(fileTree);
//  	std::cout << tree;

    /** PROBLEM DATA */
    std::ifstream fileProblem("json/problemData.json");
    ProblemData<T> problem(tree, fileProblem);
//  	std::cout << problem;

    /** CACHE */
    T tol = 1e-5;
    size_t maxOuterIters = 1;
    size_t maxInnerIters = 8;
    bool detectInfeas = false;
    Cache<T> cacheA(tree, problem, detectInfeas, tol, maxOuterIters);
    Cache<T> cacheB(tree, problem, detectInfeas, tol, maxOuterIters, maxInnerIters);

    /** ALGORITHM */
    size_t exit_status;
    std::vector<T> initState(problem.numStates(), .1);
//    exit_status = cacheA.cpTime(initState);
//    std::cout << "cp exit status: " << exit_status << std::endl;
    exit_status = cacheB.spTime(initState);
    std::cout << "spock exit status: " << exit_status << std::endl;

    return 0;
}
