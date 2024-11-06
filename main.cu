/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/gpu.cuh"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cache.cuh"

#define T double


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
    T tol = 1e-20;
    size_t maxOuterIters = 100;
    size_t maxInnerIters = 8;
    size_t andersonBuffer = 3;
    bool log = true;
    bool detectInfeas = false;
    bool allowK0Updates = true;
    Cache<T> cacheA(tree, problem, tol, maxOuterIters, log);
    Cache<T> cacheB(tree, problem, tol, maxOuterIters, false, detectInfeas, maxInnerIters, andersonBuffer, allowK0Updates);

    /** ALGORITHM */
    size_t exit_status;
    std::vector<T> initState(problem.numStates(), 1.);
    /* CP */
//    exit_status = cacheA.timeCp(initState);
//    std::cout << "cp exit status: " << exit_status << std::endl;
    /* SP */
    exit_status = cacheB.timeSp(initState);
    std::cout << "spock exit status: " << exit_status << std::endl;

    return 0;
}
