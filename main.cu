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
    bool detectInfeas = false;
    T tol = 1e-5;
    size_t maxOuterIters = 3000;
    size_t maxInnerIters = 8;
    size_t andersonBuffer = 3;
    bool allowK0Updates = false;
    Cache<T> cacheA(tree, problem, detectInfeas, tol, maxOuterIters);
    Cache<T> cacheB(tree, problem, detectInfeas, tol, maxOuterIters, maxInnerIters, andersonBuffer, allowK0Updates);

    /** ALGORITHM */
    size_t exit_status;
    std::vector<T> initState(problem.numStates(), .1);
    exit_status = cacheB.timeSp(initState);
    std::cout << "spock exit status: " << exit_status << std::endl;
    exit_status = cacheA.timeCp(initState);
    std::cout << "cp exit status: " << exit_status << std::endl;

    return 0;
}
