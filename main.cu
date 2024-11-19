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
    std::cout << "Reading tree file...\n";
    std::ifstream fileTree("json/treeData.json");
    ScenarioTree<T> tree(fileTree);
//  	std::cout << tree;

    /** PROBLEM DATA */
    std::cout << "Reading problem file...\n";
    std::ifstream fileProblem("json/problemData.json");
    ProblemData<T> problem(tree, fileProblem);
//  	std::cout << problem;

    /** CACHE */
    T tol = 1e-3;
    size_t maxOuterIters = 5000;
    size_t maxInnerIters = 8;
    size_t andersonBuffer = 3;
    bool detectInfeas = false;
    bool allowK0Updates = true;

    /** TIMING ALGORITHM */
    std::vector<T> initState(problem.numStates(), .1);
    size_t runs = 10;
    std::vector<T> runTimes(runs);

    std::cout << "Computing average solve time...\n";
    for (size_t i = 0; i < runs; i++) {
        Cache<T> cacheSp(tree, problem,
                         tol, maxOuterIters, false, detectInfeas, maxInnerIters, andersonBuffer, allowK0Updates);
        runTimes[i] = cacheSp.timeSp(initState);
    }
    T total = std::reduce(runTimes.begin(), runTimes.end());
    T avg = total / runs;

    /** SAVE */
    std::ofstream timeScaling;
    timeScaling.open("json/timeScaling.csv", std::ios::app);
    timeScaling << tree.numStages()-1 << ", " << problem.numStates() << ", " << avg << std::endl;
    timeScaling.close();
    std::cout << "Saved." << std::endl;

    return 0;
}
