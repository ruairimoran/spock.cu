/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/gpu.cuh"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cache.cuh"

#define real_t double  // templates type defaults to double


int main() {
    bool debug = false;

    /* SCENARIO TREE */
    std::cout << "Reading tree files...\n";
    ScenarioTree<real_t> tree;
    if (debug) std::cout << tree;

    /* PROBLEM DATA */
    std::cout << "Reading problem files...\n";
    ProblemData<real_t> problem(tree);
    if (debug) std::cout << problem;

    /* CACHE */
    real_t tol = 1e-3;
    size_t maxOuterIters = 50000;
    size_t maxInnerIters = 8;
    size_t andersonBuffer = 3;
    bool allowK0Updates = true;
    bool admm = false;
    std::cout << "Allocating cache...\n";
    Cache cache(tree, problem, tol, tol, maxOuterIters, maxInnerIters, andersonBuffer, allowK0Updates, debug, admm);

    /* TIMING ALGORITHM */
    std::vector<real_t> initState(tree.numStates(), .9);
    size_t runs = 3;
    size_t warm = 5;
    size_t totalRuns = runs + warm;
    std::vector<real_t> runTimes(totalRuns, 0.);
    std::vector<size_t> runIters(totalRuns, 0);
    std::cout << "Computing average solve time over (" << runs << ") runs with (" << warm << ") warm up runs...\n";
    for (size_t i = 0; i < totalRuns; i++) {
        runTimes[i] = cache.timeSp(initState);
        runIters[i] = cache.iters();
        cache.reset();
        std::cout << "Run (" << i << ") : " << runIters[i] << " iters in " << runTimes[i] << " ms.\n";
    }
    real_t time = std::reduce(runTimes.begin() + warm, runTimes.end());
    real_t avgTime = time / runs;
    size_t iter = std::reduce(runIters.begin() + warm, runIters.end());
    size_t avgIter = iter / runs;

    /* SAVE */
    std::ofstream timeScaling;
    timeScaling.open("misc/timeScaling.csv", std::ios::app);
    timeScaling << tree.numNodes() << ", " << tree.numStates() << ", " << avgIter << ", " << avgTime << std::endl;
    timeScaling.close();
    std::cout << "Saved (avgIter = " << avgIter << ", avgTime = " << avgTime << " ms).\n";

    return 0;
}
