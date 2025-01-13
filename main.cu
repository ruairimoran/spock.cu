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
    bool debug = true;

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
    size_t maxOuterIters = 1000;
    size_t maxInnerIters = 8;
    size_t andersonBuffer = 3;
    bool allowK0Updates = true;
    std::cout << "Allocating cache...\n";
    Cache cache(tree, problem, tol, tol, maxOuterIters, maxInnerIters, andersonBuffer, allowK0Updates, debug);

    /* TIMING ALGORITHM */
    std::vector<real_t> initState(tree.numStates(), .1);
    size_t runs = 11;
    size_t warm = 1;
    size_t rmw = runs - warm;
    std::vector<real_t> runTimes(runs, 0.);
    std::cout << "Computing average solve time over (" << rmw << ") runs...\n";
    for (size_t i = 0; i < runs; i++) {
        runTimes[i] = cache.timeSp(initState);
        cache.reset();
        std::cout << "Run (" << i << ") = " << runTimes[i] << " ms.\n";
    }
    real_t total = std::reduce(runTimes.begin() + warm, runTimes.end());
    real_t avg = total / rmw;

    /* SAVE */
    std::ofstream timeScaling;
    timeScaling.open("misc/timeScaling.csv", std::ios::app);
    timeScaling << tree.numStages() << ", " << tree.numStates() << ", " << avg << std::endl;
    timeScaling.close();
    std::cout << "Saved (avg = " << avg << " ms)." << std::endl;

    return 0;
}
