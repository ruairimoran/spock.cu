/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/gpu.cuh"
#include "src/tree.cuh"
#include "src/problem.cuh"
#include "src/cache.cuh"
#include "src/cp.cuh"


int main() {
    /** SCENARIO TREE */
    std::ifstream fileTree("tests/default_tree_data.json"); 
    ScenarioTree tree(fileTree);
  	tree.print();

    /** PROBLEM DATA */
    std::ifstream fileProblem("tests/default_problem_data.json"); 
    ProblemData problem(tree, fileProblem);
  	problem.print();

    /** CACHE */
    real_t tol = 1e-4;
    size_t maxIters = 20;
    Cache cache(tree, problem, tol, maxIters);
    cache.print();

    /** VANILLA CP */
    std::vector<real_t> initState = {.1, -.2, .3};
    size_t exit_status = runCp(cache, initState);
    std::cout << "cp exit status: " << exit_status << std::endl;
    cache.print();

    return 0;
}
