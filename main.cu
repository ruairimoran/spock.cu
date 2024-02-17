/**
 * must be run using cmake
 * e.g., (from top level)
 * $ cmake -S . -B ./build -Wno-dev && cmake --build ./build && ./build/spock
*/

#include "include/stdgpu.h"
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
    Cache cache(tree, problem);
    cache.print();

    /** VANILLA CP */
    size_t exit_status = cp(cache);
    std::cout << "cp exit status: " << exit_status << std::endl;

    return 0;
}
