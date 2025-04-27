#include <spock.cuh>
#define real_t double  // templates type defaults to double


int main() {
    real_t minute = 60.;
    real_t avgTime = 0.;
    try {
        /* SCENARIO TREE */
        std::cout << "Reading tree files...\n";
        ScenarioTree<real_t> tree;

        /* PROBLEM DATA */
        std::cout << "Reading problem files...\n";
        ProblemData<real_t> problem(tree);

        /* CACHE */
        real_t tol = 1e-3;
        real_t maxTime = .25 * minute;
        std::cout << "Allocating cache...\n";
        CacheBuilder builder(tree, problem);
        Cache cache = builder.toleranceAbsolute(tol).maxTimeSecs(maxTime).build();

        /* TIMING ALGORITHM */
        size_t runs = 1;
        size_t warm = 0;
        size_t totalRuns = runs + warm;
        std::vector<real_t> runTimes(totalRuns, 0.);
        DTensor<real_t> d_initState = DTensor<real_t>::parseFromFile(tree.path() + "initialState" + tree.fpFileExt());
        std::vector<real_t> initState(tree.numStates());
        d_initState.download(initState);
        std::cout << "Computing average solve time over (" << runs << ") runs with (" << warm << ") warm up runs...\n";
        for (size_t i = 0; i < totalRuns; i++) {
            int status = cache.runSpock(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimes[i] = cache.solveTime();
            if (i < totalRuns - 1) cache.reset();
            std::cout << "Run (" << i << ") : " << runTimes[i] << " s.\n";
        }
        real_t time = std::reduce(runTimes.begin() + warm, runTimes.end());
        avgTime = time / runs;

        /* PRINT */
        std::vector<real_t> states = cache.states();
        std::vector<real_t> inputs = cache.inputs();
        std::cout << "States:\n";
        for (size_t i = 0; i < tree.numStates() * tree.numNodes(); i++) {
            std::cout << "    " << states[i] << "\n";
        }
        std::cout << "Inputs:\n";
        for (size_t i = 0; i < tree.numInputs() * tree.numNonleafNodes(); i++) {
            std::cout << "    " << inputs[i] << "\n";
        }
    } catch (const std::exception &e) {
        std::cout << "SPOCK failed! : " << e.what() << std::endl;
    } catch (...) {
        std::cout << "SPOCK failed! : No error info.\n";
    }

    /* PRINT */
    std::cout << "Avg time = " << avgTime << " s.\n";

    return 0;
}
