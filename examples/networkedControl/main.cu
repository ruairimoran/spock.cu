#include <spock.cuh>
#define real_t double  // templates type defaults to double


int main() {
    real_t minute = 60.;
    real_t avgTime = 0.;
    real_t avgIter = 0.;
    /* SCENARIO TREE */
    std::cout << "Reading tree files...\n";
    ScenarioTree<real_t> tree;

    /* PROBLEM DATA */
    std::cout << "Reading problem files...\n";
    ProblemData<real_t> problem(tree);

    try {
        /* CACHE */
        real_t tol = 1e-3;
        real_t maxTime = 5 * minute;
        std::cout << "Allocating cache...\n";
        CacheBuilder builder(tree, problem);
        Cache cache = builder.tol(tol).maxTimeSecs(maxTime).build();

        /* TIMING ALGORITHM */
        DTensor<real_t> d_initState = DTensor<real_t>::parseFromFile(tree.path() + "initialState" + tree.fpFileExt());
        std::vector<real_t> initState(tree.numStates());
        d_initState.download(initState);
        size_t runs = 1;
        size_t warm = 1;
        size_t totalRuns = runs + warm;
        std::vector<real_t> runTimes(totalRuns, 0.);
        std::vector<real_t> runIters(totalRuns, 0.);
        std::cout << "Computing average solve time over (" << runs << ") runs with (" << warm << ") warm up runs...\n";
        for (size_t i = 0; i < totalRuns; i++) {
            int status = cache.runSpock(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimes[i] = cache.solveTime();
            runIters[i] = cache.solveIter();
            if (i < totalRuns - 1) cache.reset();
            std::cout << "Run (" << i << ") : " << runTimes[i] << " s, " << runIters[i] << " iters." << std::endl;
        }
        real_t time = std::reduce(runTimes.begin() + warm, runTimes.end());
        real_t iter = std::reduce(runIters.begin() + warm, runIters.end());
        avgTime = time / runs;
        avgIter = iter / runs;

        /* PRINT */
        if (false) {
            std::cout << std::fixed << std::setprecision(2);
            std::vector<real_t> states = cache.states();
            std::vector<real_t> inputs = cache.inputs();
            std::cout << "States:\n";
            for (size_t i = 0; i < tree.numNodes(); i++) {
                for (size_t j = 0; j < tree.numStates(); j++) {
                    std::cout << "\t\t" << states[i * tree.numStates() + j];// << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "Inputs:\n";
            for (size_t i = 0; i < tree.numNonleafNodes(); i++) {
                for (size_t j = 0; j < tree.numInputs(); j++) {
                    std::cout << "\t\t" << inputs[i * tree.numInputs() + j];// << "\n";
                }
                std::cout << "\n";
            }
        }
    } catch (const std::exception &e) {
        std::cout << "SPOCK failed! : " << e.what() << std::endl;
    } catch (...) {
        std::cout << "SPOCK failed! : No error info.\n";
    }

    /* SAVE */
    std::ofstream timeScaling;
    timeScaling.open("time.csv", std::ios::app);
    timeScaling << avgTime << ", " << avgIter << std::endl;
    timeScaling.close();
    std::cout << "Saved (avgTime = " << avgTime << " s) (avgIter = " << avgIter << ").\n";

    return 0;
}
