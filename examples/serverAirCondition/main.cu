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
        real_t maxTime = 5 * minute;
        std::cout << "Allocating cache...\n";
        CacheBuilder builder(tree, problem);
        Cache cache = builder.tol(tol).maxTimeSecs(maxTime).build();

        /* TIMING ALGORITHM */
        DTensor<real_t> d_initState = DTensor<real_t>::parseFromFile(tree.path() + "initialState" + tree.fpFileExt());
        std::vector<real_t> initState(tree.numStates());
        d_initState.download(initState);
        size_t runs = 2;
        size_t warm = 3;
        size_t totalRuns = runs + warm;
        std::vector<real_t> runTimes(totalRuns, 0.);
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
    } catch (const std::exception &e) {
        std::cout << "SPOCK failed! : " << e.what() << std::endl;
    } catch (...) {
        std::cout << "SPOCK failed! : No error info.\n";
    }

    /* SAVE */
    std::ofstream file;
    file.open("time.csv", std::ios::app);
    file << avgTime << std::endl;
    file.close();
    std::cout << "Saved (avgTime = " << avgTime << " s).\n";

    return 0;
}
