#include <spock.cuh>
#define real_t double  // templates type defaults to double


int main() {
    real_t minute = 60.;
    real_t runTimes3 = 0.;
    real_t runTimes5 = 0.;
    real_t runTimes7 = 0.;
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
        Cache cache3 = builder.tol(tol).maxTimeSecs(maxTime).andersonBuffer(3).build();
        Cache cache5 = builder.tol(tol).maxTimeSecs(maxTime).andersonBuffer(5).build();
        Cache cache7 = builder.tol(tol).maxTimeSecs(maxTime).andersonBuffer(7).build();

        /* TIMING ALGORITHM */
        DTensor<real_t> d_initState = DTensor<real_t>::parseFromFile(tree.path() + "initialState" + tree.fpFileExt());
        std::vector<real_t> initState(tree.numStates());
        d_initState.download(initState);
        size_t runs = 1;
        size_t warm = 1;
        size_t totalRuns = runs + warm;
        std::cout << "Computing average solve time over (" << runs << ") runs with (" << warm << ") warm up runs...\n";
        int status = 0;
        for (size_t i = 0; i < totalRuns; i++) {
            status = cache3.runSpock(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimes3 = cache3.solveTime();
            cache3.reset();
            std::cout << "Run (" << i << "), buff (3) : " << runTimes3 << " s.\n";
            status = cache5.runSpock(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimes5 = cache5.solveTime();
            cache5.reset();
            std::cout << "Run (" << i << "), buff (5) : " << runTimes5 << " s.\n";
            status = cache7.runSpock(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimes7 = cache7.solveTime();
            cache7.reset();
            std::cout << "Run (" << i << "), buff (7) : " << runTimes7 << " s.\n";
        }
    } catch (const std::exception &e) {
        std::cout << "SPOCK failed! : " << e.what() << std::endl;
    } catch (...) {
        std::cout << "SPOCK failed! : No error info.\n";
    }

    /* SAVE */
    std::ofstream timeScaling;
    timeScaling.open("time.csv", std::ios::app);
    timeScaling << tree.numNodes() * tree.numStatesAndInputs() << ", "
                << tree.numNodes() << ", "
                << tree.numStatesAndInputs() << ", "
                << runTimes3 << ", "
                << runTimes5 << ", "
                << runTimes7
                << std::endl;
    timeScaling.close();

    return 0;
}
