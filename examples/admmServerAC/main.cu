#include <spock.cuh>
#define real_t double  // templates type defaults to double


int main() {
    real_t minute = 60.;
    real_t avgTimeCp = 0.;
    real_t avgTimeAdmm = 0.;
    try {
        /* SCENARIO TREE */
        std::cout << "Reading tree files...\n";
        ScenarioTree<real_t> tree;

        /* PROBLEM DATA */
        std::cout << "Reading problem files...\n";
        ProblemData<real_t> problem(tree);

        /* CACHE::CP */
        real_t tol = 1e-3;
        real_t maxTime = 5 * minute;
        std::cout << "Allocating CP cache...\n";
        CacheBuilder builder(tree, problem);
        Cache cacheCp = builder.tol(tol).maxTimeSecs(maxTime).enableDebug().build();
        /* ::ADMM */
        std::cout << "Allocating ADMM cache...\n";
        Cache cacheAdmm = builder.tol(tol).enableAdmm().maxIters(10000).build();

        /* TIMING ALGORITHM */
        DTensor<real_t> d_initState = DTensor<real_t>::parseFromFile(tree.path() + "initialState" + tree.fpFileExt());
        std::vector<real_t> initState(tree.numStates());
        d_initState.download(initState);
        size_t runs = 1;
        size_t warm = 0;
        size_t totalRuns = runs + warm;
        std::vector<real_t> runTimesCp(totalRuns, 0.);
        std::vector<real_t> runTimesAdmm(totalRuns, 0.);
        std::cout << "Computing average solve time over (" << runs << ") runs with (" << warm << ") warm up runs...\n";
        for (size_t i = 0; i < totalRuns; i++) {
            int status = cacheCp.runCp(initState);
            if (status != converged) throw std::runtime_error(toString(status));
            runTimesCp[i] = cacheCp.solveTime();
            if (i < totalRuns - 1) cacheCp.reset();
            std::cout << "CP  : Run (" << i << ") : " << runTimesCp[i] << " s.\n";
            status = cacheAdmm.runCp(initState);
//            if (status != converged) throw std::runtime_error(toString(status));
            runTimesAdmm[i] = cacheAdmm.solveTime();
            if (i < totalRuns - 1) cacheAdmm.reset();
            std::cout << "ADMM: Run (" << i << ") : " << runTimesAdmm[i] << " s.\n";
        }
        real_t timeCp = std::reduce(runTimesCp.begin() + warm, runTimesCp.end());
        avgTimeCp = timeCp / runs;
        real_t timeAdmm = std::reduce(runTimesAdmm.begin() + warm, runTimesAdmm.end());
        avgTimeAdmm = timeAdmm / runs;

        /* PRINT */
        std::vector<real_t> states = cacheCp.states();
        std::vector<real_t> inputs = cacheCp.inputs();
        std::cout << "Cp: States:\n";
        for (size_t i = 0; i < tree.numStates() * tree.numNodes(); i++) {
            std::cout << "    " << states[i] << "\n";
        }
        states = cacheAdmm.states();
        std::cout << "Admm: States:\n";
        for (size_t i = 0; i < tree.numStates() * tree.numNodes(); i++) {
            std::cout << "    " << states[i] << "\n";
        }
        std::cout << "Cp: Inputs:\n";
        for (size_t i = 0; i < tree.numInputs() * tree.numNonleafNodes(); i++) {
            std::cout << "    " << inputs[i] << "\n";
        }
        inputs = cacheAdmm.inputs();
        std::cout << "Admm: Inputs:\n";
        for (size_t i = 0; i < tree.numInputs() * tree.numNonleafNodes(); i++) {
            std::cout << "    " << inputs[i] << "\n";
        }
    } catch (const std::exception &e) {
        std::cout << "SPOCK failed! : " << e.what() << std::endl;
    } catch (...) {
        std::cout << "SPOCK failed! : No error info.\n";
    }

    /* SAVE */
    std::ofstream file;
    file.open("time.csv", std::ios::app);
    file << avgTimeCp << ", " << avgTimeAdmm << std::endl;
    file.close();
    std::cout << "Saved (avgTimeCp = " << avgTimeCp << " s, avgTimeAdmm = " << avgTimeAdmm << " s).\n";

    return 0;
}
