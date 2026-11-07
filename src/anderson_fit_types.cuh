#ifndef SPOCK_ANDERSON_FIT_TYPES_CUH
#define SPOCK_ANDERSON_FIT_TYPES_CUH

#include <cstddef>
#include <vector>

enum class AndersonFitBackend { gpu, cpu };

inline const char *andersonFitBackendName(AndersonFitBackend backend) {
    return backend == AndersonFitBackend::cpu ? "cpu" : "gpu";
}

enum class AndersonFitStatus {
    accepted, rankZero, nonfiniteHistory, solverFailure,
    nonfiniteSolution, coefficientLimit, directionLimit
};

inline const char *andersonStatusName(AndersonFitStatus status) {
    switch (status) {
        case AndersonFitStatus::accepted: return "accepted";
        case AndersonFitStatus::rankZero: return "rank_zero";
        case AndersonFitStatus::nonfiniteHistory: return "nonfinite_history";
        case AndersonFitStatus::solverFailure: return "solver_failure";
        case AndersonFitStatus::nonfiniteSolution: return "nonfinite_solution";
        case AndersonFitStatus::coefficientLimit: return "coefficient_limit";
        case AndersonFitStatus::directionLimit: return "direction_limit";
    }
    return "unknown";
}

struct AndersonFitInfo {
    AndersonFitStatus status = AndersonFitStatus::rankZero;
    std::vector<size_t> selectedColumns;
    std::vector<double> coefficients;
    double scale = 0.;
    double coefficientNorm = 0.;
    double directionRatio = 0.;

    bool accepted() const { return status == AndersonFitStatus::accepted; }
};

#endif
