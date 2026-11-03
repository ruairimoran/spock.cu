#include <spock.cuh>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// One JSON request per solve. Parameters are read outside the timed interval.
// The full primal-dual iterate survives changes to the prediction matrices.
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: networked_mpc /absolute/path/to/data/\n";
        return 2;
    }
    std::string dataPath = argv[1];
    if (dataPath.back() != '/') dataPath += '/';
    std::vector<double> previous;
    std::vector<double> previousScaling;
    checked(cudaFree(nullptr));
    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            rapidjson::Document request;
            request.Parse(line.c_str());
            if (request.HasParseError() || !request.HasMember("state"))
                throw std::runtime_error("Invalid MPC request");
            const bool warm = request["warm_start"].GetBool();
            const double tol = request["tolerance"].GetDouble();
            const double maxTime = request["max_time_s"].GetDouble();
            std::vector<double> state;
            for (const auto &x : request["state"].GetArray()) state.push_back(x.GetDouble());

            const auto setupStart = Clock::now();
            ScenarioTree<double> tree(dataPath);
            ProblemData<double> problem(tree);
            CacheBuilder<double> builder(tree, problem);
            auto cache = builder.tol(tol).maxTimeSecs(maxTime).build();
            const size_t size = cache.sizePrimal() + cache.sizeDual();
            if (warm && previous.size() != size)
                throw std::runtime_error("Warm-start dimension changed or previous solution missing");
            if (warm && previousScaling != problem.scaling())
                throw std::runtime_error("Warm-start coordinates changed: preconditioning must remain fixed");
            checked(cudaDeviceSynchronize());
            const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();

            checked(cudaEventRecord(start));
            const auto wallStart = Clock::now();
            const int status = cache.runSpock(state, warm ? &previous : nullptr);
            checked(cudaDeviceSynchronize());
            checked(cudaEventRecord(stop));
            checked(cudaEventSynchronize(stop));
            const double wallSeconds = std::chrono::duration<double>(Clock::now() - wallStart).count();
            float milliseconds = 0;
            checked(cudaEventElapsedTime(&milliseconds, start, stop));

            std::vector<double> rootInput;
            if (status == converged) {
                const auto inputs = cache.inputs();
                rootInput.assign(inputs.begin(), inputs.begin() + tree.numInputs());
                previous.resize(size);
                cache.solution().download(previous);
                previousScaling = problem.scaling();
                for (double value : rootInput)
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite control action");
            }
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> out(buffer);
            out.StartObject();
            out.Key("converged"); out.Bool(status == converged);
            out.Key("status"); out.Int(status);
            out.Key("iterations"); out.Uint64(cache.solveIter());
            out.Key("gpu_time_s"); out.Double(milliseconds / 1000.0);
            out.Key("solver_wall_s"); out.Double(wallSeconds);
            out.Key("setup_wall_s"); out.Double(setupSeconds);
            out.Key("warm_start"); out.Bool(warm);
            out.Key("primal_size"); out.Uint64(cache.sizePrimal());
            out.Key("dual_size"); out.Uint64(cache.sizeDual());
            out.Key("control"); out.StartArray();
            for (double value : rootInput) out.Double(value);
            out.EndArray();
            out.EndObject();
            std::cout << buffer.GetString() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "MPC worker failed: " << e.what() << std::endl;
            return 1;
        }
    }
    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
    return 0;
}
