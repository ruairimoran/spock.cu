#include <anderson.cuh>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template<typename T>
void checkFixture(const rapidjson::Document &input, bool qrUpdates) {
    const auto &values = input["residual"];
    const size_t rows = values.Size();
    const size_t memory = input["Y"][0].Size();
    std::vector<T> c(rows * memory), y(rows * memory), residual(rows), direction(rows);
    for (size_t row = 0; row < rows; ++row) {
        residual[row] = T(values[row].GetDouble());
        for (size_t col = 0; col < memory; ++col) {
            c[col * rows + row] = T(input["C"][row][col].GetDouble());
            y[col * rows + row] = T(input["Y"][row][col].GetDouble());
        }
    }
    if (input.HasMember("inject_nonfinite")) {
        const auto &injection = input["inject_nonfinite"];
        auto &target = std::string(injection["target"].GetString()) == "C" ? c : y;
        const size_t row = injection["row"].GetUint64(), col = injection["column"].GetUint64();
        if (row >= rows || col >= memory) throw std::invalid_argument("Invalid nonfinite fixture index");
        target[col * rows + row] = std::string(injection["value"].GetString()) == "nan"
            ? std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::infinity();
    }
    AndersonOptions options;
    options.enabled = true;
    options.qr.incremental = qrUpdates;
    const auto &settings = input["options"];
    options.regularization = settings["regularization"].GetDouble();
    options.rankTolerance = settings["rank_tolerance"].GetDouble();
    options.maxCoefficientNorm = settings["max_coefficient_norm"].GetDouble();
    options.maxDirectionRatio = settings["max_direction_ratio"].GetDouble();
    options.validate();
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> out(buffer);
    out.StartObject();
    out.Key("name"); out.String(input["name"].GetString());
    out.Key("regularization"); out.Double(options.regularization);
    if (memory == 0) {
        out.Key("status"); out.String("empty_history");
        out.Key("rank"); out.Uint64(0);
        out.Key("scale"); out.Double(0);
        out.Key("coefficient_norm"); out.Double(0);
        out.Key("accepted_indices"); out.StartArray(); out.EndArray();
        out.Key("gamma"); out.StartArray(); out.EndArray();
        out.Key("direction"); out.StartArray();
        for (T value : residual) out.Double(-value);
        out.EndArray();
    } else {
        DTensor<T> dC(c, rows, memory), dY(y, rows, memory), dResidual(residual, rows);
        DTensor<T> qr(rows, memory), rhs(rows), gamma(memory), dDirection(rows);
        RegularisedAnderson<T> fit(dC, dY, qr, rhs, gamma, options);
        const size_t allocated = Session::getInstance().totalAllocatedBytes();
        const auto &info = fit.compute(dResidual, dDirection);
        dDirection.download(direction);
        if (Session::getInstance().totalAllocatedBytes() != allocated)
            throw std::runtime_error("Anderson fit changed its device allocation");
        out.Key("status"); out.String(andersonStatusName(info.status));
        out.Key("rank"); out.Uint64(info.selectedColumns.size());
        out.Key("scale"); out.Double(info.scale);
        out.Key("coefficient_norm"); out.Double(info.coefficientNorm);
        out.Key("accepted_indices"); out.StartArray();
        for (size_t index : info.selectedColumns) out.Uint64(index);
        out.EndArray();
        out.Key("gamma"); out.StartArray();
        for (double value : info.coefficients) out.Double(value);
        out.EndArray();
        out.Key("direction"); out.StartArray();
        for (T value : direction) out.Double(value);
        out.EndArray();
    }
    out.EndObject();
    std::cout << buffer.GetString() << std::endl;
}

int main(int argc, char **argv) {
    bool single = false, qrUpdates = false;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--float") single = true;
        else if (flag == "--qr-updates") qrUpdates = true;
        else { std::cerr << "Unknown option: " << flag << '\n'; return 2; }
    }
    try {
        std::string line;
        while (std::getline(std::cin, line)) {
            rapidjson::Document input;
            input.Parse(line.c_str());
            if (input.HasParseError() || !input.HasMember("residual") || input["residual"].Empty())
                throw std::invalid_argument("Invalid Anderson fixture");
            if (single) checkFixture<float>(input, qrUpdates);
            else checkFixture<double>(input, qrUpdates);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
