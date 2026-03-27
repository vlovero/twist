#include "H5Cpp.h"
#include "cli/colors.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "magic_enum.hpp"
#include "serialize.h"
#include "shared.h"

#include <string>
#include <string_view>
#include <vector>

std::vector<std::string_view> split(const std::string &text, const std::string &delimeter)
{
    std::vector<std::string_view> tokens;
    size_t start = 0;
    size_t pos;
    while ((pos = text.find(delimeter, start)) != std::string::npos) {
        tokens.emplace_back(&text[start], pos - start);
        start = pos + delimeter.size();
    }
    tokens.emplace_back(&text[start], text.size() - start);
    return tokens;
}

int parse_part(std::string_view sv, int default_val)
{
    return sv.empty() ? default_val : std::stoi(std::string{ sv });
}

std::vector<int> parse_indices_and_slices(const std::vector<std::string> &inputs, const int high)
{
    std::vector<int> all_indices;
    int i, start, stop, step, nparts;

    for (const auto &input : inputs) {
        std::vector<std::string_view> split_input{ split(input, ":") };
        nparts = split_input.size();

        stop = high;
        step = 1;

        if ((nparts == 0) || (nparts > 3)) {
            throw std::runtime_error(fmt::format("invalid slice entry: '{}'", input));
        }
        if (nparts == 1) {
            all_indices.emplace_back(std::stoi(input));
            continue;
        }

        try {
            start = parse_part(split_input[0], 0);
            if (nparts == 1) {
                stop = high;
            }
            else {
                stop = parse_part(split_input[1], high);
                if (nparts == 3) {
                    step = parse_part(split_input[2], 1);
                }
            }
        }
        catch (...) {
            throw std::runtime_error(fmt::format("invalid slice entry: '{}'", input));
        }

        start = ((start % high) + high) % high;
        stop = ((stop % (high + 1)) + (high + 1)) % (high + 1);
        for (i = start; i < stop; i += step) {
            all_indices.emplace_back(i);
        }
    }
    return all_indices;
}

std::pair<double, size_t> aargmin_and_shift(RP(double) alphar, RP(double) alphai, RP(double) beta, const size_t N)
{
    auto _hypot = [](double x, double y) -> double {
        const double m = std::max(std::abs(x), std::abs(y));
        x /= m;
        y /= m;
        return m * std::sqrt(x * x + y * y);
    };

    size_t i;
    size_t imin = -1;
    double dmin = std::numeric_limits<double>::infinity();
    double val;
    const double cutoff = N * std::numeric_limits<double>::epsilon();

    // find where smallest eigenvalue is
    for (i = 0; i < N; i++) {
        if (std::abs(beta[i]) < cutoff) {
            continue;
        }
        val = _hypot(alphar[i] / beta[i], alphai[i] / beta[i]);
        if (val < dmin) {
            imin = i;
            dmin = val;
        }
    }
    // shift all eigenvalues to center smallest at the origin
    for (i = 0; i < N; i++) {
        if (std::abs(beta[i]) < cutoff) {
            continue;
        }
        if (i == imin) {
            alphar[i] = 0;
            alphai[i] = 0;
            continue;
        }
        alphar[i] -= beta[i] * (alphar[imin] / beta[imin]);
        alphai[i] -= beta[i] * (alphai[imin] / beta[imin]);
    }
    return { dmin, imin };
}

void print_spectrum_info(const H5::H5File &file, const std::vector<int> &indices, const int ngroups, const std::optional<std::vector<TWIST::SolutionTypes>> &stype_filter)
{
    size_t dim;
    size_t i;
    uint64_t stype;
    double cutoff;
    double *alphar = nullptr, *alphai = nullptr, *beta = nullptr;
    std::vector<std::string> gz;
    for (const int index : indices) {
        if (index < 0 || (ngroups <= index)) {
            continue;
        }
        // load data from h5 file
        H5::Group group = file.openGroup(fmt::format("{}", index));
        deserialize(group, "stype", stype);
        if (stype_filter) {
            const auto &filters = stype_filter.value();
            if (std::find(filters.begin(), filters.end(), stype) == filters.end()) {
                continue;
            }
        }
        group = file.openGroup(fmt::format("{}/spectrum", index));
        deserialize(group, "alphar", alphar, dim);
        deserialize(group, "alphai", alphai, dim);
        deserialize(group, "beta", beta, dim);

        cutoff = std::numeric_limits<double>::epsilon() * dim;
        auto [dmin, imin] = aargmin_and_shift(alphar, alphai, beta, dim);
        gz.clear();
        for (i = 0; i < (size_t)dim; i++) {
            if (std::abs(beta[i]) < cutoff) {
                continue;
            }
            if (alphar[i] > 0) {
                gz.emplace_back(fmt::format("{}{:+g}i", alphar[i] / beta[i], alphai[i] / beta[i]));
            }
        }
        fmt::println(COLOR_BLUE "[{:4d}]" COLOR_RESET COLOR_GREEN, index);
        fmt::println("{:<25s}: {}", "size", dim);
        fmt::println("{:<25s}: {}", "type", magic_enum::enum_name((TWIST::SolutionTypes)stype));
        fmt::println("{:<25s}: {}", "min|\u03BB|", dmin);
        fmt::println("{:<25s}: [{}]", "Re(\u03BB) > 0 (shifted)", fmt::join(gz, ", "));
        fmt::println("{:<25s}: {}", "# Re(\u03BB) > 0 (shifted)", gz.size());
        puts(COLOR_RESET);
    }
    free(alphar);
    free(alphai);
    free(beta);
}

void print_parameter_values(const H5::H5File &file, const std::vector<int> &indices, const int ngroups, const std::vector<std::string> &param_names, const std::vector<int> &param_indices, const std::optional<std::vector<TWIST::SolutionTypes>> &stype_filter)
{
    size_t dim;
    uint64_t stype;
    double *p = nullptr;
    const size_t nspace = std::max<size_t>(6, std::max_element(param_names.begin(), param_names.end(), [](const std::string &a, const std::string &b) { return a.size() < b.size(); })->size());
    std::vector<std::string> gz;
    for (const int index : indices) {
        if (index < 0 || (ngroups <= index)) {
            continue;
        }
        // load data from h5 file
        H5::Group group = file.openGroup(fmt::format("{}", index));
        deserialize(group, "stype", stype);
        if (stype_filter) {
            const auto &filters = stype_filter.value();
            if (std::find(filters.begin(), filters.end(), stype) == filters.end()) {
                continue;
            }
        }
        deserialize(group, "p", p, dim);

        fmt::println(COLOR_BLUE "[{:4d}]" COLOR_RESET COLOR_GREEN, index);
        fmt::println("{:>{}s} : {}", "type", nspace, magic_enum::enum_name((TWIST::SolutionTypes)stype));
        for (size_t j = 0; j < param_names.size(); j++) {
            fmt::println("{:>{}s} : {}", param_names[j], nspace, p[param_indices[j]]);
        }
        puts(COLOR_RESET);
    }
    free(p);
}
