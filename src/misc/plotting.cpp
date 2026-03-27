#include "python/plotting.h"
#include "cli/load.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "serialize.h"

#include <filesystem>


void solve_plot(bool plot_after_solve_init, bool plot_dense, bool spectrum, int N, int node, int ncol, int nnodes, double *t, double *tc, double *y, double *td, double *yd, const std::vector<double> &alphar, const std::vector<double> &alphai, const std::vector<double> &beta, const std::vector<std::string> &var_names)
{
    if (!(plot_after_solve_init || plot_dense || spectrum)) {
        return;
    }

    // dump all data
    {
        H5::H5File h5_file(".cache/solve_res.h5", H5F_ACC_TRUNC);
        H5::Group group = h5_file.createGroup("data");
        if (plot_after_solve_init) {
            assert(t);
            assert(tc);
            assert(y);
            serialize(group, "nnodes", nnodes);
            serialize(group, "node", node);
            serialize(group, "ncol", ncol);
            serialize<double, 1>(group, "t", t, { (hsize_t)nnodes });
            serialize<double, 1>(group, "tc", tc, { (hsize_t)((nnodes - 1) * (ncol + 1) + 1) });
            serialize<double, 1>(group, "y", y, { (hsize_t)N });
        }
        if (plot_dense) {
            assert(td);
            assert(yd);
            serialize<double, 1>(group, "td", td, { (hsize_t)10000 });
            serialize<double, 1>(group, "yd", yd, { (hsize_t)(10000 * node) });
        }
        if (spectrum) {
            // assert(alphar);
            // assert(alphai);
            // assert(beta);
            serialize<double, 1>(group, "alphar", alphar.data(), { alphar.size() });
            serialize<double, 1>(group, "alphai", alphai.data(), { alphai.size() });
            serialize<double, 1>(group, "beta", beta.data(), { beta.size() });
        }
    }

    std::vector<std::string> flags = {};
    if (plot_after_solve_init) {
        flags.emplace_back("--solve-init");
        flags.emplace_back(fmt::format("--names {}", fmt::join(var_names, " ")));
    }
    if (plot_dense) {
        flags.emplace_back("--dense");
    }
    if (spectrum) {
        flags.emplace_back("--spectrum");
    }
    run_python_script(python::plot::get_script(), flags, false);
    // std::string cmd = fmt::format("{}", fmt::join(flags, " "));
    // system(cmd.c_str());
}

int run_python_script(const std::string_view &script, const std::vector<std::string> &args, const bool escape)
{
    int code;
    FILE *process;
    std::string cmd;

    // ensure python path still exists
    if (std::filesystem::exists(PYTHON_EXEC)) {
        cmd = fmt::format("{} - {}", PYTHON_EXEC, fmt::join(args, " "));
    }
    else if (const char *path = std::getenv("TWIST_PYTHON_PATH"); path && std::filesystem::exists(path)) {
        // use a different one if specified
        cmd = fmt::format("{} - {}", path, fmt::join(args, " "));
    }
    else {
        throw std::runtime_error(fmt::format("TWIST was compiled with {} as the python path which no longer exists. Set the 'TWIST_PYTHON_PATH' environment variable to a valid path to python or rebuild TWIST.", PYTHON_EXEC));
    }

    if (escape) {
        replace(cmd, "\\", "\\\\");
    }

    process = popen(cmd.c_str(), "w");
    if (process == NULL) {
        return -1;
    }

    fwrite(script.data(), 1, script.size(), process);

    code = pclose(process);

    return code;
}