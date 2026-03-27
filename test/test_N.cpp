#include "argparse/argparse.hpp"
#include "collocator.h"
#include "numpy_like.h"
#include "serialize.h"
#include "tools/helpers.h"
#include <numeric>

extern void solve_vanderT(const ptrdiff_t n, const double *alpha, const double *b, double *x);

std::pair<double, double> get_coeffs(ptrdiff_t order)
{
    const ptrdiff_t n = order + 1;
    double alpha[n], b[n], coeffs[n];
    double C1, C2;

    // setup for coeffs
    std::iota(alpha, alpha + n, -order / 2);

    // get coeffs (1st derivative)
    memset(b, 0, n * sizeof(double));
    b[1] = 1;
    solve_vanderT(n, alpha, b, coeffs);
    C1 = std::accumulate(alpha, alpha + n, 0.0, [&](double acc, double it) {
        ptrdiff_t i = it;
        ptrdiff_t j = i + order / 2;
        return acc + coeffs[j] * std::pow(i, order + 1) / std::tgamma(order + 2);
    });

    // get coeffs (2nd derivative)
    b[1] = 0;
    b[2] = 2;
    solve_vanderT(n, alpha, b, coeffs);
    C2 = std::accumulate(alpha, alpha + n, 0.0, [&](double acc, double it) {
        ptrdiff_t i = it;
        ptrdiff_t j = i + order / 2;
        return acc + coeffs[j] * std::pow(i, order + 2) / std::tgamma(order + 3);
    });

    return { std::abs(C1), std::abs(C2) };
}

std::tuple<size_t, double, double> get_N(H5::H5File &data, const int index, const double tol, const int order)
{
    double norm_1st, norm_2nd, Dmax, scale, hmin;
    int Nmin;
    int pmask[2];

    auto t = linspace(0, 1, 10001);
    TWIST::Collocator collocator(data, index);
    auto group = data.openGroup("basic_info");
    // deserialize(data, "basic_info/pmask", )
    const auto &diffusion = collocator.getDiffusion();
    std::vector<double> y(t.size() * collocator.getNode());
    double *p = collocator.p();
    deserialize<int, 2>(group, "pmask", pmask);


    Dmax = (*std::max_element(diffusion.begin(), diffusion.end(), [](const auto a, const auto b) { return a.second < b.second; })).second;

    if (order == -1) {
        const int eta = 2;
        collocator.denseNuthDerivativeOutput(eta, t.data(), t.size(), y.data());
        // norm_1st = norminf(y.data(), y.size());
        norm_2nd = 0;
        for (size_t i = 0; i < (t.size() - 1); i++) {
            norm_2nd += (t[i + 1] - t[i]) * inner(&y[i * collocator.getNode()], &y[i * collocator.getNode()], collocator.getNode());
        }
        // cn <= B / (L * N^s) <= tol -> B / (L * tol) <= N^s
        hmin = collocator.spatialPeriod() / (std::pow(norm_2nd / (collocator.spatialPeriod() * tol), 1.0 / eta) - 1);
        // cn <= (1/(L n^s) Int[y^(s), {xi, 0, L}])
    }
    if ((order & 1) == 0) {
        if (order >= (int)collocator.getNStages()) {
            throw std::runtime_error(fmt::format("Cannot estimate points needed for order {} (collocation was done using order {} polynomials)", order, collocator.getNStages()));
        }
        auto &&[C1, C2] = get_coeffs(order);

        // compute error terms
        collocator.denseNuthDerivativeOutput(order + 1, t.data(), t.size(), y.data());
        norm_1st = norminf(y.data(), y.size());
        collocator.denseNuthDerivativeOutput(order + 2, t.data(), t.size(), y.data());
        norm_2nd = norminf(y.data(), y.size());
        scale = Dmax * norm_2nd * C2 + std::abs(collocator.waveSpeed()) * norm_1st * C1;
        hmin = std::pow(tol / scale, 1.0 / order);
    }
    else {
        throw std::runtime_error(fmt::format("order {} not implemented", order));
    }


    Nmin = std::ceil(collocator.spatialPeriod() / hmin + 1);
    return { Nmin, p[pmask[0]], p[pmask[1]] };
}

struct TestNArgs : public argparse::Args
{
    std::string &name = arg("name", "name of model");
    std::string &pset = arg("pset", "parameter set");
    std::string &param = arg("param", "which param");
    std::string &tag = arg("tag", "which tag");
    double &tol = kwarg("tol", "tolerance").set_default(1e-8);
    int &order = kwarg("order", "order of FDM").set_default(2);

    int run() override
    {
        const std::string path = fmt::format(".cache/continuation_data/{}-{}-{}-{}.h5", name, pset, param, tag);
        const std::string output = fmt::format("test/data/N-needed-{}-{}-{}-{}.csv", name, pset, param, tag);
        verify_hdf5_file(path);
        H5::H5File data(path, H5F_ACC_RDONLY);
        const int nsolutions = get_number_of_solutions(data);

        std::filesystem::create_directories("test/data");
        FILE *file = fopen(output.c_str(), "w");

        for (int i = 0; i < nsolutions; i++) {
            const auto [N, x, y] = get_N(data, i, tol, order);
            fmt::println(file, "{:.16e},{:.16e},{:.16e}", (double)N, x, y);
        }

        return 0;
    }
};

int main(int argc, char **argv)
{
    auto args = argparse::parse<TestNArgs>(argc, argv, true);
    args.run();

    return 0;
}