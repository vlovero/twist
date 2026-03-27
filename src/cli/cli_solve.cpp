#include "cli/cli_solve.h"
#include "H5File.h"
#include "cli/load.h"
#include "collocator.h"
#include "fmt/base.h"
#include "fmt/color.h"
#include "indicators/progress_bar.hpp"
#include "indicators/setting.hpp"
#include "libloader.h"
#include "preprocess/drxd.h"
#include "serialize.h"
#include "tinyexpr.h"
#include "tools/helpers.h"
#include <cstdio>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <unistd.h>

void filter_eigenvalues(const std::string &filter, std::vector<int> &indices, const std::vector<double> alphar, const std::vector<double> &alphai, const std::vector<double> &beta)
{
    double real, imag, result;
    te_parser tep;

    tep.set_variables_and_functions({ { "real", &real }, { "imag", &imag } });
    real = 0.69;
    imag = 0.420;

    result = tep.evaluate(filter);

    if (!tep.success()) {
        throw std::runtime_error(fmt::format("Parse error at {}\n{}\n{:^{}s}^", std::to_string(tep.get_last_error_position()), filter, " ", tep.get_last_error_position()));
    }

    for (size_t i = 0; i < alphar.size(); i++) {
        if (std::abs(beta[i]) <= (alphar.size() * pow(0.5, 52))) {
            continue;
        }
        real = alphar[i] / beta[i];
        imag = alphai[i] / beta[i];

        result = tep.evaluate();
        if (result != 0.0) {
            indices.emplace_back(i);
        }
    }
}

int SolveArgs::run()
{
    if (!solve_quietly) {
        // this->print();
        puts("");
    }

    if (nthreads) {
        if (nthreads <= 0) {
            throw std::runtime_error("number of threads must be positive");
        }
        omp_set_num_threads(nthreads.value());
    }

    size_t node, nt, N;
    TWIST::Collocator collocator;
    double *t = NULL, *y = NULL, *yd = NULL;
    std::vector<double> alphar, alphai, beta;
    std::vector<double> tc, td;
    std::vector<std::string> var_names;
    void *lib = nullptr;

    if (!std::filesystem::exists(model)) {
        throw std::runtime_error(fmt::format("file {} does not exist", model));
    }

    if (from_data.size() != 0) {
        if (from_data.size() != 2) {
            puts("--from-data keyword must have the form '--from-data <path> <index>'");
            return 1;
        }
        verify_hdf5_file(from_data[0]);
        H5::H5File h5data(from_data[0], H5F_ACC_RDONLY);
        int solution_index = atoi(from_data[1].c_str());
        int nsolutions = get_number_of_solutions(h5data);
        solution_index = ((solution_index % nsolutions) + nsolutions) % nsolutions;
        collocator = load_collocator_from_h5data_and_index(h5data, solution_index, ncol, &lib);
        collocator.setLibHandle(lib);
        h5data.close();
    }
    else {
        collocator = load_collocator_from_spec_and_init(ncol, model, parameter_set, &lib);
        collocator.setLibHandle(lib);
    }
    node = collocator.getNode();

    if (solve_init) {
        const bool failed = collocator.solveWithAdaptation(solve_nadapt, solve_geps, solve_min_nodes, solve_max_nodes, solve_tol, solve_min_damp, solve_max_iter, !solve_quietly);

#if ENABLE_MAKE_EQUISPACED
        if (const char *env = std::getenv("MAKE_EQUISPACED"); env && (*env != '0')) {
            TWIST::Collocator backup(collocator);
            const size_t n = collocator.getNNodes();
            double *t = collocator.t();
            double *h = collocator.h();
            double *y = collocator.y();
            double *yprev = collocator.yprev();
            double *yp = collocator.yp();
            for (size_t i = 0; i < n; i++) {
                t[i] = ((double)i) / (n - 1);
                if (i < (n - 1)) {
                    h[i] = 1.0 / (n - 1);
                }
            }
            auto tmp = collocator.plottablePoints();
            backup.denseOutput(tmp.data(), tmp.size(), y);
            backup.denseOutput(tmp.data(), tmp.size(), yprev);
            backup.denseDerivativeOutput(tmp.data(), tmp.size(), yp);

            collocator.solveWithAdaptation(0, solve_geps, solve_min_nodes, solve_max_nodes, solve_tol, solve_min_damp, solve_max_iter, !solve_quietly);
        }
#endif

        if (replace_existing_initial_guess && (!failed)) {
            if (from_data.size()) {
                throw std::runtime_error("replacing initial wave with existing data is probably a mistake");
            }

            replace_existing_starting_solution(collocator, model, parameter_set);
        }

        if (add_bumps > 0) {
            collocator.copyCollocatorDataIntoSelf(extend_to_multiwave_solution(collocator, add_bumps));
            collocator.setLibHandle(lib);
            collocator.solveWithAdaptation(solve_nadapt, solve_geps, (add_bumps + 1) * solve_min_nodes, solve_max_nodes, solve_tol, solve_min_damp, solve_max_iter, !solve_quietly);
        }
        if (!solve_quietly) {
            fmt::println("{}", fmt::format(fmt::fg(fmt::color::cyan), "Finished with {} unknowns", ((collocator.getNNodes() - 1) * (collocator.getNStages() + 1) + 1) * node));
        }
#if ENABLE_DUMP_COL_MAT
        if (const char *env = std::getenv("DUMP_COL_MAT"); env && (*env != '0')) {
            sparse::COOMatrix &coo_tmp{ collocator.getBaseJacobianCOO() };
            FILE *file = fopen(".dev/coo_tmp.dat", "w");
            for (int64_t i = 0; i < coo_tmp.nnz; i++) {
                fmt::println(file, "{:15d},{:15d},{: .15e}", coo_tmp.irow[i], coo_tmp.icol[i], coo_tmp.data[i]);
            }
            fclose(file);
        }
#endif
#if ENABLE_DUMP_PENCIL
        if (const char *env = std::getenv("DUMP_PENCIL"); env && (*env != '0')) {
            sparse::RealCSCMatrix A, B;
            H5::H5File file(".dev/csc_tmp.h5", H5F_ACC_TRUNC);
            H5::Group group = file.createGroup("A");

            collocator.setupSparseABPencilForSpectrum(A, B, 0.0);

            serialize<int64_t, 1>(group, "Ap", A.Ap, { (hsize_t)(A.ncols + 1) });
            serialize<int64_t, 1>(group, "Ai", A.Ai, { (hsize_t)(A.nnz()) });
            serialize<double, 1>(group, "Ax", A.Ax, { (hsize_t)(A.nnz()) });

            group = file.createGroup("B");
            serialize<int64_t, 1>(group, "Ap", B.Ap, { (hsize_t)(B.ncols + 1) });
            serialize<int64_t, 1>(group, "Ai", B.Ai, { (hsize_t)(B.nnz()) });
            serialize<double, 1>(group, "Ax", B.Ax, { (hsize_t)(B.nnz()) });
            group.close();
            file.close();
        }
#endif
    }
    nt = collocator.getNNodes();
    node = collocator.getNode();
    N = collocator.NUnknowns();
    t = (double *)collocator.t();
    y = collocator.y();

    if (solve_init && plot_after_solve_init) {
        tc = collocator.plottablePoints();
        if (plot_dense) {
            td = linspace(0, 1, 10000);
            yd = (double *)malloc(10000 * node * sizeof(double));
            collocator.denseOutput(td.data(), td.size(), yd);
            // collocator.denseDerivativeOutput(td.data(), td.size(), yd);
        }

        // get variable names for plotting
        auto contains = [](int i, const diffusion_t &diffusion) -> bool {
            for (const auto &[loc, _] : diffusion) {
                if (loc == i) {
                    return true;
                }
            }
            return false;
        };
        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.load(model);
        auto system = entry_to_map("system", doc);
        auto diffusion = load_diffusion(doc);
        int i = 0;
        for (const auto &[name, _] : system) {
            var_names.emplace_back(name);
            if (contains(i, diffusion)) {
                var_names.emplace_back(fmt::format(R"(\$\\frac{{\\partial}}{{\\partial\\xi}}\${})", name));
            }
            i++;
        }
    }

    if (spectrum) {
        if (essential_spectrum) {
            std::vector<double> alphar_g, alphai_g, beta_g;
            using namespace indicators;
            ProgressBar pbar(option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[essential spectrum] " }, option::ShowPercentage(true));
            std::vector<std::complex<double>> tmp;
            collocator.spectrum(alphar, alphai, beta, strategy, false, NULL);

            std::vector<int> indices;
            int count;
            indices.reserve(alphar.size());
            if (essential_spectrum_filter.size() != 0) {
                filter_eigenvalues(essential_spectrum_filter, indices, alphar, alphai, beta);
                omp_set_max_active_levels(1);
                count = 0;
                pbar.set_option(option::MaxProgress{ indices.size() });
                const int ndigits = std::ceil(std::log10(indices.size()));

#pragma omp parallel for private(tmp) schedule(dynamic, 1)
                for (const int index : indices) {
                    collocator.generateEssentialSpectrumBranch({ alphar[index] / beta[index], alphai[index] / beta[index] }, tmp);
#pragma omp critical
                    {
                        count += 1;
                        pbar.set_option(option::PostfixText{ fmt::format("{0:{2}d}/{1:{2}d}", count, indices.size(), ndigits) });
                        pbar.tick();
                        for (const auto &value : tmp) {
                            alphar_g.emplace_back(value.real());
                            alphai_g.emplace_back(value.imag());
                            beta_g.emplace_back(1.0);
                        }
                    }
                }
                alphar.insert(alphar.end(), alphar_g.begin(), alphar_g.end());
                alphai.insert(alphai.end(), alphai_g.begin(), alphai_g.end());
                beta.insert(beta.end(), beta_g.begin(), beta_g.end());
            }
        }
        else {
            TWIST::time_code("spectrum", [&]() { collocator.spectrum(alphar, alphai, beta, strategy, false, NULL); });
        }
    }
    if (subspace) {
        collocator.generateSubspace(subspace_size, alphar, alphai, subspace_sigma);
        for (const auto &_ : alphar) {
            (void)_;
            beta.emplace_back(1.0);
        }
    }

    if (simulate) {
        if (simulate_nloops < 0.0) {
            puts("--simulate-nloops keyword must be non-negative");
            return 1;
        }
        ptrdiff_t i, j, k;
        func_t func_ode = (func_t)dlsym(lib, "func_ode");
        func_t fjac_ode = (func_t)dlsym(lib, "fjac_ode");
        const double wave_speed = collocator.waveSpeed();
        const double L = collocator.spatialPeriod();
        const double *p = collocator.p();
        auto diffusion = collocator.getDiffusion();

        const size_t nx = simulate_nspace;
        const size_t nt = simulate_nsample;
        std::vector<double> x = linspace(0, 1, nx);
        std::vector<double> t = linspace(0, simulate_nloops * (L / std::abs(wave_speed)), nt);
        double *yeval = (double *)malloc(nx * node * nt * sizeof(double));
        double *ynov = (double *)malloc(nx * (node - diffusion.size()) * sizeof(double));
        double *ywithv = (double *)malloc(node * sizeof(double));

        for (i = 0; i < (ptrdiff_t)nx; i++) {
            collocator.denseOutput(&x[i], 1, ywithv);
            x[i] *= L;

            for (j = 0, k = 0; j < (ptrdiff_t)node; j++, k++) {
                ynov[i * (node - diffusion.size()) + k] = ywithv[j];
                for (const auto &[loc, _] : diffusion) {
                    if (loc == k) {
                        j++;
                        break;
                    }
                }
            }
            ynov[i * (node - diffusion.size())] *= 1 + simulate_vscale;
        }

        std::tuple<double *, double *, long, double> sim_result;
        switch (simulation_method) {
        case SimulationMethods::DIRK5:
            sim_result = rxdiff_simulate<Tables::DIRK5>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, 1e-6, 2e-3, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
            break;
        case SimulationMethods::DIRK865:
            sim_result = rxdiff_simulate<Tables::DIRK865>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, 1e-6, 2e-3, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
            break;
        case SimulationMethods::DIRK965:
            sim_result = rxdiff_simulate<Tables::DIRK965>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, 1e-6, 2e-3, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
            break;
        case SimulationMethods::DIRK1175:
            sim_result = rxdiff_simulate<Tables::DIRK1175>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, 1e-6, 2e-3, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
            break;
        case SimulationMethods::IIF2:
            sim_result = rxdiff_simulate_iif2(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], -1, 1e-6, 2e-4, false, nt, t.data(), nx, x.data(), yeval, simulate_animate);
            break;
        }

        if (1) {
            int code;
            std::vector<std::string> cmd;
            const std::string &file_path = simulate_save_path ? simulate_save_path.value() : ".cache/solve_res.h5";
            H5::H5File res_file(file_path, H5F_ACC_TRUNC);
            H5::Group group = res_file.createGroup("data");
            serialize<double, 1>(group, "x", x.data(), { (hsize_t)nx });
            serialize<double, 3>(group, "y", yeval, { (hsize_t)nt, (hsize_t)nx, (hsize_t)(node - diffusion.size()) });
            group.close();
            res_file.close();
            if (simulate_save_path) {
                cmd.emplace_back(file_path);
            }
            code = run_python_script(python::interim::get_script(), cmd);
            if (code) {
                return 0;
            }
        }
        free(std::get<0>(sim_result));
        free(std::get<1>(sim_result));
        free(yeval);
        free(ynov);
        free(ywithv);
    }

    solve_plot(plot_after_solve_init, plot_dense, spectrum | subspace, N, node, ncol, nt, t, tc.data(), y, td.data(), yd, alphar, alphai, beta, var_names);

    return 0;
}

void SolveArgs::welcome()
{
    fmt::println("Refine traveling wave, compute spectrum, simulate");
}

// [N] MAKE_EQUISPACED: after last solve finished, resolve on uniformt grid using same num of grid points that solver finished with
// [N] DUMP_COL_MAT: dump colmat into plain text file .dev/coo_tmp.dat
// [N] DUMP_PENCIL: dump matrix pencil into hdf5 file, .dev/csc_tmp.h5
// [Y] NO_USE_COL_MAT: do not use col_mat (falls back to UMFPack)
// [Y] NO_USE_TREE_MAT: do not use tree mat (falls back to col_mat)
// [N] SHOW_MONITOR_FUNCTION: show the monitor function every time the grid is adapted