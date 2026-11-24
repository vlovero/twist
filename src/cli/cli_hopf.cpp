#include "cli/cli_hopf.h"
#include "bif_points/hopf_point.h"
#include "cli/colors.h"
#include "cli/load.h"
#include "fmt/core.h"
#include "modulating/hopf_branch_solver.h"
#include "modulating/hopf_continuation.h"
#include "serialize.h"
#include "tools/helpers.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <omp.h>
#include <valarray>

namespace
{
    std::string infer_parameter_name(const std::vector<std::string> &params, int parnum, size_t np)
    {
        if (parnum == (int)(np - 1)) {
            return "sps";
        }
        const int idx = parnum - 1;
        if (idx >= 0 && idx < (int)params.size()) {
            return params[idx];
        }
        return fmt::format("param{}", parnum);
    }

    int resolve_parameter_index(const std::vector<std::string> &params, const std::string &param_name)
    {
        if (param_name == "sps" || param_name == "spatial_period_scale") {
            return (int)params.size() + 1;
        }
        auto it = std::find(params.begin(), params.end(), param_name);
        if (it == params.end()) {
            throw std::runtime_error(fmt::format("'{}' is not a valid parameter name", param_name));
        }
        return 1 + (int)(it - params.begin());
    }

    bool compute_hopf_eigenpair_fallback(TWIST::Collocator &col, double &out_omega, std::vector<std::complex<double>> &out_psi, H5::H5File *h5_file, int sol_idx)
    {
        std::vector<double> alphar, alphai, beta;
        if (h5_file && h5_file->nameExists(fmt::format("{}/spectrum", sol_idx))) {
            H5::Group sp = h5_file->openGroup(fmt::format("{}/spectrum", sol_idx));
            double *ar = nullptr, *ai = nullptr, *bt = nullptr;
            size_t len = 0;
            deserialize(sp, "alphar", ar, len);
            deserialize(sp, "alphai", ai, len);
            deserialize(sp, "beta", bt, len);
            alphar.assign(ar, ar + len);
            alphai.assign(ai, ai + len);
            beta.assign(bt, bt + len);
            free(ar);
            free(ai);
            free(bt);
        }
        else {
            col.spectrum(alphar, alphai, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, nullptr);
        }

        // Find complex pair closest to imaginary axis (min |Re(lambda)| with Im(lambda) > 1e-6)
        double best_re = std::numeric_limits<double>::infinity();
        std::complex<double> best_k(0.0, 0.0);
        const size_t N = beta.size();
        for (size_t i = 0; i < N; i++) {
            if (std::abs(beta[i]) < N * std::numeric_limits<double>::epsilon()) {
                continue;
            }
            const double re = alphar[i] / beta[i];
            const double im = alphai[i] / beta[i];
            if (im > 1e-6 && std::abs(re) < best_re) {
                best_re = std::abs(re);
                best_k = std::complex<double>(re, im);
            }
        }
        if (!std::isfinite(best_re) || best_k.imag() <= 1e-6) {
            return false;
        }

        const size_t node = col.getNode();
        double M[node * node], D[node];
        Hopf::setup_M_and_D(node, M, D, col.indicesToDelete(), col.getDiffusion(), col.spatialPeriod(), col.waveSpeed());
        DEV::ComplexMatrix A, B;
        DEV::ComplexBoundaryConditions bc, zbc;
        std::valarray<std::complex<double>> v;
        Hopf::get_initial_eigenvector(best_k, v, &col, A, B, &bc, &zbc, M, col.colA(), col.colB());
        Hopf::refine_eigenpair(best_k, v, &col, A, B, &bc, &zbc, M, col.colA(), col.colB());

        out_omega = std::abs(best_k.imag());
        out_psi.resize(v.size());
        const bool conj_v = (best_k.imag() < 0.0);
        for (size_t i = 0; i < v.size(); i++) {
            out_psi[i] = conj_v ? std::conj(v[i]) : v[i];
        }
        return true;
    }
} // namespace

int HopfArgs::run()
{
    if (nthreads) {
        if (nthreads.value() <= 0) {
            throw std::runtime_error("number of threads must be positive");
        }
        omp_set_num_threads(nthreads.value());
    }
    omp_set_max_active_levels(1);
    OpenBLASThreadContext __openblas_context(1);

    if (!std::filesystem::exists(model)) {
        throw std::runtime_error(fmt::format("file {} does not exist", model));
    }
    if (from_data.empty() && resume.empty()) {
        fmt::println(COLOR_RED "Error: must specify either --from-data <steady.h5> [idx] or --resume <hopf.h5> [idx]" COLOR_RESET);
        return 1;
    }
    if (!from_data.empty() && !resume.empty()) {
        fmt::println(COLOR_RED "Error: --from-data and --resume are mutually exclusive" COLOR_RESET);
        return 1;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    verified_json_load(model, doc, parser);
    const std::string name = std::string(get_json_string(doc, "name"));
    const std::vector<std::string> params = entry_to_list<std::string>("params", doc, false);

    TWIST::HopfBranchSolver solver;
    std::string param_name;
    void *lib = nullptr;

    if (!resume.empty()) {
        if (resume.size() > 2) {
            fmt::println("--resume expects <hopf.h5> [index]");
            return 1;
        }
        const std::string &resume_path = resume[0];
        verify_hdf5_file(resume_path);
        H5::H5File h5file(resume_path, H5F_ACC_RDWR);
        const int nsol = get_number_of_solutions(h5file);
        int idx = (resume.size() == 2) ? std::atoi(resume[1].c_str()) : (nsol - 1);
        idx = ((idx % nsol) + nsol) % nsol;

        H5::Group basic = h5file.openGroup("basic_info");
        int pmask_file[2] = { 0, 1 };
        size_t np_file = 0;
        deserialize(basic, "pmask", pmask_file);
        deserialize(basic, "np", np_file);
        param_name = parameter ? parameter.value() : infer_parameter_name(params, pmask_file[1], np_file);

        solver.loadFromH5(h5file, idx);
        solver.solveNewton(0.0, 1e-10, 1e-4, 10, verbose);
        solver.computeTangent();
        solver.acceptStep();
        // Set serialization to append onto a new tag (or continue in new file)
        fmt::println("Resumed modulating wave from '{}' index {} ({} = {:.6e}, T = {:.6f})",
            resume_path, idx, param_name, solver.continuationParam(), solver.period());
    }
    else {
        if (from_data.size() > 2) {
            fmt::println("--from-data expects <steady.h5> [index]");
            return 1;
        }
        const std::string &h5_path = from_data[0];
        verify_hdf5_file(h5_path);
        H5::H5File h5data(h5_path, H5F_ACC_RDONLY);
        const int nsol = get_number_of_solutions(h5data);

        int sol_idx = -1;
        if (from_data.size() == 2) {
            sol_idx = std::atoi(from_data[1].c_str());
            sol_idx = ((sol_idx % nsol) + nsol) % nsol;
        }
        else {
            // Auto-scan for SolutionTypes::torusPoint entries
            std::vector<int> hopf_indices;
            for (int k = 0; k < nsol; k++) {
                H5::Group g = h5data.openGroup(fmt::format("{}", k));
                TWIST::SolutionTypes st = TWIST::SolutionTypes::regular;
                deserialize<TWIST::SolutionTypes>(g, "stype", st);
                if (st == TWIST::SolutionTypes::torusPoint) {
                    hopf_indices.push_back(k);
                }
            }
            if (hopf_indices.empty()) {
                fmt::println(COLOR_RED "No Hopf bifurcation (torusPoint) found in '{}'." COLOR_RESET, h5_path);
                fmt::println("Run `twist postprocess {} -s -b` first to locate Hopf points.", h5_path);
                return 1;
            }
            if (hopf_indices.size() > 1) {
                H5::Group basic = h5data.openGroup("basic_info");
                int pmask_file[2] = { 0, 1 };
                deserialize(basic, "pmask", pmask_file);
                fmt::println("Multiple Hopf points found in '{}'. Re-run with `--from-data {} <index>`:", h5_path, h5_path);
                fmt::println("  {:>6s} | {:>14s} | {:>14s} | {:>14s}", "index", "parameter", "wave speed", "omega");
                for (int hi : hopf_indices) {
                    H5::Group g = h5data.openGroup(fmt::format("{}", hi));
                    double *p_arr = nullptr;
                    deserialize(g, "p", p_arr);
                    std::string omega_str = "-";
                    if (g.nameExists("hopf_info")) {
                        double om = 0.0;
                        H5::Group hg = g.openGroup("hopf_info");
                        deserialize(hg, "omega", om);
                        omega_str = fmt::format("{:.6e}", om);
                    }
                    fmt::println("  {:6d} | {:14.6e} | {:14.6f} | {:>14s}", hi, p_arr[pmask_file[1]], p_arr[0], omega_str);
                    free(p_arr);
                }
                return 1;
            }
            sol_idx = hopf_indices[0];
        }

        H5::Group basic = h5data.openGroup("basic_info");
        int pmask_file[2] = { 0, 1 };
        size_t np_file = 0, nstages_file = 10;
        deserialize(basic, "pmask", pmask_file);
        deserialize(basic, "np", np_file);
        deserialize(basic, "nstages", nstages_file);

        param_name = parameter ? parameter.value() : infer_parameter_name(params, pmask_file[1], np_file);
        const int parnum = parameter ? resolve_parameter_index(params, param_name) : pmask_file[1];
        const size_t use_ncol = ncol ? (size_t)ncol.value() : nstages_file;

        TWIST::Collocator col;
        bool use_stored_hopf = false;
        if (use_ncol == nstages_file) {
            TWIST::Collocator tmp(h5data, sol_idx);
            lib = tmp.getLibHandle();
            col = tmp;
            tmp.clearLibHandle();
        }
        else {
            col = load_collocator_from_h5data_and_index(h5data, sol_idx, use_ncol, &lib);
        }

        double omega = 0.0;
        std::vector<std::complex<double>> psi;
        H5::Group sol_group = h5data.openGroup(fmt::format("{}", sol_idx));
        if (use_ncol == nstages_file && sol_group.nameExists("hopf_info")) {
            H5::Group hg = sol_group.openGroup("hopf_info");
            double *pr = nullptr, *pi = nullptr;
            size_t len = 0;
            deserialize(hg, "omega", omega);
            deserialize(hg, "psi_real", pr, len);
            deserialize(hg, "psi_imag", pi, len);
            if (len == col.NUnknowns() && omega > 0.0) {
                psi.resize(len);
                for (size_t i = 0; i < len; i++) {
                    psi[i] = std::complex<double>(pr[i], pi[i]);
                }
                use_stored_hopf = true;
            }
            free(pr);
            free(pi);
        }

        if (!use_stored_hopf) {
            fmt::println("Computing Hopf eigenpair at solution index {}...", sol_idx);
            if (!compute_hopf_eigenpair_fallback(col, omega, psi, &h5data, sol_idx)) {
                fmt::println(COLOR_RED "Failed to locate/refine Hopf eigenpair at index {}." COLOR_RESET, sol_idx);
                if (lib) {
                    dlclose(lib);
                }
                return 1;
            }
        }

        fmt::println("Initialized from Hopf point {} ({} = {:.6e}, c = {:.6f}, omega = {:.6f}, T0 = {:.6f})",
            sol_idx, param_name, col.p()[parnum], col.waveSpeed(), omega, (2.0 * M_PI) / omega);

        solver.initFromHopfPoint(col, omega, psi.data(), (size_t)nt, (size_t)ncol_time, parnum);
        col.clearLibHandle();
    }

    TWIST::HopfContinuationBounds bounds;
    bounds.ds = ds;
    bounds.dsmin = dsmin;
    bounds.dsmax = dsmax;
    bounds.parmin = parmin;
    bounds.parmax = parmax;
    bounds.geps = geps;
    bounds.min_nodes_adapt = (size_t)min_nodes;
    bounds.max_nodes_adapt = (size_t)max_nodes;
    bounds.min_nt_adapt = (size_t)min_nt;
    bounds.max_nt_adapt = (size_t)max_nt;
    bounds.allow_mesh_adaptation = !no_adapt;

    hopf_continuation_work_loop(solver, bounds, name, parameter_set, param_name, tag, prefix, verbose, max_steps);
    if (lib) {
        dlclose(lib);
    }
    return 0;
}

void HopfArgs::welcome()
{
    fmt::println("Continue modulating traveling wave (relative periodic orbit) branches from Hopf points");
}
