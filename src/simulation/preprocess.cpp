#include "preprocess/preprocess.h"
#include "cli/colors.h"
#include "fmt/ranges.h"
#include "numpy_like.h"
#include "preprocess/drxd.h"
#include "preprocess/lsoda.h"
#include "preprocess/roots.h"
#include "python/plotting.h"
#include "serialize.h"
#include "shared.h"
#include <cstdio>
#include <filesystem>
#include <valarray>

double *add_velocities(const double h, const double *y, const ptrdiff_t Nx, const ptrdiff_t node, const std::vector<std::pair<int, double>> &diffusion)
{
    const ptrdiff_t ndiff = diffusion.size();
    double *u = new double[Nx * (node + ndiff)];
    double velocity;
    ptrdiff_t i, j, k;

    auto contains = [](int i, const std::vector<std::pair<int, double>> &diffusion) -> bool {
        for (const auto &[loc, _] : diffusion) {
            if (loc == i) {
                return true;
            }
        }
        return false;
    };

    auto mod = [](ptrdiff_t a, ptrdiff_t b) -> ptrdiff_t { return ((a % b) + b) % b; };

    for (i = 0; i < Nx; i++) {
        for (j = 0, k = 0; j < node; j++, k++) {
            u[i * (node + ndiff) + k] = y[i * node + j];
            if (contains(j, diffusion)) {
                velocity = (y[((i + 1) % Nx) * node + j] - y[mod(i - 1, Nx) * node + j]) / (2 * h);
                u[i * (node + ndiff) + k + 1] = velocity;
                k++;
            }
        }
    }

    return u;
}

double *add_velocities(const double *x, const double *y, const ptrdiff_t Nx, const ptrdiff_t node, const std::vector<std::pair<int, double>> &diffusion)
{
    const ptrdiff_t ndiff = diffusion.size();
    double *u = new double[Nx * (node + ndiff)];
    double velocity, hs, hd, c1, c2, c3;
    ptrdiff_t i, j, k;

    auto contains = [](int i, const std::vector<std::pair<int, double>> &diffusion) -> bool {
        for (const auto &[loc, _] : diffusion) {
            if (loc == i) {
                return true;
            }
        }
        return false;
    };

    auto mod = [](ptrdiff_t a, ptrdiff_t b) -> ptrdiff_t { return ((a % b) + b) % b; };

    for (i = 0; i < Nx; i++) {
        hs = (i != 0) ? (x[i] - x[i - 1]) : (x[Nx - 1] - x[Nx - 2]);
        hd = (i != (Nx - 1)) ? (x[i + 1] - x[i]) : (x[1] - x[0]);
        c1 = -hd / (hs * (hs + hd));
        c2 = (hd - hs) / (hd * hs);
        c3 = +hs / (hd * (hs + hd));
        for (j = 0, k = 0; j < node; j++, k++) {
            u[i * (node + ndiff) + k] = y[i * node + j];
            if (contains(j, diffusion)) {
                velocity = c1 * y[mod(i - 1, Nx) * node + j];
                velocity += c2 * y[i * node + j];
                velocity += c3 * y[((i + 1) % Nx) * node + j];
                u[i * (node + ndiff) + k + 1] = velocity;
                k++;
            }
        }
    }
    return u;
}

void make_alphas(const ptrdiff_t nx, const double *x, const ptrdiff_t i, const ptrdiff_t nside, double *alpha)
{
    ptrdiff_t k, j;
    const bool left_ok = nside <= i;
    const bool right_ok = i < (nx - nside);
    double acc_left, acc_right;
    assert(nx >= (2 * nside + 1));

    alpha[nside] = 0;
    acc_left = 0;
    acc_right = 0;
    if (left_ok && right_ok) {
        // no special treatment
        for (k = 0; k < nside; k++) {
            acc_right += x[i + k + 1] - x[i + k];
            acc_left -= x[i - k] - x[i - k - 1];
            alpha[nside + k + 1] = acc_right;
            alpha[nside - k - 1] = acc_left;
        }
    }
    else if (left_ok) {
        // special treatment of right
        for (k = 0; k < nside; k++) {
            if ((i + k) >= (nx - 1)) {
                j = k;
                acc_right += x[j + 1] - x[j];
            }
            else {
                acc_right += x[i + k + 1] - x[i + k];
            }
            acc_left -= x[i - k] - x[i - k - 1];
            alpha[nside + k + 1] = acc_right;
            alpha[nside - k - 1] = acc_left;
        }
    }
    else if (right_ok) {
        // special treatment of left
        for (k = 0; k < nside; k++) {
            acc_right += x[i + k + 1] - x[i + k];
            if ((i - k) <= 0) {
                j = nx - 1 - k;
                acc_left -= x[j] - x[j - 1];
            }
            else {
                acc_left -= x[i - k] - x[i - k - 1];
            }
            alpha[nside + k + 1] = acc_right;
            alpha[nside - k - 1] = acc_left;
        }
    }
}

double *add_velocities(const int nside, const double *x, const double *y, const ptrdiff_t Nx, const ptrdiff_t node, const std::vector<std::pair<int, double>> &diffusion)
{
    const ptrdiff_t ndiff = diffusion.size();
    double *u = new double[Nx * (node + ndiff)];
    double velocity;
    ptrdiff_t i, j, k, l;
    double alpha[2 * nside + 1];
    double coeff[2 * nside + 1];
    double rhs[2 * nside + 1];

    auto contains = [](int i, const std::vector<std::pair<int, double>> &diffusion) -> bool {
        for (const auto &[loc, _] : diffusion) {
            if (loc == i) {
                return true;
            }
        }
        return false;
    };

    auto mod = [](ptrdiff_t a, ptrdiff_t b) -> ptrdiff_t { return ((a % b) + b) % b; };

    for (i = 0; i < (2 * nside + 1); i++) {
        rhs[i] = 0;
    }
    rhs[1] = 1;

    for (i = 0; i < Nx; i++) {
        make_alphas(Nx, x, i, nside, alpha);
        solve_vanderT(2 * nside + 1, alpha, rhs, coeff);
        for (j = 0, k = 0; j < node; j++, k++) {
            u[i * (node + ndiff) + k] = y[i * node + j];
            if (contains(j, diffusion)) {
                velocity = 0;
                for (l = -nside; l <= nside; l++) {
                    velocity += coeff[l + nside] * y[mod(i + l, Nx) * node + j];
                }
                u[i * (node + ndiff) + k + 1] = velocity;
                k++;
            }
        }
    }
    return u;
}


bool check_rest_state_is_stable(func_t fjac_ode, const double *rest_state, const size_t node, const double *p)
{
    int info, lwork;
    size_t k;
    double temp;
    double jac[node * node];
    double wr[node];
    double wi[node];
    double *work = nullptr;
    bool is_stable = true;
    dgeev("N", "N", node, jac, node, wr, wi, NULL, node, NULL, node, &temp, -1, &info);
    lwork = temp;
    work = (double *)malloc(lwork * sizeof(double));

    fjac_ode(rest_state, p, jac);
    dgeev("N", "N", node, jac, node, wr, wi, NULL, node, NULL, node, work, lwork, &info);

    for (k = 0; k < node; k++) {
        if ((0.0 < wr[k]) && (wi[k] == 0)) {
            is_stable = false;
            break;
        }
    }

    if constexpr (false) {
        std::valarray<double> tmp(wr, node);
        const double rbar = tmp.sum() / node;
        const double rstd = ((tmp - rbar) * (tmp - rbar)).sum() / node;
        const double stiffness = std::abs(tmp).max() / std::abs(tmp).min();
        fmt::println("{:^14s} & {:^14s} & {:^14s} \\\\", "stiffness", "mean", "std. dev.");
        fmt::println("{:.8e} & {:.8e} & {:.8e} \\\\", stiffness, rbar, rstd);
        for (k = 0; k < node; k++) {
            fmt::println("{:.8e}{:+.8e}", wr[k], wi[k]);
        }
    }

    free(work);

    return is_stable;
}

inline auto run_simulation(const PreprocessingMethods method, func_t func_ode, func_t fjac_ode, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t node, const double *y, const double *p, const double tf_pde)
{
    const double xeps = 2e-3;
    const double atol = 1e-6;
    const double rtol = 2e-3;

    switch (method) {
    case PreprocessingMethods::DIRK5_N:
        return rxdiff_simulate<Tables::DIRK5>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, false, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK865_N:
        return rxdiff_simulate<Tables::DIRK865>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, false, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK965_N:
        return rxdiff_simulate<Tables::DIRK965>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, false, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK1175_N:
        return rxdiff_simulate<Tables::DIRK1175>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, false, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK5_A:
        return rxdiff_simulate<Tables::DIRK5>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, true, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK865_A:
        return rxdiff_simulate<Tables::DIRK865>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, true, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK965_A:
        return rxdiff_simulate<Tables::DIRK965>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, true, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::DIRK1175_A:
        return rxdiff_simulate<Tables::DIRK1175>(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, atol, rtol, true, true, 0, NULL, 0, NULL, NULL, false);
    case PreprocessingMethods::IIF2_N:
        return rxdiff_simulate_iif2(func_ode, fjac_ode, diffusion, Nx, x, node, y, p, tf_pde, xeps, 1e-3, 1e-2, true, 0, NULL, 0, NULL, NULL, false);
    }
}

void generate_initial_guess(func_t func_ode, func_t fjac_ode, const char *name, const char *parameter_set, const double L, const double *p, size_t Nx, const size_t node, const RP(double) rest_state, const double tf_ode, double tf_pde, const std::tuple<double, double, double> stim, const std::vector<std::pair<int, double>> &diffusion, PreprocessingMethods method, const bool display, const bool refine_rest_state, const bool stop_at_ode)
{
    size_t i;
    char go_again;
    double *y = new double[Nx * node];
    double *t = nullptr;
    std::vector<double> x{ linspace(0, L, Nx) };
    std::vector<double> teval{ linspace(0, tf_ode, Nx) };
    double *u = nullptr;
    const double stim_time = std::get<0>(stim);
    const double stim_dur = std::get<1>(stim);
    const double stim_amp = std::get<2>(stim);
    double max_norms[node + 1];
    double wave_speed;

    std::tuple<double *, double *, ptrdiff_t, double> result = { nullptr, nullptr, 0, 0.0 };

    auto fstim = [=](double tt, double *z, double *res, const void *) {
        func_ode(z, p, res);
        if ((stim_time <= tt) && (tt <= (stim_time + stim_dur))) {
            res[0] += stim_amp;
        }
    };

    // make this an option later
    if (refine_rest_state) {
        fsolve(func_ode, fjac_ode, node, (double *)rest_state, 1e-8, 1e-8, p);
        fmt::println(COLOR_GREEN "The computed rest state is\n{}" COLOR_RESET, fmt::join(std::vector<double>(rest_state, rest_state + node), ",\n"));
        if (!check_rest_state_is_stable(fjac_ode, rest_state, node, p)) {
            puts(COLOR_RED "ERROR: The rest state above found via root-finding is not a stable fixed point." COLOR_RESET);
            puts(COLOR_YELLOW "Consider skipping the rest state refinement." COLOR_RESET);
            exit(1);
        }
        puts("");
    }
    memset(max_norms, 0, sizeof(double) * (node + 1));
    max_norms[0] = std::numeric_limits<double>::infinity();
    lsoda::integrate<double>(fstim, rest_state, teval.data(), y, 0.0, std::min(stim_dur, 0.5 * tf_ode / (Nx - 1)), 0.0, 1e-8, 1e-8, node, Nx, max_norms, NULL);
    if (display) {
        {
            H5::H5File res_file(".cache/solve_res.h5", H5F_ACC_TRUNC);
            H5::Group group = res_file.createGroup("data");
            serialize<double, 1>(group, "x", teval.data(), { Nx });
            serialize<double, 2>(group, "y", y, { Nx, node });
        }
        int code = run_python_script(python::interim::get_script(), {});
        if (code) {
            exit(0);
        }
    }
    if (stop_at_ode) {
        return;
    }

    do {
        // this should only be true after a simulation has already ran
        if (t != nullptr) {
            x.resize(Nx);
            memcpy(x.data(), t, Nx * sizeof(double));
            free(t);
        }
        result = run_simulation(method, func_ode, fjac_ode, diffusion, Nx, x.data(), node, y, p, tf_pde);

        free(y); // clear old memory before reassigning
        t = std::get<0>(result);
        y = std::get<1>(result);
        Nx = std::get<2>(result);
        wave_speed = std::isfinite(std::get<3>(result)) ? std::get<3>(result) : wave_speed;

        if (display) {
            {
                H5::H5File res_file(".cache/solve_res.h5", H5F_ACC_TRUNC);
                H5::Group group = res_file.createGroup("data");
                serialize<double, 1>(group, "x", t, { Nx });
                serialize<double, 2>(group, "y", y, { Nx, node });
            }
            int code = run_python_script(python::interim::get_script(), { "--space-ap-xlabel", "x" });
            if (code) {
                exit(0);
            }

            // only allow for longer sim if in display mode
            // otherwise there's no point...
            fmt::print("Simulate longer? (y/n) ");
            fflush(stdout);
            if ((scanf(" %c", &go_again) != 1) || ((go_again != 'y') && (go_again != 'Y'))) {
                break;
            }
            fmt::print("Enter how much longer to simulate: ");
            fflush(stdout);
            if ((scanf(" %lf", &tf_pde) != 1)) {
                break;
            }
            if (tf_pde <= 0) {
                // code won't allow going backwards
                break;
            }
        }
        else {
            break;
        }
    } while (true);

    u = add_velocities(2, t, y, Nx, node, diffusion);
    for (i = 0; i < Nx; i++) {
        t[i] /= L;
    }

    {
        std::filesystem::create_directories(".cache/models/init_data");
        std::string file_path = fmt::format(".cache/models/init_data/{}-{}.h5", name, parameter_set);
        H5::H5File file(file_path, H5F_ACC_TRUNC);
        {
            H5::Group group(file.createGroup("meta_data"));
            const std::string program_argv{ get_program_argv() };
            const std::string cwd = std::filesystem::current_path();
            serialize<int>(group, "file_type", 1);
            serialize<char, 1>(group, "cmdline", program_argv.c_str(), { program_argv.size() });
            serialize<char, 1>(group, "directory", cwd.c_str(), { cwd.size() });
        }
        H5::Group group(file.createGroup("data"));
        serialize<double, 1>(group, "y", u, { Nx * (node + diffusion.size()) });
        serialize<size_t>(group, "nnodes", Nx);
        serialize<double>(group, "wave_speed", wave_speed);
        serialize<double, 1>(group, "t", t, { Nx });
    }

    free(y);
    free(u);
}