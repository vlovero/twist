#include "cli/cli_simulate.h"
#include "collocator.h"
#include "libloader.h"
#include "python/plotting.h"
#include "serialize.h"
#include "tools/helpers.h"


int SimulateArgs::run()
{
    verify_hdf5_file(data);

    const int nsolutions = get_number_of_solutions(data);
    index = ((index % nsolutions) + nsolutions) % nsolutions;
    TWIST::Collocator collocator(data, index);
    const ptrdiff_t node = collocator.getNode();
    void *lib = collocator.getLibHandle();

    if (simulate_nloops < 0.0) {
        puts("--nloops keyword must be non-negative");
        return 1;
    }

    ptrdiff_t i, j, k;
    func_t func_ode = (func_t)dlsym(lib, "func_ode");
    func_t fjac_ode = (func_t)dlsym(lib, "fjac_ode");
    const double wave_speed = collocator.waveSpeed();
    const double L = collocator.spatialPeriod();
    const double *p = collocator.p();
    auto diffusion = collocator.getDiffusion();
    std::filesystem::path save_path(simulate_save_path ? simulate_save_path.value() : ".cache/sim_res.h5");

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
        sim_result = rxdiff_simulate<Tables::DIRK5>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, simulate_atol, simulate_rtol, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
        break;
    case SimulationMethods::DIRK865:
        sim_result = rxdiff_simulate<Tables::DIRK865>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, simulate_atol, simulate_rtol, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
        break;
    case SimulationMethods::DIRK965:
        sim_result = rxdiff_simulate<Tables::DIRK965>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, simulate_atol, simulate_rtol, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
        break;
    case SimulationMethods::DIRK1175:
        sim_result = rxdiff_simulate<Tables::DIRK1175>(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], simulate_xeps, simulate_atol, simulate_rtol, false, simulate_adaptive_mesh, nt, t.data(), nx, x.data(), yeval, simulate_animate);
        break;
    case SimulationMethods::IIF2:
        sim_result = rxdiff_simulate_iif2(func_ode, fjac_ode, diffusion, nx, x.data(), node - diffusion.size(), ynov, p, t[nt - 1], -1, simulate_atol, simulate_rtol, false, nt, t.data(), nx, x.data(), yeval, simulate_animate);
        break;
    }

    // make sure directories exist
    std::filesystem::create_directories(save_path.parent_path());

    {
        // local scope to ensure file is actually closed
        H5::H5File res_file(save_path.c_str(), H5F_ACC_TRUNC);
        H5::Group group = res_file.createGroup("meta_data");
        const std::string cmdline = get_program_argv();
        const std::string cwd = std::filesystem::current_path();
        // store simulation data
        serialize<double, 1>(res_file, "x", x.data(), { (hsize_t)x.size() });
        serialize<double, 1>(res_file, "t", t.data(), { (hsize_t)t.size() });
        serialize<double, 3>(res_file, "Y", yeval, { (hsize_t)t.size(), (hsize_t)x.size(), (hsize_t)(node - diffusion.size()) });

        // store meta data
        serialize<char, 1>(group, "cmdline", cmdline.data(), { cmdline.size() });
        serialize<char, 1>(group, "directory", cwd.data(), { cwd.size() });
        serialize<int>(group, "file_type", TWIST::FileType::simulation);
    }

    std::vector<std::string> args = { data, fmt::format("{}", index), save_path.c_str() };
    // add extra args here
    if (with_initial_profile) {
        args.emplace_back("--with-initial-profile");
    }
    if (plotting_colorbar) {
        args.emplace_back("--colorbar");
    }
    if (plotting_apply_transforms) {
        args.emplace_back("--apply-transforms");
    }
    if (plotting_hide_ticks) {
        args.emplace_back("--hide_ticks");
    }
    if (plotting_rc_file) {
        args.emplace_back("--rc-file");
        args.emplace_back(plotting_rc_file.value());
    }
    if (plotting_color_bar_label) {
        args.emplace_back("--colorbar-label");
        args.emplace_back(plotting_color_bar_label.value());
    }
    if (plotting_xlabel) {
        args.emplace_back("--xlabel");
        args.emplace_back(plotting_xlabel.value());
    }
    if (plotting_ylabel) {
        args.emplace_back("--ylabel");
        args.emplace_back(plotting_ylabel.value());
    }
    if (plotting_save_fig) {
        args.emplace_back("--save-fig");
        args.emplace_back(plotting_save_fig.value());
    }
    if (profile_ylabel) {
        args.emplace_back("--profile-ylabel");
        args.emplace_back(profile_ylabel.value());
    }

    run_python_script(python::simviz::get_script(), args);

    free(std::get<0>(sim_result));
    free(std::get<1>(sim_result));
    free(yeval);
    free(ynov);
    free(ywithv);

    return 0;
}

void SimulateArgs::welcome()
{
    fmt::println("Simulate a traveling wave computed from continuation");
}
