#pragma once

#include "cli/cli_common.h"

struct SimulateArgs : public argparse::Args
{
    std::string &data = arg("data", "path to continuation data");
    int &index = arg("index", "solution index in `data` to start simulation from");
    // std::vec

    // simulate options
    bool &simulate_adaptive_mesh = flag("adaptive-mesh", "Enable adaptive mesh for PDE simulation. The adaptive mesh can introduce discretization errors that can alter the behavior of unstable solutions").set_default(false);
    int &simulate_nspace = kwarg("nspace", "number of spatial grid points for PDE simulation.").set_default(1200);
    int &simulate_nsample = kwarg("nsample", "number of temporal points at which to sample PDE simulation.").set_default(1000);
    double &simulate_nloops = kwarg("nloops", "number of (theoretical) times the solution should loop around the ring.").set_default(1.0);
    double &simulate_xeps = kwarg("xeps", "spatial error tolerance (ignored if not using adaptive mesh)").set_default(2e-3);
    bool &simulate_animate = flag("animate", "animate the simulation (leads to much slower simulations)").set_default(false);
    SimulationMethods &simulation_method = kwarg("method", "integration method to be used for simulation").set_default(SimulationMethods::DIRK965);
    std::optional<std::string> &simulate_save_path = kwarg("save-data", "where to save the data (only saved if specified)");
    double &simulate_vscale = kwarg("vscale", "[EXPERIMENTAL] Scale voltage before sim starts").set_default(0.0);
    double &simulate_atol = kwarg("atol", "absolute tolerance for time stepper").set_default(1e-6);
    double &simulate_rtol = kwarg("rtol", "relative tolerance for time stepper").set_default(2e-3);

    // plotting options
    bool &plotting_colorbar = flag("colorbar", "show colorbar on the plot").set_default(false);
    bool &plotting_apply_transforms = flag("apply-transforms", "apply any transforms specified by spec file for the simulated model").set_default(false);
    bool &plotting_hide_ticks = flag("hide-ticks", "hide the x/y-axis ticks").set_default(false);
    bool &with_initial_profile = flag("with-initial-profile", "show the starting profile").set_default(false);
    std::optional<std::string> &plotting_rc_file = kwarg("rc-file", "path to rc/style file");
    std::optional<std::string> &plotting_color_bar_label = kwarg("colorbar-label", "label for the colorbar");
    std::optional<std::string> &plotting_xlabel = kwarg("xlabel", "x-axis label");
    std::optional<std::string> &plotting_ylabel = kwarg("ylabel", "y-axis label");
    std::optional<std::string> &plotting_save_fig = kwarg("save-fig", "path to save the figure (does not show plot)");
    std::optional<std::string> &profile_ylabel = kwarg("profile-ylabel", "y-axis label for profile (if using --with-initial-profile flag)");

    int run() override;
    void welcome() override;
};
