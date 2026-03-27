#pragma once

#include "cli/cli_common.h"

struct VisualizeArgs : public argparse::Args
{
    std::vector<std::string> &files = arg("file", "continuation data file(s)").multi_argument().set_default(std::vector<std::string>{});
    bool &l2 = flag("l2", "Plot the L2 norm on the y-axis instead of wave speed");
    bool &apd = flag("apd", "Plot the APD on the y-axis instead of wave speed");
    bool &individual = flag("individual", "Instead of one single figure, plot each subplot in its own figure");
    bool &no_stability = flag("no-stability", "Do not show stability, even if already computed");
    bool &actual_spatial_period = flag("actual-spatial-period", "Use the actual spatial period instead of normalizing it on the x-axis");
    bool &center_wave = flag("center-wave", "(if --actual-spatial-period set) center the waves on [-L/2, L/2] instead of [0, L]");
    bool &hide_bifurcation_points = flag("hide-bifurcation-points", "Do not show any bifurcation points, even if already computed");
    bool &print_rc_file = flag("print-rc-file", "Print the current rc file to the terminal");
    bool &dashed_unstable = flag("dashed-unstable", "Make unstable branches dashed on figure");
    bool &apply_transforms = flag("apply-transforms", "Apply specified transforms to any potentially transformed states");
    bool &stability_legend = flag("stability-legend", "Show a legend for the stability (if stability data available)");

    int &apd_value = kwarg("apd-value", "(if --apd set) What precentage of APD to use").set_default(90);
    std::string &dispersion_mode = kwarg("dispersion-mode", "Display mode for the dispersion curve. Usage: ('default' | 'multi:color1,color2,...' | 'cmap:name_of_color_map')").set_default("default");
    std::optional<std::string> &rc_file = kwarg("rc-file", "Path to an rc or mplstyle file for plot settings");
    std::optional<std::string> &save = kwarg("save", "Save figures to specified path if set");
    std::optional<std::string> &spatial_period_units = kwarg("spatial-period-units", "(if --actual-spatial-period) add units to axes");
    std::optional<std::string> &wave_speed_units = kwarg("wave-speed-units", "Add units to wave speed axis");
    std::optional<std::string> &wave_profile_units = kwarg("wave-profile-units", "Add units to wave profile axis");
    std::optional<std::vector<std::string>> &dispersion_labels = kwarg("dispersion-labels", "Add a legend with the specified labels for each dipsersion curve file").multi_argument();
    std::optional<std::vector<int>> &fixed_solutions = kwarg("fixed-solutions", "Disable interactive mode and only show the specified solutions (can be empty to only show dispersion curve)").multi_argument();
    std::optional<std::vector<std::string>> &starting_extents = kwarg("starting-extents", "Set the extents of each axes. Usage ('<index>:xlo,xhi,ylo,yhi')").multi_argument();
    std::string &stable_color = kwarg("stable-color", "Color to indicate stable solutions").set_default("red");
    std::string &unstable_color = kwarg("unstable-color", "Color to indicate unstable solutions").set_default("black");

    int run() override;
    void welcome() override;
};
