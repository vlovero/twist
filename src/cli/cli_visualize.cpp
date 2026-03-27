#include "cli/cli_visualize.h"
#include "python/plotting.h"
#include "tools/helpers.h"

int VisualizeArgs::run()
{
    std::vector<std::string> cmd;
    if (l2 && apd) {
        fmt::println("--apd and --l2 are mutually exclusive (only set one)");
        return 1;
    }
    if (!((dispersion_mode == "default") || dispersion_mode.starts_with("multi:") || dispersion_mode.starts_with("cmap:"))) {
        fmt::println("Invalid usage of --dispersion-mode. See --help");
        return 1;
    }
    if (dispersion_mode.starts_with("multi:")) {
        auto colors = split(&dispersion_mode[6], ",");
        if (colors.size() < files.size()) {
            fmt::println("Only {} colors specified but provided {} files", colors.size(), files.size());
            return 1;
        }
    }
    if (dispersion_labels && (dispersion_labels.value().size() != files.size())) {
        fmt::println("Only {} labels specified but provided {} files", dispersion_labels.value().size(), files.size());
        return 1;
    }
    for (const auto &file : files) {
        verify_hdf5_file(file);
        cmd.emplace_back(file);
    }
    if (l2) {
        cmd.emplace_back("--l2");
    }
    if (apd) {
        cmd.emplace_back("--apd");
    }
    if (individual) {
        cmd.emplace_back("--individual");
    }
    if (no_stability) {
        cmd.emplace_back("--no-stability");
    }
    if (actual_spatial_period) {
        cmd.emplace_back("--actual-spatial-period");
    }
    if (center_wave) {
        cmd.emplace_back("--center-wave");
    }
    if (hide_bifurcation_points) {
        cmd.emplace_back("--hide-bifurcation-points");
    }
    if (print_rc_file) {
        cmd.emplace_back("--print-rc-file");
    }
    if (dashed_unstable) {
        cmd.emplace_back("--dashed-unstable");
    }
    if (apply_transforms) {
        cmd.emplace_back("--apply-transforms");
    }
    if (stability_legend) {
        cmd.emplace_back("--stability-legend");
    }
    if (rc_file) {
        if (!std::filesystem::exists(rc_file.value())) {
            fmt::println("No such file '{}'", rc_file.value());
            return 1;
        }
        cmd.emplace_back("--rc-file");
        cmd.emplace_back(rc_file.value());
    }
    if (save) {
        cmd.emplace_back("--save");
        cmd.emplace_back(save.value());
    }
    if (spatial_period_units) {
        cmd.emplace_back("--spatial-period-units");
        cmd.emplace_back(spatial_period_units.value());
    }
    if (wave_speed_units) {
        cmd.emplace_back("--wave-speed-units");
        cmd.emplace_back(wave_speed_units.value());
    }
    if (wave_profile_units) {
        cmd.emplace_back("--wave-profile-units");
        cmd.emplace_back(wave_profile_units.value());
    }
    if (dispersion_labels) {
        cmd.emplace_back(fmt::format("--dispersion-labels {}", fmt::join(dispersion_labels.value(), " ")));
    }
    if (fixed_solutions) {
        cmd.emplace_back(fmt::format("--fixed-solutions {}", fmt::join(fixed_solutions.value(), " ")));
    }
    if (starting_extents) {
        cmd.emplace_back(fmt::format("--starting-extents {}", fmt::join(starting_extents.value(), " ")));
    }
    cmd.emplace_back(fmt::format("--apd-value {}", apd_value));
    cmd.emplace_back(fmt::format("--dispersion-mode {}", dispersion_mode));
    cmd.emplace_back(fmt::format("--stable-color {}", stable_color));
    cmd.emplace_back(fmt::format("--unstable-color {}", unstable_color));

    return run_python_script(python::visualize::get_script(), cmd);
}

void VisualizeArgs::welcome()
{
    fmt::println("Visualize data using python");
}