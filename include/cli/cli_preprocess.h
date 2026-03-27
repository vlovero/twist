#pragma once

#include "cli/cli_common.h"

struct PreprocessArgs : public argparse::Args
{
    std::string &file = arg("file", "path to JSON spec file of the model (it's assumed the model is already compiled)");
    std::string &parameter_set = kwarg("p,parameter-set", "which parameter set to be used").set_default("default");
    double &tf_ode = kwarg("time-ode", "time duration for local dynamics simulation");
    double &tf_pde = kwarg("time-pde", "time duration for PDE simulation");
    double &stim_time = kwarg("stim-time", "time which stimulation is applied to variable 0").set_default(std::numeric_limits<double>::infinity());
    double &stim_dur = kwarg("stim-dur", "duration which stimulation is applied to variable 0").set_default(2.0);
    double &stim_amp = kwarg("stim-amp", "amplitude if stimulation applied to variable 0").set_default(0.4);
    size_t &Nx = kwarg("num-space", "number of points in spatial discretization").set_default(600);
    PreprocessingMethods &method = kwarg("m,method", "method").set_default(PreprocessingMethods::DIRK965_A);
    bool &display_start_and_end = flag("d,display-start-and-end", "show the solutions before and after performing the preprocessing").set_default(false);
    bool &refine_rest_state = flag("r,refine-rest-state", "Refine the supplied rest state using root-finding.").set_default(false);
    bool &stop_at_ode = flag("o,stop-at-ode", "Stop the preprocessing after the local dynamics have been intergrated.").set_default(false);

    int run() override;
    void welcome() override;
};