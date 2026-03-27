#pragma once

#include "cli/cli_common.h"

struct ImportArgs : public argparse::Args
{
    std::string &spec_path = arg("spec", "path to JSON spec file for model");
    std::string &data_path = arg("data", "path to data that is to be imported");
    std::string &parameter_set = kwarg("p,parameter-set", "which parameter set in spec file to use").set_default("default");
    TWIST::TextFileTypes &file_type = kwarg("f,file-type", "which type of file is being imported").set_default(TWIST::TextFileTypes::csv);
    double &wave_speed = kwarg("c,wave-speed", "initial estimate of the wave speed").set_default(std::numeric_limits<double>::infinity());
    bool &derivatives_included = flag("derivatives-included", "Assume that derivatives of diffusive variables are already present").set_default(false);
    int run() override;
    void welcome() override;
};