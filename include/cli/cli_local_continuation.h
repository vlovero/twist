#pragma once

#include "cli/cli_common.h"


struct LocalContinuationArgs : public argparse::Args
{
    std::string &model = arg("model", "path to JSON spec file of the model (assumed that model is compiled and processed)");
    std::string &parameter_set = kwarg("parameter-set", "which parameter set to use").set_default("default");
    std::vector<std::string> &from_data = kwarg("from-data", "using solution from continuation output").set_default(std::vector<std::string>{}).multi_argument();
    ;

    // continuation options
    bool &forward = flag("f,forward", "perform continuation in forward direction").set_default(false);
    bool &backward = flag("b,backward", "perform continuation in backward direction").set_default(false);
    std::string &parameter = kwarg("p,parameter", "name of parameter");
    double &wave_speed = kwarg("wave-speed", "initial wave speed (if not starting from data)");
    double &ds = kwarg("ds", "initial step size");
    double &dsmin = kwarg("dsmin", "min step size");
    double &dsmax = kwarg("dsmax", "max step size");
    double &parmin = kwarg("parmin", "min paramter value");
    double &parmax = kwarg("parmax", "max paramter value");

    // output options
    std::string &tag = kwarg("t,tag", "tag for output file").set_default("latest");
    std::string &prefix = kwarg("prefix", "tag for output file").set_default(".cache/continuation_data");

    int run() override;
    void welcome() override;
};