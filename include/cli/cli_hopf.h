#pragma once

#include "cli/cli_common.h"

struct HopfArgs : public argparse::Args
{
    std::string &model = arg("model", "path to JSON spec file of the model");
    std::vector<std::string> &from_data = kwarg("from-data", "steady continuation HDF5 file and optional Hopf solution index: <file.h5> [index]").set_default(std::vector<std::string>{}).multi_argument();
    std::vector<std::string> &resume = kwarg("resume", "resume from an existing hopf continuation HDF5 file: <hopf.h5> [index]").set_default(std::vector<std::string>{}).multi_argument();

    std::string &parameter_set = kwarg("parameter-set", "which parameter set to use").set_default("default");
    std::optional<std::string> &parameter = kwarg("p,parameter", "continuation parameter name (auto-inferred from HDF5 pmask if omitted)");
    double &parmin = kwarg("parmin", "minimum continuation parameter value");
    double &parmax = kwarg("parmax", "maximum continuation parameter value");

    double &ds = kwarg("ds", "initial continuation step size").set_default(0.05);
    double &dsmin = kwarg("dsmin", "minimum continuation step size").set_default(1e-6);
    double &dsmax = kwarg("dsmax", "maximum continuation step size").set_default(0.2);
    double &geps = kwarg("geps", "unified space-time global error tolerance").set_default(1e-10);

    std::optional<int> &ncol = kwarg("ncol", "spatial GL collocation stages (auto-inferred from HDF5 if omitted)");
    int &ncol_time = kwarg("ncol-time", "temporal Radau IIA dG stages (2=dG1, 3=dG2, 4=dG3)").set_default(3);
    int &nt = kwarg("nt", "initial number of temporal elements").set_default(20);
    int &min_nodes = kwarg("min-nodes", "minimum spatial nodes during adaptation").set_default(16);
    int &max_nodes = kwarg("max-nodes", "maximum spatial nodes during adaptation").set_default(200);
    int &min_nt = kwarg("min-nt", "minimum temporal elements during adaptation").set_default(16);
    int &max_nt = kwarg("max-nt", "maximum temporal elements during adaptation").set_default(48);
    bool &no_adapt = flag("no-adapt", "disable adaptive space-time mesh refinement").set_default(false);
    int &max_steps = kwarg("max-steps", "maximum number of continuation steps").set_default(5000);

    std::optional<int> &nthreads = kwarg("nthreads", "number of OpenMP threads");
    bool &verbose = flag("v,verbose", "print Newton/GMRES/adaptation diagnostics").set_default(false);

    std::string &tag = kwarg("t,tag", "tag for output file").set_default("latest");
    std::string &prefix = kwarg("prefix", "output directory prefix").set_default(".cache/continuation_data");

    int run() override;
    void welcome() override;
};
