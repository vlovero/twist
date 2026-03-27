#pragma once

#include "cli/cli_common.h"


struct ContinuationArgs : public argparse::Args
{
    std::string &model = arg("model", "path to JSON spec file of the model (assumed that model is compiled and processed)");
    std::string &parameter_set = kwarg("parameter-set", "which parameter set to use").set_default("default");
    std::vector<std::string> &from_data = kwarg("from-data", "using solution from continuation output").set_default(std::vector<std::string>{}).multi_argument();

    // continuation options
    bool &forward = flag("f,forward", "perform continuation in forward direction").set_default(false);
    bool &backward = flag("b,backward", "perform continuation in backward direction").set_default(false);
    std::string &parameter = kwarg("p,parameter", "name of parameter");
    double &ds = kwarg("ds", "initial step size");
    double &dsmin = kwarg("dsmin", "min step size");
    double &dsmax = kwarg("dsmax", "max step size");
    double &parmin = kwarg("parmin", "min paramter value");
    double &parmax = kwarg("parmax", "max paramter value");

    // solver options
    bool &solve_init = flag("solve-init", "solve for initial traveling wave").set_default(false);
    int &ncol = kwarg("ncol", "number of collocation points for solver to use").set_default(10);
    double &solve_tol = kwarg("solve-tol", "tolerance for traveling wave solver").set_default(0.0);
    double &solve_geps = kwarg("solve-geps", "global tolerance for traveling wave solution when adapting the mesh").set_default(1e-12);
    double &solve_min_damp = kwarg("solve-min-damp", "minimum damping factor when performing Newton's method").set_default(std::pow(0.5, 15));
    int &solve_max_iter = kwarg("solve-max-iter", "maximum number of newton iterations for solver").set_default(100);
    int &solve_nadapt = kwarg("solve-nadapt", "number of times to adapt the mesh").set_default(2);
    int &solve_min_nodes = kwarg("solve-min-nodes", "minimum number of nodes required when adapting the mesh").set_default(2);
    int &solve_max_nodes = kwarg("solve-max-nodes", "maximum number of nodes required when adapting the mesh").set_default(-1);
    bool &quiet = flag("q,solve-quietly", "don't show solver information").set_default(false);
    std::optional<int> &nthreads = kwarg("nthreads", "set the number of OpenMP threads to be used for the solver");

    // output options
    std::string &tag = kwarg("t,tag", "tag for output file").set_default("latest");
    std::string &prefix = kwarg("prefix", "tag for output file").set_default(".cache/continuation_data");

    // experimental
    int &add_bumps = kwarg("add-bumps", "[EXPERIMENTAL] repeat wave").set_default(0);

    int run() override;
    void welcome() override;
};