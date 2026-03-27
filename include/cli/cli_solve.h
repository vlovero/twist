#pragma once

#include "cli/cli_common.h"

struct SolveArgs : public argparse::Args
{
    std::string &model = arg("model", "path to JSON spec file of the model (assumed that model is compiled and processed)");
    std::string &parameter_set = kwarg("parameter-set", "which parameter set to use").set_default("default");
    std::vector<std::string> &from_data = kwarg("from-data", "using solution from continuation output").set_default(std::vector<std::string>{}).multi_argument();
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
    bool &solve_quietly = flag("q,solve-quietly", "don't show solver information").set_default(false);
    std::optional<int> &nthreads = kwarg("nthreads", "set the number of OpenMP threads to be used for the solver");

    // plotting options
    bool &plot_after_solve_init = flag("plot-after-solve-init", "Plot the computed traveling wave").set_default(false);
    bool &plot_dense = flag("plot-dense", "Plot the dense output (smooth) of the computed traveling wave solution (requires --plot-after-solve-init)").set_default(false);

    // spectrum options
    bool &spectrum = flag("spectrum", "Compute the spectrum after computing a traveling wave solution").set_default(false);
    bool &essential_spectrum = flag("essential-spectrum", "Compute the essential spectrum (first computes the spectrum)").set_default(false);
    std::string &essential_spectrum_filter = kwarg("essential-spectrum-filter", "expression for choosing which eigenvalues to start from for essential spectrum. use `real` for real component. `imag` for imaginary").set_default("");
    TWIST::SpectrumStrategy &strategy = kwarg("spectrum-strategy", "Which algorithm to compute the spectrum with").set_default(TWIST::SpectrumStrategy::shiftAndInvert);
    bool &subspace = flag("subspace", "Compute a subset of the spectrum using the Krylov-Schur algorithm with the Cayley transform").set_default(false);
    int &subspace_size = kwarg("subspace-size", "How many eigenvalues to search for using the Krylov-Schur method").set_default(10);
    double &subspace_sigma = kwarg("subspace-sigma", "Shift for the Cayley transform").set_default(0.2);

    // simulate options
    bool &simulate = flag("simulate", "Perform a time based simulation of the PDE using the computed wave profile as the initial state").set_default(false);
    bool &simulate_adaptive_mesh = flag("simulate-adaptive-mesh", "Enable adaptive mesh for PDE simulation. The adaptive mesh can introduce discretization errors that can alter the behavior of unstable solutions").set_default(false);
    int &simulate_nspace = kwarg("simulate-nspace", "number of spatial grid points for PDE simulation.").set_default(1200);
    int &simulate_nsample = kwarg("simulate-nsample", "number of temporal points at which to sample PDE simulation.").set_default(1000);
    double &simulate_nloops = kwarg("simulate-nloops", "number of (theoretical) times the solution should loop around the ring.").set_default(1.0);
    double &simulate_xeps = kwarg("simulate-xeps", "spatial error tolerance (ignored if not using adaptive mesh)").set_default(2e-3);
    bool &simulate_animate = flag("simulate-animate", "animate the simulation (leads to much slower simulations)").set_default(false);
    SimulationMethods &simulation_method = kwarg("simulate-method", "integration method to be used for simulation").set_default(SimulationMethods::DIRK965);
    std::optional<std::string> &simulate_save_path = kwarg("simulate-save-path", "where to save the data (only saved it specified)");

    // misc/experimental
    int &add_bumps = kwarg("add-bumps", "[EXPERIMENTAL] repeat wave").set_default(0);
    double &simulate_vscale = kwarg("simulate-vscale", "[EXPERIMENTAL] Scale voltage before sim starts").set_default(0.0);
    bool &replace_existing_initial_guess = flag("replace-existing-initial-guess", "replace the initial traveling wave solution during initialization with the computed one (if converged)").set_default(false);

    int run() override;
    void welcome() override;
};
