#pragma once

#include "shared.h"
#include <tuple>

enum PreprocessingMethods
{
    DIRK5_N = 0,
    DIRK865_N,
    DIRK965_N,
    DIRK1175_N,
    DIRK5_A,
    DIRK865_A,
    DIRK965_A,
    DIRK1175_A,
    IIF2_N
};

enum SimulationMethods
{
    DIRK5 = 0,
    DIRK865,
    DIRK965,
    DIRK1175,
    IIF2
};

void generate_initial_guess(func_t func_ode, func_t fjac_ode, const char *name, const char *paramter_set, const double L, const double *p, const size_t Nx, const size_t node, const RP(double) rest_state, const double tf_ode, const double tf_pde, const std::tuple<double, double, double> stim, const std::vector<std::pair<int, double>> &diffusion, PreprocessingMethods method, const bool display, const bool refine_rest_state, const bool stop_at_ode);
