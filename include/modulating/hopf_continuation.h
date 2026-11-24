#pragma once

#include "modulating/hopf_branch_solver.h"
#include <string>

void hopf_continuation_work_loop(TWIST::HopfBranchSolver &solver, TWIST::HopfContinuationBounds &bounds, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix = ".cache/continuation_data", bool verbose = false, int max_steps = 5000, const std::string &target_h5_path = "", const std::string &branch_name = "", int source_index = -1, bool skip_initial_serialize = false);
