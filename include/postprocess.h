#ifndef TWIST_POSTPROCESS_H
#define TWIST_POSTPROCESS_H

#include "H5Cpp.h"
#include "collocator.h"
#include "fmt/core.h"
#include <utility>
#include <vector>

void locate_all_branch_points(const std::string &file, TWIST::Collocator &collocator_cache);

// NEW
void evaluate_spectrum_for_each_solution_inplace(const std::vector<std::string> &files, const TWIST::SpectrumStrategy strategy);
void evaluate_spectrum_for_each_solution_inplace_distributed(const std::vector<std::string> &files, const TWIST::SpectrumStrategy strategy);
void locate_all_saddle_nodes_inplace(const std::string &file, TWIST::Collocator &collocator_cache);
void locate_all_hopf_points_inplace_new(const std::string &file);
void locate_all_branch_points_inplace_new(const std::string &file);

#endif // TWIST_POSTPROCESS_H