#pragma once

#include <cmath>
#include <cstdint>

#include "argparse/argparse.hpp"
#include "colors.h"
#include "continuation.h"
#include "indicators/multi_progress.hpp"
#include "indicators/progress_bar.hpp"
#include "load.h"
#include "model_generation.h"
#include "postprocess.h"
#include "preprocess/drxd.h"
#include "preprocess/preprocess.h"
#include "python/plotting.h"

namespace TWIST
{
    enum TextFileTypes
    {
        csv = 0,
        tsv,
        dat,
        npy,
        h5
    };
}


std::vector<int> parse_indices_and_slices(const std::vector<std::string> &inputs, const int high);
void print_spectrum_info(const H5::H5File &file, const std::vector<int> &indices, const int ngroups, const std::optional<std::vector<TWIST::SolutionTypes>> &stype_filter);
void print_parameter_values(const H5::H5File &file, const std::vector<int> &indices, const int ngroups);
void print_parameter_values(const H5::H5File &file, const std::vector<int> &indices, const int ngroups, const std::vector<std::string> &param_names, const std::vector<int> &param_indices, const std::optional<std::vector<TWIST::SolutionTypes>> &stype_filter);