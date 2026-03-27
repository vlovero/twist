#pragma once

#include "cli/cli_common.h"

struct InfoArgs : public argparse::Args
{
    std::string &file = arg("file", "data file");
    bool &get_length = flag("l,length", "print how many solutions/spectra are stored in the given file").set_default(false);
    bool &spectrum_info = flag("s,spectrum", "print info about each spectrum in the given file").set_default(false);
    std::vector<std::string> &which_indices = kwarg("i,indices", "only use specified indices").set_default(std::vector<std::string>{}).multi_argument();
    bool &show_params = flag("p,show-params", "Display parameter values for specified solutions.").set_default(false);
    bool &print_meta_data = flag("m,print-meta-data", "Print meta data for given file.").set_default(false);
    std::vector<std::string> &which_params = kwarg("w,which-params", "Only show specified parametr(s)").set_default(std::vector<std::string>{}).multi_argument();
    std::optional<std::vector<TWIST::SolutionTypes>> &stype_filter = kwarg("f,filter-solution-types", "Only show specified solution types").multi_argument();

    int run() override;
    void welcome() override;
};
