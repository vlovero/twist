#pragma once

#include "cli/cli_common.h"
#include <vector>


struct ToolsArgs : public argparse::Args
{
    std::string &file = arg("file", "data file");
    std::vector<std::string> &which_indices = kwarg("i,indices", "which indices to apply tools to").set_default(std::vector<std::string>{}).multi_argument();
    bool &prune = flag("p,prune", "Prune the specified solutions").set_default(false);
    bool &reverse = flag("r,reverse", "Reverse the order of solutions").set_default(false);
    std::vector<std::string> &files_to_append = kwarg("a,append", "append the following").set_default(std::vector<std::string>{}).multi_argument();
    std::optional<std::string> &file_to_insert = kwarg("n,insert", "append the following");
    bool &flip_direction = flag("f,flip-direction", "flip the direction of the nullspace vector").set_default(false);
    std::optional<TWIST::TextFileTypes> &export_solutions = kwarg("x,export-to", "export the selected solutions in specified format");
    int &export_solution_index = kwarg("s,solution-to-export", "index of the desired solution to be exported").set_default(0);
    std::string &export_path = kwarg("o,export-path", "path to export the data generated using the -x/--export-to argument").set_default("");
    std::vector<std::string> &import_init_wave = kwarg("import-initial-wave", "Generate an initial wave from an existing CSV file").set_default(std::vector<std::string>{});
    std::optional<std::vector<int>> &refine_between = kwarg("refine-between", "Take smaller continuation steps between the provided solutions").multi_argument();
    double &refine_ds = kwarg("refine-ds", "step size to use during refinement").set_default(1e-4);

    int run() override;
    void welcome() override;
};
