#ifndef TWIST_CLI_SETUP_H
#define TWIST_CLI_SETUP_H

#include "argparse/argparse.hpp"

#include "cli/cli_build.h"
#include "cli/cli_continuation.h"
#include "cli/cli_import.h"
#include "cli/cli_info.h"
#ifdef TWIST_INCLUDE_LOCAL_CONTINUATION
#include "cli/cli_local_continuation.h"
#endif
#include "cli/cli_postprocess.h"
#include "cli/cli_preprocess.h"
#include "cli/cli_simulate.h"
#include "cli/cli_solve.h"
#include "cli/cli_tools.h"
#include "cli/cli_visualize.h"

#define TWIST_VERSION_MAJOR 0
#define TWIST_VERSION_MINOR 1
#define TWIST_VERSION_PATCH 0

struct TWISTArgs : public argparse::Args
{
    bool &print_version = flag("v,version", "show version").set_default(false);

    BuildArgs &build_args = subcommand("build");
    PreprocessArgs &initialize_args = subcommand("initialize");
    SolveArgs &solve_args = subcommand("solve");
    ContinuationArgs &continuation_args = subcommand("continuation");
    PostprocessArgs &postprocess_args = subcommand("postprocess");
    VisualizeArgs &visualize_args = subcommand("visualize");
    InfoArgs &info_args = subcommand("info");
    ToolsArgs &tools_args = subcommand("tools");
    ImportArgs &import_args = subcommand("import");
    SimulateArgs &simulate_args = subcommand("simulate");
#ifdef TWIST_INCLUDE_LOCAL_CONTINUATION
    LocalContinuationArgs &local_continuation_args = subcommand("local-continuation");
#endif

    int run() override
    {
        if (print_version) {
            fmt::println("TWIST {}.{}.{}", TWIST_VERSION_MAJOR, TWIST_VERSION_MINOR, TWIST_VERSION_PATCH);
        }
        return 0;
    }

    void welcome() override
    {
        fmt::println("TWIST: A tool for the computation and analysis of dispersion curves in reaction-diffusion systems");
    }
};


#endif // TWIST_CLI_SETUP_H