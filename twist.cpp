#include "argparse/argparse.hpp"
#include "cli/cli.h"
#include "fmt/base.h"
#include <stdexcept>

int main(int argc, char **argv)
{
    try {
        set_program_args(argc, argv);
        auto args = argparse::parse<TWISTArgs>(argc, argv, true);
        if (argc == 1) {
            args.welcome();
            return 0;
        }
        else if (args.print_version) {
            return args.run();
        }
        return args.run_subcommands();
    }
    catch (const std::runtime_error &e) {
        fmt::println(stderr, "{}", e.what());
        return 1;
    }
}
