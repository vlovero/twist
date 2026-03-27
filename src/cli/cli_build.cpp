#include "cli/cli_build.h"

int BuildArgs::run()
{
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error(fmt::format("file {} does not exist", file));
    }
    compile_model(file.c_str(), compiler.c_str(), source_only);
    return 0;
}

void BuildArgs::welcome()
{
    fmt::println("Generate source code and compile from a given spec (JSON) file");
}