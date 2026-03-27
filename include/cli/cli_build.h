#pragma once

#include "cli/cli_common.h"


struct BuildArgs : public argparse::Args
{
    std::string &file = arg("file", "path to JSON spec file of the model to be compiled");
    std::string &compiler = kwarg("c,compiler", "set the compiler that will compiler generated model code").set_default("g++");
    bool &source_only = flag("s,source-only", "Only create the source file which can be compiled manually by user").set_default(false);

    void welcome() override;
    int run() override;
};
