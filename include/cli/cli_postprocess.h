#pragma once

#include "cli/cli_common.h"
#include <vector>

enum WhichBifPoints : int32_t
{
    all = 0,
    saddle,
    branch,
    periodDoubling,
    hopf,
    _numBifPoints
};

struct PostprocessArgs : public argparse::Args
{
    std::vector<std::string> &files = arg("file", "continuation data file(s) to do postprocessing on").multi_argument();
    bool &spectrum = flag("s,spectrum", "compute spectrum for given file(s)").set_default(false);
    std::vector<WhichBifPoints> &which = kwarg("w,which", "which bifurcation points too look for").multi_argument().set_default(std::vector<WhichBifPoints>({ WhichBifPoints::all }));
    bool &bifurcations = flag("b,bifurcations", "Locate bifurcation points for given file(s)").set_default(false);
    bool &get_length = flag("l,length", "compute how many solutions are stored in the given file(s)").set_default(false);
    TWIST::SpectrumStrategy &strategy = kwarg("spectrum-strategy", "Which algorithm to compute the spectrum").set_default(TWIST::SpectrumStrategy::shiftAndInvert);
    bool &use_less_memory = flag("less-memory", "Compute one spectrum at a time instead of many at once (assumes -s/--spectrum flag)").set_default(false);

    int run() override;
    void welcome() override;
};
