#include "cli/cli_postprocess.h"
#include "postprocess.h"
#include "serialize.h"
#include "tools/helpers.h"

int PostprocessArgs::run()
{
    using namespace indicators;
    ProgressBar pbar1{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ COLOR_MAGENTA "[postprocessing]" COLOR_RESET " " COLOR_GREEN }, option::PostfixText{ COLOR_RESET }, option::ShowPercentage(true), option::ShowElapsedTime(true) };
    ProgressBar pbar2{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PostfixText{ COLOR_RESET }, option::ShowPercentage(true), option::ShowElapsedTime(true), option::ShowRemainingTime(true) };
    MultiProgress<ProgressBar, 2> pbars(pbar1, pbar2);
    std::vector<double> alphar, alphai, beta;
    std::string spectrum_file_path;

    const std::string prefix = ".cache/spectrum_data";
    std::filesystem::create_directories(prefix);

    // do not allow for repeated work
    // make list of files unique
    {
        std::set unique_files(files.begin(), files.end());
        files.clear();
        for (const auto &item : unique_files) {
            verify_hdf5_file(item);
            files.emplace_back(item);
        }
    }

    if (get_length) {
        for (const auto &file_name : files) {
            H5::H5File file(file_name, H5F_ACC_RDONLY);
            fmt::println("There are {} solutions in '{}'", get_number_of_solutions(file), file_name);
            for (const auto &bname : get_hopf_branch_names(file)) {
                H5::Group bg = file.openGroup(fmt::format("hopf_branches/{}", bname));
                fmt::println("  hopf branch '{}': {} solutions", bname, get_number_of_solutions(bg));
            }
        }
    }

    std::vector<std::string> steady_files;
    std::vector<std::pair<std::string, std::string>> hopf_targets;
    for (const auto &file_name : files) {
        H5::H5File f(file_name, H5F_ACC_RDONLY);
        int ftype = TWIST::FileType::continuation;
        if (f.nameExists("meta_data/file_type")) {
            H5::Group meta = f.openGroup("meta_data");
            deserialize(meta, "file_type", ftype);
        }
        if (ftype == TWIST::FileType::hopf_continuation) {
            hopf_targets.emplace_back(file_name, "");
        }
        else {
            steady_files.push_back(file_name);
            for (const auto &bname : get_hopf_branch_names(f)) {
                hopf_targets.emplace_back(file_name, fmt::format("hopf_branches/{}", bname));
            }
        }
    }

    if (spectrum) {
        if (!steady_files.empty()) {
            if (use_less_memory) {
                evaluate_spectrum_for_each_solution_inplace(steady_files, strategy);
            }
            else {
                OpenBLASThreadContext __openblas_context(1);
                evaluate_spectrum_for_each_solution_inplace_distributed(steady_files, strategy);
            }
        }
    }

    if (floquet) {
        if (!hopf_targets.empty()) {
            if (use_less_memory) {
                evaluate_floquet_for_each_solution_inplace(hopf_targets);
            }
            else {
                OpenBLASThreadContext __openblas_context(1);
                evaluate_floquet_for_each_solution_inplace_distributed(hopf_targets);
            }
        }
    }

    if (bifurcations) {
        if (std::find(which.begin(), which.end(), WhichBifPoints::all) != which.end()) {
            which.clear();
            for (int32_t i = 0; i < WhichBifPoints::_numBifPoints; i++) {
                which.emplace_back((WhichBifPoints)i);
            }
        }
        for (const std::string &file : steady_files) {
            TWIST::Collocator collocator_cache(file, 0);
            if (std::find(which.begin(), which.end(), WhichBifPoints::saddle) != which.end()) {
                locate_all_saddle_nodes_inplace(file, collocator_cache);
            }
            if (std::find(which.begin(), which.end(), WhichBifPoints::hopf) != which.end()) {
                locate_all_hopf_points_inplace_new(file);
            }
            if (std::find(which.begin(), which.end(), WhichBifPoints::branch) != which.end()) {
                locate_all_branch_points_inplace_new(file);
                // locate_all_branch_points(file, collocator_cache);
            }
            if (std::find(which.begin(), which.end(), WhichBifPoints::periodDoubling) != which.end()) {
                fmt::println("skipping period doubling because it's not implemented correctly");
                continue;
            }
        }
        for (const auto &[file, branch_path] : hopf_targets) {
            locate_all_floquet_bifurcations_inplace(file, branch_path);
        }
    }

    return 0;
}

void PostprocessArgs::welcome()
{
    fmt::println("Compute spectra and bifurcation points along dispersion curve(s) computed during continuation");
}