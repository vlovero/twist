#include "postprocess.h"
#include "bif_points/branch_point.h"
#include "bif_points/hopf_point.h"
#include "cli/colors.h"
#include "cli/load.h"
#include "collocator.h"
#include "colmat/tcolmat.h"
#include "fmt/base.h"
#include "fmt/core.h"
#include "indicators/multi_progress.hpp"
#include "indicators/progress_bar.hpp"
#include "indicators/setting.hpp"
#include "serialize.h"
#include "tools/helpers.h"

void insert_solution_at_index(H5::H5File &file, const int index, const int nsolutions, TWIST::Collocator &collocator, TWIST::SolutionTypes solution_type)
{
    /*
        1. shift everything from index:nsolusion up by one
            ^ do this in reverse order
            ^ also need to move spectra if they are present
            ^ set a flag if spectra are present so that this routine
              knows that the spectrum of the soltion needs to be computed too
        2. set file and index for collocator
        3. serialize with solution type
    */

    int i;
    bool do_spectra;
    std::vector<double> alphar, alphai, beta;
    H5::Group solution_group, spectrum_group;

    do_spectra = file.nameExists("0/spectrum");

    // shift everything from index:nsolusion up by one
    for (i = nsolutions - 1; index <= i; i--) {
        file.move(fmt::format("{}", i), fmt::format("{}", i + 1));
    }

    // set file and index for collocator
    collocator.setSerializationFileAndIndex(file, index);

    // serialize
    collocator.serialize("", "", "", "", "", solution_type);
    if (do_spectra) {
        // open groups that will be written to
        solution_group = file.openGroup(fmt::format("{}", index));
        spectrum_group = solution_group.createGroup("spectrum");

        // compute spectrum
        collocator.spectrum(alphar, alphai, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

        // serialize spectrum
        serialize<double, 1>(spectrum_group, "alphar", alphar.data(), { alphar.size() });
        serialize<double, 1>(spectrum_group, "alphai", alphai.data(), { alphai.size() });
        serialize<double, 1>(spectrum_group, "beta", beta.data(), { beta.size() });
    }
}

bool is_saddle_node_solution(H5::H5File &file, const int index)
{
    H5::Group solution_group;
    TWIST::SolutionTypes solution_type;

    solution_group = file.openGroup(fmt::format("{}", index));
    deserialize<TWIST::SolutionTypes>(solution_group, "stype", solution_type);

    return solution_type == TWIST::SolutionTypes::saddleNode;
}

bool is_branch_point_solution(H5::H5File &file, const int index)
{
    H5::Group solution_group;
    TWIST::SolutionTypes solution_type;

    solution_group = file.openGroup(fmt::format("{}", index));
    deserialize<TWIST::SolutionTypes>(solution_group, "stype", solution_type);

    return solution_type == TWIST::SolutionTypes::branchPoint;
}

bool is_torus_point_solution(H5::H5File &file, const int index)
{
    H5::Group solution_group;
    TWIST::SolutionTypes solution_type;

    solution_group = file.openGroup(fmt::format("{}", index));
    deserialize<TWIST::SolutionTypes>(solution_group, "stype", solution_type);

    return solution_type == TWIST::SolutionTypes::torusPoint;
}


void locate_all_branch_points(const std::string &file, TWIST::Collocator &collocator_cache)
{
    using namespace indicators;

    size_t i;
    H5::H5File h5_file(file, H5F_ACC_RDWR);
    const size_t nsolutions = get_number_of_solutions(file);
    std::string file_no_ext{ file };
    replace(file_no_ext, ".h5", "");
    std::vector<std::string_view> split_res{ split(file_no_ext, "-") };
    std::string model_name = std::string{ split(std::string{ split_res[0] }, "/").back() };
    fmt::println("name = {}", model_name);
    DEV::Matrix A;

    ProgressBar pbar(option::MaxProgress{ nsolutions - 1 }, option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[checking] " }, option::ShowPercentage(true));
    const int ndigits = std::ceil(std::log10(nsolutions - 1));
    omp_set_max_active_levels(1);
    int count = 0;
    OpenBLASThreadContext __openblas_context(1);

#pragma omp parallel for private(i)
    for (i = 0; i < nsolutions - 1; i++) {
        TWIST::Collocator collocator1, collocator2;
        bool skip = false;
#pragma omp critical
        {
            if (is_saddle_node_solution(h5_file, i) || is_saddle_node_solution(h5_file, i + 1)) {
                skip = true;
            }
            collocator1 = TWIST::Collocator(h5_file, i + 0);
            collocator2 = TWIST::Collocator(h5_file, i + 1);
        }
        if (skip) {
            continue;
        }
        auto A1 = collocator1.getBaseJacobian();
        auto A2 = collocator2.getBaseJacobian();
        A1.factor();
        A2.factor();
        auto det1 = A1.det();
        auto det2 = A2.det();
        auto dir1 = collocator1.getDirection();
        auto dir2 = collocator2.getDirection();
        bool cond1 = TWIST::sign(det1.first) != TWIST::sign(det2.first);
        bool cond2 = dir1 == dir2;
        if (cond1 && cond2) {
            fmt::println("\nBranch point between solutions {} and {}", i, i + 1);
            // bool converged = locate_saddle_node_newton(&collocator1);
            bool converged = false;
            if (converged) {
                collocator_cache.copyCollocatorDataIntoSelf(collocator1);
#pragma omp critical
                {
                    collocator_cache.serialize(model_name, std::string{ split_res[1] }, std::string{ split_res[2] }, std::string{ split_res[3] } + "-bifpoints", ".cache/continuation_data", TWIST::SolutionTypes::branchPoint, TWIST::FileType::bifurcation_points);
                }
            }
            else {
                // converged = locate_saddle_node_newton(&collocator2);
                bool converged = false;
                if (converged) {
                    collocator_cache.copyCollocatorDataIntoSelf(collocator2);
#pragma omp critical
                    {
                        collocator_cache.serialize(model_name, std::string{ split_res[1] }, std::string{ split_res[2] }, std::string{ split_res[3] } + "-bifpoints", ".cache/continuation_data", TWIST::SolutionTypes::branchPoint, TWIST::FileType::bifurcation_points);
                    }
                }
            }
        }
#pragma omp critical
        {
            count += 1;
            pbar.set_option(option::PostfixText{ fmt::format("{0:{2}d}/{1:{2}d}", count, nsolutions - 1, ndigits) });
            pbar.tick();
        }
    }
    puts("");
}


// NEW "INPLACE" versions of functions above
void evaluate_spectrum_for_each_solution_inplace(const std::vector<std::string> &files, const TWIST::SpectrumStrategy strategy)
{
    using namespace indicators;
    ProgressBar pbar1{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[postprocessing] " }, option::ShowPercentage(true), option::ShowElapsedTime(true) };
    ProgressBar pbar2{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::ShowPercentage(true), option::ShowElapsedTime(true), option::ShowRemainingTime(true) };
    MultiProgress<ProgressBar, 2> pbars(pbar1, pbar2);

    ptrdiff_t i, j, jj, nsolutions;
    H5::Group solution_group, spectrum_group;
    std::vector<double> alphar, alphai, beta;
    void *lib = nullptr;
    char answer;
    bool computing_spectra[files.size()];

    // first check for existing spectra
    for (i = 0; i < (ptrdiff_t)files.size(); i++) {
        H5::H5File file(files[i], H5F_ACC_RDWR);
        computing_spectra[i] = true;
        if (!file.nameExists("0/spectrum")) {
            continue;
        }

        fmt::println("The spectra have already been computed for the file '{}'", files[i]);
        fflush(stdin);
        fmt::print("Recompute spectra? [y/n] ");
        fflush(stdout);
        if (scanf(" %c", &answer) != 1) {
            fmt::println("invaid input...");
            return;
        }

        if ((answer != 'y') && (answer != 'Y')) {
            computing_spectra[i] = false;
            continue;
        }

        nsolutions = get_number_of_solutions(file);
        for (jj = 0; jj < nsolutions; jj++) {
            const auto dset = fmt::format("{}/spectrum", jj);
            if (file.exists(dset)) {
                file.unlink(dset);
            }
        }
    }

    for (i = 0; i < (ptrdiff_t)files.size(); i++) {
        pbar1.set_option(option::PostfixText{ fmt::format(" '{}'", files[i]) });
        if (!computing_spectra[i]) {
            continue;
        }
        H5::H5File file(files[i], H5F_ACC_RDWR);

        nsolutions = get_number_of_solutions(file);
        for (j = 0; j < nsolutions; j++) {
            pbar2.set_option(option::MaxProgress{ nsolutions });
            pbar2.set_option(option::PrefixText{ fmt::format("[soln.{:4d}/{:4d}] ", j, nsolutions) });

            // open/create groups
            solution_group = file.openGroup(fmt::format("{}", j));
            spectrum_group = solution_group.createGroup("spectrum");

            // compute spectrum
            TWIST::Collocator collocator(file, j);
            collocator.setLibHandle(lib);
            collocator.spectrum(alphar, alphai, beta, strategy, false, NULL);

            // dump data
            serialize<double, 1>(spectrum_group, "alphar", alphar.data(), { alphar.size() });
            serialize<double, 1>(spectrum_group, "alphai", alphai.data(), { alphai.size() });
            serialize<double, 1>(spectrum_group, "beta", beta.data(), { beta.size() });

            // update progress bar
            pbars.set_progress<1lu>((size_t)j + 1);
        }
        pbar2.set_option(option::PrefixText{ fmt::format("[soln.{:4d}/{:4d}] ", nsolutions, nsolutions) });
        pbars.set_progress<0lu>((size_t)(100 * (double)(i + 1) / files.size()));
    }
}

void evaluate_spectrum_for_each_solution_inplace_distributed(const std::vector<std::string> &files, const TWIST::SpectrumStrategy strategy)
{
    using namespace indicators;
    ProgressBar pbar1{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[postprocessing]    " }, option::ShowPercentage(true), option::ShowElapsedTime(true) };
    ProgressBar pbar2{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::ShowPercentage(true), option::ShowElapsedTime(true), option::ShowRemainingTime(true) };
    MultiProgress<ProgressBar, 2> pbars(pbar1, pbar2);

    ptrdiff_t i, j, jj, nsolutions;
    int count;
    char answer;
    bool computing_spectra[files.size()];
    // don't use multithreading inside openblas
    OpenBLASThreadContext __openblas_context(1);
    // incase openblas using openmp make sure multithreading
    // limited to this scope
    omp_set_max_active_levels(1);

    // first check for existing spectra
    for (i = 0; i < (ptrdiff_t)files.size(); i++) {
        H5::H5File file(files[i], H5F_ACC_RDWR);
        computing_spectra[i] = true;
        if (!file.nameExists("0/spectrum")) {
            continue;
        }

        fmt::println("The spectra have already been computed for the file '{}'", files[i]);
        fflush(stdin);
        fmt::print("Recompute spectra? [y/n] ");
        fflush(stdout);
        if (scanf(" %c", &answer) != 1) {
            fmt::println("invaid input...");
            return;
        }

        if ((answer != 'y') && (answer != 'Y')) {
            computing_spectra[i] = false;
            continue;
        }

        nsolutions = get_number_of_solutions(file);
        for (jj = 0; jj < nsolutions; jj++) {
            const auto dset = fmt::format("{}/spectrum", jj);
            if (file.exists(dset)) {
                file.unlink(dset);
            }
        }
    }

    pbar1.set_option(option::MaxProgress{ files.size() });
    pbar2.set_option(option::PrefixText{ fmt::format("[computing spectra] ", nsolutions, nsolutions) });
    for (i = 0; i < (ptrdiff_t)files.size(); i++) {
        pbar1.set_option(option::PostfixText{ fmt::format(" '{}'", files[i]) });
        if (!computing_spectra[i]) {
            continue;
        }
        hid_t file_id = H5Fopen(files[i].c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0) {
            continue;
        }

        {
            H5::H5File tmp(file_id);
            nsolutions = get_number_of_solutions(tmp);
        }

        count = 0;
        pbar2.set_option(option::MaxProgress{ nsolutions });
        const int ndigits = std::ceil(std::log10(nsolutions));

#pragma omp parallel for private(j) schedule(dynamic, 1)
        for (j = 0; j < nsolutions; j++) {
            std::optional<TWIST::Collocator> collocator;
            std::vector<double> alphar, alphai, beta;

#pragma omp critical
            {
                collocator.emplace(file_id, j);
            }

            // compute spectrum
            collocator->spectrum(alphar, alphai, beta, strategy, false, NULL);

#pragma omp critical
            {
                std::string sol_name = fmt::format("{}", j);
                hid_t solution_group_id = H5Gopen2(file_id, sol_name.c_str(), H5P_DEFAULT);
                hid_t spectrum_group_id = H5Gcreate2(solution_group_id, "spectrum", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

                // dump data
                {
                    H5::Group spectrum_group(spectrum_group_id);
                    serialize<double, 1>(spectrum_group, "alphar", alphar.data(), { alphar.size() });
                    serialize<double, 1>(spectrum_group, "alphai", alphai.data(), { alphai.size() });
                    serialize<double, 1>(spectrum_group, "beta", beta.data(), { beta.size() });
                }

                // close groups
                H5Gclose(spectrum_group_id);
                H5Gclose(solution_group_id);

                collocator.reset();

                // update progress bar
                count += 1;
                pbar2.set_option(option::PostfixText{ fmt::format("{0:{2}d}/{1:{2}d}", count, nsolutions, ndigits) });
                pbars.tick<1lu>();
            }
        }

        H5Fclose(file_id);
        pbars.tick<0lu>();
    }
}


void locate_all_saddle_nodes_inplace(const std::string &file, TWIST::Collocator &)
{
    size_t i;
    bool converged;
    size_t nsolutions = get_number_of_solutions(file);
    H5::H5File h5_file(file, H5F_ACC_RDWR);
    std::string file_no_ext{ file };
    replace(file_no_ext, ".h5", "");
    std::vector<std::string_view> split_res{ split(file_no_ext, "-") };
    std::string model_name = std::string{ split(std::string{ split_res[0] }, "/").back() };
    fmt::println("name = {}", model_name);

    for (i = 0; i < nsolutions - 1; i++) {
        if (is_saddle_node_solution(h5_file, i + 0) || (is_saddle_node_solution(h5_file, i + 1))) {
            // skip already computed saddle nodes
            continue;
        }
        TWIST::Collocator collocator1(h5_file, i + 0);
        TWIST::Collocator collocator2(h5_file, i + 1);

        if (collocator1.getDirection() != collocator2.getDirection()) {
            fmt::println("Saddle node between solutions {} and {}", i, i + 1);
            // continue;
            converged = collocator1.locateSaddleNode();
            if (converged) {
                insert_solution_at_index(h5_file, i + 1, nsolutions, collocator1, TWIST::SolutionTypes::saddleNode);
                nsolutions++;
                i++;
                continue;
            }
            // first one failed -> try second
            converged = collocator2.locateSaddleNode();
            if (converged) {
                insert_solution_at_index(h5_file, i + 1, nsolutions, collocator2, TWIST::SolutionTypes::saddleNode);
                nsolutions++;
                i++;
                continue;
            }
        }
    }
}

bool is_between(double x, double a, double b)
{
    if (b < a) {
        std::swap(a, b);
    }
    // fmt::println("checking if {} is between ({}, {})", x, a, b);
    return (a <= x) && (x <= b);
}


void locate_all_hopf_points_inplace_new(const std::string &file)
{
    size_t nsolutions, i, j, N;
    bool converged;
    int npairs1, npairs2, nunstable1, nunstable2, count;
    double parmin, parmax, tmp1, tmp2;
    double *alphar = nullptr, *alphai = nullptr, *beta = nullptr;
    std::vector<std::tuple<size_t, double, TWIST::Collocator>> all_computed;
    std::vector<std::complex<double>> unstable1, unstable2;

    // the min and max nodes won't be used in the solver so set them to whatever
    TWIST::ContinuationBounds bounds = {
        .ds = 1e-3,
        .dsmin = 1e-12,
        .dsmax = 1,
        .parmin = -std::numeric_limits<double>::infinity(),
        .parmax = +std::numeric_limits<double>::infinity(),
        .geps = 1e-12,
        .min_nodes_adapt = 20,
        .max_nodes_adapt = 20,
        .allow_mesh_adaptation = 0,
    };

    // for now read only until I implement insert function
    H5::H5File solution_data(file, H5F_ACC_RDWR);
    nsolutions = get_number_of_solutions(solution_data);

    if (!solution_data.nameExists("0/spectrum")) {
        fmt::println("skipping file '{0}' because no spectrum data was found\n"
                     "make sure to run `twist postprocess {0} -s` before (or with) bifurcation detection",
        file);
        return;
    }

    for (i = 1; i < nsolutions; i++) {
        if (is_torus_point_solution(solution_data, i) || is_torus_point_solution(solution_data, i - 1)) {
            continue;
        }
        H5::Group group1 = solution_data.openGroup(fmt::format("{}/spectrum", i - 0));
        H5::Group group2 = solution_data.openGroup(fmt::format("{}/spectrum", i - 1));

        deserialize(group1, "alphar", alphar, N);
        deserialize(group1, "alphai", alphai, N);
        deserialize(group1, "beta", beta, N);
        Hopf::get_unstable_pairs(unstable1, alphar, alphai, beta, N);
        npairs1 = Hopf::count_complex_pairs(alphar, alphai, beta, N);
        nunstable1 = Hopf::count_unstable(alphar, alphai, beta, N);

        deserialize(group2, "alphar", alphar, N);
        deserialize(group2, "alphai", alphai, N);
        deserialize(group2, "beta", beta, N);
        Hopf::get_unstable_pairs(unstable2, alphar, alphai, beta, N);
        npairs2 = Hopf::count_complex_pairs(alphar, alphai, beta, N);
        nunstable2 = Hopf::count_unstable(alphar, alphai, beta, N);

        bool cond1 = npairs1 != npairs2;
        bool cond2 = nunstable1 != nunstable2;
        bool cond3 = (nunstable1 & 1) == (nunstable2 & 1);
        bool hopf_occurred = cond1 && cond2 && cond3;

        if (hopf_occurred) {
            assert((unstable1.size() & 1) == 0);
            assert((unstable2.size() & 1) == 0);

            fmt::println("Hopf bifurcation detected bettween solutions {} and {}", i - 1, i);
            fmt::println("went from {} to {} pairs", npairs2, npairs1);
            TWIST::Collocator collocator1(solution_data, i - 0);
            TWIST::Collocator collocator2(solution_data, i - 1);
            parmin = collocator1.getContinuationParameterValue();
            parmax = collocator2.getContinuationParameterValue();
            if (parmax < parmin) {
                std::swap(parmin, parmax);
            }
            bounds.parmin = parmin;
            bounds.parmax = parmax;
            std::sort(unstable1.begin(), unstable1.end(), [](const std::complex<double> &a, const std::complex<double> &b) { return a.real() < b.real(); });
            std::sort(unstable2.begin(), unstable2.end(), [](const std::complex<double> &a, const std::complex<double> &b) { return a.real() < b.real(); });
            count = 0;

            if (npairs2 < npairs1) {
                for (j = 0; j < unstable1.size(); j += 2) {
                    TWIST::Collocator collocator(solution_data, i);
                    bounds.ds = 1e-3;
                    converged = collocator.locateTorusPoint(bounds, unstable1, j);
                    if (converged) {
                        if (is_between(collocator.getContinuationParameterValue(), parmin, parmax)) {
                            count++;
                            all_computed.push_back({ i, unstable1[j].imag(), collocator });
                        }
                    }
                    if ((npairs1 - npairs2) <= count) {
                        break;
                    }
                }
            }
            else {
                for (j = 0; j < unstable2.size(); j += 2) {
                    TWIST::Collocator collocator(solution_data, i - 1);
                    bounds.ds = 1e-3;
                    converged = collocator.locateTorusPoint(bounds, unstable2, j);
                    if (converged) {
                        if (is_between(collocator.getContinuationParameterValue(), parmin, parmax)) {
                            count++;
                            all_computed.push_back({ i, unstable2[j].imag(), collocator });
                        }
                    }
                    if ((npairs2 - npairs1) <= count) {
                        break;
                    }
                }
            }

            // don't insert yet, sort based on parameter value, reverse if needed, then insert
            std::sort(all_computed.begin(), all_computed.end(), [](const auto &a, const auto &b) { return std::get<2>(a).getContinuationParameterValue() < std::get<2>(b).getContinuationParameterValue(); });
            tmp1 = collocator1.getContinuationParameterValue();
            tmp2 = collocator2.getContinuationParameterValue();
            if (tmp2 > tmp1) {
                std::reverse(all_computed.begin(), all_computed.end());
            }
            for (auto &[_1, _2, collocator] : all_computed) {
                insert_solution_at_index(solution_data, i, nsolutions, collocator, TWIST::SolutionTypes::torusPoint);
                nsolutions++;
                i++;
            }
            all_computed.clear();
            if (count < std::abs(npairs1 - npairs2)) {
                fmt::print(COLOR_YELLOW "[Warning]: Only {} of the {} bifurcation points were able to be found. " COLOR_RESET, count, std::abs(npairs1 - npairs2));
                fmt::print(COLOR_YELLOW "This is likely due to discretization errors between the two successive meshes. " COLOR_RESET);
                fmt::println(COLOR_YELLOW "A likely fix would be to use a smaller step size in this region during continuation. " COLOR_RESET);
            }
        }
    }

    fmt::println("found {} crossings", all_computed.size());

    if (alphar) {
        free(alphar);
    }
    if (alphai) {
        free(alphai);
    }
    if (beta) {
        free(beta);
    }
}


void locate_all_branch_points_inplace_new(const std::string &file)
{
    /*
    loop through all spectra pairs like in HB
    skip any that are already marked as saddle/branch
    ??if # positive real increases by odd # then branch point??
    compare eigenfunctions:
        for positive (less):
            for positive (more):
                diffs[j] = ||more - less||
            skip[i] = diffs.argmin()

        for index in indices\skip:
            ds = ds0
            k = eigval
            v = eigvec
            while abs(ds) > dstol and abs(k0) > ktol:
                backup = collocator
                delta = sign(ds) * sqeps * (1 + abs(ds))
                solve/step(ds)
                k1 = refine_eig(k, vec)
                delta = delta * k / (k - k1)
                ds += delta
                ?? k = k1 ??
                collocator = backup

    */

    size_t i, N;
    bool converged;
    size_t nsolutions;
    H5::H5File solution_data(file, H5F_ACC_RDWR);
    double *alphar = nullptr, *alphai = nullptr, *beta = nullptr;
    std::vector<double> unstable1, unstable2;
    int nu1, nu2;

    nsolutions = get_number_of_solutions(solution_data);

    if (!solution_data.nameExists("0/spectrum")) {
        fmt::println("skipping file '{0}' because no spectrum data was found\n"
                     "make sure to run `twist postprocess {0} -s` before (or with) bifurcation detection",
        file);
        return;
    }

    for (i = 0; i < nsolutions - 1; i++) {
        if (is_saddle_node_solution(solution_data, i + 0) || (is_saddle_node_solution(solution_data, i + 1))) {
            // skip already computed saddle nodes
            continue;
        }
        else if (is_branch_point_solution(solution_data, i + 0) || (is_branch_point_solution(solution_data, i + 1))) {
            // skip already computed branch points
            continue;
        }
        H5::Group group1 = solution_data.openGroup(fmt::format("{}/spectrum", i - 0));
        H5::Group group2 = solution_data.openGroup(fmt::format("{}/spectrum", i + 1));

        deserialize(group1, "alphar", alphar, N);
        deserialize(group1, "alphai", alphai, N);
        deserialize(group1, "beta", beta, N);
        BranchPoint::shift_to_origin(alphar, alphai, beta, N);
        BranchPoint::get_real_unstable(unstable1, alphar, alphai, beta, N);

        deserialize(group2, "alphar", alphar, N);
        deserialize(group2, "alphai", alphai, N);
        deserialize(group2, "beta", beta, N);
        BranchPoint::shift_to_origin(alphar, alphai, beta, N);
        BranchPoint::get_real_unstable(unstable2, alphar, alphai, beta, N);

        nu1 = unstable1.size();
        nu2 = unstable2.size();

        if (std::abs(nu1 - nu2) == 1) {
            fmt::println("branch point between solutions {} and {}", i, i + 1);
            continue;
            TWIST::Collocator collocator(solution_data, nu1 > nu2 ? i : i + 1);
            // TODO: replace this with correct logic later!!
            double &k = nu1 > nu2 ? unstable1[0] : unstable2[0];
            converged = collocator.locateBranchPoint(k);
            if (converged) {
                insert_solution_at_index(solution_data, i + 1, nsolutions, collocator, TWIST::SolutionTypes::branchPoint);
                nsolutions++;
                i++;
                continue;
            }
        }
    }
}