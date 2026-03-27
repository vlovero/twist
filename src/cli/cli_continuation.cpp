#include "cli/cli_continuation.h"
#include "tools/helpers.h"

int ContinuationArgs::run()
{
    if (!quiet) {
        // this->print();
        puts("");
    }

    if (nthreads) {
        if (nthreads <= 0) {
            throw std::runtime_error("number of threads must be positive");
        }
        omp_set_num_threads(nthreads.value());
    }

    int parnum;
    std::string name;
    TWIST::Collocator collocator;
    void *lib;

    if (!std::filesystem::exists(model)) {
        throw std::runtime_error(fmt::format("file {} does not exist", model));
    }

    if (from_data.size() != 0) {
        if (from_data.size() != 2) {
            puts("--from-data keyword must have the form '--from-data <path> <index>'");
            return 1;
        }
        verify_hdf5_file(from_data[0]);

        H5::H5File h5data(from_data[0], H5F_ACC_RDONLY);
        int solution_index = atoi(from_data[1].c_str());
        int nsolutions = get_number_of_solutions(h5data);
        solution_index = ((solution_index % nsolutions) + nsolutions) % nsolutions;
        collocator = load_collocator_from_h5data_and_index(h5data, solution_index, ncol, &lib);
        collocator.setLibHandle(lib);
        h5data.close();
    }
    else {
        collocator = load_collocator_from_spec_and_init(ncol, model, parameter_set, &lib);
        collocator.setLibHandle(lib);
    }

    {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        verified_json_load(model, doc, parser);
        name = get_json_string(doc, "name");
        auto params = entry_to_list<std::string>("params", doc, false);
        auto loc = std::find(params.begin(), params.end(), parameter);
        if ((parameter == "sps") || (parameter == "spatial_period_scale")) {
            parnum = params.size() + 1;
        }
#if TEST_WAVE_NUMBER_PARAM
        else if (parameter == "wave_number") {
            parnum = params.size() + 2;
        }
#endif
        else if (loc == params.end()) {
            fmt::println("'{}' is not a valid parameter name", parameter);
            exit(1);
        }
        else {
            parnum = 1 + (loc - params.begin());
        }
    }

    collocator.solveWithAdaptation(solve_nadapt, solve_geps, solve_min_nodes, solve_max_nodes, solve_tol, solve_min_damp, solve_max_iter, !quiet);
    if (add_bumps > 0) {
        collocator.copyCollocatorDataIntoSelf(extend_to_multiwave_solution(collocator, add_bumps));
        collocator.setLibHandle(lib);
        collocator.solveWithAdaptation(solve_nadapt, solve_geps, (add_bumps + 1) * solve_min_nodes, solve_max_nodes, solve_tol, solve_min_damp, solve_max_iter, !quiet);
    }
    puts("");

    ds = std::abs(ds) * (backward ? -1 : 1);

    // void continuation_work_loop(collocator_t &collocator, int parnum, bounds, const std::string &name, const std::string &paramter_set, const std::string &tag)
    TWIST::ContinuationBounds bounds{ ds, dsmin, dsmax, parmin, parmax, solve_geps, solve_min_nodes, solve_max_nodes, true };
    continuation_work_loop(collocator, parnum, bounds, name, parameter_set, parameter, tag, prefix);
    return 0;
}

void ContinuationArgs::welcome()
{
    fmt::println("Generate a dispersion curve using numerical continuation");
}