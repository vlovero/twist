#include "cli/cli_local_continuation.h"
#include "libloader.h"

extern void local_continuation_loop(TWIST::ContinuationBounds &bounds, void *handle, const int pindex, const ptrdiff_t ny, const double *y, const ptrdiff_t np, const double *p, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix);


static std::vector<int> indices_to_insert(const diffusion_t &diffusion)
{
    int offset;
    std::vector<int> to_delete;
    to_delete.reserve(diffusion.size());

    offset = 1;
    for (const auto &[k, _] : diffusion) {
        to_delete.push_back(k + offset);
        offset++;
    }
    return to_delete;
}

static void load_local(const std::string &spec, const std::string &parameter, const std::string &parameter_set, const double c0, std::string &name, void *&lib, std::vector<double> &p, std::vector<double> &rest_state, int &pindex)
{
    bool is_missing;
    size_t node;
    diffusion_t diffusion;
    std::string lib_path;
    std::vector<std::string> params;
    std::vector<int> indices;
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.load(spec);

    node = entry_to_map("system", doc).size();
    name = get_json_entry(doc, "name", false, is_missing);
    lib_path = fmt::format(".cache/models/lib/{}.so", name);
    if (!std::filesystem::exists(lib_path)) {
        fmt::println("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", name);
        exit(1);
    }
    lib = dlopen(lib_path.c_str(), RTLD_LAZY);
    params = entry_to_list<std::string>("params", doc, false);
    rest_state = entry_to_list<double>("rest_state", doc, false);
    if (rest_state.size() != node) {
        throw std::runtime_error(fmt::format("number of states in 'rest_state' ({}) does not match number of states in 'system' ({})", rest_state.size(), node));
    }

    diffusion = load_diffusion(doc);
    indices = indices_to_insert(diffusion);
    for (const auto &index : indices) {
        rest_state.insert(rest_state.begin() + index, 0.0);
    }

    p.clear();
    p.emplace_back(c0);

    {
        auto loc = std::find(params.begin(), params.end(), parameter);
        if ((parameter == "sps") || (parameter == "spatial_period_scale")) {
            pindex = params.size() + 1;
        }
#if TEST_WAVE_NUMBER_PARAM
        else if (parameter == "wave_number") {
            pindex = params.size() + 2;
        }
#endif
        else if (loc == params.end()) {
            fmt::println("'{}' is not a valid parameter name", parameter);
            exit(1);
        }
        else {
            pindex = 1 + (loc - params.begin());
        }
    }

    {
        auto parameter_sets = get_json_object(doc, "parameter_sets");
        auto pset_obj_ref = get_json_object(parameter_sets.value(), parameter_set.c_str());
        if (pset_obj_ref.size() != params.size()) {
            throw std::runtime_error(fmt::format("size of '{}' parameter set ({}) does not match number of model paramters ({})", parameter_set, pset_obj_ref.size(), params.size()));
        }
#ifdef TEST_WAVE_NUMBER_PARAM
        p.resize(params.size() + 3);
#else
        p.resize(params.size() + 2);
#endif
        for (auto [pname, value] : pset_obj_ref) {
            auto where = std::find(params.begin(), params.end(), pname);
            if (!value.is_double()) {
                throw std::runtime_error(fmt::format("provided value for paramter_sets/{}/{} is not a number", parameter_set.c_str(), pname));
            }
            if (where != params.end()) {
                p[1 + where - params.begin()] = value;
            }
            else {
                throw std::runtime_error(fmt::format("parameter '{}' does not match any paramters from ({})", pname, fmt::join(params, ", ")));
            }
        }
        // default scaling of 1
        p[params.size() + 1] = 1.0;
#ifdef TEST_WAVE_NUMBER_PARAM
        p[params.size() + 2] = 1.0;
#endif
    }
}

int LocalContinuationArgs::run()
{
    int pindex;
    std::string name;
    void *lib;
    std::vector<double> p, rest_state;

    if (from_data.size() != 0) {
        throw std::runtime_error("--from-data not implemented");
        // if (from_data.size() != 2) {
        //     puts("--from-data keyword must have the form '--from-data <path> <index>'");
        //     return 1;
        // }
        // H5::H5File h5data(from_data[0], H5F_ACC_RDONLY);
        // int solution_index = atoi(from_data[1].c_str());
        // int nsolutions = get_number_of_solutions(h5data);
        // solution_index = ((solution_index % nsolutions) + nsolutions) % nsolutions;
        // h5data.close();
    }
    else {
        load_local(model, parameter, parameter_set, wave_speed, name, lib, p, rest_state, pindex);
    }

    TWIST::ContinuationBounds bounds = { .ds = ds, .dsmin = dsmin, .dsmax = dsmax, .parmin = parmin, .parmax = parmax, .geps = 0.0, .min_nodes_adapt = 0, .max_nodes_adapt = 0, .allow_mesh_adaptation = 0 };
    local_continuation_loop(bounds, lib, pindex, rest_state.size(), rest_state.data(), p.size(), p.data(), name, parameter_set, parameter, tag, prefix);

    return 0;
}
void LocalContinuationArgs::welcome()
{
    fmt::println("Generate a bifurcation diagram for the local dynamics (in traveling wave coordinates) using numerical continuation");
}