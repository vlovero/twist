#include "cli/cli_preprocess.h"
#include "cli/load.h"
#include "libloader.h"
#include "serialize.h"
#include <fmt/format.h>
#include <stdexcept>

int PreprocessArgs::run()
{
    double L;
    bool is_missing;
    // add checks for file path later
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.load(file);

    auto name = get_json_entry(doc, "name", false, is_missing);
    std::string lib_path = fmt::format(".cache/models/lib/{}.so", std::string_view{ name.get_c_str() });
    if (!std::filesystem::exists(lib_path)) {
        fmt::println("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", std::string_view{ name.get_c_str() });
        exit(1);
    }

    void *lib = dlopen(lib_path.c_str(), RTLD_LAZY);
    func_t func_ode = (func_t)dlsym(lib, "func_ode");
    func_t fjac_ode = (func_t)dlsym(lib, "fjac_ode");

    std::vector<double> p = { 0.0 };
    auto diffusion = load_diffusion(doc);
    auto transforms = load_transforms(doc);
    auto system = entry_to_map("system", doc);
    auto params = entry_to_list<std::string>("params", doc, false);
    size_t node = system.size();
    std::vector<double> rest_state;
    std::vector<std::string> state_vars;

    if (stim_time == std::numeric_limits<double>::infinity()) {
        stim_time = 0.1 * tf_ode;
    }

    for (const auto &[key, _] : system) {
        state_vars.emplace_back(key);
    }

    L = get_json_double(doc, "spatial_period");
    rest_state = entry_to_list<double>("rest_state", doc, false);
    if (rest_state.size() != system.size()) {
        throw std::runtime_error(fmt::format("number of states in 'rest_state' ({}) does not match number of states in 'system' ({})", rest_state.size(), system.size()));
    }
    // apply transforms to rest state
    for (const auto &[var_name, shift_and_scale] : transforms) {
        const auto [shift, scale] = shift_and_scale;
        auto where = std::find(state_vars.begin(), state_vars.end(), var_name);
        if (where == state_vars.end()) {
            continue;
        }
        ptrdiff_t i = where - state_vars.begin();
        rest_state[i] = (rest_state[i] + shift) / scale;
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

    generate_initial_guess(func_ode, fjac_ode, name.get_c_str(), parameter_set.c_str(), L, p.data(), Nx, node, rest_state.data(), tf_ode, tf_pde, { stim_time, stim_dur, stim_amp }, diffusion, method, display_start_and_end, refine_rest_state, stop_at_ode);
    return 0;
}


void PreprocessArgs::welcome()
{
    fmt::println("Generate an initial traveling wave using simulation techniques");
}