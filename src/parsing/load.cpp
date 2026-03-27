#include "cli/load.h"
#include "H5File.h"
#include "H5Fpublic.h"
#include "H5Group.h"
#include "collocator.h"
#include "fmt/format.h"
#include "libloader.h"
#include "serialize.h"
#include "simdjson.h"
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>


json_res_t get_json_entry(const json_elm_t &doc, const std::string_view &name, const bool allow_missing, bool &is_missing)
{
    auto name_ref = doc[name];
    const auto code = name_ref.error();

    is_missing = false;

    if (code == simdjson::error_code::NO_SUCH_FIELD) {
        is_missing = true;
        if (!allow_missing) {
            throw std::runtime_error(fmt::format("entry '{}' must be specified", name));
        }
    }
    else if (code != simdjson::error_code::SUCCESS) {
        std::ostringstream stream;
        stream << code;
        throw std::runtime_error(fmt::format("entry '{}' resulted in error: {}", name, stream.str()));
    }

    return name_ref;
}


json_array_t get_json_array(const json_res_t &ref, const std::string_view &name)
{
    auto array_ref = ref.get_array();
    auto code = array_ref.error();
    if (code == simdjson::error_code::INCORRECT_TYPE) {
        throw std::runtime_error(fmt::format("entry '{}' is not an array entry", name));
    }
    return array_ref;
}


json_object_t get_json_object(const json_res_t &ref, const std::string_view &name)
{
    auto object_ref = ref.get_object();
    auto code = object_ref.error();
    if (code == simdjson::error_code::INCORRECT_TYPE) {
        throw std::runtime_error(fmt::format("entry '{}' is not a map/dict entry", name));
    }
    return object_ref;
}


json_object_t get_json_object(const json_elm_t &ref, const std::string_view &name)
{
    bool is_missing;
    auto name_ref = get_json_entry(ref, name, false, is_missing);
    return get_json_object(name_ref, name);
}


double get_json_double(const json_elm_t &ref, const std::string_view &name)
{
    bool is_missing;
    auto entry_ref = get_json_entry(ref, name, false, is_missing);
    auto double_ref = entry_ref.get_double();
    auto code = double_ref.error();

    if (code == simdjson::error_code::INCORRECT_TYPE) {
        throw std::runtime_error(fmt::format("entry '{}' is not a number", name));
    }

    return double_ref.value();
}

std::string_view get_json_string(const json_elm_t &ref, const std::string_view &name)
{
    bool is_missing;
    auto entry_ref = get_json_entry(ref, name, false, is_missing);
    auto string_ref = entry_ref.get_string();
    auto code = string_ref.error();

    if (code == simdjson::error_code::INCORRECT_TYPE) {
        throw std::runtime_error(fmt::format("entry '{}' is not a string", name));
    }

    return string_ref.value();
}


void replace(std::string &text, const std::string &from, const std::string &to)
{
    if (from.empty()) {
        return;
    }
    for (auto at = text.find(from, 0); at != std::string::npos; at = text.find(from, at + to.length())) {
        text.replace(at, from.length(), to);
    }
}


std::vector<std::pair<int, double>> load_diffusion(const simdjson::dom::element &doc)
{
    std::vector<std::pair<int, double>> fields;

    auto system = entry_to_map("system", doc);
    auto diffusion_map = entry_to_map<double>("diffusion", doc);
    for (const auto &[var, coefficient] : diffusion_map) {
        size_t index = std::distance(system.begin(), system.find(var));
        if (index == system.size()) {
            throw std::runtime_error(fmt::format("found diffusion coefficent for {1} but {1} is not present in system variables", var, var));
        }
        if (coefficient != 0.0) {
            fields.emplace_back(index, coefficient);
        }
    }
    std::sort(fields.begin(), fields.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first < b.first; });
    return fields;
}


std::tuple<std::string, func_t, func_t, pjac_t, size_t, std::vector<double>, std::vector<std::pair<int, double>>, double, void *> load_base_twave_data_from_spec(const std::string &file, const std::string &parameter_set)
{
    double L;
    bool is_missing;
    // add checks for file path later
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    verified_json_load(file, doc, parser);

    auto name = get_json_entry(doc, "name", false, is_missing);
    std::string lib_path = fmt::format(".cache/models/lib/{}.so", std::string_view{ name.get_c_str() });
    if (!std::filesystem::exists(lib_path)) {
        fmt::println("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", std::string_view{ name.get_c_str() });
        exit(1);
    }

    void *lib = dlopen(lib_path.c_str(), RTLD_LAZY);
    func_t func = (func_t)dlsym(lib, "func");
    func_t fjac = (func_t)dlsym(lib, "fjac");
    pjac_t pjac = (pjac_t)dlsym(lib, "pjac");

    std::vector<double> p = { 0.0 };
    auto diffusion = load_diffusion(doc);
    size_t node = entry_to_map("system", doc).size();
    auto params = entry_to_list<std::string>("params", doc, false);
    std::vector<double> rest_state;

    rest_state = entry_to_list<double>("rest_state", doc, false);
    if (rest_state.size() != node) {
        throw std::runtime_error(fmt::format("number of states in 'rest_state' ({}) does not match number of states in 'system' ({})", rest_state.size(), node));
    }
    L = get_json_double(doc, "spatial_period");

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

    return { std::string{ name.get_c_str() }, func, fjac, pjac, node + diffusion.size(), p, diffusion, L, lib };
}


TWIST::Collocator load_collocator_from_spec_and_init(const size_t ncol, const std::string &file, const std::string &parameter_set, void **lib)
{
    double *t = nullptr;
    double *y = nullptr;
    size_t nnodes_init;
    double wave_speed;
    auto [name, func, fjac, pjac, node, p, diffusion, L, lib_handle] = load_base_twave_data_from_spec(file, parameter_set);
    // deserialize data generated from preprocessing
    const std::string init_data_path{};
    H5::H5File h5_file(fmt::format(".cache/models/init_data/{}-{}.h5", name, parameter_set), H5F_ACC_RDONLY);
    H5::Group group = h5_file.openGroup("data");

    deserialize(group, "nnodes", nnodes_init);
    deserialize(group, "wave_speed", wave_speed);
    deserialize(group, "t", t);
    deserialize(group, "y", y);

    // set this here for debugging
    p[0] = wave_speed ? wave_speed : 0.014658;
    *lib = lib_handle;

    if ((ncol < 1) || (ncol > (GL_methods.size() - 1))) {
        fmt::println("Error: Invalid number of collocation points (1-{} supported)", GL_methods.size() - 1);
        exit(1);
    }

    TWIST::Collocator collocator(ncol, func, fjac, pjac, node, nnodes_init, t, y, p.size(), p.data(), diffusion, 0.5 * L);

    free(t);
    t = nullptr;
    free(y);
    y = nullptr;

    return collocator;
}


void replace_existing_starting_solution(TWIST::Collocator &collocator, const std::string &file, const std::string &parameter_set)
{
    double *y;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    verified_json_load(file, doc, parser);

    auto name = doc["name"];
    H5::H5File h5_file(fmt::format(".cache/models/init_data/{}-{}.h5", std::string_view{ name.get_c_str() }, parameter_set), H5F_ACC_RDWR);
    H5::Group data_group = h5_file.openGroup("data");
    data_group.unlink("nnodes");
    data_group.unlink("wave_speed");
    data_group.unlink("t");
    data_group.unlink("y");

    y = (double *)malloc(collocator.getNNodes() * collocator.getNode() * sizeof(double));
    collocator.denseOutput(collocator.t(), collocator.getNNodes(), y);

    serialize(data_group, "nnodes", collocator.getNNodes());
    serialize(data_group, "wave_speed", collocator.waveSpeed());
    serialize<double, 1>(data_group, "t", collocator.t(), { collocator.getNNodes() });
    serialize<double, 1>(data_group, "y", y, { collocator.getNNodes() * collocator.getNode() });

    free(y);
}


TWIST::Collocator load_collocator_from_h5data_and_index(const H5::H5File &h5data, const int solution_index, const size_t ncol_wanted, void **handle)
{
    size_t ncol;
    {
        H5::Group group = h5data.openGroup("basic_info");
        deserialize(group, "nstages", ncol);
    }
    TWIST::Collocator collocator(h5data, solution_index);

    if ((ncol == ncol_wanted) || (ncol_wanted == (size_t)(-1))) {
        *handle = collocator.getLibHandle();
        collocator.clearLibHandle();
        return collocator;
    }
    else {
        if ((ncol < 1) || (ncol > (GL_methods.size() - 1))) {
            fmt::println("Error: Invalid number of collocation points (1-{} supported)", GL_methods.size() - 1);
            exit(1);
        }
        if (ncol_wanted < ncol) {
            fmt::println(COLOR_YELLOW "[WARNING] : Going from {} to {} collocation points. Decreasing # collocation points might lead to convergence failure" COLOR_RESET, ncol, ncol_wanted);
        }
        const size_t nnodes = collocator.getNNodes();
        const size_t node = collocator.getNode();
        const size_t np = collocator.getNParam();
        auto diffusion = collocator.getDiffusion();
        const double L = collocator.unscaledSpatialPeriod();

        // only these two need to get copied
        // these are aliases that get freed after starting collocator_t gets
        // destructed before constructor of new collocator gets called
        const double *p_old = collocator.p();
        double *p = (double *)malloc(np * sizeof(double));
        memcpy(p, p_old, np * sizeof(double));

        const double *t_old = collocator.t();
        double *t = (double *)malloc(nnodes * sizeof(double));
        memcpy(t, t_old, nnodes * sizeof(double));

        const double *y = collocator.y();

        void *lib = collocator.getLibHandle();
        collocator.clearLibHandle();
        func_t func = (func_t)dlsym(lib, "func");
        func_t fjac = (func_t)dlsym(lib, "fjac");
        pjac_t pjac = (pjac_t)dlsym(lib, "pjac");
        double *y0 = (double *)malloc(((nnodes - 1) * (ncol_wanted + 1) + 1) * node * sizeof(double));

        // extract y_k's from current collocator
        for (size_t i = 0; i < nnodes; i++) {
            for (size_t j = 0; j < node; j++) {
                y0[i * node + j] = y[i * (ncol + 1) * node + j];
            }
        }
        TWIST::Collocator collocator_wanted(ncol_wanted, func, fjac, pjac, node, nnodes, t, y0, np, p, diffusion, 0.5 * L);
        *handle = lib;

        free(y0);
        free(p);
        free(t);
        return collocator_wanted;
    }
}


TWIST::Collocator extend_to_multiwave_solution(TWIST::Collocator &collocator, const int nwaves)
{
    const ptrdiff_t node = collocator.getNode();
    const ptrdiff_t nt1 = collocator.getNNodes();
    const ptrdiff_t s = collocator.getNStages();
    double *y1 = collocator.y();
    double *t1 = collocator.t();
    void *lib = collocator.getLibHandle();
    func_t func = (func_t)dlsym(lib, "func");
    func_t fjac = (func_t)dlsym(lib, "fjac");
    pjac_t pjac = (pjac_t)dlsym(lib, "pjac");
    const size_t np = collocator.getNParam();
    const double *p = collocator.p();
    const diffusion_t diffusion = collocator.getDiffusion();
    const double spatial_period = 0.5 * collocator.unscaledSpatialPeriod();

    const ptrdiff_t nt2 = nt1 + nwaves * (nt1 - 1);
    double *t2 = (double *)malloc(nt2 * sizeof(double));
    double *y2 = (double *)malloc(nt2 * node * sizeof(double));


    memcpy(t2, t1, nt1 * sizeof(double));
    for (ptrdiff_t j = 0; j < nt1; j++) {
        memcpy(&y2[node * j], &y1[node * j * (s + 1)], node * sizeof(double));
    }
    for (int i = 0; i < nwaves; i++) {
        // t2[nt1 + i * (nt1 - 1)]
        for (ptrdiff_t j = 0; j < (nt1 - 1); j++) {
            t2[nt1 + i * (nt1 - 1) + j] = (i + 1) + t1[j + 1];
            memcpy(&y2[node * (nt1 + i * (nt1 - 1) + j)], &y2[node * (j + 1)], node * sizeof(double));
        }
    }

    for (ptrdiff_t j = 0; j < nt2; j++) {
        t2[j] /= nwaves + 1;
    }

    TWIST::Collocator collocator2(s, func, fjac, pjac, node, nt2, t2, y2, np, p, diffusion, spatial_period);
    collocator2.p()[np - 1] *= nwaves + 1;
    free(t2);
    free(y2);

    y2 = collocator2.y();
    for (int i = 0; i < (nwaves + 1); i++) {
        ptrdiff_t j = node * ((nt1 - 1) * (s + 1) + 1);
        memcpy(&y2[i * j - (i * node)], y1, j * sizeof(double));
    }

    return collocator2;
}
