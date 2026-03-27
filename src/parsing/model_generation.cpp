#include "cli/load.h"
#include <filesystem>
#include <stdexcept>


#ifdef USE_BETTER_AUX
#include "experimental/expander.h"
#include <utility>

using aux_entry_t = std::pair<std::vector<std::string>, std::string>;
using aux_list_t = tsl::ordered_map<std::string, aux_entry_t>;

static const std::regex empty_call_pattern(R"(\(\s*\))");

aux_list_t create_aux_funcs(const simdjson::dom::element &doc)
{
    aux_list_t result;
    json_res_t aux_entry;
    bool is_missing;

    aux_entry = get_json_entry(doc, "auxillary", true, is_missing);
    if (is_missing) {
        return result;
    }

    json_object_t aux_data = get_json_object(aux_entry, "auxillary");
    for (const auto &[aux_name, aux_detail] : aux_data) {
        auto inputs_entry = get_json_entry(aux_detail, "inputs", false, is_missing);
        auto inputs_array = get_json_array(inputs_entry, "inputs");
        auto output_string = std::string{ get_json_string(aux_detail, "output") };

        output_string = std::regex_replace(output_string, empty_call_pattern, "");
        result[std::string{ aux_name }] = { std::vector<std::string>(inputs_array.begin(), inputs_array.end()), output_string };
    }
    return aux_list_t(result.rbegin(), result.rend());
}

void process_aux_funcs(std::vector<std::string> &rhs, const simdjson::dom::element &doc)
{
    FunctionExpander expander;
    std::ostringstream stream;
    SymEngine::Expression tmp_expr;

    const auto &aux = create_aux_funcs(doc);

    // register all auxillary functions
    for (const auto &[aux_name, aux_detail] : aux) {
        const auto &[aux_func_args, aux_func_defn] = aux_detail;
        expander.register_function(aux_name, aux_func_args, aux_func_defn);
    }

    for (auto &rhs_defn : rhs) {
        // get rid of any empty parentheses first to prevent
        // symengine syntax error exception
        rhs_defn = std::regex_replace(rhs_defn, empty_call_pattern, "");

        SymEngine::Expression sym_rhs_defn(rhs_defn);
        // loop through aux first and attempt direct subs first
        for (const auto &[aux_name, aux_detail] : aux) {
            const auto &[aux_inputs, aux_output] = aux_detail;
            SymEngine::Expression sym_output(aux_output);

            if (aux_inputs.empty()) {
                // no args -> just name replacement
                // args will be handled by expander
                sym_rhs_defn = sym_rhs_defn.subs({ { SymEngine::Expression(aux_name), sym_output } });
            }
        }
        stream.str("");
        stream.clear();
        stream << sym_rhs_defn;
        rhs_defn = stream.str();
        rhs_defn = expander.expand_string(rhs_defn);
    }
}

#endif

constexpr std::string_view code_template = R"(#include <cmath>

extern "C" {{
    void func_ode(const double *__dummy1, const double *__dummy2, double *__dummy_result)
    {{
        {const_decls_ode}
        {transforms}
        {rhs_func_ode}
    }}

    void fjac_ode(const double *__dummy1, const double *__dummy2, double *__dummy_result)
    {{
        {const_decls_ode}
        {transforms}
        {rhs_fjac_ode}
    }}

    void func(const double *__dummy1, const double *__dummy2, double *__dummy_result)
    {{
        {const_decls}
        {transforms}
        {rhs_func}
    }}

    void fjac(const double *__dummy1, const double *__dummy2, double *__dummy_result)
    {{
        {const_decls}
        {transforms}
        {rhs_fjac}
    }}

    void pjac(const double *__dummy1, const double *__dummy2, const int *pmask, const int nparam, double *__dummy_result)
    {{
        {const_decls}
        {transforms}
        for (int i = 0; i < nparam; i++) {{
            int k = pmask[i];
            switch (k) {{
                {rhs_pjac}
            }}
        }}
    }}
}}
)";


std::string generate_constant_delcations(const tsl::ordered_map<std::string, double> &constants, const std::vector<std::string> &state_vars, const std::vector<std::string> &params)
{
    size_t i;
    std::stringstream stream;

    for (const auto &[name, value] : constants) {
        stream << fmt::format("constexpr double {} = {:.16g};\n        ", name, value);
    }
    for (i = 0; i < state_vars.size(); i++) {
        stream << fmt::format("auto {} = __dummy1[{}];\n        ", state_vars[i], i);
    }
    for (i = 0; i < params.size(); i++) {
        stream << fmt::format("const auto {} = __dummy2[{}];\n        ", params[i], i);
    }
    return stream.str();
}

std::string escape(const std::string &s)
{
    static const std::regex specialChars{ R"([-[\]{}()*+?.,\^$|#\s])" };
    return std::regex_replace(s, specialChars, R"(\$&)");
}

void process_auxillary_functions(std::vector<std::string> &rhs, const tsl::ordered_map<std::string, std::string> &aux)
{
    // for (const auto &[rep, aux_expr] : aux) {
    for (auto it = aux.rbegin(); it != aux.rend(); it++) {
        const auto &[rep, aux_expr] = *it;
        const std::regex pattern{ fmt::format(R"(\b{}(?!\w))", escape(rep)) };
        for (std::string &rhs_expr : rhs) {
            rhs_expr = fmt::format(" {} ", rhs_expr);
            rhs_expr = std::regex_replace(rhs_expr, pattern, fmt::format("({})", aux_expr));
            // std::cout << rhs_expr << '\n';
        }
    }
}

tsl::ordered_map<std::string, std::pair<double, double>> load_transforms(const simdjson::dom::element &doc)
{
    bool is_missing;
    tsl::ordered_map<std::string, std::pair<double, double>> fields;

    auto name_ref = get_json_entry(doc, "transforms", true, is_missing);
    if (is_missing) {
        return fields;
    }
    auto object = get_json_object(name_ref, "transforms");

    for (const auto &[pname, value] : object) {
        // fields[std::string_vi ew{ field.key }.data()] = { pair_ref.at(0).get_double(), pair_ref.at(1).get_double() };
        if (!value.is_array()) {
            throw std::runtime_error(fmt::format("transforms/{} is not an array", pname));
        }
        auto array_ref = value.get_array();
        auto entry1 = array_ref.at(0);
        auto entry2 = array_ref.at(1);
        if ((!entry1.is_double()) || (!entry2.is_double())) {
            throw std::runtime_error("Syntax for transforms must be \"<variable name>\": [ <shift>, <scale> ]");
        }
        fields[std::string{ pname }] = { entry1.get_double(), entry2.get_double() };
    }
    return fields;
}

std::pair<SymEngine::DenseMatrix, SymEngine::DenseMatrix> extend_to_twave_coordinates(const std::vector<std::pair<int, double>> &diff_list, std::vector<std::string> &vars, SymEngine::DenseMatrix &rhs)
{
    const size_t ndiff = diff_list.size();
    const size_t node = vars.size();
    size_t i;
    SymEngine::Expression c("c");
    SymEngine::Expression sps("spatial_period_scale");
#ifdef TEST_WAVE_NUMBER_PARAM
    SymEngine::Expression wave_number("wave_number");
#endif
    SymEngine::DenseMatrix J(node + ndiff, node + ndiff);
    SymEngine::DenseMatrix vars_sym(node + ndiff, 1);
    std::vector<size_t> to_skip;

    to_skip.reserve(node);

    for (i = 0; i < ndiff; i++) {
        const auto [diff_loc, diff_val] = diff_list[i];
        const std::string u_name = fmt::format("_velocity_{}", i);
        SymEngine::Expression u(u_name);
        to_skip.push_back(diff_loc + i);
#ifdef TEST_WAVE_NUMBER_PARAM
        rhs.set(diff_loc + i, 0, (c * u - rhs.get(diff_loc + i, 0)) / (wave_number * wave_number * diff_val));
#else
        if (diff_val != 1) {
            rhs.set(diff_loc + i, 0, (c * u - rhs.get(diff_loc + i, 0)) / diff_val);
        }
        else {
            rhs.set(diff_loc + i, 0, (c * u - rhs.get(diff_loc + i, 0)));
        }
#endif
        rhs.row_insert(SymEngine::DenseMatrix{ { u } }, diff_loc + i);
        vars.insert(vars.begin() + diff_loc + i + 1, u_name);
    }

    for (i = 0; i < vars.size(); i++) {
        vars_sym.set(i, 0, SymEngine::Expression(vars[i]));
        rhs.set(i, 0, rhs.get(i, 0) * sps);
        rhs.set(i, 0, SymEngine::simplify(rhs.get(i, 0)));
    }

    for (i = 0; i < (node + ndiff); i++) {
        if (std::find(to_skip.begin(), to_skip.end(), i) != to_skip.end()) {
            continue;
        }
        if (std::find(to_skip.begin(), to_skip.end(), i - 1) != to_skip.end()) {
            continue;
        }
        rhs.set(i, 0, rhs.get(i, 0) / c);
    }
    SymEngine::jacobian(rhs, vars_sym, J);

    return { J, vars_sym };
}

std::string generate_func_code(const SymEngine::DenseMatrix &rhs)
{
    std::stringstream stream;
    for (unsigned int i = 0; i < rhs.nrows(); i++) {
        stream << fmt::format("__dummy_result[{}] = {};\n{:8s}", i, SymEngine::cxxcode(SymEngine::Expression{ rhs.get(i, 0) }), "");
    }
    return stream.str();
}

std::string generate_fjac_code(const SymEngine::DenseMatrix &jac)
{
    std::stringstream stream;
    for (unsigned int i = 0; i < jac.nrows(); i++) {
        for (unsigned int j = 0; j < jac.ncols(); j++) {
            stream << fmt::format("__dummy_result[{} * {} + {}] = {};\n{:8s}", i, jac.ncols(), j, SymEngine::cxxcode(SymEngine::Expression{ jac.get(i, j) }), "");
        }
    }
    return stream.str();
}

void generate_hjac_code(std::stringstream &stream, const int hess_num, const SymEngine::DenseMatrix &jac)
{
    for (unsigned int i = 0; i < jac.nrows(); i++) {
        for (unsigned int j = 0; j < jac.ncols(); j++) {
            stream << fmt::format("__dummy_temporary[{} * {} + {}] = {};\n{:8s}", i, jac.ncols(), j, SymEngine::cxxcode(SymEngine::Expression{ jac.get(i, j) }), "");
        }
    }
    stream << fmt::format("matvec({}, __dummy_temporary, __dummy1, &__dummy_result[{} * {}]);\n{:8s}", jac.nrows(), hess_num, jac.nrows(), "");
}

std::string generate_pjac_code(const SymEngine::DenseMatrix &pjac)
{
    std::stringstream stream;
    for (unsigned int i = 0; i < pjac.ncols(); i++) {
        stream << fmt::format("{:{}s}case {}:\n", "", i ? 16 : 0, i);
        for (unsigned int j = 0; j < pjac.nrows(); j++) {
            stream << fmt::format("{:20s}__dummy_result[{} * nparam + i] = {};\n", "", j, SymEngine::cxxcode(SymEngine::Expression{ pjac.get(j, i) }));
        }
        stream << fmt::format("{:20s}break;\n", "");
    }
    return stream.str();
}

std::string generate_transform_code(const tsl::ordered_map<std::string, std::pair<double, double>> &transforms)
{
    std::stringstream stream;
    for (const auto &[variable, transform] : transforms) {
        const auto &[shift, scale] = transform;
        // v = 120 * v - 80
        stream << fmt::format("{1} = {2} * {1} - {3};\n{0:8s}", "", variable, scale, shift);
    }
    return stream.str();
}

void check_invalid_names(const std::vector<std::string> &names)
{
    if (std::find(names.begin(), names.end(), "c") != names.end()) {
        fmt::println(COLOR_RED "Error: 'c' is a reserved name for the wave speed. Use a different name for this parameter");
        exit(1);
    }
    else if (std::find(names.begin(), names.end(), "sps") != names.end()) {
        fmt::println(COLOR_RED "Error: 'sps' is a reserved name for the spatial period scale alias. Use a different name for this parameter");
        exit(1);
    }
    else if (std::find(names.begin(), names.end(), "spatial_period_scale") != names.end()) {
        fmt::println(COLOR_RED "Error: 'spatial_period_scale' is a reserved name for the spatial period scale. Use a different name for this parameter");
        exit(1);
    }
}

std::string generate_compilable_code(const simdjson::dom::element &doc)
{
    std::vector<std::string> state_vars, rhs;
    auto system = entry_to_map("system", doc);
    auto aux = entry_to_map("aux", doc, true);
    auto params = entry_to_list<std::string>("params", doc, false);
    auto constants = entry_to_map<double>("constants", doc, true);
    auto transforms = load_transforms(doc);
    auto diffusion = load_diffusion(doc);
    SymEngine::DenseMatrix rhs_sym(system.size(), 1);
    SymEngine::DenseMatrix vars_sym(system.size(), 1);
    SymEngine::DenseMatrix params_sym(params.size() + 2, 1);
#ifdef TEST_WAVE_NUMBER_PARAM
    SymEngine::DenseMatrix pjac(system.size() + diffusion.size(), params.size() + 3);
#else
    SymEngine::DenseMatrix pjac(system.size() + diffusion.size(), params.size() + 2);
#endif
    SymEngine::DenseMatrix Jode(system.size(), system.size());

    check_invalid_names(params);

    // add wave speed and spatial period paramters
    // to list
    params.insert(params.begin(), "c");
    params.push_back("spatial_period_scale");
#ifdef TEST_WAVE_NUMBER_PARAM
    params.push_back("wave_number");
#endif

    // create symbolic version of paramters
    for (unsigned int i = 0; i < params.size(); i++) {
        params_sym.set(i, 0, SymEngine::Expression(params[i]));
    }

    // extract state variables and rhs defintions
    // for easier manipuation later
    state_vars.reserve(system.size());
    rhs.reserve(system.size());
    for (const auto &[key, value] : system) {
        state_vars.emplace_back(key);
        rhs.emplace_back(value);
    }
    check_invalid_names(state_vars);
#ifdef USE_BETTER_AUX
    process_aux_funcs(rhs, doc);
#else
    process_auxillary_functions(rhs, aux);
#endif

    // create symbolic version of RHS
    for (unsigned int i = 0; i < state_vars.size(); i++) {
        rhs_sym.set(i, 0, SymEngine::Expression(rhs[i]));
    }

    // apply any transforms
    for (const auto &[var_name, shift_and_scale] : transforms) {
        const auto [_, scale] = shift_and_scale;
        if (scale == 1) {
            continue;
        }
        auto where = std::find(state_vars.begin(), state_vars.end(), var_name);
        if (where == state_vars.end()) {
            continue;
        }
        ptrdiff_t i = where - state_vars.begin();
        rhs_sym.set(i, 0, SymEngine::Expression{ rhs_sym.get(i, 0) } / scale);
    }

    // BEGIN: create local dynamics stuff
    auto rhs_func_ode = generate_func_code(rhs_sym);
    for (unsigned int i = 0; i < state_vars.size(); i++) {
        vars_sym.set(i, 0, SymEngine::Expression(state_vars[i]));
    }
    SymEngine::jacobian(rhs_sym, vars_sym, Jode);
    // need to undo double scaling from transform applied to RHS
    for (const auto &[var_name, shift_and_scale] : transforms) {
        const auto [_, scale] = shift_and_scale;
        if (scale == 1) {
            continue;
        }
        auto where = std::find(state_vars.begin(), state_vars.end(), var_name);
        if (where == state_vars.end()) {
            continue;
        }
        ptrdiff_t i = where - state_vars.begin();
        for (unsigned int j = 0; j < Jode.nrows(); j++) {
            Jode.set(j, i, scale * SymEngine::Expression{ Jode.get(j, i) });
        }
    }
    for (unsigned int i = 0; i < Jode.nrows(); i++) {
        for (unsigned int j = 0; j < Jode.ncols(); j++) {
            Jode.set(i, j, SymEngine::simplify(SymEngine::expand(Jode.get(i, j))));
        }
    }
    auto rhs_fjac_ode = generate_fjac_code(Jode);
    auto const_decls_ode = generate_constant_delcations(constants, state_vars, params);
    // END: create local dynamics stuff

    // move system to traveling wave coordinates
    auto &&[J, extended_vars] = extend_to_twave_coordinates(diffusion, state_vars, rhs_sym);

    // need to undo double scaling from transform applied to RHS
    for (const auto &[var_name, shift_and_scale] : transforms) {
        const auto [_, scale] = shift_and_scale;
        if (scale == 1) {
            continue;
        }
        auto where = std::find(state_vars.begin(), state_vars.end(), var_name);
        if (where == state_vars.end()) {
            continue;
        }
        ptrdiff_t i = where - state_vars.begin();
        for (unsigned int j = 0; j < J.nrows(); j++) {
            J.set(j, i, scale * SymEngine::Expression{ J.get(j, i) });
        }
    }

    // create partial derivatives for parameters
    SymEngine::jacobian(rhs_sym, params_sym, pjac);

    // experiment for future: local hessian for two param continuation
    if (0) {
        std::stringstream stream;
        SymEngine::DenseMatrix H(J.nrows(), J.ncols());
        SymEngine::DenseMatrix rowi(1, J.ncols());
        SymEngine::DenseMatrix grad(J.ncols(), 1);
        for (unsigned int hess_num = 0; hess_num < J.nrows(); hess_num++) {
            J.submatrix(rowi, hess_num, 0, hess_num + 1, J.ncols());
            rowi.transpose(grad);
            SymEngine::jacobian(grad, extended_vars, H);
            generate_hjac_code(stream, hess_num, H);
        }
        fmt::println("{}", stream.str());
    }

    // leave this is as if for now in case I see a reason to make it optional
    if (constexpr bool simplity_eqs = true; simplity_eqs) {
        for (unsigned int i = 0; i < rhs_sym.nrows(); i++) {
            rhs_sym.set(i, 0, SymEngine::simplify(rhs_sym.get(i, 0)));
        }
        for (unsigned int i = 0; i < J.nrows(); i++) {
            for (unsigned int j = 0; j < J.ncols(); j++) {
                J.set(i, j, SymEngine::simplify(SymEngine::expand(J.get(i, j))));
            }
        }
        for (unsigned int i = 0; i < pjac.nrows(); i++) {
            for (unsigned int j = 0; j < pjac.ncols(); j++) {
                pjac.set(i, j, SymEngine::simplify(SymEngine::expand(pjac.get(i, j))));
            }
        }
    }

    auto const_decls = generate_constant_delcations(constants, state_vars, params);
    auto rhs_func = generate_func_code(rhs_sym);
    auto rhs_fjac = generate_fjac_code(J);
    auto rhs_pjac = generate_pjac_code(pjac);
    auto transforms_code = generate_transform_code(transforms);

    return fmt::format(code_template, fmt::arg("const_decls", const_decls), fmt::arg("transforms", transforms_code), fmt::arg("rhs_func", rhs_func), fmt::arg("rhs_fjac", rhs_fjac), fmt::arg("rhs_pjac", rhs_pjac), fmt::arg("const_decls_ode", const_decls_ode), fmt::arg("rhs_func_ode", rhs_func_ode), fmt::arg("rhs_fjac_ode", rhs_fjac_ode));
}

void compile_model(const char *path, const char *compiler, const bool source_only)
{
    int sys_code;
    FILE *file;
    std::string cpp_file, cmd;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    try {
        doc = parser.load(path);
    }
    catch (simdjson::simdjson_error &e) {
        throw std::runtime_error(fmt::format("Could not parse file {} ({})", path, e.what()));
    }
    std::string code = generate_compilable_code(doc);
    std::string_view name = get_json_string(doc, "name");

    // create directories for output
    std::filesystem::create_directories(".cache/models/src");
    std::filesystem::create_directories(".cache/models/lib");

    // create save file
    cpp_file = fmt::format(".cache/models/src/{}.cpp", name);
    file = fopen(cpp_file.c_str(), "w");
    fwrite(code.c_str(), 1, code.size(), file);
    fclose(file);

    if (source_only) {
        fmt::println("source file has been saved to {}\nIf manually compiling make sure to save it under .cache/models/lib/{}.so", cpp_file, name);
        return;
    }

    // don't allow overwrites by default
    if (std::filesystem::exists(fmt::format(".cache/models/lib/{}.so", name))) {
        fmt::print("A model with the name '{}' has already been built. Do you want to replace this model? (y/n) ", name);
        fflush(stdout);
        if (char proceed; (scanf(" %c", &proceed) != 1) || ((proceed != 'y') && (proceed != 'Y'))) {
            return;
        }
    }
    // compile code
    cmd = fmt::format("{0} -std=c++20 -march=native -O3 .cache/models/src/{1}.cpp -shared -o .cache/models/lib/{1}.so", compiler, name);
    fmt::println("running: {}", cmd);
    sys_code = system(cmd.c_str());
    if (sys_code != 0) {
        exit(sys_code);
    }
}
