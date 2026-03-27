#ifndef TWIST_LOAD_H
#define TWIST_LOAD_H

#include <cstdlib>
#include <fmt/base.h>
#include <fmt/format.h>
#include <ostream>
#include <regex>
#include <stdexcept>

#include "cli/colors.h"
#include "collocator.h"
#include "simdjson.h"
#include "tsl/ordered_map.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "cxxcodeprinter.h"
#include "symengine/expression.h"
#include "symengine/matrix.h"
#include "symengine/simplify.h"
#include "symengine/symbol.h"

#pragma GCC diagnostic pop


using json_elm_t = simdjson::dom::element;
using json_res_t = simdjson::simdjson_result<json_elm_t>;
using json_array_t = simdjson::simdjson_result<simdjson::dom::array>;
using json_object_t = simdjson::simdjson_result<simdjson::dom::object>;
using json_double_t = simdjson::simdjson_result<double>;

json_res_t get_json_entry(const json_elm_t &doc, const std::string_view &name, const bool allow_missing, bool &is_missing);
json_array_t get_json_array(const json_res_t &ref, const std::string_view &name);
json_object_t get_json_object(const json_res_t &ref, const std::string_view &name);

json_object_t get_json_object(const json_elm_t &ref, const std::string_view &name);
double get_json_double(const json_elm_t &ref, const std::string_view &name);
std::string_view get_json_string(const json_elm_t &ref, const std::string_view &name);

std::vector<std::pair<int, double>> load_diffusion(const simdjson::dom::element &doc);
std::tuple<std::string, func_t, func_t, pjac_t, size_t, std::vector<double>, std::vector<std::pair<int, double>>, double, void *> load_base_twave_data_from_spec(const std::string &file, const std::string &parameter_set);
TWIST::Collocator load_collocator_from_spec_and_init(const size_t ncol, const std::string &file, const std::string &parameter_set, void **lib);
TWIST::Collocator load_collocator_from_h5data_and_index(const H5::H5File &h5data, const int solution_index, const size_t ncol_wanted, void **lib);
TWIST::Collocator extend_to_multiwave_solution(TWIST::Collocator &collocator, const int nwaves);
void replace_existing_starting_solution(TWIST::Collocator &collocator, const std::string &file, const std::string &parameter_set);
void replace(std::string &text, const std::string &from, const std::string &to);


template <typename T = std::string>
std::vector<T> entry_to_list(const std::string_view &name, const simdjson::dom::element &doc, const bool allow_empty)
{
    bool is_missing;
    std::vector<T> fields;

    auto name_ref = get_json_entry(doc, name, allow_empty, is_missing);
    if (allow_empty && is_missing) {
        return fields;
    }
    auto array_ref = get_json_array(name_ref, name);

    for (simdjson::dom::element field : array_ref) {
        try {
            fields.emplace_back(field);
        }
        catch (simdjson::simdjson_error &e) {
            auto code = e.error();
            if (code == simdjson::error_code::INCORRECT_TYPE) {
                throw std::runtime_error(fmt::format("invalid entry in '{}' array", name));
            }
        }
    }
    return fields;
}


template <typename T = std::string>
tsl::ordered_map<std::string, T> entry_to_map(const std::string_view &name, const simdjson::dom::element &doc, const bool allow_empty = false)
{
    bool is_missing;
    tsl::ordered_map<std::string, T> fields;

    auto name_ref = get_json_entry(doc, name, allow_empty, is_missing);
    if (allow_empty && is_missing) {
        return fields;
    }
    auto object_ref = get_json_object(name_ref, name);

    for (auto field : object_ref) {
        if (fields.size() && (fields.find(std::string{ field.key }) != fields.end())) {
            throw std::runtime_error(fmt::format("duplicate entry found for {} in {} field", field.key, name));
        }
        fields[std::string_view{ field.key }.data()] = field.value;
    }

    return fields;
}


#endif // TWIST_LOAD_H