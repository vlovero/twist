#pragma once

#include "H5Cpp.h"
#include "linalg.h"
#include "simdjson.h"

#include <filesystem>
#include <string>
#include <vector>


class OpenBLASThreadContext
{
    int m_nwanted;
    int m_ndefault;

public:
    OpenBLASThreadContext() = delete;
    OpenBLASThreadContext(const int nwanted);
    ~OpenBLASThreadContext();
};

int get_number_of_solutions(const std::string &path);
int get_number_of_solutions(H5::H5File &h5_file);

void verified_json_load(const std::string &path, simdjson::dom::element &doc, simdjson::dom::parser &parser);
void verify_basic_info(const H5::H5File &file1, const H5::H5File &file2);
void copy_dataset(const H5::Group &group_src, H5::Group &group_dst, const std::string_view &name);
void check_export_path(const std::filesystem::path &path);
void verify_hdf5_file(const std::string &path);
void get_group_hierarchy(const H5::Group &group, std::vector<std::string> &names, const std::string &base, int recurse_level = 0);
