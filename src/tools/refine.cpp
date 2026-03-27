#include "tools/refine.h"
#include "serialize.h"
#include "tools/insert.h"

#include "collocator.h"
#include <algorithm>
#include <filesystem>
#include <vector>


void do_refine(H5::H5File &data, const int index1, const int index2, const double dsabs, const std::filesystem::path &tmp_dir, const std::string &name)
{
    void continuation_work_loop(TWIST::Collocator & collocator, int parnum, TWIST::ContinuationBounds &bounds, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, const bool get_nullspace = true);

    // H5::H5File data(path, H5F_ACC_RDWR);
    H5::Group group;
    std::string cmdline_str;
    std::vector<std::string_view> split_cmdline;
    // const std::string basename = std::filesystem::path(path).filename().string();

    double par1, par2;
    TWIST::ContinuationBounds bounds;
    const double sgn = TWIST::sign(TWIST::Collocator(data, 1).getContinuationParameterValue() - TWIST::Collocator(data, 0).getContinuationParameterValue());
    TWIST::Collocator collocator1(data, index1);
    TWIST::Collocator collocator2(data, index2);

    // get paramter range
    par1 = collocator1.getContinuationParameterValue();
    par2 = collocator2.getContinuationParameterValue();

    // setup continuation bounds
    bounds.ds = sgn * dsabs;
    bounds.dsmin = dsabs;
    bounds.dsmax = dsabs;
    bounds.parmin = std::min(par1, par2);
    bounds.parmax = std::max(par1, par2);
    bounds.geps = 1e-12;
    bounds.min_nodes_adapt = 20;
    bounds.max_nodes_adapt = 1 << 30;
    bounds.allow_mesh_adaptation = true;

    // do continuation
    continuation_work_loop(collocator1, collocator1.getContinuationParameterIndex(), bounds, name, "delete", "delete", "delete", tmp_dir, false);
}


void refine_between_solutions(H5::H5File &data, const int index1, const int index2, const double dsabs)
{
    std::string name;
    char *_name = nullptr;
    auto tmp_dir = std::filesystem::temp_directory_path();
    auto group = data.openGroup("basic_info");
    deserialize(group, "name", _name);
    name = _name;
    auto tmp_path = tmp_dir / fmt::format("{}-delete-delete-delete.h5", name);
    fmt::println("saved temporary data to {}", tmp_path.string());
    do_refine(data, index1, index2, dsabs, tmp_dir, name);
    H5::H5File tmp_data(tmp_path, H5F_ACC_RDWR);

    // insert data from temp file into main
    try {
        insert_data(data, tmp_data);
    }
    catch (...) {
        fmt::println("insertion failed");
    }

    // delete temp file
    std::filesystem::remove(tmp_path);
}