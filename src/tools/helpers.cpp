#include "tools/helpers.h"
#include "fmt/core.h"
#include "serialize.h"
#include "shared.h"


OpenBLASThreadContext::OpenBLASThreadContext(const int nwanted) : m_nwanted(nwanted), m_ndefault(openblas_get_num_threads())
{
    openblas_set_num_threads_(&m_nwanted);
}

OpenBLASThreadContext::~OpenBLASThreadContext()
{
    openblas_set_num_threads_(&m_ndefault);
}

static herr_t op_func(hid_t, const char *name, const H5L_info_t *, void *op_data)
{
    ((std::vector<std::string> *)op_data)->emplace_back(name);
    return 0;
}

std::vector<std::string> get_all_subgroup_names(const H5::Group &group)
{
    std::vector<std::string> names;
    H5Literate(group.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, op_func, &names);
    return names;
}

void get_group_hierarchy(const H5::Group &group, std::vector<std::string> &names, const std::string &base, int recurse_level)
{
    if (recurse_level == 0) {
        names.clear();
    }
    std::vector<std::string> sub_names{ get_all_subgroup_names(group) };
    for (const auto &sub_name : sub_names) {
        const auto next = fmt::format("{}/{}", base, sub_name);
        // fmt::println("{}", next);
        H5O_type_t child_type = group.childObjType(next);
        if (child_type == H5O_type_t::H5O_TYPE_GROUP) {
            get_group_hierarchy(group.openGroup(sub_name.data()), names, next, recurse_level + 1);
            continue;
        }
        names.emplace_back(next);
    }
}

template <typename T>
inline void verify_same(const std::string &name, const T a, const T b)
{
    if (a != b) {
        throw name;
    }
}

void verify_basic_info(const H5::H5File &file1, const H5::H5File &file2)
{
    char *model_name1 = nullptr;
    int *diffusion_indices1 = nullptr;
    int *pmask1 = nullptr;
    double *diffusion_coeffs1 = nullptr;
    int ndiff1;
    size_t node1, np1, nparam1;
    double spatial_period1;
    diffusion_t diffusion1;

    char *model_name2 = nullptr;
    int *diffusion_indices2 = nullptr;
    int *pmask2 = nullptr;
    double *diffusion_coeffs2 = nullptr;
    int ndiff2;
    size_t node2, np2, nparam2;
    double spatial_period2;
    diffusion_t diffusion2;

    {
        TWIST::FileType file_type;
        H5::Group group = file1.openGroup("meta_data");
        deserialize(group, "file_type", file_type);
        if (file_type == TWIST::FileType::spectrum) {
            return;
        }
    }

    {
        H5::Group group = file1.openGroup("basic_info");
        deserialize(group, "node", node1);
        deserialize(group, "np", np1);
        deserialize(group, "nparam", nparam1);
        deserialize(group, "spatial_period", spatial_period1);
        deserialize(group, "pmask", pmask1);
        deserialize(group, "name", model_name1);
        deserialize(group, "diffusion_indices", diffusion_indices1);
        deserialize(group, "diffusion_coeffs", diffusion_coeffs1);
        deserialize(group, "ndiff", ndiff1);
        for (int i = 0; i < ndiff1; i++) {
            diffusion1.emplace_back(std::pair<int, double>{ diffusion_indices1[i], diffusion_coeffs1[i] });
        }
    }
    {
        H5::Group group = file2.openGroup("basic_info");
        deserialize(group, "node", node2);
        deserialize(group, "np", np2);
        deserialize(group, "nparam", nparam2);
        deserialize(group, "spatial_period", spatial_period2);
        deserialize(group, "pmask", pmask2);
        deserialize(group, "name", model_name2);
        deserialize(group, "diffusion_indices", diffusion_indices2);
        deserialize(group, "diffusion_coeffs", diffusion_coeffs2);
        deserialize(group, "ndiff", ndiff2);
        for (int i = 0; i < ndiff2; i++) {
            diffusion2.emplace_back(std::pair<int, double>{ diffusion_indices2[i], diffusion_coeffs2[i] });
        }
    }

    verify_same("model name", std::string_view(model_name1), std::string_view(model_name2));
    verify_same("# of ODEs", node1, node2);
    verify_same("# of model parameters", np1, np2);
    verify_same("# of free parameters", nparam1, nparam2);
    verify_same("free parameters", std::vector<int>(pmask1, pmask1 + nparam1), std::vector<int>(pmask2, pmask2 + nparam1));
    verify_same("base spatial period", spatial_period1, spatial_period2);
    verify_same("diffusion matrix", diffusion1, diffusion2);

    free(model_name1);
    free(diffusion_indices1);
    free(diffusion_coeffs1);
    free(pmask1);
    free(model_name2);
    free(diffusion_indices2);
    free(diffusion_coeffs2);
    free(pmask2);
}

void copy_dataset(const H5::Group &group_src, H5::Group &group_dst, const std::string_view &name)
{
    hsize_t dim;
    std::string buffer;
    H5::DataSet dataset_src = group_src.openDataSet(name.data());
    H5::DataSpace dataspace_src = dataset_src.getSpace();
    dataspace_src.getSimpleExtentDims(&dim);
    H5::DataSpace dataspace_dst(1, &dim);
    H5::DataType dtype = dataset_src.getDataType();
    H5::DataSet dataset_dst = group_dst.createDataSet(name.data(), dtype, dataspace_dst);
    dataset_src.read(buffer, dtype);
    dataset_dst.write(buffer, dtype);
}

void check_export_path(const std::filesystem::path &path)
{
    if (path.string().empty()) {
        throw std::runtime_error("A path must be provided when exporting data");
    }
    else if (path.has_parent_path()) {
        const auto &parent_path = path.parent_path();
        if (!std::filesystem::is_directory(parent_path)) {
            std::filesystem::create_directories(parent_path);
        }
    }
}

void verify_hdf5_file(const std::string &path)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(fmt::format("file {} does not exist", path));
    }
    else if (!H5::H5File::isHdf5(path)) {
        throw std::runtime_error(fmt::format("{} is not a valid hdf5 file", path));
    }
}

void verified_json_load(const std::string &path, simdjson::dom::element &doc, simdjson::dom::parser &parser)
{
    try {
        doc = parser.load(path);
    }
    catch (simdjson::simdjson_error &e) {
        throw std::runtime_error(fmt::format("failed to parse '{}' with error: {}", path, e.what()));
    }
}
