#include "fmt/core.h"
#include "serialize.h"


size_t extract_nstages(const H5::H5File &file)
{
    size_t ncol;
    if (!file.exists("basic_info")) {
        throw std::runtime_error(fmt::format("basic_info is not present. {} was not created by TWIST's continuation command or has been corruputed", file.getFileName()));
    }
    H5::Group group = file.openGroup("basic_info");
    deserialize(group, "nstages", ncol);
    return ncol;
}

size_t extract_nstages(const std::string &h5data_path)
{
    return extract_nstages(H5::H5File(h5data_path, H5F_ACC_RDONLY));
}
