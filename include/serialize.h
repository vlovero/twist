#pragma once

#include "H5Cpp.h"
#include <array>

template <typename data_t>
constexpr const H5::PredType &get_h5_type()
{
    if constexpr (std::is_same_v<data_t, double>) {
        return H5::PredType::NATIVE_DOUBLE;
    }
    else if constexpr (std::is_same_v<data_t, int>) {
        return H5::PredType::NATIVE_INT;
    }
    else if constexpr (std::is_same_v<data_t, int64_t>) {
        return H5::PredType::NATIVE_INT64;
    }
    else if constexpr (std::is_same_v<data_t, size_t>) {
        return H5::PredType::NATIVE_ULONG;
    }
    else if constexpr (std::is_same_v<data_t, uint64_t>) {
        return H5::PredType::NATIVE_UINT64;
    }
    else if constexpr (std::is_same_v<data_t, char>) {
        return H5::PredType::NATIVE_CHAR;
    }
    else if constexpr (std::is_enum_v<data_t>) {
        return get_h5_type<std::underlying_type_t<data_t>>();
    }
    else {
        static_assert(!sizeof(data_t), "data type is not implemented");
    }
}

template <typename data_t>
void serialize(H5::Group &group, const std::string &name, data_t value)
{
    hsize_t len[1] = { 1 };
    H5::DataSpace dataspace(1, len);
    auto h5_type = get_h5_type<data_t>();
    H5::DataSet dataset{ group.createDataSet(name, h5_type, dataspace) };
    dataset.write(&value, h5_type);
}

template <typename data_t, hsize_t nelements>
void serialize(H5::Group &group, const std::string &name, const data_t *values, const std::array<hsize_t, nelements> &length)
{
    H5::DataSpace dataspace(length.size(), length.data());
    auto h5_type = get_h5_type<data_t>();
    H5::DataSet dataset{ group.createDataSet(name, h5_type, dataspace) };
    dataset.write(values, h5_type);
}

template <typename data_t>
void deserialize(H5::Group &group, const std::string &name, data_t &values)
{
    size_t i, ndim, size;
    hsize_t *dims;
    H5::DataSet dataset = group.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    auto h5_type = get_h5_type<std::remove_pointer_t<data_t>>();

    if constexpr (std::is_pointer_v<data_t>) {
        ndim = dataspace.getSimpleExtentNdims();
        dims = (hsize_t *)alloca(ndim * sizeof(hsize_t));
        dataspace.getSimpleExtentDims(dims);

        for (i = 0, size = 1; i < ndim; i++) {
            size *= dims[i];
        }

        if (values != nullptr) {
            free(values);
        }
        if constexpr (std::is_same_v<char *, data_t>) {
            values = (data_t)calloc(size + 1, sizeof(char));
        }
        else {
            size *= sizeof(std::remove_pointer_t<data_t>);
            values = (data_t)malloc(size);
        }
        dataset.read(values, h5_type);
    }
    else {
        dataset.read(&values, h5_type);
    }
}

template <typename data_t>
void deserialize(H5::Group &group, const std::string &name, data_t &values, size_t &size)
{
    size_t i, ndim;
    hsize_t *dims;
    H5::DataSet dataset = group.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    auto h5_type = get_h5_type<std::remove_pointer_t<data_t>>();

    size = 1;
    if constexpr (std::is_pointer_v<data_t>) {
        ndim = dataspace.getSimpleExtentNdims();
        dims = (hsize_t *)alloca(ndim * sizeof(hsize_t));
        dataspace.getSimpleExtentDims(dims);

        for (i = 0, size = 1; i < ndim; i++) {
            size *= dims[i];
        }

        if (values != nullptr) {
            free(values);
        }
        if constexpr (std::is_same_v<char *, data_t>) {
            values = (data_t)calloc(size + 1, sizeof(char));
        }
        else {
            size *= sizeof(std::remove_pointer_t<data_t>);
            values = (data_t)malloc(size);
            size /= sizeof(std::remove_pointer_t<data_t>);
        }
        dataset.read(values, h5_type);
    }
    else {
        dataset.read(&values, h5_type);
    }
}

template <typename data_t, size_t __size>
void deserialize(H5::Group &group, const std::string &name, data_t (&values)[__size])
{
    H5::DataSet dataset = group.openDataSet(name);
    H5::DataSpace dataspace = dataset.getSpace();
    auto h5_type = get_h5_type<std::remove_pointer_t<data_t>>();

    dataset.read(&values, h5_type);
}