#include "cli/cli_tools.h"
#include "cli/colors.h"
#include "collocator.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "serialize.h"
#include "tools/helpers.h"
#include "tools/insert.h"
#include "tools/refine.h"

#include <filesystem>
#include <numeric>


// static void export_to_csv(H5::H5File &file, const int index, const std::filesystem::path &path)
template <const char delimeter>
void export_to_delimited(H5::H5File &file, const int index, const std::filesystem::path &path)
{
    FILE *csv_file;
    const char _delimeter[2] = { delimeter, '\0' };

    TWIST::Collocator collocator(file, index);
    const double *y = collocator.y();
    const size_t node = collocator.getNode();

    const auto xi = collocator.plottablePoints();
    const double L = collocator.spatialPeriod();

    check_export_path(path);
    csv_file = fopen(path.c_str(), "w");

    for (size_t i = 0; i < xi.size(); i++) {
        fmt::println(csv_file, "{:.15e},{: .15e}", L * xi[i], fmt::join(&y[i * node], &y[(i + 1) * node], _delimeter));
    }

    fclose(csv_file);
}

static auto export_to_csv = export_to_delimited<','>;
static auto export_to_tsv = export_to_delimited<'\t'>;
static auto export_to_dat = export_to_delimited<' '>;


int ToolsArgs::run()
{
    this->validate(true);

    verify_hdf5_file(file);

    H5::PropList object_copy_prop = H5::PropList::DEFAULT;
    H5::PropList link_creation_prop = H5::PropList::DEFAULT;

    ptrdiff_t i, offset;
    H5::H5File h5_file(file, H5F_ACC_RDWR);
    size_t ngroups = get_number_of_solutions(h5_file);
    std::vector<int> indices{ parse_indices_and_slices(which_indices, ngroups) };

    std::vector<int> all_indices(ngroups);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    if (prune) {
        offset = 0;
        for (i = 0; i < (ptrdiff_t)indices.size(); i++) {
            if ((indices[i] < 0) || (ngroups <= (size_t)indices[i])) {
                fmt::println(COLOR_YELLOW "[WARNING]: Solution index {} is not present in {} which only has {} solutions" COLOR_RESET, indices[i], file, ngroups);
                continue;
            }
            h5_file.unlink(fmt::format("{}", indices[i]));
            fmt::println(COLOR_GREEN "Successfully removed solution {} from {}" COLOR_RESET, indices[i], file);
            all_indices.erase(all_indices.begin() + indices[i] - offset);
            offset++;
        }

        for (i = 0; i < (ptrdiff_t)all_indices.size(); i++) {
            if (all_indices[i] != i) {
                h5_file.move(fmt::format("{}", all_indices[i]), fmt::format("{}", i));
            }
        }
    }

    if (reverse) {
        for (i = 0; i < ((ptrdiff_t)all_indices.size() / 2); i++) {
            const auto name1 = fmt::format("{}", i);
            const auto name2 = fmt::format("{}", all_indices.size() - 1 - i);
            h5_file.move(name1, "tmp");
            h5_file.move(name2, name1);
            h5_file.move("tmp", name2);
        }
    }

    for (const auto &file_to_append : files_to_append) {
        verify_hdf5_file(file_to_append);

        H5::H5File h5_file_other(file_to_append, H5F_ACC_RDONLY);
        ptrdiff_t ngroups_other = get_number_of_solutions(h5_file_other);

        try {
            verify_basic_info(h5_file, h5_file_other);
        }
        catch (const std::string &which) {
            fmt::println(COLOR_YELLOW "[WARNING]: Could not append {} to {} because the {} did not match" COLOR_RESET, file_to_append, file, which);
            continue;
        }

        std::vector<std::string> paths;
        for (i = 0; i < ngroups_other; i++) {
            const std::string name1 = fmt::format("{}", i);
            const std::string name2 = fmt::format("{}", i + ngroups);
            herr_t status = H5Ocopy(h5_file_other.getId(), name1.c_str(), h5_file.getId(), name2.c_str(), object_copy_prop.getId(), link_creation_prop.getId());
            if (status < 0) {
                fmt::println("copying {}/{} to {}/{} failed", file_to_append, i, file, i + ngroups);
                return 1;
            }
        }
    }

    if (file_to_insert) {
        H5::H5File h5_file_other(file_to_insert.value(), H5F_ACC_RDONLY);
        try {
            verify_basic_info(h5_file, h5_file_other);
        }
        catch (const std::string &which) {
            fmt::println(COLOR_YELLOW "[WARNING]: Could not insert {} into {} because the {} did not match" COLOR_RESET, file_to_insert.value(), file, which);
            return 1;
        }
        insert_data(h5_file, h5_file_other);
    }

    if (flip_direction) {
        double *buffer = nullptr;
        size_t size;
        for (i = 0; i < (ptrdiff_t)indices.size(); i++) {
            if ((indices[i] < 0) || (ngroups <= (size_t)indices[i])) {
                fmt::println(COLOR_YELLOW "[WARNING]: Solution index {} is not present in {} which only has {} solutions" COLOR_RESET, indices[i], file, ngroups);
                continue;
            }
            H5::Group group{ h5_file.openGroup(fmt::format("/{}", indices[i])) };
            deserialize(group, "nullspacev", buffer, size);
            buffer[size - 1] *= -1;
            h5_file.move(fmt::format("/{}/nullspacev", indices[i]), "tmp");
            serialize<double, 1>(group, "nullspacev", buffer, { size });
            h5_file.unlink("tmp");
            fmt::println(COLOR_GREEN "Successfully flipped direction for solution {} from {}" COLOR_RESET, indices[i], file);
        }
    }

    if (refine_between) {
        if (refine_between.value().size() != 2) {
            fmt::println("TWO solution indices must be provided to refine between");
            return 1;
        }
        const auto &indices = refine_between.value();
        int index1 = indices[0];
        int index2 = indices[1];

        if (index2 < index1) {
            std::swap(index1, index2);
        }
        refine_between_solutions(h5_file, index1, index2, refine_ds);
    }

    if (export_solutions) {
        switch (export_solutions.value()) {
        case TWIST::TextFileTypes::csv:
            export_to_csv(h5_file, export_solution_index, export_path);
            break;
        case TWIST::TextFileTypes::tsv:
            export_to_tsv(h5_file, export_solution_index, export_path);
            break;
        case TWIST::TextFileTypes::dat:
            export_to_dat(h5_file, export_solution_index, export_path);
            break;
        default:
            break;
        }
    }

    return 0;
}

void ToolsArgs::welcome()
{
    fmt::println("Various tools for continuation data file manipulation and solution exports");
}