#include "tools/insert.h"
#include "collocator.h"
#include "fmt/core.h"
#include "indicators/progress_bar.hpp"
#include "serialize.h"
#include "tools/helpers.h"

#include <utility>
#include <vector>


std::vector<int> get_insertion_indices(H5::H5File &main_file, H5::H5File &other_file, ptrdiff_t &nsolutions_main, ptrdiff_t &nsolutions_other)
{
    extern bool is_between(double x, double a, double b);

    std::vector<int> indices;
    std::vector<double> x_main, y_main, x_other, y_other;
    ptrdiff_t i, j;
    H5::Group group;
    int pmask[2];
    double *p = nullptr;

    nsolutions_main = get_number_of_solutions(main_file);
    nsolutions_other = get_number_of_solutions(other_file);

    // assuming already checked that free parameters are the same
    group = main_file.openGroup("basic_info");
    deserialize(group, "pmask", pmask);

    // get main curve
    for (i = 0; i < nsolutions_main; i++) {
        try {
            group = main_file.openGroup(fmt::format("{}", i));
        }
        catch (...) {
            fmt::println("failed to open group {}", i);
            continue;
        }
        deserialize(group, "p", p);
        x_main.emplace_back(p[pmask[0]]);
        y_main.emplace_back(p[pmask[1]]);
    }

    // get other curve
    for (i = 0; i < nsolutions_other; i++) {
        group = other_file.openGroup(fmt::format("{}", i));
        deserialize(group, "p", p);
        x_other.emplace_back(p[pmask[0]]);
        y_other.emplace_back(p[pmask[1]]);
    }

    for (i = 0; i < nsolutions_other; i++) {
        const double x = x_other[i];
        const double y = y_other[i];
        for (j = 0; j < nsolutions_main - 1; j++) {
            const double x0 = x_main[j + 0];
            const double y0 = y_main[j + 0];
            const double x1 = x_main[j + 1];
            const double y1 = y_main[j + 1];
            if (is_between(x, x0, x1) && is_between(y, y0, y1)) {
                indices.emplace_back(j + 1);
                break;
            }
        }
    }

    return indices;
}

auto create_ordering(const int noriginal, const std::vector<int> where)
{
    std::vector<std::pair<int, int>> modified;
    ptrdiff_t i, j, k, total;

    total = noriginal + where.size();
    modified.reserve(total);

    for (i = 0, j = 0, k = 0; i < total; i++) {
        const auto i2 = where[k];
        if ((j < i2) || (k == (int)where.size())) {
            modified.emplace_back(j++, 0);
        }
        else {
            modified.emplace_back(k++, 1);
        }
    }

    return modified;
}

void insert_data(H5::H5File &main_file, H5::H5File &other_file)
{
    using namespace indicators;
    extern void evaluate_spectrum_for_each_solution_inplace_distributed(const std::vector<std::string> &files, const TWIST::SpectrumStrategy strategy);


    ptrdiff_t i, nsolutions_main, nsolutions_other;
    auto indices = get_insertion_indices(main_file, other_file, nsolutions_main, nsolutions_other);
    std::vector<std::pair<int, int>> ordering;
    std::vector<std::string> paths;
    H5::Group group, group_new;
    H5::PropList object_copy_prop = H5::PropList::DEFAULT;
    H5::PropList link_creation_prop = H5::PropList::DEFAULT;
    bool get_spectrum = main_file.nameExists("0/spectrum") && (!other_file.nameExists("0/spectrum"));

    // get spectrum first
    if (get_spectrum) {
        evaluate_spectrum_for_each_solution_inplace_distributed(std::vector<std::string>{ other_file.getFileName() }, TWIST::SpectrumStrategy::shiftAndInvert);
    }

    ordering = create_ordering(nsolutions_main, indices);

    // move existing data in main file to correct indices
    for (i = ordering.size() - 1; 0 <= i; i--) {
        const auto [index, which] = ordering[i];

        if ((which == 0) && (index != i)) {
            const auto name1 = fmt::format("{}", index);
            const auto name2 = fmt::format("{}", i);
            // main file
            main_file.move(name1, name2);
        }
    }

    // now that original data remapped, add new data
    for (i = 0; i < (ptrdiff_t)ordering.size(); i++) {
        const auto [index, which] = ordering[i];
        if (which == 1) {
            const auto name1 = fmt::format("{}", index);
            const auto name2 = fmt::format("{}", i);
            herr_t status = H5Ocopy(other_file.getId(), name1.c_str(), main_file.getId(), name2.c_str(), object_copy_prop.getId(), link_creation_prop.getId());
            if (status < 0) {
                fmt::println("{} -> {}", index, i);
                throw H5::Exception("twist tools append", "H5Ocopy failed to copy the group.");
            }
        }
    }
}