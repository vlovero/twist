#ifndef TWIST_CONTINUATION_H
#define TWIST_CONTINUATION_H

#include <sys/ioctl.h>

#include "collocator.h"
#include "fmt/core.h"
#include "shared.h"


struct LocationBar
{
    using time_point_t = decltype(std::chrono::high_resolution_clock::now());
    double m_a;
    double m_b;
    ptrdiff_t m_pos;
    ptrdiff_t m_nwhite;
    ptrdiff_t m_offset;
    std::string m_pbar;
    time_point_t m_prev_checkpoint;
    time_point_t m_curr_checkpoint;

    LocationBar(double a, double b);
    void update(const double y, const std::string &tail = "");
};

void continuation_work_loop(TWIST::Collocator &collocator, int parnum, TWIST::ContinuationBounds &bounds, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, const bool get_nullspace = true);

#endif // TWIST_CONTINUATION_H