
#ifndef COLLOCATION_GL1_H
#define COLLOCATION_GL1_H

#include <array>

namespace TWIST
{
    namespace GL1
    {
        constexpr size_t c_nstages = 1;
        constexpr std::array<double, c_nstages * c_nstages> c_A = { 0.5 };
        constexpr std::array<double, c_nstages> c_b = { 1.0 };
        constexpr std::array<double, c_nstages> c_c = { 0.5 };
        constexpr std::array<double, c_nstages * c_nstages> c_P = { 1.0 };
        constexpr std::array<double, 1> c_chat = { 0.125000000000000000000000000000 };
    } // namespace GL1
} // namespace TWIST

#endif // COLLOCATION_GL1_H
