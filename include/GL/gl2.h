
#ifndef COLLOCATION_GL2_H
#define COLLOCATION_GL2_H

#include <array>

namespace TWIST
{
    namespace GL2
    {
        constexpr size_t c_nstages = 2;
        constexpr std::array<double, c_nstages * c_nstages> c_A = { 0.250, -0.0386751345948128822545743902510, 0.538675134594812882254574390251, 0.250000000000000000000000000000 };
        constexpr std::array<double, c_nstages> c_b = { 0.50, 0.5 };
        constexpr std::array<double, c_nstages> c_c = { 0.211324865405187117745425609749, 0.788675134594812882254574390251 };
        constexpr std::array<double, c_nstages * c_nstages> c_P = { 1.36602540378443864676372317075, -0.866025403784438646763723170753, -0.366025403784438646763723170753, 0.866025403784438646763723170753 };
        constexpr std::array<double, 1> c_chat = { 0.00801875373874480228484928861808 };
    } // namespace GL2
} // namespace TWIST

#endif // COLLOCATION_GL2_H
