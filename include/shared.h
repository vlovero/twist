#ifndef TWIST_SHARED_H
#define TWIST_SHARED_H

#include "fmt/core.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>


#define RP(T) T *__restrict

#define ENABLE_MAKE_EQUISPACED 0
#define ENABLE_DUMP_COL_MAT 0
#define ENABLE_DUMP_PENCIL 0
#define ENABLE_SHOW_MONITOR_FUNCTION 0


template <class T, class... Ts>
std::array<T, sizeof...(Ts)> cast_array(Ts &&...ts)
{
    return { { static_cast<T>(ts)... } };
}


// #define TIME_CODE(what_it_is, code)                                                        \
//     {                                                                                      \
//         auto start_time = std::chrono::high_resolution_clock::now();                       \
//         code;                                                                              \
//         auto stop_time = std::chrono::high_resolution_clock::now();                        \
//         double total_time = std::chrono::duration<double>(stop_time - start_time).count(); \
//         fmt::println("{} took {:.3e} seconds", what_it_is, total_time);                    \
//     }

typedef std::vector<std::pair<int, double>> diffusion_t;

typedef void (*func_t)(const double *z, const double *p, double *out);
typedef void (*pjac_t)(const double *z, const double *p, const int *pmask, const int nparam, double *out);


namespace TWIST
{
    // maybe I'll use single precision later?
    // for now just define it as T so I can do `typename T`

    constexpr int max_nparam = 2;
    using T = double;

    enum FileType : int
    {
        continuation,
        spectrum,
        bifurcation_points,
        preprocess,
        simulation
    };

    enum SolutionTypes : uint64_t
    {
        regular = 0,
        saddleNode,
        branchPoint,
        torusPoint
    };

    struct ContinuationBounds
    {
        T ds, dsmin, dsmax, parmin, parmax, geps;
        int min_nodes_adapt, max_nodes_adapt, allow_mesh_adaptation;
    };

    enum SpectrumStrategy
    {
        ggev4 = 0,
        shiftAndInvert,
        ggev3,
        ggev5
    };

    template <typename T>
    T sign(const T x)
    {
        return (x < 0) ? (T)(-1) : (T)1;
    }

    template <typename T>
    void time_code(const std::string_view what_it_is, T lambda_expression)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        lambda_expression();
        auto stop_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(stop_time - start_time).count();
        fmt::println("{} took {:.3} seconds", what_it_is, total_time);
    }
} // namespace TWIST

template <typename T>
struct is_complex_t : public std::false_type
{
};


template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type
{
};

template <typename T>
constexpr bool is_complex()
{
    return is_complex_t<T>::value;
}

template <typename T>
struct type_of_data
{
    using type = T;
};

template <typename T>
struct type_of_data<std::complex<T>>
{
    using type = T;
};

template <typename T>
typename type_of_data<T>::type norm2(const T *x, size_t n)
{
    size_t i;
    std::remove_const_t<typename type_of_data<T>::type> acc = 0;

    for (i = 0; i < n; i++) {
        if constexpr (is_complex<T>()) {
            typename type_of_data<T>::type tmp = std::abs(x[i]);
            acc += tmp * tmp;
        }
        else {
            acc += x[i] * x[i];
        }
    }

    return std::sqrt(acc);
}

template <typename T>
typename type_of_data<T>::type norminf(const T *x, size_t n)
{
    size_t i;
    std::remove_const_t<typename type_of_data<T>::type> acc = 0;

    for (i = 0; i < n; i++) {
        acc = std::max(acc, std::abs(x[i]));
    }

    return acc;
}

template <typename T>
void normalize(T *x, const size_t n)
{
    const auto normx = norm2(x, n);
    for (size_t i = 0; i < n; i++) {
        x[i] /= normx;
    }
}

template <typename T>
T inner(const T *x, const T *y, size_t n)
{
    size_t i;
    std::remove_const_t<T> acc = 0;

    for (i = 0; i < n; i++) {
        if constexpr (is_complex<T>()) {
            acc += std::conj(x[i]) * y[i];
        }
        else {
            acc += x[i] * y[i];
        }
    }

    return acc;
}

std::vector<std::string_view> split(const std::string &text, const std::string &delimeter);
std::string get_program_argv();
char **get_program_raw_argv();
void set_program_args(int argc, char **argv);

#endif // TWIST_SHARED_H
