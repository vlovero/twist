#pragma once

#include "collocator.h"
#include "colmat/tcolmat.h"
#include <random>
#include <valarray>

namespace Hopf
{
    using complex_t = std::complex<double>;

    template <typename T>
    void generate_random_matrix(T &first, T &last, const int seed)
    {
        std::default_random_engine engine(seed);
        // std::uniform_real_distribution dist(0, 1);
        std::normal_distribution<double> dist;
        std::generate(first, last, [&]() { return dist(engine); });
    }

    template <typename T>
    void generate_random_matrix(T first, T last, const int seed)
    {
        std::default_random_engine engine(seed);
        // std::uniform_real_distribution dist(0, 1);
        std::normal_distribution<double> dist;
        std::generate(first, last, [&]() { return dist(engine); });
    }

    int count_complex_pairs(const double *alphar, const double *alphai, const double *beta, const size_t N);
    int count_unstable(const double *alphar, const double *alphai, const double *beta, const size_t N);

    void get_unstable_pairs(std::vector<complex_t> &unstable, const double *alphar, const double *alphai, const double *beta, const size_t N);
    void setup_M_and_D(const ptrdiff_t node, double *M, double *D, const std::vector<int> &to_delete, const diffusion_t &diffusion, const double L, const double speed);
    void add_BBlock_to_partition(const ptrdiff_t node, const ptrdiff_t nstages, complex_t *block, const ptrdiff_t ldblock, const complex_t mu, const double *M, const double *A, const double *b);
    void set_BBlock_as_partition(const ptrdiff_t node, const ptrdiff_t nstages, complex_t *block, const ptrdiff_t ldblock, const complex_t mu, const double *M, const double *A, const double *b);
    void generate_matrices(const complex_t k, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential = false);
    void get_initial_eigenvector(const complex_t k, std::valarray<complex_t> &v, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk);
    void refine_eigenpair(complex_t &k, std::valarray<complex_t> &z, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential = false);

    void add_BBlock_to_partition(const ptrdiff_t node, const ptrdiff_t nstages, double *block, const ptrdiff_t ldblock, const double mu, const double *M, const double *A, const double *b);
}; // namespace Hopf
