#pragma once

#include "collocator.h"
#include <valarray>

namespace BranchPoint
{
    void shift_to_origin(RP(double) alphar, RP(double) alphai, RP(double) beta, const ptrdiff_t N);
    void get_real_unstable(std::vector<double> &unstable, const double *alphar, const double *alphai, const double *beta, const ptrdiff_t N);
    void refine_eigenpair(double &k, std::valarray<double> &z, TWIST::Collocator *collocator, DEV::Matrix &A, DEV::Matrix &B, DEV::BoundaryConditions *bc, DEV::BoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential);
    void get_initial_eigenvector(const double k, std::valarray<double> &v, TWIST::Collocator *collocator, DEV::Matrix &A, DEV::Matrix &B, DEV::BoundaryConditions *bc, DEV::BoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk);

} // namespace BranchPoint