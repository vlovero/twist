#pragma once
#include "collocator.h"
#include <complex>
#include <cstddef>

struct PieceWiseLinearContour
{
    const ptrdiff_t m_n;
    const double *m_x;
    const double *m_y;

    std::complex<double> gamma(double t) const;
    std::complex<double> gammaPrime(double t) const;
    double arcLength() const;
};

struct FEASTResult
{
    using complex_t = std::complex<double>;
    ptrdiff_t n = 0;
    std::vector<complex_t> evals;
    std::vector<complex_t> evecs;

    FEASTResult() = default;

    void clear();
    void update(const double tol, const ptrdiff_t n, const ptrdiff_t neig, const complex_t *k, const complex_t *V);
};

FEASTResult feast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol);
FEASTResult bfeast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol);
FEASTResult nlfeast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol);