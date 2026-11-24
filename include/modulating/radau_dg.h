#pragma once

#include <cstddef>

namespace TWIST
{
    struct RadauDGMethod
    {
        size_t st;                  // number of temporal stages (2, 3, or 4)
        const double *c;            // length st, Radau IIA nodes in (0, 1], c[st-1] == 1
        const double *b;            // length st, quadrature weights (last row of A)
        const double *A;            // st x st, row-major Butcher matrix A^R
        const double *A_inv;        // st x st, row-major (A^R)^(-1)
        const double *A_inv_ones;   // length st, (A^R)^(-1) * 1
        const double *T;            // st x st, row-major real transformation T_s
        const double *T_inv;        // st x st, row-major T_s^(-1)
        size_t n_real;              // number of real eigenvalues of (A^R)^(-1)
        size_t n_complex;           // number of complex conjugate pairs of (A^R)^(-1)
        const double *real_eigs;    // length n_real
        const double *complex_re;   // length n_complex
        const double *complex_im;   // length n_complex (positive imaginary parts)
        const double *D_tau;        // st x (st + 1), row-major temporal derivative matrix at c_l from {0, c_1, ..., c_st}
        const double *err_weights;  // length st + 1, divided-difference weights for leading degree-st term
        double leading_err_scale;   // 1 / (st!) scale for temporal error monitor
        const double *P_in;         // st x st, row-major T_s^(-1) * (A^R)^(-1)
        const double *gamma_ones;   // length st, T_s^(-1) * (A^R)^(-1) * 1_{st}
    };

    const RadauDGMethod &getRadauDG(size_t st);

    // Evaluates the (st + 1) Lagrange interpolation weights on {0, c_1, ..., c_st} at theta in [0, 1].
    void radauDenseWeights(size_t st, double theta, double *weights_out);
} // namespace TWIST
