#pragma once


#include "fmt/base.h"
#include "fmt/core.h"
#include "linalg.h"
#include "shared.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

extern "C" void zgesdd_(const char *jobz, const int *m, const int *n, std::complex<double> *A, const int *lda, double *S, std::complex<double> *U, const int *ldu, std::complex<double> *VT, const int *ldvt, std::complex<double> *work, const int *lwork, double *rwork, int *iwork, int *info);

namespace Krylov
{
    namespace internal
    {
        template <typename data_t>
        void gemm(const char *transa, const char *transb, const int nrow_op_a, const int ncol_op_b, const int nrow_op_b, const data_t alpha, const data_t *a, const int lda, const data_t *b, const int ldb, const data_t beta, data_t *c, const int ldc)
        {
            if constexpr (std::is_same_v<data_t, double>) {
                dgemm_(transa, transb, &nrow_op_a, &ncol_op_b, &nrow_op_b, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zgemm_(transa, transb, &nrow_op_a, &ncol_op_b, &nrow_op_b, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        }


        template <typename data_t>
        void gesdd(const char *jobz, const int m, const int n, data_t *A, const int lda, typename type_of_data<data_t>::type *S, data_t *U, const int ldu, data_t *VT, const int ldvt, data_t *work, int lwork, int *iwork, int *info)
        {
            using real_t = typename type_of_data<data_t>::type;
            int mn, mx, lrwork;
            real_t *rwork = (real_t *)work;

            mn = std::min(m, n);
            mx = std::max(m, n);
            lrwork = std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn);

            if (lwork != -1) {
                lwork -= lrwork;
                work += lrwork;
            }

            if constexpr (std::is_same_v<data_t, double>) {
                dgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, iwork, info);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, rwork, iwork, info);
            }

            if (lwork == -1) {
                work[0] += (real_t)lrwork;
            }
        }
    } // namespace internal

    template <typename A_Op_t, typename M_Op_t, typename data_t>
    int bicgstab(A_Op_t &A, M_Op_t &M, size_t n, RP(data_t) x, const RP(data_t) b, RP(data_t) work, int maxiter, typename type_of_data<data_t>::type atol, typename type_of_data<data_t>::type rhotol, typename type_of_data<data_t>::type omegatol, int *niters, const bool verbose = false)
    {
        data_t *r, *p, *v, *s, *t, *rtilde, *phat, *shat;
        data_t rho, rho_prev, omega, alpha, beta, rv;
        size_t i;
        int itern;
        const bool allocate_work = work == nullptr;

        if (allocate_work) {
            work = (data_t *)malloc(8 * n * sizeof(data_t));
        }

        // need 8
        r = work;
        p = r + n;
        v = p + n;
        s = v + n;
        t = s + n;
        rtilde = t + n;
        phat = rtilde + n;
        shat = phat + n;

        if (A(x, r, n)) {
            *niters = 0;
            if (allocate_work) {
                free(work);
            }
            return -3;
        }
        for (i = 0; i < n; i++) {
            r[i] = b[i] - r[i];
            rtilde[i] = r[i];
        }

        for (itern = 0; itern < maxiter; itern++) {
            if (verbose) {
                fmt::println("[BiCGStab] {:4d} : |r| = {:.8e}", itern, norm2(r, n));
            }
            if (norm2(r, n) < atol) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return 0;
            }

            rho = inner(rtilde, r, n);
            if (std::abs(rho) < rhotol) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -10;
            }

            if (itern) {
                if (std::abs(omega) < omegatol) {
                    *niters = itern;
                    if (allocate_work) {
                        free(work);
                    }
                    return -11;
                }

                beta = (rho / rho_prev) * (alpha / omega);
                for (i = 0; i < n; i++) {
                    p[i] -= omega * v[i];
                    p[i] *= beta;
                    p[i] += r[i];
                }
            }
            else {
                for (i = 0; i < n; i++) {
                    p[i] = r[i];
                }
            }

            if (M(p, phat, n)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -2;
            }
            if (A(phat, v, n)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -3;
            }
            rv = inner(rtilde, v, n);
            if (rv == (data_t)0) {
                *niters = itern + 1;
                if (allocate_work) {
                    free(work);
                }
                return -11;
            }

            alpha = rho / rv;
            for (i = 0; i < n; i++) {
                r[i] -= alpha * v[i];
                s[i] = r[i];
            }

            if (norm2(s, n) < atol) {
                for (i = 0; i < n; i++) {
                    x[i] += alpha * phat[i];
                }
                *niters = itern + 1;
                if (allocate_work) {
                    free(work);
                }
                return 0;
            }

            if (M(s, shat, n)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -2;
            }
            if (A(shat, t, n)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -3;
            }
            omega = inner(t, s, n) / inner(t, t, n);
            for (i = 0; i < n; i++) {
                x[i] += alpha * phat[i];
                x[i] += omega * shat[i];
                r[i] -= omega * t[i];
            }
            rho_prev = rho;
        }
        *niters = maxiter;

        if (allocate_work) {
            free(work);
        }
        return -1;
    }

    template <typename A_Op_t, typename M_Op_t, typename data_t>
    int block_bicgstab(A_Op_t A, M_Op_t M, int n, int nrhs, data_t *__restrict X, const data_t *__restrict B, data_t *__restrict work, int &lwork, int maxiter, typename type_of_data<data_t>::type atol, typename type_of_data<data_t>::type rhotol, int *niters, const bool verbose)
    {
        // constexpr bool use_complex = is_complex<data_t>();
        using real_t = typename type_of_data<data_t>::type;
        const int n_nrhs = n * nrhs;
        const int nrhs2 = nrhs * nrhs;
        // const bool work_query = lwork == -1;
        const bool allocate_work = work == nullptr;

        int itern, iwork[8 * nrhs], info;
        ptrdiff_t i, j, rank;
        data_t tmp;
        real_t omega, sigma[nrhs];
        // these are all (n, nrhs)
        data_t *R, *Rt, *P, *T, *V, *Q, *S;
        // these are all (nrhs, nrhs)
        data_t *O, *D, *Z_VT, *K, *Z;
        // this has to be queried
        data_t *svd_work;

        iwork[0] = 7 * (n + 5 * nrhs) * nrhs;
        internal::gesdd<data_t>("O", nrhs, nrhs, NULL, nrhs, sigma, NULL, nrhs, NULL, nrhs, &tmp, -1, iwork, &info);
        iwork[0] += (int)std::real(tmp);

        if (lwork == -1) {
            lwork = iwork[0];
            return 0;
        }
        else if (allocate_work) {
            work = (data_t *)malloc(iwork[0] * sizeof(data_t));
        }
        else if (lwork < iwork[0]) {
            throw std::runtime_error("don't forget to allocate work for block BiCGStab");
        }

        (void)rhotol;

        R = work;
        Rt = R + n_nrhs;
        P = Rt + n_nrhs;
        T = P + n_nrhs;
        V = T + n_nrhs;
        Q = V + n_nrhs;
        S = Q + n_nrhs;
        O = S + n_nrhs;
        D = O + nrhs2;
        Z_VT = D + nrhs2;
        K = Z_VT + nrhs2;
        Z = K + nrhs2;
        svd_work = Z + nrhs2;

        if (A(X, R, n, nrhs)) {
            *niters = 0;
            if (allocate_work) {
                free(work);
            }
            return -1;
        }
        for (i = 0; i < n_nrhs; i++) {
            R[i] = B[i] - R[i];
            Rt[i] = R[i];
        }

        for (itern = 0; itern < maxiter; itern++) {
            if (verbose) {
                fmt::println("[BiCGStab] {:4d} : |r| = {:.8e}", itern, norm2(R, n));
            }

            // check residual norm
            if (norm2(R, n_nrhs) < atol) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return 0;
            }

            if (itern) {
                // O = -(Rt^T * V)^{-1} (Rt^T * T)
                // O = -Rt^T T
                internal::gemm<data_t>("C", "N", nrhs, nrhs, n, -1.0, Rt, n, T, n, 0.0, O, nrhs);

                // (R^T V)^{-1} already available so don't comute svd again

                // solve (U sigma VT)^{-1} = VT^T sigma^{-1} U^T
                // apply U^T (U^T)
                internal::gemm<data_t>("C", "N", nrhs, nrhs, nrhs, 1.0, Z, nrhs, O, nrhs, 0.0, D, nrhs);
                // apply sigma^{-1}
                for (i = 0; i < nrhs; i++) {
                    for (j = 0; j < rank; j++) {
                        D[i * nrhs + j] /= sigma[j];
                    }
                    for (; j < nrhs; j++) {
                        D[i * nrhs + j] = 0;
                    }
                }
                // apply Zh^T
                internal::gemm<data_t>("C", "N", nrhs, nrhs, nrhs, 1.0, Z_VT, nrhs, D, nrhs, 0.0, O, nrhs);

                // P = R + (P - oemga * V) O
                for (i = 0; i < nrhs; i++) {
                    for (j = 0; j < n; j++) {
                        T[i * n + j] = P[i * n + j] - omega * V[i * n + j];
                    }
                }
                // P = R + T O
                internal::gemm<data_t>("N", "N", n, nrhs, nrhs, 1.0, T, n, O, nrhs, 1.0, P, n);
            }
            else {
                for (i = 0; i < n_nrhs; i++) {
                    P[i] = R[i];
                }
            }

            // D = R^T R
            internal::gemm<data_t>("C", "N", nrhs, nrhs, n, 1.0, R, n, R, n, 0.0, D, nrhs);
            // compute Q = MP
            if (M(P, Q, n, nrhs)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -2;
            }
            // compute V = AQ
            if (A(Q, V, n, nrhs)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -3;
            }
            // K = (R^T V)^{-1} D
            {
                // Z = Rt^T V
                internal::gemm<data_t>("C", "N", nrhs, nrhs, n, 1.0, Rt, n, V, n, 0.0, Z, nrhs);
                // Z = U Sigma VT; U is stored in Z
                internal::gesdd<data_t>("O", nrhs, nrhs, Z, nrhs, sigma, nullptr, nrhs, Z_VT, nrhs, svd_work, lwork, iwork, &info);
                // determine rank
                for (i = 0, rank = 0; i < nrhs; i++, rank++) {
                    if (sigma[i] < (sigma[0] * std::numeric_limits<real_t>::epsilon())) {
                        break;
                    }
                }
                // apply U^T to D
                internal::gemm<data_t>("C", "N", nrhs, nrhs, nrhs, 1.0, Z, nrhs, D, nrhs, 0.0, O, nrhs);
                // apply sigma^{-1}
                for (i = 0; i < nrhs; i++) {
                    for (j = 0; j < rank; j++) {
                        O[i * nrhs + j] /= sigma[j];
                    }
                    for (; j < nrhs; j++) {
                        O[i * nrhs + j] = 0;
                    }
                }
                internal::gemm<data_t>("C", "N", nrhs, nrhs, nrhs, 1.0, Z_VT, nrhs, O, nrhs, 0.0, K, nrhs);
            }
            // X = X + QK
            internal::gemm<data_t>("N", "N", n, nrhs, nrhs, 1.0, Q, n, K, nrhs, 1.0, X, n);
            // R = R - VK
            internal::gemm<data_t>("N", "N", n, nrhs, nrhs, -1.0, V, n, K, nrhs, 1.0, R, n);

            if (norm2(R, n * nrhs) < atol) {
                if (verbose) {
                    fmt::println("finished with ||R|| = {:.8e}", norm2(R, n * nrhs));
                }
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return 0;
            }

            // S = MR
            if (M(R, S, n, nrhs)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -4;
            }
            // T = AS
            if (A(S, T, n, nrhs)) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -5;
            }

            // omega = |T^T R| / |T^T T|
            {
                internal::gemm<data_t>("C", "N", nrhs, nrhs, n, 1.0, T, n, R, n, 0.0, O, nrhs);
                omega = norm2(O, nrhs * nrhs);
                internal::gemm<data_t>("C", "N", nrhs, nrhs, n, 1.0, T, n, T, n, 0.0, O, nrhs);
                omega /= norm2(O, nrhs * nrhs);
            }
            if (omega < (std::numeric_limits<real_t>::epsilon() * std::numeric_limits<real_t>::epsilon())) {
                *niters = itern;
                if (allocate_work) {
                    free(work);
                }
                return -11;
            }

            // X += omega * S
            // R -= omega * T
            for (i = 0; i < nrhs; i++) {
                for (j = 0; j < n; j++) {
                    X[i * n + j] += omega * S[i * n + j];
                    R[i * n + j] -= omega * T[i * n + j];
                }
            }
            if (norm2(R, n * nrhs) < atol) {
                *niters = itern + 1;
                if (allocate_work) {
                    free(work);
                }
                return 0;
            }
        }
        return maxiter;
    }
} // namespace Krylov