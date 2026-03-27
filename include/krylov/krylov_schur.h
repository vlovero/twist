#pragma once

#include "fmt/base.h"
#include "fmt/core.h"
#include "linalg.h"
#include "shared.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <random>
#include <stdexcept>

extern "C" void dhseqr_(const char *job, const char *compz, const int *n, const int *ilo, const int *ihi, double *h, const int *ldh, double *wr, double *wi, double *z, const int *ldz, double *work, const int *lwork, int *info);


namespace Krylov::better
{

    void generate_random_matrix(double *x, const ptrdiff_t nx, const int seed);
    // double norminf(const double *x, const ptrdiff_t nx);
    bool leading_eigenvalue_is_complex(const double *H, const ptrdiff_t ldH);

    struct KSResult
    {
        double *H = 0;
        ptrdiff_t ldH = 0;
        double *Q = 0;
        ptrdiff_t ldQ = 0;
        ptrdiff_t nconverged = 0;
        ptrdiff_t lwork = 0;
        double *work = 0;

        KSResult() = default;
        ~KSResult();

        ptrdiff_t getLAPACKLWork(const ptrdiff_t n, const ptrdiff_t nsub) const;
        ptrdiff_t getLWork(const ptrdiff_t n, const ptrdiff_t nsub) const;
        void reserve(const ptrdiff_t n, const ptrdiff_t nsub);
        void getEigenvalues(RP(double) wr, RP(double) wi) const;
        void getEigenvectors(double *V);
    };

    template <int ndigit = 3>
    void print_forder_mat(const char *name, double *A, size_t ldA, size_t N, size_t M)
    {
        printf("%s = \n", name);
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                fmt::print("{: .{}e} ", A[j * ldA + i], ndigit);
            }
            puts("");
        }
        puts("");
    }

    template <int ndigit = 3>
    void print_mat(const char *name, const double *A, const ptrdiff_t ldA, const ptrdiff_t n, const ptrdiff_t m, const bool forder = true)
    {
        ptrdiff_t i, j;

        if (name != nullptr) {
            fmt::println("{} = ", name);
        }
        if (forder) {
            for (j = 0; j < m; j++) {
                for (i = 0; i < n; i++) {
                    fmt::println("\x1B[{}C{: .3e}", j * (9 + 3), A[i + j * ldA]);
                }
                if (j != (m - 1)) {
                    for (i = 0; i < n; i++) {
                        fmt::print("\x1b[1A");
                    }
                }
            }
        }
        else {
            for (i = 0; i < n; i++) {
                fmt::println("{: .{}e}", fmt::join(&A[i * ldA], &A[i * ldA + m], " "), ndigit);
            }
        }
    }


    template <typename Op_t>
    void next_arnoldi(Op_t Ax, const ptrdiff_t n, const ptrdiff_t k, RP(double) Q, const ptrdiff_t ldQ, RP(double) H, const ptrdiff_t ldH, RP(double) r)
    {
        ptrdiff_t i;
        double normv;
        double *v = &Q[(k + 1) * ldQ];
        double *w = &H[k * ldH];

        Ax(&Q[k * ldQ], v);
        dgemv("C", n, k + 1, +1.0, Q, ldQ, v, 1, 0.0, w, 1);
        dgemv("N", n, k + 1, -1.0, Q, ldQ, w, 1, 1.0, v, 1);
        dgemv("C", n, k + 1, +1.0, Q, ldQ, v, 1, 0.0, r, 1);
        dgemv("N", n, k + 1, -1.0, Q, ldQ, r, 1, 1.0, v, 1);

        for (i = 0; i < (k + 1); i++) {
            w[i] += r[i];
        }
        normv = norm2(v, n);
        if (normv == 0.0) {
            throw std::runtime_error("figure this out later");
        }
        w[k + 1] = normv;
        for (i = 0; i < n; i++) {
            v[i] /= normv;
        }
    }


    template <typename Op_t>
    void krylov_schur(Op_t Ax, const ptrdiff_t n, const ptrdiff_t neig, const ptrdiff_t nsub, KSResult &result, const double tol, const bool verbose = true, const int seed = 69, const int maxiter = 1000)
    {
        double *H, *Q, *work, *r, *b, *mags, *wr, *wi, *bhat, *Qhat, *Hhat, *U, *lapack_work;
        ptrdiff_t ldH, ldQ, p, s, i, j, ndim, is_c;
        int sdim[1], info[1];
        int *select, lapack_lwork;
        double eps, umax;

        result.nconverged = 0; // for now just clear them
        if (result.nconverged == 0) {
            result.reserve(n, nsub);
        }

        // get H and Q that will store schur decomp
        H = result.H;
        ldH = result.ldH;
        Q = result.Q;
        ldQ = result.ldQ;

        // assign all workspace arrays
        work = result.work;
        r = work;
        b = r + nsub + 1;
        mags = b + nsub + 1;
        wr = mags + nsub + 1;
        wi = wr + nsub + 1;
        Qhat = wi + nsub + 1;
        Hhat = Qhat + n * nsub;
        select = (int *)(Hhat + (nsub * (nsub + 1)));
        U = ((double *)select) + (nsub + 1);
        lapack_work = U + nsub * nsub;
        lapack_lwork = result.lwork - result.getLAPACKLWork(n, nsub);
        bhat = mags;

        // init number of converged eigenvalues and start index for arnoldi to zero
        p = 0;
        s = 0;

        // generate initial vector
        generate_random_matrix(Q, n, seed);
        umax = norm2(Q, n);
        for (j = 0; j < n; j++) {
            Q[j] /= umax;
        }

        for (i = 0; i < maxiter; i++) {
            // expand arnoldi
            for (j = s; j < nsub; j++) {
                next_arnoldi(Ax, n, j, Q, ldQ, H, ldH, r);
            }
            dlacpy("A", 1, nsub - p, &H[p * ldH + nsub], ldH, b, 1);

            // compute schur
            ndim = nsub - p;
            dgees("V", "N", NULL, ndim, &H[p * ldH + p], ldH, sdim, wr, wi, U, ndim, lapack_work, lapack_lwork, NULL, info);
            // sort schur
            for (j = 0; j < ndim; j++) {
                mags[j] = std::hypot(wr[j], wi[j]);
            }
            // need stable sort to preserve order of real and complex parts
            std::stable_sort(mags, mags + ndim, std::greater<double>());
            umax = mags[0];
            for (j = 0; j < ndim; j++) {
                select[j] = std::hypot(wr[j], wi[j]) >= mags[neig];
            }
            dtrsen("N", "V", select, ndim, &H[p * ldH + p], ldH, U, ndim, wr, wi, sdim, NULL, NULL, lapack_work, ndim, (int *)(&lapack_work[ndim]), lapack_lwork - ndim, info);

            // get bhat
            dgemv("C", ndim, ndim, 1.0, U, ndim, b, 1, 0.0, bhat, 1);
            // update H
            if (0 < p) {
                dgemm("N", "N", p, ndim, ndim, 1.0, &H[p * ldH], ldH, U, ndim, 0.0, Hhat, p);
                dlacpy("A", p, ndim, Hhat, p, &H[p * ldH], ldH);
            }

            // check if complex
            is_c = leading_eigenvalue_is_complex(&H[p * ldH + p], ldH);
            // update s
            s = p + 1 + is_c;

            if (verbose) {
                fmt::println("{:4d} : s = {:3d} : p = {:3d} : {} | {: .8e}", i + 1, s, p, is_c, *bhat);
            }

            // update Q
            dgemm("N", "N", n, ndim, ndim, 1.0, &Q[p * ldQ], ldQ, U, ndim, 0.0, Qhat, n);
            dlacpy("A", n, ndim, Qhat, n, &Q[p * ldQ], ldQ);
            // truncate H
            dlacpy("A", 1, ndim, bhat, 1, &H[p * ldH + s], ldH);
            // truncate Q
            memcpy(&Q[s * ldQ], &Q[nsub * ldQ], n * sizeof(double));

            // check for convergence
            eps = std::max(norminf(H, nsub * (nsub + 1)) * 2e-16, umax * tol);
            if (norminf(bhat, 1 + is_c) < eps) {
                // clear converged part of truncated H
                for (j = 0; j < (1 + is_c); j++) {
                    H[(p + j) * ldH + s] = 0;
                }
                // increment converged count
                p += 1 + is_c;
            }
            if (neig <= p) {
                break;
            }
        }
        result.nconverged = p;
    }
} // namespace Krylov::better
