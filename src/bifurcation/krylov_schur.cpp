#include "krylov/krylov_schur.h"
#include "linalg.h"
#include <cstddef>

namespace Krylov::better
{
    void generate_random_matrix(double *x, const ptrdiff_t nx, const int seed)
    {
        static std::default_random_engine engine(seed);
        static std::normal_distribution<double> dist;

        std::generate(x, x + nx, [&]() { return dist(engine); });
    }

    // double norminf(const double *x, const ptrdiff_t nx)
    // {
    //     double dnorm = 0.0;
    //     for (ptrdiff_t i = 0; i < nx; i++) {
    //         dnorm = std::max(std::abs(x[i]), dnorm);
    //     }
    //     return dnorm;
    // }

    bool leading_eigenvalue_is_complex(const double *H, const ptrdiff_t ldH)
    {
        double a, b, c, d, trace, det, delta;

        a = H[0];
        c = H[1];
        b = H[ldH];
        d = H[ldH + 1];

        trace = a + d;
        det = a * d - b * c;
        delta = trace * trace - 4 * det;

        return delta < 0;
    }


    KSResult::~KSResult()
    {
        if (H) {
            free(H);
        }
        H = nullptr;
        if (Q) {
            free(Q);
        }
        Q = nullptr;
        if (work) {
            free(work);
        }
        work = nullptr;
        ldH = 0;
        ldQ = 0;
        nconverged = 0;
    }

    ptrdiff_t KSResult::getLAPACKLWork(const ptrdiff_t n, const ptrdiff_t nsub) const
    {
        return nsub * (7 + n + 2 * nsub + 1) + 7 + n;
    }

    ptrdiff_t KSResult::getLWork(const ptrdiff_t n, const ptrdiff_t nsub) const
    {
        /*
            all work needed
                r, b, bhat, mags, select, Qhat, Hhat, wr, wi
                7 * nsub + n * nsub + (nsub + 1) * nsub + nsub^2
                nsub * (7 + n + 2 * nsub + 1) + 7 + n;
        */
        int lapack_lwork, sdim[1], info[1];
        double dummy[1];

        lapack_lwork = 0;
        dgees("V", "N", NULL, nsub, NULL, nsub, sdim, NULL, NULL, NULL, nsub, dummy, -1, NULL, info);
        lapack_lwork += std::max<int>(dummy[0], nsub);
        // dtrsen("N", "V", NULL, nsub, NULL, nsub, NULL, nsub, NULL, NULL, sdim, NULL, NULL, dummy, -1, idummy, -1, info);
        lapack_lwork += nsub;
        lapack_lwork += 1;

        return lapack_lwork + getLAPACKLWork(n, nsub);
    }

    void KSResult::reserve(const ptrdiff_t n, const ptrdiff_t nsub)
    {
        H = (double *)realloc(H, nsub * (nsub + 1) * sizeof(double));
        memset(H, 0, nsub * (nsub + 1) * sizeof(double));
        ldH = nsub + 1;
        Q = (double *)realloc(Q, n * (nsub + 1) * sizeof(double));
        memset(Q, 0, n * (nsub + 1) * sizeof(double));
        ldQ = n;
        nconverged = 0;
        lwork = getLWork(n, nsub);
        work = (double *)realloc(work, lwork * sizeof(double));
    }

    void KSResult::getEigenvalues(RP(double) wr, RP(double) wi) const
    {
        int sdim[1], info[1];
        // print_forder_mat<16>("H", H, ldH, nconverged, nconverged);

        dgees("N", "N", NULL, nconverged, H, ldH, sdim, wr, wi, NULL, nconverged, work, lwork, NULL, info);
    }

    void KSResult::getEigenvectors(double *V)
    {
        // V = QW
        int info[1];
        const int n = ldQ;
        double *wr = work;
        double *wi = wr + nconverged;
        double *W = wi + nconverged;

        dgeev("N", "V", nconverged, H, ldH, wr, wi, NULL, nconverged, W, nconverged, work + nconverged * (2 + nconverged), lwork - nconverged * (2 + nconverged), info);
        dgemm("N", "N", n, nconverged, nconverged, 1.0, Q, ldQ, W, nconverged, 0.0, V, n);
    }
} // namespace Krylov::better