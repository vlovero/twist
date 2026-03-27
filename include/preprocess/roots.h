#ifndef TWIST_PREPROCESS_ROOTS_H
#define TWIST_PREPROCESS_ROOTS_H

#include "fmt/core.h"
#include "shared.h"


template <typename func_t, typename fjac_t>
void fsolve(func_t func, fjac_t fjac, const size_t nx, RP(double) x, const double atol, const double rtol, const double *args)
{
    // convert into arguments later
    constexpr int max_iter = 1000;
    constexpr bool verbose = true;

    const double tol = std::max(rtol * norm2(x, nx), atol);

    int itern, info, lwork_lapack;
    double normfx, normfxb, normd, normdb, damp;
    double dmin, dmax, temp, rcond, threshold;
    bool use_least_squares;
    size_t k;
    int ipiv[8 * nx];

    double *rwork = nullptr;
    double *fx = nullptr;
    double *fxb = nullptr;
    double *xb = nullptr;
    double *Jx = nullptr;
    double *Jx2 = nullptr;
    double *U = nullptr;
    double *S = nullptr;
    double *VT = nullptr;
    double *work_lapack = nullptr;
    double *tmp = nullptr;
    int lrwork = (4 * nx) + (4 * nx * nx);
    dgesdd("S", nx, nx, Jx2, nx, S, U, nx, VT, nx, &temp, -1, ipiv, &info);
    lwork_lapack = temp;
    lrwork += lwork_lapack;
    rwork = (double *)malloc(lrwork * sizeof(double));

    fx = rwork;
    fxb = fx + nx;
    xb = fxb + nx;
    Jx = xb + nx;
    Jx2 = Jx + (nx * nx);
    U = Jx2 + (nx * nx);
    S = U + (nx * nx);
    VT = S + nx;
    work_lapack = VT + (nx * nx);
    tmp = work_lapack + lwork_lapack;

    func(x, args, fx);
    normfx = norm2(fx, nx);

    if (verbose) {
        fmt::println(" {:^5s} | {:^14s} | {:^14s} |", "iter", "||F||", "||dx||");
        fmt::println("{:-^42s}", "");
        fmt::println(" {:5d} | {:.8e} | {:.8e} |", 0, normfx, 0.0);
    }

    if (normfx < tol) {
        free(rwork);
        return;
    }

    damp = 1.0;
    for (itern = 0; itern < max_iter; itern++) {
        // compute jacobian
        fjac(x, args, Jx);
        memcpy(Jx2, Jx, nx * nx * sizeof(double));

        // estimate threshold for numerically singular
        threshold = norm2(Jx, nx * nx) * std::numeric_limits<double>::epsilon();
        // compute LU factorization
        dgetrf(nx, nx, Jx, nx, ipiv, &info);

        // estimate rcond
        dmin = std::numeric_limits<double>::infinity();
        dmax = 0.0;
        for (k = 0; k < nx; k++) {
            temp = std::abs(Jx[k * (nx + 1)]);
            dmin = std::min(dmin, temp);
            dmax = std::max(dmax, temp);
        }
        rcond = dmin / dmax;
        use_least_squares = rcond < threshold;

        // dx is stored in fx
        if (!use_least_squares) {
            // not numerically singular -> use LU
            dgetrs("T", nx, 1, Jx, nx, ipiv, fx, nx, &info);
        }
        else {
            // was numerically singular -> compute SVD
            // Jx2 = U S VT
            dgesdd("S", nx, nx, Jx2, nx, S, U, nx, VT, nx, work_lapack, lwork_lapack, ipiv, &info);

            // dx = ((VT)^T S^{-1} U^T)fx
            dgemv("T", nx, nx, 1.0, U, nx, fx, 1, 0.0, tmp, 1);
            for (k = 0; k < nx; k++) {
                if (threshold < S[k]) {
                    tmp[k] /= S[k];
                }
                else {
                    tmp[k] = 0;
                }
            }
            dgemv("T", nx, nx, 1.0, VT, nx, tmp, 1, 0.0, fx, 1);
        }

        normd = norm2(fx, nx);

        damp = std::min(2 * damp, 1.0);
        while (damp > pow(0.5, 15)) {
            // take (damped) step
            for (k = 0; k < nx; k++) {
                xb[k] = x[k] - damp * fx[k];
            }

            // compute residual at next iterate
            func(xb, args, fxb);
            normfxb = norm2(fxb, nx);
            if (!std::isfinite(normfxb)) {
                damp *= 0.5;
                continue;
            }
            memcpy(fx, fxb, nx * sizeof(double));

            // solve for \bar{dx}
            if (!use_least_squares) {
                dgetrs("T", nx, 1, Jx, nx, ipiv, fxb, nx, &info);
            }
            else {
                dgemv("T", nx, nx, 1.0, U, nx, fxb, 1, 0.0, tmp, 1);
                for (k = 0; k < nx; k++) {
                    if (threshold < S[k]) {
                        tmp[k] /= S[k];
                    }
                    else {
                        tmp[k] = 0;
                    }
                }
                dgemv("T", nx, nx, 1.0, VT, nx, tmp, 1, 0.0, fxb, 1);
            }
            normdb = norm2(fxb, nx);

            // monotinicity test
            if (normdb <= ((1 - 0.5 * damp) * normd)) {
                break;
            }
            damp *= 0.5;
        }

        // update states
        memcpy(x, xb, nx * sizeof(double));
        normfx = normfxb;

        if (verbose) {
            fmt::println(" {:5d} | {:.8e} | {:.8e} |", itern + 1, normfx, normd);
        }

        // convergence test
        if ((normfx < (tol * tol)) || (normd < tol)) {
            break;
        }
    }
    free(rwork);
}

/*
def dnewton(f, x0, args=(), fjac=None, xtol=pow(0.5, 26), ftol=0.0, maxiter=1000):
    xn = np.array(x0)
    fn = f(xn, *args)
    normfn = np.linalg.norm2(fn)
    if normfn <= ftol:
        return xn

    damp = 1.0
    do_least_squares = False
    for itern in range(maxiter):
        jac = fjac(xn, *args)
        LU = lu_factor(jac)
        d = abs(LU[0].diagonal())
        cond = np.linalg.norm2(jac.ravel()) * pow(2, -52)
        if (d.min() / d.max()) < cond:
            do_least_squares = True
        if not do_least_squares:
            dx = lu_solve(LU, fn)
        else:
            dx = np.linalg.lstsq(jac, fn, rcond=cond)[0]
        normdx = np.linalg.norm2(dx)

        damp = min(2 * damp, 1.0)
        while damp > pow(0.5, 15):
            xn1 = xn - damp * dx
            fn1 = f(xn1, *args)
            if not do_least_squares:
                db = lu_solve(LU, fn1)
            else:
                db = np.linalg.lstsq(jac, fn1, rcond=cond)[0]
            normdb = np.linalg.norm2(db)
            cond = normdb <= ((1 - 0.5 * damp) * normdx)
            if cond:
                break
            damp *= 0.5
        fn[:] = fn1
        normfn_old = normfn
        normfn = np.linalg.norm2(fn)
        if (normdx > xtol) or ((normdx <= xtol) and (normfn < normfn_old)):
            xn[:] = xn1

        if (normdx <= xtol) or (normfn <= ftol):
            break
    return xn

*/

#endif // TWIST_PREPROCESS_ROOTS_H