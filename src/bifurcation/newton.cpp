#include "bif_points/newton.h"

namespace Newton
{
    bool solve(const ptrdiff_t n, double *x0, residual_t residual, construct_t construct_jac, solver_t solve_system, callback_t callback, double tol, int maxiter, void *data, const bool verbose)
    {
        int itern;
        double damp, norm_dx, norm_dxb, norm_r, norm_r1;
        double *x, *x1, *r, *r1, *dx, *dxb;
        bool converged, cond1, cond2;
        x = (double *)malloc(n * sizeof(double));
        x1 = (double *)malloc(n * sizeof(double));
        r = (double *)malloc(n * sizeof(double));
        r1 = (double *)malloc(n * sizeof(double));
        dx = (double *)malloc(n * sizeof(double));
        dxb = (double *)malloc(n * sizeof(double));

        memcpy(x, x0, n * sizeof(double));
        residual(r, x, data);
        norm_r = norm2(r, n);

        if (norm_r < tol) {
            return true;
        }

        damp = 1.0;
        converged = false;
        for (itern = 0; itern < maxiter; itern++) {
            if (verbose) {
                fmt::println("{:3d} {:.8e} {:.8e}", itern, norm_r, damp);
            }
            auto J = construct_jac(x, data);
            solve_system(J, r, dx, data);
            norm_dx = norm2(dx, n);
            if (!std::isfinite(norm_r1) || !std::isfinite(norm_dx)) {
                break;
            }
            damp = std::min(2.0 * damp, 1.0);
            while (damp > 1e-2) { // pow(0.5, 15)) {
                for (ptrdiff_t i = 0; i < n; i++) {
                    x1[i] = x[i] - damp * dx[i];
                }
                residual(r1, x1, data);
                norm_r1 = norm2(r1, n);
                solve_system(J, r1, dxb, data);
                norm_dxb = norm2(dxb, n);
                cond1 = norm_dxb <= ((1 - 0.5 * damp) * norm_dx);
                cond2 = norm_r1 < norm_r;
                if (cond1 && cond2) {
                    break;
                }
                damp *= 0.5;
            }
            memcpy(r, r1, n * sizeof(double));
            norm_r = norm_r1;
            callback(x, x1, data);
            if ((norm_dx < tol) || (norm_r < tol)) {
                memcpy(x0, x, n * sizeof(double));
                converged = true;
                if (verbose) {
                    fmt::println("finished with |R| = {:.8e} |d| = {:.8e}", norm_r, norm_dx);
                }
                break;
            }
        }

        free(x);
        free(x1);
        free(r);
        free(r1);
        free(dx);
        free(dxb);

        return converged;
    }
} // namespace Newton