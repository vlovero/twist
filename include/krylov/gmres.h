#pragma once

#include "fmt/core.h"
#include "shared.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace Krylov
{
    inline size_t gmres_lwork(size_t n, int restart)
    {
        const size_t m = (size_t)restart;
        return (2 * m + 3) * n + (m + 1) * m + 4 * m + 4;
    }

    // Right-preconditioned restarted GMRES(restart) with optional diagonal quadrature weights
    // and DGKS double-reorthogonalized Modified Gram-Schmidt.
    // `weights` (length n) may be nullptr for unweighted Euclidean inner product.
    // `work` must have length >= gmres_lwork(n, restart) or be nullptr (auto-allocated and freed).
    // Returns 0 on convergence, 1 if maxiter reached without full convergence, <0 on operator error.
    template <typename A_Op_t, typename M_Op_t>
    int gmres(A_Op_t &A, M_Op_t &M, size_t n, RP(double) x, const RP(double) b, const double *weights, RP(double) work, int restart, int maxiter, double rtol, double atol, int *niters, bool verbose = false)
    {
        if (restart < 1) {
            restart = 1;
        }
        if (restart > (int)n) {
            restart = (int)n;
        }
        const bool own_work = (work == nullptr);
        if (own_work) {
            work = (double *)malloc(gmres_lwork(n, restart) * sizeof(double));
        }

        const size_t m = (size_t)restart;
        const size_t ldh = m + 1;
        double *V = work;                      // (m + 1) * n
        double *Z = V + (m + 1) * n;           // m * n
        double *H = Z + m * n;                 // (m + 1) * m
        double *cs = H + (m + 1) * m;          // m
        double *sn = cs + m;                   // m
        double *g = sn + m;                    // m + 1
        double *y = g + (m + 1);               // m
        double *r = y + m;                     // n
        double *w = r + n;                     // n

        auto w_dot = [n, weights](const RP(double) u, const RP(double) v) -> double {
            double acc = 0.0;
            if (weights == nullptr) {
                for (size_t i = 0; i < n; i++) {
                    acc += u[i] * v[i];
                }
            }
            else {
                for (size_t i = 0; i < n; i++) {
                    acc += weights[i] * u[i] * v[i];
                }
            }
            return acc;
        };

        auto w_norm = [&w_dot](const RP(double) u) -> double {
            const double val = w_dot(u, u);
            return val > 0.0 ? std::sqrt(val) : 0.0;
        };

        int total_iters = 0;
        if (A(x, w, n) != 0) {
            if (niters) *niters = 0;
            if (own_work) free(work);
            return -3;
        }
        for (size_t i = 0; i < n; i++) {
            r[i] = b[i] - w[i];
        }
        double beta0 = w_norm(r);
        double beta = beta0;
        const double target_tol = std::max(atol, rtol * beta0);

        if (verbose) {
            fmt::println("[GMRES] init |r|_W = {:.6e} (target {:.6e})", beta0, target_tol);
        }

        if (beta <= atol || beta <= target_tol) {
            if (niters) *niters = 0;
            if (own_work) free(work);
            return 0;
        }

        int status = 1;
        while (total_iters < maxiter) {
            const double inv_beta = 1.0 / beta;
            double *v0 = V;
            for (size_t i = 0; i < n; i++) {
                v0[i] = r[i] * inv_beta;
            }
            memset(g, 0, (m + 1) * sizeof(double));
            g[0] = beta;

            int k_used = 0;
            bool converged = false;

            for (int j = 0; j < restart && total_iters < maxiter; j++) {
                double *vj = &V[(size_t)j * n];
                double *zj = &Z[(size_t)j * n];
                double *hj = &H[(size_t)j * ldh];
                memset(hj, 0, ldh * sizeof(double));

                if (M(vj, zj, n) != 0) {
                    if (niters) *niters = total_iters;
                    if (own_work) free(work);
                    return -2;
                }
                if (A(zj, w, n) != 0) {
                    if (niters) *niters = total_iters;
                    if (own_work) free(work);
                    return -3;
                }

                // Modified Gram-Schmidt with DGKS double reorthogonalization
                for (int i = 0; i <= j; i++) {
                    const double *vi = &V[(size_t)i * n];
                    const double hij = w_dot(vi, w);
                    hj[i] = hij;
                    for (size_t q = 0; q < n; q++) {
                        w[q] -= hij * vi[q];
                    }
                }
                for (int i = 0; i <= j; i++) {
                    const double *vi = &V[(size_t)i * n];
                    const double corr = w_dot(vi, w);
                    hj[i] += corr;
                    for (size_t q = 0; q < n; q++) {
                        w[q] -= corr * vi[q];
                    }
                }

                const double h_next = w_norm(w);
                hj[j + 1] = h_next;
                if (h_next > 0.0) {
                    double *vj1 = &V[(size_t)(j + 1) * n];
                    const double inv_h = 1.0 / h_next;
                    for (size_t q = 0; q < n; q++) {
                        vj1[q] = w[q] * inv_h;
                    }
                }

                // Apply previous Givens rotations to column j
                for (int i = 0; i < j; i++) {
                    const double tmp = cs[i] * hj[i] + sn[i] * hj[i + 1];
                    hj[i + 1] = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                    hj[i] = tmp;
                }

                // Compute new Givens rotation
                const double denom = std::hypot(hj[j], hj[j + 1]);
                if (denom > 0.0) {
                    cs[j] = hj[j] / denom;
                    sn[j] = hj[j + 1] / denom;
                }
                else {
                    cs[j] = 1.0;
                    sn[j] = 0.0;
                }
                hj[j] = cs[j] * hj[j] + sn[j] * hj[j + 1];
                hj[j + 1] = 0.0;

                const double gj = g[j];
                g[j] = cs[j] * gj;
                g[j + 1] = -sn[j] * gj;

                total_iters++;
                k_used = j + 1;
                const double res_est = std::abs(g[j + 1]);

                if (verbose) {
                    fmt::println("[GMRES] iter {:4d} : |r|_W = {:.6e}", total_iters, res_est);
                }

                if (res_est <= target_tol || h_next <= 1e-15 * beta0) {
                    converged = true;
                    break;
                }
            }

            // Back-substitute H[0:k_used, 0:k_used] * y = g[0:k_used]
            for (int i = k_used - 1; i >= 0; i--) {
                double sum = g[i];
                for (int l = i + 1; l < k_used; l++) {
                    sum -= H[(size_t)l * ldh + (size_t)i] * y[l];
                }
                const double diag = H[(size_t)i * ldh + (size_t)i];
                y[i] = (std::abs(diag) > 1e-300) ? (sum / diag) : 0.0;
            }

            for (int i = 0; i < k_used; i++) {
                const double yi = y[i];
                const double *zi = &Z[(size_t)i * n];
                for (size_t q = 0; q < n; q++) {
                    x[q] += yi * zi[q];
                }
            }

            // Recompute true residual
            if (A(x, w, n) != 0) {
                if (niters) *niters = total_iters;
                if (own_work) free(work);
                return -3;
            }
            for (size_t q = 0; q < n; q++) {
                r[q] = b[q] - w[q];
            }
            beta = w_norm(r);
            if (beta <= target_tol || (converged && beta <= 10.0 * target_tol)) {
                status = 0;
                break;
            }
            if (converged && total_iters >= maxiter) {
                break;
            }
        }

        if (niters) {
            *niters = total_iters;
        }
        if (own_work) {
            free(work);
        }
        return status;
    }
} // namespace Krylov
