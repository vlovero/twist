#pragma once

#include "fmt/core.h"
#include "linalg.h"
#include "shared.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace Krylov
{
    inline size_t gmres_lwork(size_t n, int restart)
    {
        const size_t m = (size_t)restart;
        return (2 * m + 4) * n + (m + 1) * m + 5 * m + 4;
    }

    namespace detail
    {
        template <typename Op_t>
        inline int invoke_with_tol(Op_t &op, const RP(double) x, RP(double) y, size_t n, double tol)
        {
            if constexpr (std::is_invocable_v<Op_t &, const RP(double), RP(double), size_t, double>) {
                return op(x, y, n, tol);
            }
            else {
                (void)tol;
                return op(x, y, n);
            }
        }
    } // namespace detail

    // Right-preconditioned restarted GMRES(restart) with optional diagonal quadrature weights
    // and BLAS2 CGS2 with conditional DGKS double reorthogonalization.
    // `weights` (length n) may be nullptr for unweighted Euclidean inner product.
    // `work` must have length >= gmres_lwork(n, restart) or be nullptr (auto-allocated and freed).
    // Returns 0 on convergence, 1 if maxiter reached without full convergence, <0 on operator error.
    template <typename A_Op_t, typename M_Op_t>
    int gmres(A_Op_t &A, M_Op_t &M, size_t n, RP(double) x, const RP(double) b, const double *weights, RP(double) work, int restart, int maxiter, double rtol, double atol, int *niters, bool verbose = false, bool verify_residual = true)
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
        double *h_corr = y + m;                // m
        double *r = h_corr + m;                // n
        double *w = r + n;                     // n
        double *w_w = w + n;                   // n

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
        bool x_is_zero = true;
        for (size_t i = 0; i < n; i++) {
            if (x[i] != 0.0) {
                x_is_zero = false;
                break;
            }
        }
        if (x_is_zero) {
            memcpy(r, b, n * sizeof(double));
        }
        else {
            if (detail::invoke_with_tol(A, x, w, n, -1.0) != 0) {
                if (niters) *niters = 0;
                if (own_work) free(work);
                return -3;
            }
            for (size_t i = 0; i < n; i++) {
                r[i] = b[i] - w[i];
            }
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

                if (detail::invoke_with_tol(M, vj, zj, n, -1.0) != 0) {
                    if (niters) *niters = total_iters;
                    if (own_work) free(work);
                    return -2;
                }
                if (detail::invoke_with_tol(A, zj, w, n, -1.0) != 0) {
                    if (niters) *niters = total_iters;
                    if (own_work) free(work);
                    return -3;
                }

                // BLAS2 CGS2 with conditional DGKS reorthogonalization
                const int in = (int)n;
                const int inc = 1;
                const double norm_before = w_norm(w);
                const double *w_weighted = w;
                if (weights != nullptr) {
                    for (size_t q = 0; q < n; q++) {
                        w_w[q] = weights[q] * w[q];
                    }
                    w_weighted = w_w;
                }
                dgemv("T", in, j + 1, 1.0, V, in, w_weighted, inc, 0.0, hj, inc);
                dgemv("N", in, j + 1, -1.0, V, in, hj, inc, 1.0, w, inc);

                double h_next = w_norm(w);
                if (h_next < 0.7 * norm_before) {
                    if (weights != nullptr) {
                        for (size_t q = 0; q < n; q++) {
                            w_w[q] = weights[q] * w[q];
                        }
                        w_weighted = w_w;
                    }
                    dgemv("T", in, j + 1, 1.0, V, in, w_weighted, inc, 0.0, h_corr, inc);
                    dgemv("N", in, j + 1, -1.0, V, in, h_corr, inc, 1.0, w, inc);
                    for (int i = 0; i <= j; i++) {
                        hj[i] += h_corr[i];
                    }
                    h_next = w_norm(w);
                }
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

            if (k_used > 0) {
                dgemv("N", (int)n, k_used, 1.0, Z, (int)n, y, 1, 1.0, x, 1);
            }

            if (!verify_residual && converged && total_iters <= restart && rtol >= 1e-8) {
                status = 0;
                break;
            }

            // Recompute true residual
            if (detail::invoke_with_tol(A, x, w, n, -1.0) != 0) {
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

    enum class KrylovMethod
    {
        GMRES = 0,
        FGMRES = 1,
        LGMRES = 2,
        GCRODR = 3
    };

    inline KrylovMethod parseKrylovMethod(const char *name)
    {
        if (!name) return KrylovMethod::GMRES;
        if (strcmp(name, "fgmres") == 0 || strcmp(name, "FGMRES") == 0) return KrylovMethod::FGMRES;
        if (strcmp(name, "lgmres") == 0 || strcmp(name, "LGMRES") == 0) return KrylovMethod::LGMRES;
        if (strcmp(name, "gcrodr") == 0 || strcmp(name, "GCRODR") == 0 || strcmp(name, "fgcrodr") == 0) return KrylovMethod::GCRODR;
        return KrylovMethod::GMRES;
    }

    inline const char *krylovMethodName(KrylovMethod m)
    {
        switch (m) {
        case KrylovMethod::GMRES: return "gmres";
        case KrylovMethod::FGMRES: return "fgmres";
        case KrylovMethod::LGMRES: return "lgmres";
        case KrylovMethod::GCRODR: return "gcrodr";
        }
        return "gmres";
    }

    struct GmresRecycleState
    {
        size_t n = 0;
        size_t k = 0;
        size_t cap_n = 0;
        size_t cap_k = 0;
        bool has_c_valid = false;
        double *U = nullptr; // cap_n * cap_k
        double *C = nullptr; // cap_n * cap_k
        double *tmp = nullptr; // cap_n * cap_k (for subspace rotation)

        ~GmresRecycleState() { freeBuffers(); }

        void clear()
        {
            k = 0;
            has_c_valid = false;
        }

        void reserve(size_t new_n, size_t new_k)
        {
            if (new_n != n) {
                k = 0;
                has_c_valid = false;
                n = new_n;
            }
            if (new_n > cap_n || new_k > cap_k) {
                const size_t cn = std::max(cap_n, new_n);
                const size_t ck = std::max(cap_k, new_k);
                double *U_new = (double *)malloc(cn * ck * sizeof(double));
                double *C_new = (double *)malloc(cn * ck * sizeof(double));
                double *T_new = (double *)malloc(cn * ck * sizeof(double));
                if (k > 0 && U && C && cap_n >= n) {
                    const size_t copy_k = std::min(k, ck);
                    for (size_t j = 0; j < copy_k; j++) {
                        memcpy(&U_new[j * cn], &U[j * cap_n], n * sizeof(double));
                        memcpy(&C_new[j * cn], &C[j * cap_n], n * sizeof(double));
                    }
                    k = copy_k;
                }
                else {
                    k = 0;
                    has_c_valid = false;
                }
                free(U);
                free(C);
                free(tmp);
                U = U_new;
                C = C_new;
                tmp = T_new;
                cap_n = cn;
                cap_k = ck;
            }
        }

        void freeBuffers()
        {
            if (U) { free(U); U = nullptr; }
            if (C) { free(C); C = nullptr; }
            if (tmp) { free(tmp); tmp = nullptr; }
            n = 0;
            k = 0;
            cap_n = 0;
            cap_k = 0;
            has_c_valid = false;
        }
    };

    inline size_t fgmres_lwork(size_t n, int restart, int k_rec = 0)
    {
        const size_t m = (size_t)std::max(1, restart);
        const size_t kr = (size_t)std::max(0, k_rec);
        const size_t ktot = m + kr + 2;
        return (2 * m + kr + 5) * n + 2 * (m + 1) * m + 4 * ktot * ktot + 32 * ktot + 16;
    }

    // Flexible GMRES with optional inexact Krylov relaxation (FGMRES), LGMRES error augmentation,
    // and FGCRO-DR harmonic Ritz subspace recycling (Parks et al. / Carvalho et al.).
    template <typename A_Op_t, typename M_Op_t>
    int fgmres(A_Op_t &A, M_Op_t &M, size_t n, RP(double) x, const RP(double) b, const double *weights,
               RP(double) work, int restart, int maxiter, double rtol, double atol, int *niters,
               KrylovMethod method, GmresRecycleState *rec, int k_rec, bool recompute_rec_c,
               double base_local_rtol, bool verbose = false, bool verify_residual = true)
    {
        if (method == KrylovMethod::GMRES) {
            return gmres(A, M, n, x, b, weights, work, restart, maxiter, rtol, atol, niters, verbose, verify_residual);
        }
        if (restart < 2) {
            restart = std::min<int>((int)n, 2);
        }
        if (restart > (int)n) {
            restart = (int)n;
        }
        const int kr = (rec != nullptr && k_rec > 0) ? std::min(k_rec, std::max(1, restart - 2)) : 0;
        const bool own_work = (work == nullptr);
        if (own_work) {
            work = (double *)malloc(fgmres_lwork(n, restart, kr) * sizeof(double));
        }

        const size_t m = (size_t)restart;
        const size_t ldh = m + 1;
        double *V = work;                      // (m + 1) * n
        double *Z = V + (m + 1) * n;           // m * n
        double *H = Z + m * n;                 // (m + 1) * m
        double *H_orig = H + (m + 1) * m;      // (m + 1) * m
        double *cs = H_orig + (m + 1) * m;     // m
        double *sn = cs + m;                   // m
        double *g = sn + m;                    // m + 1
        double *y = g + (m + 1);               // m
        double *h_corr = y + m;                // m
        double *r = h_corr + m;                // n
        double *w = r + n;                     // n
        double *w_w = w + n;                   // n
        double *dx_acc = w_w + n;              // n (LGMRES cycle update)
        double *small_buf = dx_acc + n;

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

        if (rec != nullptr && kr > 0) {
            rec->reserve(n, (size_t)kr);
            if (method == KrylovMethod::LGMRES) {
                rec->clear();
            }
        }

        int total_iters = 0;
        bool x_is_zero = true;
        for (size_t i = 0; i < n; i++) {
            if (x[i] != 0.0) {
                x_is_zero = false;
                break;
            }
        }
        if (x_is_zero) {
            memcpy(r, b, n * sizeof(double));
        }
        else {
            if (detail::invoke_with_tol(A, x, w, n, base_local_rtol) != 0) {
                if (niters) *niters = 0;
                if (own_work) free(work);
                return -3;
            }
            for (size_t i = 0; i < n; i++) {
                r[i] = b[i] - w[i];
            }
        }

        double beta0 = w_norm(r);
        double beta = beta0;
        const double target_tol = std::max(atol, rtol * beta0);

        if (beta <= atol || beta <= target_tol) {
            if (niters) *niters = 0;
            if (own_work) free(work);
            return 0;
        }

        // GCRO-DR initial setup & residual projection onto recycled C_k
        if (method == KrylovMethod::GCRODR && rec != nullptr && kr > 0 && rec->k > 0) {
            const size_t ld_rec = rec->cap_n;
            if (!rec->has_c_valid || recompute_rec_c) {
                size_t valid_k = 0;
                for (size_t i = 0; i < rec->k; i++) {
                    double *ui = &rec->U[valid_k * ld_rec];
                    double *ci = &rec->C[valid_k * ld_rec];
                    if (valid_k != i) {
                        memcpy(ui, &rec->U[i * ld_rec], n * sizeof(double));
                    }
                    if (detail::invoke_with_tol(A, ui, ci, n, base_local_rtol) != 0) {
                        rec->clear();
                        break;
                    }
                    // W-orthonormalize C while keeping A * U = C
                    for (int pass = 0; pass < 2; pass++) {
                        for (size_t p = 0; p < valid_k; p++) {
                            const double *cp = &rec->C[p * ld_rec];
                            const double *up = &rec->U[p * ld_rec];
                            const double alpha = w_dot(cp, ci);
                            for (size_t q = 0; q < n; q++) {
                                ci[q] -= alpha * cp[q];
                                ui[q] -= alpha * up[q];
                            }
                        }
                    }
                    const double nc = w_norm(ci);
                    if (nc > 1e-12) {
                        const double inv_nc = 1.0 / nc;
                        for (size_t q = 0; q < n; q++) {
                            ci[q] *= inv_nc;
                            ui[q] *= inv_nc;
                        }
                        valid_k++;
                    }
                }
                rec->k = valid_k;
                rec->has_c_valid = (valid_k > 0);
            }

            if (rec->k > 0) {
                const size_t kc = rec->k;
                const double *r_w = r;
                if (weights != nullptr) {
                    for (size_t q = 0; q < n; q++) {
                        w_w[q] = weights[q] * r[q];
                    }
                    r_w = w_w;
                }
                double *alpha_u = small_buf;
                dgemv("T", (int)n, (int)kc, 1.0, rec->C, (int)ld_rec, r_w, 1, 0.0, alpha_u, 1);
                dgemv("N", (int)n, (int)kc, 1.0, rec->U, (int)ld_rec, alpha_u, 1, 1.0, x, 1);
                dgemv("N", (int)n, (int)kc, -1.0, rec->C, (int)ld_rec, alpha_u, 1, 1.0, r, 1);
                beta = w_norm(r);
                if (beta <= atol || beta <= target_tol) {
                    if (niters) *niters = 0;
                    if (own_work) free(work);
                    return 0;
                }
            }
        }

        if (verbose) {
            fmt::println("[{}] init |r|_W = {:.6e} (target {:.6e})", krylovMethodName(method), beta0, target_tol);
        }

        int status = 1;
        while (total_iters < maxiter) {
            const size_t k_cur = (method == KrylovMethod::GCRODR && rec != nullptr && kr > 0) ? std::min<size_t>(rec->k, (size_t)kr) : 0;
            const int k_lgm = (method == KrylovMethod::LGMRES && rec != nullptr && kr > 0) ? std::min<int>((int)rec->k, std::max(0, restart - 2)) : 0;
            const int cycle_dim = (method == KrylovMethod::GCRODR && k_cur > 0)
                ? std::max(2, restart - (int)k_cur)
                : restart;
            const int m_arn = (method == KrylovMethod::LGMRES && k_lgm > 0)
                ? std::max(2, cycle_dim - k_lgm)
                : cycle_dim;

            double *B_mat = small_buf; // k_cur * cycle_dim (col-major, ld = k_cur)
            if (k_cur > 0) {
                memset(B_mat, 0, k_cur * (size_t)cycle_dim * sizeof(double));
            }

            const double inv_beta = 1.0 / beta;
            double *v0 = V;
            for (size_t i = 0; i < n; i++) {
                v0[i] = r[i] * inv_beta;
            }
            memset(g, 0, (m + 1) * sizeof(double));
            g[0] = beta;

            int k_used = 0;
            bool converged = false;

            for (int j = 0; j < cycle_dim && total_iters < maxiter; j++) {
                double *vj = &V[(size_t)j * n];
                double *zj = &Z[(size_t)j * n];
                double *hj = &H[(size_t)j * ldh];
                double *hj_o = &H_orig[(size_t)j * ldh];
                memset(hj, 0, ldh * sizeof(double));
                memset(hj_o, 0, ldh * sizeof(double));

                // Inexact Krylov tolerance relaxation (Simoncini & Szyld)
                const double prev_res = std::abs(g[j]);
                const double cur_ratio = std::max(1e-15, prev_res / std::max(1e-30, beta0));
                const double iter_tol = std::clamp(base_local_rtol / cur_ratio, base_local_rtol, 1e-4);

                if (method == KrylovMethod::LGMRES && k_lgm > 0 && j >= m_arn) {
                    const size_t l_idx = (size_t)(j - m_arn);
                    memcpy(zj, &rec->U[l_idx * rec->cap_n], n * sizeof(double));
                }
                else {
                    if (detail::invoke_with_tol(M, vj, zj, n, iter_tol) != 0) {
                        if (niters) *niters = total_iters;
                        if (own_work) free(work);
                        return -2;
                    }
                }
                if (detail::invoke_with_tol(A, zj, w, n, iter_tol) != 0) {
                    if (niters) *niters = total_iters;
                    if (own_work) free(work);
                    return -3;
                }

                const int in = (int)n;
                const double norm_before = w_norm(w);

                // In GCRO-DR, project w orthogonal to recycled C_k first
                if (k_cur > 0) {
                    double *bj = &B_mat[(size_t)j * k_cur];
                    double *b_corr = B_mat + k_cur * (size_t)cycle_dim;
                    const double *w_weighted = w;
                    if (weights != nullptr) {
                        for (size_t q = 0; q < n; q++) {
                            w_w[q] = weights[q] * w[q];
                        }
                        w_weighted = w_w;
                    }
                    dgemv("T", in, (int)k_cur, 1.0, rec->C, (int)rec->cap_n, w_weighted, 1, 0.0, bj, 1);
                    dgemv("N", in, (int)k_cur, -1.0, rec->C, (int)rec->cap_n, bj, 1, 1.0, w, 1);
                    const double nw_c = w_norm(w);
                    if (nw_c < 0.7 * norm_before) {
                        if (weights != nullptr) {
                            for (size_t q = 0; q < n; q++) {
                                w_w[q] = weights[q] * w[q];
                            }
                            w_weighted = w_w;
                        }
                        dgemv("T", in, (int)k_cur, 1.0, rec->C, (int)rec->cap_n, w_weighted, 1, 0.0, b_corr, 1);
                        dgemv("N", in, (int)k_cur, -1.0, rec->C, (int)rec->cap_n, b_corr, 1, 1.0, w, 1);
                        for (size_t p = 0; p < k_cur; p++) {
                            bj[p] += b_corr[p];
                        }
                    }
                }

                // Orthogonalize w against V[:, 0..j]
                const double norm_pre_v = w_norm(w);
                const double *w_weighted = w;
                if (weights != nullptr) {
                    for (size_t q = 0; q < n; q++) {
                        w_w[q] = weights[q] * w[q];
                    }
                    w_weighted = w_w;
                }
                dgemv("T", in, j + 1, 1.0, V, in, w_weighted, 1, 0.0, hj, 1);
                dgemv("N", in, j + 1, -1.0, V, in, hj, 1, 1.0, w, 1);

                double h_next = w_norm(w);
                if (h_next < 0.7 * std::max(norm_pre_v, 1e-30 * norm_before)) {
                    if (weights != nullptr) {
                        for (size_t q = 0; q < n; q++) {
                            w_w[q] = weights[q] * w[q];
                        }
                        w_weighted = w_w;
                    }
                    dgemv("T", in, j + 1, 1.0, V, in, w_weighted, 1, 0.0, h_corr, 1);
                    dgemv("N", in, j + 1, -1.0, V, in, h_corr, 1, 1.0, w, 1);
                    for (int i = 0; i <= j; i++) {
                        hj[i] += h_corr[i];
                    }
                    h_next = w_norm(w);
                }
                hj[j + 1] = h_next;
                memcpy(hj_o, hj, (size_t)(j + 2) * sizeof(double));

                if (h_next > 0.0) {
                    double *vj1 = &V[(size_t)(j + 1) * n];
                    const double inv_h = 1.0 / h_next;
                    for (size_t q = 0; q < n; q++) {
                        vj1[q] = w[q] * inv_h;
                    }
                }

                // Givens rotations
                for (int i = 0; i < j; i++) {
                    const double tmp = cs[i] * hj[i] + sn[i] * hj[i + 1];
                    hj[i + 1] = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                    hj[i] = tmp;
                }
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
                    fmt::println("[{}] iter {:4d} : |r|_W = {:.6e}", krylovMethodName(method), total_iters, res_est);
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

            if (k_used > 0) {
                dgemv("N", (int)n, k_used, 1.0, Z, (int)n, y, 1, 0.0, dx_acc, 1);
                if (k_cur > 0) {
                    double *y_u = B_mat + k_cur * (size_t)cycle_dim;
                    dgemv("N", (int)k_cur, k_used, -1.0, B_mat, (int)k_cur, y, 1, 0.0, y_u, 1);
                    dgemv("N", (int)n, (int)k_cur, 1.0, rec->U, (int)rec->cap_n, y_u, 1, 1.0, dx_acc, 1);
                }
                for (size_t q = 0; q < n; q++) {
                    x[q] += dx_acc[q];
                }
            }

            // Update LGMRES FIFO error approximation buffer
            if (method == KrylovMethod::LGMRES && rec != nullptr && kr > 0 && k_used > 0) {
                const double nrm_dx = w_norm(dx_acc);
                if (nrm_dx > 1e-30) {
                    const double inv_nrm = 1.0 / nrm_dx;
                    const size_t ld_rec = rec->cap_n;
                    if (rec->k < (size_t)kr) {
                        double *dst = &rec->U[rec->k * ld_rec];
                        for (size_t q = 0; q < n; q++) dst[q] = dx_acc[q] * inv_nrm;
                        rec->k++;
                    }
                    else {
                        for (size_t c = 0; c + 1 < rec->k; c++) {
                            memcpy(&rec->U[c * ld_rec], &rec->U[(c + 1) * ld_rec], n * sizeof(double));
                        }
                        double *dst = &rec->U[(rec->k - 1) * ld_rec];
                        for (size_t q = 0; q < n; q++) dst[q] = dx_acc[q] * inv_nrm;
                    }
                }
            }

            // Update FGCRO-DR harmonic Ritz subspace if cycling or worth harvesting
            if (method == KrylovMethod::GCRODR && rec != nullptr && kr > 0 && (k_cur + (size_t)k_used) >= 2 && (k_used >= 2 || k_cur > 0)) {
                const size_t ktot = k_cur + (size_t)k_used;
                const size_t rows_g = ktot + 1;
                const size_t ld_rec = rec->cap_n;
                const size_t ld_g = rows_g;
                const size_t ld_sq = ktot;

                // Layout in scratch vectors (small (ktot+1) x ktot dense ops, O(100^3))
                double *G_full = B_mat + k_cur * (size_t)cycle_dim + k_cur;
                double *d_scale = G_full + ld_g * ktot;
                double *S_ov = d_scale + k_cur;                 // (rows_g * ktot) = V_hat^T W Z_hat
                double *M_A = S_ov + ld_g * ktot;               // ktot * ktot = G^T G
                double *M_B = M_A + ld_sq * ktot;               // ktot * ktot = G^T S_ov
                double *alphar = M_B + ld_sq * ktot;
                double *alphai = alphar + ktot;
                double *beta_ev = alphai + ktot;
                double *vr = beta_ev + ktot;                    // ktot * ktot
                double *P_mat = vr + ld_sq * ktot;              // ktot * kr
                double *T_mat = P_mat + ld_sq * (size_t)kr;     // rows_g * kr
                double *tau_qr = T_mat + ld_g * (size_t)kr;     // kr
                double *lap_work = tau_qr + kr;

                for (size_t i = 0; i < k_cur; i++) {
                    const double nu = w_norm(&rec->U[i * ld_rec]);
                    d_scale[i] = (nu > 1e-30) ? (1.0 / nu) : 1.0;
                }
                memset(G_full, 0, ld_g * ktot * sizeof(double));
                for (size_t i = 0; i < k_cur; i++) {
                    G_full[i * ld_g + i] = d_scale[i];
                }
                for (int j = 0; j < k_used; j++) {
                    const size_t col = k_cur + (size_t)j;
                    for (size_t i = 0; i < k_cur; i++) {
                        G_full[col * ld_g + i] = B_mat[(size_t)j * k_cur + i];
                    }
                    for (int i = 0; i <= j + 1 && i <= k_used; i++) {
                        G_full[col * ld_g + k_cur + (size_t)i] = H_orig[(size_t)j * ldh + (size_t)i];
                    }
                }

                // Compute S_ov = V_hat^T W Z_hat via w_dot on blocks [C, V] and [U*D, Z]
                for (size_t col = 0; col < ktot; col++) {
                    const double *zc = (col < k_cur) ? &rec->U[col * ld_rec] : &Z[(col - k_cur) * n];
                    const double sc_z = (col < k_cur) ? d_scale[col] : 1.0;
                    const double *zw = zc;
                    if (weights != nullptr) {
                        for (size_t q = 0; q < n; q++) w_w[q] = weights[q] * zc[q];
                        zw = w_w;
                    }
                    if (k_cur > 0) {
                        dgemv("T", (int)n, (int)k_cur, sc_z, rec->C, (int)ld_rec, zw, 1, 0.0, &S_ov[col * ld_g], 1);
                    }
                    dgemv("T", (int)n, k_used + 1, sc_z, V, (int)n, zw, 1, 0.0, &S_ov[col * ld_g + k_cur], 1);
                }

                const int ik = (int)ktot;
                const int irg = (int)rows_g;
                dgemm("T", "N", ik, ik, irg, 1.0, G_full, irg, G_full, irg, 0.0, M_A, ik);
                dgemm("T", "N", ik, ik, irg, 1.0, G_full, irg, S_ov, irg, 0.0, M_B, ik);

                // Query & run dggev3
                int info_ev = 0;
                double q_work = 0.0;
                dggev3("N", "V", ik, M_A, ik, M_B, ik, alphar, alphai, beta_ev, nullptr, 1, vr, ik, &q_work, -1, &info_ev);
                const int lwork_ev = std::max(1, (int)q_work);
                std::vector<double> ev_work_vec((size_t)lwork_ev);
                dggev3("N", "V", ik, M_A, ik, M_B, ik, alphar, alphai, beta_ev, nullptr, 1, vr, ik, ev_work_vec.data(), lwork_ev, &info_ev);

                if (info_ev == 0) {
                    struct EvIdx { double mag; size_t idx; bool is_conj; };
                    std::vector<EvIdx> order;
                    order.reserve(ktot);
                    for (size_t i = 0; i < ktot; ) {
                        const bool cplx = (std::abs(alphai[i]) > 1e-14 && i + 1 < ktot);
                        const double den = std::abs(beta_ev[i]);
                        const double mag = (den > 1e-14) ? (std::hypot(alphar[i], alphai[i]) / den) : 1e300;
                        order.push_back({ mag, i, cplx });
                        i += cplx ? 2 : 1;
                    }
                    std::sort(order.begin(), order.end(), [](const EvIdx &a, const EvIdx &b) {
                        return a.mag < b.mag;
                    });

                    size_t k_sel = 0;
                    const size_t k_target = std::min((size_t)kr, ktot);
                    for (const EvIdx &ev : order) {
                        const size_t need = ev.is_conj ? 2 : 1;
                        if (k_sel + need > k_target) continue;
                        memcpy(&P_mat[k_sel * ktot], &vr[ev.idx * ktot], ktot * sizeof(double));
                        if (ev.is_conj) {
                            memcpy(&P_mat[(k_sel + 1) * ktot], &vr[(ev.idx + 1) * ktot], ktot * sizeof(double));
                        }
                        k_sel += need;
                        if (k_sel >= k_target) break;
                    }

                    if (k_sel > 0) {
                        dgemm("N", "N", irg, (int)k_sel, ik, 1.0, G_full, irg, P_mat, ik, 0.0, T_mat, irg);
                        int info_qr = 0;
                        double q_qr = 0.0;
                        dgeqrf(irg, (int)k_sel, T_mat, irg, tau_qr, &q_qr, -1, &info_qr);
                        int lw_qr = (int)q_qr;
                        double q_org = 0.0;
                        dorgqr(irg, (int)k_sel, (int)k_sel, T_mat, irg, tau_qr, &q_org, -1, &info_qr);
                        lw_qr = std::max({ lw_qr, (int)q_org, 1 });
                        (void)lap_work;
                        std::vector<double> qr_work((size_t)lw_qr);
                        dgeqrf(irg, (int)k_sel, T_mat, irg, tau_qr, qr_work.data(), lw_qr, &info_qr);
                        if (info_qr == 0) {
                            dtrsm("R", "U", "N", "N", ik, (int)k_sel, 1.0, T_mat, irg, P_mat, ik);
                            dorgqr(irg, (int)k_sel, (int)k_sel, T_mat, irg, tau_qr, qr_work.data(), lw_qr, &info_qr);
                        }
                        if (info_qr == 0) {
                            // U_new = Z_hat * P_mat (preserving column scale d_scale[0..k_cur-1]) into rec->tmp
                            if (k_cur > 0) {
                                // Scale upper rows of P_mat by d_scale
                                for (size_t c = 0; c < k_sel; c++) {
                                    for (size_t r_p = 0; r_p < k_cur; r_p++) {
                                        P_mat[c * ktot + r_p] *= d_scale[r_p];
                                    }
                                }
                                dgemm("N", "N", (int)n, (int)k_sel, (int)k_cur, 1.0, rec->U, (int)ld_rec, P_mat, ik, 0.0, rec->tmp, (int)ld_rec);
                            }
                            dgemm("N", "N", (int)n, (int)k_sel, k_used, 1.0, Z, (int)n, &P_mat[k_cur], ik, (k_cur > 0) ? 1.0 : 0.0, rec->tmp, (int)ld_rec);
                            for (size_t c = 0; c < k_sel; c++) {
                                memcpy(&rec->U[c * ld_rec], &rec->tmp[c * ld_rec], n * sizeof(double));
                            }
                            // C_new = V_hat * Q_mat (= T_mat) into rec->tmp then copy to rec->C
                            if (k_cur > 0) {
                                dgemm("N", "N", (int)n, (int)k_sel, (int)k_cur, 1.0, rec->C, (int)ld_rec, T_mat, irg, 0.0, rec->tmp, (int)ld_rec);
                            }
                            dgemm("N", "N", (int)n, (int)k_sel, k_used + 1, 1.0, V, (int)n, &T_mat[k_cur], irg, (k_cur > 0) ? 1.0 : 0.0, rec->tmp, (int)ld_rec);
                            for (size_t c = 0; c < k_sel; c++) {
                                memcpy(&rec->C[c * ld_rec], &rec->tmp[c * ld_rec], n * sizeof(double));
                            }
                            rec->k = k_sel;
                            rec->has_c_valid = true;
                        }
                    }
                }
            }

            if (!verify_residual && converged && total_iters <= restart && rtol >= 1e-8) {
                status = 0;
                break;
            }

            // Recompute exact true residual with base_local_rtol
            if (detail::invoke_with_tol(A, x, w, n, base_local_rtol) != 0) {
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
            // Project updated residual orthogonal to C_k before next GCRO-DR cycle
            if (method == KrylovMethod::GCRODR && rec != nullptr && kr > 0 && rec->k > 0) {
                const size_t kc = rec->k;
                const double *r_w = r;
                if (weights != nullptr) {
                    for (size_t q = 0; q < n; q++) {
                        w_w[q] = weights[q] * r[q];
                    }
                    r_w = w_w;
                }
                double *alpha_u = small_buf;
                dgemv("T", (int)n, (int)kc, 1.0, rec->C, (int)rec->cap_n, r_w, 1, 0.0, alpha_u, 1);
                dgemv("N", (int)n, (int)kc, 1.0, rec->U, (int)rec->cap_n, alpha_u, 1, 1.0, x, 1);
                dgemv("N", (int)n, (int)kc, -1.0, rec->C, (int)rec->cap_n, alpha_u, 1, 1.0, r, 1);
                beta = w_norm(r);
                if (beta <= target_tol) {
                    status = 0;
                    break;
                }
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
