#pragma once

#include "linalg.h"
#include "shared.h"
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace Krylov
{
    // Computes the Periodic Real Schur Decomposition of p matrices H[0..p-1] (each n x n, column-major, ldH = n),
    // representing the cycle H[p-1] * H[p-2] * ... * H[0].
    // On exit:
    //   H[0..p-2] are upper triangular, H[p-1] is real quasi-upper-triangular.
    //   Z[0..p-1] (each n x n, column-major) satisfy:
    //     H_new[n] = Z[(n+1)%p]^T * H_old[n] * Z[n]
    //   wr[0..n-1], wi[0..n-1] hold eigenvalues of the product (computed safely from diagonal 1x1 / 2x2 blocks).
    void periodic_real_schur(size_t p, size_t n, double *const *H, double *const *Z, double *wr, double *wi, double *work = nullptr);

    // Reorders the Periodic Real Schur form (H[0..p-1], Z[0..p-1]) so that eigenvalues with select[i] != 0
    // appear in the leading top-left k x k blocks. Updates wr, wi in place and returns k.
    size_t periodic_schur_reorder(size_t p, size_t n, double *const *H, double *const *Z, const int *select, double *wr, double *wi, double *work = nullptr);

    struct PeriodicKSResult
    {
        size_t p = 0;
        size_t nsub = 0;
        size_t nconverged = 0;
        size_t *dims = nullptr;
        size_t *q_offsets = nullptr;
        double *Q_all = nullptr; // p blocks, n-th block has dims[n] * (nsub + 1)
        double *H_all = nullptr; // p matrices, each nsub * nsub (column-major)
        double *wr = nullptr;    // length nsub
        double *wi = nullptr;    // length nsub
        double *work = nullptr;
        size_t work_cap = 0;

        PeriodicKSResult() = default;
        PeriodicKSResult(const PeriodicKSResult &) = delete;
        PeriodicKSResult &operator=(const PeriodicKSResult &) = delete;
        ~PeriodicKSResult();

        void freeBuffers();
        void reserve(size_t in_p, const size_t *in_dims, size_t in_nsub);

        double *Q(size_t n) { return Q_all + q_offsets[n]; }
        const double *Q(size_t n) const { return Q_all + q_offsets[n]; }
        double *H(size_t n) { return H_all + n * nsub * nsub; }
        const double *H(size_t n) const { return H_all + n * nsub * nsub; }

        // Computes complex right eigenvectors in space V_0 for the first `k` eigenvalues (k <= nconverged).
        // `V_out` must have size dims[0] * k (column-major).
        void getEigenvectorsAtNode0(size_t k, std::complex<double> *V_out) const;
    };

    namespace internal
    {
        inline double weighted_dot(size_t n, const double *w, const double *x, const double *y)
        {
            double acc = 0.0;
            if (w == nullptr) {
                for (size_t i = 0; i < n; i++) {
                    acc += x[i] * y[i];
                }
            }
            else {
                for (size_t i = 0; i < n; i++) {
                    acc += w[i] * x[i] * y[i];
                }
            }
            return acc;
        }

        inline double weighted_norm(size_t n, const double *w, const double *x)
        {
            const double val = weighted_dot(n, w, x, x);
            return val > 0.0 ? std::sqrt(val) : 0.0;
        }
    } // namespace internal

    // Periodic Krylov-Schur eigensolver for the cyclic operator product A_{p-1} * ... * A_1 * A_0,
    // where `apply_op(size_t n, const double *v_in, double *v_out)` applies A_n : V_n -> V_{(n+1)%p}.
    // `weights[n]` (length dims[n], or weights == nullptr) specifies diagonal L2 quadrature weights on V_n.
    template <typename ApplyOp_t>
    void periodic_krylov_schur(ApplyOp_t &apply_op, size_t p, const size_t *dims, const double *const *weights, size_t neig, size_t nsub, PeriodicKSResult &result, double tol, bool verbose = false, int seed = 17, int maxiter = 100)
    {
        if (nsub < neig + 4) {
            nsub = neig + 4;
        }
        size_t min_dim = dims[0];
        for (size_t n = 1; n < p; n++) {
            min_dim = std::min(min_dim, dims[n]);
        }
        if (nsub > min_dim) {
            nsub = min_dim;
        }
        if (neig > nsub) {
            neig = nsub;
        }

        result.reserve(p, dims, nsub);
        result.nconverged = 0;

        double *H_ptrs[p];
        double *Z_ptrs[p];
        double *Z_buf = result.work;
        for (size_t n = 0; n < p; n++) {
            H_ptrs[n] = result.H(n);
            Z_ptrs[n] = Z_buf + n * nsub * nsub;
            memset(H_ptrs[n], 0, nsub * nsub * sizeof(double));
        }

        double *schur_work = Z_buf + p * nsub * nsub;
        const size_t schur_work_len = std::max(4 * p * nsub * nsub + 32 * nsub, 32 * p * p + 64 * p) + 256;
        double *b_row = schur_work + schur_work_len;
        double *q_tmp = b_row + nsub; // size max(dims) * nsub

        size_t max_dim = dims[0];
        for (size_t n = 1; n < p; n++) {
            max_dim = std::max(max_dim, dims[n]);
        }
        double *v_scratch = q_tmp + max_dim * nsub;
        double *r_vec = v_scratch + max_dim; // size dims[0]

        // Seed starting vector q_0^{(0)} in V_0
        {
            double *q0 = result.Q(0);
            const size_t d0 = dims[0];
            uint64_t rng = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
            for (size_t i = 0; i < d0; i++) {
                rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                q0[i] = ((double)(int64_t)(rng >> 11) / (double)(1ULL << 52)) - 1.0;
            }
            const double *w0 = weights ? weights[0] : nullptr;
            const double nm = internal::weighted_norm(d0, w0, q0);
            if (nm > 0.0) {
                for (size_t i = 0; i < d0; i++) {
                    q0[i] /= nm;
                }
            }
        }

        auto ortho_col = [&](size_t space_idx, size_t col_idx, double *vec, double *h_col) {
            const size_t dim = dims[space_idx];
            const double *w = weights ? weights[space_idx] : nullptr;
            const double *Qmat = result.Q(space_idx);
            for (size_t i = 0; i < col_idx; i++) {
                const double hij = internal::weighted_dot(dim, w, &Qmat[i * dim], vec);
                h_col[i] = hij;
                for (size_t r = 0; r < dim; r++) {
                    vec[r] -= hij * Qmat[i * dim + r];
                }
            }
            // DGKS second pass
            for (size_t i = 0; i < col_idx; i++) {
                const double corr = internal::weighted_dot(dim, w, &Qmat[i * dim], vec);
                h_col[i] += corr;
                for (size_t r = 0; r < dim; r++) {
                    vec[r] -= corr * Qmat[i * dim + r];
                }
            }
        };

        auto fill_random_orthonormal = [&](size_t space_idx, size_t col_idx, double *vec, uint64_t &rng) {
            const size_t dim = dims[space_idx];
            const double *w = weights ? weights[space_idx] : nullptr;
            const double *Qmat = result.Q(space_idx);
            for (int attempt = 0; attempt < 4; attempt++) {
                for (size_t i = 0; i < dim; i++) {
                    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                    vec[i] = ((double)(int64_t)(rng >> 11) / (double)(1ULL << 52)) - 1.0;
                }
                for (size_t i = 0; i < col_idx; i++) {
                    const double hij = internal::weighted_dot(dim, w, &Qmat[i * dim], vec);
                    for (size_t r = 0; r < dim; r++) {
                        vec[r] -= hij * Qmat[i * dim + r];
                    }
                }
                const double nm = internal::weighted_norm(dim, w, vec);
                if (nm > 1e-12) {
                    for (size_t r = 0; r < dim; r++) {
                        vec[r] /= nm;
                    }
                    return;
                }
            }
        };

        uint64_t rng = (uint64_t)seed + 1337ULL;
        size_t k_start = 0;
        double beta_last = 0.0;

        for (int iter = 0; iter < maxiter; iter++) {
            // Periodic Arnoldi expansion from k_start .. nsub - 1
            for (size_t j = k_start; j < nsub; j++) {
                for (size_t n = 0; n < p - 1; n++) {
                    const size_t next_n = n + 1;
                    const size_t d_next = dims[next_n];
                    const double *w_next = weights ? weights[next_n] : nullptr;
                    double *qn = &result.Q(n)[j * dims[n]];
                    double *qnext = &result.Q(next_n)[j * d_next];
                    double *hcol = &H_ptrs[n][j * nsub];
                    memset(hcol, 0, nsub * sizeof(double));

                    apply_op(n, qn, v_scratch);
                    ortho_col(next_n, j, v_scratch, hcol);
                    const double nm = internal::weighted_norm(d_next, w_next, v_scratch);
                    hcol[j] = nm;
                    if (nm > 1e-14) {
                        const double inv_nm = 1.0 / nm;
                        for (size_t r = 0; r < d_next; r++) {
                            qnext[r] = v_scratch[r] * inv_nm;
                        }
                    }
                    else {
                        hcol[j] = 0.0;
                        fill_random_orthonormal(next_n, j, qnext, rng);
                    }
                }

                // Closing step: A_{p-1}: V_{p-1} -> V_0
                {
                    const size_t d0 = dims[0];
                    const double *w0 = weights ? weights[0] : nullptr;
                    double *qp = &result.Q(p - 1)[j * dims[p - 1]];
                    double *hcol = &H_ptrs[p - 1][j * nsub];
                    memset(hcol, 0, nsub * sizeof(double));

                    apply_op(p - 1, qp, v_scratch);
                    ortho_col(0, j + 1, v_scratch, hcol);
                    const double nm = internal::weighted_norm(d0, w0, v_scratch);
                    beta_last = nm;
                    if (j + 1 < nsub) {
                        H_ptrs[p - 1][j * nsub + (j + 1)] = nm;
                        double *q0_next = &result.Q(0)[(j + 1) * d0];
                        if (nm > 1e-14) {
                            const double inv_nm = 1.0 / nm;
                            for (size_t r = 0; r < d0; r++) {
                                q0_next[r] = v_scratch[r] * inv_nm;
                            }
                        }
                        else {
                            H_ptrs[p - 1][j * nsub + (j + 1)] = 0.0;
                            fill_random_orthonormal(0, j + 1, q0_next, rng);
                        }
                    }
                    else {
                        if (nm > 1e-14) {
                            const double inv_nm = 1.0 / nm;
                            for (size_t r = 0; r < d0; r++) {
                                r_vec[r] = v_scratch[r] * inv_nm;
                            }
                        }
                        else {
                            beta_last = 0.0;
                            fill_random_orthonormal(0, nsub, r_vec, rng);
                        }
                    }
                }
            }

            // Periodic Real Schur decomposition of (H_{p-1}, ..., H_0)
            periodic_real_schur(p, nsub, H_ptrs, Z_ptrs, result.wr, result.wi, schur_work);

            // Sort eigenvalues by descending magnitude |mu|
            int select[nsub];
            double mags[nsub];
            size_t order[nsub];
            for (size_t i = 0; i < nsub; i++) {
                mags[i] = std::hypot(result.wr[i], result.wi[i]);
                order[i] = i;
                select[i] = 0;
            }
            std::sort(order, order + nsub, [&](size_t a, size_t b) { return mags[a] > mags[b]; });

            // Pick top `neig` eigenvalues (do not split complex conjugate pairs)
            size_t k_target = 0;
            for (size_t idx = 0; idx < nsub && k_target < neig; idx++) {
                const size_t pos = order[idx];
                if (select[pos]) {
                    continue;
                }
                if (result.wi[pos] != 0.0) {
                    // Find conjugate partner (adjacent in Schur form)
                    if (pos + 1 < nsub && std::abs(result.wi[pos] + result.wi[pos + 1]) < 1e-12 * (1.0 + std::abs(result.wi[pos])) && std::abs(result.wr[pos] - result.wr[pos + 1]) < 1e-12 * (1.0 + std::abs(result.wr[pos]))) {
                        select[pos] = 1;
                        select[pos + 1] = 1;
                        k_target += 2;
                    }
                    else if (pos > 0 && std::abs(result.wi[pos] + result.wi[pos - 1]) < 1e-12 * (1.0 + std::abs(result.wi[pos])) && std::abs(result.wr[pos] - result.wr[pos - 1]) < 1e-12 * (1.0 + std::abs(result.wr[pos]))) {
                        select[pos - 1] = 1;
                        select[pos] = 1;
                        k_target += 2;
                    }
                    else {
                        select[pos] = 1;
                        k_target += 1;
                    }
                }
                else {
                    select[pos] = 1;
                    k_target += 1;
                }
            }

            // Reorder selected eigenvalues to top-left 0..k_kept-1
            // Note: periodic_schur_reorder also updates Z_ptrs[0..p-1]
            // We need full Z_ptrs including the initial periodic_real_schur Z_ptrs, which periodic_schur_reorder accumulates into!
            const size_t k_kept = periodic_schur_reorder(p, nsub, H_ptrs, Z_ptrs, select, result.wr, result.wi, schur_work);

            // Update basis Q_n <- Q_n * Z_n (using q_tmp)
            for (size_t n = 0; n < p; n++) {
                const size_t d = dims[n];
                dgemm("N", "N", (int)d, (int)k_kept, (int)nsub, 1.0, result.Q(n), (int)d, Z_ptrs[n], (int)nsub, 0.0, q_tmp, (int)d);
                memcpy(result.Q(n), q_tmp, d * k_kept * sizeof(double));
            }

            // Compute residual row b_row = beta_last * e_{nsub-1}^T * Z_{p-1}
            for (size_t j = 0; j < k_kept; j++) {
                b_row[j] = beta_last * Z_ptrs[p - 1][j * nsub + (nsub - 1)];
            }

            // Check convergence of leading `neig` (or `k_kept`) eigenvalues
            size_t nconv = 0;
            for (size_t j = 0; j < k_kept; j++) {
                const double mu_mag = std::hypot(result.wr[j], result.wi[j]);
                const double res_j = std::abs(b_row[j]);
                if (res_j <= tol * std::max(1.0, mu_mag)) {
                    nconv++;
                }
                else {
                    break;
                }
            }
            result.nconverged = nconv;

            if (verbose) {
                double max_res = 0.0;
                for (size_t j = 0; j < std::min(neig, k_kept); j++) {
                    max_res = std::max(max_res, std::abs(b_row[j]) / std::max(1.0, std::hypot(result.wr[j], result.wi[j])));
                }
                fmt::println("[PeriodicKS] iter {:3d} : nconv = {}/{} : max_rel_res = {:.3e}", iter + 1, nconv, neig, max_res);
            }

            if (nconv >= neig || nsub >= min_dim) {
                if (nconv < neig && nsub >= min_dim) {
                    result.nconverged = k_kept;
                }
                break;
            }

            // Restart from k_kept: place r_vec into Q_0[:, k_kept], and b_row into H_{p-1}[k_kept, 0..k_kept-1]
            memcpy(&result.Q(0)[k_kept * dims[0]], r_vec, dims[0] * sizeof(double));
            for (size_t n = 0; n < p; n++) {
                // Clear columns k_kept..nsub-1 and rows >= k_kept (except row k_kept of H_{p-1})
                for (size_t j = k_kept; j < nsub; j++) {
                    memset(&H_ptrs[n][j * nsub], 0, nsub * sizeof(double));
                }
                for (size_t j = 0; j < k_kept; j++) {
                    for (size_t i = k_kept; i < nsub; i++) {
                        H_ptrs[n][j * nsub + i] = 0.0;
                    }
                }
            }
            for (size_t j = 0; j < k_kept; j++) {
                H_ptrs[p - 1][j * nsub + k_kept] = b_row[j];
            }
            k_start = k_kept;
        }
    }
} // namespace Krylov
