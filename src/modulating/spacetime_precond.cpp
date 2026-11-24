#include "modulating/spacetime_precond.h"
#include "linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

namespace TWIST
{
    RealSchurSpaceTimePrecond::~RealSchurSpaceTimePrecond()
    {
        freeBuffers();
    }

    void RealSchurSpaceTimePrecond::freeBuffers()
    {
        if (m_offsets) {
            free(m_offsets);
            m_offsets = nullptr;
        }
        if (m_dtau) {
            free(m_dtau);
            m_dtau = nullptr;
        }
        if (m_to_ref) {
            delete[] m_to_ref;
            m_to_ref = nullptr;
        }
        if (m_from_ref) {
            delete[] m_from_ref;
            m_from_ref = nullptr;
        }
        m_slice_cap = 0;

        if (m_S_tau) {
            free(m_S_tau);
            m_S_tau = nullptr;
        }
        if (m_Q_tau) {
            free(m_Q_tau);
            m_Q_tau = nullptr;
        }
        if (m_W_in) {
            free(m_W_in);
            m_W_in = nullptr;
        }
        if (m_lapack_work) {
            free(m_lapack_work);
            m_lapack_work = nullptr;
        }
        if (m_ipiv) {
            free(m_ipiv);
            m_ipiv = nullptr;
        }
        if (m_bwork) {
            free(m_bwork);
            m_bwork = nullptr;
        }
        if (m_blocks) {
            free(m_blocks);
            m_blocks = nullptr;
        }
        m_nt_cap = 0;

        if (m_real_mats) {
            delete[] m_real_mats;
            m_real_mats = nullptr;
            m_real_cap = 0;
        }
        if (m_complex_mats) {
            delete[] m_complex_mats;
            m_complex_mats = nullptr;
            m_complex_cap = 0;
        }
        if (m_work) {
            free(m_work);
            m_work = nullptr;
            m_work_cap = 0;
        }
        if (m_cwork) {
            free(m_cwork);
            m_cwork = nullptr;
            m_cwork_cap = 0;
        }
    }

    void RealSchurSpaceTimePrecond::ensureBuffers(size_t nt, size_t n_ref, size_t total_end)
    {
        if (m_slice_cap < nt) {
            m_offsets = (size_t *)realloc(m_offsets, (nt + 1) * sizeof(size_t));
            m_dtau = (double *)realloc(m_dtau, nt * sizeof(double));
            delete[] m_to_ref;
            delete[] m_from_ref;
            m_to_ref = new SparseDenseOutputOp[nt];
            m_from_ref = new SparseDenseOutputOp[nt];
            m_slice_cap = nt;
        }
        if (m_nt_cap < nt) {
            m_S_tau = (double *)realloc(m_S_tau, nt * nt * sizeof(double));
            m_Q_tau = (double *)realloc(m_Q_tau, nt * nt * sizeof(double));
            m_W_in = (double *)realloc(m_W_in, nt * nt * sizeof(double));
            m_lapack_work = (double *)realloc(m_lapack_work, (8 * nt + 64) * sizeof(double));
            m_ipiv = (int *)realloc(m_ipiv, nt * sizeof(int));
            m_bwork = (int *)realloc(m_bwork, nt * sizeof(int));
            m_blocks = (SchurBlock *)realloc(m_blocks, nt * sizeof(SchurBlock));
            m_nt_cap = nt;
        }
        if (m_real_cap < nt) {
            delete[] m_real_mats;
            m_real_mats = new DEV::Matrix[nt];
            m_real_cap = nt;
        }
        if (m_complex_cap < nt) {
            delete[] m_complex_mats;
            m_complex_mats = new DEV::ComplexMatrix[nt];
            m_complex_cap = nt;
        }

        // Scratch needed in apply():
        //   r_scaled (total_end? no, per-slice in ref mesh!)
        //   r_ref (n_ref * nt), rhs_hat (n_ref * nt), z_hat (n_ref * nt), z_ref (n_ref * nt)
        //   plus tmp_ref (n_ref * 4), mass_work (node * nstages)
        const size_t needed_work = 4 * n_ref * nt + 4 * n_ref + nt * nt * 2 + 256;
        if (m_work_cap < needed_work) {
            m_work = (double *)realloc(m_work, needed_work * sizeof(double));
            m_work_cap = needed_work;
        }
        const size_t needed_cwork = 2 * n_ref;
        if (m_cwork_cap < needed_cwork) {
            m_cwork = (complex_t *)realloc(m_cwork, needed_cwork * sizeof(complex_t));
            m_cwork_cap = needed_cwork;
        }
        (void)total_end;
    }

    void RealSchurSpaceTimePrecond::update(size_t nt, const SpatialSlice *slices, const SubintervalCondenser *condensers, const double *y_end, const double *dtau, double T, const double *p, double theta)
    {
        (void)condensers;
        m_nt = nt;
        m_T = T;
        m_theta = theta;

        // Choose reference mesh: slice with the largest number of subintervals (already adapted to sharpest front),
        // or copy slices[0] if all equal.
        size_t best_slice = 0;
        size_t max_nnodes = slices[0].nnodes;
        size_t total_end = 0;
        for (size_t n = 0; n < nt; n++) {
            total_end += slices[n].n_space;
            if (slices[n].nnodes > max_nnodes) {
                max_nnodes = slices[n].nnodes;
                best_slice = n;
            }
        }
        m_total_end = total_end;
        m_ref_slice = slices[best_slice];
        m_n_ref = m_ref_slice.n_space;

        ensureBuffers(nt, m_n_ref, total_end);

        m_offsets[0] = 0;
        for (size_t n = 0; n < nt; n++) {
            m_offsets[n + 1] = m_offsets[n] + slices[n].n_space;
            m_dtau[n] = dtau[n];
            m_to_ref[n].build(slices[n], m_ref_slice);
            m_from_ref[n].build(m_ref_slice, slices[n]);
        }

        // Compute time-averaged state on m_ref_slice
        double *y_avg_ref = m_work;
        double *y_tmp_ref = y_avg_ref + m_n_ref;
        memset(y_avg_ref, 0, m_n_ref * sizeof(double));
        const double inv_nt = 1.0 / (double)nt;
        for (size_t n = 0; n < nt; n++) {
            m_to_ref[n].apply(&y_end[m_offsets[n]], y_tmp_ref);
            for (size_t i = 0; i < m_n_ref; i++) {
                y_avg_ref[i] += inv_nt * y_tmp_ref[i];
            }
        }

        m_real_bc.makePeriodic((ptrdiff_t)m_ref_slice.node, 0);
        m_complex_bc.makePeriodic((ptrdiff_t)m_ref_slice.node, 0);

        int pmask_dummy[2] = { 0, 0 };
        m_ref_jac.update(m_ref_slice.fjac, m_ref_slice.pjac, (ptrdiff_t)m_ref_slice.nnodes, (ptrdiff_t)m_ref_slice.node, (ptrdiff_t)m_ref_slice.nstages, 0, m_ref_slice.unscaled_spatial_period, m_ref_slice.h, y_avg_ref, p, pmask_dummy, nullptr, &m_real_bc);

        // Build Nt x Nt matrices D_tau and M_tau (column-major)
        const int int_nt = (int)nt;
        double *D_tau = m_S_tau;
        double *M_tau = m_W_in;
        memset(D_tau, 0, nt * nt * sizeof(double));
        memset(M_tau, 0, nt * nt * sizeof(double));

        for (size_t n = 0; n < nt; n++) {
            const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
            const double inv_dt = 1.0 / m_dtau[n];
            // Column-major: (row n, col j) -> j * nt + n
            D_tau[n * nt + n] += inv_dt;
            D_tau[prev * nt + n] -= inv_dt;

            M_tau[n * nt + n] += m_theta;
            M_tau[prev * nt + n] += (1.0 - m_theta);
        }

        // Solve M_tau * H_tau = D_tau  -> D_tau overwritten with H_tau = M_tau^(-1) * D_tau
        // Also compute M_tau^(-1) in m_Q_tau temporarily so we can form W_in = Q_tau^T * M_tau^(-1)
        double *M_inv = m_work;
        memset(M_inv, 0, nt * nt * sizeof(double));
        for (size_t i = 0; i < nt; i++) {
            M_inv[i * nt + i] = 1.0;
        }
        int info = 0;
        dgetrf(int_nt, int_nt, M_tau, int_nt, m_ipiv, &info);
        dgetrs("N", int_nt, int_nt, M_tau, int_nt, m_ipiv, M_inv, int_nt, &info);
        dgetrs("N", int_nt, int_nt, M_tau, int_nt, m_ipiv, D_tau, int_nt, &info);

        // Copy M_inv into temp buffer m_work + nt*nt before dgees uses m_work? No, dgees uses m_lapack_work!
        double *M_inv_save = m_work + nt * nt;
        memcpy(M_inv_save, M_inv, nt * nt * sizeof(double));

        // Orthogonal real Schur decomposition: H_tau = Q_tau * S_tau * Q_tau^T
        double *wr = m_work;
        double *wi = wr + nt;
        int sdim = 0;
        int lwork = (int)(8 * nt + 64);
        dgees("V", "N", nullptr, int_nt, m_S_tau, int_nt, &sdim, wr, wi, m_Q_tau, int_nt, m_lapack_work, lwork, m_bwork, &info);

        // Compute m_W_in = Q_tau^T * M_tau^(-1)
        dgemm("T", "N", int_nt, int_nt, int_nt, 1.0, m_Q_tau, int_nt, M_inv_save, int_nt, 0.0, m_W_in, int_nt);

        // Parse 1x1 and 2x2 Schur blocks from m_S_tau
        m_nblocks = 0;
        m_nreal_mats = 0;
        m_ncomplex_mats = 0;
        size_t k = 0;
        while (k < nt) {
            if (k + 1 < nt && std::abs(m_S_tau[k * nt + (k + 1)]) > 1e-13 &&
                (m_S_tau[k * nt + (k + 1)] * m_S_tau[(k + 1) * nt + k]) < 0.0) {
                const double a = m_S_tau[k * nt + k];
                const double b = m_S_tau[k * nt + (k + 1)];       // S(k+1, k)
                const double c = m_S_tau[(k + 1) * nt + k];       // S(k, k+1)
                const double prod = -b * c;
                const double beta = prod > 0.0 ? std::sqrt(prod) : 0.0;
                const double kappa = std::sqrt(std::abs(c) / std::abs(b));
                const double sgn = (c >= 0.0) ? 1.0 : -1.0;

                m_blocks[m_nblocks++] = { true, k, a, beta, kappa, sgn, m_ncomplex_mats++ };
                k += 2;
            }
            else {
                const double a = m_S_tau[k * nt + k];
                m_blocks[m_nblocks++] = { false, k, a, 0.0, 1.0, 1.0, m_nreal_mats++ };
                k += 1;
            }
        }

        // Factor the decoupled spatial blocks in parallel across OpenMP threads
        OpenBLASThreadContext __openblas_context(1);

#pragma omp parallel for schedule(dynamic, 1)
        for (size_t blk_i = 0; blk_i < m_nblocks; blk_i++) {
            const SchurBlock &blk = m_blocks[blk_i];
            // Ensure tiny positive regularization on zero/near-zero modes (like k=0 translational mode)
            double a_eff = blk.a;
            if (!blk.is_complex && std::abs(a_eff) < 1e-4) {
                a_eff = (a_eff >= 0.0) ? 1e-4 : -1e-4;
            }
            else if (blk.is_complex && std::abs(a_eff) < 1e-5 && blk.beta < 1e-5) {
                a_eff = 1e-4;
            }

            if (!blk.is_complex) {
                DEV::Matrix &mat = m_real_mats[blk.mat_idx];
                mat.copyPartitionsFrom(m_ref_jac, &m_real_bc);
                mat.addMassBlock(m_ref_slice.M, m_ref_slice.c_A, m_ref_slice.c_b, m_ref_slice.h, -a_eff / m_T);
                mat.factor();
            }
            else {
                DEV::ComplexMatrix &mat = m_complex_mats[blk.mat_idx];
                mat.copyPartitionsFrom(m_ref_jac, &m_complex_bc);
                // Solve (A_ref - (a - i * sgn * beta)/T * B_ref) (z_k + i z_{k+1}) = -(1/T) (r_k + i r_{k+1})
                const complex_t scale(-a_eff / m_T, +(blk.sgn * blk.beta) / m_T);
                mat.addMassBlock(m_ref_slice.M, m_ref_slice.c_A, m_ref_slice.c_b, m_ref_slice.h, scale);
                mat.factor();
            }
        }
    }

    void RealSchurSpaceTimePrecond::apply(const RP(double) r, RP(double) z) const
    {
        const size_t nt = m_nt;
        const size_t n_ref = m_n_ref;
        const int in_ref = (int)n_ref;
        const int int_nt = (int)nt;

        double *r_ref = m_work;                     // n_ref * nt
        double *rhs_hat = r_ref + n_ref * nt;       // n_ref * nt
        double *z_hat = rhs_hat + n_ref * nt;       // n_ref * nt
        double *z_ref = z_hat + n_ref * nt;         // n_ref * nt
        double *tmp_ref = z_ref + n_ref * nt;       // n_ref
        double *tmp_Br = tmp_ref + n_ref;           // n_ref
        double *mass_work = tmp_Br + n_ref;         // >= node * nstages

        // 1. Identity-split form: z_n = r_n + P_from[n]( w_ref[n] ) where
        //    (D_tau x B_ref - M_tau x T * A_ref) w_ref = (1/dtau_n B_ref + (1 - theta) * T * A_ref) P_to[prev](r_prev).
        //    Preserves exact identity on algebraic & stiff modes across arbitrary spatial mesh jumps.
        for (size_t n = 0; n < nt; n++) {
            const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
            m_to_ref[prev].apply(&r[m_offsets[prev]], tmp_ref);
            m_ref_slice.applyMass(tmp_ref, &r_ref[n * n_ref], 1.0 / m_dtau[n], 0.0, mass_work);
            if (m_theta != 1.0) {
                m_ref_jac.gemv(1.0, &r_ref[n * n_ref], (1.0 - m_theta) * m_T, tmp_ref);
            }
        }

        // 2. Transform RHS: rhs_hat = r_ref * W_in^T  (where W_in = Q_tau^T * M_tau^(-1))
        dgemm("N", "T", in_ref, int_nt, int_nt, 1.0, r_ref, in_ref, m_W_in, int_nt, 0.0, rhs_hat, in_ref);

        // 3. Quasi-upper-triangular block back-substitution from k = m_nblocks - 1 down to 0
        double *bz0 = tmp_ref;
        double *bz1 = tmp_Br;
        complex_t *bc = m_cwork;
        complex_t *zc = bc + n_ref;
        const double inv_T = -1.0 / m_T;

        for (size_t blk_rev = m_nblocks; blk_rev > 0; blk_rev--) {
            const SchurBlock &blk = m_blocks[blk_rev - 1];
            const size_t c = blk.col;

            if (!blk.is_complex) {
                double *rk = &rhs_hat[c * n_ref];
                for (size_t i = 0; i < n_ref; i++) {
                    rk[i] *= inv_T;
                }
                double *zk = &z_hat[c * n_ref];
                m_real_mats[blk.mat_idx].solve(rk, zk);

                // Update earlier blocks i < c: rhs_hat[i] -= S_tau(i, c) * (B_ref * zk)
                if (c > 0) {
                    m_ref_slice.applyMass(zk, bz0, 1.0, 0.0, mass_work);
                    for (size_t i = 0; i < c; i++) {
                        const double s_ic = m_S_tau[c * nt + i];
                        if (s_ic != 0.0) {
                            double *ri = &rhs_hat[i * n_ref];
                            for (size_t q = 0; q < n_ref; q++) {
                                ri[q] -= s_ic * bz0[q];
                            }
                        }
                    }
                }
            }
            else {
                const double *rk0 = &rhs_hat[c * n_ref];
                const double *rk1 = &rhs_hat[(c + 1) * n_ref];
                const double inv_kappa = 1.0 / blk.kappa;
                for (size_t i = 0; i < n_ref; i++) {
                    bc[i] = complex_t(inv_T * inv_kappa * rk0[i], inv_T * rk1[i]);
                }
                m_complex_mats[blk.mat_idx].solve(bc, zc);

                double *zk0 = &z_hat[c * n_ref];
                double *zk1 = &z_hat[(c + 1) * n_ref];
                for (size_t i = 0; i < n_ref; i++) {
                    zk0[i] = blk.kappa * zc[i].real();
                    zk1[i] = zc[i].imag();
                }

                // Update earlier blocks i < c: rhs_hat[i] -= S_tau(i, c)*(B_ref*zk0) + S_tau(i, c+1)*(B_ref*zk1)
                if (c > 0) {
                    m_ref_slice.applyMass(zk0, bz0, 1.0, 0.0, mass_work);
                    m_ref_slice.applyMass(zk1, bz1, 1.0, 0.0, mass_work);
                    for (size_t i = 0; i < c; i++) {
                        const double s_ic0 = m_S_tau[c * nt + i];
                        const double s_ic1 = m_S_tau[(c + 1) * nt + i];
                        double *ri = &rhs_hat[i * n_ref];
                        if (s_ic0 != 0.0 || s_ic1 != 0.0) {
                            for (size_t q = 0; q < n_ref; q++) {
                                ri[q] -= s_ic0 * bz0[q] + s_ic1 * bz1[q];
                            }
                        }
                    }
                }
            }
        }

        // 4. Transform back: z_ref = z_hat * Q_tau^T
        dgemm("N", "T", in_ref, int_nt, int_nt, 1.0, z_hat, in_ref, m_Q_tau, int_nt, 0.0, z_ref, in_ref);

        // 5. Project w_ref_n -> slice n and add back r_n: z_n = r_n + P_from[n](w_ref_n)
        for (size_t n = 0; n < nt; n++) {
            const size_t ns = m_offsets[n + 1] - m_offsets[n];
            const size_t off = m_offsets[n];
            m_from_ref[n].apply(&z_ref[n * n_ref], tmp_ref);
            for (size_t i = 0; i < ns; i++) {
                z[off + i] = r[off + i] + tmp_ref[i];
            }
        }
    }
} // namespace TWIST
