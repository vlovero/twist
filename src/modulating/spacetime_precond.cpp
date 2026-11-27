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
        if (m_bdf_b1) {
            free(m_bdf_b1);
            m_bdf_b1 = nullptr;
        }
        if (m_bdf_b2) {
            free(m_bdf_b2);
            m_bdf_b2 = nullptr;
        }
        if (m_to_ref) {
            delete[] m_to_ref;
            m_to_ref = nullptr;
        }
        if (m_from_ref) {
            delete[] m_from_ref;
            m_from_ref = nullptr;
        }
        if (m_sig_t) {
            for (size_t i = 0; i < m_slice_cap; i++) {
                if (m_sig_t[i]) free(m_sig_t[i]);
            }
            free(m_sig_t);
            m_sig_t = nullptr;
        }
        if (m_sig_nnodes) {
            free(m_sig_nnodes);
            m_sig_nnodes = nullptr;
        }
        if (m_sig_cap) {
            free(m_sig_cap);
            m_sig_cap = nullptr;
        }
        if (m_ref_sig_t) {
            free(m_ref_sig_t);
            m_ref_sig_t = nullptr;
            m_ref_sig_nnodes = 0;
            m_ref_sig_cap = 0;
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
        if (m_V_out) {
            free(m_V_out);
            m_V_out = nullptr;
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
            m_bdf_b1 = (double *)realloc(m_bdf_b1, nt * sizeof(double));
            m_bdf_b2 = (double *)realloc(m_bdf_b2, nt * sizeof(double));
            delete[] m_to_ref;
            delete[] m_from_ref;
            m_to_ref = new SparseDenseOutputOp[nt];
            m_from_ref = new SparseDenseOutputOp[nt];

            m_sig_nnodes = (size_t *)realloc(m_sig_nnodes, nt * sizeof(size_t));
            m_sig_t = (double **)realloc(m_sig_t, nt * sizeof(double *));
            m_sig_cap = (size_t *)realloc(m_sig_cap, nt * sizeof(size_t));
            for (size_t i = m_slice_cap; i < nt; i++) {
                m_sig_nnodes[i] = 0;
                m_sig_t[i] = nullptr;
                m_sig_cap[i] = 0;
            }
            for (size_t i = 0; i < m_slice_cap; i++) {
                m_sig_nnodes[i] = 0; // transfers recreated above
            }
            m_slice_cap = nt;
        }
        if (m_nt_cap < nt) {
            m_S_tau = (double *)realloc(m_S_tau, nt * nt * sizeof(double));
            m_Q_tau = (double *)realloc(m_Q_tau, nt * nt * sizeof(double));
            m_V_out = (double *)realloc(m_V_out, nt * nt * sizeof(double));
            m_W_in = (double *)realloc(m_W_in, nt * nt * sizeof(double));
            m_lapack_work = (double *)realloc(m_lapack_work, (8 * nt * nt + 8 * nt + 64) * sizeof(double));
            m_ipiv = (int *)realloc(m_ipiv, nt * sizeof(int));
            m_bwork = (int *)realloc(m_bwork, (8 * nt + 64) * sizeof(int));
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

        const size_t max_threads = (size_t)std::max(1, omp_get_max_threads());
        // Scratch needed in update() and apply():
        //   r_ref (n_ref * nt), rhs_hat (n_ref * nt), z_hat (n_ref * nt), z_ref (n_ref * nt)
        //   plus tmp_ref per thread (max_threads * n_ref), bz0/bz1 (2 * n_ref), Y_tau/M_inv/SVD (6 * nt * nt + 8 * nt)
        const size_t needed_work = 4 * n_ref * nt + (max_threads + 4) * n_ref + 6 * nt * nt + 8 * nt + 256;
        if (m_work_cap < needed_work) {
            m_work = (double *)realloc(m_work, needed_work * sizeof(double));
            m_work_cap = needed_work;
        }
        const size_t needed_cwork = max_threads * 2 * n_ref;
        if (m_cwork_cap < needed_cwork) {
            m_cwork = (complex_t *)realloc(m_cwork, needed_cwork * sizeof(complex_t));
            m_cwork_cap = needed_cwork;
        }
        (void)total_end;
    }

    void RealSchurSpaceTimePrecond::update(size_t nt, const SpatialSlice *slices, const SubintervalCondenser *condensers, const double *y_end, const double *dtau, double T, const double *p, double theta, bool use_bdf2)
    {
        std::vector<const SpatialSlice *> ptrs(nt);
        for (size_t n = 0; n < nt; n++) {
            ptrs[n] = &slices[n];
        }
        update(nt, ptrs.data(), condensers, y_end, dtau, T, p, theta, use_bdf2);
    }

    void RealSchurSpaceTimePrecond::update(size_t nt, const SpatialSlice *const *slice_ptrs, const SubintervalCondenser *condensers, const double *y_end, const double *dtau, double T, const double *p, double theta, bool use_bdf2)
    {
        (void)condensers;
        const bool nt_changed = (m_nt != nt);
        m_nt = nt;
        m_T = T;
        m_theta = (use_bdf2 && nt >= 3) ? 1.0 : theta;
        m_use_bdf2 = (use_bdf2 && nt >= 3);

        // Choose reference mesh: slice with the largest number of subintervals (already adapted to sharpest front),
        // or copy slice_ptrs[0] if all equal.
        size_t best_slice = 0;
        size_t max_nnodes = slice_ptrs[0]->nnodes;
        size_t total_end = 0;
        for (size_t n = 0; n < nt; n++) {
            total_end += slice_ptrs[n]->n_space;
            if (slice_ptrs[n]->nnodes > max_nnodes) {
                max_nnodes = slice_ptrs[n]->nnodes;
                best_slice = n;
            }
        }
        m_total_end = total_end;
        const SpatialSlice *ref_src = slice_ptrs[best_slice];
        bool ref_changed = nt_changed || (m_ref_sig_nnodes != ref_src->nnodes) ||
            (m_ref_sig_t == nullptr) || (memcmp(m_ref_sig_t, ref_src->t, ref_src->nnodes * sizeof(double)) != 0);
        if (ref_changed) {
            m_ref_slice = *ref_src;
            m_n_ref = m_ref_slice.n_space;
            m_ref_sig_nnodes = ref_src->nnodes;
            if (m_ref_sig_cap < ref_src->nnodes) {
                m_ref_sig_t = (double *)realloc(m_ref_sig_t, ref_src->nnodes * sizeof(double));
                m_ref_sig_cap = ref_src->nnodes;
            }
            memcpy(m_ref_sig_t, ref_src->t, ref_src->nnodes * sizeof(double));
        }

        ensureBuffers(nt, m_n_ref, total_end);

        m_offsets[0] = 0;
        for (size_t n = 0; n < nt; n++) {
            const SpatialSlice *sn = slice_ptrs[n];
            m_offsets[n + 1] = m_offsets[n] + sn->n_space;
            m_dtau[n] = dtau[n];
            const bool slice_changed = ref_changed || (m_sig_nnodes[n] != sn->nnodes) ||
                (m_sig_t[n] == nullptr) || (memcmp(m_sig_t[n], sn->t, sn->nnodes * sizeof(double)) != 0);
            if (slice_changed) {
                m_to_ref[n].build(*sn, m_ref_slice);
                m_from_ref[n].build(m_ref_slice, *sn);
                m_sig_nnodes[n] = sn->nnodes;
                if (m_sig_cap[n] < sn->nnodes) {
                    m_sig_t[n] = (double *)realloc(m_sig_t[n], sn->nnodes * sizeof(double));
                    m_sig_cap[n] = sn->nnodes;
                }
                memcpy(m_sig_t[n], sn->t, sn->nnodes * sizeof(double));
            }
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

        // Build Nt x Nt matrix D_tau (and M_tau when theta != 1.0)
        const int int_nt = (int)nt;
        double *D_tau = m_S_tau;
        memset(D_tau, 0, nt * nt * sizeof(double));
        if (m_use_bdf2) {
            for (size_t n = 0; n < nt; n++) {
                const size_t prev1 = (n == 0) ? (nt - 1) : (n - 1);
                const size_t prev2 = (prev1 == 0) ? (nt - 1) : (prev1 - 1);
                const double h1 = m_dtau[n];
                const double h0 = m_dtau[prev1];
                const double a_n = 1.0 / h1 + 1.0 / (h1 + h0);
                const double b_n = -(h1 + h0) / (h1 * h0);
                const double c_n = h0 / (h1 * (h1 + h0));
                m_bdf_b1[n] = b_n;
                m_bdf_b2[n] = c_n;
                D_tau[n * nt + n] += a_n;
                D_tau[prev1 * nt + n] += b_n;
                D_tau[prev2 * nt + n] += c_n;
            }
        }
        else {
            for (size_t n = 0; n < nt; n++) {
                const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
                const double inv_dt = 1.0 / m_dtau[n];
                D_tau[n * nt + n] += inv_dt;
                D_tau[prev * nt + n] -= inv_dt;
            }
        }

        double *M_inv_save = m_work + nt * nt;
        int info = 0;
        if (m_theta == 1.0) {
            // M_tau = I -> H_tau = D_tau, W_in_schur = Q_tau^T
            memset(M_inv_save, 0, nt * nt * sizeof(double));
            for (size_t i = 0; i < nt; i++) {
                M_inv_save[i * nt + i] = 1.0;
            }
        }
        else {
            double *M_tau = m_W_in;
            memset(M_tau, 0, nt * nt * sizeof(double));
            for (size_t n = 0; n < nt; n++) {
                const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
                M_tau[n * nt + n] += m_theta;
                M_tau[prev * nt + n] += (1.0 - m_theta);
            }
            double *M_inv = m_work;
            memset(M_inv, 0, nt * nt * sizeof(double));
            for (size_t i = 0; i < nt; i++) {
                M_inv[i * nt + i] = 1.0;
            }
            dgetrf(int_nt, int_nt, M_tau, int_nt, m_ipiv, &info);
            dgetrs("N", int_nt, int_nt, M_tau, int_nt, m_ipiv, M_inv, int_nt, &info);
            dgetrs("N", int_nt, int_nt, M_tau, int_nt, m_ipiv, D_tau, int_nt, &info);
            memcpy(M_inv_save, M_inv, nt * nt * sizeof(double));
        }

        // Orthogonal real Schur decomposition: H_tau = Q_tau * S_tau * Q_tau^T
        double *wr = m_work;
        double *wi = wr + nt;
        int sdim = 0;
        int lwork = (int)(8 * nt * nt + 8 * nt + 64);
        dgees("V", "N", nullptr, int_nt, m_S_tau, int_nt, &sdim, wr, wi, m_Q_tau, int_nt, m_lapack_work, lwork, m_bwork, &info);

        // Compute orthogonal W_in = Q_tau^T * M_tau^(-1) (or Q_tau^T when theta == 1.0)
        if (m_theta == 1.0) {
            for (size_t r = 0; r < nt; r++) {
                for (size_t c = 0; c < nt; c++) {
                    m_W_in[c * nt + r] = m_Q_tau[r * nt + c];
                }
            }
        }
        else {
            dgemm("T", "N", int_nt, int_nt, int_nt, 1.0, m_Q_tau, int_nt, M_inv_save, int_nt, 0.0, m_W_in, int_nt);
        }

        // Parse 1x1 and 2x2 Schur blocks from m_S_tau & measure strict upper-block off-diagonal norm
        m_nblocks = 0;
        m_nreal_mats = 0;
        m_ncomplex_mats = 0;
        size_t k = 0;
        while (k < nt) {
            if (k + 1 < nt && std::abs(m_S_tau[k * nt + (k + 1)]) > 1e-13 &&
                (m_S_tau[k * nt + (k + 1)] * m_S_tau[(k + 1) * nt + k]) < 0.0) {
                const double a = m_S_tau[k * nt + k];
                const double b = m_S_tau[k * nt + (k + 1)];
                const double c = m_S_tau[(k + 1) * nt + k];
                const double prod = -b * c;
                const double beta = prod > 0.0 ? std::sqrt(prod) : 0.0;
                const double kappa = std::sqrt(std::abs(c) / std::abs(b));
                const double sgn = (c >= 0.0) ? 1.0 : -1.0;

                m_blocks[m_nblocks++] = { true, false, k, a, beta, kappa, sgn, m_ncomplex_mats++ };
                k += 2;
            }
            else {
                const double a = m_S_tau[k * nt + k];
                m_blocks[m_nblocks++] = { false, false, k, a, 0.0, 1.0, 1.0, m_nreal_mats++ };
                k += 1;
            }
        }

        double diag_norm = 0.0;
        double off_norm = 0.0;
        for (size_t blk_i = 0; blk_i < m_nblocks; blk_i++) {
            SchurBlock &blk = m_blocks[blk_i];
            const size_t c = blk.col;
            diag_norm = std::max(diag_norm, std::abs(blk.a) + blk.beta);
            bool has_u = false;
            const size_t ncols = blk.is_complex ? 2 : 1;
            for (size_t dc = 0; dc < ncols; dc++) {
                for (size_t r = 0; r < c; r++) {
                    const double v = std::abs(m_S_tau[(c + dc) * nt + r]);
                    if (v > 0.0) {
                        has_u = true;
                        if (v > off_norm) off_norm = v;
                    }
                }
            }
            blk.has_upper = has_u;
        }

        m_is_block_diag = false;
        memcpy(m_V_out, m_Q_tau, nt * nt * sizeof(double));
        if (off_norm <= 1e-12 * std::max(1.0, diag_norm)) {
            m_is_block_diag = true;
        }
        else {
            double *Y_tau = m_work + 2 * nt * nt;
            double *Y_copy = Y_tau + nt * nt;
            double *svals = Y_copy + nt * nt;
            int m_out = 0;
            int info_ev = 0;
            dtrevc3("R", "A", nullptr, int_nt, m_S_tau, int_nt, nullptr, int_nt, Y_tau, int_nt, int_nt, &m_out, m_lapack_work, lwork, &info_ev);
            if (info_ev == 0 && m_out == int_nt) {
                memcpy(Y_copy, Y_tau, nt * nt * sizeof(double));
                int info_svd = 0;
                dgesdd("N", int_nt, int_nt, Y_copy, int_nt, svals, nullptr, int_nt, nullptr, int_nt, m_lapack_work, lwork, m_bwork, &info_svd);
                const double cond = (info_svd == 0 && svals[nt - 1] > 0.0) ? (svals[0] / svals[nt - 1]) : 1e30;
                if (cond <= 10.0) {
                    // Solve Y_tau * W_in_new = W_in_schur (via dgetrf/dgetrs) and V_out = Q_tau * Y_tau
                    memcpy(Y_copy, Y_tau, nt * nt * sizeof(double));
                    int info_lu = 0;
                    dgetrf(int_nt, int_nt, Y_copy, int_nt, m_ipiv, &info_lu);
                    if (info_lu == 0) {
                        double *W_new = svals + nt; // nt * nt
                        memcpy(W_new, m_W_in, nt * nt * sizeof(double));
                        dgetrs("N", int_nt, int_nt, Y_copy, int_nt, m_ipiv, W_new, int_nt, &info_lu);
                        if (info_lu == 0) {
                            memcpy(m_W_in, W_new, nt * nt * sizeof(double));
                            dgemm("N", "N", int_nt, int_nt, int_nt, 1.0, m_Q_tau, int_nt, Y_tau, int_nt, 0.0, m_V_out, int_nt);
                            // Normalize 2x2 diagonal blocks of B_tau to [a, beta; -beta, a] (kappa=1, sgn=+1)
                            for (size_t blk_i = 0; blk_i < m_nblocks; blk_i++) {
                                if (m_blocks[blk_i].is_complex) {
                                    m_blocks[blk_i].kappa = 1.0;
                                    m_blocks[blk_i].sgn = 1.0;
                                }
                            }
                            m_is_block_diag = true;
                        }
                    }
                }
            }
        }

        // Factor the decoupled spatial blocks in parallel across OpenMP threads
        omp_set_max_active_levels(1);
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
                mat.copyPartitionsFrom(m_ref_jac, &m_real_bc, false);
                mat.addMassBlockSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, m_ref_slice.c_A, m_ref_slice.c_b, m_ref_slice.h, -a_eff / m_T, false);
                mat.factor();
            }
            else {
                DEV::ComplexMatrix &mat = m_complex_mats[blk.mat_idx];
                mat.copyPartitionsFrom(m_ref_jac, &m_complex_bc, false);
                // Solve (A_ref - (a - i * sgn * beta)/T * B_ref) (z_k + i z_{k+1}) = -(1/T) (r_k + i r_{k+1})
                const complex_t scale(-a_eff / m_T, +(blk.sgn * blk.beta) / m_T);
                mat.addMassBlockSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, m_ref_slice.c_A, m_ref_slice.c_b, m_ref_slice.h, scale, false);
                mat.factor();
            }
        }
    }

    void RealSchurSpaceTimePrecond::apply(const RP(double) r, RP(double) z) const
    {
        omp_set_max_active_levels(1);
        OpenBLASThreadContext __openblas_context(1);

        const size_t nt = m_nt;
        const size_t n_ref = m_n_ref;
        const int in_ref = (int)n_ref;
        const int int_nt = (int)nt;

        double *r_ref = m_work;                     // n_ref * nt
        double *rhs_hat = r_ref + n_ref * nt;       // n_ref * nt
        double *z_hat = rhs_hat + n_ref * nt;       // n_ref * nt
        double *z_ref = z_hat + n_ref * nt;         // n_ref * nt
        double *thread_tmp = z_ref + n_ref * nt;    // max_threads * n_ref
        const size_t max_threads = (size_t)std::max(1, omp_get_max_threads());
        double *bz_scratch = thread_tmp + max_threads * n_ref;

        // 1. Identity-split form in parallel across slices
#pragma omp parallel for schedule(static) if (nt * n_ref >= 65536)
        for (size_t n = 0; n < nt; n++) {
            const size_t tid = (size_t)omp_get_thread_num();
            double *tmp_ref = thread_tmp + tid * n_ref;
            const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
            m_to_ref[prev].apply(&r[m_offsets[prev]], tmp_ref);
            if (m_use_bdf2) {
                const size_t prev2 = (prev == 0) ? (nt - 1) : (prev - 1);
                m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, tmp_ref, &r_ref[n * n_ref], -m_bdf_b1[n], 0.0);
                m_to_ref[prev2].apply(&r[m_offsets[prev2]], tmp_ref);
                m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, tmp_ref, &r_ref[n * n_ref], -m_bdf_b2[n], 1.0);
            }
            else {
                m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, tmp_ref, &r_ref[n * n_ref], 1.0 / m_dtau[n], 0.0);
                if (m_theta != 1.0) {
                    m_ref_jac.gemv(1.0, &r_ref[n * n_ref], (1.0 - m_theta) * m_T, tmp_ref);
                }
            }
        }

        // 2. Transform RHS: rhs_hat = r_ref * W_in^T
        dgemm("N", "T", in_ref, int_nt, int_nt, 1.0, r_ref, in_ref, m_W_in, int_nt, 0.0, rhs_hat, in_ref);

        const double inv_T = -1.0 / m_T;
        if (m_is_block_diag) {
            // 3a. Parallel decoupled block solves with zero back-substitution or applyMass overhead
#pragma omp parallel for schedule(dynamic, 1) if (m_nblocks >= 2 && n_ref >= 1500)
            for (size_t blk_i = 0; blk_i < m_nblocks; blk_i++) {
                const SchurBlock &blk = m_blocks[blk_i];
                const size_t c = blk.col;
                if (!blk.is_complex) {
                    double *rk = &rhs_hat[c * n_ref];
                    for (size_t i = 0; i < n_ref; i++) {
                        rk[i] *= inv_T;
                    }
                    m_real_mats[blk.mat_idx].solve(rk, &z_hat[c * n_ref]);
                }
                else {
                    const size_t tid = (size_t)omp_get_thread_num();
                    complex_t *bc = m_cwork + tid * 2 * n_ref;
                    complex_t *zc = bc + n_ref;
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
                }
            }
        }
        else {
            // 3b. Quasi-upper-triangular block back-substitution fallback
            double *bz0 = bz_scratch;
            double *bz1 = bz0 + n_ref;
            complex_t *bc = m_cwork;
            complex_t *zc = bc + n_ref;

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

                    if (c > 0 && blk.has_upper) {
                        m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, zk, bz0, 1.0, 0.0);
                        dgemm("N", "T", in_ref, (int)c, 1, -1.0, bz0, in_ref, &m_S_tau[c * nt], (int)c, 1.0, rhs_hat, in_ref);
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

                    if (c > 0 && blk.has_upper) {
                        m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, zk0, bz0, 1.0, 0.0);
                        m_ref_slice.applyMassOpSparse(m_ref_slice.m_nnz, m_ref_slice.m_rows, m_ref_slice.m_cols, m_ref_slice.m_vals, zk1, bz1, 1.0, 0.0);
                        dgemm("N", "T", in_ref, (int)c, 1, -1.0, bz0, in_ref, &m_S_tau[c * nt], (int)c, 1.0, rhs_hat, in_ref);
                        dgemm("N", "T", in_ref, (int)c, 1, -1.0, bz1, in_ref, &m_S_tau[(c + 1) * nt], (int)c, 1.0, rhs_hat, in_ref);
                    }
                }
            }
        }

        // 4. Transform back: z_ref = z_hat * V_out^T
        dgemm("N", "T", in_ref, int_nt, int_nt, 1.0, z_hat, in_ref, m_V_out, int_nt, 0.0, z_ref, in_ref);

        // 5. Project w_ref_n -> slice n and add back r_n: z_n = r_n + P_from[n](w_ref_n)
#pragma omp parallel for schedule(static) if (nt * n_ref >= 65536)
        for (size_t n = 0; n < nt; n++) {
            const size_t tid = (size_t)omp_get_thread_num();
            double *tmp_ref = thread_tmp + tid * n_ref;
            const size_t ns = m_offsets[n + 1] - m_offsets[n];
            const size_t off = m_offsets[n];
            m_from_ref[n].apply(&z_ref[n * n_ref], tmp_ref);
            for (size_t i = 0; i < ns; i++) {
                z[off + i] = r[off + i] + tmp_ref[i];
            }
        }
    }
} // namespace TWIST
