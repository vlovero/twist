#include "modulating/subinterval_condenser.h"
#include "krylov/gmres.h"
#include "linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace TWIST
{
    SubintervalCondenser::~SubintervalCondenser()
    {
        freeBuffers();
    }

    void SubintervalCondenser::freeBuffers()
    {
        if (m_stage_jacs) {
            delete[] m_stage_jacs;
            m_stage_jacs = nullptr;
            m_stage_jacs_cap = 0;
        }
        if (m_y_avg) {
            free(m_y_avg);
            m_y_avg = nullptr;
        }
        if (m_p_in) {
            free(m_p_in);
            m_p_in = nullptr;
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
        if (m_pjac_work) {
            free(m_pjac_work);
            m_pjac_work = nullptr;
            m_pjac_work_cap = 0;
        }
        if (m_gmres_work) {
            free(m_gmres_work);
            m_gmres_work = nullptr;
            m_gmres_work_cap = 0;
        }
    }

    void SubintervalCondenser::ensureWorkspace(size_t n_space, size_t st)
    {
        if (m_stage_jacs_cap < st) {
            delete[] m_stage_jacs;
            m_stage_jacs = new DEV::Matrix[st];
            m_stage_jacs_cap = st;
        }
        if (m_n_space != n_space || m_st != st) {
            m_y_avg = (double *)realloc(m_y_avg, n_space * sizeof(double));
            m_p_in = (double *)realloc(m_p_in, st * st * sizeof(double));
        }

        // Needs enough space for:
        //  0..1: b_hat, z_hat (2*total)
        //  2..3: ay, mass_work (2*total)
        //  4..6: rhs_copy, res, corr (3*total)
        //  7..9: rhs_stages, dk_stages, transfer_buf (3*total)
        const size_t needed_real = 10 * n_space * st + 256;
        if (m_work_cap < needed_real) {
            m_work = (double *)realloc(m_work, needed_real * sizeof(double));
            m_work_cap = needed_real;
        }

        const size_t needed_complex = 2 * n_space;
        if (m_cwork_cap < needed_complex) {
            m_cwork = (complex_t *)realloc(m_cwork, needed_complex * sizeof(complex_t));
            m_cwork_cap = needed_complex;
        }

        const size_t node = m_slice ? m_slice->node : 1;
        const size_t needed_pjac = m_slice ? (m_slice->nstages * node) : node;
        if (m_pjac_work_cap < needed_pjac) {
            m_pjac_work = (double *)realloc(m_pjac_work, needed_pjac * sizeof(double));
            m_pjac_work_cap = needed_pjac;
        }
    }

    void SubintervalCondenser::factor(const SpatialSlice *slice, size_t st, const double *y_stages, const double *p, double dtau_T, bool build_stage_jacs)
    {
        m_slice = slice;
        m_dtau_T = dtau_T;
        const size_t n_space = slice->n_space;
        ensureWorkspace(n_space, st);
        m_st = st;
        m_n_space = n_space;
        m_has_stage_jacs = build_stage_jacs;

        const RadauDGMethod &m = getRadauDG(st);

        // Precompute row-major m_p_in = T_s^(-1) * A_inv (size st x st)
        for (size_t i = 0; i < st; i++) {
            for (size_t j = 0; j < st; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < st; k++) {
                    sum += m.T_inv[i * st + k] * m.A_inv[k * st + j];
                }
                m_p_in[i * st + j] = sum;
            }
        }

        // Compute stage-averaged state y_avg = (1/st) * sum_l y_stages[l]
        const double inv_st = 1.0 / (double)st;
        memset(m_y_avg, 0, n_space * sizeof(double));
        for (size_t l = 0; l < st; l++) {
            const double *yl = &y_stages[l * n_space];
            for (size_t i = 0; i < n_space; i++) {
                m_y_avg[i] += inv_st * yl[i];
            }
        }

        m_real_bc.makePeriodic((ptrdiff_t)slice->node, 0);
        m_complex_bc.makePeriodic((ptrdiff_t)slice->node, 0);

        int pmask_dummy[2] = { 0, 0 };
        m_avg_jac.update(slice->fjac, slice->pjac, (ptrdiff_t)slice->nnodes, (ptrdiff_t)slice->node, (ptrdiff_t)slice->nstages, 0, slice->unscaled_spatial_period, slice->h, m_y_avg, p, pmask_dummy, nullptr, &m_real_bc);

        if (build_stage_jacs) {
            for (size_t l = 0; l < st; l++) {
                m_stage_jacs[l].update(slice->fjac, slice->pjac, (ptrdiff_t)slice->nnodes, (ptrdiff_t)slice->node, (ptrdiff_t)slice->nstages, 0, slice->unscaled_spatial_period, slice->h, &y_stages[l * n_space], p, pmask_dummy, nullptr, &m_real_bc);
            }
        }

        for (size_t r = 0; r < m.n_real; r++) {
            m_real_mats[r].copyPartitionsFrom(m_avg_jac, &m_real_bc);
            m_real_mats[r].addMassBlock(slice->M, slice->c_A, slice->c_b, slice->h, -m.real_eigs[r] / dtau_T);
            m_real_mats[r].factor();
        }

        for (size_t r = 0; r < m.n_complex; r++) {
            m_complex_mats[r].copyPartitionsFrom(m_avg_jac, &m_complex_bc);
            const complex_t scale(-m.complex_re[r] / dtau_T, +m.complex_im[r] / dtau_T);
            m_complex_mats[r].addMassBlock(slice->M, slice->c_A, slice->c_b, slice->h, scale);
            m_complex_mats[r].factor();
        }
    }

    void SubintervalCondenser::applyPrecondOnly(const double *rhs_stages, double *dk_stages) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;

        double *b_hat = m_work;
        double *z_hat = b_hat + n * st;
        complex_t *b_c = m_cwork;
        complex_t *z_c = b_c + n;

        // b_hat = (-1 / dtau_T) * rhs_stages * (T_s^(-1) * A_inv)^T
        const double inv_neg_dt = -1.0 / m_dtau_T;
        for (size_t k = 0; k < st; k++) {
            double *bk = &b_hat[k * n];
            memset(bk, 0, n * sizeof(double));
            for (size_t l = 0; l < st; l++) {
                const double fac = inv_neg_dt * m_p_in[k * st + l];
                const double *rl = &rhs_stages[l * n];
                for (size_t i = 0; i < n; i++) {
                    bk[i] += fac * rl[i];
                }
            }
        }

        size_t col = 0;
        for (size_t r = 0; r < m.n_real; r++) {
            m_real_mats[r].solve(&b_hat[col * n], &z_hat[col * n]);
            col++;
        }

        for (size_t r = 0; r < m.n_complex; r++) {
            const double *br = &b_hat[col * n];
            const double *bi = &b_hat[(col + 1) * n];
            for (size_t i = 0; i < n; i++) {
                b_c[i] = complex_t(br[i], bi[i]);
            }
            m_complex_mats[r].solve(b_c, z_c);
            double *zr = &z_hat[col * n];
            double *zi = &z_hat[(col + 1) * n];
            for (size_t i = 0; i < n; i++) {
                zr[i] = z_c[i].real();
                zi[i] = z_c[i].imag();
            }
            col += 2;
        }

        // dk_stages = z_hat * T_s^T
        for (size_t j = 0; j < st; j++) {
            double *dj = &dk_stages[j * n];
            memset(dj, 0, n * sizeof(double));
            for (size_t k = 0; k < st; k++) {
                const double fac = m.T[j * st + k];
                const double *zk = &z_hat[k * n];
                for (size_t i = 0; i < n; i++) {
                    dj[i] += fac * zk[i];
                }
            }
        }
    }

    void SubintervalCondenser::applyStageOperator(const double *dy_stages, double *out_stages) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;

        // Place ay in a disjoint region from m_work[0..2*n*st] (used by applyPrecondOnly)
        double *ay = m_work + 2 * n * st;
        double *mass_work = ay + n * st;

        for (size_t l = 0; l < st; l++) {
            m_slice->applyMass(&dy_stages[l * n], &out_stages[l * n], 1.0, 0.0, mass_work);
            if (m_has_stage_jacs) {
                m_stage_jacs[l].gemv(0.0, &ay[l * n], 1.0, &dy_stages[l * n]);
            }
            else {
                m_avg_jac.gemv(0.0, &ay[l * n], 1.0, &dy_stages[l * n]);
            }
        }

        // out_stages -= dtau_T * ay * (A^R)^T
        for (size_t j = 0; j < st; j++) {
            double *oj = &out_stages[j * n];
            for (size_t l = 0; l < st; l++) {
                const double fac = -m_dtau_T * m.A[j * st + l];
                const double *al = &ay[l * n];
                for (size_t i = 0; i < n; i++) {
                    oj[i] += fac * al[i];
                }
            }
        }
    }

    void SubintervalCondenser::solveLocalStages(const double *rhs_stages, double *dk_stages, bool exact_stage_jac) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t total = n * st;

        // First apply stage-averaged Hairer-Wanner solver
        // If rhs_stages aliases dk_stages, copy rhs_stages first if we need Richardson sweeps
        double *rhs_copy = nullptr;
        if (exact_stage_jac && m_has_stage_jacs) {
            rhs_copy = m_work + 4 * total;
            memcpy(rhs_copy, rhs_stages, total * sizeof(double));
        }

        applyPrecondOnly(rhs_stages, dk_stages);

        if (!exact_stage_jac || !m_has_stage_jacs) {
            return;
        }

        const double rhs_norm = norm2(rhs_copy, total);
        if (rhs_norm == 0.0) {
            return;
        }

        double *res = m_work + 5 * total;
        double *corr = m_work + 6 * total;
        double prev_res_norm = rhs_norm;
        bool richardson_ok = false;

        for (int sweep = 0; sweep < 4; sweep++) {
            applyStageOperator(dk_stages, res);
            for (size_t i = 0; i < total; i++) {
                res[i] = rhs_copy[i] - res[i];
            }
            const double res_norm = norm2(res, total);
            if (res_norm <= 1e-12 * rhs_norm) {
                richardson_ok = true;
                break;
            }
            if (res_norm > 0.5 * prev_res_norm) {
                break;
            }
            prev_res_norm = res_norm;
            applyPrecondOnly(res, corr);
            for (size_t i = 0; i < total; i++) {
                dk_stages[i] += corr[i];
            }
        }

        if (!richardson_ok) {
            // Check residual after last sweep before resorting to local GMRES
            applyStageOperator(dk_stages, res);
            for (size_t i = 0; i < total; i++) {
                res[i] = rhs_copy[i] - res[i];
            }
            const double res_norm = norm2(res, total);
            if (res_norm > 1e-10 * rhs_norm) {
                const size_t restart = 16;
                const size_t lwork = Krylov::gmres_lwork(total, restart);
                if (m_gmres_work_cap < lwork) {
                    m_gmres_work = (double *)realloc(m_gmres_work, lwork * sizeof(double));
                    m_gmres_work_cap = lwork;
                }
                auto A_loc = [&](const RP(double) x, RP(double) y, size_t) -> int {
                    applyStageOperator(x, y);
                    return 0;
                };
                auto M_loc = [&](const RP(double) r, RP(double) z, size_t) -> int {
                    applyPrecondOnly(r, z);
                    return 0;
                };
                int local_nit = 0;
                Krylov::gmres(A_loc, M_loc, total, dk_stages, rhs_copy, nullptr, m_gmres_work, (int)restart, 32, 1e-12, 1e-13 * rhs_norm, &local_nit, false);
            }
        }
    }

    void SubintervalCondenser::applyElementPropagator(const RP(double) w_in, RP(double) w_out, bool exact_stage_jac) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t total = n * st;

        double *rhs_stages = m_work + 7 * total;
        double *dk_stages = m_work + 8 * total;
        double *bw = m_work + 9 * total;
        double *mass_work = bw + n;

        m_slice->applyMass(w_in, bw, 1.0, 0.0, mass_work);
        for (size_t j = 0; j < st; j++) {
            memcpy(&rhs_stages[j * n], bw, n * sizeof(double));
        }

        solveLocalStages(rhs_stages, dk_stages, exact_stage_jac);
        memcpy(w_out, &dk_stages[(st - 1) * n], n * sizeof(double));
    }

    void SubintervalCondenser::applyCondensedC(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) w_out, bool exact_stage_jac) const
    {
        if (transfer_prev_to_cur.is_identity) {
            applyElementPropagator(v_prev, w_out, exact_stage_jac);
            return;
        }
        double *v_cur = m_y_avg;
        transfer_prev_to_cur.apply(v_prev, v_cur);
        applyElementPropagator(v_cur, w_out, exact_stage_jac);
    }

    void SubintervalCondenser::applyPrecondOnlyBatch(size_t K, const double *rhs_stages, double *dk_stages, double *batch_work, complex_t *cbatch_work) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t block_len = n * K;

        double *b_hat = batch_work;
        double *z_hat = b_hat + st * block_len;
        complex_t *b_c = cbatch_work;
        complex_t *z_c = b_c + block_len;

        const double inv_neg_dt = -1.0 / m_dtau_T;
        for (size_t k = 0; k < st; k++) {
            double *bk = &b_hat[k * block_len];
            memset(bk, 0, block_len * sizeof(double));
            for (size_t l = 0; l < st; l++) {
                const double fac = inv_neg_dt * m_p_in[k * st + l];
                const double *rl = &rhs_stages[l * block_len];
                for (size_t p = 0; p < block_len; p++) {
                    bk[p] += fac * rl[p];
                }
            }
        }

        size_t col = 0;
        for (size_t r = 0; r < m.n_real; r++) {
            if (K == 1) {
                m_real_mats[r].solve(&b_hat[col * block_len], &z_hat[col * block_len]);
            }
            else {
                m_real_mats[r].solve(&b_hat[col * block_len], &z_hat[col * block_len], (ptrdiff_t)K);
            }
            col++;
        }

        for (size_t r = 0; r < m.n_complex; r++) {
            const double *br = &b_hat[col * block_len];
            const double *bi = &b_hat[(col + 1) * block_len];
            for (size_t p = 0; p < block_len; p++) {
                b_c[p] = complex_t(br[p], bi[p]);
            }
            if (K == 1) {
                m_complex_mats[r].solve(b_c, z_c);
            }
            else {
                m_complex_mats[r].solve(b_c, z_c, (ptrdiff_t)K);
            }
            double *zr = &z_hat[col * block_len];
            double *zi = &z_hat[(col + 1) * block_len];
            for (size_t p = 0; p < block_len; p++) {
                zr[p] = z_c[p].real();
                zi[p] = z_c[p].imag();
            }
            col += 2;
        }

        for (size_t j = 0; j < st; j++) {
            double *dj = &dk_stages[j * block_len];
            memset(dj, 0, block_len * sizeof(double));
            for (size_t k = 0; k < st; k++) {
                const double fac = m.T[j * st + k];
                const double *zk = &z_hat[k * block_len];
                for (size_t p = 0; p < block_len; p++) {
                    dj[p] += fac * zk[p];
                }
            }
        }
    }

    void SubintervalCondenser::applyStageOperatorBatch(size_t K, const double *dy_stages, double *out_stages, double *ay_buf, double *mass_work) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t block_len = n * K;

        for (size_t l = 0; l < st; l++) {
            const double *dy_l = &dy_stages[l * block_len];
            double *out_l = &out_stages[l * block_len];
            for (size_t j = 0; j < K; j++) {
                m_slice->applyMass(&dy_l[j * n], &out_l[j * n], 1.0, 0.0, mass_work);
            }
            const DEV::Matrix &jac = m_has_stage_jacs ? m_stage_jacs[l] : m_avg_jac;
            if (K == 1) {
                jac.gemv(0.0, &ay_buf[l * block_len], 1.0, dy_l);
            }
            else {
                jac.gemm((ptrdiff_t)K, 0.0, &ay_buf[l * block_len], 1.0, dy_l);
            }
        }

        for (size_t j = 0; j < st; j++) {
            double *oj = &out_stages[j * block_len];
            for (size_t l = 0; l < st; l++) {
                const double fac = -m_dtau_T * m.A[j * st + l];
                const double *al = &ay_buf[l * block_len];
                for (size_t p = 0; p < block_len; p++) {
                    oj[p] += fac * al[p];
                }
            }
        }
    }

    void SubintervalCondenser::computeCondensedMatrix(const SparseDenseOutputOp &transfer_prev_to_cur, double *C_out) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t K = transfer_prev_to_cur.is_identity ? n : (transfer_prev_to_cur.src_npts * transfer_prev_to_cur.node);
        const size_t chunk_max = 64;

        std::vector<double> e_vec(K, 0.0);
        std::vector<double> v_cur(n);
        std::vector<double> rhs_chunk(st * n * chunk_max);
        std::vector<double> dk_chunk(st * n * chunk_max);
        std::vector<double> res_chunk(st * n * chunk_max);
        std::vector<double> corr_chunk(st * n * chunk_max);
        std::vector<double> ay_chunk(st * n * chunk_max);
        std::vector<double> pc_work(2 * st * n * chunk_max);
        std::vector<complex_t> pc_cwork(2 * n * chunk_max);
        std::vector<double> rhs_1col(st * n), dk_1col(st * n);
        std::vector<double> col_rhs_norm(chunk_max), col_prev_res(chunk_max);
        std::vector<unsigned char> col_active(chunk_max);
        std::vector<double> mass_work_buf(m_slice->node * m_slice->nstages + 64);
        double *mass_work = mass_work_buf.data();

        for (size_t c0 = 0; c0 < K; c0 += chunk_max) {
            const size_t ck = std::min(chunk_max, K - c0);
            const size_t block_len = n * ck;

            // Assemble rhs_chunk for columns [c0, c0 + ck)
            for (size_t j = 0; j < ck; j++) {
                if (transfer_prev_to_cur.is_identity) {
                    memset(v_cur.data(), 0, n * sizeof(double));
                    v_cur[c0 + j] = 1.0;
                }
                else {
                    e_vec[c0 + j] = 1.0;
                    transfer_prev_to_cur.apply(e_vec.data(), v_cur.data());
                    e_vec[c0 + j] = 0.0;
                }
                m_slice->applyMass(v_cur.data(), &rhs_chunk[0 * block_len + j * n], 1.0, 0.0, mass_work);
            }
            for (size_t l = 1; l < st; l++) {
                memcpy(&rhs_chunk[l * block_len], &rhs_chunk[0], block_len * sizeof(double));
            }

            applyPrecondOnlyBatch(ck, rhs_chunk.data(), dk_chunk.data(), pc_work.data(), pc_cwork.data());

            if (m_has_stage_jacs) {
                for (size_t j = 0; j < ck; j++) {
                    double sq = 0.0;
                    for (size_t l = 0; l < st; l++) {
                        const double *col = &rhs_chunk[l * block_len + j * n];
                        for (size_t i = 0; i < n; i++) {
                            sq += col[i] * col[i];
                        }
                    }
                    col_rhs_norm[j] = std::sqrt(sq);
                    col_prev_res[j] = col_rhs_norm[j];
                    col_active[j] = (col_rhs_norm[j] > 0.0) ? 1 : 0;
                }

                for (int sweep = 0; sweep < 4; sweep++) {
                    applyStageOperatorBatch(ck, dk_chunk.data(), res_chunk.data(), ay_chunk.data(), mass_work);
                    for (size_t p = 0; p < st * block_len; p++) {
                        res_chunk[p] = rhs_chunk[p] - res_chunk[p];
                    }

                    bool any_active = false;
                    for (size_t j = 0; j < ck; j++) {
                        if (!col_active[j]) continue;
                        double sq = 0.0;
                        for (size_t l = 0; l < st; l++) {
                            const double *col = &res_chunk[l * block_len + j * n];
                            for (size_t i = 0; i < n; i++) {
                                sq += col[i] * col[i];
                            }
                        }
                        const double rn = std::sqrt(sq);
                        if (rn <= 1e-12 * col_rhs_norm[j] || rn > 0.5 * col_prev_res[j]) {
                            col_active[j] = 0;
                        }
                        else {
                            col_prev_res[j] = rn;
                            any_active = true;
                        }
                    }
                    if (!any_active) {
                        break;
                    }

                    applyPrecondOnlyBatch(ck, res_chunk.data(), corr_chunk.data(), pc_work.data(), pc_cwork.data());
                    for (size_t j = 0; j < ck; j++) {
                        if (!col_active[j]) continue;
                        for (size_t l = 0; l < st; l++) {
                            double *dcol = &dk_chunk[l * block_len + j * n];
                            const double *ccol = &corr_chunk[l * block_len + j * n];
                            for (size_t i = 0; i < n; i++) {
                                dcol[i] += ccol[i];
                            }
                        }
                    }
                }

                // Final residual check per column; polish only unconverged columns with local GMRES
                applyStageOperatorBatch(ck, dk_chunk.data(), res_chunk.data(), ay_chunk.data(), mass_work);
                for (size_t j = 0; j < ck; j++) {
                    if (col_rhs_norm[j] == 0.0) continue;
                    double sq = 0.0;
                    for (size_t l = 0; l < st; l++) {
                        const double *r_src = &rhs_chunk[l * block_len + j * n];
                        const double *a_dk = &res_chunk[l * block_len + j * n];
                        for (size_t i = 0; i < n; i++) {
                            const double d = r_src[i] - a_dk[i];
                            sq += d * d;
                        }
                    }
                    if (std::sqrt(sq) > 1e-10 * col_rhs_norm[j]) {
                        for (size_t l = 0; l < st; l++) {
                            memcpy(&rhs_1col[l * n], &rhs_chunk[l * block_len + j * n], n * sizeof(double));
                            memcpy(&dk_1col[l * n], &dk_chunk[l * block_len + j * n], n * sizeof(double));
                        }
                        const size_t total = st * n;
                        const size_t restart = 16;
                        const size_t lwork = Krylov::gmres_lwork(total, restart);
                        if (m_gmres_work_cap < lwork) {
                            m_gmres_work = (double *)realloc(m_gmres_work, lwork * sizeof(double));
                            m_gmres_work_cap = lwork;
                        }
                        auto A_loc = [&](const RP(double) x, RP(double) y, size_t) -> int {
                            applyStageOperator(x, y);
                            return 0;
                        };
                        auto M_loc = [&](const RP(double) r, RP(double) z, size_t) -> int {
                            applyPrecondOnly(r, z);
                            return 0;
                        };
                        int local_nit = 0;
                        Krylov::gmres(A_loc, M_loc, total, dk_1col.data(), rhs_1col.data(), nullptr, m_gmres_work, (int)restart, 32, 1e-12, 1e-13 * col_rhs_norm[j], &local_nit, false);
                        memcpy(&dk_chunk[(st - 1) * block_len + j * n], &dk_1col[(st - 1) * n], n * sizeof(double));
                    }
                }
            }

            memcpy(&C_out[c0 * n], &dk_chunk[(st - 1) * block_len], block_len * sizeof(double));
        }
    }

    void SubintervalCondenser::evalSpatialParamDeriv(size_t /*stage_l*/, const double *y_stage, const double *p, int param_index, RP(double) out) const
    {
        const size_t n_sub = m_slice->nSub();
        const size_t m = m_slice->node;
        const size_t s = m_slice->nstages;
        const double L0 = m_slice->unscaled_spatial_period;
        const double *A_gl = m_slice->c_A;
        const double *b_gl = m_slice->c_b;
        const int pmask_single[1] = { param_index };

        memset(out, 0, m_n_space * sizeof(double));
        for (size_t k = 0; k < n_sub; k++) {
            const size_t stride_k = m * (s + 1) * k;
            const double hk = m_slice->h[k];
            const double Lh = L0 * hk;

            for (size_t i = 0; i < s; i++) {
                m_slice->pjac(&y_stage[stride_k + (i + 1) * m], p, pmask_single, 1, &m_pjac_work[i * m]);
            }

            // Top row F_k
            double *Fk = &out[stride_k];
            for (size_t i = 0; i < s; i++) {
                const double fac = Lh * b_gl[i];
                const double *df = &m_pjac_work[i * m];
                for (size_t l = 0; l < m; l++) {
                    Fk[l] += fac * df[l];
                }
            }

            // Stage rows H_{k,j}
            for (size_t j = 0; j < s; j++) {
                double *Hkj = &out[stride_k + (j + 1) * m];
                for (size_t i = 0; i < s; i++) {
                    const double fac = Lh * A_gl[j * s + i];
                    const double *df = &m_pjac_work[i * m];
                    for (size_t l = 0; l < m; l++) {
                        Hkj[l] += fac * df[l];
                    }
                }
            }
        }
    }
} // namespace TWIST
