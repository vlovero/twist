#include "modulating/subinterval_condenser.h"
#include "krylov/gmres.h"
#include "linalg.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace TWIST
{
    int subinterval_local_gmres_subspace_dim = read_positive_int_env("TWIST_HOPF_LOCAL_GMRES_DIM", 16);

    SubintervalCondenser::~SubintervalCondenser()
    {
        freeBuffers();
    }

    void SubintervalCondenser::freeBuffers()
    {
        if (m_avg_fjac) {
            free(m_avg_fjac);
            m_avg_fjac = nullptr;
            m_avg_fjac_cap = 0;
        }
        if (m_delta_fjac) {
            free(m_delta_fjac);
            m_delta_fjac = nullptr;
            m_delta_fjac_cap = 0;
        }
        if (m_y_avg) {
            free(m_y_avg);
            m_y_avg = nullptr;
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
        if (m_cmat_buf) {
            free(m_cmat_buf);
            m_cmat_buf = nullptr;
            m_cmat_cap = 0;
        }
        if (m_cmat_cbuf) {
            free(m_cmat_cbuf);
            m_cmat_cbuf = nullptr;
            m_cmat_ccap = 0;
        }
        if (m_cmat_cols) {
            free(m_cmat_cols);
            m_cmat_cols = nullptr;
            m_cmat_cols_cap = 0;
        }
        m_bc_node = 0;
    }

    void SubintervalCondenser::ensureWorkspace(size_t n_space, size_t st)
    {
        if (m_n_space != n_space || m_st != st) {
            m_y_avg = (double *)realloc(m_y_avg, n_space * sizeof(double));
        }

        const size_t n_sub = m_slice ? m_slice->nSub() : 1;
        const size_t ncol = m_slice ? m_slice->nstages : 1;
        const size_t node = m_slice ? m_slice->node : 1;
        const size_t avg_fjac_sz = n_sub * ncol * node * node;
        if (m_avg_fjac_cap < avg_fjac_sz) {
            m_avg_fjac = (double *)realloc(m_avg_fjac, avg_fjac_sz * sizeof(double));
            m_avg_fjac_cap = avg_fjac_sz;
        }
        const size_t delta_fjac_sz = st * avg_fjac_sz;
        if (m_delta_fjac_cap < delta_fjac_sz) {
            m_delta_fjac = (double *)realloc(m_delta_fjac, delta_fjac_sz * sizeof(double));
            m_delta_fjac_cap = delta_fjac_sz;
        }

        // 11 * n_space * st + 256 (0..9: b_hat, z_hat, ay, w_buf, rhs_copy, res, corr, rhs_stages, dk_stages, bw; 10: delta_out)
        const size_t needed_real = 11 * n_space * st + 256;
        if (m_work_cap < needed_real) {
            m_work = (double *)realloc(m_work, needed_real * sizeof(double));
            m_work_cap = needed_real;
        }

        const size_t needed_complex = 2 * n_space;
        if (m_cwork_cap < needed_complex) {
            m_cwork = (complex_t *)realloc(m_cwork, needed_complex * sizeof(complex_t));
            m_cwork_cap = needed_complex;
        }

        const size_t needed_pjac = ncol * node;
        if (m_pjac_work_cap < needed_pjac) {
            m_pjac_work = (double *)realloc(m_pjac_work, needed_pjac * sizeof(double));
            m_pjac_work_cap = needed_pjac;
        }
    }

    void SubintervalCondenser::ensureBatchWorkspace(size_t K_batch, size_t n_space, size_t st, size_t extra_doubles) const
    {
        const size_t block_len = n_space * K_batch;
        const size_t needed_d = 6 * st * block_len + extra_doubles + 256;
        if (m_cmat_cap < needed_d) {
            m_cmat_buf = (double *)realloc(m_cmat_buf, needed_d * sizeof(double));
            m_cmat_cap = needed_d;
        }
        const size_t needed_c = 2 * block_len;
        if (m_cmat_ccap < needed_c) {
            m_cmat_cbuf = (complex_t *)realloc(m_cmat_cbuf, needed_c * sizeof(complex_t));
            m_cmat_ccap = needed_c;
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
        m_max_delta_fjac = 0.0;

        const RadauDGMethod &m = getRadauDG(st);
        const size_t n_sub = slice->nSub();
        const size_t ncol = slice->nstages;
        const size_t node = slice->node;
        const size_t node2 = node * node;
        const size_t fjac_stride = n_sub * ncol * node2;

        // Compute stage-averaged state y_avg = (1/st) * sum_l y_stages[l]
        const double inv_st = 1.0 / (double)st;
        if (st == 3) {
            const double *y0 = &y_stages[0 * n_space];
            const double *y1 = &y_stages[1 * n_space];
            const double *y2 = &y_stages[2 * n_space];
            for (size_t i = 0; i < n_space; i++) {
                m_y_avg[i] = inv_st * (y0[i] + y1[i] + y2[i]);
            }
        }
        else {
            memset(m_y_avg, 0, n_space * sizeof(double));
            for (size_t l = 0; l < st; l++) {
                const double *yl = &y_stages[l * n_space];
                for (size_t i = 0; i < n_space; i++) {
                    m_y_avg[i] += inv_st * yl[i];
                }
            }
        }

        if (m_bc_node != node) {
            m_real_bc.makePeriodic((ptrdiff_t)node, 0);
            m_complex_bc.makePeriodic((ptrdiff_t)node, 0);
            m_bc_node = node;
        }

        // Evaluate fjac at y_avg once into m_avg_fjac, then assemble m_avg_jac directly without re-calling fjac or latrn
        for (size_t k = 0; k < n_sub; k++) {
            const double *yk = &m_y_avg[k * node * (ncol + 1) + node];
            for (size_t i = 0; i < ncol; i++) {
                slice->fjac(&yk[i * node], p, &m_avg_fjac[(k * ncol + i) * node2]);
            }
        }
        m_avg_jac.updateFromFjacs(m_avg_fjac, (ptrdiff_t)slice->nnodes, (ptrdiff_t)node, (ptrdiff_t)ncol, slice->unscaled_spatial_period, slice->h, &m_real_bc, true);

        if (build_stage_jacs) {
            double max_dj = 0.0;
            for (size_t l = 0; l < st; l++) {
                const double *yl = &y_stages[l * n_space];
                double *dl_fjac = &m_delta_fjac[l * fjac_stride];
                for (size_t k = 0; k < n_sub; k++) {
                    const double *ylk = &yl[k * node * (ncol + 1) + node];
                    for (size_t i = 0; i < ncol; i++) {
                        const size_t off = (k * ncol + i) * node2;
                        double *dst = &dl_fjac[off];
                        const double *avg = &m_avg_fjac[off];
                        slice->fjac(&ylk[i * node], p, dst);
                        for (size_t q = 0; q < node2; q++) {
                            const double d = dst[q] - avg[q];
                            dst[q] = d;
                            const double ad = std::abs(d);
                            if (ad > max_dj) {
                                max_dj = ad;
                            }
                        }
                    }
                }
            }
            m_max_delta_fjac = max_dj;
        }

        for (size_t r = 0; r < m.n_real; r++) {
            m_real_mats[r].copyPartitionsFrom(m_avg_jac, &m_real_bc, false);
            m_real_mats[r].addMassBlockSparse(slice->m_nnz, slice->m_rows, slice->m_cols, slice->m_vals, slice->c_A, slice->c_b, slice->h, -m.real_eigs[r] / dtau_T, false);
            m_real_mats[r].factor();
        }

        for (size_t r = 0; r < m.n_complex; r++) {
            m_complex_mats[r].copyPartitionsFrom(m_avg_jac, &m_complex_bc, false);
            const complex_t scale(-m.complex_re[r] / dtau_T, +m.complex_im[r] / dtau_T);
            m_complex_mats[r].addMassBlockSparse(slice->m_nnz, slice->m_rows, slice->m_cols, slice->m_vals, slice->c_A, slice->c_b, slice->h, scale, false);
            m_complex_mats[r].factor();
        }
    }

    void SubintervalCondenser::applyPrecondOnlyFromBhat(const double *b_hat, double *dk_stages) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;

        double *z_hat = m_work + n * st;
        complex_t *b_c = m_cwork;
        complex_t *z_c = b_c + n;

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
            const double t0 = m.T[j * st + 0];
            const double *z0 = &z_hat[0 * n];
            for (size_t i = 0; i < n; i++) {
                dj[i] = t0 * z0[i];
            }
            for (size_t k = 1; k < st; k++) {
                const double fac = m.T[j * st + k];
                const double *zk = &z_hat[k * n];
                for (size_t i = 0; i < n; i++) {
                    dj[i] += fac * zk[i];
                }
            }
        }
    }

    void SubintervalCondenser::applyPrecondOnly(const double *rhs_stages, double *dk_stages) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        double *b_hat = m_work;

        // b_hat = (-1 / dtau_T) * rhs_stages * (T_s^(-1) * A_inv)^T
        const double inv_neg_dt = -1.0 / m_dtau_T;
        for (size_t k = 0; k < st; k++) {
            double *bk = &b_hat[k * n];
            const double f0 = inv_neg_dt * m.P_in[k * st + 0];
            const double *r0 = &rhs_stages[0 * n];
            for (size_t i = 0; i < n; i++) {
                bk[i] = f0 * r0[i];
            }
            for (size_t l = 1; l < st; l++) {
                const double fac = inv_neg_dt * m.P_in[k * st + l];
                const double *rl = &rhs_stages[l * n];
                for (size_t i = 0; i < n; i++) {
                    bk[i] += fac * rl[i];
                }
            }
        }
        applyPrecondOnlyFromBhat(b_hat, dk_stages);
    }

    void SubintervalCondenser::applyDeltaStageOp(const double *in_stages, double *out_stages) const
    {
        const RadauDGMethod &rm = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t n_sub = m_slice->nSub();
        const size_t ncol = m_slice->nstages;
        const size_t m = m_slice->node;
        const size_t m2 = m * m;
        const size_t n_no_bc = n_sub * (ncol + 1) * m;
        const size_t fjac_stride = n_sub * ncol * m2;
        const double L0 = m_slice->unscaled_spatial_period;
        const double *c_A = m_slice->c_A;
        const double *c_b = m_slice->c_b;
        const double *h = m_slice->h;

        double *w_buf = m_work + 3 * n * st;
        double tmp[m];

        for (size_t l = 0; l < st; l++) {
            const double *in_l = &in_stages[l * n];
            double *w_l = &w_buf[l * n_no_bc];
            const double *dl_fjac = &m_delta_fjac[l * fjac_stride];

            for (size_t k = 0; k < n_sub; k++) {
                const size_t stride_k = k * (ncol + 1) * m;
                const double Lh = L0 * h[k];
                double *wk = &w_l[stride_k];
                memset(wk, 0, (ncol + 1) * m * sizeof(double));

                for (size_t i = 0; i < ncol; i++) {
                    const double *dJ = &dl_fjac[(k * ncol + i) * m2];
                    const double *xi = &in_l[stride_k + (i + 1) * m];

                    for (size_t r = 0; r < m; r++) {
                        const double *row_r = &dJ[r * m];
                        double dot = 0.0;
                        for (size_t c = 0; c < m; c++) {
                            dot += row_r[c] * xi[c];
                        }
                        tmp[r] = Lh * dot;
                    }

                    const double bi = c_b[i];
                    for (size_t r = 0; r < m; r++) {
                        wk[r] += bi * tmp[r];
                    }
                    for (size_t j = 0; j < ncol; j++) {
                        const double aji = c_A[j * ncol + i];
                        double *wkj = &wk[(j + 1) * m];
                        for (size_t r = 0; r < m; r++) {
                            wkj[r] += aji * tmp[r];
                        }
                    }
                }
            }
        }

        // out_stages[j] = dtau_T * sum_l A^R[j, l] * w_l (BC rows = 0)
        for (size_t j = 0; j < st; j++) {
            double *oj = &out_stages[j * n];
            const double f0 = m_dtau_T * rm.A[j * st + 0];
            const double *w0 = &w_buf[0 * n_no_bc];
            for (size_t q = 0; q < n_no_bc; q++) {
                oj[q] = f0 * w0[q];
            }
            for (size_t l = 1; l < st; l++) {
                const double fac = m_dtau_T * rm.A[j * st + l];
                const double *wl = &w_buf[l * n_no_bc];
                for (size_t q = 0; q < n_no_bc; q++) {
                    oj[q] += fac * wl[q];
                }
            }
            memset(&oj[n_no_bc], 0, m * sizeof(double));
        }
    }

    void SubintervalCondenser::applyDeltaStageOpBatch(size_t K, const double *in_stages, double *out_stages, double *w_buf) const
    {
        const RadauDGMethod &rm = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t block_len = n * K;
        const size_t n_sub = m_slice->nSub();
        const size_t ncol = m_slice->nstages;
        const size_t m = m_slice->node;
        const size_t m2 = m * m;
        const size_t n_no_bc = n_sub * (ncol + 1) * m;
        const size_t fjac_stride = n_sub * ncol * m2;
        const double L0 = m_slice->unscaled_spatial_period;
        const double *c_A = m_slice->c_A;
        const double *c_b = m_slice->c_b;
        const double *h = m_slice->h;

        double tmp[m];
        for (size_t l = 0; l < st; l++) {
            const double *in_l = &in_stages[l * block_len];
            double *w_l = &w_buf[l * n_no_bc * K];
            const double *dl_fjac = &m_delta_fjac[l * fjac_stride];

            for (size_t c_idx = 0; c_idx < K; c_idx++) {
                const double *in_lc = &in_l[c_idx * n];
                double *w_lc = &w_l[c_idx * n_no_bc];

                for (size_t k = 0; k < n_sub; k++) {
                    const size_t stride_k = k * (ncol + 1) * m;
                    const double Lh = L0 * h[k];
                    double *wk = &w_lc[stride_k];
                    memset(wk, 0, (ncol + 1) * m * sizeof(double));

                    for (size_t i = 0; i < ncol; i++) {
                        const double *dJ = &dl_fjac[(k * ncol + i) * m2];
                        const double *xi = &in_lc[stride_k + (i + 1) * m];

                        for (size_t r = 0; r < m; r++) {
                            const double *row_r = &dJ[r * m];
                            double dot = 0.0;
                            for (size_t cc = 0; cc < m; cc++) {
                                dot += row_r[cc] * xi[cc];
                            }
                            tmp[r] = Lh * dot;
                        }

                        const double bi = c_b[i];
                        for (size_t r = 0; r < m; r++) {
                            wk[r] += bi * tmp[r];
                        }
                        for (size_t j = 0; j < ncol; j++) {
                            const double aji = c_A[j * ncol + i];
                            double *wkj = &wk[(j + 1) * m];
                            for (size_t r = 0; r < m; r++) {
                                wkj[r] += aji * tmp[r];
                            }
                        }
                    }
                }
            }
        }

        for (size_t j = 0; j < st; j++) {
            double *oj = &out_stages[j * block_len];
            for (size_t c_idx = 0; c_idx < K; c_idx++) {
                double *ojc = &oj[c_idx * n];
                const double f0 = m_dtau_T * rm.A[j * st + 0];
                const double *w0 = &w_buf[(0 * K + c_idx) * n_no_bc];
                for (size_t q = 0; q < n_no_bc; q++) {
                    ojc[q] = f0 * w0[q];
                }
                for (size_t l = 1; l < st; l++) {
                    const double fac = m_dtau_T * rm.A[j * st + l];
                    const double *wl = &w_buf[(l * K + c_idx) * n_no_bc];
                    for (size_t q = 0; q < n_no_bc; q++) {
                        ojc[q] += fac * wl[q];
                    }
                }
                memset(&ojc[n_no_bc], 0, m * sizeof(double));
            }
        }
    }

    void SubintervalCondenser::applyStageOperator(const double *dy_stages, double *out_stages) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;

        double *ay = m_work + 2 * n * st;

        for (size_t l = 0; l < st; l++) {
            m_slice->applyMass(&dy_stages[l * n], &out_stages[l * n], 1.0, 0.0, nullptr);
            m_avg_jac.gemv(0.0, &ay[l * n], 1.0, &dy_stages[l * n]);
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

        if (m_has_stage_jacs && m_max_delta_fjac > 1e-14) {
            double *delta_out = m_work + 10 * n * st;
            applyDeltaStageOp(dy_stages, delta_out);
            for (size_t q = 0; q < st * n; q++) {
                out_stages[q] -= delta_out[q];
            }
        }
    }

    void SubintervalCondenser::solveLocalStages(const double *rhs_stages, double *dk_stages, bool exact_stage_jac, double local_rtol) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t total = n * st;

        const bool need_refine = exact_stage_jac && m_has_stage_jacs && (m_max_delta_fjac > 1e-14);
        double *rhs_copy = nullptr;
        double rhs_norm = 0.0;
        if (need_refine) {
            rhs_copy = m_work + 4 * total;
            memcpy(rhs_copy, rhs_stages, total * sizeof(double));
            rhs_norm = norm2(rhs_copy, total);
        }

        applyPrecondOnly(rhs_stages, dk_stages);

        if (!need_refine || rhs_norm == 0.0) {
            return;
        }

        double *res = m_work + 5 * total;
        double *corr = m_work + 6 * total;

        // First residual: r^(1) = b - A_n dk^(0) = (A_avg - A_n) dk^(0) = applyDeltaStageOp(dk^(0))
        applyDeltaStageOp(dk_stages, res);
        double res_norm = norm2(res, total);
        const double target_tol = local_rtol * rhs_norm;
        if (res_norm <= target_tol) {
            return;
        }

        double prev_res_norm = res_norm;
        bool richardson_ok = false;

        for (int sweep = 0; sweep < 4; sweep++) {
            m_stat_richardson_sweeps++;
            applyPrecondOnly(res, corr);
            for (size_t i = 0; i < total; i++) {
                dk_stages[i] += corr[i];
            }
            applyDeltaStageOp(corr, res);
            res_norm = norm2(res, total);
            if (res_norm <= target_tol) {
                richardson_ok = true;
                break;
            }
            if (res_norm > 0.5 * prev_res_norm) {
                break;
            }
            prev_res_norm = res_norm;
        }

        const double gmres_trigger = std::max(10.0 * local_rtol, 1e-10) * rhs_norm;
        if (!richardson_ok && res_norm > gmres_trigger) {
            runLocalGmres(dk_stages, rhs_copy, local_rtol, 0.1 * local_rtol * rhs_norm);
        }
    }

    void SubintervalCondenser::runLocalGmres(double *x, const double *b, double local_rtol, double atol) const
    {
        const size_t total = m_st * m_n_space;
        const size_t clamped_dim = std::min<size_t>((size_t)std::max(1, subinterval_local_gmres_subspace_dim), std::max<size_t>(1, total));
        const int restart = (clamped_dim > (size_t)std::numeric_limits<int>::max())
            ? std::numeric_limits<int>::max()
            : static_cast<int>(clamped_dim);
        const size_t lwork = Krylov::gmres_lwork(total, restart);
        if (m_gmres_work_cap < lwork) {
            m_gmres_work = (double *)realloc(m_gmres_work, lwork * sizeof(double));
            m_gmres_work_cap = lwork;
        }
        auto A_loc = [&](const RP(double) vx, RP(double) vy, size_t) -> int {
            applyStageOperator(vx, vy);
            return 0;
        };
        auto M_loc = [&](const RP(double) r, RP(double) z, size_t) -> int {
            applyPrecondOnly(r, z);
            return 0;
        };
        int local_nit = 0;
        const int maxiter = std::max(32, restart);
        m_stat_gmres_fallbacks++;
        Krylov::gmres(A_loc, M_loc, total, x, b, nullptr, m_gmres_work, restart, maxiter, local_rtol, atol, &local_nit, false);
    }

    void SubintervalCondenser::applyElementPropagatorAllStages(const RP(double) w_in, RP(double) dk_stages_out, bool exact_stage_jac, double local_rtol) const
    {
        const RadauDGMethod &rm = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t total = n * st;

        double *bw = m_work + 9 * total;
        m_slice->applyMassOpSparse(m_slice->m_nnz, m_slice->m_rows, m_slice->m_cols, m_slice->m_vals, w_in, bw, 1.0, 0.0);

        // Compute b_hat[k] = (-gamma_ones[k] / dtau_T) * bw directly without forming st copies of bw
        double *b_hat = m_work;
        const double inv_neg_dt = -1.0 / m_dtau_T;
        for (size_t k = 0; k < st; k++) {
            const double fac = inv_neg_dt * rm.gamma_ones[k];
            double *bk = &b_hat[k * n];
            for (size_t i = 0; i < n; i++) {
                bk[i] = fac * bw[i];
            }
        }
        applyPrecondOnlyFromBhat(b_hat, dk_stages_out);

        const bool need_refine = exact_stage_jac && m_has_stage_jacs && (m_max_delta_fjac > 1e-14);
        if (!need_refine) {
            return;
        }

        const double bw_norm = norm2(bw, n);
        if (bw_norm == 0.0) {
            return;
        }
        const double rhs_norm = std::sqrt((double)st) * bw_norm;
        const double target_tol = local_rtol * rhs_norm;

        double *res = m_work + 5 * total;
        double *corr = m_work + 6 * total;

        applyDeltaStageOp(dk_stages_out, res);
        double res_norm = norm2(res, total);
        if (res_norm <= target_tol) {
            return;
        }

        double prev_res_norm = res_norm;
        bool richardson_ok = false;

        for (int sweep = 0; sweep < 4; sweep++) {
            m_stat_richardson_sweeps++;
            applyPrecondOnly(res, corr);
            for (size_t i = 0; i < total; i++) {
                dk_stages_out[i] += corr[i];
            }
            applyDeltaStageOp(corr, res);
            res_norm = norm2(res, total);
            if (res_norm <= target_tol) {
                richardson_ok = true;
                break;
            }
            if (res_norm > 0.5 * prev_res_norm) {
                break;
            }
            prev_res_norm = res_norm;
        }

        const double gmres_trigger = std::max(10.0 * local_rtol, 1e-10) * rhs_norm;
        if (!richardson_ok && res_norm > gmres_trigger) {
            double *rhs_stages = m_work + 7 * total;
            for (size_t j = 0; j < st; j++) {
                memcpy(&rhs_stages[j * n], bw, n * sizeof(double));
            }
            runLocalGmres(dk_stages_out, rhs_stages, local_rtol, 0.1 * local_rtol * rhs_norm);
        }
    }

    void SubintervalCondenser::applyElementPropagator(const RP(double) w_in, RP(double) w_out, bool exact_stage_jac, double local_rtol) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        double *dk_stages = m_work + 8 * n * st;
        applyElementPropagatorAllStages(w_in, dk_stages, exact_stage_jac, local_rtol);
        memcpy(w_out, &dk_stages[(st - 1) * n], n * sizeof(double));
    }

    void SubintervalCondenser::applyCondensedC(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) w_out, bool exact_stage_jac, double local_rtol) const
    {
        if (transfer_prev_to_cur.is_identity) {
            applyElementPropagator(v_prev, w_out, exact_stage_jac, local_rtol);
            return;
        }
        double *v_cur = m_y_avg;
        transfer_prev_to_cur.apply(v_prev, v_cur);
        applyElementPropagator(v_cur, w_out, exact_stage_jac, local_rtol);
    }

    void SubintervalCondenser::applyCondensedCAllStages(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) dk_stages_out, bool exact_stage_jac, double local_rtol) const
    {
        if (transfer_prev_to_cur.is_identity) {
            applyElementPropagatorAllStages(v_prev, dk_stages_out, exact_stage_jac, local_rtol);
            return;
        }
        double *v_cur = m_y_avg;
        transfer_prev_to_cur.apply(v_prev, v_cur);
        applyElementPropagatorAllStages(v_cur, dk_stages_out, exact_stage_jac, local_rtol);
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
            const double f0 = inv_neg_dt * m.P_in[k * st + 0];
            const double *r0 = &rhs_stages[0 * block_len];
            for (size_t p = 0; p < block_len; p++) {
                bk[p] = f0 * r0[p];
            }
            for (size_t l = 1; l < st; l++) {
                const double fac = inv_neg_dt * m.P_in[k * st + l];
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
            const double t0 = m.T[j * st + 0];
            const double *z0 = &z_hat[0 * block_len];
            for (size_t p = 0; p < block_len; p++) {
                dj[p] = t0 * z0[p];
            }
            for (size_t k = 1; k < st; k++) {
                const double fac = m.T[j * st + k];
                const double *zk = &z_hat[k * block_len];
                for (size_t p = 0; p < block_len; p++) {
                    dj[p] += fac * zk[p];
                }
            }
        }
    }

    void SubintervalCondenser::applyStageOperatorBatch(size_t K, const double *dy_stages, double *out_stages, double *ay_buf, double * /*mass_work*/) const
    {
        const RadauDGMethod &m = getRadauDG(m_st);
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t block_len = n * K;

        for (size_t l = 0; l < st; l++) {
            const double *dy_l = &dy_stages[l * block_len];
            double *out_l = &out_stages[l * block_len];
            for (size_t j = 0; j < K; j++) {
                m_slice->applyMass(&dy_l[j * n], &out_l[j * n], 1.0, 0.0, nullptr);
            }
            if (K == 1) {
                m_avg_jac.gemv(0.0, &ay_buf[l * block_len], 1.0, dy_l);
            }
            else {
                m_avg_jac.gemm((ptrdiff_t)K, 0.0, &ay_buf[l * block_len], 1.0, dy_l);
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

        if (m_has_stage_jacs && m_max_delta_fjac > 1e-14) {
            applyDeltaStageOpBatch(K, dy_stages, ay_buf, ay_buf + st * block_len);
            for (size_t p = 0; p < st * block_len; p++) {
                out_stages[p] -= ay_buf[p];
            }
        }
    }

    void SubintervalCondenser::solveLocalStagesBatch(size_t K, const double *rhs_stages, double *dk_stages, bool exact_stage_jac, double local_rtol) const
    {
        if (K == 0) {
            return;
        }
        if (K == 1) {
            solveLocalStages(rhs_stages, dk_stages, exact_stage_jac, local_rtol);
            return;
        }

        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t block_len = n * K;
        const size_t st_block = st * block_len;
        const bool need_refine = exact_stage_jac && m_has_stage_jacs && (m_max_delta_fjac > 1e-14);

        ensureBatchWorkspace(K, n, st, 2 * K + 2 * st * n);
        double *pc_work = m_cmat_buf;                        // 2 * st_block
        double *res_buf = pc_work + 2 * st_block;            // st_block
        double *corr_buf = res_buf + st_block;               // st_block
        double *w_buf = corr_buf + st_block;                 // st_block
        double *col_rhs_norm = w_buf + st_block;             // K
        double *col_prev_res = col_rhs_norm + K;             // K

        double *rhs_copy = nullptr;
        if (need_refine) {
            rhs_copy = col_prev_res + K; // st_block (we reserved 6*st_block total)
            memcpy(rhs_copy, rhs_stages, st_block * sizeof(double));
        }

        applyPrecondOnlyBatch(K, rhs_stages, dk_stages, pc_work, m_cmat_cbuf);
        if (!need_refine) {
            return;
        }

        unsigned char col_active[K];
        bool any_active = false;
        for (size_t j = 0; j < K; j++) {
            double sq = 0.0;
            for (size_t l = 0; l < st; l++) {
                const double *col = &rhs_copy[l * block_len + j * n];
                for (size_t i = 0; i < n; i++) {
                    sq += col[i] * col[i];
                }
            }
            col_rhs_norm[j] = std::sqrt(sq);
            col_active[j] = (col_rhs_norm[j] > 0.0) ? 1 : 0;
            if (col_active[j]) any_active = true;
        }
        if (!any_active) {
            return;
        }

        applyDeltaStageOpBatch(K, dk_stages, res_buf, w_buf);
        any_active = false;
        for (size_t j = 0; j < K; j++) {
            if (!col_active[j]) continue;
            double sq = 0.0;
            for (size_t l = 0; l < st; l++) {
                const double *col = &res_buf[l * block_len + j * n];
                for (size_t i = 0; i < n; i++) {
                    sq += col[i] * col[i];
                }
            }
            const double rn = std::sqrt(sq);
            col_prev_res[j] = rn;
            if (rn <= local_rtol * col_rhs_norm[j]) {
                col_active[j] = 0;
            }
            else {
                any_active = true;
            }
        }

        for (int sweep = 0; sweep < 4 && any_active; sweep++) {
            m_stat_richardson_sweeps += K;
            applyPrecondOnlyBatch(K, res_buf, corr_buf, pc_work, m_cmat_cbuf);
            for (size_t j = 0; j < K; j++) {
                if (!col_active[j]) continue;
                for (size_t l = 0; l < st; l++) {
                    double *dcol = &dk_stages[l * block_len + j * n];
                    const double *ccol = &corr_buf[l * block_len + j * n];
                    for (size_t i = 0; i < n; i++) {
                        dcol[i] += ccol[i];
                    }
                }
            }
            applyDeltaStageOpBatch(K, corr_buf, res_buf, w_buf);
            any_active = false;
            for (size_t j = 0; j < K; j++) {
                if (!col_active[j]) continue;
                double sq = 0.0;
                for (size_t l = 0; l < st; l++) {
                    const double *col = &res_buf[l * block_len + j * n];
                    for (size_t i = 0; i < n; i++) {
                        sq += col[i] * col[i];
                    }
                }
                const double rn = std::sqrt(sq);
                if (rn <= local_rtol * col_rhs_norm[j] || rn > 0.5 * col_prev_res[j]) {
                    col_active[j] = 0;
                    col_prev_res[j] = rn;
                }
                else {
                    col_prev_res[j] = rn;
                    any_active = true;
                }
            }
        }

        // Polish any remaining unconverged columns with local GMRES
        const double gmres_trigger_fac = std::max(10.0 * local_rtol, 1e-10);
        for (size_t j = 0; j < K; j++) {
            if (col_rhs_norm[j] == 0.0 || col_prev_res[j] <= gmres_trigger_fac * col_rhs_norm[j]) {
                continue;
            }
            double *rhs_1col = corr_buf;
            double *dk_1col = corr_buf + st * n;
            for (size_t l = 0; l < st; l++) {
                memcpy(&rhs_1col[l * n], &rhs_copy[l * block_len + j * n], n * sizeof(double));
                memcpy(&dk_1col[l * n], &dk_stages[l * block_len + j * n], n * sizeof(double));
            }
            runLocalGmres(dk_1col, rhs_1col, local_rtol, 0.1 * local_rtol * col_rhs_norm[j]);
            for (size_t l = 0; l < st; l++) {
                memcpy(&dk_stages[l * block_len + j * n], &dk_1col[l * n], n * sizeof(double));
            }
        }
    }

    void SubintervalCondenser::computeCondensedMatrix(const SparseDenseOutputOp &transfer_prev_to_cur, double *C_out) const
    {
        const size_t n = m_n_space;
        const size_t st = m_st;
        const size_t K_full = transfer_prev_to_cur.is_identity ? n : (transfer_prev_to_cur.src_npts * transfer_prev_to_cur.node);
        const size_t chunk_max = 64;
        const size_t m = m_slice->node;
        const size_t ncol = m_slice->nstages;
        const size_t n_sub = m_slice->nSub();

        memset(C_out, 0, n * K_full * sizeof(double));

        if (m_cmat_cols_cap < K_full) {
            m_cmat_cols = (size_t *)realloc(m_cmat_cols, K_full * sizeof(size_t));
            m_cmat_cols_cap = K_full;
        }

        // Precompute extra scratch: rhs_chunk (st*n*chunk_max) + dk_chunk (st*n*chunk_max) + v_cur (n) + e_vec (K_full) + bw_tmp (n) + 2*chunk_max
        const size_t extra = 2 * st * n * chunk_max + 2 * n + K_full + 2 * chunk_max + 64;
        ensureBatchWorkspace(chunk_max, n, st, extra);
        // Place computeCondensedMatrix buffers at the tail of m_cmat_buf (after 6 * st * n * chunk_max + 2 * chunk_max + 64 used by solveLocalStagesBatch)
        double *tail = m_cmat_buf + 6 * st * n * chunk_max + 2 * chunk_max + 64;
        double *rhs_chunk = tail;
        double *dk_chunk = rhs_chunk + st * n * chunk_max;
        double *v_cur = dk_chunk + st * n * chunk_max;
        double *bw_tmp = v_cur + n;
        double *e_vec = bw_tmp + n;

        if (transfer_prev_to_cur.is_identity) {
            bool var_active[m];
            memset(var_active, 0, m * sizeof(bool));
            for (size_t nz = 0; nz < m_slice->m_nnz; nz++) {
                var_active[m_slice->m_cols[nz]] = true;
            }

            size_t n_active = 0;
            for (size_t j = 0; j < n; j++) {
                const size_t pt = j / m;
                const size_t v = j % m;
                if (pt < n_sub * (ncol + 1) && (pt % (ncol + 1)) != 0 && var_active[v]) {
                    m_cmat_cols[n_active++] = j;
                }
            }

            for (size_t c0 = 0; c0 < n_active; c0 += chunk_max) {
                const size_t ck = std::min(chunk_max, n_active - c0);
                const size_t block_len = n * ck;

                memset(rhs_chunk, 0, block_len * sizeof(double));
                for (size_t slot = 0; slot < ck; slot++) {
                    const size_t orig_j = m_cmat_cols[c0 + slot];
                    const size_t pt = orig_j / m;
                    const size_t var_c = orig_j % m;
                    const size_t k = pt / (ncol + 1);
                    const size_t i_s = (pt % (ncol + 1)) - 1;
                    const size_t stride_k = k * (ncol + 1) * m;
                    const double hk = m_slice->h[k];
                    double *dst_col = &rhs_chunk[slot * n];

                    for (size_t nz = 0; nz < m_slice->m_nnz; nz++) {
                        if ((size_t)m_slice->m_cols[nz] != var_c) continue;
                        const size_t r = (size_t)m_slice->m_rows[nz];
                        const double scale_mv = hk * m_slice->m_vals[nz];
                        dst_col[stride_k + r] += scale_mv * m_slice->c_b[i_s];
                        for (size_t jr = 0; jr < ncol; jr++) {
                            dst_col[stride_k + (jr + 1) * m + r] += scale_mv * m_slice->c_A[jr * ncol + i_s];
                        }
                    }
                }
                for (size_t l = 1; l < st; l++) {
                    memcpy(&rhs_chunk[l * block_len], &rhs_chunk[0], block_len * sizeof(double));
                }

                solveLocalStagesBatch(ck, rhs_chunk, dk_chunk, true, 1e-12);

                const double *last_stage = &dk_chunk[(st - 1) * block_len];
                for (size_t slot = 0; slot < ck; slot++) {
                    memcpy(&C_out[m_cmat_cols[c0 + slot] * n], &last_stage[slot * n], n * sizeof(double));
                }
            }
        }
        else {
            memset(e_vec, 0, K_full * sizeof(double));
            size_t slot = 0;
            auto flush_batch = [&](size_t ck) {
                if (ck == 0) return;
                const size_t block_len = n * ck;
                for (size_t l = 1; l < st; l++) {
                    memcpy(&rhs_chunk[l * block_len], &rhs_chunk[0], block_len * sizeof(double));
                }
                solveLocalStagesBatch(ck, rhs_chunk, dk_chunk, true, 1e-12);
                const double *last_stage = &dk_chunk[(st - 1) * block_len];
                for (size_t s_idx = 0; s_idx < ck; s_idx++) {
                    memcpy(&C_out[m_cmat_cols[s_idx] * n], &last_stage[s_idx * n], n * sizeof(double));
                }
            };

            for (size_t j = 0; j < K_full; j++) {
                e_vec[j] = 1.0;
                transfer_prev_to_cur.apply(e_vec, v_cur);
                e_vec[j] = 0.0;
                m_slice->applyMass(v_cur, bw_tmp, 1.0, 0.0, nullptr);

                bool nonzero = false;
                for (size_t i = 0; i < n; i++) {
                    if (bw_tmp[i] != 0.0) {
                        nonzero = true;
                        break;
                    }
                }
                if (!nonzero) continue;

                m_cmat_cols[slot] = j;
                memcpy(&rhs_chunk[slot * n], bw_tmp, n * sizeof(double));
                slot++;
                if (slot == chunk_max) {
                    // Compact stage 0 already in rhs_chunk[0..chunk_max*n), flush!
                    flush_batch(chunk_max);
                    slot = 0;
                }
            }
            if (slot > 0) {
                flush_batch(slot);
            }
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
