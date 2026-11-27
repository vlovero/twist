#include "modulating/spatial_slice.h"
#include "linalg.h"
#include "numpy_like.h"
#include "tools/helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace TWIST
{
    static size_t slice_searchsorted(const double *arr, size_t size, double key)
    {
        size_t lo = 0;
        size_t hi = size;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (arr[mid] <= key) {
                lo = mid + 1;
            }
            else {
                hi = mid;
            }
        }
        return lo;
    }

    SpatialSlice::SpatialSlice(const SpatialSlice &other)
    {
        *this = other;
    }

    SpatialSlice &SpatialSlice::operator=(const SpatialSlice &other)
    {
        if (this == &other) {
            return *this;
        }
        freeBuffers();

        nstages = other.nstages;
        node = other.node;
        nnodes = other.nnodes;
        n_space = other.n_space;
        unscaled_spatial_period = other.unscaled_spatial_period;
        c_A = other.c_A;
        c_b = other.c_b;
        c_c = other.c_c;
        c_P = other.c_P;
        c_chat = other.c_chat;
        func = other.func;
        fjac = other.fjac;
        pjac = other.pjac;
        ndiff = other.ndiff;

        if (nnodes > 0) {
            t = (double *)malloc(nnodes * sizeof(double));
            memcpy(t, other.t, nnodes * sizeof(double));
        }
        if (nnodes > 1) {
            h = (double *)malloc((nnodes - 1) * sizeof(double));
            memcpy(h, other.h, (nnodes - 1) * sizeof(double));
        }
        if (ndiff > 0) {
            to_delete = (int *)malloc(ndiff * sizeof(int));
            diff_coeffs = (double *)malloc(ndiff * sizeof(double));
            memcpy(to_delete, other.to_delete, ndiff * sizeof(int));
            memcpy(diff_coeffs, other.diff_coeffs, ndiff * sizeof(double));
        }
        if (node > 0) {
            M = (double *)malloc(node * node * sizeof(double));
            dM_dc = (double *)malloc(node * node * sizeof(double));
            memcpy(M, other.M, node * node * sizeof(double));
            memcpy(dM_dc, other.dM_dc, node * node * sizeof(double));
        }
        if (nstages > 0 && other.A_gl_invT != nullptr) {
            A_gl_invT = (double *)malloc(nstages * nstages * sizeof(double));
            memcpy(A_gl_invT, other.A_gl_invT, nstages * nstages * sizeof(double));
        }
        if (nstages > 0 && other.D_gl != nullptr) {
            const size_t n_basis = nstages + 1;
            D_gl = (double *)malloc(n_basis * n_basis * sizeof(double));
            memcpy(D_gl, other.D_gl, n_basis * n_basis * sizeof(double));
        }
        if (node > 0) {
            m_nnz = other.m_nnz;
            m_rows = (int *)malloc(node * sizeof(int));
            m_cols = (int *)malloc(node * sizeof(int));
            m_vals = (double *)malloc(node * sizeof(double));
            if (m_nnz > 0 && other.m_rows) {
                memcpy(m_rows, other.m_rows, m_nnz * sizeof(int));
                memcpy(m_cols, other.m_cols, m_nnz * sizeof(int));
                memcpy(m_vals, other.m_vals, m_nnz * sizeof(double));
            }
            dm_nnz = other.dm_nnz;
            dm_rows = (int *)malloc(node * sizeof(int));
            dm_cols = (int *)malloc(node * sizeof(int));
            dm_vals = (double *)malloc(node * sizeof(double));
            if (dm_nnz > 0 && other.dm_rows) {
                memcpy(dm_rows, other.dm_rows, dm_nnz * sizeof(int));
                memcpy(dm_cols, other.dm_cols, dm_nnz * sizeof(int));
                memcpy(dm_vals, other.dm_vals, dm_nnz * sizeof(double));
            }
        }
        return *this;
    }

    SpatialSlice::~SpatialSlice()
    {
        freeBuffers();
    }

    void SpatialSlice::freeBuffers()
    {
        if (t) {
            free(t);
            t = nullptr;
        }
        if (h) {
            free(h);
            h = nullptr;
        }
        if (to_delete) {
            free(to_delete);
            to_delete = nullptr;
        }
        if (diff_coeffs) {
            free(diff_coeffs);
            diff_coeffs = nullptr;
        }
        if (M) {
            free(M);
            M = nullptr;
        }
        if (dM_dc) {
            free(dM_dc);
            dM_dc = nullptr;
        }
        if (A_gl_invT) {
            free(A_gl_invT);
            A_gl_invT = nullptr;
        }
        if (D_gl) {
            free(D_gl);
            D_gl = nullptr;
        }
        if (m_rows) { free(m_rows); m_rows = nullptr; }
        if (m_cols) { free(m_cols); m_cols = nullptr; }
        if (m_vals) { free(m_vals); m_vals = nullptr; }
        if (dm_rows) { free(dm_rows); dm_rows = nullptr; }
        if (dm_cols) { free(dm_cols); dm_cols = nullptr; }
        if (dm_vals) { free(dm_vals); dm_vals = nullptr; }
        m_nnz = 0;
        dm_nnz = 0;
    }

    void SpatialSlice::init(size_t in_nstages, size_t in_node, size_t in_nnodes, const double *in_t, func_t in_func, func_t in_fjac, pjac_t in_pjac, const diffusion_t &diffusion, double in_unscaled_spatial_period)
    {
        freeBuffers();

        nstages = in_nstages;
        node = in_node;
        unscaled_spatial_period = in_unscaled_spatial_period;
        func = in_func;
        fjac = in_fjac;
        pjac = in_pjac;

        const _method_t &method = GL_methods[nstages];
        c_A = std::get<1>(method);
        c_b = std::get<2>(method);
        c_c = std::get<3>(method);
        c_P = std::get<4>(method);
        c_chat = std::get<5>(method);

        ndiff = diffusion.size();
        if (ndiff > 0) {
            to_delete = (int *)malloc(ndiff * sizeof(int));
            diff_coeffs = (double *)malloc(ndiff * sizeof(double));
            int offset = 1;
            for (size_t i = 0; i < ndiff; i++) {
                to_delete[i] = diffusion[i].first + offset;
                diff_coeffs[i] = diffusion[i].second;
                offset++;
            }
        }

        M = (double *)calloc(node * node, sizeof(double));
        dM_dc = (double *)calloc(node * node, sizeof(double));
        m_rows = (int *)malloc(node * sizeof(int));
        m_cols = (int *)malloc(node * sizeof(int));
        m_vals = (double *)malloc(node * sizeof(double));
        dm_rows = (int *)malloc(node * sizeof(int));
        dm_cols = (int *)malloc(node * sizeof(int));
        dm_vals = (double *)malloc(node * sizeof(double));
        m_nnz = 0;
        dm_nnz = 0;

        // Precompute (A_GL)^{-T} for exact stage-derivative extraction from collocation state
        A_gl_invT = (double *)malloc(nstages * nstages * sizeof(double));
        // c_A is row-major A_GL, so in Fortran column-major it is A_GL^T!
        memcpy(A_gl_invT, c_A, nstages * nstages * sizeof(double));
        int s = (int)nstages;
        int ipiv[nstages];
        int info = 0;
        double work_inv[nstages * nstages];
        // Invert column-major A_GL^T -> (A_GL)^{-T}
        dgetrf(s, s, A_gl_invT, s, ipiv, &info);
        // Use dgesv on identity if dgetri isn't declared in linalg.h
        double rhs[nstages * nstages];
        memset(rhs, 0, sizeof(rhs));
        for (size_t i = 0; i < nstages; i++) {
            rhs[i * nstages + i] = 1.0;
        }
        memcpy(work_inv, c_A, nstages * nstages * sizeof(double));
        dgesv(s, s, work_inv, s, ipiv, rhs, s, &info);
        memcpy(A_gl_invT, rhs, nstages * nstages * sizeof(double));

        // Precompute (nstages+1) x (nstages+1) Lagrange differentiation matrix D_gl
        const size_t n_basis = nstages + 1;
        D_gl = (double *)malloc(n_basis * n_basis * sizeof(double));
        double nodes_gl[n_basis];
        nodes_gl[0] = 0.0;
        for (size_t j = 0; j < nstages; j++) {
            nodes_gl[j + 1] = c_c[j];
        }
        for (size_t l = 0; l < n_basis; l++) {
            const double th = nodes_gl[l];
            for (size_t j = 0; j < n_basis; j++) {
                double den = 1.0;
                for (size_t q = 0; q < n_basis; q++) {
                    if (q != j) den *= (nodes_gl[j] - nodes_gl[q]);
                }
                double num = 0.0;
                for (size_t k = 0; k < n_basis; k++) {
                    if (k == j) continue;
                    double prod = 1.0;
                    for (size_t q = 0; q < n_basis; q++) {
                        if (q != j && q != k) prod *= (th - nodes_gl[q]);
                    }
                    num += prod;
                }
                D_gl[l * n_basis + j] = num / den;
            }
        }

        updateGrid(in_nnodes, in_t);
    }

    void SpatialSlice::initCoarseFrom(const SpatialSlice &fine, size_t coarse_nstages)
    {
        diffusion_t diff(fine.ndiff);
        for (size_t i = 0; i < fine.ndiff; i++) {
            diff[i].first = fine.to_delete[i] - (int)(i + 1);
            diff[i].second = fine.diff_coeffs[i];
        }
        init(coarse_nstages, fine.node, fine.nnodes, fine.t, fine.func, fine.fjac, fine.pjac, diff, fine.unscaled_spatial_period);
        if (M && fine.M && node > 0) {
            memcpy(M, fine.M, node * node * sizeof(double));
            memcpy(dM_dc, fine.dM_dc, node * node * sizeof(double));
            m_nnz = fine.m_nnz;
            if (m_nnz > 0 && fine.m_rows) {
                memcpy(m_rows, fine.m_rows, m_nnz * sizeof(int));
                memcpy(m_cols, fine.m_cols, m_nnz * sizeof(int));
                memcpy(m_vals, fine.m_vals, m_nnz * sizeof(double));
            }
            dm_nnz = fine.dm_nnz;
            if (dm_nnz > 0 && fine.dm_rows) {
                memcpy(dm_rows, fine.dm_rows, dm_nnz * sizeof(int));
                memcpy(dm_cols, fine.dm_cols, dm_nnz * sizeof(int));
                memcpy(dm_vals, fine.dm_vals, dm_nnz * sizeof(double));
            }
        }
    }

    void SpatialSlice::initFromCollocator(const Collocator &col)
    {
        // const_cast avoided by reading non-const via Collocator copy or pointer if needed;
        // Collocator::t() is non-const in existing API, so cast away constness on pointer safely.
        Collocator &col_mut = *((Collocator *)&col);
        init(col.getNStages(), col.getNode(), col.getNNodes(), col_mut.t(), col.func(), col.fjac(), col.pjac(), col.getDiffusion(), col.unscaledSpatialPeriod());
        updateMassMatrices(col.waveSpeed(), col.p()[col.getNParam() - 1]);
    }

    void SpatialSlice::updateGrid(size_t new_nnodes, const double *new_t)
    {
        nnodes = new_nnodes;
        n_space = node * ((nnodes - 1) * (nstages + 1) + 1);
        t = (double *)realloc(t, nnodes * sizeof(double));
        h = (double *)realloc(h, (nnodes - 1) * sizeof(double));
        memcpy(t, new_t, nnodes * sizeof(double));
        for (size_t k = 0; k < nnodes - 1; k++) {
            h[k] = t[k + 1] - t[k];
        }
    }

    void SpatialSlice::buildSparseMass(double c, double sps, size_t &nnz_out, int *rows_out, int *cols_out, double *vals_out) const
    {
        const double L = sps * unscaled_spatial_period;
        double D_full[node];
        memset(D_full, 0, node * sizeof(double));
        for (size_t idx = 0; idx < ndiff; idx++) {
            const int diff_loc = to_delete[idx] - (int)idx - 1;
            if (diff_loc >= 0 && (size_t)diff_loc < node) {
                D_full[diff_loc] = diff_coeffs[idx];
            }
        }

        size_t nnz = 0;
        size_t k = 0;
        for (size_t i = 0; i < node; i++) {
            bool next_is_del = false;
            bool cur_is_del = false;
            for (size_t q = 0; q < ndiff; q++) {
                if (to_delete[q] == (int)(i + 1)) {
                    next_is_del = true;
                }
                if (to_delete[q] == (int)i) {
                    cur_is_del = true;
                }
            }
            if (next_is_del) {
                continue;
            }
            if (cur_is_del) {
                const size_t j = (size_t)to_delete[k];
                k++;
                rows_out[nnz] = (int)j;
                cols_out[nnz] = (int)(j - 1);
                vals_out[nnz] = -L / D_full[j - k];
                nnz++;
                continue;
            }
            rows_out[nnz] = (int)i;
            cols_out[nnz] = (int)i;
            vals_out[nnz] = L / c;
            nnz++;
        }
        nnz_out = nnz;
    }

    void SpatialSlice::updateMassMatrices(double c, double sps)
    {
        const double L = sps * unscaled_spatial_period;
        memset(M, 0, node * node * sizeof(double));
        memset(dM_dc, 0, node * node * sizeof(double));

        buildSparseMass(c, sps, m_nnz, m_rows, m_cols, m_vals);

        dm_nnz = 0;
        const double inv_c2 = -L / (c * c);
        for (size_t nz = 0; nz < m_nnz; nz++) {
            const int r = m_rows[nz];
            const int col = m_cols[nz];
            M[(size_t)col * node + (size_t)r] = m_vals[nz];
            if (r == col) {
                dM_dc[(size_t)col * node + (size_t)r] = inv_c2;
                dm_rows[dm_nnz] = r;
                dm_cols[dm_nnz] = col;
                dm_vals[dm_nnz] = inv_c2;
                dm_nnz++;
            }
        }
    }

    void SpatialSlice::evaluateResidual(const RP(double) Y, const RP(double) p, RP(double) residual, RP(double) yp) const
    {
        const size_t n = nnodes - 1;
        const size_t m = node;
        const size_t s = nstages;
        const double scale = unscaled_spatial_period;

        for (size_t k = 0; k < n; k++) {
            const double hk = h[k];
            const size_t stride_k = m * (s + 1) * k;
            const size_t stride_k1 = m * (s + 1) * (k + 1);

            const double *yk = &Y[stride_k];
            const double *yk1 = &Y[stride_k1];
            double *Fk = &residual[stride_k];

            for (size_t l = 0; l < m; l++) {
                Fk[l] = yk[l] - yk1[l];
            }

            for (size_t i = 0; i < s; i++) {
                const size_t stride_i = (i + 1) * m;
                const double *yk_i = &Y[stride_k + stride_i];
                double *Sk_i = &yp[stride_k + stride_i];

                func(yk_i, p, Sk_i);
                for (size_t l = 0; l < m; l++) {
                    Sk_i[l] *= scale;
                }

                const double fac = hk * c_b[i];
                for (size_t l = 0; l < m; l++) {
                    Fk[l] += fac * Sk_i[l];
                }
            }

            for (size_t i = 0; i < s; i++) {
                const size_t stride_i = (i + 1) * m;
                const double *yki = &Y[stride_k + stride_i];
                double *Hk_i = &residual[stride_k + stride_i];

                for (size_t l = 0; l < m; l++) {
                    Hk_i[l] = yk[l] - yki[l];
                }

                for (size_t j = 0; j < s; j++) {
                    const double *Sk_j = &yp[stride_k + (j + 1) * m];
                    const double fac = hk * c_A[i * s + j];
                    for (size_t l = 0; l < m; l++) {
                        Hk_i[l] += fac * Sk_j[l];
                    }
                }
            }
        }

        // Periodic spatial boundary condition: y(0) - y(1) = 0
        const double *y0 = Y;
        const double *yn = &Y[n * m * (s + 1)];
        double *bc_out = &residual[n * m * (s + 1)];
        for (size_t l = 0; l < m; l++) {
            bc_out[l] = y0[l] - yn[l];
        }
    }

    void SpatialSlice::applyMassOp(const double *M_colmajor, const RP(double) v, RP(double) out, double alpha, double beta, RP(double) work) const
    {
        const size_t n = nnodes - 1;
        const size_t m = node;
        const size_t s = nstages;
        const int im = (int)m;
        const int is = (int)s;

        for (size_t k = 0; k < n; k++) {
            const size_t stride_k = m * (s + 1) * k;
            const double hk = h[k];
            const double scale = alpha * hk;

            // work = M_true * v_stages  (m x s)
            // Note: M_colmajor is column-major (M_true^T in row-major, i.e., M_true in column-major!)
            dgemm("N", "N", im, is, im, 1.0, M_colmajor, im, &v[stride_k + m], im, 0.0, work, im);

            // Top block (continuity/quadrature row): out_0 = beta * out_0 + scale * work * c_b
            dgemv("N", im, is, scale, work, im, c_b, 1, beta, &out[stride_k], 1);

            // Stage rows: out_{1..s} = beta * out_{1..s} + scale * work * c_A^T
            // Since c_A is row-major A_GL, in Fortran column-major it is A_GL^T, so transb = "N" gives work * A_GL^T!
            dgemm("N", "N", im, is, is, scale, work, im, c_A, is, beta, &out[stride_k + m], im);
        }

        // Periodic BC rows have zero mass
        double *bc_out = &out[n * m * (s + 1)];
        if (beta == 0.0) {
            memset(bc_out, 0, m * sizeof(double));
        }
        else if (beta != 1.0) {
            for (size_t l = 0; l < m; l++) {
                bc_out[l] *= beta;
            }
        }
    }

    void SpatialSlice::applyMassOpSparse(size_t nnz, const int *rows, const int *cols, const double *vals, const RP(double) v, RP(double) out, double alpha, double beta) const
    {
        const size_t n = nnodes - 1;
        const size_t m = node;
        const size_t s = nstages;

        if (beta == 0.0) {
            memset(out, 0, n_space * sizeof(double));
        }
        else if (beta != 1.0) {
            for (size_t i = 0; i < n_space; i++) {
                out[i] *= beta;
            }
        }

        double tmp[s];
        for (size_t k = 0; k < n; k++) {
            const size_t stride_k = m * (s + 1) * k;
            const double scale_k = alpha * h[k];

            for (size_t nz = 0; nz < nnz; nz++) {
                const size_t r = (size_t)rows[nz];
                const size_t c = (size_t)cols[nz];
                const double s_mv = scale_k * vals[nz];

                for (size_t i = 0; i < s; i++) {
                    tmp[i] = v[stride_k + (i + 1) * m + c];
                }

                double sum_b = 0.0;
                for (size_t i = 0; i < s; i++) {
                    sum_b += c_b[i] * tmp[i];
                }
                out[stride_k + r] += s_mv * sum_b;

                for (size_t j = 0; j < s; j++) {
                    const double *Aj = &c_A[j * s];
                    double sum_a = 0.0;
                    for (size_t i = 0; i < s; i++) {
                        sum_a += Aj[i] * tmp[i];
                    }
                    out[stride_k + (j + 1) * m + r] += s_mv * sum_a;
                }
            }
        }
    }

    void SpatialSlice::applyMass(const RP(double) v, RP(double) out, double alpha, double beta, RP(double) /*work*/) const
    {
        applyMassOpSparse(m_nnz, m_rows, m_cols, m_vals, v, out, alpha, beta);
    }

    void SpatialSlice::applyMassDerivC(const RP(double) v, RP(double) out, double alpha, double beta, RP(double) /*work*/) const
    {
        applyMassOpSparse(dm_nnz, dm_rows, dm_cols, dm_vals, v, out, alpha, beta);
    }

    void SpatialSlice::applyMassDerivSps(double sps, const RP(double) v, RP(double) out, double alpha, double beta, RP(double) /*work*/) const
    {
        applyMassOpSparse(m_nnz, m_rows, m_cols, m_vals, v, out, alpha / sps, beta);
    }

    void SpatialSlice::fillQuadratureWeights(RP(double) weights, bool include_nodes) const
    {
        const size_t n = nnodes - 1;
        const size_t m = node;
        const size_t s = nstages;

        memset(weights, 0, n_space * sizeof(double));
        for (size_t k = 0; k < n; k++) {
            const size_t stride_k = m * (s + 1) * k;
            const double hk = h[k];
            for (size_t i = 0; i < s; i++) {
                const double w = hk * c_b[i];
                for (size_t l = 0; l < m; l++) {
                    weights[stride_k + (i + 1) * m + l] = w;
                }
            }
            if (include_nodes) {
                const double w_node = 0.25 * hk / (double)(s + 1);
                for (size_t l = 0; l < m; l++) {
                    weights[stride_k + l] += w_node;
                    weights[stride_k + (s + 1) * m + l] += w_node;
                }
            }
        }
    }

    void SpatialSlice::computeMonitorFromState(const double *Y, RP(double) F, RP(double) v, double &theta, double &err_exp, RP(double) work) const
    {
        const size_t n = nnodes - 1;
        const size_t m = node;
        const size_t s = nstages;
        const double fac = std::tgamma((double)(s + 2));
        err_exp = 1.0 / (double)(s + 1);

        double *diff_Y = work;
        double *hk_Sk = diff_Y + m * s;
        double *tmp = hk_Sk + m * s;

        F[0] = 0.0;
        for (size_t k = 0; k < n; k++) {
            const size_t stride_k = m * (s + 1) * k;
            const double *yk = &Y[stride_k];
            for (size_t i = 0; i < s; i++) {
                const double *yki = &Y[stride_k + (i + 1) * m];
                for (size_t l = 0; l < m; l++) {
                    diff_Y[i * m + l] = yki[l] - yk[l];
                }
            }
            // hk * Sk = diff_Y * A_gl_invT
            // Note: A_gl_invT is column-major A_GL^{-T}, so (diff_Y) * A_GL^{-T} in column-major is diff_Y * A_gl_invT
            dgemm("N", "N", (int)m, (int)s, (int)s, 1.0, diff_Y, (int)m, A_gl_invT, (int)s, 0.0, hk_Sk, (int)m);
            dgemv("N", (int)m, (int)s, 1.0, hk_Sk, (int)m, &c_P[s - 1], (int)s, 0.0, tmp, 1);

            double vk = fac * norm2(tmp, m);
            vk = std::pow(vk, err_exp);
            v[k] = vk;
            F[k + 1] = F[k] + vk;
        }
        theta = F[n];
        if (theta > 0.0) {
            for (size_t i = 0; i < nnodes; i++) {
                F[i] /= theta;
            }
        }
        else {
            for (size_t i = 0; i < nnodes; i++) {
                F[i] = t[i];
            }
        }
    }

    double *SpatialSlice::equidistributeMesh(const double *F_cum, const double *v_max, double theta, double err_exp, double geps, size_t min_nnodes, size_t max_nnodes, size_t &nnodes_opt, bool small_change) const
    {
        const double leading_error_scale = c_chat[0];
        if (theta <= 1e-15) {
            nnodes_opt = std::max<size_t>({ 3ul, min_nnodes, nnodes });
            nnodes_opt = std::min(max_nnodes, nnodes_opt);
            double *t_opt = (double *)malloc(nnodes_opt * sizeof(double));
            for (size_t i = 0; i < nnodes_opt; i++) {
                t_opt[i] = (double)i / (double)(nnodes_opt - 1);
            }
            return t_opt;
        }

        nnodes_opt = (size_t)(theta * std::pow(leading_error_scale / geps, err_exp) + 1.0);
        double r2 = 0.0;
        for (size_t i = 0; i < nnodes - 1; i++) {
            const double tmp = h[i] * std::pow((leading_error_scale / geps) * v_max[i], err_exp);
            r2 += tmp;
        }
        const size_t r2_int = (size_t)((int)r2 + 1);
        nnodes_opt = std::min(nnodes_opt, std::max(nnodes_opt, r2_int) / 2) + 1;
        nnodes_opt = (size_t)std::ceil((double)nnodes_opt * 1.15);
        if (small_change) {
            nnodes_opt = std::min(nnodes_opt, (size_t)(1.2 * (double)nnodes));
            nnodes_opt = std::max(nnodes_opt, (size_t)(0.8 * (double)nnodes));
        }
        nnodes_opt = std::max(min_nnodes, nnodes_opt);
        nnodes_opt = std::min(max_nnodes, nnodes_opt);
        nnodes_opt = std::max<size_t>(3ul, nnodes_opt);

        double *t_opt = (double *)malloc(nnodes_opt * sizeof(double));
        for (size_t i = 0; i < nnodes_opt; i++) {
            t_opt[i] = (double)i / (double)(nnodes_opt - 1);
        }
        interp(t_opt, nnodes_opt, t_opt, F_cum, t, nnodes, 1);
        t_opt[0] = 0.0;
        t_opt[nnodes_opt - 1] = 1.0;
        return t_opt;
    }

    SparseDenseOutputOp::SparseDenseOutputOp(const SparseDenseOutputOp &other)
    {
        *this = other;
    }

    SparseDenseOutputOp &SparseDenseOutputOp::operator=(const SparseDenseOutputOp &other)
    {
        if (this == &other) {
            return *this;
        }
        freeBuffers();
        nstages = other.nstages;
        node = other.node;
        src_npts = other.src_npts;
        dst_npts = other.dst_npts;
        is_identity = other.is_identity;

        if (dst_npts > 0 && !is_identity) {
            src_sub = (int *)malloc(dst_npts * sizeof(int));
            weights = (double *)malloc(dst_npts * (nstages + 1) * sizeof(double));
            memcpy(src_sub, other.src_sub, dst_npts * sizeof(int));
            memcpy(weights, other.weights, dst_npts * (nstages + 1) * sizeof(double));
        }
        return *this;
    }

    SparseDenseOutputOp::~SparseDenseOutputOp()
    {
        freeBuffers();
    }

    void SparseDenseOutputOp::freeBuffers()
    {
        if (src_sub) {
            free(src_sub);
            src_sub = nullptr;
        }
        if (weights) {
            free(weights);
            weights = nullptr;
        }
        is_identity = false;
    }

    void SparseDenseOutputOp::build(const SpatialSlice &src, const SpatialSlice &dst)
    {
        freeBuffers();
        nstages = src.nstages;
        node = src.node;
        src_npts = src.nPoints();
        dst_npts = dst.nPoints();

        if (src.nnodes == dst.nnodes && src.nstages == dst.nstages && src.node == dst.node) {
            double max_diff = 0.0;
            for (size_t i = 0; i < src.nnodes; i++) {
                max_diff = std::max(max_diff, std::abs(src.t[i] - dst.t[i]));
            }
            if (max_diff == 0.0) {
                is_identity = true;
                return;
            }
        }

        const size_t s = nstages;
        const size_t n_basis = s + 1;
        src_sub = (int *)malloc(dst_npts * sizeof(int));
        weights = (double *)malloc(dst_npts * n_basis * sizeof(double));

        double nodes[n_basis];
        nodes[0] = 0.0;
        for (size_t j = 0; j < s; j++) {
            nodes[j + 1] = src.c_c[j];
        }

        const size_t n_src_sub = src.nnodes - 1;

        for (size_t q = 0; q < dst_npts; q++) {
            double xi = 0.0;
            if (q == dst_npts - 1) {
                // Exact right endpoint maps to exact right endpoint of src
                src_sub[q] = (int)(n_src_sub - 1);
                double *w = &weights[q * n_basis];
                for (size_t j = 0; j < n_basis; j++) {
                    w[j] = 0.0;
                }
                // Or evaluate at theta = 1.0 in last subinterval; wait:
                // Using the exact endpoint index is handled in apply() via q == dst_npts - 1,
                // or compute Lagrange weights at theta = 1.0 as fallback.
                xi = 1.0;
            }
            else {
                const size_t dst_basis = dst.nstages + 1;
                const size_t k_dst = q / dst_basis;
                const size_t s_dst = q % dst_basis;
                xi = (s_dst == 0) ? dst.t[k_dst] : (dst.t[k_dst] + dst.c_c[s_dst - 1] * dst.h[k_dst]);
                if (xi < 0.0) {
                    xi = 0.0;
                }
                if (xi >= 1.0) {
                    xi = 1.0;
                }
            }

            size_t idx = (q == dst_npts - 1) ? n_src_sub : slice_searchsorted(src.t, n_src_sub, xi);
            if (idx == 0) {
                idx = 1;
            }
            const size_t k_src = idx - 1;
            src_sub[q] = (int)k_src;
            const double theta = (xi - src.t[k_src]) / src.h[k_src];

            double *w = &weights[q * n_basis];
            for (size_t j = 0; j < n_basis; j++) {
                double prod = 1.0;
                for (size_t m = 0; m < n_basis; m++) {
                    if (m != j) {
                        prod *= (theta - nodes[m]) / (nodes[j] - nodes[m]);
                    }
                }
                w[j] = prod;
            }
        }
    }

    void SparseDenseOutputOp::apply(const RP(double) src_y, RP(double) dst_y) const
    {
        if (is_identity) {
            memcpy(dst_y, src_y, dst_npts * node * sizeof(double));
            return;
        }

        const size_t n_basis = nstages + 1;
        const size_t m = node;

        for (size_t q = 0; q < dst_npts - 1; q++) {
            const double *w = &weights[q * n_basis];
            const double *src_blk = &src_y[(size_t)src_sub[q] * n_basis * m];
            double *out = &dst_y[q * m];
            for (size_t d = 0; d < m; d++) {
                out[d] = w[0] * src_blk[d];
            }
            for (size_t j = 1; j < n_basis; j++) {
                const double wj = w[j];
                const double *sj = &src_blk[j * m];
                for (size_t d = 0; d < m; d++) {
                    out[d] += wj * sj[d];
                }
            }
        }
        // Exact endpoint copy
        memcpy(&dst_y[(dst_npts - 1) * m], &src_y[(src_npts - 1) * m], m * sizeof(double));
    }
} // namespace TWIST
