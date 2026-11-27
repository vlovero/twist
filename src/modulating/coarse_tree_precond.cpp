#include "modulating/coarse_tree_precond.h"
#include "linalg.h"
#include "modulating/radau_dg.h"
#include "tools/helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

namespace TWIST
{
    // =========================================================================
    // CyclicWindowTree
    // =========================================================================

    CyclicWindowTree::~CyclicWindowTree()
    {
        freeBuffers();
    }

    void CyclicWindowTree::resetLevels()
    {
        if (m_levels) {
            for (size_t l = 0; l < m_levels_cap; l++) {
                free(m_levels[l].in_dims);
                free(m_levels[l].even_offs);
                free(m_levels[l].odd_offs);
                free(m_levels[l].B_in);
                free(m_levels[l].B_prod);
            }
            free(m_levels);
            m_levels = nullptr;
            m_levels_cap = 0;
        }
        m_nlevels = 0;
    }

    void CyclicWindowTree::freeBuffers()
    {
        resetLevels();
        if (m_win_offs) { free(m_win_offs); m_win_offs = nullptr; }
        if (m_win_dims) { free(m_win_dims); m_win_dims = nullptr; }
        m_win_cap = 0;
        if (m_tree_mem) { free(m_tree_mem); m_tree_mem = nullptr; m_tree_mem_cap = 0; }
        if (m_root_mat) { free(m_root_mat); m_root_mat = nullptr; m_root_cap = 0; }
        if (m_root_ipiv) { free(m_root_ipiv); m_root_ipiv = nullptr; }
        if (m_f_work) { free(m_f_work); m_f_work = nullptr; }
        if (m_u_work) { free(m_u_work); m_u_work = nullptr; }
        m_vec_work_cap = 0;
        m_M = 0;
        m_total_dim = 0;
        m_root_dim = 0;
        m_ready = false;
    }

    void CyclicWindowTree::buildAndFactor(size_t M, const size_t *dims, const double *const *B_ptrs)
    {
        m_ready = false;
        m_M = M;
        if (M == 0) {
            m_total_dim = 0;
            return;
        }

        if (m_win_cap < M + 1) {
            m_win_offs = (size_t *)realloc(m_win_offs, (M + 1) * sizeof(size_t));
            m_win_dims = (size_t *)realloc(m_win_dims, (M + 1) * sizeof(size_t));
            m_win_cap = M + 1;
        }
        m_win_offs[0] = 0;
        for (size_t w = 0; w < M; w++) {
            m_win_dims[w] = dims[w];
            m_win_offs[w + 1] = m_win_offs[w] + dims[w];
        }
        m_total_dim = m_win_offs[M];

        // Count levels
        size_t n_lvls = 0;
        size_t cnt_tmp = M;
        while (cnt_tmp > 1) {
            cnt_tmp = (cnt_tmp / 2) + (cnt_tmp & 1);
            n_lvls++;
        }
        if (m_levels_cap < n_lvls) {
            const size_t old_cap = m_levels_cap;
            m_levels = (TreeLevel *)realloc(m_levels, n_lvls * sizeof(TreeLevel));
            for (size_t l = old_cap; l < n_lvls; l++) {
                memset(&m_levels[l], 0, sizeof(TreeLevel));
            }
            m_levels_cap = n_lvls;
        }
        m_nlevels = n_lvls;

        // First pass: plan level dimensions & compute total matrix pool doubles + max vector work
        size_t total_mat_doubles = 0;
        size_t max_lvl_vec = m_total_dim;
        std::vector<size_t> cur_dims(M);
        for (size_t w = 0; w < M; w++) {
            cur_dims[w] = dims[w];
        }

        size_t cnt = M;
        for (size_t l = 0; l < n_lvls; l++) {
            TreeLevel &lvl = m_levels[l];
            lvl.cnt_in = cnt;
            lvl.npairs = cnt / 2;
            lvl.has_tail = ((cnt & 1) != 0);

            lvl.in_dims = (size_t *)realloc(lvl.in_dims, cnt * sizeof(size_t));
            lvl.even_offs = (size_t *)realloc(lvl.even_offs, (lvl.npairs + 1) * sizeof(size_t));
            lvl.odd_offs = (size_t *)realloc(lvl.odd_offs, (cnt + 1) * sizeof(size_t));
            lvl.B_in = (double **)realloc(lvl.B_in, cnt * sizeof(double *));
            lvl.B_prod = (double **)realloc(lvl.B_prod, std::max<size_t>(1, lvl.npairs) * sizeof(double *));

            size_t vec_len = 0;
            for (size_t i = 0; i < cnt; i++) {
                lvl.in_dims[i] = cur_dims[i];
                vec_len += cur_dims[i];
            }
            if (vec_len > max_lvl_vec) {
                max_lvl_vec = vec_len;
            }
            for (size_t j = 0; j < lvl.npairs; j++) {
                const size_t prev_even = (j == 0) ? (cnt - 1) : (2 * j - 1);
                total_mat_doubles += cur_dims[2 * j + 1] * cur_dims[prev_even];
            }

            std::vector<size_t> next_dims(lvl.npairs + (lvl.has_tail ? 1 : 0));
            for (size_t j = 0; j < lvl.npairs; j++) {
                next_dims[j] = cur_dims[2 * j + 1];
            }
            if (lvl.has_tail) {
                next_dims[lvl.npairs] = cur_dims[cnt - 1];
            }
            cur_dims.swap(next_dims);
            cnt = cur_dims.size();
        }
        m_root_dim = cur_dims[0];

        if (m_tree_mem_cap < total_mat_doubles) {
            m_tree_mem = (double *)realloc(m_tree_mem, std::max<size_t>(1, total_mat_doubles) * sizeof(double));
            m_tree_mem_cap = total_mat_doubles;
        }
        const size_t lvl_vec_tot = (n_lvls + 1) * m_total_dim;
        if (m_vec_work_cap < lvl_vec_tot) {
            m_f_work = (double *)realloc(m_f_work, std::max<size_t>(1, lvl_vec_tot) * sizeof(double));
            m_u_work = (double *)realloc(m_u_work, std::max<size_t>(1, lvl_vec_tot) * sizeof(double));
            m_vec_work_cap = lvl_vec_tot;
        }

        // Second pass: assign pointers and compute product matrices level-by-level in parallel
        double *pool_ptr = m_tree_mem;
        std::vector<const double *> active_B(M);
        for (size_t w = 0; w < M; w++) {
            active_B[w] = B_ptrs[w];
        }

        OpenBLASThreadContext __ob_ctx(1);
        for (size_t l = 0; l < n_lvls; l++) {
            TreeLevel &lvl = m_levels[l];
            for (size_t i = 0; i < lvl.cnt_in; i++) {
                lvl.B_in[i] = (double *)active_B[i];
            }
            for (size_t j = 0; j < lvl.npairs; j++) {
                const size_t prev_even = (j == 0) ? (lvl.cnt_in - 1) : (2 * j - 1);
                const size_t r = lvl.in_dims[2 * j + 1];
                const size_t c = lvl.in_dims[prev_even];
                lvl.B_prod[j] = pool_ptr;
                pool_ptr += r * c;
            }

            const size_t npairs = lvl.npairs;
#pragma omp parallel for schedule(dynamic, 1) if (npairs >= 2)
            for (size_t j = 0; j < npairs; j++) {
                const size_t prev_even = (j == 0) ? (lvl.cnt_in - 1) : (2 * j - 1);
                const int r = (int)lvl.in_dims[2 * j + 1];
                const int inner = (int)lvl.in_dims[2 * j];
                const int c = (int)lvl.in_dims[prev_even];
                dgemm("N", "N", r, c, inner, 1.0, active_B[2 * j + 1], r, active_B[2 * j], inner, 0.0, lvl.B_prod[j], r);
            }

            std::vector<const double *> next_B(npairs + (lvl.has_tail ? 1 : 0));
            for (size_t j = 0; j < npairs; j++) {
                next_B[j] = lvl.B_prod[j];
            }
            if (lvl.has_tail) {
                next_B[npairs] = active_B[lvl.cnt_in - 1];
            }
            active_B.swap(next_B);
        }

        // Factor root system (I - B_root)
        const size_t rd = m_root_dim;
        if (m_root_cap < rd * rd) {
            m_root_mat = (double *)realloc(m_root_mat, rd * rd * sizeof(double));
            m_root_ipiv = (int *)realloc(m_root_ipiv, rd * sizeof(int));
            m_root_cap = rd * rd;
        }
        const double *B_root = active_B[0];
        for (size_t col = 0; col < rd; col++) {
            for (size_t row = 0; row < rd; row++) {
                m_root_mat[col * rd + row] = ((row == col) ? 1.0 : 0.0) - B_root[col * rd + row];
            }
        }
        int info = 0;
        {
            OpenBLASThreadContext __ob_lu((int)std::max(1, omp_get_max_threads()));
            dgetrf((int)rd, (int)rd, m_root_mat, (int)rd, m_root_ipiv, &info);
        }
        m_ready = (info == 0);
    }

    void CyclicWindowTree::apply(const RP(double) f, RP(double) u) const
    {
        if (!m_ready || m_M == 0) {
            if (u != f && m_total_dim > 0) {
                memcpy(u, f, m_total_dim * sizeof(double));
            }
            return;
        }

        OpenBLASThreadContext __ob_ctx(1);

        // Level 0 RHS stored in m_f_work[0 .. m_total_dim)
        double *f_lvl0 = m_f_work;
        memcpy(f_lvl0, f, m_total_dim * sizeof(double));

        std::vector<size_t> cur_offs(m_M + 1);
        for (size_t w = 0; w <= m_M; w++) {
            cur_offs[w] = m_win_offs[w];
        }

        // Track level offsets for back-substitution
        std::vector<double *> lvl_f_ptrs(m_nlevels + 1);
        std::vector<double *> lvl_u_ptrs(m_nlevels + 1);
        std::vector<std::vector<size_t>> all_offs(m_nlevels + 1);
        lvl_f_ptrs[0] = m_f_work;
        lvl_u_ptrs[0] = m_u_work;
        all_offs[0] = cur_offs;

        double *f_cursor = m_f_work + m_total_dim;
        double *u_cursor = m_u_work + m_total_dim;

        // Forward reduction sweep
        for (size_t l = 0; l < m_nlevels; l++) {
            const TreeLevel &lvl = m_levels[l];
            const size_t n_out = lvl.npairs + (lvl.has_tail ? 1 : 0);
            std::vector<size_t> next_offs(n_out + 1, 0);
            for (size_t j = 0; j < lvl.npairs; j++) {
                next_offs[j + 1] = next_offs[j] + lvl.in_dims[2 * j + 1];
            }
            if (lvl.has_tail) {
                next_offs[n_out] = next_offs[lvl.npairs] + lvl.in_dims[lvl.cnt_in - 1];
            }

            const double *f_in = lvl_f_ptrs[l];
            double *f_out = f_cursor;
            lvl_f_ptrs[l + 1] = f_out;
            lvl_u_ptrs[l + 1] = u_cursor;
            all_offs[l + 1] = next_offs;

            const size_t npairs = lvl.npairs;
#pragma omp parallel for schedule(dynamic, 1) if (npairs >= 2)
            for (size_t j = 0; j < npairs; j++) {
                const int r = (int)lvl.in_dims[2 * j + 1];
                const int inner = (int)lvl.in_dims[2 * j];
                const double *f_even = &f_in[all_offs[l][2 * j]];
                const double *f_odd = &f_in[all_offs[l][2 * j + 1]];
                double *dst = &f_out[next_offs[j]];
                memcpy(dst, f_odd, (size_t)r * sizeof(double));
                dgemv("N", r, inner, 1.0, lvl.B_in[2 * j + 1], r, f_even, 1, 1.0, dst, 1);
            }
            if (lvl.has_tail) {
                memcpy(&f_out[next_offs[npairs]], &f_in[all_offs[l][lvl.cnt_in - 1]], lvl.in_dims[lvl.cnt_in - 1] * sizeof(double));
            }

            f_cursor += next_offs[n_out];
            u_cursor += next_offs[n_out];
        }

        // Solve root (I - B_root) u_root = f_root
        double *u_root = lvl_u_ptrs[m_nlevels];
        const double *f_root = lvl_f_ptrs[m_nlevels];
        memcpy(u_root, f_root, m_root_dim * sizeof(double));
        int info = 0;
        dgetrs("N", (int)m_root_dim, 1, m_root_mat, (int)m_root_dim, m_root_ipiv, u_root, (int)m_root_dim, &info);

        // Backward substitution sweep
        for (size_t l = m_nlevels; l-- > 0;) {
            const TreeLevel &lvl = m_levels[l];
            const double *u_in = lvl_u_ptrs[l + 1];
            const double *f_out_prev = lvl_f_ptrs[l];
            double *u_out = lvl_u_ptrs[l];
            const auto &offs_cur = all_offs[l];
            const auto &offs_next = all_offs[l + 1];

            const size_t npairs = lvl.npairs;
            for (size_t j = 0; j < npairs; j++) {
                memcpy(&u_out[offs_cur[2 * j + 1]], &u_in[offs_next[j]], lvl.in_dims[2 * j + 1] * sizeof(double));
            }
            if (lvl.has_tail) {
                memcpy(&u_out[offs_cur[lvl.cnt_in - 1]], &u_in[offs_next[npairs]], lvl.in_dims[lvl.cnt_in - 1] * sizeof(double));
            }

#pragma omp parallel for schedule(dynamic, 1) if (npairs >= 2)
            for (size_t j = 0; j < npairs; j++) {
                const size_t prev_even = (j == 0) ? (lvl.cnt_in - 1) : (2 * j - 1);
                const int r = (int)lvl.in_dims[2 * j];
                const int c = (int)lvl.in_dims[prev_even];
                double *dst = &u_out[offs_cur[2 * j]];
                memcpy(dst, &f_out_prev[offs_cur[2 * j]], (size_t)r * sizeof(double));
                dgemv("N", r, c, 1.0, lvl.B_in[2 * j], r, &u_out[offs_cur[prev_even]], 1, 1.0, dst, 1);
            }
        }

        memcpy(u, lvl_u_ptrs[0], m_total_dim * sizeof(double));
    }

    // =========================================================================
    // TwoLevelTreePrecond
    // =========================================================================

    TwoLevelTreePrecond::~TwoLevelTreePrecond()
    {
        freeBuffers();
    }

    void TwoLevelTreePrecond::invalidate()
    {
        m_ready = false;
        m_nt = 0;
        m_M = 0;
        m_tree.invalidate();
        m_smoother.invalidate();
    }

    void TwoLevelTreePrecond::freeBuffers()
    {
        m_tree.freeBuffers();
        m_smoother.freeBuffers();
        if (m_c_slices) { delete[] m_c_slices; m_c_slices = nullptr; }
        if (m_R_slice) { delete[] m_R_slice; m_R_slice = nullptr; }
        if (m_C_trans) { delete[] m_C_trans; m_C_trans = nullptr; }
        if (m_sig_t) {
            for (size_t i = 0; i < m_slice_cap; i++) {
                if (m_sig_t[i]) free(m_sig_t[i]);
            }
            free(m_sig_t);
            m_sig_t = nullptr;
        }
        if (m_sig_nnodes) { free(m_sig_nnodes); m_sig_nnodes = nullptr; }
        if (m_sig_cap) { free(m_sig_cap); m_sig_cap = nullptr; }
        m_slice_cap = 0;

        if (m_R_win) { delete[] m_R_win; m_R_win = nullptr; }
        if (m_P_win) { delete[] m_P_win; m_P_win = nullptr; }
        if (m_win_fine_offs) { free(m_win_fine_offs); m_win_fine_offs = nullptr; }
        if (m_win_coarse_offs) { free(m_win_coarse_offs); m_win_coarse_offs = nullptr; }
        if (m_win_coarse_dims) { free(m_win_coarse_dims); m_win_coarse_dims = nullptr; }
        if (m_win_dtau) { free(m_win_dtau); m_win_dtau = nullptr; }
        if (m_win_fine_ptrs) { free(m_win_fine_ptrs); m_win_fine_ptrs = nullptr; }
        if (m_B_ptrs) { free(m_B_ptrs); m_B_ptrs = nullptr; }
        m_win_cap = 0;

        if (m_c_mats) { delete[] m_c_mats; m_c_mats = nullptr; }
        if (m_c_cmats) { delete[] m_c_cmats; m_c_cmats = nullptr; }
        m_c_mats_cap = 0;
        if (m_fjac_buf) { free(m_fjac_buf); m_fjac_buf = nullptr; m_fjac_cap = 0; }
        if (m_rhs_c_buf) { free(m_rhs_c_buf); m_rhs_c_buf = nullptr; m_rhs_c_cap = 0; }
        if (m_crhs_buf) { free(m_crhs_buf); m_crhs_buf = nullptr; m_crhs_cap = 0; }
        if (m_C_pool) { free(m_C_pool); m_C_pool = nullptr; m_C_pool_cap = 0; }
        if (m_B_pool) { free(m_B_pool); m_B_pool = nullptr; m_B_pool_cap = 0; }
        if (m_win_y_end) { free(m_win_y_end); m_win_y_end = nullptr; m_win_y_cap = 0; }
        if (m_tmp_per_thread) { free(m_tmp_per_thread); m_tmp_per_thread = nullptr; m_tmp_thread_cap = 0; }
        if (m_apply_work) { free(m_apply_work); m_apply_work = nullptr; m_apply_work_cap = 0; }

        m_nt = 0;
        m_M = 0;
        m_W_end = 0;
        m_C_end = 0;
        m_ready = false;
    }

    void TwoLevelTreePrecond::ensureSliceStorage(size_t nt, size_t M)
    {
        if (m_slice_cap < nt) {
            delete[] m_c_slices;
            delete[] m_R_slice;
            delete[] m_C_trans;
            m_c_slices = new SpatialSlice[nt];
            m_R_slice = new SparseDenseOutputOp[nt];
            m_C_trans = new SparseDenseOutputOp[nt];
            m_sig_nnodes = (size_t *)realloc(m_sig_nnodes, nt * sizeof(size_t));
            if (m_sig_t) {
                m_sig_t = (double **)realloc(m_sig_t, nt * sizeof(double *));
            }
            else {
                m_sig_t = (double **)malloc(nt * sizeof(double *));
            }
            m_sig_cap = (size_t *)realloc(m_sig_cap, nt * sizeof(size_t));
            for (size_t i = m_slice_cap; i < nt; i++) {
                m_sig_nnodes[i] = 0;
                m_sig_t[i] = nullptr;
                m_sig_cap[i] = 0;
            }
            for (size_t i = 0; i < m_slice_cap; i++) {
                m_sig_nnodes[i] = 0;
            }
            m_slice_cap = nt;
        }
        if (m_c_mats_cap < nt) {
            delete[] m_c_mats;
            delete[] m_c_cmats;
            m_c_mats = new DEV::Matrix[nt];
            m_c_cmats = new DEV::ComplexMatrix[nt];
            m_c_mats_cap = nt;
        }
        if (m_win_cap < M) {
            delete[] m_R_win;
            delete[] m_P_win;
            m_R_win = new SparseDenseOutputOp[M];
            m_P_win = new SparseDenseOutputOp[M];
            m_win_fine_offs = (size_t *)realloc(m_win_fine_offs, (M + 1) * sizeof(size_t));
            m_win_coarse_offs = (size_t *)realloc(m_win_coarse_offs, (M + 1) * sizeof(size_t));
            m_win_coarse_dims = (size_t *)realloc(m_win_coarse_dims, M * sizeof(size_t));
            m_win_dtau = (double *)realloc(m_win_dtau, M * sizeof(double));
            m_win_fine_ptrs = (const SpatialSlice **)realloc(m_win_fine_ptrs, M * sizeof(const SpatialSlice *));
            m_B_ptrs = (double **)realloc(m_B_ptrs, M * sizeof(double *));
            m_win_cap = M;
        }
    }

    void TwoLevelTreePrecond::update(size_t nt, size_t M, const size_t *w_start, const size_t *w_end,
                                     const SpatialSlice *slices, size_t st, const double *y_stages,
                                     const double *dtau, double T, const double *p, bool smoother_bdf2)
    {
        m_ready = false;
        if (nt == 0 || M == 0) {
            return;
        }
        const bool grid_reset = (m_nt != nt) || (m_M != M);
        m_nt = nt;
        m_M = M;
        ensureSliceStorage(nt, M);

        const size_t node = slices[0].node;
        m_real_bc.makePeriodic((ptrdiff_t)node, 0);
        m_complex_bc.makePeriodic((ptrdiff_t)node, 0);
        const size_t coarse_spatial_s = (size_t)std::clamp(read_positive_int_env("TWIST_HOPF_COARSE_SPATIAL_S", 1), 1, (int)slices[0].nstages);
        const size_t coarse_time_st = (size_t)std::clamp(read_positive_int_env("TWIST_HOPF_COARSE_ST", 2), 1, 2);

        size_t max_nsub = 0;
        size_t max_nc = 0;
        size_t sum_c_blk = 0;
        size_t sum_b_blk = 0;

        bool any_slice_changed = grid_reset;
        for (size_t n = 0; n < nt; n++) {
            const SpatialSlice &fs = slices[n];
            const bool changed = grid_reset || (m_sig_nnodes[n] != fs.nnodes) ||
                (m_sig_t[n] == nullptr) || (memcmp(m_sig_t[n], fs.t, fs.nnodes * sizeof(double)) != 0) ||
                (m_c_slices[n].nstages != coarse_spatial_s);
            if (changed) {
                any_slice_changed = true;
                m_c_slices[n].initCoarseFrom(fs, coarse_spatial_s);
                m_R_slice[n].build(fs, m_c_slices[n]);
                m_sig_nnodes[n] = fs.nnodes;
                if (m_sig_cap[n] < fs.nnodes) {
                    m_sig_t[n] = (double *)realloc(m_sig_t[n], fs.nnodes * sizeof(double));
                    m_sig_cap[n] = fs.nnodes;
                }
                memcpy(m_sig_t[n], fs.t, fs.nnodes * sizeof(double));
            }
            else {
                // Keep mass matrix synced to current (c, sps) without full grid rebuild
                m_c_slices[n].m_nnz = fs.m_nnz;
                if (fs.m_nnz > 0) {
                    memcpy(m_c_slices[n].m_rows, fs.m_rows, fs.m_nnz * sizeof(int));
                    memcpy(m_c_slices[n].m_cols, fs.m_cols, fs.m_nnz * sizeof(int));
                    memcpy(m_c_slices[n].m_vals, fs.m_vals, fs.m_nnz * sizeof(double));
                }
            }
            max_nsub = std::max(max_nsub, m_c_slices[n].nSub());
            max_nc = std::max(max_nc, m_c_slices[n].n_space);
        }

        std::vector<size_t> c_offs(nt + 1, 0);
        std::vector<size_t> fine_stage_offs(nt + 1, 0);
        std::vector<size_t> fine_end_offs(nt + 1, 0);
        size_t max_nf = 0;
        for (size_t n = 0; n < nt; n++) {
            const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
            if (any_slice_changed) {
                m_C_trans[n].build(m_c_slices[prev], m_c_slices[n]);
            }
            c_offs[n] = sum_c_blk;
            sum_c_blk += m_c_slices[n].n_space * m_c_slices[prev].n_space;
            fine_end_offs[n + 1] = fine_end_offs[n] + slices[n].n_space;
            fine_stage_offs[n + 1] = fine_stage_offs[n] + st * slices[n].n_space;
            max_nf = std::max(max_nf, slices[n].n_space);
        }
        c_offs[nt] = sum_c_blk;

        // Window boundary mappings
        m_win_fine_offs[0] = 0;
        m_win_coarse_offs[0] = 0;
        std::vector<size_t> b_offs(M + 1, 0);
        for (size_t w = 0; w < M; w++) {
            const size_t e = w_end[w] - 1;
            const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
            const size_t pe = w_end[prev_w] - 1;
            if (any_slice_changed) {
                m_R_win[w].build(slices[e], m_c_slices[e]);
                m_P_win[w].build(m_c_slices[e], slices[e]);
            }
            m_win_fine_ptrs[w] = &slices[e];
            m_win_coarse_dims[w] = m_c_slices[e].n_space;
            m_win_fine_offs[w + 1] = m_win_fine_offs[w] + slices[e].n_space;
            m_win_coarse_offs[w + 1] = m_win_coarse_offs[w] + m_c_slices[e].n_space;
            double dt_sum = 0.0;
            for (size_t n = w_start[w]; n < w_end[w]; n++) {
                dt_sum += dtau[n];
            }
            m_win_dtau[w] = dt_sum;
            b_offs[w] = sum_b_blk;
            sum_b_blk += m_c_slices[e].n_space * m_c_slices[pe].n_space;
        }
        m_W_end = m_win_fine_offs[M];
        m_C_end = m_win_coarse_offs[M];

        // Buffers
        const size_t max_threads = (size_t)std::max(1, omp_get_max_threads());
        const size_t fjac_stride = max_nsub * coarse_spatial_s * node * node;
        if (m_fjac_cap < nt * fjac_stride) {
            m_fjac_buf = (double *)realloc(m_fjac_buf, nt * fjac_stride * sizeof(double));
            m_fjac_cap = nt * fjac_stride;
        }
        const size_t needed_rhs_c = nt * max_nc * max_nc + max_threads * max_nf;
        if (m_rhs_c_cap < needed_rhs_c) {
            m_rhs_c_buf = (double *)realloc(m_rhs_c_buf, needed_rhs_c * sizeof(double));
            m_rhs_c_cap = needed_rhs_c;
        }
        if (m_crhs_cap < nt * 2 * max_nc * max_nc) {
            m_crhs_buf = (std::complex<double> *)realloc(m_crhs_buf, nt * 2 * max_nc * max_nc * sizeof(std::complex<double>));
            m_crhs_cap = nt * 2 * max_nc * max_nc;
        }
        if (m_C_pool_cap < sum_c_blk) {
            m_C_pool = (double *)realloc(m_C_pool, sum_c_blk * sizeof(double));
            m_C_pool_cap = sum_c_blk;
        }
        if (m_B_pool_cap < sum_b_blk) {
            m_B_pool = (double *)realloc(m_B_pool, sum_b_blk * sizeof(double));
            m_B_pool_cap = sum_b_blk;
        }
        const size_t t_stride = 4 * max_nc + max_nc * max_nc + 32;
        if (m_tmp_thread_cap < max_threads * t_stride) {
            m_tmp_per_thread = (double *)realloc(m_tmp_per_thread, max_threads * t_stride * sizeof(double));
            m_tmp_thread_cap = max_threads * t_stride;
        }
        if (m_win_y_cap < m_W_end) {
            m_win_y_end = (double *)realloc(m_win_y_end, m_W_end * sizeof(double));
            m_win_y_cap = m_W_end;
        }
        for (size_t w = 0; w < M; w++) {
            const size_t e = w_end[w] - 1;
            const size_t ns_e = slices[e].n_space;
            memcpy(&m_win_y_end[m_win_fine_offs[w]], &y_stages[fine_stage_offs[e] + (st - 1) * ns_e], ns_e * sizeof(double));
        }

        // Update smoother on window boundaries
        const char *env_sm = std::getenv("TWIST_HOPF_COARSE_SMOOTHER");
        const bool use_smoother = !(env_sm && env_sm[0] == '0' && env_sm[1] == '\0');
        if (use_smoother) {
            m_smoother.update(M, m_win_fine_ptrs, nullptr, m_win_y_end, m_win_dtau, T, p, 1.0, smoother_bdf2);
        }

        // Assemble coarse Radau-2 (3rd-order L-stable) or BE-1 transfer C_n^(c) per slice in parallel
        OpenBLASThreadContext __ob_ctx(1);
        const size_t node2 = node * node;
        const RadauDGMethod &radau2 = getRadauDG(2);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t n = 0; n < nt; n++) {
            const size_t tid = (size_t)omp_get_thread_num();
            double *tbuf = m_tmp_per_thread + tid * t_stride;
            double *y_c = tbuf;
            double *y_f_avg = m_rhs_c_buf + nt * max_nc * max_nc + tid * max_nf;
            double *v_tmp = y_c + max_nc;
            double *e_vec = v_tmp + max_nc;

            const SpatialSlice &cs = m_c_slices[n];
            const SpatialSlice &fs = slices[n];
            const size_t ns_f = fs.n_space;
            const size_t nc = cs.n_space;
            const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
            const size_t nc_prev = m_c_slices[prev].n_space;
            const size_t n_sub = cs.nSub();
            const size_t sc = cs.nstages;
            const double dtau_T = dtau[n] * T;

            // Time-averaged fine state on slice n -> restrict onto coarse slice
            const double *yn_base = &y_stages[fine_stage_offs[n]];
            const double inv_st = 1.0 / (double)st;
            memset(y_f_avg, 0, ns_f * sizeof(double));
            for (size_t l = 0; l < st; l++) {
                const double *yl = &yn_base[l * ns_f];
                for (size_t i = 0; i < ns_f; i++) {
                    y_f_avg[i] += inv_st * yl[i];
                }
            }
            m_R_slice[n].apply(y_f_avg, y_c);

            // Evaluate Jacobian at the n_sub * sc coarse GL collocation points
            double *fj = &m_fjac_buf[n * fjac_stride];
            for (size_t k = 0; k < n_sub; k++) {
                const double *yk = &y_c[k * (sc + 1) * node + node];
                for (size_t q = 0; q < sc; q++) {
                    cs.fjac(&yk[q * node], p, &fj[(k * sc + q) * node2]);
                }
            }

            double *C_out = &m_C_pool[c_offs[n]];
            memset(C_out, 0, nc * nc_prev * sizeof(double));
            double *rhs_mat = &m_rhs_c_buf[n * max_nc * max_nc];

            std::vector<size_t> active_cols;
            size_t n_solve_cols = nc_prev;
            if (m_C_trans[n].is_identity) {
                bool var_active[node];
                memset(var_active, 0, node * sizeof(bool));
                for (size_t nz = 0; nz < cs.m_nnz; nz++) {
                    var_active[cs.m_cols[nz]] = true;
                }
                active_cols.reserve(n_sub * sc * node);
                for (size_t k = 0; k < n_sub; k++) {
                    for (size_t q = 0; q < sc; q++) {
                        for (size_t v = 0; v < node; v++) {
                            if (var_active[v]) {
                                active_cols.push_back((k * (sc + 1) + (q + 1)) * node + v);
                            }
                        }
                    }
                }
                n_solve_cols = active_cols.size();
            }

            if (n_solve_cols == 0) {
                continue;
            }

            const double rhs_scale = (coarse_time_st == 2) ? 1.0 : (-1.0 / dtau_T);
            if (m_C_trans[n].is_identity) {
                memset(rhs_mat, 0, nc * n_solve_cols * sizeof(double));
                for (size_t slot = 0; slot < n_solve_cols; slot++) {
                    const size_t col_j = active_cols[slot];
                    const size_t pt = col_j / node;
                    const size_t vc = col_j % node;
                    const size_t k = pt / (sc + 1);
                    const size_t i_s = (pt % (sc + 1)) - 1;
                    const size_t stride_k = k * (sc + 1) * node;
                    const double hk = cs.h[k];
                    double *dst = &rhs_mat[slot * nc];
                    for (size_t nz = 0; nz < cs.m_nnz; nz++) {
                        if ((size_t)cs.m_cols[nz] != vc) continue;
                        const size_t r = (size_t)cs.m_rows[nz];
                        const double s_mv = rhs_scale * hk * cs.m_vals[nz];
                        dst[stride_k + r] += s_mv * cs.c_b[i_s];
                        for (size_t jr = 0; jr < sc; jr++) {
                            dst[stride_k + (jr + 1) * node + r] += s_mv * cs.c_A[jr * sc + i_s];
                        }
                    }
                }
            }
            else {
                for (size_t j = 0; j < nc_prev; j++) {
                    memset(e_vec, 0, nc_prev * sizeof(double));
                    e_vec[j] = 1.0;
                    m_C_trans[n].apply(e_vec, v_tmp);
                    cs.applyMassOpSparse(cs.m_nnz, cs.m_rows, cs.m_cols, cs.m_vals, v_tmp, &rhs_mat[j * nc], rhs_scale, 0.0);
                }
            }

            if (coarse_time_st == 2) {
                // 2-stage Radau IIA (3rd order L-stable) via single complex banded matrix solve
                m_c_cmats[n].updateFromFjacs(fj, (ptrdiff_t)cs.nnodes, (ptrdiff_t)node, (ptrdiff_t)sc, cs.unscaled_spatial_period, cs.h, &m_complex_bc, false);
                const std::complex<double> scale(-radau2.complex_re[0] / dtau_T, +radau2.complex_im[0] / dtau_T);
                m_c_cmats[n].addMassBlockSparse(cs.m_nnz, cs.m_rows, cs.m_cols, cs.m_vals, cs.c_A, cs.c_b, cs.h, scale, false);
                m_c_cmats[n].factor();

                std::complex<double> *bc = &m_crhs_buf[n * 2 * max_nc * max_nc];
                std::complex<double> *zc = bc + max_nc * max_nc;
                const double g0 = -radau2.gamma_ones[0] / dtau_T;
                const double g1 = -radau2.gamma_ones[1] / dtau_T;
                const size_t total_elems = nc * n_solve_cols;
                for (size_t idx = 0; idx < total_elems; idx++) {
                    bc[idx] = std::complex<double>(g0 * rhs_mat[idx], g1 * rhs_mat[idx]);
                }
                if (n_solve_cols == 1) {
                    m_c_cmats[n].solve(bc, zc);
                }
                else {
                    m_c_cmats[n].solve(bc, zc, (ptrdiff_t)n_solve_cols);
                }
                const double t10 = radau2.T[1 * 2 + 0];
                const double t11 = radau2.T[1 * 2 + 1];
                if (m_C_trans[n].is_identity) {
                    for (size_t slot = 0; slot < n_solve_cols; slot++) {
                        double *dst = &C_out[active_cols[slot] * nc];
                        const std::complex<double> *zcol = &zc[slot * nc];
                        for (size_t i = 0; i < nc; i++) {
                            dst[i] = t10 * zcol[i].real() + t11 * zcol[i].imag();
                        }
                    }
                }
                else {
                    for (size_t j = 0; j < nc_prev; j++) {
                        double *dst = &C_out[j * nc];
                        const std::complex<double> *zcol = &zc[j * nc];
                        for (size_t i = 0; i < nc; i++) {
                            dst[i] = t10 * zcol[i].real() + t11 * zcol[i].imag();
                        }
                    }
                }
            }
            else {
                m_c_mats[n].updateFromFjacs(fj, (ptrdiff_t)cs.nnodes, (ptrdiff_t)node, (ptrdiff_t)sc, cs.unscaled_spatial_period, cs.h, &m_real_bc, false);
                m_c_mats[n].addMassBlockSparse(cs.m_nnz, cs.m_rows, cs.m_cols, cs.m_vals, cs.c_A, cs.c_b, cs.h, -1.0 / dtau_T, false);
                m_c_mats[n].factor();

                if (m_C_trans[n].is_identity) {
                    double *sol_mat = tbuf + 3 * max_nc;
                    if (n_solve_cols == 1) {
                        m_c_mats[n].solve(rhs_mat, sol_mat);
                    }
                    else {
                        m_c_mats[n].solve(rhs_mat, sol_mat, (ptrdiff_t)n_solve_cols);
                    }
                    for (size_t slot = 0; slot < n_solve_cols; slot++) {
                        memcpy(&C_out[active_cols[slot] * nc], &sol_mat[slot * nc], nc * sizeof(double));
                    }
                }
                else {
                    m_c_mats[n].solve(rhs_mat, C_out, (ptrdiff_t)nc_prev);
                }
            }
        }

        // Compose per-window coarse product B_w^(c) = C_{we}^(c) * ... * C_{ws}^(c) in parallel
#pragma omp parallel for schedule(dynamic, 1) if (M > 1)
        for (size_t w = 0; w < M; w++) {
            const size_t tid = (size_t)omp_get_thread_num();
            double *tbuf = m_tmp_per_thread + tid * t_stride;
            double *prod_a = tbuf + 3 * max_nc;
            double *prod_b = m_rhs_c_buf + w * max_nc * max_nc;

            const size_t s = w_start[w];
            const size_t e = w_end[w] - 1;
            const size_t s_prev = (s == 0) ? (nt - 1) : (s - 1);
            const size_t nc_in = m_c_slices[s_prev].n_space;
            size_t nc_cur = m_c_slices[s].n_space;

            memcpy(prod_a, &m_C_pool[c_offs[s]], nc_cur * nc_in * sizeof(double));
            double *src = prod_a;
            double *dst = prod_b;

            for (size_t n = s + 1; n <= e; n++) {
                const size_t nc_next = m_c_slices[n].n_space;
                dgemm("N", "N", (int)nc_next, (int)nc_in, (int)nc_cur, 1.0, &m_C_pool[c_offs[n]], (int)nc_next, src, (int)nc_cur, 0.0, dst, (int)nc_next);
                std::swap(src, dst);
                nc_cur = nc_next;
            }
            double *B_dst = &m_B_pool[b_offs[w]];
            memcpy(B_dst, src, nc_cur * nc_in * sizeof(double));
            m_B_ptrs[w] = B_dst;
        }

        m_tree.buildAndFactor(M, m_win_coarse_dims, m_B_ptrs);
        m_ready = m_tree.isReady();
    }

    void TwoLevelTreePrecond::apply(const RP(double) r, RP(double) z) const
    {
        if (!isReady()) {
            m_smoother.apply(r, z);
            return;
        }

        const size_t needed_w = 2 * m_C_end + 3 * m_W_end;
        if (m_apply_work_cap < needed_w) {
            m_apply_work = (double *)realloc(m_apply_work, needed_w * sizeof(double));
            m_apply_work_cap = needed_w;
        }
        double *r_c = m_apply_work;
        double *z_c = r_c + m_C_end;
        double *r_f = z_c + m_C_end;
        double *z_f = r_f + m_W_end;
        double *p_zc = z_f + m_W_end;

        const size_t M = m_M;
        // 1. Restrict fine residual r -> r_c and compute high-frequency complement r_f = r - P R r
#pragma omp parallel for schedule(static) if (M * m_C_end >= 4096)
        for (size_t w = 0; w < M; w++) {
            const size_t f_off = m_win_fine_offs[w];
            const size_t c_off = m_win_coarse_offs[w];
            const size_t nf = m_win_fine_offs[w + 1] - f_off;
            const double *rw = &r[f_off];
            double *rcw = &r_c[c_off];
            double *rfw = &r_f[f_off];
            m_R_win[w].apply(rw, rcw);
            m_P_win[w].apply(rcw, rfw);
            for (size_t i = 0; i < nf; i++) {
                rfw[i] = rw[i] - rfw[i];
            }
        }

        // 2. Coarse time-varying cyclic binary-tree solve + fine Schur high-freq smoother (or identity if disabled)
        m_tree.apply(r_c, z_c);
        const char *env_sm = std::getenv("TWIST_HOPF_COARSE_SMOOTHER");
        const bool use_smoother = !(env_sm && env_sm[0] == '0' && env_sm[1] == '\0');
        if (use_smoother) {
            m_smoother.apply(r_f, z_f);
        }
        else {
            memcpy(z_f, r_f, m_W_end * sizeof(double));
        }

        // 3. Combine: z = P * z_c + (I - P * R) * z_f (preserves R * z = z_c exact!)
#pragma omp parallel for schedule(static) if (M * m_C_end >= 4096)
        for (size_t w = 0; w < M; w++) {
            const size_t f_off = m_win_fine_offs[w];
            const size_t c_off = m_win_coarse_offs[w];
            const size_t nf = m_win_fine_offs[w + 1] - f_off;
            double *pzw = &p_zc[f_off];
            double *zfw = &z_f[f_off];
            double *zw = &z[f_off];
            double *tmp_c = &r_c[c_off];
            m_P_win[w].apply(&z_c[c_off], pzw);
            m_R_win[w].apply(zfw, tmp_c);
            m_P_win[w].apply(tmp_c, &r_f[f_off]);
            for (size_t i = 0; i < nf; i++) {
                zw[i] = pzw[i] + (zfw[i] - r_f[f_off + i]);
            }
        }
    }

    // =========================================================================
    // DirectTreePrecond
    // =========================================================================

    DirectTreePrecond::~DirectTreePrecond()
    {
        freeBuffers();
    }

    void DirectTreePrecond::invalidate()
    {
        m_ready = false;
        m_nt = 0;
        m_M = 0;
        m_tree.invalidate();
    }

    void DirectTreePrecond::freeBuffers()
    {
        m_tree.freeBuffers();
        if (m_c_pool) { free(m_c_pool); m_c_pool = nullptr; m_c_pool_cap = 0; }
        if (m_B_pool) { free(m_B_pool); m_B_pool = nullptr; m_B_pool_cap = 0; }
        if (m_tmp_pool) { free(m_tmp_pool); m_tmp_pool = nullptr; m_tmp_pool_cap = 0; }
        if (m_B_ptrs) { free(m_B_ptrs); m_B_ptrs = nullptr; }
        if (m_dims) { free(m_dims); m_dims = nullptr; }
        m_win_cap = 0;
        m_nt = 0;
        m_M = 0;
        m_ready = false;
    }

    void DirectTreePrecond::update(size_t nt, size_t M, const size_t *w_start, const size_t *w_end,
                                   const SpatialSlice *slices, const SubintervalCondenser *condensers,
                                   const SparseDenseOutputOp *transfers, bool fast_approx)
    {
        m_ready = false;
        if (nt == 0 || M == 0) {
            return;
        }
        m_nt = nt;
        m_M = M;
        if (m_win_cap < M) {
            m_B_ptrs = (double **)realloc(m_B_ptrs, M * sizeof(double *));
            m_dims = (size_t *)realloc(m_dims, M * sizeof(size_t));
            m_win_cap = M;
        }

        size_t sum_b = 0, max_ns = 0;
        std::vector<size_t> b_offs(M + 1, 0);
        for (size_t n = 0; n < nt; n++) {
            max_ns = std::max(max_ns, slices[n].n_space);
        }
        for (size_t w = 0; w < M; w++) {
            const size_t e = w_end[w] - 1;
            const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
            const size_t pe = w_end[prev_w] - 1;
            m_dims[w] = slices[e].n_space;
            b_offs[w] = sum_b;
            sum_b += slices[e].n_space * slices[pe].n_space;
        }

        if (m_c_pool) {
            free(m_c_pool);
            m_c_pool = nullptr;
            m_c_pool_cap = 0;
        }
        if (m_B_pool_cap < sum_b) {
            m_B_pool = (double *)realloc(m_B_pool, sum_b * sizeof(double));
            m_B_pool_cap = sum_b;
        }
        const size_t chunk_max = 32;
        const size_t needed_tmp = M * 2 * max_ns * chunk_max;
        if (m_tmp_pool_cap < needed_tmp) {
            m_tmp_pool = (double *)realloc(m_tmp_pool, needed_tmp * sizeof(double));
            m_tmp_pool_cap = needed_tmp;
        }

        const bool exact_jac = !fast_approx;
        const double rtol_eff = fast_approx ? 1e-3 : 1e-12;
        OpenBLASThreadContext __ob_ctx(1);
#pragma omp parallel for schedule(dynamic, 1) if (M > 1)
        for (size_t w = 0; w < M; w++) {
            const size_t s = w_start[w];
            const size_t e = w_end[w] - 1;
            const size_t s_prev = (s == 0) ? (nt - 1) : (s - 1);
            const size_t ns_in = slices[s_prev].n_space;
            const size_t ns_e = slices[e].n_space;
            double *B_dst = &m_B_pool[b_offs[w]];

            double *cur = &m_tmp_pool[w * 2 * max_ns * chunk_max];
            double *nxt = cur + max_ns * chunk_max;

            for (size_t c0 = 0; c0 < ns_in; c0 += chunk_max) {
                const size_t K = std::min(chunk_max, ns_in - c0);
                memset(cur, 0, max_ns * K * sizeof(double));
                for (size_t col = 0; col < K; col++) {
                    cur[col * max_ns + c0 + col] = 1.0;
                }
                for (size_t n = s; n <= e; n++) {
                    condensers[n].applyCondensedCBatch(K, cur, max_ns, transfers[n], nxt, max_ns, exact_jac, rtol_eff);
                    std::swap(cur, nxt);
                }
                for (size_t col = 0; col < K; col++) {
                    memcpy(&B_dst[(c0 + col) * ns_e], &cur[col * max_ns], ns_e * sizeof(double));
                }
            }
            m_B_ptrs[w] = B_dst;
        }

        m_tree.buildAndFactor(M, m_dims, m_B_ptrs);
        m_ready = m_tree.isReady();
    }

    void DirectTreePrecond::remapSpatial(size_t nt, size_t M, const size_t *w_end,
                                         const SpatialSlice *old_end_slices, const SpatialSlice *new_slices)
    {
        if (!m_ready || nt == 0 || M == 0 || M != m_M || nt != m_nt) {
            m_ready = false;
            return;
        }
        std::vector<size_t> new_dims(M, 0), new_offs(M + 1, 0), old_offs(M + 1, 0);
        size_t sum_new = 0, max_T = 0;
        for (size_t w = 0; w < M; w++) {
            const size_t e = w_end[w] - 1;
            const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
            const size_t pe = w_end[prev_w] - 1;
            new_dims[w] = new_slices[e].n_space;
            new_offs[w] = sum_new;
            sum_new += new_slices[e].n_space * new_slices[pe].n_space;
            old_offs[w + 1] = old_offs[w] + old_end_slices[w].n_space * old_end_slices[prev_w].n_space;
            max_T = std::max(max_T, new_slices[e].n_space * old_end_slices[prev_w].n_space);
        }

        std::vector<double> new_B(sum_new, 0.0);
        OpenBLASThreadContext __ob_ctx(1);
#pragma omp parallel for schedule(dynamic, 1) if (M > 1)
        for (size_t w = 0; w < M; w++) {
            const size_t e = w_end[w] - 1;
            const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
            const size_t pe = w_end[prev_w] - 1;
            const SpatialSlice &old_e = old_end_slices[w];
            const SpatialSlice &old_pe = old_end_slices[prev_w];
            const SpatialSlice &new_e = new_slices[e];
            const SpatialSlice &new_pe = new_slices[pe];
            const size_t n_eo = old_e.n_space, n_po = old_pe.n_space;
            const size_t n_en = new_e.n_space, n_pn = new_pe.n_space;
            const double *B_old = &m_B_pool[old_offs[w]];
            double *B_dst = &new_B[new_offs[w]];

            SparseDenseOutputOp P_out, R_in;
            P_out.build(old_e, new_e);
            R_in.build(new_pe, old_pe);

            std::vector<double> T_buf(n_en * n_po);
            for (size_t j = 0; j < n_po; j++) {
                P_out.apply(&B_old[j * n_eo], &T_buf[j * n_en]);
            }

            if (R_in.is_identity) {
                memcpy(B_dst, T_buf.data(), n_en * n_pn * sizeof(double));
            }
            else {
                const size_t node = R_in.node;
                const size_t st1 = R_in.nstages + 1;
                for (size_t i_pt = 0; i_pt < R_in.dst_npts; i_pt++) {
                    const size_t src_base = (size_t)R_in.src_sub[i_pt] * st1;
                    const double *wts = &R_in.weights[i_pt * st1];
                    for (size_t q = 0; q < st1; q++) {
                        const double wq = wts[q];
                        if (wq == 0.0) continue;
                        const size_t src_pt = src_base + q;
                        for (size_t c = 0; c < node; c++) {
                            const double *T_col = &T_buf[(i_pt * node + c) * n_en];
                            double *B_col = &B_dst[(src_pt * node + c) * n_en];
                            for (size_t r = 0; r < n_en; r++) {
                                B_col[r] += wq * T_col[r];
                            }
                        }
                    }
                }
            }
        }

        if (m_B_pool_cap < sum_new) {
            m_B_pool = (double *)realloc(m_B_pool, sum_new * sizeof(double));
            m_B_pool_cap = sum_new;
        }
        memcpy(m_B_pool, new_B.data(), sum_new * sizeof(double));
        for (size_t w = 0; w < M; w++) {
            m_dims[w] = new_dims[w];
            m_B_ptrs[w] = &m_B_pool[new_offs[w]];
        }
        m_tree.buildAndFactor(M, m_dims, m_B_ptrs);
        m_ready = m_tree.isReady();
    }

    void DirectTreePrecond::apply(const RP(double) r, RP(double) z) const
    {
        m_tree.apply(r, z);
    }
} // namespace TWIST
