#pragma once

#include "colmat/tcolmat.h"
#include "modulating/spacetime_precond.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include "shared.h"
#include <cstddef>

namespace TWIST
{
    // Parallel O(log M) cyclic binary-tree direct solver for M coupled window transfer blocks:
    //   u_w - B_w * u_{prev(w)} = f_w,   w = 0..M-1, prev(0) = M-1.
    // Supports arbitrary block dimensions dims[w] (adaptive meshes) and odd/even M.
    class CyclicWindowTree
    {
    public:
        CyclicWindowTree() = default;
        CyclicWindowTree(const CyclicWindowTree &) = delete;
        CyclicWindowTree &operator=(const CyclicWindowTree &) = delete;
        ~CyclicWindowTree();

        void freeBuffers();
        void invalidate() { m_M = 0; m_ready = false; }

        // Builds & factors the binary reduction tree from M column-major matrices B_w (dims[w] x dims[prev(w)]).
        void buildAndFactor(size_t M, const size_t *dims, const double *const *B_ptrs);

        // Solves for u (concatenated dims[0..M-1]) from rhs f (concatenated dims[0..M-1]).
        // `f` and `u` may alias.
        void apply(const RP(double) f, RP(double) u) const;

        bool isReady() const { return m_ready; }
        size_t numWindows() const { return m_M; }
        size_t totalDim() const { return m_total_dim; }

    private:
        struct TreeLevel
        {
            size_t cnt_in = 0;
            size_t npairs = 0;
            bool has_tail = false;
            size_t *in_dims = nullptr;
            size_t *even_offs = nullptr;
            size_t *odd_offs = nullptr;
            double **B_in = nullptr;     // B_in[i]: in_dims[i] x in_dims[prev(i)]
            double **B_prod = nullptr;   // B_prod[j]: in_dims[2*j+1] x in_dims[prev(2*j)]
        };

        void resetLevels();

        size_t m_M = 0;
        size_t m_total_dim = 0;
        bool m_ready = false;

        size_t *m_win_offs = nullptr;
        size_t *m_win_dims = nullptr;
        size_t m_win_cap = 0;

        TreeLevel *m_levels = nullptr;
        size_t m_nlevels = 0;
        size_t m_levels_cap = 0;

        double *m_tree_mem = nullptr; // backing pool for all level matrices + level vectors
        size_t m_tree_mem_cap = 0;

        double *m_root_mat = nullptr;
        int *m_root_ipiv = nullptr;
        size_t m_root_dim = 0;
        size_t m_root_cap = 0;

        mutable double *m_f_work = nullptr; // per-level forward/back vectors
        mutable double *m_u_work = nullptr;
        mutable size_t m_vec_work_cap = 0;
    };

    // Two-level p-coarsened time-varying preconditioner across M shooting windows:
    //   M_two_lvl(r) = P * A_coarse^(-1) * (R * r) + Q_f * M_smooth^(-1) * (Q_f * r),
    // where A_coarse is the time-varying implicit Euler condensed window tree at coarse GL degree
    // (nstages_c = 1, st_c = 1), R/P are exact Lagrange restriction/prolongation (R * P = I),
    // Q_f = I - P * R, and M_smooth is a BDF2/BE RealSchurSpaceTimePrecond smoothing high spatial modes.
    class TwoLevelTreePrecond
    {
    public:
        TwoLevelTreePrecond() = default;
        TwoLevelTreePrecond(const TwoLevelTreePrecond &) = delete;
        TwoLevelTreePrecond &operator=(const TwoLevelTreePrecond &) = delete;
        ~TwoLevelTreePrecond();

        void freeBuffers();
        void invalidate();

        void update(size_t nt, size_t M, const size_t *w_start, const size_t *w_end,
                    const SpatialSlice *slices, size_t st, const double *y_stages,
                    const double *dtau, double T, const double *p, bool smoother_bdf2 = true);

        void apply(const RP(double) r, RP(double) z) const;

        bool isReady() const { return m_ready && m_tree.isReady(); }
        size_t nt() const { return m_nt; }
        size_t numWindows() const { return m_M; }

    private:
        void ensureSliceStorage(size_t nt, size_t M);

        size_t m_nt = 0;
        size_t m_M = 0;
        size_t m_W_end = 0;
        size_t m_C_end = 0;
        bool m_ready = false;

        SpatialSlice *m_c_slices = nullptr;      // nt coarse spatial slices (nstages_c = 1)
        SparseDenseOutputOp *m_R_slice = nullptr; // fine slice n -> coarse slice n (for state restriction)
        SparseDenseOutputOp *m_C_trans = nullptr; // coarse slice prev(n) -> coarse slice n
        size_t *m_sig_nnodes = nullptr;
        double **m_sig_t = nullptr;
        size_t *m_sig_cap = nullptr;
        size_t m_slice_cap = 0;

        SparseDenseOutputOp *m_R_win = nullptr;   // fine window end w -> coarse window end w
        SparseDenseOutputOp *m_P_win = nullptr;   // coarse window end w -> fine window end w
        size_t *m_win_fine_offs = nullptr;
        size_t *m_win_coarse_offs = nullptr;
        size_t *m_win_coarse_dims = nullptr;
        double *m_win_dtau = nullptr;
        const SpatialSlice **m_win_fine_ptrs = nullptr;
        size_t m_win_cap = 0;

        DEV::BoundaryConditions m_real_bc;
        DEV::ComplexBoundaryConditions m_complex_bc;
        DEV::Matrix *m_c_mats = nullptr;         // nt banded coarse matrices (st_c = 1)
        DEV::ComplexMatrix *m_c_cmats = nullptr; // nt banded coarse complex matrices (st_c = 2)
        size_t m_c_mats_cap = 0;

        double *m_fjac_buf = nullptr;
        size_t m_fjac_cap = 0;
        double *m_rhs_c_buf = nullptr;
        size_t m_rhs_c_cap = 0;
        std::complex<double> *m_crhs_buf = nullptr;
        size_t m_crhs_cap = 0;
        double *m_C_pool = nullptr;              // nt coarse condensed matrices C_n^(c)
        size_t m_C_pool_cap = 0;
        double *m_B_pool = nullptr;              // M coarse window product matrices B_w^(c)
        size_t m_B_pool_cap = 0;
        double **m_B_ptrs = nullptr;
        double *m_win_y_end = nullptr;
        size_t m_win_y_cap = 0;
        double *m_tmp_per_thread = nullptr;
        size_t m_tmp_thread_cap = 0;

        CyclicWindowTree m_tree;
        RealSchurSpaceTimePrecond m_smoother;

        mutable double *m_apply_work = nullptr;
        mutable size_t m_apply_work_cap = 0;
    };

    // Exact lagged direct window-tree preconditioner (experimental: gated by speed benchmark).
    class DirectTreePrecond
    {
    public:
        DirectTreePrecond() = default;
        DirectTreePrecond(const DirectTreePrecond &) = delete;
        DirectTreePrecond &operator=(const DirectTreePrecond &) = delete;
        ~DirectTreePrecond();

        void freeBuffers();
        void invalidate();

        void update(size_t nt, size_t M, const size_t *w_start, const size_t *w_end,
                    const SpatialSlice *slices, const SubintervalCondenser *condensers,
                    const SparseDenseOutputOp *transfers, bool fast_approx = true);

        void remapSpatial(size_t nt, size_t M, const size_t *w_end,
                          const SpatialSlice *old_end_slices, const SpatialSlice *new_slices);

        void apply(const RP(double) r, RP(double) z) const;

        bool isReady() const { return m_ready && m_tree.isReady(); }
        size_t nt() const { return m_nt; }
        size_t numWindows() const { return m_M; }

    private:
        size_t m_nt = 0;
        size_t m_M = 0;
        bool m_ready = false;

        double *m_c_pool = nullptr;
        size_t m_c_pool_cap = 0;
        double *m_B_pool = nullptr;
        size_t m_B_pool_cap = 0;
        double *m_tmp_pool = nullptr;
        size_t m_tmp_pool_cap = 0;
        double **m_B_ptrs = nullptr;
        size_t *m_dims = nullptr;
        size_t m_win_cap = 0;

        CyclicWindowTree m_tree;
    };
} // namespace TWIST
