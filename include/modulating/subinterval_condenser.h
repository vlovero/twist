#pragma once

#include "colmat/tcolmat.h"
#include "modulating/radau_dg.h"
#include "modulating/spatial_slice.h"
#include "shared.h"
#include <complex>
#include <cstddef>
#include <cstdint>

namespace TWIST
{
    // Maximum Krylov subspace dimension (restart) for local stage-solve GMRES fallback.
    // Initialized from TWIST_HOPF_LOCAL_GMRES_DIM (default 16).
    extern int subinterval_local_gmres_subspace_dim;

    enum class LocalStageSolverMode
    {
        HairerWanner,
        BlockGaussSeidel,
        Hybrid,
        Umfpack,
        HybridUmfpack
    };

    // Default mode initialized from TWIST_HOPF_LOCAL_SOLVER ({hw, bgs, hybrid, umfpack, hybrid_umf}).
    LocalStageSolverMode defaultLocalStageSolverMode();
    const char *localStageSolverModeName(LocalStageSolverMode mode);

    class SubintervalCondenser
    {
    public:
        using complex_t = std::complex<double>;

        SubintervalCondenser() = default;
        SubintervalCondenser(const SubintervalCondenser &) = delete;
        SubintervalCondenser &operator=(const SubintervalCondenser &) = delete;
        ~SubintervalCondenser();

        void freeBuffers();

        void setMode(LocalStageSolverMode mode) { m_mode = mode; }
        LocalStageSolverMode mode() const { return m_mode; }

        // Factors element I_n on `slice` using the active local solver backend.
        void factor(const SpatialSlice *slice, size_t st, const double *y_stages, const double *p, double dtau_T, bool build_stage_jacs = true);

        // Solves A_n * dk_stages = rhs_stages where A_n = I x B - dtau_T * (A^R x A_{n,l}).
        // `rhs_stages` and `dk_stages` both have shape (n_space x st) in column-major (st contiguous slices of length n_space).
        // `dk_stages` may alias `rhs_stages`.
        void solveLocalStages(const double *rhs_stages, double *dk_stages, bool exact_stage_jac = true, double local_rtol = 1e-12) const;

        // Batched solve for K RHS columns (layout: st blocks of (n_space * K) column-major).
        void solveLocalStagesBatch(size_t K, const double *rhs_stages, double *dk_stages, bool exact_stage_jac = true, double local_rtol = 1e-12) const;

        // Evaluates out_stages = A_n * dy_stages using exact stage Jacobians (or avg Jacobian if !has_stage_jacs).
        void applyStageOperator(const double *dy_stages, double *out_stages) const;

        // Evaluates the condensed element propagator on slice M_n:
        //   w_out = [A_n^{-1} (1_{st} x B_{M_n} w_in)]_{st}
        // `w_in` and `w_out` have length n_space (on `m_slice`).
        void applyElementPropagator(const RP(double) w_in, RP(double) w_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;
        void applyElementPropagatorAllStages(const RP(double) w_in, RP(double) dk_stages_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;

        // Evaluates condensed propagator from previous slice M_{n-1} to M_n:
        //   w_out = [A_n^{-1} (1_{st} x B_{M_n} P_{n-1 -> n} v_prev)]_{st}
        // `w_in` and `w_out` have length n_space (on `m_slice`).
        void applyCondensedC(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) w_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;
        void applyCondensedCBatch(size_t K, const RP(double) v_prev, size_t ld_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) w_out, size_t ld_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;
        void applyCondensedCAllStages(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) dk_stages_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;

        // Assembles the explicit column-major condensed propagator matrix C_out (n_space x K)
        // where column j is applyCondensedC(e_j, transfer_prev_to_cur, col_j, true), using batched BLAS3 solves.
        void computeCondensedMatrix(const SparseDenseOutputOp &transfer_prev_to_cur, double *C_out, bool exact_stage_jac = true, double local_rtol = 1e-12) const;

        // Computes d(R_space)/dp_col at stage `l` (0 <= l < st) on `m_slice`, length n_space.
        // `param_index` is the index in `p[]` (e.g., 0 for c, pmask[1] for alpha).
        void evalSpatialParamDeriv(size_t stage_l, const double *y_stage, const double *p, int param_index, RP(double) out) const;

        size_t nSpace() const { return m_n_space; }
        size_t st() const { return m_st; }
        const SpatialSlice *slice() const { return m_slice; }
        const DEV::Matrix &avgJacobian() const { return m_avg_jac; }

        size_t statRichardsonSweeps() const { return m_stat_richardson_sweeps; }
        size_t statGmresFallbacks() const { return m_stat_gmres_fallbacks; }
        void resetStats() const { m_stat_richardson_sweeps = 0; m_stat_gmres_fallbacks = 0; }

    private:
        void ensureWorkspace(size_t n_space, size_t st);
        void ensureBatchWorkspace(size_t K_batch, size_t n_space, size_t st, size_t extra_doubles = 0) const;
        void ensureHwFactored() const;
        void applyPrecondOnly(const double *rhs_stages, double *dk_stages) const;
        void applyPrecondOnlyFromBhat(const double *b_hat, double *dk_stages) const;
        void applyPrecondOnlyBatch(size_t K, const double *rhs_stages, double *dk_stages, double *batch_work, complex_t *cbatch_work) const;
        void applyDeltaStageOp(const double *in_stages, double *out_stages) const;
        void applyDeltaStageOpBatch(size_t K, const double *in_stages, double *out_stages, double *w_buf) const;
        void applySpatialJacFromFjac(const double *fjac_stage, const double *in, double *out) const;
        void applySpatialJacFromFjacBatch(size_t K, const double *fjac_stage, const double *in, double *out) const;
        void applyBgsSweep(const double *rhs_stages, double *e_stages, double *v_stages, double *next_res_stages, double *b_ej_tmp) const;
        void applyBgsSweepBatch(size_t K, const double *rhs_stages, double *e_stages, double *v_stages, double *next_res_stages, double *b_ej_tmp) const;
        void buildUmfpackSystem(const SpatialSlice *slice, size_t st, double dtau_T, bool refactor_numeric);
        void applyUmfpackSolve(const double *rhs_stages, double *dk_stages) const;
        void freeUmfpack();
        void applyStageOperatorBatch(size_t K, const double *dy_stages, double *out_stages, double *ay_buf, double *mass_work) const;
        void runLocalGmres(double *x, const double *b, double local_rtol, double atol) const;

        LocalStageSolverMode m_mode = defaultLocalStageSolverMode();
        bool m_active_bgs = false;
        bool m_active_umf = false;

        const SpatialSlice *m_slice = nullptr;
        size_t m_st = 0;
        size_t m_n_space = 0;
        size_t m_bc_node = 0;
        double m_dtau_T = 1.0;
        bool m_has_stage_jacs = false;
        double m_max_delta_fjac = 0.0;

        mutable DEV::BoundaryConditions m_real_bc;
        mutable DEV::ComplexBoundaryConditions m_complex_bc;

        mutable DEV::Matrix m_avg_jac;
        mutable double *m_avg_fjac = nullptr;
        size_t m_avg_fjac_cap = 0;
        double *m_delta_fjac = nullptr;
        size_t m_delta_fjac_cap = 0;

        mutable DEV::Matrix m_real_mats[2];
        mutable DEV::ComplexMatrix m_complex_mats[2];
        mutable DEV::Matrix m_stage_mats[4];
        mutable bool m_hw_factored = false;
        double *m_loc_weights = nullptr;
        size_t m_loc_weights_cap = 0;

        // UMFPACK sparse direct / precond state (only allocated if Umfpack/UmfpackPrecond used)
        void *m_umf_symbolic = nullptr;
        void *m_umf_numeric = nullptr;
        int32_t *m_umf_Ap = nullptr;
        int32_t *m_umf_Ai = nullptr;
        double *m_umf_Ax = nullptr;
        int32_t *m_umf_Wi = nullptr;
        double *m_umf_W = nullptr;
        mutable double *m_umf_rhs = nullptr;
        size_t m_umf_n = 0;
        size_t m_umf_nnz = 0;
        size_t m_umf_nnz_cap = 0;
        size_t m_umf_w_cap = 0;
        size_t m_umf_sig_nnodes = 0;
        size_t m_umf_sig_node = 0;
        size_t m_umf_sig_ncol = 0;
        size_t m_umf_sig_st = 0;
        size_t m_umf_sig_mnnz = 0;
        int m_umf_lag_count = 0;

        // Pre-allocated scratch buffers (mutable for const solve/apply methods)
        mutable double *m_y_avg = nullptr;
        mutable double *m_work = nullptr;     // general real scratch
        mutable complex_t *m_cwork = nullptr; // complex solve scratch
        size_t m_work_cap = 0;
        size_t m_cwork_cap = 0;
        mutable double *m_pjac_work = nullptr;
        size_t m_pjac_work_cap = 0;
        mutable double *m_gmres_work = nullptr;
        mutable size_t m_gmres_work_cap = 0;
        mutable double *m_cmat_buf = nullptr;
        mutable size_t m_cmat_cap = 0;
        mutable complex_t *m_cmat_cbuf = nullptr;
        mutable size_t m_cmat_ccap = 0;
        mutable size_t *m_cmat_cols = nullptr;
        mutable size_t m_cmat_cols_cap = 0;
        mutable double *m_cond_batch_buf = nullptr;
        mutable size_t m_cond_batch_cap = 0;
        mutable size_t m_stat_richardson_sweeps = 0;
        mutable size_t m_stat_gmres_fallbacks = 0;
        mutable bool m_triggered_local_gmres = false;
    };
} // namespace TWIST
