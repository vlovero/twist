#pragma once

#include "colmat/tcolmat.h"
#include "modulating/radau_dg.h"
#include "modulating/spatial_slice.h"
#include "shared.h"
#include <complex>
#include <cstddef>

namespace TWIST
{
    class SubintervalCondenser
    {
    public:
        using complex_t = std::complex<double>;

        SubintervalCondenser() = default;
        SubintervalCondenser(const SubintervalCondenser &) = delete;
        SubintervalCondenser &operator=(const SubintervalCondenser &) = delete;
        ~SubintervalCondenser();

        void freeBuffers();

        // Factors the stage-averaged Hairer-Wanner matrices for element I_n on `slice`,
        // and updates unfactored stage Jacobians A_{n,l} (l = 0..st-1) for exact matvec / Richardson.
        void factor(const SpatialSlice *slice, size_t st, const double *y_stages, const double *p, double dtau_T, bool build_stage_jacs = true);

        // Solves A_n * dk_stages = rhs_stages where A_n = I x B - dtau_T * (A^R x A_{n,l}).
        // `rhs_stages` and `dk_stages` both have shape (n_space x st) in column-major (st contiguous slices of length n_space).
        // `dk_stages` may alias `rhs_stages`.
        void solveLocalStages(const double *rhs_stages, double *dk_stages, bool exact_stage_jac = true) const;

        // Evaluates out_stages = A_n * dy_stages using exact stage Jacobians (or avg Jacobian if !has_stage_jacs).
        void applyStageOperator(const double *dy_stages, double *out_stages) const;

        // Evaluates the condensed element propagator on slice M_n:
        //   w_out = [A_n^{-1} (1_{st} x B_{M_n} w_in)]_{st}
        // `w_in` and `w_out` have length n_space (on `m_slice`).
        void applyElementPropagator(const RP(double) w_in, RP(double) w_out, bool exact_stage_jac = true) const;

        // Evaluates condensed propagator from previous slice M_{n-1} to M_n:
        //   w_out = [A_n^{-1} (1_{st} x B_{M_n} P_{n-1 -> n} v_prev)]_{st}
        void applyCondensedC(const RP(double) v_prev, const SparseDenseOutputOp &transfer_prev_to_cur, RP(double) w_out, bool exact_stage_jac = true) const;

        // Computes d(R_space)/dp_col at stage `l` (0 <= l < st) on `m_slice`, length n_space.
        // `param_index` is the index in `p[]` (e.g., 0 for c, pmask[1] for alpha).
        void evalSpatialParamDeriv(size_t stage_l, const double *y_stage, const double *p, int param_index, RP(double) out) const;

        size_t nSpace() const { return m_n_space; }
        size_t st() const { return m_st; }
        const SpatialSlice *slice() const { return m_slice; }
        const DEV::Matrix &avgJacobian() const { return m_avg_jac; }

    private:
        void ensureWorkspace(size_t n_space, size_t st);
        void applyPrecondOnly(const double *rhs_stages, double *dk_stages) const;

        const SpatialSlice *m_slice = nullptr;
        size_t m_st = 0;
        size_t m_n_space = 0;
        double m_dtau_T = 1.0;
        bool m_has_stage_jacs = false;

        DEV::BoundaryConditions m_real_bc;
        DEV::ComplexBoundaryConditions m_complex_bc;

        DEV::Matrix m_avg_jac;
        DEV::Matrix *m_stage_jacs = nullptr;
        size_t m_stage_jacs_cap = 0;

        mutable DEV::Matrix m_real_mats[2];
        mutable DEV::ComplexMatrix m_complex_mats[2];

        // Pre-allocated scratch buffers (mutable for const solve/apply methods)
        mutable double *m_y_avg = nullptr;
        mutable double *m_p_in = nullptr;     // st x st: T_s^(-1) * A_inv (row-major)
        mutable double *m_work = nullptr;     // general real scratch
        mutable complex_t *m_cwork = nullptr; // complex solve scratch
        size_t m_work_cap = 0;
        size_t m_cwork_cap = 0;
        mutable double *m_pjac_work = nullptr;
        size_t m_pjac_work_cap = 0;
        mutable double *m_gmres_work = nullptr;
        mutable size_t m_gmres_work_cap = 0;
    };
} // namespace TWIST
