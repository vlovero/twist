#pragma once

#include "colmat/tcolmat.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include "shared.h"
#include <complex>
#include <cstddef>

namespace TWIST
{
    class RealSchurSpaceTimePrecond
    {
    public:
        using complex_t = std::complex<double>;

        RealSchurSpaceTimePrecond() = default;
        RealSchurSpaceTimePrecond(const RealSchurSpaceTimePrecond &) = delete;
        RealSchurSpaceTimePrecond &operator=(const RealSchurSpaceTimePrecond &) = delete;
        ~RealSchurSpaceTimePrecond();

        void freeBuffers();
        void invalidate() { m_nt = 0; }

        // Updates the preconditioner from the current N_t slices, condensers, temporal step sizes dtau[0..nt-1],
        // scalar period T, parameter vector p, and theta (default 1.0, L-stable for DAE algebraic modes).
        void update(size_t nt, const SpatialSlice *slices, const SubintervalCondenser *condensers, const double *y_end, const double *dtau, double T, const double *p, double theta = 1.0);

        // Applies z = P_ST^(-1) * r on the condensed endpoint space (sum_n slices[n].n_space).
        // `r` and `z` may NOT alias.
        void apply(const RP(double) r, RP(double) z) const;

        size_t totalEndSize() const { return m_total_end; }
        size_t nt() const { return m_nt; }
        const SpatialSlice &refSlice() const { return m_ref_slice; }

    private:
        struct SchurBlock
        {
            bool is_complex;
            size_t col; // starting index in 0..nt-1
            double a;
            double beta;
            double kappa; // scale for 2x2 diagonal similarity
            double sgn;   // sign of S(col, col+1)
            size_t mat_idx;
        };

        void ensureBuffers(size_t nt, size_t n_ref, size_t total_end);

        size_t m_nt = 0;
        size_t m_n_ref = 0;
        size_t m_total_end = 0;
        double m_T = 1.0;
        double m_theta = 0.55;

        SpatialSlice m_ref_slice;
        DEV::BoundaryConditions m_real_bc;
        DEV::ComplexBoundaryConditions m_complex_bc;
        DEV::Matrix m_ref_jac;

        // Per-slice offsets & mesh transfers
        size_t *m_offsets = nullptr;
        double *m_dtau = nullptr;
        SparseDenseOutputOp *m_to_ref = nullptr;
        SparseDenseOutputOp *m_from_ref = nullptr;
        size_t m_slice_cap = 0;

        // Temporal Nt x Nt matrices (column-major)
        double *m_S_tau = nullptr;
        double *m_Q_tau = nullptr;
        double *m_W_in = nullptr; // Q_tau^T * M_tau^(-1)
        double *m_lapack_work = nullptr;
        int *m_ipiv = nullptr;
        int *m_bwork = nullptr;
        size_t m_nt_cap = 0;

        // Schur block solvers
        SchurBlock *m_blocks = nullptr;
        size_t m_nblocks = 0;
        mutable DEV::Matrix *m_real_mats = nullptr;
        mutable DEV::ComplexMatrix *m_complex_mats = nullptr;
        size_t m_nreal_mats = 0;
        size_t m_ncomplex_mats = 0;
        size_t m_real_cap = 0;
        size_t m_complex_cap = 0;

        // Pre-allocated scratch buffers
        mutable double *m_work = nullptr;
        size_t m_work_cap = 0;
        mutable complex_t *m_cwork = nullptr;
        size_t m_cwork_cap = 0;
    };
} // namespace TWIST
