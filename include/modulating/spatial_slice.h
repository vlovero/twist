#pragma once

#include "GL/gl.h"
#include "collocator.h"
#include "shared.h"
#include <cstddef>

namespace TWIST
{
    struct SpatialSlice
    {
        size_t nstages = 0;
        size_t node = 0;
        size_t nnodes = 0;
        size_t n_space = 0;
        double unscaled_spatial_period = 0.0;

        const double *c_A = nullptr;
        const double *c_b = nullptr;
        const double *c_c = nullptr;
        const double *c_P = nullptr;
        const double *c_chat = nullptr;

        func_t func = nullptr;
        func_t fjac = nullptr;
        pjac_t pjac = nullptr;

        double *t = nullptr;
        double *h = nullptr;

        size_t ndiff = 0;
        int *to_delete = nullptr;
        double *diff_coeffs = nullptr;
        double *M = nullptr;       // column-major (transposed) M(c, L) matching Hopf::setup_M_and_D
        double *dM_dc = nullptr;   // column-major dM/dc
        double *A_gl_invT = nullptr; // (A_GL)^{-T}, size nstages * nstages

        SpatialSlice() = default;
        SpatialSlice(const SpatialSlice &other);
        SpatialSlice &operator=(const SpatialSlice &other);
        ~SpatialSlice();

        void freeBuffers();

        void init(size_t in_nstages, size_t in_node, size_t in_nnodes, const double *in_t, func_t in_func, func_t in_fjac, pjac_t in_pjac, const diffusion_t &diffusion, double in_unscaled_spatial_period);
        void initFromCollocator(const Collocator &col);
        void updateGrid(size_t new_nnodes, const double *new_t);

        // Builds M(c, L) and dM/dc(c, L) in column-major layout (where L = spatial_period = sps * unscaled_spatial_period)
        void updateMassMatrices(double c, double sps);

        // Evaluates R_space(Y; p) with periodic BC Y_0 - Y_end at the last `node` entries.
        // `yp` must have size >= n_space (can be scratch).
        void evaluateResidual(const RP(double) Y, const RP(double) p, RP(double) residual, RP(double) yp) const;

        // Evaluates out = beta * out + alpha * B_M * v using the supplied column-major M_colmajor (node x node).
        // `work` must have size >= node * nstages.
        void applyMassOp(const double *M_colmajor, const RP(double) v, RP(double) out, double alpha, double beta, RP(double) work) const;

        void applyMass(const RP(double) v, RP(double) out, double alpha, double beta, RP(double) work) const;
        void applyMassDerivC(const RP(double) v, RP(double) out, double alpha, double beta, RP(double) work) const;
        void applyMassDerivSps(double sps, const RP(double) v, RP(double) out, double alpha, double beta, RP(double) work) const;

        // Fills quadrature weights of length n_space.
        // If include_nodes is true, node entries get small positive trapezoidal weights so W is strictly positive definite for GMRES.
        void fillQuadratureWeights(RP(double) weights, bool include_nodes) const;

        // Computes spatial monitor function data from stage values of a state vector `Y` (length n_space).
        // `F` length >= nnodes, `v` length >= nnodes - 1, `work` length >= node * nstages + nstages * nstages.
        void computeMonitorFromState(const double *Y, RP(double) F, RP(double) v, double &theta, double &err_exp, RP(double) work) const;

        // Equidistributes a new mesh `out_t` (caller frees or pass realloc'd buffer) given pre-computed elementwise monitor `v_max` (length nnodes - 1) and `F_cum` (length nnodes).
        double *equidistributeMesh(const double *F_cum, const double *v_max, double theta, double err_exp, double geps, size_t min_nnodes, size_t max_nnodes, size_t &nnodes_opt, bool small_change) const;

        size_t nSub() const { return nnodes > 0 ? nnodes - 1 : 0; }
        size_t nPoints() const { return nnodes > 0 ? (nnodes - 1) * (nstages + 1) + 1 : 0; }
    };

    struct SparseDenseOutputOp
    {
        size_t nstages = 0;
        size_t node = 0;
        size_t src_npts = 0;
        size_t dst_npts = 0;
        bool is_identity = false;

        int *src_sub = nullptr;   // length dst_npts
        double *weights = nullptr; // length dst_npts * (nstages + 1)

        SparseDenseOutputOp() = default;
        SparseDenseOutputOp(const SparseDenseOutputOp &other);
        SparseDenseOutputOp &operator=(const SparseDenseOutputOp &other);
        ~SparseDenseOutputOp();

        void freeBuffers();
        void build(const SpatialSlice &src, const SpatialSlice &dst);
        void apply(const RP(double) src_y, RP(double) dst_y) const;
    };
} // namespace TWIST
