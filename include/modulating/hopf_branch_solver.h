#pragma once

#include "H5Cpp.h"
#include "collocator.h"
#include "modulating/floquet.h"
#include "modulating/radau_dg.h"
#include "modulating/spacetime_precond.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include "shared.h"
#include <complex>
#include <cstddef>
#include <string>

namespace TWIST
{
    struct HopfContinuationBounds
    {
        double ds = 0.05;
        double dsmin = 1e-6;
        double dsmax = 0.2;
        double parmin = 0.0;
        double parmax = 1.0;
        double geps = 1e-10;
        size_t min_nodes_adapt = 16;
        size_t max_nodes_adapt = 200;
        size_t min_nt_adapt = 16;
        size_t max_nt_adapt = 96;
        bool allow_mesh_adaptation = true;
    };

    class HopfBranchSolver
    {
    public:
        using complex_t = std::complex<double>;

        HopfBranchSolver() = default;
        HopfBranchSolver(const HopfBranchSolver &other);
        HopfBranchSolver &operator=(const HopfBranchSolver &other);
        ~HopfBranchSolver();

        void freeBuffers();

        // Initializes at the Hopf bifurcation point (A = 0) from a converged steady Collocator,
        // Hopf frequency `omega`, and complex eigenfunction `psi` (length col.NUnknowns()).
        void initFromHopfPoint(const Collocator &hopf_col, double omega, const complex_t *psi, size_t nt = 20, size_t ncol_time = 3, int parnum = -1);

        // Loads step `solution_index` from an existing hopf_continuation HDF5 file (for --resume or postprocessing).
        void loadFromH5(const std::string &h5_path, int solution_index);
        void loadFromH5(const H5::H5File &file, int solution_index);

        // Sets up continuation predictor step: z_curr = z_prev + ds * tangent, updates m_y_stages accordingly.
        void predict(double ds);

        // Globalized damped Inexact Newton-GMRES solve for the space-time dG(q) + phase + pseudo-arclength system.
        // Returns true if Newton converged, false on failure.
        bool solveNewton(double ds, double tol = 1e-10, double min_damp = 1e-4, int max_iter = 25, bool verbose = false);

        // Adapts temporal grid {tau_n} and per-element spatial meshes {M_n} to maintain `geps`,
        // then polishes with 1-2 Newton steps.
        void adaptSpaceTimeMesh(const HopfContinuationBounds &bounds, double ds, bool small_change = true, bool verbose = false);

        // Computes unit continuation tangent `m_tangent` via one bordered GMRES solve at converged state.
        void computeTangent();

        // Updates reference state & orthogonalized (g_space, g_time) phase templates from current solution.
        void updatePhaseConditions();

        // Commits current solution as `m_z_prev` (after convergence + adaptation).
        void acceptStep();

        // Restores state from `m_z_prev` (e.g. on step failure / ds reduction).
        void restoreAcceptedStep();

        // Computes Floquet spectrum at current converged state using FloquetAnalyzer.
        FloquetSpectrum computeFloquet(size_t neig = 12, size_t nsub = 24, double tol = 1e-10, bool verbose = false);

        // Serializes current solution step to HDF5 (creates file on first call).
        void serialize(const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix = ".cache/continuation_data", SolutionTypes stype = SolutionTypes::regular);

        void setSerializationFileAndIndex(H5::H5File &file, int index);

        // Accessors
        size_t nt() const { return m_nt; }
        size_t st() const { return m_st; }
        size_t node() const { return m_node; }
        size_t spatialNStages() const { return m_spatial_nstages; }
        size_t maxSpatialNodes() const;
        size_t totalEndSize() const { return m_total_end; }
        size_t totalStageSize() const { return m_total_stages; }

        double waveSpeed() const { return m_p ? m_p[0] : 0.0; }
        double period() const { return m_T; }
        double continuationParam() const { return m_p ? m_p[m_pmask[1]] : 0.0; }
        int continuationParamIndex() const { return m_pmask[1]; }
        double oscillationAmplitude() const;
        double solutionNormL2() const;
        double tangentAlpha() const { return m_tangent ? m_tangent[m_total_end + 2] : 0.0; }

        const SpatialSlice *slices() const { return m_slices; }
        const SparseDenseOutputOp *transfers() const { return m_transfers; }
        const SubintervalCondenser *condensers() const { return m_condensers; }
        const double *yStages() const { return m_y_stages; }
        const double *dtau() const { return m_dtau; }
        const double *p() const { return m_p; }
        void *libHandle() const { return m_handle; }
        void setLibHandle(void *h) { m_handle = h; }

    private:
        void allocateForGrid(size_t nt, size_t st, const size_t *slice_nnodes);
        void rebuildOffsetsAndWeights();
        void rebuildTransfers();
        void syncEndpointsFromStages(double *end_out) const;
        void updateMassMatricesAll();

        // Evaluates uncondensed stage residual G (size m_total_stages) and 3 scalar constraints R_p[3].
        // Also computes spatial residual R_space(Y_{n,l}) into `m_r_space` (for dG/dT) if `store_r_space` is true.
        void evaluateFullResidual(const double *y_stages, double c, double T, double alpha, double ds, double *G_out, double *Rp_out, bool store_r_space);

        // Evaluates full residual norm (Euclidean/weighted)
        double computeResidualNorm(const double *G, const double *Rp) const;

        // Factors condensers, computes condensed RHS `b_Y`, scalar param columns `J_Yp`, and updates preconditioner.
        void assembleCondensedSystem(const double *G, bool update_precond);

        // Solves the (m_total_end + 3) bordered system via right-preconditioned weighted GMRES
        // with adaptive M-window exact block condensation.
        int solveBorderedGMRES(const double *rhs_Y, const double *rhs_p, const double *row3_Y, const double *row3_p, double *sol_Y, double *sol_p, double rtol, int maxiter, bool verbose);

        // Weighted L2 inner product on endpoint space (length m_total_end)
        double endpointDot(const double *a, const double *b) const;
        double endpointNorm(const double *a) const;

        // Weighted L2 inner product of bordered vector (m_total_end + 3)
        double borderedDot(const double *a, const double *b) const;
        double borderedNorm(const double *a) const;

        // Model metadata & function pointers
        void *m_handle = nullptr;
        bool m_owns_handle = false;
        func_t m_func = nullptr;
        func_t m_fjac = nullptr;
        pjac_t m_pjac = nullptr;
        size_t m_node = 0;
        size_t m_spatial_nstages = 0;
        size_t m_np = 0;
        int m_pmask[2] = { 0, 1 };
        double m_unscaled_spatial_period = 0.0;
        diffusion_t m_diffusion;
        std::string m_model_name;

        // Time-space grid
        size_t m_nt = 0;
        size_t m_st = 3;
        double *m_tau = nullptr;   // length m_nt + 1
        double *m_dtau = nullptr;  // length m_nt
        SpatialSlice *m_slices = nullptr;
        SparseDenseOutputOp *m_transfers = nullptr;
        SubintervalCondenser *m_condensers = nullptr;
        RealSchurSpaceTimePrecond m_precond;

        size_t *m_end_offsets = nullptr;   // length m_nt + 1
        size_t *m_stage_offsets = nullptr; // length m_nt + 1
        size_t m_total_end = 0;
        size_t m_total_stages = 0;

        // Primary state & continuation vectors
        double *m_y_stages = nullptr;      // m_total_stages
        double *m_y_prev_stages = nullptr; // m_total_stages (accepted backup)
        double *m_p = nullptr;             // m_np
        double m_c_prev = 0.0;
        double m_T = 0.0;
        double m_T_prev = 0.0;
        double m_T_scale = 1.0;
        double m_alpha_prev = 0.0;

        double *m_y_ref_end = nullptr;     // m_total_end (reference for phase conditions)
        double *m_y_prev_end = nullptr;    // m_total_end (for pseudo-arclength)
        double *m_g_space = nullptr;       // m_total_end (normalized spatial phase template)
        double *m_g_time = nullptr;        // m_total_end (normalized orthogonalized temporal phase template)
        double *m_tangent = nullptr;       // m_total_end + 3
        double *m_weights_end = nullptr;   // m_total_end (temporal * spatial quadrature weights)
        double *m_weights_bord = nullptr;  // m_total_end + 3

        // Pre-allocated workspace for Newton + GMRES (zero alloc inside inner loops)
        double *m_r_space = nullptr;       // m_total_stages (stores R_space(Y_{n,l}))
        double *m_G = nullptr;             // m_total_stages
        double *m_G_trial = nullptr;       // m_total_stages
        double *m_y_trial = nullptr;       // m_total_stages
        double *m_delta_stages = nullptr;  // m_total_stages
        double *m_b_Y = nullptr;           // m_total_end
        double *m_J_Yp = nullptr;          // 3 * m_total_end (columns for c, T, alpha)
        double *m_Minv_U = nullptr;        // 3 * m_total_end (P_ST^(-1) * J_Yp)
        double *m_bord_rhs = nullptr;      // m_total_end + 3
        double *m_bord_sol = nullptr;      // m_total_end + 3
        double *m_gmres_work = nullptr;    // gmres_lwork(m_total_end + 3, 40)
        size_t m_gmres_work_cap = 0;

        double *m_stage_scratch = nullptr; // per-thread/general scratch
        size_t m_stage_scratch_cap = 0;
        size_t m_stage_scratch_stride = 0;
        mutable size_t m_win_M = 1;

        // HDF5 serialization state
        int m_serialization_index = -1;
        H5::H5File m_serialization_file;
    };
} // namespace TWIST
