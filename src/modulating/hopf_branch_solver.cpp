#include "modulating/hopf_branch_solver.h"
#include "cli/load.h"
#include "fmt/core.h"
#include "krylov/gmres.h"
#include "libloader.h"
#include "linalg.h"
#include "numpy_like.h"
#include "profiling.h"
#include "serialize.h"
#include "tools/helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <omp.h>
#include <stdexcept>

namespace TWIST
{
    namespace
    {
        void eval_slice_spatial_deriv(const SpatialSlice &slice, const double *Y, double *dY_dxi)
        {
            const size_t nsub = slice.nSub();
            const size_t m = slice.node;
            const size_t s = slice.nstages;
            const size_t n_basis = s + 1;

            double nodes[n_basis];
            nodes[0] = 0.0;
            for (size_t j = 0; j < s; j++) {
                nodes[j + 1] = slice.c_c[j];
            }

            double D_gl[n_basis * n_basis];
            for (size_t l = 0; l < n_basis; l++) {
                const double th = nodes[l];
                for (size_t j = 0; j < n_basis; j++) {
                    double den = 1.0;
                    for (size_t q = 0; q < n_basis; q++) {
                        if (q != j) den *= (nodes[j] - nodes[q]);
                    }
                    double num = 0.0;
                    for (size_t k = 0; k < n_basis; k++) {
                        if (k == j) continue;
                        double prod = 1.0;
                        for (size_t q = 0; q < n_basis; q++) {
                            if (q != j && q != k) prod *= (th - nodes[q]);
                        }
                        num += prod;
                    }
                    D_gl[l * n_basis + j] = num / den;
                }
            }

            for (size_t k = 0; k < nsub; k++) {
                const size_t stride_k = k * n_basis * m;
                const double inv_hk = 1.0 / slice.h[k];
                for (size_t l = 0; l < n_basis; l++) {
                    double *out = &dY_dxi[stride_k + l * m];
                    memset(out, 0, m * sizeof(double));
                    for (size_t j = 0; j < n_basis; j++) {
                        const double w = inv_hk * D_gl[l * n_basis + j];
                        const double *src = &Y[stride_k + j * m];
                        for (size_t d = 0; d < m; d++) {
                            out[d] += w * src[d];
                        }
                    }
                }
            }
            memcpy(&dY_dxi[nsub * n_basis * m], dY_dxi, m * sizeof(double));
        }

        size_t find_interval(const double *tau, size_t nt, double t)
        {
            if (t <= tau[0]) return 0;
            if (t >= tau[nt]) return nt - 1;
            size_t lo = 0, hi = nt;
            while (lo + 1 < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (tau[mid] <= t) lo = mid;
                else hi = mid;
            }
            return lo;
        }
    } // namespace

    void HopfTimingStats::reset()
    {
        t_residual = 0.0;
        t_condense_factor = 0.0;
        t_condense_rhs_p = 0.0;
        t_precond_update = 0.0;
        t_gmres_total = 0.0;
        t_gmres_preelim = 0.0;
        t_gmres_A_op = 0.0;
        t_gmres_M_op = 0.0;
        t_gmres_direct_lu = 0.0;
        t_gmres_reconstruct = 0.0;
        t_stage_recover = 0.0;
        t_adapt = 0.0;
        t_phase = 0.0;
        t_tangent = 0.0;
        t_amp = 0.0;
        t_serialize = 0.0;
        n_newton_iters = 0;
        n_gmres_iters = 0;
        n_local_gmres_fallbacks = 0;
        n_richardson_sweeps = 0;
        n_failed_steps = 0;
    }

    void HopfTimingStats::printSummary(double wall_total) const
    {
        const double inv = (wall_total > 0.0) ? (100.0 / wall_total) : 0.0;
        const double accounted = t_residual + t_condense_factor + t_condense_rhs_p + t_precond_update
                               + t_gmres_total + t_stage_recover + t_adapt + t_phase
                               + t_tangent + t_amp + t_serialize;
        fmt::println("\n=== Hopf Timing Summary (wall = {:.3f} s, accounted = {:.3f} s [{:.1f}%]) ===",
            wall_total, accounted, accounted * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "residual", t_residual, t_residual * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "condense_factor", t_condense_factor, t_condense_factor * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "condense_rhs_p", t_condense_rhs_p, t_condense_rhs_p * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "precond_update", t_precond_update, t_precond_update * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "gmres_total", t_gmres_total, t_gmres_total * inv);
        fmt::println("    {:22s}: {:8.3f} s ({:5.1f}%)", "gmres_preelim", t_gmres_preelim, t_gmres_preelim * inv);
        fmt::println("    {:22s}: {:8.3f} s ({:5.1f}%)", "gmres_A_op", t_gmres_A_op, t_gmres_A_op * inv);
        fmt::println("    {:22s}: {:8.3f} s ({:5.1f}%)", "gmres_M_op", t_gmres_M_op, t_gmres_M_op * inv);
        fmt::println("    {:22s}: {:8.3f} s ({:5.1f}%)", "gmres_direct_lu", t_gmres_direct_lu, t_gmres_direct_lu * inv);
        fmt::println("    {:22s}: {:8.3f} s ({:5.1f}%)", "gmres_reconstruct", t_gmres_reconstruct, t_gmres_reconstruct * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "stage_recover", t_stage_recover, t_stage_recover * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "adapt", t_adapt, t_adapt * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "phase", t_phase, t_phase * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "tangent", t_tangent, t_tangent * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "amplitude", t_amp, t_amp * inv);
        fmt::println("  {:24s}: {:8.3f} s ({:5.1f}%)", "serialize", t_serialize, t_serialize * inv);
        fmt::println("  Counters: newton_iters={}, gmres_iters={}, richardson_sweeps={}, local_gmres={}, failed_steps={}",
            n_newton_iters, n_gmres_iters, n_richardson_sweeps, n_local_gmres_fallbacks, n_failed_steps);
    }

    void HopfBranchSolver::resetTimingStats()
    {
        m_timing.reset();
        for (size_t n = 0; n < m_nt; n++) {
            m_condensers[n].resetStats();
        }
    }

    void HopfBranchSolver::collectCondenserStats()
    {
        size_t rich = 0, lgm = 0;
        for (size_t n = 0; n < m_nt; n++) {
            rich += m_condensers[n].statRichardsonSweeps();
            lgm += m_condensers[n].statGmresFallbacks();
        }
        m_timing.n_richardson_sweeps = rich;
        m_timing.n_local_gmres_fallbacks = lgm;
    }

    HopfBranchSolver::HopfBranchSolver(const HopfBranchSolver &other)
    {
        *this = other;
    }

    HopfBranchSolver &HopfBranchSolver::operator=(const HopfBranchSolver &other)
    {
        if (this == &other) {
            return *this;
        }
        freeBuffers();

        m_handle = other.m_handle;
        m_owns_handle = false;
        m_func = other.m_func;
        m_fjac = other.m_fjac;
        m_pjac = other.m_pjac;
        m_node = other.m_node;
        m_spatial_nstages = other.m_spatial_nstages;
        m_np = other.m_np;
        m_pmask[0] = other.m_pmask[0];
        m_pmask[1] = other.m_pmask[1];
        m_unscaled_spatial_period = other.m_unscaled_spatial_period;
        m_diffusion = other.m_diffusion;
        m_model_name = other.m_model_name;
        m_c_prev = other.m_c_prev;
        m_T = other.m_T;
        m_T_prev = other.m_T_prev;
        m_T_scale = other.m_T_scale;
        m_alpha_prev = other.m_alpha_prev;
        m_win_M = other.m_win_M;
        m_serialization_index = other.m_serialization_index;
        m_serialization_file = other.m_serialization_file;
        m_serialization_root = other.m_serialization_root;
        m_serialization_branch = other.m_serialization_branch;

        if (other.m_nt > 0) {
            std::vector<size_t> nn(other.m_nt);
            for (size_t n = 0; n < other.m_nt; n++) {
                nn[n] = other.m_slices[n].nnodes;
            }
            allocateForGrid(other.m_nt, other.m_st, nn.data());
            memcpy(m_tau, other.m_tau, (m_nt + 1) * sizeof(double));
            memcpy(m_dtau, other.m_dtau, m_nt * sizeof(double));
            for (size_t n = 0; n < m_nt; n++) {
                m_slices[n] = other.m_slices[n];
            }
            rebuildOffsetsAndWeights();
            rebuildTransfers();

            memcpy(m_y_stages, other.m_y_stages, m_total_stages * sizeof(double));
            memcpy(m_y_prev_stages, other.m_y_prev_stages, m_total_stages * sizeof(double));
            memcpy(m_p, other.m_p, m_np * sizeof(double));
            memcpy(m_y_ref_end, other.m_y_ref_end, m_total_end * sizeof(double));
            memcpy(m_y_prev_end, other.m_y_prev_end, m_total_end * sizeof(double));
            memcpy(m_g_space, other.m_g_space, m_total_end * sizeof(double));
            memcpy(m_g_time, other.m_g_time, m_total_end * sizeof(double));
            memcpy(m_tangent, other.m_tangent, (m_total_end + 3) * sizeof(double));
        }

        return *this;
    }

    HopfBranchSolver::~HopfBranchSolver()
    {
        freeBuffers();
    }

    void HopfBranchSolver::freeBuffers()
    {
        if (m_owns_handle && m_handle) {
            dlclose(m_handle);
        }
        m_handle = nullptr;
        m_owns_handle = false;

        if (m_tau) { free(m_tau); m_tau = nullptr; }
        if (m_dtau) { free(m_dtau); m_dtau = nullptr; }
        if (m_slices) { delete[] m_slices; m_slices = nullptr; }
        if (m_transfers) { delete[] m_transfers; m_transfers = nullptr; }
        if (m_condensers) { delete[] m_condensers; m_condensers = nullptr; }
        m_precond.freeBuffers();

        if (m_end_offsets) { free(m_end_offsets); m_end_offsets = nullptr; }
        if (m_stage_offsets) { free(m_stage_offsets); m_stage_offsets = nullptr; }
        if (m_y_stages) { free(m_y_stages); m_y_stages = nullptr; }
        if (m_y_prev_stages) { free(m_y_prev_stages); m_y_prev_stages = nullptr; }
        if (m_p) { free(m_p); m_p = nullptr; }
        if (m_y_ref_end) { free(m_y_ref_end); m_y_ref_end = nullptr; }
        if (m_y_prev_end) { free(m_y_prev_end); m_y_prev_end = nullptr; }
        if (m_g_space) { free(m_g_space); m_g_space = nullptr; }
        if (m_g_time) { free(m_g_time); m_g_time = nullptr; }
        if (m_tangent) { free(m_tangent); m_tangent = nullptr; }
        if (m_weights_end) { free(m_weights_end); m_weights_end = nullptr; }
        if (m_weights_bord) { free(m_weights_bord); m_weights_bord = nullptr; }

        if (m_r_space) { free(m_r_space); m_r_space = nullptr; }
        if (m_G) { free(m_G); m_G = nullptr; }
        if (m_G_trial) { free(m_G_trial); m_G_trial = nullptr; }
        if (m_y_trial) { free(m_y_trial); m_y_trial = nullptr; }
        if (m_delta_stages) { free(m_delta_stages); m_delta_stages = nullptr; }
        if (m_b_Y) { free(m_b_Y); m_b_Y = nullptr; }
        if (m_J_Yp) { free(m_J_Yp); m_J_Yp = nullptr; }
        if (m_Minv_U) { free(m_Minv_U); m_Minv_U = nullptr; }
        if (m_bord_rhs) { free(m_bord_rhs); m_bord_rhs = nullptr; }
        if (m_bord_sol) { free(m_bord_sol); m_bord_sol = nullptr; }
        if (m_gmres_work) { free(m_gmres_work); m_gmres_work = nullptr; m_gmres_work_cap = 0; }
        if (m_stage_scratch) { free(m_stage_scratch); m_stage_scratch = nullptr; m_stage_scratch_cap = 0; }

        m_nt = 0;
        m_total_end = 0;
        m_total_stages = 0;
    }

    void HopfBranchSolver::allocateForGrid(size_t nt, size_t st, const size_t *slice_nnodes)
    {
        if (m_nt != nt) {
            delete[] m_slices;
            delete[] m_transfers;
            delete[] m_condensers;
            m_slices = new SpatialSlice[nt];
            m_transfers = new SparseDenseOutputOp[nt];
            m_condensers = new SubintervalCondenser[nt];
            m_tau = (double *)realloc(m_tau, (nt + 1) * sizeof(double));
            m_dtau = (double *)realloc(m_dtau, nt * sizeof(double));
            m_end_offsets = (size_t *)realloc(m_end_offsets, (nt + 1) * sizeof(size_t));
            m_stage_offsets = (size_t *)realloc(m_stage_offsets, (nt + 1) * sizeof(size_t));
        }
        m_nt = nt;
        m_st = st;

        size_t total_end = 0;
        for (size_t n = 0; n < nt; n++) {
            const size_t ns = m_node * ((slice_nnodes[n] - 1) * (m_spatial_nstages + 1) + 1);
            total_end += ns;
        }
        const size_t total_stages = total_end * st;

        if (m_np > 0 && m_p == nullptr) {
            m_p = (double *)calloc(m_np, sizeof(double));
        }

        m_y_stages = (double *)realloc(m_y_stages, total_stages * sizeof(double));
        m_y_prev_stages = (double *)realloc(m_y_prev_stages, total_stages * sizeof(double));
        m_y_ref_end = (double *)realloc(m_y_ref_end, total_end * sizeof(double));
        m_y_prev_end = (double *)realloc(m_y_prev_end, total_end * sizeof(double));
        m_g_space = (double *)realloc(m_g_space, total_end * sizeof(double));
        m_g_time = (double *)realloc(m_g_time, total_end * sizeof(double));
        m_tangent = (double *)realloc(m_tangent, (total_end + 3) * sizeof(double));
        m_weights_end = (double *)realloc(m_weights_end, total_end * sizeof(double));
        m_weights_bord = (double *)realloc(m_weights_bord, (total_end + 3) * sizeof(double));

        m_r_space = (double *)realloc(m_r_space, total_stages * sizeof(double));
        m_G = (double *)realloc(m_G, total_stages * sizeof(double));
        m_G_trial = (double *)realloc(m_G_trial, total_stages * sizeof(double));
        m_y_trial = (double *)realloc(m_y_trial, total_stages * sizeof(double));
        m_delta_stages = (double *)realloc(m_delta_stages, total_stages * sizeof(double));
        m_b_Y = (double *)realloc(m_b_Y, total_end * sizeof(double));
        m_J_Yp = (double *)realloc(m_J_Yp, 3 * total_end * sizeof(double));
        m_Minv_U = (double *)realloc(m_Minv_U, 3 * total_end * sizeof(double));
        m_bord_rhs = (double *)realloc(m_bord_rhs, (total_end + 3) * sizeof(double));
        m_bord_sol = (double *)realloc(m_bord_sol, (total_end + 3) * sizeof(double));

        const size_t needed_gmres = Krylov::gmres_lwork(total_end + 3, 100);
        if (m_gmres_work_cap < needed_gmres) {
            m_gmres_work = (double *)realloc(m_gmres_work, needed_gmres * sizeof(double));
            m_gmres_work_cap = needed_gmres;
        }

        size_t max_ns = 0;
        for (size_t n = 0; n < nt; n++) {
            const size_t ns = m_node * ((slice_nnodes[n] - 1) * (m_spatial_nstages + 1) + 1);
            max_ns = std::max(max_ns, ns);
        }
        const size_t max_threads = (size_t)std::max(1, omp_get_max_threads());
        m_stage_scratch_stride = max_ns * (st + 6) + 64;
        const size_t needed_scratch = max_threads * m_stage_scratch_stride;
        if (m_stage_scratch_cap < needed_scratch) {
            m_stage_scratch = (double *)realloc(m_stage_scratch, needed_scratch * sizeof(double));
            m_stage_scratch_cap = needed_scratch;
        }
        m_win_M = 1;
        m_precond.invalidate();
    }

    void HopfBranchSolver::rebuildOffsetsAndWeights()
    {
        m_end_offsets[0] = 0;
        m_stage_offsets[0] = 0;
        for (size_t n = 0; n < m_nt; n++) {
            m_dtau[n] = m_tau[n + 1] - m_tau[n];
            const size_t ns = m_slices[n].n_space;
            m_end_offsets[n + 1] = m_end_offsets[n] + ns;
            m_stage_offsets[n + 1] = m_stage_offsets[n] + m_st * ns;
            m_slices[n].fillQuadratureWeights(&m_weights_end[m_end_offsets[n]], true);
            for (size_t i = 0; i < ns; i++) {
                m_weights_end[m_end_offsets[n] + i] *= m_dtau[n];
            }
        }
        m_total_end = m_end_offsets[m_nt];
        m_total_stages = m_stage_offsets[m_nt];
        memcpy(m_weights_bord, m_weights_end, m_total_end * sizeof(double));
        const double w_T = 1.0 / (m_T_scale * m_T_scale);
        m_weights_bord[m_total_end + 0] = 1.0;
        m_weights_bord[m_total_end + 1] = w_T;
        m_weights_bord[m_total_end + 2] = 1.0;
    }

    void HopfBranchSolver::rebuildTransfers()
    {
        for (size_t n = 0; n < m_nt; n++) {
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            m_transfers[n].build(m_slices[prev], m_slices[n]);
        }
    }

    void HopfBranchSolver::syncEndpointsFromStages(double *end_out) const
    {
        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            memcpy(&end_out[m_end_offsets[n]], &m_y_stages[m_stage_offsets[n] + (m_st - 1) * ns], ns * sizeof(double));
        }
    }

    void HopfBranchSolver::updateMassMatricesAll()
    {
        const double c = m_p[0];
        const double sps = m_p[m_np - 1];
        for (size_t n = 0; n < m_nt; n++) {
            m_slices[n].updateMassMatrices(c, sps);
        }
    }

    double HopfBranchSolver::endpointDot(const double *a, const double *b) const
    {
        double acc = 0.0;
        for (size_t i = 0; i < m_total_end; i++) {
            acc += m_weights_end[i] * a[i] * b[i];
        }
        return acc;
    }

    double HopfBranchSolver::endpointNorm(const double *a) const
    {
        const double val = endpointDot(a, a);
        return val > 0.0 ? std::sqrt(val) : 0.0;
    }

    double HopfBranchSolver::borderedDot(const double *a, const double *b) const
    {
        double acc = endpointDot(a, b);
        acc += m_weights_bord[m_total_end + 0] * a[m_total_end + 0] * b[m_total_end + 0];
        acc += m_weights_bord[m_total_end + 1] * a[m_total_end + 1] * b[m_total_end + 1];
        acc += m_weights_bord[m_total_end + 2] * a[m_total_end + 2] * b[m_total_end + 2];
        return acc;
    }

    double HopfBranchSolver::borderedNorm(const double *a) const
    {
        const double val = borderedDot(a, a);
        return val > 0.0 ? std::sqrt(val) : 0.0;
    }

    size_t HopfBranchSolver::maxSpatialNodes() const
    {
        size_t mx = 0;
        for (size_t n = 0; n < m_nt; n++) {
            mx = std::max(mx, m_slices[n].nnodes);
        }
        return mx;
    }

    void HopfBranchSolver::initFromHopfPoint(const Collocator &hopf_col, double omega, const complex_t *psi, size_t nt, size_t ncol_time, int parnum)
    {
        freeBuffers();

        m_handle = hopf_col.getLibHandle();
        m_owns_handle = false;
        m_func = hopf_col.func();
        m_fjac = hopf_col.fjac();
        m_pjac = hopf_col.pjac();
        m_node = hopf_col.getNode();
        m_spatial_nstages = hopf_col.getNStages();
        m_np = hopf_col.getNParam();
        m_pmask[0] = 0;
        m_pmask[1] = (parnum >= 0) ? parnum : (int)hopf_col.getContinuationParameterIndex();
        m_unscaled_spatial_period = hopf_col.unscaledSpatialPeriod();
        m_diffusion = hopf_col.getDiffusion();

        m_p = (double *)malloc(m_np * sizeof(double));
        memcpy(m_p, hopf_col.p(), m_np * sizeof(double));
        m_T = (2.0 * M_PI) / std::abs(omega);
        m_c_prev = m_p[0];
        m_T_prev = m_T;
        m_T_scale = std::max(1.0, m_T);
        m_alpha_prev = m_p[m_pmask[1]];

        std::vector<size_t> nn(nt, hopf_col.getNNodes());
        allocateForGrid(nt, ncol_time, nn.data());

        for (size_t n = 0; n <= nt; n++) {
            m_tau[n] = (double)n / (double)nt;
        }
        for (size_t n = 0; n < nt; n++) {
            m_slices[n].initFromCollocator(hopf_col);
        }
        rebuildOffsetsAndWeights();
        rebuildTransfers();

        const Collocator &col_mut = hopf_col;
        const double *y_steady = ((Collocator &)col_mut).y();
        const size_t ns = m_slices[0].n_space;
        for (size_t n = 0; n < nt; n++) {
            for (size_t l = 0; l < m_st; l++) {
                memcpy(&m_y_stages[m_stage_offsets[n] + l * ns], y_steady, ns * sizeof(double));
            }
        }
        memcpy(m_y_prev_stages, m_y_stages, m_total_stages * sizeof(double));
        syncEndpointsFromStages(m_y_prev_end);
        memcpy(m_y_ref_end, m_y_prev_end, m_total_end * sizeof(double));

        // Build spatial phase weight g_space = y_steady_xi / ||y_steady_xi||
        double *dy_dxi = (double *)malloc(ns * sizeof(double));
        eval_slice_spatial_deriv(m_slices[0], y_steady, dy_dxi);
        for (size_t n = 0; n < nt; n++) {
            memcpy(&m_g_space[m_end_offsets[n]], dy_dxi, ns * sizeof(double));
        }
        const double gsp_norm = endpointNorm(m_g_space);
        if (gsp_norm > 0.0) {
            for (size_t i = 0; i < m_total_end; i++) {
                m_g_space[i] /= gsp_norm;
            }
        }
        free(dy_dxi);

        // Normalize Hopf eigenfunction psi in spatial L2 norm so ||dot_Y_0|| = 1
        double psi_l2 = 0.0;
        const double *w_s0 = m_slices[0].c_b;
        const size_t nsub0 = m_slices[0].nSub();
        const size_t s_x = m_slices[0].nstages;
        // Use the exact slice weights on node 0 divided by dtau[0]
        const double inv_dt0 = 1.0 / m_dtau[0];
        for (size_t i = 0; i < ns; i++) {
            const double wi = m_weights_end[i] * inv_dt0;
            psi_l2 += wi * std::norm(psi[i]);
        }
        (void)w_s0; (void)nsub0; (void)s_x;
        psi_l2 = (psi_l2 > 0.0) ? std::sqrt(psi_l2) : 1.0;

        // Branch-switching tangent on endpoints:
        //   dot_Y_0(tau, xi) = sqrt(2) * (psi_R * cos(2*pi*tau) - psi_I * sin(2*pi*tau))
        // Orthogonal temporal phase weight:
        //   g_time_0(tau, xi) = sqrt(2) * (-psi_R * sin(2*pi*tau) - psi_I * cos(2*pi*tau))
        const double sqrt2_inv_norm = std::sqrt(2.0) / psi_l2;
        for (size_t n = 0; n < nt; n++) {
            const double tau_end = m_tau[n + 1];
            const double cs = std::cos(2.0 * M_PI * tau_end);
            const double sn = std::sin(2.0 * M_PI * tau_end);
            double *tang_n = &m_tangent[m_end_offsets[n]];
            double *gt_n = &m_g_time[m_end_offsets[n]];
            for (size_t i = 0; i < ns; i++) {
                const double pr = psi[i].real();
                const double pi = psi[i].imag();
                tang_n[i] = sqrt2_inv_norm * (pr * cs - pi * sn);
                gt_n[i] = sqrt2_inv_norm * (-pr * sn - pi * cs);
            }
        }
        m_tangent[m_total_end + 0] = 0.0; // dc/ds = 0
        m_tangent[m_total_end + 1] = 0.0; // dT/ds = 0
        m_tangent[m_total_end + 2] = 0.0; // dalpha/ds = 0

        // Orthogonalize g_time against g_space and normalize
        const double proj_ts = endpointDot(m_g_space, m_g_time);
        for (size_t i = 0; i < m_total_end; i++) {
            m_g_time[i] -= proj_ts * m_g_space[i];
        }
        const double gt_norm = endpointNorm(m_g_time);
        if (gt_norm > 0.0) {
            for (size_t i = 0; i < m_total_end; i++) {
                m_g_time[i] /= gt_norm;
            }
        }
        // Also ensure m_tangent is unit length and orthogonal to g_space, g_time
        const double proj_ys = endpointDot(m_g_space, m_tangent);
        const double proj_yt = endpointDot(m_g_time, m_tangent);
        for (size_t i = 0; i < m_total_end; i++) {
            m_tangent[i] -= proj_ys * m_g_space[i] + proj_yt * m_g_time[i];
        }
        const double t_norm = borderedNorm(m_tangent);
        if (t_norm > 0.0) {
            for (size_t i = 0; i < m_total_end + 3; i++) {
                m_tangent[i] /= t_norm;
            }
        }
    }

    void HopfBranchSolver::predict(double ds)
    {
        const RadauDGMethod &rm = getRadauDG(m_st);
        const double *tang_Y = m_tangent;

        // Predict parameters from accepted step
        m_p[0] = m_c_prev + ds * m_tangent[m_total_end + 0];
        m_T = m_T_prev + ds * m_tangent[m_total_end + 1];
        m_p[m_pmask[1]] = m_alpha_prev + ds * m_tangent[m_total_end + 2];
        if (m_T <= 1e-12) {
            m_T = m_T_prev;
        }
        updateMassMatricesAll();

        // Predict internal stages via linear/Fourier interpolation between endpoint updates on each element
        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            double *d_prev_cur = m_stage_scratch;
            m_transfers[n].apply(&tang_Y[m_end_offsets[prev]], d_prev_cur);
            const double *d_cur = &tang_Y[m_end_offsets[n]];

            for (size_t l = 0; l < m_st; l++) {
                const double th = rm.c[l];
                double *yl = &m_y_stages[m_stage_offsets[n] + l * ns];
                const double *yl_prev = &m_y_prev_stages[m_stage_offsets[n] + l * ns];
                for (size_t i = 0; i < ns; i++) {
                    yl[i] = yl_prev[i] + ds * ((1.0 - th) * d_prev_cur[i] + th * d_cur[i]);
                }
            }
        }
    }

    void HopfBranchSolver::evaluateFullResidual(const double *y_stages, double c, double T, double alpha, double ds, double *G_out, double *Rp_out, bool store_r_space)
    {
        TWIST_MARK_BEGIN("hopf_residual");
        const double t0 = omp_get_wtime();
        omp_set_max_active_levels(1);
        OpenBLASThreadContext __openblas_context(1);
        const RadauDGMethod &rm = getRadauDG(m_st);
        // Temporarily set parameters on m_p (caller restores if needed, or already set)
        m_p[0] = c;
        m_p[m_pmask[1]] = alpha;
        const double sps = m_p[m_np - 1];

#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < m_nt; n++) {
            SpatialSlice slice_local = m_slices[n];
            slice_local.updateMassMatrices(c, sps);

            const size_t ns = slice_local.n_space;
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            const size_t prev_ns = m_slices[prev].n_space;
            const double *y_prev_end = &y_stages[m_stage_offsets[prev] + (m_st - 1) * prev_ns];
            const double *yn = &y_stages[m_stage_offsets[n]];
            double *Gn = &G_out[m_stage_offsets[n]];

            const size_t tid = (size_t)omp_get_thread_num();
            double *tbuf = m_stage_scratch + tid * m_stage_scratch_stride;
            double *y_in_cur = tbuf;
            double *yp_tmp = y_in_cur + ns;
            double *r_sp = yp_tmp + ns; // m_st * ns
            double *mass_tmp = r_sp + m_st * ns;
            double *mass_work = mass_tmp + ns;

            m_transfers[n].apply(y_prev_end, y_in_cur);

            for (size_t l = 0; l < m_st; l++) {
                slice_local.evaluateResidual(&yn[l * ns], m_p, &r_sp[l * ns], yp_tmp);
            }
            if (store_r_space) {
                memcpy(&m_r_space[m_stage_offsets[n]], r_sp, m_st * ns * sizeof(double));
            }

            const double dtau_T = m_dtau[n] * T;
            for (size_t j = 0; j < m_st; j++) {
                double *Gnj = &Gn[j * ns];
                for (size_t i = 0; i < ns; i++) {
                    mass_tmp[i] = yn[j * ns + i] - y_in_cur[i];
                }
                slice_local.applyMass(mass_tmp, Gnj, 1.0, 0.0, mass_work);
                for (size_t l = 0; l < m_st; l++) {
                    const double scale = -dtau_T * rm.A[j * m_st + l];
                    const double *rl = &r_sp[l * ns];
                    for (size_t i = 0; i < ns; i++) {
                        Gnj[i] += scale * rl[i];
                    }
                }
            }
        }

        // Evaluate 3 scalar conditions on endpoints: psi_space, psi_time, pseudo-arclength
        double psi_sp = 0.0, psi_tm = 0.0, arclen = 0.0;
        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const double *yn_end = &y_stages[m_stage_offsets[n] + (m_st - 1) * ns];
            const double *yref_n = &m_y_ref_end[m_end_offsets[n]];
            const double *yprev_n = &m_y_prev_end[m_end_offsets[n]];
            const double *gs_n = &m_g_space[m_end_offsets[n]];
            const double *gt_n = &m_g_time[m_end_offsets[n]];
            const double *tang_n = &m_tangent[m_end_offsets[n]];
            const double *wn = &m_weights_end[m_end_offsets[n]];

            for (size_t i = 0; i < ns; i++) {
                const double w = wn[i];
                const double dref = yn_end[i] - yref_n[i];
                const double dprev = yn_end[i] - yprev_n[i];
                psi_sp += w * gs_n[i] * dref;
                psi_tm += w * gt_n[i] * dref;
                arclen += w * tang_n[i] * dprev;
            }
        }
        arclen += m_weights_bord[m_total_end + 0] * m_tangent[m_total_end + 0] * (c - m_c_prev)
                + m_weights_bord[m_total_end + 1] * m_tangent[m_total_end + 1] * (T - m_T_prev)
                + m_weights_bord[m_total_end + 2] * m_tangent[m_total_end + 2] * (alpha - m_alpha_prev) - ds;

        Rp_out[0] = psi_sp;
        Rp_out[1] = psi_tm;
        Rp_out[2] = arclen;
        m_timing.t_residual += omp_get_wtime() - t0;
        TWIST_MARK_END("hopf_residual");
    }

    double HopfBranchSolver::computeResidualNorm(const double *G, const double *Rp) const
    {
        double acc = 0.0;
        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const double *wn = &m_weights_end[m_end_offsets[n]];
            for (size_t l = 0; l < m_st; l++) {
                const double *Gnl = &G[m_stage_offsets[n] + l * ns];
                for (size_t i = 0; i < ns; i++) {
                    acc += wn[i] * Gnl[i] * Gnl[i];
                }
            }
        }
        acc += Rp[0] * Rp[0] + Rp[1] * Rp[1] + Rp[2] * Rp[2];
        return std::sqrt(acc);
    }

    void HopfBranchSolver::assembleCondensedSystem(const double *G, bool update_precond)
    {
        TWIST_MARK_BEGIN("hopf_assemble");
        const RadauDGMethod &rm = getRadauDG(m_st);
        const int alpha_idx = m_pmask[1];
        const bool alpha_is_sps = (alpha_idx == (int)(m_np - 1));
        const double sps = m_p[m_np - 1];

        const double t_loop_start = omp_get_wtime();
        updateMassMatricesAll();

        OpenBLASThreadContext __openblas_context(1);

        double sum_factor = 0.0;
        double sum_rhs_p = 0.0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+:sum_factor, sum_rhs_p)
        for (size_t n = 0; n < m_nt; n++) {
            const SpatialSlice *sl = &m_slices[n];
            const size_t ns = sl->n_space;
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            const size_t prev_ns = m_slices[prev].n_space;
            const double *yn = &m_y_stages[m_stage_offsets[n]];
            const double *y_prev_end = &m_y_stages[m_stage_offsets[prev] + (m_st - 1) * prev_ns];
            const double dtau_T = m_dtau[n] * m_T;

            const double tf0 = omp_get_wtime();
            m_condensers[n].factor(sl, m_st, yn, m_p, dtau_T, true);
            const double tf1 = omp_get_wtime();
            sum_factor += (tf1 - tf0);

            const size_t tid = (size_t)omp_get_thread_num();
            double *tbuf = m_stage_scratch + tid * m_stage_scratch_stride;
            //Note: size per_thread is ns*(m_st+6)+64; we need rhs_tmp(m_st*ns) + dk_tmp(m_st*ns) + y_in(ns) + dp_s(ns) + mass_tmp(ns) + mass_work(ns) -> ns*(2*m_st+4)
            // Let's allocate safely on stack/ensure stage_scratch has (2*m_st+6) in allocateForGrid, OR use m_delta_stages[m_stage_offsets[n]] as dk_tmp!
            double *rhs_tmp = tbuf; // size m_st * ns
            double *dk_tmp = &m_delta_stages[m_stage_offsets[n]]; // size m_st * ns (safe disjoint per element n!)
            double *y_in = rhs_tmp + m_st * ns;
            double *dp_s = y_in + ns;
            double *mass_tmp = dp_s + ns;
            double *mass_work = mass_tmp + ns;

            m_transfers[n].apply(y_prev_end, y_in);

            // 1. Condensed RHS: b_{Y,n} = [A_n^(-1) (-G_n)]_{st}
            if (G != nullptr) {
                const double *Gn = &G[m_stage_offsets[n]];
                for (size_t i = 0; i < m_st * ns; i++) {
                    rhs_tmp[i] = -Gn[i];
                }
                m_condensers[n].solveLocalStages(rhs_tmp, dk_tmp, true);
                memcpy(&m_b_Y[m_end_offsets[n]], &dk_tmp[(m_st - 1) * ns], ns * sizeof(double));
            }
            else {
                memset(&m_b_Y[m_end_offsets[n]], 0, ns * sizeof(double));
            }

            // 2. Column 0 (c): dG_n/dc
            for (size_t l = 0; l < m_st; l++) {
                for (size_t i = 0; i < ns; i++) {
                    mass_tmp[i] = yn[l * ns + i] - y_in[i];
                }
                sl->applyMassDerivC(mass_tmp, &rhs_tmp[l * ns], 1.0, 0.0, mass_work);
            }
            for (size_t l = 0; l < m_st; l++) {
                m_condensers[n].evalSpatialParamDeriv(l, &yn[l * ns], m_p, 0, dp_s);
                for (size_t j = 0; j < m_st; j++) {
                    const double scale = -dtau_T * rm.A[j * m_st + l];
                    double *rj = &rhs_tmp[j * ns];
                    for (size_t i = 0; i < ns; i++) {
                        rj[i] += scale * dp_s[i];
                    }
                }
            }
            m_condensers[n].solveLocalStages(rhs_tmp, dk_tmp, true);
            memcpy(&m_J_Yp[0 * m_total_end + m_end_offsets[n]], &dk_tmp[(m_st - 1) * ns], ns * sizeof(double));

            // 3. Column 1 (T): dG_n/dT = -dtau_n * sum_l A_{j,l} R_space(Y_{n,l})
            memset(rhs_tmp, 0, m_st * ns * sizeof(double));
            const double *rn_sp = &m_r_space[m_stage_offsets[n]];
            for (size_t j = 0; j < m_st; j++) {
                double *rj = &rhs_tmp[j * ns];
                for (size_t l = 0; l < m_st; l++) {
                    const double scale = -m_dtau[n] * rm.A[j * m_st + l];
                    const double *r_l = &rn_sp[l * ns];
                    for (size_t i = 0; i < ns; i++) {
                        rj[i] += scale * r_l[i];
                    }
                }
            }
            m_condensers[n].solveLocalStages(rhs_tmp, dk_tmp, true);
            memcpy(&m_J_Yp[1 * m_total_end + m_end_offsets[n]], &dk_tmp[(m_st - 1) * ns], ns * sizeof(double));

            // 4. Column 2 (alpha): dG_n/dalpha
            if (alpha_is_sps) {
                for (size_t l = 0; l < m_st; l++) {
                    for (size_t i = 0; i < ns; i++) {
                        mass_tmp[i] = yn[l * ns + i] - y_in[i];
                    }
                    sl->applyMassDerivSps(sps, mass_tmp, &rhs_tmp[l * ns], 1.0, 0.0, mass_work);
                }
            }
            else {
                memset(rhs_tmp, 0, m_st * ns * sizeof(double));
            }
            for (size_t l = 0; l < m_st; l++) {
                m_condensers[n].evalSpatialParamDeriv(l, &yn[l * ns], m_p, alpha_idx, dp_s);
                for (size_t j = 0; j < m_st; j++) {
                    const double scale = -dtau_T * rm.A[j * m_st + l];
                    double *rj = &rhs_tmp[j * ns];
                    for (size_t i = 0; i < ns; i++) {
                        rj[i] += scale * dp_s[i];
                    }
                }
            }
            m_condensers[n].solveLocalStages(rhs_tmp, dk_tmp, true);
            memcpy(&m_J_Yp[2 * m_total_end + m_end_offsets[n]], &dk_tmp[(m_st - 1) * ns], ns * sizeof(double));
            sum_rhs_p += omp_get_wtime() - tf1;
        }
        const double t_loop = omp_get_wtime() - t_loop_start;
        const double sum_tot = sum_factor + sum_rhs_p;
        if (sum_tot > 0.0) {
            m_timing.t_condense_factor += t_loop * (sum_factor / sum_tot);
            m_timing.t_condense_rhs_p += t_loop * (sum_rhs_p / sum_tot);
        }

        if (update_precond) {
            TWIST_MARK_BEGIN("hopf_precond_update");
            const double tp0 = omp_get_wtime();
            const size_t M = (m_nt <= 4) ? m_nt : std::clamp<size_t>(m_win_M, 1, m_nt);
            double *y_end = m_bord_sol; // temporary use before GMRES
            syncEndpointsFromStages(y_end);
            if (M == m_nt) {
                m_precond.update(m_nt, m_slices, m_condensers, y_end, m_dtau, m_T, m_p, 1.0);
            }
            else if (M >= 2) {
                std::vector<SpatialSlice> win_slices(M);
                std::vector<double> win_dtau(M, 0.0);
                size_t W_end = 0;
                for (size_t w = 0; w < M; w++) {
                    const size_t ws = (w * m_nt) / M;
                    const size_t we = ((w + 1) * m_nt) / M;
                    const size_t e = we - 1;
                    win_slices[w] = m_slices[e];
                    W_end += m_slices[e].n_space;
                    for (size_t n = ws; n < we; n++) {
                        win_dtau[w] += m_dtau[n];
                    }
                }
                std::vector<double> win_y_end(W_end);
                size_t woff = 0;
                for (size_t w = 0; w < M; w++) {
                    const size_t e = ((w + 1) * m_nt) / M - 1;
                    const size_t ns_e = m_slices[e].n_space;
                    memcpy(&win_y_end[woff], &y_end[m_end_offsets[e]], ns_e * sizeof(double));
                    woff += ns_e;
                }
                m_precond.update(M, win_slices.data(), nullptr, win_y_end.data(), win_dtau.data(), m_T, m_p, 1.0);
            }
            m_timing.t_precond_update += omp_get_wtime() - tp0;
            TWIST_MARK_END("hopf_precond_update");
        }
        TWIST_MARK_END("hopf_assemble");
    }

    int HopfBranchSolver::solveBorderedGMRES(const double *rhs_Y, const double *rhs_p, const double *row3_Y, const double *row3_p, double *sol_Y, double *sol_p, double rtol, int maxiter, bool verbose)
    {
        TWIST_MARK_BEGIN("hopf_gmres");
        const double t_gmres_start = omp_get_wtime();
        omp_set_max_active_levels(1);
        OpenBLASThreadContext __openblas_context(1);

        const size_t N_end = m_total_end;
        const size_t N_tot = N_end + 3;
        const double w0 = m_weights_bord[N_end + 0];
        const double w1 = m_weights_bord[N_end + 1];
        const double w2 = m_weights_bord[N_end + 2];

        size_t max_ns = 0;
        for (size_t n = 0; n < m_nt; n++) {
            max_ns = std::max(max_ns, m_slices[n].n_space);
        }

        double orig_rhs_sq = w0 * rhs_p[0] * rhs_p[0] + w1 * rhs_p[1] * rhs_p[1] + w2 * rhs_p[2] * rhs_p[2];
        double orig_J_sq = 0.0;
        for (size_t i = 0; i < N_end; i++) {
            orig_rhs_sq += m_weights_end[i] * rhs_Y[i] * rhs_Y[i];
            const double j0i = m_J_Yp[0 * N_end + i];
            const double j1i = m_J_Yp[1 * N_end + i];
            const double j2i = m_J_Yp[2 * N_end + i];
            orig_J_sq += m_weights_end[i] * (j0i * j0i + j1i * j1i + j2i * j2i);
        }
        const double orig_rhs_norm = std::sqrt(std::max(0.0, orig_rhs_sq));
        const double orig_J_norm = std::sqrt(std::max(0.0, orig_J_sq));
        const double orig_scale = std::max(1e-12, std::max(orig_rhs_norm, orig_J_norm));

        for (size_t M = (m_nt <= 4) ? m_nt : std::max<size_t>(1, m_win_M / 2); M < m_nt;) {
            std::vector<size_t> w_start(M), w_end(M), end_idx(M), w_offs(M + 1, 0);
            std::vector<double> win_dtau(M, 0.0);
            for (size_t w = 0; w < M; w++) {
                w_start[w] = (w * m_nt) / M;
                w_end[w] = ((w + 1) * m_nt) / M;
                end_idx[w] = w_end[w] - 1;
                w_offs[w + 1] = w_offs[w] + m_slices[end_idx[w]].n_space;
                for (size_t n = w_start[w]; n < w_end[w]; n++) {
                    win_dtau[w] += m_dtau[n];
                }
            }
            const size_t W_end = w_offs[M];
            const size_t W_tot = W_end + 3;

            std::vector<double> g_end(W_end, 0.0);
            std::vector<double> J_win_end(3 * W_end, 0.0);
            std::vector<double> V_win_lumped(3 * W_end, 0.0);
            std::vector<double> w_weights(W_tot, 0.0);
            std::vector<double> vt_g_win(3 * M, 0.0), vt_J_win(9 * M, 0.0);

            for (size_t w = 0; w < M; w++) {
                const size_t e = end_idx[w];
                const size_t ns_e = m_slices[e].n_space;
                const double scale_w = win_dtau[w] / m_dtau[e];
                for (size_t i = 0; i < ns_e; i++) {
                    w_weights[w_offs[w] + i] = m_weights_end[m_end_offsets[e] + i] * scale_w;
                    V_win_lumped[0 * W_end + w_offs[w] + i] = m_g_space[m_end_offsets[e] + i];
                    V_win_lumped[1 * W_end + w_offs[w] + i] = m_g_time[m_end_offsets[e] + i];
                    V_win_lumped[2 * W_end + w_offs[w] + i] = row3_Y[m_end_offsets[e] + i];
                }
            }
            w_weights[W_end + 0] = w0;
            w_weights[W_end + 1] = w1;
            w_weights[W_end + 2] = w2;

            // Pre-eliminate window RHS (g) and parameter columns (J_win)
            const double t_pe0 = omp_get_wtime();
#pragma omp parallel for schedule(static) if (M > 1)
            for (size_t w = 0; w < M; w++) {
                std::vector<double> cur(max_ns), nxt(max_ns);
                const size_t s = w_start[w];
                const size_t e = w_end[w] - 1;

                auto run_col = [&](const double *src_Y, double *end_out, double *dot_out) {
                    const size_t ns0 = m_slices[s].n_space;
                    memcpy(cur.data(), &src_Y[m_end_offsets[s]], ns0 * sizeof(double));
                    for (size_t i = 0; i < ns0; i++) {
                        const double wt = m_weights_end[m_end_offsets[s] + i] * cur[i];
                        dot_out[0] += wt * m_g_space[m_end_offsets[s] + i];
                        dot_out[1] += wt * m_g_time[m_end_offsets[s] + i];
                        dot_out[2] += wt * row3_Y[m_end_offsets[s] + i];
                    }
                    for (size_t n = s + 1; n <= e; n++) {
                        const size_t ns_n = m_slices[n].n_space;
                        m_condensers[n].applyCondensedC(cur.data(), m_transfers[n], nxt.data(), true);
                        for (size_t i = 0; i < ns_n; i++) {
                            cur[i] = src_Y[m_end_offsets[n] + i] + nxt[i];
                            const double wt = m_weights_end[m_end_offsets[n] + i] * cur[i];
                            dot_out[0] += wt * m_g_space[m_end_offsets[n] + i];
                            dot_out[1] += wt * m_g_time[m_end_offsets[n] + i];
                            dot_out[2] += wt * row3_Y[m_end_offsets[n] + i];
                        }
                    }
                    memcpy(&end_out[w_offs[w]], cur.data(), m_slices[e].n_space * sizeof(double));
                };

                run_col(rhs_Y, g_end.data(), &vt_g_win[3 * w]);
                for (size_t q = 0; q < 3; q++) {
                    run_col(&m_J_Yp[q * N_end], &J_win_end[q * W_end], &vt_J_win[9 * w + 3 * q]);
                }
            }

            double rhs_p_win[3] = { rhs_p[0], rhs_p[1], rhs_p[2] };
            double D_tilde[9] = {
                0.0, 0.0, w0 * row3_p[0],
                0.0, 0.0, w1 * row3_p[1],
                0.0, 0.0, w2 * row3_p[2]
            };
            for (size_t w = 0; w < M; w++) {
                for (size_t r_idx = 0; r_idx < 3; r_idx++) {
                    rhs_p_win[r_idx] -= vt_g_win[3 * w + r_idx];
                    for (size_t q = 0; q < 3; q++) {
                        D_tilde[q * 3 + r_idx] -= vt_J_win[9 * w + 3 * q + r_idx];
                    }
                }
            }
            m_timing.t_gmres_preelim += omp_get_wtime() - t_pe0;

            memcpy(m_bord_rhs, g_end.data(), W_end * sizeof(double));
            m_bord_rhs[W_end + 0] = rhs_p_win[0];
            m_bord_rhs[W_end + 1] = rhs_p_win[1];
            m_bord_rhs[W_end + 2] = rhs_p_win[2];

            double win_beta0_sq = 0.0;
            for (size_t i = 0; i < W_tot; i++) {
                win_beta0_sq += w_weights[i] * m_bord_rhs[i] * m_bord_rhs[i];
            }
            const double win_beta0 = std::sqrt(std::max(0.0, win_beta0_sq));
            double win_J_sq = 0.0;
            for (size_t i = 0; i < W_end; i++) {
                const double j0i = J_win_end[0 * W_end + i];
                const double j1i = J_win_end[1 * W_end + i];
                const double j2i = J_win_end[2 * W_end + i];
                win_J_sq += w_weights[i] * (j0i * j0i + j1i * j1i + j2i * j2i);
            }
            const double win_J_norm = std::sqrt(std::max(0.0, win_J_sq));
            const double win_scale = std::max(win_beta0, win_J_norm);

            // Subdivide into finer windows if single-window exponential amplification risks roundoff loss
            if (win_scale > 500.0 * orig_scale) {
                M = std::min(m_nt, 2 * M);
                m_win_M = M;
                continue;
            }
            m_win_M = M;

            auto solve_direct_single_window = [&]() -> bool {
                // Assemble exact condensed slice matrices in parallel and solve the single-window system directly with dense LU
                std::vector<std::vector<double>> C_mats(m_nt);
#pragma omp parallel for schedule(dynamic, 1)
                for (size_t n = 0; n < m_nt; n++) {
                    const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
                    C_mats[n].resize(m_slices[n].n_space * m_slices[prev].n_space);
                    m_condensers[n].computeCondensedMatrix(m_transfers[n], C_mats[n].data());
                }

                const size_t ns0 = m_slices[0].n_space;
                std::vector<double> P_cur(ns0 * W_end);
                memcpy(P_cur.data(), C_mats[0].data(), ns0 * W_end * sizeof(double));
                std::vector<double> P_next;
                std::vector<double> B_rows(3 * W_end, 0.0);
                std::vector<double> wt_rows;

                for (size_t n = 0; n < m_nt; n++) {
                    const size_t ns_n = m_slices[n].n_space;
                    const size_t off_n = m_end_offsets[n];
                    if (n > 0) {
                        const size_t ns_prev = m_slices[n - 1].n_space;
                        P_next.resize(ns_n * W_end);
                        dgemm("N", "N", (int)ns_n, (int)W_end, (int)ns_prev, 1.0, C_mats[n].data(), (int)ns_n, P_cur.data(), (int)ns_prev, 0.0, P_next.data(), (int)ns_n);
                        P_cur.swap(P_next);
                    }
                    wt_rows.resize(ns_n * 3);
                    for (size_t i = 0; i < ns_n; i++) {
                        const double wi = m_weights_end[off_n + i];
                        wt_rows[0 * ns_n + i] = wi * m_g_space[off_n + i];
                        wt_rows[1 * ns_n + i] = wi * m_g_time[off_n + i];
                        wt_rows[2 * ns_n + i] = wi * row3_Y[off_n + i];
                    }
                    dgemm("T", "N", 3, (int)W_end, (int)ns_n, 1.0, wt_rows.data(), (int)ns_n, P_cur.data(), (int)ns_n, 1.0, B_rows.data(), 3);
                }

                std::vector<double> A_full(W_tot * W_tot, 0.0);
                for (size_t j = 0; j < W_end; j++) {
                    for (size_t i = 0; i < W_end; i++) {
                        A_full[j * W_tot + i] = ((i == j) ? 1.0 : 0.0) - P_cur[j * W_end + i];
                    }
                    A_full[j * W_tot + W_end + 0] = B_rows[j * 3 + 0];
                    A_full[j * W_tot + W_end + 1] = B_rows[j * 3 + 1];
                    A_full[j * W_tot + W_end + 2] = B_rows[j * 3 + 2];
                }
                for (size_t q = 0; q < 3; q++) {
                    for (size_t i = 0; i < W_end; i++) {
                        A_full[(W_end + q) * W_tot + i] = J_win_end[q * W_end + i];
                    }
                    A_full[(W_end + q) * W_tot + W_end + 0] = D_tilde[q * 3 + 0];
                    A_full[(W_end + q) * W_tot + W_end + 1] = D_tilde[q * 3 + 1];
                    A_full[(W_end + q) * W_tot + W_end + 2] = D_tilde[q * 3 + 2];
                }

                memcpy(m_bord_sol, m_bord_rhs, W_tot * sizeof(double));
                std::vector<int> ipiv(W_tot);
                int info_lu = 0;
                dgetrf((int)W_tot, (int)W_tot, A_full.data(), (int)W_tot, ipiv.data(), &info_lu);
                if (info_lu == 0) {
                    dgetrs("N", (int)W_tot, 1, A_full.data(), (int)W_tot, ipiv.data(), m_bord_sol, (int)W_tot, &info_lu);
                }
                if (info_lu != 0) {
                    return false;
                }

                sol_p[0] = m_bord_sol[W_end + 0];
                sol_p[1] = m_bord_sol[W_end + 1];
                sol_p[2] = m_bord_sol[W_end + 2];

                std::vector<double> cur(max_ns), nxt(max_ns);
                memcpy(cur.data(), m_bord_sol, W_end * sizeof(double));
                const double c0 = sol_p[0], c1 = sol_p[1], c2 = sol_p[2];
                for (size_t n = 0; n < m_nt; n++) {
                    const size_t ns_n = m_slices[n].n_space;
                    const size_t ns_prev = m_slices[(n == 0) ? (m_nt - 1) : (n - 1)].n_space;
                    double *sn = &sol_Y[m_end_offsets[n]];
                    dgemv("N", (int)ns_n, (int)ns_prev, 1.0, C_mats[n].data(), (int)ns_n, cur.data(), 1, 0.0, nxt.data(), 1);
                    const double *rn = &rhs_Y[m_end_offsets[n]];
                    const double *j0 = &m_J_Yp[0 * N_end + m_end_offsets[n]];
                    const double *j1 = &m_J_Yp[1 * N_end + m_end_offsets[n]];
                    const double *j2 = &m_J_Yp[2 * N_end + m_end_offsets[n]];
                    for (size_t i = 0; i < ns_n; i++) {
                        sn[i] = rn[i] + nxt[i] - c0 * j0[i] - c1 * j1[i] - c2 * j2[i];
                        cur[i] = sn[i];
                    }
                }
                return true;
            };

            const bool use_pc = (M >= 2);
            if (use_pc && m_precond.nt() != M) {
                const double t_pu0 = omp_get_wtime();
                std::vector<SpatialSlice> win_slices(M);
                std::vector<double> win_y_end(W_end);
                for (size_t w = 0; w < M; w++) {
                    const size_t e = end_idx[w];
                    const size_t ns_e = m_slices[e].n_space;
                    win_slices[w] = m_slices[e];
                    memcpy(&win_y_end[w_offs[w]], &m_y_stages[m_stage_offsets[e] + (m_st - 1) * ns_e], ns_e * sizeof(double));
                }
                m_precond.update(M, win_slices.data(), nullptr, win_y_end.data(), win_dtau.data(), m_T, m_p, 1.0);
                m_timing.t_precond_update += omp_get_wtime() - t_pu0;
            }

            auto win_dot = [&](const double *a, const double *b) -> double {
                double acc = 0.0;
                for (size_t i = 0; i < W_end; i++) {
                    acc += w_weights[i] * a[i] * b[i];
                }
                return acc;
            };

            std::vector<double> Minv_U_win(3 * W_end, 0.0);
            double S3_win[9];
            memcpy(S3_win, D_tilde, 9 * sizeof(double));
            for (size_t q = 0; q < 3; q++) {
                const double *Uq = &J_win_end[q * W_end];
                double *MUq = &Minv_U_win[q * W_end];
                if (use_pc) {
                    m_precond.apply(Uq, MUq);
                }
                else {
                    memcpy(MUq, Uq, W_end * sizeof(double));
                }
                S3_win[q * 3 + 0] -= win_dot(&V_win_lumped[0 * W_end], MUq);
                S3_win[q * 3 + 1] -= win_dot(&V_win_lumped[1 * W_end], MUq);
                S3_win[q * 3 + 2] -= win_dot(&V_win_lumped[2 * W_end], MUq);
            }
            int ipiv_w[3], info_w = 0;
            dgetrf(3, 3, S3_win, 3, ipiv_w, &info_w);
            const bool s3_w_ok = (info_w == 0);

            std::vector<double> loc_v(3 * M, 0.0);
            std::vector<double> win_cur_buf(M * max_ns), win_nxt_buf(M * max_ns);

            auto A_win_op = [&](const RP(double) u_in, RP(double) out, size_t) -> int {
                const double t_a0 = omp_get_wtime();
                const double *uY = u_in;
                const double *xp = u_in + W_end;
                double *outY = out;
                double *outp = out + W_end;

                // Only parallelize across windows inside inner GMRES for large slices to avoid small-task OpenMP barrier overhead
#pragma omp parallel for schedule(static) if (M > 1 && max_ns >= 800)
                for (size_t w = 0; w < M; w++) {
                    double *cur = &win_cur_buf[w * max_ns];
                    double *nxt = &win_nxt_buf[w * max_ns];
                    const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
                    const size_t s = w_start[w];
                    const size_t e = w_end[w] - 1;
                    memcpy(cur, &uY[w_offs[prev_w]], m_slices[end_idx[prev_w]].n_space * sizeof(double));

                    double lv0 = 0.0, lv1 = 0.0, lv2 = 0.0;
                    for (size_t n = s; n <= e; n++) {
                        const size_t ns_n = m_slices[n].n_space;
                        m_condensers[n].applyCondensedC(cur, m_transfers[n], nxt, true);
                        for (size_t i = 0; i < ns_n; i++) {
                            cur[i] = nxt[i];
                            const double wt = m_weights_end[m_end_offsets[n] + i] * cur[i];
                            lv0 += wt * m_g_space[m_end_offsets[n] + i];
                            lv1 += wt * m_g_time[m_end_offsets[n] + i];
                            lv2 += wt * row3_Y[m_end_offsets[n] + i];
                        }
                    }
                    loc_v[3 * w + 0] = lv0;
                    loc_v[3 * w + 1] = lv1;
                    loc_v[3 * w + 2] = lv2;

                    const size_t ns_e = m_slices[e].n_space;
                    const double *uw = &uY[w_offs[w]];
                    const double *j0 = &J_win_end[0 * W_end + w_offs[w]];
                    const double *j1 = &J_win_end[1 * W_end + w_offs[w]];
                    const double *j2 = &J_win_end[2 * W_end + w_offs[w]];
                    double *ow = &outY[w_offs[w]];
                    const double c0 = xp[0], c1 = xp[1], c2 = xp[2];
                    for (size_t i = 0; i < ns_e; i++) {
                        ow[i] = uw[i] - cur[i] + c0 * j0[i] + c1 * j1[i] + c2 * j2[i];
                    }
                }

                outp[0] = D_tilde[0 * 3 + 0] * xp[0] + D_tilde[1 * 3 + 0] * xp[1] + D_tilde[2 * 3 + 0] * xp[2];
                outp[1] = D_tilde[0 * 3 + 1] * xp[0] + D_tilde[1 * 3 + 1] * xp[1] + D_tilde[2 * 3 + 1] * xp[2];
                outp[2] = D_tilde[0 * 3 + 2] * xp[0] + D_tilde[1 * 3 + 2] * xp[1] + D_tilde[2 * 3 + 2] * xp[2];
                for (size_t w = 0; w < M; w++) {
                    outp[0] += loc_v[3 * w + 0];
                    outp[1] += loc_v[3 * w + 1];
                    outp[2] += loc_v[3 * w + 2];
                }
                m_timing.t_gmres_A_op += omp_get_wtime() - t_a0;
                return 0;
            };

            auto M_win_op = [&](const RP(double) r, RP(double) z, size_t) -> int {
                const double t_m0 = omp_get_wtime();
                const double *rY = r;
                const double *rp = r + W_end;
                double *zY = z;
                double *zp = z + W_end;

                if (use_pc) {
                    m_precond.apply(rY, zY);
                }
                else {
                    memcpy(zY, rY, W_end * sizeof(double));
                }

                if (s3_w_ok) {
                    double rhs3[3] = {
                        rp[0] - win_dot(&V_win_lumped[0 * W_end], zY),
                        rp[1] - win_dot(&V_win_lumped[1 * W_end], zY),
                        rp[2] - win_dot(&V_win_lumped[2 * W_end], zY)
                    };
                    int info_s = 0;
                    dgetrs("N", 3, 1, S3_win, 3, ipiv_w, rhs3, 3, &info_s);
                    zp[0] = rhs3[0];
                    zp[1] = rhs3[1];
                    zp[2] = rhs3[2];
                    const double p0 = zp[0], p1 = zp[1], p2 = zp[2];
                    const double *mu0 = &Minv_U_win[0 * W_end];
                    const double *mu1 = &Minv_U_win[1 * W_end];
                    const double *mu2 = &Minv_U_win[2 * W_end];
                    for (size_t i = 0; i < W_end; i++) {
                        zY[i] -= p0 * mu0[i] + p1 * mu1[i] + p2 * mu2[i];
                    }
                }
                else {
                    zp[0] = rp[0];
                    zp[1] = rp[1];
                    zp[2] = rp[2];
                }
                m_timing.t_gmres_M_op += omp_get_wtime() - t_m0;
                return 0;
            };

            memset(m_bord_sol, 0, W_tot * sizeof(double));
            const double rtol_win = (win_beta0 > 0.0 && orig_rhs_norm < win_beta0) ? std::max(1e-13, rtol * (orig_rhs_norm / win_beta0)) : rtol;

            int niters = 0;
            const int try_max = (M == 1 && W_end <= 1024) ? std::min(maxiter, 60) : maxiter;
            const int status = Krylov::gmres(A_win_op, M_win_op, W_tot, m_bord_sol, m_bord_rhs, w_weights.data(), m_gmres_work, 100, try_max, rtol_win, 1e-13, &niters, verbose);
            m_timing.n_gmres_iters += (size_t)niters;
            if (status != 0 && M == 1 && W_end <= 1024) {
                const double t_d0 = omp_get_wtime();
                const bool d_ok = solve_direct_single_window();
                m_timing.t_gmres_direct_lu += omp_get_wtime() - t_d0;
                if (d_ok) {
                    m_timing.t_gmres_total += omp_get_wtime() - t_gmres_start;
                    TWIST_MARK_END("hopf_gmres");
                    return niters;
                }
            }

            sol_p[0] = m_bord_sol[W_end + 0];
            sol_p[1] = m_bord_sol[W_end + 1];
            sol_p[2] = m_bord_sol[W_end + 2];

            // Reconstruct all slice updates sol_Y inside each window
            const double t_rec0 = omp_get_wtime();
#pragma omp parallel for schedule(static) if (M > 1)
            for (size_t w = 0; w < M; w++) {
                std::vector<double> cur(max_ns);
                const size_t prev_w = (w == 0) ? (M - 1) : (w - 1);
                const size_t s = w_start[w];
                const size_t e = w_end[w] - 1;
                memcpy(cur.data(), &m_bord_sol[w_offs[prev_w]], m_slices[end_idx[prev_w]].n_space * sizeof(double));

                const double c0 = sol_p[0], c1 = sol_p[1], c2 = sol_p[2];
                for (size_t n = s; n <= e; n++) {
                    const size_t ns_n = m_slices[n].n_space;
                    double *sn = &sol_Y[m_end_offsets[n]];
                    m_condensers[n].applyCondensedC(cur.data(), m_transfers[n], sn, true);
                    const double *rn = &rhs_Y[m_end_offsets[n]];
                    const double *j0 = &m_J_Yp[0 * N_end + m_end_offsets[n]];
                    const double *j1 = &m_J_Yp[1 * N_end + m_end_offsets[n]];
                    const double *j2 = &m_J_Yp[2 * N_end + m_end_offsets[n]];
                    for (size_t i = 0; i < ns_n; i++) {
                        sn[i] = rn[i] + sn[i] - c0 * j0[i] - c1 * j1[i] - c2 * j2[i];
                        cur[i] = sn[i];
                    }
                }
            }
            m_timing.t_gmres_reconstruct += omp_get_wtime() - t_rec0;
            m_timing.t_gmres_total += omp_get_wtime() - t_gmres_start;
            TWIST_MARK_END("hopf_gmres");
            (void)status;
            return niters;
        }

        // Direct uncondensed path when M == m_nt (e.g. very small nt <= 4)
        if (m_precond.nt() != m_nt) {
            const double t_pu0 = omp_get_wtime();
            double *y_end = m_bord_rhs; // temporary before copying rhs_Y
            syncEndpointsFromStages(y_end);
            m_precond.update(m_nt, m_slices, m_condensers, y_end, m_dtau, m_T, m_p, 1.0);
            m_timing.t_precond_update += omp_get_wtime() - t_pu0;
        }
        double S3[9]; // column-major 3x3
        double D3[9] = {
            0.0, 0.0, w0 * row3_p[0],
            0.0, 0.0, w1 * row3_p[1],
            0.0, 0.0, w2 * row3_p[2]
        };

        for (size_t q = 0; q < 3; q++) {
            const double *Uq = &m_J_Yp[q * N_end];
            double *Minv_Uq = &m_Minv_U[q * N_end];
            m_precond.apply(Uq, Minv_Uq);

            const double v0 = endpointDot(m_g_space, Minv_Uq);
            const double v1 = endpointDot(m_g_time, Minv_Uq);
            const double v2 = endpointDot(row3_Y, Minv_Uq);

            S3[q * 3 + 0] = D3[q * 3 + 0] - v0;
            S3[q * 3 + 1] = D3[q * 3 + 1] - v1;
            S3[q * 3 + 2] = D3[q * 3 + 2] - v2;
        }

        int ipiv3[3], info3 = 0;
        dgetrf(3, 3, S3, 3, ipiv3, &info3);
        const bool s3_ok = (info3 == 0);

        auto A_op = [&](const RP(double) x, RP(double) y, size_t) -> int {
            const double t_a0 = omp_get_wtime();
            const double *xY = x;
            const double *xp = x + N_end;
            double *yY = y;
            double *yp = y + N_end;

            // yY = xY - C_tilde * xY + J_Yp * xp
#pragma omp parallel for schedule(static) if (m_nt > 1 && max_ns >= 800)
            for (size_t n = 0; n < m_nt; n++) {
                const size_t ns = m_slices[n].n_space;
                const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
                const size_t off_n = m_end_offsets[n];
                const size_t off_prev = m_end_offsets[prev];
                double *yn_out = &yY[off_n];

                m_condensers[n].applyCondensedC(&xY[off_prev], m_transfers[n], yn_out, true);

                const double *xn = &xY[off_n];
                const double *u0 = &m_J_Yp[0 * N_end + off_n];
                const double *u1 = &m_J_Yp[1 * N_end + off_n];
                const double *u2 = &m_J_Yp[2 * N_end + off_n];
                const double c0 = xp[0], c1 = xp[1], c2 = xp[2];

                for (size_t i = 0; i < ns; i++) {
                    yn_out[i] = xn[i] - yn_out[i] + c0 * u0[i] + c1 * u1[i] + c2 * u2[i];
                }
            }

            yp[0] = endpointDot(m_g_space, xY);
            yp[1] = endpointDot(m_g_time, xY);
            yp[2] = endpointDot(row3_Y, xY) + w0 * row3_p[0] * xp[0] + w1 * row3_p[1] * xp[1] + w2 * row3_p[2] * xp[2];
            m_timing.t_gmres_A_op += omp_get_wtime() - t_a0;
            return 0;
        };

        auto M_op = [&](const RP(double) r, RP(double) z, size_t) -> int {
            const double t_m0 = omp_get_wtime();
            const double *rY = r;
            const double *rp = r + N_end;
            double *zY = z;
            double *zp = z + N_end;

            m_precond.apply(rY, zY);

            if (s3_ok) {
                double rhs3[3];
                rhs3[0] = rp[0] - endpointDot(m_g_space, zY);
                rhs3[1] = rp[1] - endpointDot(m_g_time, zY);
                rhs3[2] = rp[2] - endpointDot(row3_Y, zY);
                int info_s = 0;
                dgetrs("N", 3, 1, S3, 3, ipiv3, rhs3, 3, &info_s);
                zp[0] = rhs3[0];
                zp[1] = rhs3[1];
                zp[2] = rhs3[2];

                const double *mu0 = &m_Minv_U[0 * N_end];
                const double *mu1 = &m_Minv_U[1 * N_end];
                const double *mu2 = &m_Minv_U[2 * N_end];
                const double p0 = zp[0], p1 = zp[1], p2 = zp[2];
                for (size_t i = 0; i < N_end; i++) {
                    zY[i] -= p0 * mu0[i] + p1 * mu1[i] + p2 * mu2[i];
                }
            }
            else {
                zp[0] = rp[0];
                zp[1] = rp[1];
                zp[2] = rp[2];
            }
            m_timing.t_gmres_M_op += omp_get_wtime() - t_m0;
            return 0;
        };

        memcpy(m_bord_rhs, rhs_Y, N_end * sizeof(double));
        m_bord_rhs[N_end + 0] = rhs_p[0];
        m_bord_rhs[N_end + 1] = rhs_p[1];
        m_bord_rhs[N_end + 2] = rhs_p[2];
        memset(m_bord_sol, 0, N_tot * sizeof(double));

        int niters = 0;
        const int status = Krylov::gmres(A_op, M_op, N_tot, m_bord_sol, m_bord_rhs, m_weights_bord, m_gmres_work, 100, maxiter, rtol, 1e-13, &niters, verbose);
        m_timing.n_gmres_iters += (size_t)niters;

        memcpy(sol_Y, m_bord_sol, N_end * sizeof(double));
        sol_p[0] = m_bord_sol[N_end + 0];
        sol_p[1] = m_bord_sol[N_end + 1];
        sol_p[2] = m_bord_sol[N_end + 2];

        m_timing.t_gmres_total += omp_get_wtime() - t_gmres_start;
        TWIST_MARK_END("hopf_gmres");
        (void)status;
        return niters;
    }

    bool HopfBranchSolver::solveNewton(double ds, double tol, double min_damp, int max_iter, bool verbose)
    {
        omp_set_max_active_levels(1);
        OpenBLASThreadContext __openblas_context(1);

        const RadauDGMethod &rm = getRadauDG(m_st);
        double c = m_p[0];
        double T = m_T;
        double alpha = m_p[m_pmask[1]];
        const int alpha_idx = m_pmask[1];
        const bool alpha_is_sps = (alpha_idx == (int)(m_np - 1));

        double Rp[3], Rp_trial[3];
        evaluateFullResidual(m_y_stages, c, T, alpha, ds, m_G, Rp, true);
        double norm_res = computeResidualNorm(m_G, Rp);

        if (verbose) {
            fmt::println("  [HopfNewton] init |R| = {:.4e}, c = {:.6f}, T = {:.6f}, alpha = {:.6f}", norm_res, c, T, alpha);
        }
        if (norm_res < tol) {
            assembleCondensedSystem(nullptr, true);
            return true;
        }

        double damp = 1.0;
        bool converged = false;
        double *sol_Y = (double *)malloc(m_total_end * sizeof(double));
        double sol_p[3];

        for (int iter = 0; iter < max_iter; iter++) {
            m_timing.n_newton_iters++;
            // Re-assemble & update preconditioner on iter 0 or when convergence slowed / damping occurred
            const bool update_precond = (iter == 0 || damp < 1.0 || (iter % 3 == 0));
            assembleCondensedSystem(m_G, update_precond);

            // Eisenstat-Walker forcing
            const double rtol_gmres = std::clamp(0.1 * norm_res, 1e-11, 1e-2);
            const double neg_Rp[3] = { -Rp[0], -Rp[1], -Rp[2] };
            const int gmres_iters = solveBorderedGMRES(m_b_Y, neg_Rp, m_tangent, &m_tangent[m_total_end], sol_Y, sol_p, rtol_gmres, 200, false);

            // Recover full internal stage updates delta_stages in parallel
            const double dc = sol_p[0];
            const double dT = sol_p[1];
            const double dalpha = sol_p[2];
            const double sps = m_p[m_np - 1];

            TWIST_MARK_BEGIN("hopf_stage_recover");
            const double t_sr0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t n = 0; n < m_nt; n++) {
                const SpatialSlice *sl = &m_slices[n];
                const size_t ns = sl->n_space;
                const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
                const size_t prev_ns = m_slices[prev].n_space;
                const double *yn = &m_y_stages[m_stage_offsets[n]];
                const double *y_prev_end = &m_y_stages[m_stage_offsets[prev] + (m_st - 1) * prev_ns];
                const double *Gn = &m_G[m_stage_offsets[n]];
                const double dtau_T = m_dtau[n] * T;

                const size_t tid = (size_t)omp_get_thread_num();
                double *tbuf = m_stage_scratch + tid * m_stage_scratch_stride;
                double *rhs_tmp = tbuf;
                double *y_in = rhs_tmp + m_st * ns;
                double *dy_in = y_in + ns;
                double *dp_s = dy_in + ns;
                double *mass_tmp = dp_s + ns;
                double *mass_work = mass_tmp + ns;

                m_transfers[n].apply(y_prev_end, y_in);
                m_transfers[n].apply(&sol_Y[m_end_offsets[prev]], dy_in);

                sl->applyMass(dy_in, mass_tmp, 1.0, 0.0, mass_work);
                for (size_t j = 0; j < m_st; j++) {
                    double *rj = &rhs_tmp[j * ns];
                    const double *Gnj = &Gn[j * ns];
                    for (size_t i = 0; i < ns; i++) {
                        rj[i] = -Gnj[i] + mass_tmp[i];
                    }
                }

                // Subtract dG_n/dc * dc
                for (size_t l = 0; l < m_st; l++) {
                    for (size_t i = 0; i < ns; i++) {
                        mass_tmp[i] = yn[l * ns + i] - y_in[i];
                    }
                    sl->applyMassDerivC(mass_tmp, &rhs_tmp[l * ns], -dc, 1.0, mass_work);
                }
                for (size_t l = 0; l < m_st; l++) {
                    m_condensers[n].evalSpatialParamDeriv(l, &yn[l * ns], m_p, 0, dp_s);
                    for (size_t j = 0; j < m_st; j++) {
                        const double scale = +dtau_T * rm.A[j * m_st + l] * dc;
                        double *rj = &rhs_tmp[j * ns];
                        for (size_t i = 0; i < ns; i++) {
                            rj[i] += scale * dp_s[i];
                        }
                    }
                }

                // Subtract dG_n/dT * dT
                const double *rn_sp = &m_r_space[m_stage_offsets[n]];
                for (size_t j = 0; j < m_st; j++) {
                    double *rj = &rhs_tmp[j * ns];
                    for (size_t l = 0; l < m_st; l++) {
                        const double scale = +m_dtau[n] * rm.A[j * m_st + l] * dT;
                        const double *r_l = &rn_sp[l * ns];
                        for (size_t i = 0; i < ns; i++) {
                            rj[i] += scale * r_l[i];
                        }
                    }
                }

                // Subtract dG_n/dalpha * dalpha
                if (alpha_is_sps) {
                    for (size_t l = 0; l < m_st; l++) {
                        for (size_t i = 0; i < ns; i++) {
                            mass_tmp[i] = yn[l * ns + i] - y_in[i];
                        }
                        sl->applyMassDerivSps(sps, mass_tmp, &rhs_tmp[l * ns], -dalpha, 1.0, mass_work);
                    }
                }
                for (size_t l = 0; l < m_st; l++) {
                    m_condensers[n].evalSpatialParamDeriv(l, &yn[l * ns], m_p, alpha_idx, dp_s);
                    for (size_t j = 0; j < m_st; j++) {
                        const double scale = +dtau_T * rm.A[j * m_st + l] * dalpha;
                        double *rj = &rhs_tmp[j * ns];
                        for (size_t i = 0; i < ns; i++) {
                            rj[i] += scale * dp_s[i];
                        }
                    }
                }

                m_condensers[n].solveLocalStages(rhs_tmp, &m_delta_stages[m_stage_offsets[n]], true);
            }
            m_timing.t_stage_recover += omp_get_wtime() - t_sr0;
            TWIST_MARK_END("hopf_stage_recover");

            const double delta_norm = std::sqrt(endpointDot(sol_Y, sol_Y) + dc * dc + dT * dT + dalpha * dalpha);
            if (!std::isfinite(delta_norm)) {
                break;
            }

            // Damped backtracking
            damp = std::min(2.0 * damp, 1.0);
            bool step_accepted = false;
            double norm_trial = norm_res;
            while (damp >= min_damp) {
                const double c_tr = c + damp * dc;
                const double T_tr = T + damp * dT;
                const double alpha_tr = alpha + damp * dalpha;
                if (T_tr <= 1e-10 || std::abs(c_tr) < 1e-12 || (alpha_is_sps && alpha_tr <= 1e-12)) {
                    damp *= 0.5;
                    continue;
                }
                for (size_t i = 0; i < m_total_stages; i++) {
                    m_y_trial[i] = m_y_stages[i] + damp * m_delta_stages[i];
                }
                evaluateFullResidual(m_y_trial, c_tr, T_tr, alpha_tr, ds, m_G_trial, Rp_trial, true);
                norm_trial = computeResidualNorm(m_G_trial, Rp_trial);

                if (std::isfinite(norm_trial) && (norm_trial < norm_res || norm_trial < tol)) {
                    step_accepted = true;
                    break;
                }
                damp *= 0.5;
            }

            if (!step_accepted) {
                // Restore m_p if evaluateFullResidual modified it on rejected trial
                m_p[0] = c;
                m_p[alpha_idx] = alpha;
                updateMassMatricesAll();
                break;
            }

            memcpy(m_y_stages, m_y_trial, m_total_stages * sizeof(double));
            memcpy(m_G, m_G_trial, m_total_stages * sizeof(double));
            memcpy(Rp, Rp_trial, sizeof(Rp));
            c += damp * dc;
            T += damp * dT;
            alpha += damp * dalpha;
            m_p[0] = c;
            m_p[alpha_idx] = alpha;
            m_T = T;
            norm_res = norm_trial;

            if (verbose) {
                fmt::println("  [HopfNewton] iter {:2d} | |R|={:.3e} | |dz|={:.3e} | damp={:.3f} | gmres={:3d} | c={:.6f} T={:.6f} alpha={:.6f}",
                    iter + 1, norm_res, damp * delta_norm, damp, gmres_iters, c, T, alpha);
            }

            if (norm_res < tol || damp * delta_norm < tol) {
                converged = true;
                break;
            }
        }

        free(sol_Y);
        if (converged) {
            assembleCondensedSystem(nullptr, true);
        }
        return converged;
    }

    void HopfBranchSolver::computeTangent()
    {
        TWIST_MARK_BEGIN("hopf_tangent");
        const double t0 = omp_get_wtime();
        const double gmres_before = m_timing.t_gmres_total;
        const double pu_before = m_timing.t_precond_update;
        double *sol_Y = (double *)malloc(m_total_end * sizeof(double));
        double *zero_Y = (double *)calloc(m_total_end, sizeof(double));
        double rhs_p[3] = { 0.0, 0.0, 1.0 };
        double sol_p[3] = { 0.0, 0.0, 0.0 };

        solveBorderedGMRES(zero_Y, rhs_p, m_tangent, &m_tangent[m_total_end], sol_Y, sol_p, 1e-11, 160, false);

        // Enforce orthogonality to g_space and g_time (should hold to ~1e-11 from solve)
        const double ps = endpointDot(m_g_space, sol_Y);
        const double pt = endpointDot(m_g_time, sol_Y);
        for (size_t i = 0; i < m_total_end; i++) {
            sol_Y[i] -= ps * m_g_space[i] + pt * m_g_time[i];
        }

        // Preserve branch orientation <z_new, z_prev> > 0
        double dot_prev = endpointDot(m_tangent, sol_Y)
                        + m_weights_bord[m_total_end + 0] * m_tangent[m_total_end + 0] * sol_p[0]
                        + m_weights_bord[m_total_end + 1] * m_tangent[m_total_end + 1] * sol_p[1]
                        + m_weights_bord[m_total_end + 2] * m_tangent[m_total_end + 2] * sol_p[2];
        const double sgn = (dot_prev >= 0.0) ? 1.0 : -1.0;

        memcpy(m_tangent, sol_Y, m_total_end * sizeof(double));
        m_tangent[m_total_end + 0] = sol_p[0];
        m_tangent[m_total_end + 1] = sol_p[1];
        m_tangent[m_total_end + 2] = sol_p[2];

        const double nrm = borderedNorm(m_tangent);
        if (nrm > 0.0) {
            const double inv = sgn / nrm;
            for (size_t i = 0; i < m_total_end + 3; i++) {
                m_tangent[i] *= inv;
            }
        }

        free(zero_Y);
        free(sol_Y);
        const double inner_gmres = (m_timing.t_gmres_total - gmres_before) + (m_timing.t_precond_update - pu_before);
        const double dt_tang = (omp_get_wtime() - t0) - inner_gmres;
        if (dt_tang > 0.0) m_timing.t_tangent += dt_tang;
        TWIST_MARK_END("hopf_tangent");
    }

    void HopfBranchSolver::updatePhaseConditions()
    {
        TWIST_MARK_BEGIN("hopf_phase");
        const double t0 = omp_get_wtime();
        syncEndpointsFromStages(m_y_ref_end);

        // g_space = y_xi / ||y_xi||
        for (size_t n = 0; n < m_nt; n++) {
            eval_slice_spatial_deriv(m_slices[n], &m_y_ref_end[m_end_offsets[n]], &m_g_space[m_end_offsets[n]]);
        }
        const double gs_norm = endpointNorm(m_g_space);
        if (gs_norm > 0.0) {
            for (size_t i = 0; i < m_total_end; i++) {
                m_g_space[i] /= gs_norm;
            }
        }

        // g_time = y_tau at tau_end, orthogonalized against g_space (if amplitude is non-negligible)
        const RadauDGMethod &rm = getRadauDG(m_st);
        const double *d_row = &rm.D_tau[(m_st - 1) * (m_st + 1)];
        double *y_in_tmp = m_stage_scratch;
        double *gt_cand = (double *)malloc(m_total_end * sizeof(double));

        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            m_transfers[n].apply(&m_y_ref_end[m_end_offsets[prev]], y_in_tmp);
            const double *yn = &m_y_stages[m_stage_offsets[n]];
            const double inv_dt = 1.0 / m_dtau[n];
            double *gt_n = &gt_cand[m_end_offsets[n]];
            for (size_t i = 0; i < ns; i++) {
                double val = d_row[0] * y_in_tmp[i];
                for (size_t j = 0; j < m_st; j++) {
                    val += d_row[j + 1] * yn[j * ns + i];
                }
                gt_n[i] = val * inv_dt;
            }
        }

        const double gt_raw_norm = endpointNorm(gt_cand);
        if (gt_raw_norm > 1e-7) {
            const double proj = endpointDot(m_g_space, gt_cand);
            for (size_t i = 0; i < m_total_end; i++) {
                gt_cand[i] -= proj * m_g_space[i];
            }
            const double gt_norm = endpointNorm(gt_cand);
            if (gt_norm > 1e-10) {
                for (size_t i = 0; i < m_total_end; i++) {
                    m_g_time[i] = gt_cand[i] / gt_norm;
                }
            }
        }
        free(gt_cand);
        m_timing.t_phase += omp_get_wtime() - t0;
        TWIST_MARK_END("hopf_phase");
    }

    void HopfBranchSolver::acceptStep()
    {
        memcpy(m_y_prev_stages, m_y_stages, m_total_stages * sizeof(double));
        syncEndpointsFromStages(m_y_prev_end);
        m_c_prev = m_p[0];
        m_T_prev = m_T;
        m_alpha_prev = m_p[m_pmask[1]];
    }

    void HopfBranchSolver::restoreAcceptedStep()
    {
        memcpy(m_y_stages, m_y_prev_stages, m_total_stages * sizeof(double));
        m_p[0] = m_c_prev;
        m_T = m_T_prev;
        m_p[m_pmask[1]] = m_alpha_prev;
        updateMassMatricesAll();
    }

    void HopfBranchSolver::adaptSpaceTimeMesh(const HopfContinuationBounds &bounds, double ds, bool small_change, bool verbose)
    {
        if (!bounds.allow_mesh_adaptation) {
            return;
        }
        TWIST_MARK_BEGIN("hopf_adapt");
        const double t0 = omp_get_wtime();
        const RadauDGMethod &rm = getRadauDG(m_st);

        // 1. Temporal mesh error estimation & adaptation
        std::vector<double> v_tau(m_nt, 0.0), F_tau(m_nt + 1, 0.0);
        const double t_fac = std::tgamma((double)(m_st + 1));
        const double t_exp = 1.0 / (double)m_st;
        double *y_in_tmp = m_stage_scratch;
        double *err_tmp = y_in_tmp + m_slices[0].n_space;

        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const size_t prev = (n == 0) ? (m_nt - 1) : (n - 1);
            const size_t prev_ns = m_slices[prev].n_space;
            const double *y_prev_end = &m_y_stages[m_stage_offsets[prev] + (m_st - 1) * prev_ns];
            const double *yn = &m_y_stages[m_stage_offsets[n]];

            double *e_buf = (ns <= m_slices[0].n_space) ? err_tmp : (double *)malloc(ns * sizeof(double));
            double *yin = (ns <= m_slices[0].n_space) ? y_in_tmp : (double *)malloc(ns * sizeof(double));

            m_transfers[n].apply(y_prev_end, yin);
            for (size_t i = 0; i < ns; i++) {
                double val = rm.err_weights[0] * yin[i];
                for (size_t l = 0; l < m_st; l++) {
                    val += rm.err_weights[l + 1] * yn[l * ns + i];
                }
                e_buf[i] = val;
            }
            const double inv_dt = 1.0 / m_dtau[n];
            const double *wn = &m_weights_end[m_end_offsets[n]];
            double l2_sq = 0.0;
            for (size_t i = 0; i < ns; i++) {
                l2_sq += (wn[i] * inv_dt) * e_buf[i] * e_buf[i];
            }
            const double vk = std::pow(t_fac * std::sqrt(std::max(0.0, l2_sq)), t_exp);
            v_tau[n] = vk;
            F_tau[n + 1] = F_tau[n] + vk;

            if (ns > m_slices[0].n_space) {
                free(e_buf);
                free(yin);
            }
        }

        const double theta_tau = F_tau[m_nt];
        size_t nt_new = m_nt;
        std::vector<double> tau_new(m_tau, m_tau + m_nt + 1);
        bool adapt_time = false;

        if (theta_tau > 1e-12) {
            for (size_t n = 0; n <= m_nt; n++) {
                F_tau[n] /= theta_tau;
            }
            nt_new = (size_t)(theta_tau * std::pow(rm.leading_err_scale / bounds.geps, t_exp) + 1.0);
            nt_new = (size_t)std::ceil((double)nt_new * 1.1);
            if (small_change) {
                nt_new = std::min(nt_new, (size_t)std::ceil(1.25 * (double)m_nt));
                nt_new = std::max(nt_new, (size_t)std::floor(0.8 * (double)m_nt));
            }
            nt_new = std::clamp(nt_new, bounds.min_nt_adapt, bounds.max_nt_adapt);

            double vmin = v_tau[0], vmax = v_tau[0];
            for (size_t n = 1; n < m_nt; n++) {
                vmin = std::min(vmin, v_tau[n]);
                vmax = std::max(vmax, v_tau[n]);
            }
            if (nt_new != m_nt || (vmin > 0.0 && vmax / vmin > 1.6)) {
                adapt_time = true;
                tau_new.resize(nt_new + 1);
                for (size_t n = 0; n <= nt_new; n++) {
                    tau_new[n] = (double)n / (double)nt_new;
                }
                interp(tau_new.data(), nt_new + 1, tau_new.data(), F_tau.data(), m_tau, m_nt + 1, 1);
                tau_new[0] = 0.0;
                tau_new[nt_new] = 1.0;
            }
            else {
                nt_new = m_nt;
            }
        }

        // 2. Per-element spatial mesh adaptation
        std::vector<size_t> nnodes_new(nt_new);
        std::vector<std::vector<double>> t_space_new(nt_new);
        bool adapt_space = false;

        // First map temporal stages if temporal grid changed; otherwise adapt slices in-place
        std::vector<SpatialSlice> old_slices(m_nt);
        std::vector<double> old_tau(m_tau, m_tau + m_nt + 1);
        std::vector<size_t> old_stage_off(m_stage_offsets, m_stage_offsets + m_nt + 1);
        std::vector<size_t> old_end_off(m_end_offsets, m_end_offsets + m_nt + 1);
        std::vector<double> old_y_stages(m_y_stages, m_y_stages + m_total_stages);
        std::vector<double> old_y_ref(m_y_ref_end, m_y_ref_end + m_total_end);
        std::vector<double> old_y_prev(m_y_prev_end, m_y_prev_end + m_total_end);
        std::vector<double> old_g_sp(m_g_space, m_g_space + m_total_end);
        std::vector<double> old_g_tm(m_g_time, m_g_time + m_total_end);
        std::vector<double> old_tang(m_tangent, m_tangent + m_total_end + 3);
        for (size_t n = 0; n < m_nt; n++) {
            old_slices[n] = m_slices[n];
        }

        // Build intermediate stages on new temporal grid (keeping slice of containing old element)
        std::vector<SpatialSlice> inter_slices(nt_new);
        std::vector<double> inter_y_stages;
        std::vector<double> inter_y_ref;
        std::vector<double> inter_y_prev;
        std::vector<double> inter_g_sp;
        std::vector<double> inter_g_tm;
        std::vector<double> inter_tang;

        if (!adapt_time) {
            inter_slices = old_slices;
            inter_y_stages = old_y_stages;
            inter_y_ref = old_y_ref;
            inter_y_prev = old_y_prev;
            inter_g_sp = old_g_sp;
            inter_g_tm = old_g_tm;
            inter_tang = old_tang;
        }
        else {
            size_t tot_end = 0;
            std::vector<size_t> inter_end_off(nt_new + 1, 0), inter_st_off(nt_new + 1, 0);
            for (size_t n = 0; n < nt_new; n++) {
                const double tau_mid = 0.5 * (tau_new[n] + tau_new[n + 1]);
                const size_t src_el = find_interval(old_tau.data(), m_nt, tau_mid);
                inter_slices[n] = old_slices[src_el];
                const size_t ns = inter_slices[n].n_space;
                tot_end += ns;
                inter_end_off[n + 1] = tot_end;
                inter_st_off[n + 1] = inter_st_off[n] + m_st * ns;
            }
            inter_y_stages.resize(inter_st_off[nt_new]);
            inter_y_ref.resize(tot_end);
            inter_y_prev.resize(tot_end);
            inter_g_sp.resize(tot_end);
            inter_g_tm.resize(tot_end);
            inter_tang.resize(tot_end + 3);

            double w_rad[5];
            for (size_t n = 0; n < nt_new; n++) {
                const size_t ns = inter_slices[n].n_space;
                const double dt_new = tau_new[n + 1] - tau_new[n];

                // Interpolate each Radau stage l = 0..m_st-1 from containing old element
                for (size_t l = 0; l < m_st; l++) {
                    const double t_q = tau_new[n] + rm.c[l] * dt_new;
                    const size_t src_el = find_interval(old_tau.data(), m_nt, t_q);
                    const double dt_old = old_tau[src_el + 1] - old_tau[src_el];
                    const double theta = std::clamp((t_q - old_tau[src_el]) / dt_old, 0.0, 1.0);
                    radauDenseWeights(m_st, theta, w_rad);

                    const size_t src_ns = old_slices[src_el].n_space;
                    const size_t src_prev = (src_el == 0) ? (m_nt - 1) : (src_el - 1);
                    const size_t src_prev_ns = old_slices[src_prev].n_space;

                    std::vector<double> y0_src(src_ns);
                    SparseDenseOutputOp op_prev;
                    op_prev.build(old_slices[src_prev], old_slices[src_el]);
                    op_prev.apply(&old_y_stages[old_stage_off[src_prev] + (m_st - 1) * src_prev_ns], y0_src.data());

                    std::vector<double> y_eval_src(src_ns);
                    const double *yn_old = &old_y_stages[old_stage_off[src_el]];
                    for (size_t i = 0; i < src_ns; i++) {
                        double val = w_rad[0] * y0_src[i];
                        for (size_t j = 0; j < m_st; j++) {
                            val += w_rad[j + 1] * yn_old[j * src_ns + i];
                        }
                        y_eval_src[i] = val;
                    }

                    SparseDenseOutputOp op_sp;
                    op_sp.build(old_slices[src_el], inter_slices[n]);
                    op_sp.apply(y_eval_src.data(), &inter_y_stages[inter_st_off[n] + l * ns]);
                }

                // Linearly interpolate endpoint auxiliary vectors (g_space, g_time, tangent, y_ref, y_prev)
                const double t_end_q = tau_new[n + 1];
                const size_t el_r = find_interval(old_tau.data(), m_nt, t_end_q);
                const size_t el_l = (el_r == 0) ? (m_nt - 1) : (el_r - 1);
                const double dt_old = old_tau[el_r + 1] - old_tau[el_r];
                const double th_e = std::clamp((t_end_q - old_tau[el_r]) / dt_old, 0.0, 1.0);

                SparseDenseOutputOp op_l, op_r;
                op_l.build(old_slices[el_l], inter_slices[n]);
                op_r.build(old_slices[el_r], inter_slices[n]);
                std::vector<double> tmp_l(ns), tmp_r(ns);
                auto interp_end_vec = [&](const std::vector<double> &src, std::vector<double> &dst) {
                    op_l.apply(&src[old_end_off[el_l]], tmp_l.data());
                    op_r.apply(&src[old_end_off[el_r]], tmp_r.data());
                    for (size_t i = 0; i < ns; i++) {
                        dst[inter_end_off[n] + i] = (1.0 - th_e) * tmp_l[i] + th_e * tmp_r[i];
                    }
                };
                interp_end_vec(old_y_ref, inter_y_ref);
                interp_end_vec(old_y_prev, inter_y_prev);
                interp_end_vec(old_g_sp, inter_g_sp);
                interp_end_vec(old_g_tm, inter_g_tm);
                interp_end_vec(old_tang, inter_tang);
            }
            inter_tang[tot_end + 0] = old_tang[m_total_end + 0];
            inter_tang[tot_end + 1] = old_tang[m_total_end + 1];
            inter_tang[tot_end + 2] = old_tang[m_total_end + 2];
        }

        // Compute spatial monitor per element of inter_slices
        size_t inter_st_off_cur = 0;
        for (size_t n = 0; n < nt_new; n++) {
            const SpatialSlice &sl = inter_slices[n];
            const size_t ns = sl.n_space;
            const size_t nsub = sl.nSub();
            std::vector<double> F_sp(sl.nnodes, 0.0), v_sp(nsub, 0.0), v_max(nsub, 0.0), F_cum(sl.nnodes, 0.0);
            std::vector<double> mon_work(2 * sl.node * sl.nstages + sl.node + 16);

            double err_exp = 1.0 / (double)(sl.nstages + 1);
            for (size_t l = 0; l < m_st; l++) {
                double th_l = 0.0;
                sl.computeMonitorFromState(&inter_y_stages[inter_st_off_cur + l * ns], F_sp.data(), v_sp.data(), th_l, err_exp, mon_work.data());
                for (size_t k = 0; k < nsub; k++) {
                    v_max[k] = std::max(v_max[k], v_sp[k]);
                }
            }
            inter_st_off_cur += m_st * ns;

            F_cum[0] = 0.0;
            for (size_t k = 0; k < nsub; k++) {
                F_cum[k + 1] = F_cum[k] + v_max[k];
            }
            const double theta = F_cum[nsub];
            if (theta > 0.0) {
                for (size_t k = 0; k < sl.nnodes; k++) {
                    F_cum[k] /= theta;
                }
            }
            size_t n_opt = sl.nnodes;
            double *t_opt = sl.equidistributeMesh(F_cum.data(), v_max.data(), theta, err_exp, bounds.geps, bounds.min_nodes_adapt, bounds.max_nodes_adapt, n_opt, small_change);

            // Check if spatial mesh actually needs updating
            bool need_sp = (n_opt != sl.nnodes);
            if (!need_sp) {
                double max_dt = 0.0;
                for (size_t k = 0; k < sl.nnodes; k++) {
                    max_dt = std::max(max_dt, std::abs(t_opt[k] - sl.t[k]));
                }
                if (max_dt > 0.02 / (double)sl.nnodes) {
                    need_sp = true;
                }
            }

            if (need_sp) {
                adapt_space = true;
                nnodes_new[n] = n_opt;
                t_space_new[n].assign(t_opt, t_opt + n_opt);
            }
            else {
                nnodes_new[n] = sl.nnodes;
                t_space_new[n].assign(sl.t, sl.t + sl.nnodes);
            }
            free(t_opt);
        }

        if (!adapt_time && !adapt_space) {
            m_timing.t_adapt += omp_get_wtime() - t0;
            TWIST_MARK_END("hopf_adapt");
            return;
        }

        if (verbose) {
            size_t mx_n = 0;
            for (size_t n = 0; n < nt_new; n++) mx_n = std::max(mx_n, nnodes_new[n]);
            fmt::println("  [Adapt] Nt: {} -> {}, max(Nx): {} -> {}", m_nt, nt_new, maxSpatialNodes(), mx_n);
        }

        // Reallocate solver buffers to new (nt_new, nnodes_new)
        allocateForGrid(nt_new, m_st, nnodes_new.data());
        memcpy(m_tau, tau_new.data(), (nt_new + 1) * sizeof(double));
        for (size_t n = 0; n < nt_new; n++) {
            m_slices[n] = inter_slices[n];
            m_slices[n].updateGrid(nnodes_new[n], t_space_new[n].data());
        }
        rebuildOffsetsAndWeights();
        rebuildTransfers();

        // Spatially transfer inter_* vectors onto new m_slices[n]
        size_t inter_e_off = 0, inter_s_off = 0;
        for (size_t n = 0; n < nt_new; n++) {
            const size_t ns_old = inter_slices[n].n_space;
            const size_t ns_new = m_slices[n].n_space;
            SparseDenseOutputOp op;
            op.build(inter_slices[n], m_slices[n]);

            for (size_t l = 0; l < m_st; l++) {
                op.apply(&inter_y_stages[inter_s_off + l * ns_old], &m_y_stages[m_stage_offsets[n] + l * ns_new]);
            }
            op.apply(&inter_y_ref[inter_e_off], &m_y_ref_end[m_end_offsets[n]]);
            op.apply(&inter_y_prev[inter_e_off], &m_y_prev_end[m_end_offsets[n]]);
            op.apply(&inter_g_sp[inter_e_off], &m_g_space[m_end_offsets[n]]);
            op.apply(&inter_g_tm[inter_e_off], &m_g_time[m_end_offsets[n]]);
            op.apply(&inter_tang[inter_e_off], &m_tangent[m_end_offsets[n]]);

            inter_e_off += ns_old;
            inter_s_off += m_st * ns_old;
        }
        m_tangent[m_total_end + 0] = inter_tang[inter_e_off + 0];
        m_tangent[m_total_end + 1] = inter_tang[inter_e_off + 1];
        m_tangent[m_total_end + 2] = inter_tang[inter_e_off + 2];

        // Re-orthonormalize g_space, g_time, tangent on adapted grid
        const double gs_n = endpointNorm(m_g_space);
        if (gs_n > 0.0) {
            for (size_t i = 0; i < m_total_end; i++) m_g_space[i] /= gs_n;
        }
        const double proj_ts = endpointDot(m_g_space, m_g_time);
        for (size_t i = 0; i < m_total_end; i++) m_g_time[i] -= proj_ts * m_g_space[i];
        const double gt_n = endpointNorm(m_g_time);
        if (gt_n > 0.0) {
            for (size_t i = 0; i < m_total_end; i++) m_g_time[i] /= gt_n;
        }
        const double tn_n = borderedNorm(m_tangent);
        if (tn_n > 0.0) {
            for (size_t i = 0; i < m_total_end + 3; i++) m_tangent[i] /= tn_n;
        }

        // Polish on adapted mesh with 1-2 Newton steps (fix phase / arclength at current ds)
        m_timing.t_adapt += omp_get_wtime() - t0;
        TWIST_MARK_END("hopf_adapt");
        solveNewton(ds, 1e-10, 1e-4, 6, verbose);
    }

    double HopfBranchSolver::oscillationAmplitude() const
    {
        if (m_nt == 0 || m_node == 0) {
            return 0.0;
        }
        TWIST_MARK_BEGIN("hopf_amp");
        const double t0 = omp_get_wtime();
        // Compute L2 deviation from time average on reference mesh slices[0]
        const SpatialSlice &ref = m_slices[0];
        const size_t ns = ref.n_space;
        std::vector<double> y_mean(ns, 0.0), y_tmp(ns);
        std::vector<double> w0(ns);
        ref.fillQuadratureWeights(w0.data(), false);

        for (size_t n = 0; n < m_nt; n++) {
            SparseDenseOutputOp op;
            op.build(m_slices[n], ref);
            op.apply(&m_y_stages[m_stage_offsets[n] + (m_st - 1) * m_slices[n].n_space], y_tmp.data());
            for (size_t i = 0; i < ns; i++) {
                y_mean[i] += m_dtau[n] * y_tmp[i];
            }
        }

        double amp2 = 0.0;
        for (size_t n = 0; n < m_nt; n++) {
            SparseDenseOutputOp op;
            op.build(m_slices[n], ref);
            op.apply(&m_y_stages[m_stage_offsets[n] + (m_st - 1) * m_slices[n].n_space], y_tmp.data());
            double slice_l2 = 0.0;
            for (size_t i = 0; i < ns; i++) {
                const double diff = y_tmp[i] - y_mean[i];
                slice_l2 += w0[i] * diff * diff;
            }
            amp2 += m_dtau[n] * slice_l2;
        }
        m_timing.t_amp += omp_get_wtime() - t0;
        TWIST_MARK_END("hopf_amp");
        return std::sqrt(std::max(0.0, amp2));
    }

    double HopfBranchSolver::solutionNormL2() const
    {
        double nrm2 = 0.0;
        for (size_t n = 0; n < m_nt; n++) {
            const size_t ns = m_slices[n].n_space;
            const double *yn_end = &m_y_stages[m_stage_offsets[n] + (m_st - 1) * ns];
            const double *wn = &m_weights_end[m_end_offsets[n]];
            for (size_t i = 0; i < ns; i++) {
                nrm2 += wn[i] * yn_end[i] * yn_end[i];
            }
        }
        return std::sqrt(std::max(0.0, nrm2));
    }

    FloquetSpectrum HopfBranchSolver::computeFloquet(size_t neig, size_t nsub, double tol, bool verbose)
    {
        assembleCondensedSystem(nullptr, false);
        FloquetAnalyzer fa;
        return fa.compute(m_nt, m_st, m_slices, m_transfers, m_condensers, m_y_stages, m_dtau, m_T, m_p, neig, nsub, tol, 1e-4, verbose);
    }

    void HopfBranchSolver::setSerializationFileAndIndex(H5::H5File &file, int index)
    {
        setSerializationTarget(file, "", index);
    }

    void HopfBranchSolver::setSerializationTarget(H5::H5File &file, const std::string &branch_group_path, int index)
    {
        m_serialization_file = file;
        m_serialization_branch = branch_group_path;
        m_serialization_root = branch_group_path.empty() ? file.openGroup("/") : file.openGroup(branch_group_path);
        m_serialization_index = index;
    }

    void HopfBranchSolver::serialize(const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, SolutionTypes stype, const std::string &target_h5_path, const std::string &branch_name, int source_index)
    {
        TWIST_MARK_BEGIN("hopf_serialize");
        const double t0 = omp_get_wtime();
        const double amp_before = m_timing.t_amp;
        if (m_serialization_index == -1) {
            m_serialization_index = 0;
            m_model_name = name;
            std::vector<int> diffusion_indices;
            std::vector<double> diffusion_coeffs;
            const int ndiff = (int)m_diffusion.size();
            diffusion_indices.reserve(ndiff);
            diffusion_coeffs.reserve(ndiff);
            for (const auto &[idx, coeff] : m_diffusion) {
                diffusion_indices.emplace_back(idx);
                diffusion_coeffs.emplace_back(coeff);
            }

            if (!target_h5_path.empty()) {
                m_serialization_file = H5::H5File(target_h5_path.c_str(), H5F_ACC_RDWR);
                if (!m_serialization_file.nameExists("hopf_branches")) {
                    m_serialization_file.createGroup("hopf_branches");
                }
                H5::Group hb = m_serialization_file.openGroup("hopf_branches");
                const std::string use_branch = branch_name.empty() ? tag : branch_name;
                if (hb.nameExists(use_branch)) {
                    hb.unlink(use_branch);
                }
                m_serialization_root = hb.createGroup(use_branch);
                m_serialization_branch = fmt::format("hopf_branches/{}", use_branch);
            }
            else {
                if (!prefix.empty()) {
                    std::filesystem::create_directories(prefix);
                }
                std::filesystem::path p_prefix(prefix);
                std::filesystem::path p_name(fmt::format("{}-{}-{}-{}-hopf.h5", name, parameter_set, parameter, tag));
                std::filesystem::path p_fpath = p_prefix / p_name;
                m_serialization_file = H5::H5File(p_fpath.c_str(), H5F_ACC_TRUNC);
                m_serialization_root = m_serialization_file.openGroup("/");
                m_serialization_branch.clear();
            }

            {
                H5::Group meta(m_serialization_root.createGroup("meta_data"));
                const std::string program_argv{ get_program_argv() };
                const std::string cwd = std::filesystem::current_path();
                ::serialize<int>(meta, "file_type", (int)FileType::hopf_continuation);
                ::serialize<char, 1>(meta, "directory", cwd.c_str(), { cwd.size() });
                ::serialize<char, 1>(meta, "cmdline", program_argv.c_str(), { program_argv.size() });
                if (source_index >= 0) {
                    ::serialize<int>(meta, "source_index", source_index);
                }
            }
            {
                H5::Group basic(m_serialization_root.createGroup("basic_info"));
                ::serialize<size_t>(basic, "node", m_node);
                ::serialize<size_t>(basic, "np", m_np);
                ::serialize<size_t>(basic, "nparam", 3);
                ::serialize<size_t>(basic, "nstages", m_spatial_nstages);
                ::serialize<size_t>(basic, "ncol_time", m_st);
                ::serialize<double>(basic, "spatial_period", 0.5 * m_unscaled_spatial_period);
                ::serialize<int, 1>(basic, "pmask", m_pmask, { (hsize_t)2 });
                ::serialize<char, 1>(basic, "name", name.c_str(), { name.size() });
                ::serialize<int, 1>(basic, "diffusion_indices", diffusion_indices.data(), { (hsize_t)ndiff });
                ::serialize<double, 1>(basic, "diffusion_coeffs", diffusion_coeffs.data(), { (hsize_t)ndiff });
                ::serialize<int>(basic, "ndiff", ndiff);
                ::serialize<double, 2>(basic, "P", m_slices[0].c_P, { (hsize_t)m_spatial_nstages, (hsize_t)m_spatial_nstages });
            }
        }

        H5::Group group = m_serialization_root.createGroup(fmt::format("{}", m_serialization_index));
        m_serialization_index++;

        const double amp = oscillationAmplitude();
        const double l2 = solutionNormL2();

        ::serialize<double, 1>(group, "p", m_p, { (hsize_t)m_np });
        ::serialize<double>(group, "c", m_p[0]);
        ::serialize<double>(group, "T", m_T);
        ::serialize<double>(group, "amplitude", amp);
        ::serialize<double>(group, "norm_l2", l2);
        ::serialize<size_t>(group, "nt", m_nt);
        ::serialize<double, 1>(group, "tau", m_tau, { (hsize_t)(m_nt + 1) });
        ::serialize<uint64_t>(group, "stype", (uint64_t)stype);

        H5::Group elems = group.createGroup("elements");
        for (size_t n = 0; n < m_nt; n++) {
            H5::Group el = elems.createGroup(fmt::format("{}", n));
            const size_t ns = m_slices[n].n_space;
            ::serialize<size_t>(el, "nnodes", m_slices[n].nnodes);
            ::serialize<double, 1>(el, "t", m_slices[n].t, { (hsize_t)m_slices[n].nnodes });
            ::serialize<double, 1>(el, "h", m_slices[n].h, { (hsize_t)(m_slices[n].nnodes - 1) });
            ::serialize<double, 1>(el, "y_stages", &m_y_stages[m_stage_offsets[n]], { (hsize_t)(m_st * ns) });
        }

        // Build lightweight resampled kymograph (nt + 1) x nx_kymo x node on uniform spatial grid
        constexpr size_t nx_kymo = 96;
        std::vector<double> t_unif(nx_kymo);
        for (size_t j = 0; j < nx_kymo; j++) {
            t_unif[j] = (double)j / (double)(nx_kymo - 1);
        }
        SpatialSlice kymo_slice;
        kymo_slice.init(m_spatial_nstages, m_node, nx_kymo, t_unif.data(), m_func, m_fjac, m_pjac, m_diffusion, m_unscaled_spatial_period);
        //Wait: kymo_slice has (nx_kymo - 1)*(s+1)+1 points; we just want values at t_unif (the nx_kymo grid nodes)!
        std::vector<double> kymo((m_nt + 1) * nx_kymo * m_node);
        std::vector<double> kymo_full(kymo_slice.n_space);
        const size_t n_basis = m_spatial_nstages + 1;
        for (size_t n = 0; n <= m_nt; n++) {
            const size_t el_idx = (n == 0) ? (m_nt - 1) : (n - 1);
            const size_t ns_el = m_slices[el_idx].n_space;
            SparseDenseOutputOp op;
            op.build(m_slices[el_idx], kymo_slice);
            op.apply(&m_y_stages[m_stage_offsets[el_idx] + (m_st - 1) * ns_el], kymo_full.data());
            for (size_t j = 0; j < nx_kymo; j++) {
                const size_t src_idx = (j == nx_kymo - 1) ? ((nx_kymo - 1) * n_basis) : (j * n_basis);
                memcpy(&kymo[(n * nx_kymo + j) * m_node], &kymo_full[src_idx * m_node], m_node * sizeof(double));
            }
        }
        ::serialize<double, 3>(group, "kymograph", kymo.data(), { (hsize_t)(m_nt + 1), (hsize_t)nx_kymo, (hsize_t)m_node });
        m_serialization_file.flush(H5F_SCOPE_GLOBAL);
        const double dt_ser = (omp_get_wtime() - t0) - (m_timing.t_amp - amp_before);
        if (dt_ser > 0.0) m_timing.t_serialize += dt_ser;
        TWIST_MARK_END("hopf_serialize");
    }

    void HopfBranchSolver::loadFromH5(const std::string &h5_path, int solution_index)
    {
        H5::H5File file(h5_path, H5F_ACC_RDONLY);
        loadFromH5(file, solution_index);
    }

    void HopfBranchSolver::loadFromH5(const H5::H5File &file, int solution_index)
    {
        H5::Group root = file.openGroup("/");
        loadFromH5(root, solution_index);
    }

    void HopfBranchSolver::loadFromH5(const H5::Group &root, int solution_index)
    {
        freeBuffers();

        char *model_name = nullptr;
        int *diffusion_indices = nullptr;
        double *diffusion_coeffs = nullptr;
        int ndiff = 0;
        double half_L = 0.0;
        size_t ncol_time = 3;

        H5::Group basic = root.openGroup("basic_info");
        ::deserialize(basic, "node", m_node);
        ::deserialize(basic, "np", m_np);
        ::deserialize(basic, "nstages", m_spatial_nstages);
        ::deserialize(basic, "ncol_time", ncol_time);
        ::deserialize(basic, "spatial_period", half_L);
        ::deserialize(basic, "pmask", m_pmask);
        ::deserialize(basic, "name", model_name);
        ::deserialize(basic, "diffusion_indices", diffusion_indices);
        ::deserialize(basic, "diffusion_coeffs", diffusion_coeffs);
        ::deserialize(basic, "ndiff", ndiff);

        m_unscaled_spatial_period = 2.0 * half_L;
        m_model_name = std::string(model_name);

        std::string lib_path = fmt::format(".cache/models/lib/{}{}", model_name, TWIST_LIB_EXT);
        if (!std::filesystem::exists(lib_path)) {
            throw std::runtime_error(fmt::format("{} model does not have a compiled library at {}", model_name, lib_path));
        }
        m_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
        m_owns_handle = (m_handle != nullptr);
        m_func = (func_t)dlsym(m_handle, "func");
        m_fjac = (func_t)dlsym(m_handle, "fjac");
        m_pjac = (pjac_t)dlsym(m_handle, "pjac");
        m_diffusion.clear();
        for (int i = 0; i < ndiff; i++) {
            m_diffusion.emplace_back(diffusion_indices[i], diffusion_coeffs[i]);
        }
        free(model_name);
        free(diffusion_indices);
        free(diffusion_coeffs);

        H5::Group group = root.openGroup(fmt::format("{}", solution_index));
        size_t nt_read = 0;
        ::deserialize(group, "nt", nt_read);
        ::deserialize(group, "p", m_p);
        ::deserialize(group, "T", m_T);
        m_T_scale = std::max(1.0, m_T);

        double *tau_read = nullptr;
        ::deserialize(group, "tau", tau_read);

        H5::Group elems = group.openGroup("elements");
        std::vector<size_t> nn(nt_read);
        std::vector<double *> t_vecs(nt_read, nullptr);
        std::vector<double *> y_vecs(nt_read, nullptr);

        for (size_t n = 0; n < nt_read; n++) {
            H5::Group el = elems.openGroup(fmt::format("{}", n));
            ::deserialize(el, "nnodes", nn[n]);
            ::deserialize(el, "t", t_vecs[n]);
            ::deserialize(el, "y_stages", y_vecs[n]);
        }

        allocateForGrid(nt_read, ncol_time, nn.data());
        memcpy(m_tau, tau_read, (nt_read + 1) * sizeof(double));
        free(tau_read);

        for (size_t n = 0; n < nt_read; n++) {
            m_slices[n].init(m_spatial_nstages, m_node, nn[n], t_vecs[n], m_func, m_fjac, m_pjac, m_diffusion, m_unscaled_spatial_period);
            free(t_vecs[n]);
        }
        rebuildOffsetsAndWeights();
        rebuildTransfers();
        updateMassMatricesAll();

        for (size_t n = 0; n < nt_read; n++) {
            const size_t ns = m_slices[n].n_space;
            memcpy(&m_y_stages[m_stage_offsets[n]], y_vecs[n], m_st * ns * sizeof(double));
            free(y_vecs[n]);
        }

        m_c_prev = m_p[0];
        m_T_prev = m_T;
        m_alpha_prev = m_p[m_pmask[1]];
        memcpy(m_y_prev_stages, m_y_stages, m_total_stages * sizeof(double));
        syncEndpointsFromStages(m_y_prev_end);
        updatePhaseConditions();
        memset(m_tangent, 0, (m_total_end + 3) * sizeof(double));
        m_tangent[m_total_end + 2] = 1.0;
    }
} // namespace TWIST
