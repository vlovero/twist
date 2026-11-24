#pragma once

#include "krylov/periodic_krylov_schur.h"
#include "modulating/radau_dg.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include "shared.h"
#include <complex>
#include <cstddef>
#include <vector>

namespace TWIST
{
    struct FloquetSpectrum
    {
        std::vector<double> mu_real;
        std::vector<double> mu_imag;
        int n_unstable = 0;
        double trivial_err = 0.0;
        int trivial_idx[2] = { -1, -1 };
        double max_nontrivial_mag = 0.0;
    };

    class FloquetAnalyzer
    {
    public:
        FloquetAnalyzer() = default;
        FloquetAnalyzer(const FloquetAnalyzer &) = delete;
        FloquetAnalyzer &operator=(const FloquetAnalyzer &) = delete;
        ~FloquetAnalyzer() = default;

        // Computes leading `neig` Floquet multipliers using pre-factored condensers & transfers
        // (or factors them if `condensers == nullptr`).
        // `y_stages` has length sum_{n=0}^{nt-1} st * slices[n].n_space.
        FloquetSpectrum compute(size_t nt, size_t st, const SpatialSlice *slices, const SparseDenseOutputOp *transfers, const SubintervalCondenser *condensers, const double *y_stages, const double *dtau, double T, const double *p, size_t neig = 12, size_t nsub = 24, double tol = 1e-10, double unstable_tol = 1e-4, bool verbose = false);

        // Computes Bloch-Floquet multipliers with spatial phase shift e^{i gamma} y(tau, 0) = y(tau, 1)
        // along the converged periodic orbit (useful for essential stability checks).
        void computeBloch(size_t nt, size_t st, const SpatialSlice *slices, const SparseDenseOutputOp *transfers, const double *y_stages, const double *dtau, double T, const double *p, double gamma, std::vector<double> &out_mu_real, std::vector<double> &out_mu_imag, size_t neig = 8, size_t nsub = 20, double tol = 1e-9);

    private:
        Krylov::PeriodicKSResult m_ks_result;
    };
} // namespace TWIST
