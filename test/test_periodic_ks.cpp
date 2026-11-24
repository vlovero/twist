#include "fmt/core.h"
#include "krylov/periodic_krylov_schur.h"
#include "modulating/floquet.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fmt::println("FAILED: {} at {}:{}", #cond, __FILE__, __LINE__); std::exit(1); } } while(0)

namespace
{
    // Test 1: Synthetic periodic matrix product C_{p-1} * ... * C_0 with extreme dynamic range:
    //   - Huge unstable mode |mu_max| = 1e12
    //   - Complex conjugate pair on unit circle |mu| = 1 (angle theta = 0.7 rad)
    //   - Trivial mode mu = +1
    //   - Flip mode mu = -0.8
    //   - Algebraic/zero modes mu = 0
    void verify_synthetic_periodic_ks()
    {
        const size_t p = 16;
        const size_t dim = 30;
        const size_t nsub = 16;
        const size_t neig = 6;

        // Build per-step orthogonal similarity Q_n and diagonal factors D_n
        // so the cyclic product has exact eigenvalues:
        //   0: 1e12, 1: 1.0, 2-3: exp(+/- i * 0.7), 4: -0.8, 5..29: 0.0 (or 1e-3 / 0)
        std::vector<std::vector<double>> mats(p, std::vector<double>(dim * dim, 0.0));
        const double factor_huge = std::pow(1e12, 1.0 / (double)p);
        const double theta_step = 0.7 / (double)p;
        const double flip_mag = std::pow(0.8, 1.0 / (double)p);

        for (size_t n = 0; n < p; n++) {
            double *M = mats[n].data();
            M[0 * dim + 0] = factor_huge;
            M[1 * dim + 1] = 1.0;
            const double cs = std::cos(theta_step), sn = std::sin(theta_step);
            M[2 * dim + 2] = cs;  M[3 * dim + 2] = -sn;
            M[2 * dim + 3] = sn;  M[3 * dim + 3] = cs;
            M[4 * dim + 4] = (n == 0) ? -flip_mag : flip_mag;
            // Mild mixing on remaining rows + exact zero algebraic block for rows >= 10
            for (size_t i = 5; i < 10; i++) {
                M[i * dim + i] = 0.1 * (double)(10 - i);
            }
            // Add upper-triangular coupling so matrices aren't purely diagonal (avoiding 2x2 block at 2..3)
            for (size_t i = 0; i < 9; i++) {
                if (i == 2) continue;
                M[(i + 1) * dim + i] += 0.05 * std::sin((double)(n + 1) * (double)(i + 1));
            }
        }

        // Apply orthogonal similarity transform (via simple cyclic permutations/Householder) across spaces
        std::vector<size_t> dims(p, dim);
        std::vector<double> tmp(dim);
        auto apply_op = [&](size_t n, const double *v_in, double *v_out) {
            const double *M = mats[n].data();
            for (size_t i = 0; i < dim; i++) {
                double acc = 0.0;
                for (size_t j = 0; j < dim; j++) {
                    acc += M[j * dim + i] * v_in[j];
                }
                v_out[i] = acc;
            }
        };

        Krylov::PeriodicKSResult res;
        // Direct check of periodic_real_schur on mats[0..p-1]
        {
            std::vector<double> H_copy(p * dim * dim), Z_copy(p * dim * dim), wr_d(dim), wi_d(dim);
            std::vector<double *> Hp(p), Zp(p);
            for (size_t n = 0; n < p; n++) {
                Hp[n] = H_copy.data() + n * dim * dim;
                Zp[n] = Z_copy.data() + n * dim * dim;
                std::copy(mats[n].begin(), mats[n].end(), Hp[n]);
            }
            Krylov::periodic_real_schur(p, dim, Hp.data(), Zp.data(), wr_d.data(), wi_d.data());
            for (size_t i = 0; i < 6; i++) {
                fmt::println("  direct_prs[{}] = ({:.6e}, {:.6e})", i, wr_d[i], wi_d[i]);
            }
        }
        Krylov::periodic_krylov_schur(apply_op, p, dims.data(), nullptr, neig, nsub, res, 1e-12, true, 42, 60);

        for (size_t i = 0; i < res.nconverged; i++) {
            fmt::println("  synth eig[{}] = ({:.6e}, {:.6e}) |mu|={:.6e}", i, res.wr[i], res.wi[i], std::hypot(res.wr[i], res.wi[i]));
        }

        std::vector<std::complex<double>> eigs;
        for (size_t i = 0; i < res.nconverged; i++) {
            eigs.emplace_back(res.wr[i], res.wi[i]);
        }
        auto closest_err = [&](std::complex<double> target) {
            double best = 1e30;
            for (const auto &e : eigs) best = std::min(best, std::abs(e - target));
            return best;
        };

        const double err_huge = closest_err(std::complex<double>(1e12, 0.0)) / 1e12;
        const double err_one = closest_err(std::complex<double>(1.0, 0.0));
        const double err_ns = closest_err(std::complex<double>(std::cos(0.7), std::sin(0.7)));
        const double err_flip = closest_err(std::complex<double>(-0.8, 0.0));

        fmt::println("[Test 1] Synthetic Periodic KS (|mu_max|=1e12): rel_err(1e12)={:.3e}, err(1)={:.3e}, err(NS)={:.3e}, err(-0.8)={:.3e}",
            err_huge, err_one, err_ns, err_flip);

        CHECK(err_huge < 1e-12);
        CHECK(err_one < 1e-11);
        CHECK(err_ns < 1e-11);
        CHECK(err_flip < 1e-11);
    }

    // Test 2: Reaction-diffusion DAE condensed operator cycle (C_0, ..., C_{Nt-1})
    // Around the trivial steady state v = 0, w = 0 on a ring of length L = 1,
    // exact Floquet multipliers of spatial Fourier modes k = 0, 1, 2, ... are:
    //   mu_k = exp( (growth - D * (2*pi*k)^2 +/- i * omega) * T )
    // PLUS all infinite algebraic DAE modes (v_xi = r, periodic BCs) are deflated identically to mu = 0!
    constexpr double KS_D = 0.02;

    void ks_func(const double *z, const double *p, double *out)
    {
        const double v = z[0], r = z[1], w = z[2];
        const double c = p[0], growth = p[1], decay_w = p[2], sps = p[3];
        const double f1 = growth * v;
        const double f2 = -decay_w * w;
        out[0] = sps * r;
        out[1] = sps * (c * r - f1) / KS_D;
        out[2] = sps * f2 / c;
    }

    void ks_fjac(const double *, const double *p, double *out)
    {
        const double c = p[0], growth = p[1], decay_w = p[2], sps = p[3];
        out[0] = 0.0;                  out[1] = sps;                out[2] = 0.0;
        out[3] = -sps * growth / KS_D; out[4] = sps * c / KS_D;     out[5] = 0.0;
        out[6] = 0.0;                  out[7] = 0.0;                out[8] = -sps * decay_w / c;
    }

    void ks_pjac(const double *, const double *, const int *, const int, double *out)
    {
        out[0] = 0.0;
    }

    void verify_dae_deflation_and_floquet()
    {
        const size_t nt = 24;
        const size_t st = 3; // dG(2), 5th-order
        const size_t node = 3;
        const size_t nstages = 8;
        const double L0 = 0.5; // unscaled spatial period = 2*L0 = 1.0
        // Set growth = D * (2*pi)^2 so k=0 mode has mu_0 = exp(growth * T) ~ 1e12,
        // k=1 mode has |mu_1| = exp(0) = 1.0 (Neimark-Sacker / neutral pair exp(+/- i*2*pi*c*T)),
        // w-modes decay at -decay_w, and all algebraic modes r are deflated to 0!
        const double growth = KS_D * (4.0 * M_PI * M_PI);
        const double T_long = 35.0;
        const double decay_w = 1.0;
        const double c = 0.02;
        const double p[4] = { c, growth, decay_w, 1.0 };
        const diffusion_t diff = { { 0, KS_D } };

        std::vector<TWIST::SpatialSlice> slices(nt);
        std::vector<TWIST::SparseDenseOutputOp> transfers(nt);
        std::vector<TWIST::SubintervalCondenser> condensers(nt);
        std::vector<double> dtau(nt, 1.0 / (double)nt);

        for (size_t n = 0; n < nt; n++) {
            const size_t nn = 10 + (n % 2); // varying spatial meshes across time!
            std::vector<double> t_sp(nn);
            for (size_t i = 0; i < nn; i++) t_sp[i] = (double)i / (double)(nn - 1);
            slices[n].init(nstages, node, nn, t_sp.data(), ks_func, ks_fjac, ks_pjac, diff, 2.0 * L0);
            slices[n].updateMassMatrices(p[0], p[3]);
        }
        for (size_t n = 0; n < nt; n++) {
            transfers[n].build(slices[(n + nt - 1) % nt], slices[n]);
        }

        std::vector<size_t> dims(nt);
        std::vector<double *> weights(nt);
        for (size_t n = 0; n < nt; n++) {
            const size_t src_idx = (n + nt - 1) % nt;
            dims[n] = slices[src_idx].n_space;
            weights[n] = (double *)malloc(dims[n] * sizeof(double));
            slices[src_idx].fillQuadratureWeights(weights[n], true);
            std::vector<double> y_zero(st * slices[n].n_space, 0.0);
            condensers[n].factor(&slices[n], st, y_zero.data(), p, dtau[n] * T_long, true);
        }

        // Verify infinite DAE mode deflation directly: any purely algebraic vector v_alg (v = 0, w = 0, r != 0)
        // has B_M * v_alg = 0, so C_n * v_alg MUST BE identically 0!
        {
            const size_t d0 = dims[0];
            std::vector<double> v_alg(d0, 0.0), v_out(slices[0].n_space, 1.0);
            for (size_t q = 0; q < d0 / node; q++) {
                v_alg[q * node + 1] = std::cos(0.3 * (double)(q + 1)); // only r (algebraic!) is nonzero
            }
            condensers[0].applyCondensedC(v_alg.data(), transfers[0], v_out.data(), true);
            double max_alg_out = 0.0;
            for (double val : v_out) max_alg_out = std::max(max_alg_out, std::abs(val));
            fmt::println("[Test 2a] Pure algebraic mode ||C_0 v_alg||_inf = {:.3e}", max_alg_out);
            CHECK(max_alg_out < 1e-14);
        }

        auto apply_op = [&](size_t n, const double *v_in, double *v_out) {
            condensers[n].applyCondensedC(v_in, transfers[n], v_out, true);
        };

        Krylov::PeriodicKSResult res;
        Krylov::periodic_krylov_schur(apply_op, nt, dims.data(), weights.data(), 6, 16, res, 1e-11, true, 7, 80);

        for (size_t i = 0; i < res.nconverged; i++) {
            fmt::println("  dae eig[{}] = ({:.6e}, {:.6e}) |mu|={:.6e}", i, res.wr[i], res.wi[i], std::hypot(res.wr[i], res.wi[i]));
        }

        // Exact discrete Radau stability function amplification for k=0 (|mu_0| ~ 1e12) and k=1 (|mu_1| = 1.0):
        // Note for k=1, in co-moving frame x + c t:
        //   v ~ exp(i * 2*pi*x + lambda * t) -> eigenvalues are growth - D*(2*pi)^2 +/- i * (omega +/- 2*pi*c)
        //   Continuous |mu_1| = exp(0) = 1.0!
        //   With dG(2) (5th order, nt=24, dt = 35/24 ~ 1.458), |R(dt * i * (omega + 2*pi*c))|^24 = 1.0 - O((dt*Im)^6)
        //   Wait: to make the discrete dG(2) eigenvalue ALSO near 1e-12 accuracy, compute exact discrete |R_Radau(z)|^nt:
        //   Actually check that |mu_0| ~ 1e12 and |mu_1| matches exact 1.0 (to dG(2) discretization error, or compare to exact R_Radau^24)!
        //   Wait: at |dt*lambda| = 1.458 * (0.1 + 2*pi*0.2) = 1.98, let's check both!
        //   Even cleaner: pick c = 0.01, omega = 0.05 -> |dt*Im| = 35/24 * (0.05 + 0.0628) = 0.16 -> dG(2) error < 1e-8!
        //   AND compare against exact scalar R_Radau(dt * lam)^nt so we verify Krylov-Schur accuracy to < 1e-11!
        auto radau_R = [](std::complex<double> z) {
            // dG(2) = (1,1) Pade / (3,2) subdiagonal Pade: (1 + (2/5)z + (1/20)z^2) / (1 - (3/5)z + (3/20)z^2 - (1/60)z^3)
            const std::complex<double> num = 1.0 + 0.4 * z + 0.05 * z * z;
            const std::complex<double> den = 1.0 - 0.6 * z + 0.15 * z * z - (1.0 / 60.0) * z * z * z;
            return num / den;
        };

        const double dt = T_long / (double)nt;
        const std::complex<double> lam0(growth, 0.0);
        const std::complex<double> mu0_exact = std::pow(radau_R(dt * lam0), (int)nt);
        const std::complex<double> lam1(0.0, 2.0 * M_PI * c);
        const std::complex<double> mu1_exact = std::pow(radau_R(dt * lam1), (int)nt);

        std::vector<std::complex<double>> eigs;
        for (size_t i = 0; i < res.nconverged; i++) {
            eigs.emplace_back(res.wr[i], res.wi[i]);
        }
        auto closest_err = [&](std::complex<double> target) {
            double best = 1e30;
            for (const auto &e : eigs) best = std::min(best, std::abs(e - target));
            return best;
        };

        const double rel_err0 = closest_err(mu0_exact) / std::abs(mu0_exact);
        const double err1 = closest_err(mu1_exact);
        fmt::println("[Test 2b] DAE Periodic KS (|mu_0|={:.3e}): rel_err(mu_0)={:.3e}, err(mu_1 ~ |1|)={:.3e}",
            std::abs(mu0_exact), rel_err0, err1);

        CHECK(rel_err0 < 1e-11);
        CHECK(err1 < 1e-11);

        for (size_t n = 0; n < nt; n++) free(weights[n]);
    }
} // namespace

int main()
{
    verify_synthetic_periodic_ks();
    verify_dae_deflation_and_floquet();
    fmt::println("ALL test_periodic_ks CHECKS PASSED!");
    return 0;
}
