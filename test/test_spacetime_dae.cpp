#include "collocator.h"
#include "colmat/tcolmat.h"
#include "fmt/core.h"
#include "krylov/gmres.h"
#include "linalg.h"
#include "modulating/radau_dg.h"
#include "modulating/spacetime_precond.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fmt::println("FAILED: {} at {}:{}", #cond, __FILE__, __LINE__); std::exit(1); } } while(0)

namespace
{
    // Autonomous 3-variable first-order reaction-diffusion system:
    //   v_t = D v_xx + f1(v, w),  w_t = f2(v, w)
    // State z = [v, r, w], params p = [c, growth, omega, sps]
    // Linearized around v=0, w=0 (with nonlinear cubic saturation):
    //   f1 = growth * v - omega * w - (v^2 + w^2) * v
    //   f2 = omega * v + growth * w - (v^2 + w^2) * w
    constexpr double TEST_D = 0.05;

    void test_func(const double *z, const double *p, double *out)
    {
        const double v = z[0];
        const double r = z[1];
        const double w = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];

        const double n2 = v * v + w * w;
        const double f1 = growth * v - omega * w - n2 * v;
        const double f2 = omega * v + growth * w - n2 * w;

        out[0] = sps * r;
        out[1] = sps * (c * r - f1) / TEST_D;
        out[2] = sps * f2 / c;
    }

    void test_fjac(const double *z, const double *p, double *out)
    {
        const double v = z[0];
        const double r = z[1];
        const double w = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];
        (void)r;

        const double df1_dv = growth - 3.0 * v * v - w * w;
        const double df1_dw = -omega - 2.0 * v * w;
        const double df2_dv = omega - 2.0 * v * w;
        const double df2_dw = growth - v * v - 3.0 * w * w;

        // 3x3 row-major Jacobian
        out[0] = 0.0;                  out[1] = sps;                  out[2] = 0.0;
        out[3] = -sps * df1_dv / TEST_D; out[4] = sps * c / TEST_D;   out[5] = -sps * df1_dw / TEST_D;
        out[6] = sps * df2_dv / c;     out[7] = 0.0;                  out[8] = sps * df2_dw / c;
    }

    void test_pjac(const double *z, const double *p, const int *pmask, const int nparam, double *out)
    {
        const double v = z[0];
        const double r = z[1];
        const double w = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];
        const double n2 = v * v + w * w;
        const double f1 = growth * v - omega * w - n2 * v;
        const double f2 = omega * v + growth * w - n2 * w;

        for (int i = 0; i < nparam; i++) {
            const int pid = pmask[i];
            if (pid == 0) {
                out[0 * nparam + i] = 0.0;
                out[1 * nparam + i] = sps * r / TEST_D;
                out[2 * nparam + i] = -sps * f2 / (c * c);
            }
            else if (pid == 1) {
                out[0 * nparam + i] = 0.0;
                out[1 * nparam + i] = -sps * v / TEST_D;
                out[2 * nparam + i] = sps * w / c;
            }
            else if (pid == 2) {
                out[0 * nparam + i] = 0.0;
                out[1 * nparam + i] = sps * w / TEST_D;
                out[2 * nparam + i] = sps * v / c;
            }
            else {
                out[0 * nparam + i] = r;
                out[1 * nparam + i] = (c * r - f1) / TEST_D;
                out[2 * nparam + i] = f2 / c;
            }
        }
    }

    // 1. Verify addMassBlock on DEV::Matrix and DEV::ComplexMatrix against dense LAPACK
    void verify_add_mass_block()
    {
        const size_t node = 3;
        const size_t nstages = 4;
        const size_t nnodes = 7;
        const double L0 = 1.0;
        const double p[4] = { 1.5, 0.2, 2.0, 1.0 };
        const diffusion_t diff = { { 0, TEST_D } };

        std::vector<double> t(nnodes), y_init(nnodes * node, 0.0);
        for (size_t i = 0; i < nnodes; i++) {
            t[i] = (double)i / (double)(nnodes - 1);
            y_init[i * node + 0] = 0.5 * std::cos(2.0 * M_PI * t[i]);
            y_init[i * node + 1] = -M_PI * std::sin(2.0 * M_PI * t[i]);
            y_init[i * node + 2] = 0.3 * std::sin(2.0 * M_PI * t[i]);
        }

        TWIST::SpatialSlice slice;
        slice.init(nstages, node, nnodes, t.data(), test_func, test_fjac, test_pjac, diff, 2.0 * L0);
        slice.updateMassMatrices(p[0], p[3]);

        std::vector<double> Y(slice.n_space, 0.0);
        for (size_t k = 0; k < nnodes - 1; k++) {
            for (size_t s = 0; s <= nstages; s++) {
                const double xi = (s == 0) ? slice.t[k] : (slice.t[k] + slice.c_c[s - 1] * slice.h[k]);
                const size_t off = (k * (nstages + 1) + s) * node;
                Y[off + 0] = 0.5 * std::cos(2.0 * M_PI * xi);
                Y[off + 1] = -M_PI * std::sin(2.0 * M_PI * xi);
                Y[off + 2] = 0.3 * std::sin(2.0 * M_PI * xi);
            }
        }
        Y[(nnodes - 1) * (nstages + 1) * node + 0] = 0.5;
        Y[(nnodes - 1) * (nstages + 1) * node + 2] = 0.0;

        // Real DEV::Matrix solve check
        {
            DEV::Matrix mat;
            DEV::BoundaryConditions bc;
            bc.makePeriodic(node, 0);
            int pmask[2] = { 0, 0 };
            mat.update(test_fjac, test_pjac, nnodes, node, nstages, 0, slice.unscaled_spatial_period, slice.h, Y.data(), p, pmask, nullptr, &bc);
            const double scale = -3.25;
            mat.addMassBlock(slice.M, slice.c_A, slice.c_b, slice.h, scale);

            // Exact reference: (A_base + scale * B_M) * x_true = b via gemv + applyMass
            DEV::Matrix mat_base;
            mat_base.update(test_fjac, test_pjac, nnodes, node, nstages, 0, slice.unscaled_spatial_period, slice.h, Y.data(), p, pmask, nullptr, &bc);
            std::vector<double> x_true(slice.n_space), b(slice.n_space), b_mass(slice.n_space), x_sol(slice.n_space), mwork(node * nstages);
            for (size_t i = 0; i < slice.n_space; i++) {
                x_true[i] = std::sin(0.37 * (double)(i + 1));
            }
            mat_base.gemv(0.0, b.data(), 1.0, x_true.data());
            slice.applyMass(x_true.data(), b_mass.data(), scale, 0.0, mwork.data());
            for (size_t i = 0; i < slice.n_space; i++) {
                b[i] += b_mass[i];
            }
            mat.solve(b.data(), x_sol.data());

            double max_err = 0.0;
            for (size_t i = 0; i < slice.n_space; i++) {
                max_err = std::max(max_err, std::abs(x_sol[i] - x_true[i]));
            }
            fmt::println("[Test 1a] DEV::Matrix addMassBlock forward error: {:.3e}", max_err);
            CHECK(max_err < 1e-12);
        }

        // Complex DEV::ComplexMatrix solve check
        {
            DEV::Matrix mat_base;
            DEV::BoundaryConditions bc;
            bc.makePeriodic(node, 0);
            int pmask[2] = { 0, 0 };
            mat_base.update(test_fjac, test_pjac, nnodes, node, nstages, 0, slice.unscaled_spatial_period, slice.h, Y.data(), p, pmask, nullptr, &bc);

            DEV::ComplexMatrix cmat;
            DEV::ComplexBoundaryConditions cbc;
            cbc.makePeriodic(node, 0);
            cmat.copyPartitionsFrom(mat_base, &cbc);
            const std::complex<double> cscale(-2.68, 3.05);
            cmat.addMassBlock(slice.M, slice.c_A, slice.c_b, slice.h, cscale);

            std::vector<std::complex<double>> x_true(slice.n_space), b(slice.n_space), x_sol(slice.n_space);
            std::vector<double> xr(slice.n_space), xi(slice.n_space), br(slice.n_space), bi(slice.n_space), bmr(slice.n_space), bmi(slice.n_space), mwork(node * nstages);
            for (size_t i = 0; i < slice.n_space; i++) {
                xr[i] = std::cos(0.41 * (double)(i + 1));
                xi[i] = std::sin(0.29 * (double)(i + 1));
                x_true[i] = std::complex<double>(xr[i], xi[i]);
            }
            mat_base.gemv(0.0, br.data(), 1.0, xr.data());
            mat_base.gemv(0.0, bi.data(), 1.0, xi.data());
            slice.applyMass(xr.data(), bmr.data(), 1.0, 0.0, mwork.data());
            slice.applyMass(xi.data(), bmi.data(), 1.0, 0.0, mwork.data());
            for (size_t i = 0; i < slice.n_space; i++) {
                b[i] = std::complex<double>(br[i], bi[i]) + cscale * std::complex<double>(bmr[i], bmi[i]);
            }
            cmat.solve(b.data(), x_sol.data());

            double max_err = 0.0;
            for (size_t i = 0; i < slice.n_space; i++) {
                max_err = std::max(max_err, std::abs(x_sol[i] - x_true[i]));
            }
            fmt::println("[Test 1b] DEV::ComplexMatrix addMassBlock forward error: {:.3e}", max_err);
            CHECK(max_err < 1e-12);
        }
    }

    // 2. Verify dG(q) superconvergent temporal order (dG(1)=3rd, dG(2)=5th, dG(3)=7th)
    //    + algebraic constraint satisfaction across spatial mesh jumps M_{n-1} -> M_n
    //    + RealSchurSpaceTimePrecond GMRES convergence (< 15 iters) on an unstable periodic BVP (|mu_max| > 1e10)
    void verify_dg_convergence_and_precond()
    {
        const size_t node = 3;
        const size_t nstages = 8;
        const double L0 = 1.0;
        const diffusion_t diff = { { 0, TEST_D } };
        const double omega = 2.0 * M_PI;
        const double T = 1.0;
        const double growth = -0.3; // damped stable test for manufactured BVP
        const double p[4] = { 1.2, growth, omega, 1.0 };

        // Exact periodic solution (spatially non-trivial + time-periodic):
        //   v(tau, xi) = 0.25 * cos(2*pi*xi) * (1.0 + 0.4 * cos(2*pi*tau))
        //   r(tau, xi) = v_xi / (2*L0*sps)
        //   w(tau, xi) = 0.25 * sin(2*pi*xi) * (1.0 + 0.4 * sin(2*pi*tau))
        auto exact_y = [&](double tau, double xi, double *y_out, double *dydt_out) {
            const double L = 2.0 * L0 * p[3];
            const double cx = std::cos(2.0 * M_PI * xi);
            const double sx = std::sin(2.0 * M_PI * xi);
            const double ct = std::cos(2.0 * M_PI * tau);
            const double st = std::sin(2.0 * M_PI * tau);

            y_out[0] = 0.25 * cx * (1.0 + 0.4 * ct);
            y_out[1] = -0.25 * (2.0 * M_PI / L) * sx * (1.0 + 0.4 * ct);
            y_out[2] = 0.25 * sx * (1.0 + 0.4 * st);

            if (dydt_out) {
                dydt_out[0] = 0.25 * cx * (-0.4 * 2.0 * M_PI * st);
                dydt_out[1] = -0.25 * (2.0 * M_PI / L) * sx * (-0.4 * 2.0 * M_PI * st);
                dydt_out[2] = 0.25 * sx * (0.4 * 2.0 * M_PI * ct);
            }
        };

        // Spatial PDE residual (autonomous) + forcing f_ext(tau, xi) so exact_y satisfies:
        //   B * y_t = R_space(y) + B_forcing(tau)
        auto eval_exact_slice = [&](const TWIST::SpatialSlice &sl, double tau, double *y_out, double *dydt_out) {
            const size_t npts = sl.nPoints();
            for (size_t q = 0; q < npts; q++) {
                double xi = 1.0;
                if (q < npts - 1) {
                    const size_t k = q / (sl.nstages + 1);
                    const size_t s = q % (sl.nstages + 1);
                    xi = (s == 0) ? sl.t[k] : (sl.t[k] + sl.c_c[s - 1] * sl.h[k]);
                }
                exact_y(tau, xi, &y_out[q * node], dydt_out ? &dydt_out[q * node] : nullptr);
            }
        };

        for (size_t st : { (size_t)2, (size_t)3, (size_t)4 }) {
            const TWIST::RadauDGMethod &rm = TWIST::getRadauDG(st);
            std::vector<double> errors;
            std::vector<size_t> nts = { 6, 12, 24 };
            if (st == 4) nts = { 4, 8, 16 }; // dG(3) hits machine precision quickly!

            for (size_t nt : nts) {
                std::vector<TWIST::SpatialSlice> slices(nt);
                std::vector<TWIST::SparseDenseOutputOp> transfers(nt);
                std::vector<TWIST::SubintervalCondenser> condensers(nt);
                std::vector<double> dtau(nt, 1.0 / (double)nt);

                // Varying spatial meshes M_n across temporal elements!
                for (size_t n = 0; n < nt; n++) {
                    const size_t nn = 9 + (n % 3); // alternating 9, 10, 11 spatial nodes!
                    std::vector<double> t_sp(nn);
                    for (size_t i = 0; i < nn; i++) {
                        const double u = (double)i / (double)(nn - 1);
                        t_sp[i] = u + 0.05 * ((n % 2 == 0) ? 1.0 : -1.0) * std::sin(2.0 * M_PI * u);
                    }
                    t_sp[0] = 0.0;
                    t_sp[nn - 1] = 1.0;
                    slices[n].init(nstages, node, nn, t_sp.data(), test_func, test_fjac, test_pjac, diff, 2.0 * L0);
                    slices[n].updateMassMatrices(p[0], p[3]);
                }
                for (size_t n = 0; n < nt; n++) {
                    transfers[n].build(slices[(n + nt - 1) % nt], slices[n]);
                }

                std::vector<size_t> end_off(nt + 1, 0), st_off(nt + 1, 0);
                for (size_t n = 0; n < nt; n++) {
                    end_off[n + 1] = end_off[n] + slices[n].n_space;
                    st_off[n + 1] = st_off[n] + st * slices[n].n_space;
                }
                const size_t total_end = end_off[nt];
                const size_t total_stages = st_off[nt];

                // Build exact manufactured RHS in dG(q) residual form around linear-plus-nonlinear exact solution:
                // Actually test linear periodic BVP with exact time-dependent forcing g(tau) = B*y_t - A*y,
                // OR Newton solve on B*y_t = R_space(y) + F_man(tau)!
                std::vector<double> y_ex_stages(total_stages), f_man_stages(total_stages), y_stages(total_stages, 0.0);
                for (size_t n = 0; n < nt; n++) {
                    const size_t ns = slices[n].n_space;
                    const double tau0 = (double)n * dtau[n];
                    std::vector<double> dydt(ns), R_sp(ns), yp(ns), B_dydt(ns), mwork(node * nstages);
                    for (size_t l = 0; l < st; l++) {
                        const double tau_l = tau0 + rm.c[l] * dtau[n];
                        double *yl = &y_ex_stages[st_off[n] + l * ns];
                        eval_exact_slice(slices[n], tau_l, yl, dydt.data());
                        slices[n].evaluateResidual(yl, p, R_sp.data(), yp.data());
                        slices[n].applyMass(dydt.data(), B_dydt.data(), 1.0, 0.0, mwork.data());
                        double *fl = &f_man_stages[st_off[n] + l * ns];
                        for (size_t i = 0; i < ns; i++) {
                            fl[i] = B_dydt[i] - R_sp[i];
                        }
                    }
                }

                // Initial guess: exact + perturbation
                for (size_t i = 0; i < total_stages; i++) {
                    y_stages[i] = y_ex_stages[i] + 0.05 * std::sin(0.13 * (double)(i + 1));
                }

                TWIST::RealSchurSpaceTimePrecond precond;
                std::vector<double> weights_end(total_end);
                for (size_t n = 0; n < nt; n++) {
                    slices[n].fillQuadratureWeights(&weights_end[end_off[n]], true);
                    for (size_t i = 0; i < slices[n].n_space; i++) {
                        weights_end[end_off[n] + i] *= dtau[n];
                    }
                }
                std::vector<double> gmres_work(Krylov::gmres_lwork(total_end, 40));

                // Solve nonlinear manufactured space-time dG(q) BVP with Newton-GMRES
                for (int it = 0; it < 12; it++) {
                    std::vector<double> G(total_stages), b_Y(total_end), sol_Y(total_end, 0.0);
                    for (size_t n = 0; n < nt; n++) {
                        const size_t ns = slices[n].n_space;
                        const size_t prev = (n + nt - 1) % nt;
                        const double *y_prev = &y_stages[st_off[prev] + (st - 1) * slices[prev].n_space];
                        const double *yn = &y_stages[st_off[n]];
                        std::vector<double> y_in(ns), r_sp(st * ns), yp(ns), tmp(ns), mwork(node * nstages);
                        transfers[n].apply(y_prev, y_in.data());
                        for (size_t l = 0; l < st; l++) {
                            slices[n].evaluateResidual(&yn[l * ns], p, &r_sp[l * ns], yp.data());
                            for (size_t i = 0; i < ns; i++) {
                                r_sp[l * ns + i] += f_man_stages[st_off[n] + l * ns + i];
                            }
                        }
                        const double dtT = dtau[n] * T;
                        for (size_t j = 0; j < st; j++) {
                            double *Gnj = &G[st_off[n] + j * ns];
                            for (size_t i = 0; i < ns; i++) tmp[i] = yn[j * ns + i] - y_in[i];
                            slices[n].applyMass(tmp.data(), Gnj, 1.0, 0.0, mwork.data());
                            for (size_t l = 0; l < st; l++) {
                                const double sc = -dtT * rm.A[j * st + l];
                                for (size_t i = 0; i < ns; i++) Gnj[i] += sc * r_sp[l * ns + i];
                            }
                        }
                    }

                    const double rnorm = norm2(G.data(), total_stages);
                    if (rnorm < 1e-13) break;

                    for (size_t n = 0; n < nt; n++) {
                        const size_t ns = slices[n].n_space;
                        condensers[n].factor(&slices[n], st, &y_stages[st_off[n]], p, dtau[n] * T, true);
                        std::vector<double> rhs_n(st * ns), dk_n(st * ns);
                        for (size_t i = 0; i < st * ns; i++) rhs_n[i] = -G[st_off[n] + i];
                        condensers[n].solveLocalStages(rhs_n.data(), dk_n.data(), true);
                        memcpy(&b_Y[end_off[n]], &dk_n[(st - 1) * ns], ns * sizeof(double));
                    }

                    std::vector<double> y_end(total_end);
                    for (size_t n = 0; n < nt; n++) {
                        const size_t ns = slices[n].n_space;
                        memcpy(&y_end[end_off[n]], &y_stages[st_off[n] + (st - 1) * ns], ns * sizeof(double));
                    }
                    precond.update(nt, slices.data(), condensers.data(), y_end.data(), dtau.data(), T, p, 1.0);

                    auto A_op = [&](const RP(double) x, RP(double) y, size_t) -> int {
                        for (size_t n = 0; n < nt; n++) {
                            const size_t ns = slices[n].n_space;
                            const size_t prev = (n + nt - 1) % nt;
                            condensers[n].applyCondensedC(&x[end_off[prev]], transfers[n], &y[end_off[n]], true);
                            for (size_t i = 0; i < ns; i++) {
                                y[end_off[n] + i] = x[end_off[n] + i] - y[end_off[n] + i];
                            }
                        }
                        return 0;
                    };
                    auto M_op = [&](const RP(double) r, RP(double) z, size_t) -> int {
                        precond.apply(r, z);
                        return 0;
                    };
                    int gm_iters = 0;
                    const int gm_status = Krylov::gmres(A_op, M_op, total_end, sol_Y.data(), b_Y.data(), weights_end.data(), gmres_work.data(), 40, 80, 1e-12, 1e-14, &gm_iters, false);
                    fmt::println("  [Nt={}] it={}: ||G||={:.3e}, gmres_status={}, gm_iters={}", nt, it, rnorm, gm_status, gm_iters);

                    for (size_t n = 0; n < nt; n++) {
                        const size_t ns = slices[n].n_space;
                        const size_t prev = (n + nt - 1) % nt;
                        std::vector<double> rhs_n(st * ns), dk_n(st * ns), dy_in(ns), B_dy(ns), mwork(node * nstages);
                        transfers[n].apply(&sol_Y[end_off[prev]], dy_in.data());
                        slices[n].applyMass(dy_in.data(), B_dy.data(), 1.0, 0.0, mwork.data());
                        for (size_t j = 0; j < st; j++) {
                            for (size_t i = 0; i < ns; i++) {
                                rhs_n[j * ns + i] = -G[st_off[n] + j * ns + i] + B_dy[i];
                            }
                        }
                        condensers[n].solveLocalStages(rhs_n.data(), dk_n.data(), true);
                        for (size_t i = 0; i < st * ns; i++) {
                            y_stages[st_off[n] + i] += dk_n[i];
                        }
                    }
                }

                // Check endpoint error against exact solution AND algebraic constraint error (v_xi = r, BCs)
                double max_end_err = 0.0;
                double max_alg_err = 0.0;
                for (size_t n = 0; n < nt; n++) {
                    const size_t ns = slices[n].n_space;
                    const double *y_end_n = &y_stages[st_off[n] + (st - 1) * ns];
                    const double *y_ex_n = &y_ex_stages[st_off[n] + (st - 1) * ns];
                    for (size_t i = 0; i < ns; i++) {
                        max_end_err = std::max(max_end_err, std::abs(y_end_n[i] - y_ex_n[i]));
                    }
                    // Verify algebraic rows of R_space(y) + f_man == 0 directly at stage 0 (immediately after mesh jump M_{n-1} -> M_n!)
                    const double *y_st0 = &y_stages[st_off[n]];
                    std::vector<double> r_sp(ns), yp(ns);
                    slices[n].evaluateResidual(y_st0, p, r_sp.data(), yp.data());
                    const size_t nsub = slices[n].nSub();
                    // Check periodic BC rows (last `node` entries) and v_xi - L*r rows (component 0 of every subinterval)
                    for (size_t k = 0; k < nsub; k++) {
                        for (size_t j = 0; j <= nstages; j++) {
                            const size_t idx_v = (k * (nstages + 1) + j) * node + 0;
                            max_alg_err = std::max(max_alg_err, std::abs(r_sp[idx_v] + f_man_stages[st_off[n] + idx_v]));
                        }
                    }
                    for (size_t d = 0; d < node; d++) {
                        max_alg_err = std::max(max_alg_err, std::abs(r_sp[nsub * (nstages + 1) * node + d]));
                    }
                }
                errors.push_back(max_end_err);
                fmt::println("[Test 2] dG({}) Nt={:2d}: endpoint err={:.3e}, alg constraint err={:.3e}", st - 1, nt, max_end_err, max_alg_err);
                CHECK(max_alg_err < 1e-12);
            }

            const double order1 = std::log2(errors[0] / errors[1]);
            const double order2 = std::log2(errors[1] / errors[2]);
            const double expected_order = (double)(2 * st - 1);
            fmt::println("[Test 2] dG({}) observed orders: {:.2f}, {:.2f} (expected {:.1f})", st - 1, order1, order2, expected_order);
            CHECK(order1 >= expected_order - 0.35);
            if (errors[2] > 1e-13) {
                CHECK(order2 >= expected_order - 0.35);
            }
        }

        // 3. Test RealSchurSpaceTimePrecond GMRES on a STRONGLY UNSTABLE periodic BVP (|mu_max| > 1e10)
        //    Take growth = +25.0, T = 1.0 -> Floquet multiplier exp(+25) ~ 7.2e10!
        {
            const size_t nt = 24;
            const size_t st = 3;
            // Same 1D RD system as Test 2, except unstable diffusivity-controlled mode growth = +25 on v
            // while w decays stably (realistic RD Hopf/excitable regime: finite number of unstable spatial modes!)
            const double T_unst = 1.0;
            const double p_unst[4] = { 1.2, 25.0, 2.0 * M_PI, 1.0 };
            auto f_unst = [](const double *z, const double *pp, double *out) {
                const double v = z[0], r = z[1], w = z[2];
                const double c = pp[0], gr = pp[1], om = pp[2], sps = pp[3];
                const double f1 = gr * v - om * w;
                const double f2 = om * v - 2.0 * w;
                out[0] = sps * r;
                out[1] = sps * (c * r - f1) / TEST_D;
                out[2] = sps * f2 / c;
            };
            auto fj_unst = [](const double *, const double *pp, double *out) {
                const double c = pp[0], gr = pp[1], om = pp[2], sps = pp[3];
                out[0] = 0.0;              out[1] = sps;            out[2] = 0.0;
                out[3] = -sps * gr / TEST_D; out[4] = sps * c / TEST_D; out[5] = sps * om / TEST_D;
                out[6] = sps * om / c;     out[7] = 0.0;            out[8] = -sps * 2.0 / c;
            };

            std::vector<TWIST::SpatialSlice> slices(nt);
            std::vector<TWIST::SparseDenseOutputOp> transfers(nt);
            std::vector<TWIST::SubintervalCondenser> condensers(nt);
            std::vector<double> dtau(nt, 1.0 / (double)nt);

            for (size_t n = 0; n < nt; n++) {
                const size_t nn = 10;
                std::vector<double> t_sp(nn);
                for (size_t i = 0; i < nn; i++) t_sp[i] = (double)i / (double)(nn - 1);
                slices[n].init(st == 3 ? 8 : 8, node, nn, t_sp.data(), +f_unst, +fj_unst, test_pjac, diff, 2.0 * L0);
                slices[n].updateMassMatrices(p_unst[0], p_unst[3]);
            }
            for (size_t n = 0; n < nt; n++) {
                transfers[n].build(slices[(n + nt - 1) % nt], slices[n]);
            }

            const size_t ns = slices[0].n_space;
            const size_t total_end = nt * ns;
            std::vector<double> y_stages(nt * st * ns, 0.0), y_end(total_end, 0.0), weights_end(total_end);
            for (size_t n = 0; n < nt; n++) {
                slices[n].fillQuadratureWeights(&weights_end[n * ns], true);
                for (size_t i = 0; i < ns; i++) weights_end[n * ns + i] *= dtau[n];
                condensers[n].factor(&slices[n], st, &y_stages[n * st * ns], p_unst, dtau[n] * T_unst, true);
            }

            TWIST::RealSchurSpaceTimePrecond precond;
            precond.update(nt, slices.data(), condensers.data(), y_end.data(), dtau.data(), T_unst, p_unst, 0.55);

            std::vector<double> x_true(total_end), b_Y(total_end), x_sol(total_end, 0.0);
            for (size_t n = 0; n < nt; n++) {
                const double tau_e = (double)(n + 1) / (double)nt;
                eval_exact_slice(slices[n], tau_e, &x_true[n * ns], nullptr);
            }
            auto A_op = [&](const RP(double) x, RP(double) y, size_t) -> int {
                for (size_t n = 0; n < nt; n++) {
                    const size_t prev = (n + nt - 1) % nt;
                    condensers[n].applyCondensedC(&x[prev * ns], transfers[n], &y[n * ns], true);
                    for (size_t i = 0; i < ns; i++) y[n * ns + i] = x[n * ns + i] - y[n * ns + i];
                }
                return 0;
            };
            auto M_op = [&](const RP(double) r, RP(double) z, size_t) -> int {
                precond.apply(r, z);
                return 0;
            };
            A_op(x_true.data(), b_Y.data(), total_end);
            int gm_iters = 0;
            std::vector<double> gwork(Krylov::gmres_lwork(total_end, 40));
            const int status = Krylov::gmres(A_op, M_op, total_end, x_sol.data(), b_Y.data(), weights_end.data(), gwork.data(), 40, 40, 1e-9, 1e-12, &gm_iters, false);

            double max_err = 0.0;
            for (size_t i = 0; i < total_end; i++) {
                max_err = std::max(max_err, std::abs(x_sol[i] - x_true[i]));
            }
            fmt::println("[Test 3] Unstable periodic BVP (mu_max ~ exp(25) = 7.2e10): GMRES status={}, iters={}, forward err={:.3e}", status, gm_iters, max_err);
            CHECK(status == 0);
            CHECK(gm_iters < 15);
            CHECK(max_err < 1e-8);
        }
    }
} // namespace

int main()
{
    verify_add_mass_block();
    verify_dg_convergence_and_precond();
    fmt::println("ALL test_spacetime_dae CHECKS PASSED!");
    return 0;
}
