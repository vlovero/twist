#include "benchmark/benchmark.h"
#include "colmat/tcolmat.h"
#include "krylov/gmres.h"
#include "linalg.h"
#include "modulating/radau_dg.h"
#include "modulating/spacetime_precond.h"
#include "modulating/spatial_slice.h"
#include "modulating/subinterval_condenser.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <omp.h>
#include <vector>

namespace
{
    // Parameterized reaction-diffusion system supporting arbitrary node >= 3 (diffusive var 0 + (node-2) local ODE vars)
    // State: [v, r, w_0, ..., w_{node-3}], params: [c, growth, omega, sps]
    constexpr double BENCH_D = 0.05;

    template <int NODE>
    void bench_func(const double *z, const double *p, double *out)
    {
        const double v = z[0];
        const double r = z[1];
        const double w0 = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];

        const double n2 = v * v + w0 * w0;
        const double f1 = growth * v - omega * w0 - n2 * v;
        const double f2 = omega * v + growth * w0 - n2 * w0;

        out[0] = sps * r;
        out[1] = sps * (c * r - f1) / BENCH_D;
        out[2] = sps * f2 / c;
        for (int i = 3; i < NODE; i++) {
            const double wi = z[i];
            out[i] = sps * (-0.5 * (double)(i - 1) * wi + 0.1 * v * (1.0 - wi)) / c;
        }
    }

    template <int NODE>
    void bench_fjac(const double *z, const double *p, double *out)
    {
        memset(out, 0, (size_t)NODE * (size_t)NODE * sizeof(double));
        const double v = z[0];
        const double w0 = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];

        const double df1_dv = growth - 3.0 * v * v - w0 * w0;
        const double df1_dw = -omega - 2.0 * v * w0;
        const double df2_dv = omega - 2.0 * v * w0;
        const double df2_dw = growth - v * v - 3.0 * w0 * w0;

        out[0 * NODE + 1] = sps;
        out[1 * NODE + 0] = -sps * df1_dv / BENCH_D;
        out[1 * NODE + 1] = sps * c / BENCH_D;
        out[1 * NODE + 2] = -sps * df1_dw / BENCH_D;
        out[2 * NODE + 0] = sps * df2_dv / c;
        out[2 * NODE + 2] = sps * df2_dw / c;

        for (int i = 3; i < NODE; i++) {
            const double wi = z[i];
            out[i * NODE + 0] = sps * 0.1 * (1.0 - wi) / c;
            out[i * NODE + i] = sps * (-0.5 * (double)(i - 1) - 0.1 * v) / c;
        }
    }

    template <int NODE>
    void bench_pjac(const double *z, const double *p, const int *pmask, const int nparam, double *out)
    {
        memset(out, 0, (size_t)NODE * (size_t)nparam * sizeof(double));
        const double v = z[0];
        const double r = z[1];
        const double w0 = z[2];
        const double c = p[0];
        const double growth = p[1];
        const double omega = p[2];
        const double sps = p[3];
        const double n2 = v * v + w0 * w0;
        const double f1 = growth * v - omega * w0 - n2 * v;
        const double f2 = omega * v + growth * w0 - n2 * w0;

        for (int q = 0; q < nparam; q++) {
            const int pid = pmask[q];
            if (pid == 0) {
                out[1 * nparam + q] = sps * r / BENCH_D;
                out[2 * nparam + q] = -sps * f2 / (c * c);
                for (int i = 3; i < NODE; i++) {
                    const double wi = z[i];
                    out[i * nparam + q] = -sps * (-0.5 * (double)(i - 1) * wi + 0.1 * v * (1.0 - wi)) / (c * c);
                }
            }
            else if (pid == 1) {
                out[1 * nparam + q] = -sps * v / BENCH_D;
                out[2 * nparam + q] = sps * w0 / c;
            }
            else if (pid == 2) {
                out[1 * nparam + q] = sps * w0 / BENCH_D;
                out[2 * nparam + q] = sps * v / c;
            }
            else {
                out[0 * nparam + q] = r;
                out[1 * nparam + q] = (c * r - f1) / BENCH_D;
                out[2 * nparam + q] = f2 / c;
                for (int i = 3; i < NODE; i++) {
                    const double wi = z[i];
                    out[i * nparam + q] = (-0.5 * (double)(i - 1) * wi + 0.1 * v * (1.0 - wi)) / c;
                }
            }
        }
    }

    template <int NODE, int NCOL, int NNODES, int NT>
    struct HopfBenchFixture
    {
        static constexpr size_t st = 3;
        double p[4] = { 1.5, 0.25, 2.0, 1.0 };
        TWIST::SpatialSlice slice;
        std::vector<TWIST::SpatialSlice> slices;
        std::vector<TWIST::SparseDenseOutputOp> transfers;
        std::vector<double> dtau;
        std::vector<double> y_stages;
        std::vector<double> y_end;
        double T = 3.14;

        HopfBenchFixture()
        {
            std::vector<double> t(NNODES);
            for (int i = 0; i < NNODES; i++) {
                t[i] = (double)i / (double)(NNODES - 1);
            }
            const diffusion_t diff = { { 0, BENCH_D } };
            slice.init((size_t)NCOL, (size_t)NODE, (size_t)NNODES, t.data(), bench_func<NODE>, bench_fjac<NODE>, bench_pjac<NODE>, diff, 2.0);
            slice.updateMassMatrices(p[0], p[3]);

            const size_t ns = slice.n_space;
            slices.resize(NT);
            transfers.resize(NT);
            dtau.assign(NT, 1.0 / (double)NT);
            y_stages.resize((size_t)NT * st * ns);
            y_end.resize((size_t)NT * ns);

            for (int n = 0; n < NT; n++) {
                slices[n] = slice;
            }
            for (int n = 0; n < NT; n++) {
                const int prev = (n == 0) ? (NT - 1) : (n - 1);
                transfers[n].build(slices[prev], slices[n]);
            }

            const TWIST::RadauDGMethod &rm = TWIST::getRadauDG(st);
            double tau_acc = 0.0;
            for (int n = 0; n < NT; n++) {
                for (size_t l = 0; l < st; l++) {
                    const double tau_nl = tau_acc + rm.c[l] * dtau[n];
                    const double phase_t = 2.0 * M_PI * tau_nl;
                    double *yl = &y_stages[((size_t)n * st + l) * ns];
                    for (int k = 0; k < NNODES - 1; k++) {
                        for (int s = 0; s <= NCOL; s++) {
                            const double xi = (s == 0) ? slice.t[k] : (slice.t[k] + slice.c_c[s - 1] * slice.h[k]);
                            const size_t off = ((size_t)k * (NCOL + 1) + (size_t)s) * NODE;
                            const double ph = 2.0 * M_PI * xi + 0.3 * std::sin(phase_t);
                            yl[off + 0] = 0.4 * std::cos(ph);
                            yl[off + 1] = -0.4 * 2.0 * M_PI * std::sin(ph);
                            yl[off + 2] = 0.3 * std::sin(ph);
                            for (int d = 3; d < NODE; d++) {
                                yl[off + d] = 0.1 * std::cos(ph + 0.2 * d);
                            }
                        }
                    }
                    const size_t end_off = (size_t)(NNODES - 1) * (NCOL + 1) * NODE;
                    memcpy(&yl[end_off], yl, NODE * sizeof(double));
                }
                memcpy(&y_end[(size_t)n * ns], &y_stages[((size_t)n * st + (st - 1)) * ns], ns * sizeof(double));
                tau_acc += dtau[n];
            }
        }
    };

    template <int NODE, int NCOL, int NNODES, int NT>
    const HopfBenchFixture<NODE, NCOL, NNODES, NT> &get_fixture()
    {
        static const HopfBenchFixture<NODE, NCOL, NNODES, NT> fx;
        return fx;
    }
} // namespace

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_SpatialSlice_ApplyMass(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    const size_t ns = fx.slice.n_space;
    std::vector<double> out(ns, 0.0);
    std::vector<double> work(NODE * NCOL + 64, 0.0);
    const double *in = fx.y_end.data();

    for (auto _ : state) {
        fx.slice.applyMass(in, out.data(), 1.0, 0.0, work.data());
        benchmark::DoNotOptimize(out.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_ColMat_AddMassBlock_Real(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    DEV::BoundaryConditions bc;
    bc.makePeriodic(NODE, 0);
    int pmask[2] = { 0, 0 };
    DEV::Matrix base, mat;
    base.update(bench_fjac<NODE>, bench_pjac<NODE>, NNODES, NODE, NCOL, 0, fx.slice.unscaled_spatial_period, fx.slice.h, fx.y_end.data(), fx.p, pmask, nullptr, &bc);

    for (auto _ : state) {
        mat.copyPartitionsFrom(base, &bc);
        mat.addMassBlock(fx.slice.M, fx.slice.c_A, fx.slice.c_b, fx.slice.h, -1.25);
        benchmark::DoNotOptimize(mat);
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_ColMat_AddMassBlock_Complex(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    DEV::BoundaryConditions rbc;
    rbc.makePeriodic(NODE, 0);
    DEV::ComplexBoundaryConditions cbc;
    cbc.makePeriodic(NODE, 0);
    int pmask[2] = { 0, 0 };
    DEV::Matrix base;
    DEV::ComplexMatrix mat;
    base.update(bench_fjac<NODE>, bench_pjac<NODE>, NNODES, NODE, NCOL, 0, fx.slice.unscaled_spatial_period, fx.slice.h, fx.y_end.data(), fx.p, pmask, nullptr, &rbc);
    const std::complex<double> scale(-1.25, 0.85);

    for (auto _ : state) {
        mat.copyPartitionsFrom(base, &cbc);
        mat.addMassBlock(fx.slice.M, fx.slice.c_A, fx.slice.c_b, fx.slice.h, scale);
        benchmark::DoNotOptimize(mat);
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_ColMat_Factor_Complex(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    DEV::BoundaryConditions rbc;
    rbc.makePeriodic(NODE, 0);
    DEV::ComplexBoundaryConditions cbc;
    cbc.makePeriodic(NODE, 0);
    int pmask[2] = { 0, 0 };
    DEV::Matrix base;
    DEV::ComplexMatrix mat;
    base.update(bench_fjac<NODE>, bench_pjac<NODE>, NNODES, NODE, NCOL, 0, fx.slice.unscaled_spatial_period, fx.slice.h, fx.y_end.data(), fx.p, pmask, nullptr, &rbc);
    const std::complex<double> scale(-1.25, 0.85);

    for (auto _ : state) {
        state.PauseTiming();
        mat.copyPartitionsFrom(base, &cbc);
        mat.addMassBlock(fx.slice.M, fx.slice.c_A, fx.slice.c_b, fx.slice.h, scale);
        state.ResumeTiming();
        mat.factor();
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_ColMat_Solve_Complex(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    DEV::BoundaryConditions rbc;
    rbc.makePeriodic(NODE, 0);
    DEV::ComplexBoundaryConditions cbc;
    cbc.makePeriodic(NODE, 0);
    int pmask[2] = { 0, 0 };
    DEV::Matrix base;
    DEV::ComplexMatrix mat;
    base.update(bench_fjac<NODE>, bench_pjac<NODE>, NNODES, NODE, NCOL, 0, fx.slice.unscaled_spatial_period, fx.slice.h, fx.y_end.data(), fx.p, pmask, nullptr, &rbc);
    mat.copyPartitionsFrom(base, &cbc);
    mat.addMassBlock(fx.slice.M, fx.slice.c_A, fx.slice.c_b, fx.slice.h, std::complex<double>(-1.25, 0.85));
    mat.factor();

    const size_t ns = fx.slice.n_space;
    std::vector<std::complex<double>> rhs(ns), sol(ns);
    for (size_t i = 0; i < ns; i++) {
        rhs[i] = std::complex<double>(fx.y_end[i], 0.5 * fx.y_end[(i + 1) % ns]);
    }

    for (auto _ : state) {
        mat.solve(rhs.data(), sol.data());
        benchmark::DoNotOptimize(sol.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_ColMat_SolveBatch64_Complex(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    DEV::BoundaryConditions rbc;
    rbc.makePeriodic(NODE, 0);
    DEV::ComplexBoundaryConditions cbc;
    cbc.makePeriodic(NODE, 0);
    int pmask[2] = { 0, 0 };
    DEV::Matrix base;
    DEV::ComplexMatrix mat;
    base.update(bench_fjac<NODE>, bench_pjac<NODE>, NNODES, NODE, NCOL, 0, fx.slice.unscaled_spatial_period, fx.slice.h, fx.y_end.data(), fx.p, pmask, nullptr, &rbc);
    mat.copyPartitionsFrom(base, &cbc);
    mat.addMassBlock(fx.slice.M, fx.slice.c_A, fx.slice.c_b, fx.slice.h, std::complex<double>(-1.25, 0.85));
    mat.factor();

    constexpr ptrdiff_t K = 64;
    const size_t ns = fx.slice.n_space;
    std::vector<std::complex<double>> rhs(ns * (size_t)K), sol(ns * (size_t)K);
    for (ptrdiff_t c = 0; c < K; c++) {
        for (size_t i = 0; i < ns; i++) {
            rhs[(size_t)c * ns + i] = std::complex<double>(fx.y_end[(i + (size_t)c) % ns], 0.5 * fx.y_end[(i + 1) % ns]);
        }
    }

    for (auto _ : state) {
        mat.solve(rhs.data(), sol.data(), K);
        benchmark::DoNotOptimize(sol.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_Condenser_Factor(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    TWIST::SubintervalCondenser cond;
    const double dtau_T = fx.dtau[0] * fx.T;

    for (auto _ : state) {
        cond.factor(&fx.slice, fx.st, fx.y_stages.data(), fx.p, dtau_T, true);
        benchmark::DoNotOptimize(cond);
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_Condenser_ApplyCondensedC(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    TWIST::SubintervalCondenser cond;
    const double dtau_T = fx.dtau[0] * fx.T;
    cond.factor(&fx.slice, fx.st, fx.y_stages.data(), fx.p, dtau_T, true);
    const size_t ns = fx.slice.n_space;
    std::vector<double> out(ns, 0.0);

    for (auto _ : state) {
        cond.applyCondensedC(fx.y_end.data(), fx.transfers[0], out.data(), true);
        benchmark::DoNotOptimize(out.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_Condenser_ComputeCondensedMatrix(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    TWIST::SubintervalCondenser cond;
    const double dtau_T = fx.dtau[0] * fx.T;
    cond.factor(&fx.slice, fx.st, fx.y_stages.data(), fx.p, dtau_T, true);
    const size_t ns = fx.slice.n_space;
    std::vector<double> C_out(ns * ns, 0.0);

    for (auto _ : state) {
        cond.computeCondensedMatrix(fx.transfers[0], C_out.data());
        benchmark::DoNotOptimize(C_out.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_Precond_Update(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    TWIST::RealSchurSpaceTimePrecond precond;

    for (auto _ : state) {
        precond.update(NT, fx.slices.data(), nullptr, fx.y_end.data(), fx.dtau.data(), fx.T, fx.p, 1.0);
        benchmark::DoNotOptimize(precond);
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_Precond_Apply(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    TWIST::RealSchurSpaceTimePrecond precond;
    precond.update(NT, fx.slices.data(), nullptr, fx.y_end.data(), fx.dtau.data(), fx.T, fx.p, 1.0);
    std::vector<double> out(fx.y_end.size(), 0.0);

    for (auto _ : state) {
        precond.apply(fx.y_end.data(), out.data());
        benchmark::DoNotOptimize(out.data());
    }
}

template <int NODE, int NCOL, int NNODES, int NT>
static void BM_GMRES_Weighted(benchmark::State &state)
{
    OpenBLASThreadContext ob(1);
    const auto &fx = get_fixture<NODE, NCOL, NNODES, NT>();
    const size_t n = (size_t)NT * fx.slice.n_space + 3;
    std::vector<double> x(n, 0.0), b(n, 1.0), w(n, 1.0);
    for (size_t i = 0; i < n; i++) {
        b[i] = std::sin(0.01 * (double)(i + 1));
        w[i] = 0.5 + 0.5 * ((double)(i % 13) / 13.0);
    }
    const size_t restart = 40;
    std::vector<double> work(Krylov::gmres_lwork(n, (int)restart));

    auto A_op = [n](const RP(double) in, RP(double) out, size_t) -> int {
        for (size_t i = 0; i < n; i++) {
            const size_t prev = (i == 0) ? (n - 1) : (i - 1);
            out[i] = 2.0 * in[i] - 0.35 * in[prev];
        }
        return 0;
    };
    auto M_op = [n](const RP(double) r, RP(double) z, size_t) -> int {
        for (size_t i = 0; i < n; i++) {
            z[i] = 0.5 * r[i];
        }
        return 0;
    };

    for (auto _ : state) {
        memset(x.data(), 0, n * sizeof(double));
        int nit = 0;
        Krylov::gmres(A_op, M_op, n, x.data(), b.data(), w.data(), work.data(), (int)restart, 40, 1e-8, 1e-13, &nit, false);
        benchmark::DoNotOptimize(x.data());
    }
}

static void apply_hopf_args(benchmark::internal::Benchmark *b)
{
    b->MinTime(0.25);
    b->Unit(benchmark::kMillisecond);
}

// FHN-like: node=3, ncol=10, nnodes=35, nt=20
BENCHMARK(BM_SpatialSlice_ApplyMass<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_AddMassBlock_Real<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_AddMassBlock_Complex<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_Factor_Complex<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_Solve_Complex<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_SolveBatch64_Complex<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_Factor<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_ApplyCondensedC<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_ComputeCondensedMatrix<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_Precond_Update<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_Precond_Apply<3, 10, 35, 20>)->Apply(apply_hopf_args);
BENCHMARK(BM_GMRES_Weighted<3, 10, 35, 20>)->Apply(apply_hopf_args);

// BR-like: node=9, ncol=8, nnodes=56, nt=24
BENCHMARK(BM_SpatialSlice_ApplyMass<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_AddMassBlock_Real<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_AddMassBlock_Complex<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_Factor_Complex<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_Solve_Complex<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_ColMat_SolveBatch64_Complex<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_Factor<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_ApplyCondensedC<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_Condenser_ComputeCondensedMatrix<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_Precond_Update<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_Precond_Apply<9, 8, 56, 24>)->Apply(apply_hopf_args);
BENCHMARK(BM_GMRES_Weighted<9, 8, 56, 24>)->Apply(apply_hopf_args);

BENCHMARK_MAIN();
