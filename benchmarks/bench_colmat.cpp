#include "benchmark/benchmark.h"
#include "collocator.h"
#include "libloader.h"
#include <algorithm>

template <size_t N>
struct StringLiteral
{
    constexpr StringLiteral(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }

    char value[N];
};

void make_V(double *&V, const sparse::RealCSCMatrix &A, const ptrdiff_t nparam)
{
    int64_t i, j, k, l, start, stop;
    // allocate V
    V = (double *)realloc(V, A.nrows * sizeof(double));
    // zero out V
    memset(V, 0, A.nrows * sizeof(double));
    // extract last nparam rows from CSC matrix
    for (i = 0; i < A.ncols; i++) {
        start = A.Ap[i];
        stop = A.Ap[i + 1];
        for (j = stop - nparam; j < stop; j++) {
            k = A.Ai[j];
            if (k < (A.nrows - nparam)) {
                continue;
            }
            l = k - A.nrows + nparam;
            V[i * nparam + l] = A.Ax[j];
        }
    }
}

template <StringLiteral name, int N_nodes = -1>
static void BM_umfpack(benchmark::State &state)
{
    TWIST::Collocator collocator(fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", name.value), 0);
    if (N_nodes > 0) {
        if (collocator.solveWithAdaptation(1, 1e-12, N_nodes, N_nodes, 0.0, pow(0.5, 15), 100, false)) {
            throw std::runtime_error(fmt::format("{} solver failed using {} grid points", __PRETTY_FUNCTION__, N_nodes));
        }
    }
    sparse::CollocationMatrix mat;

    for (auto _ : state) {
        auto A = collocator.getBaseJacobian();
        A.factor();
    }
}

template <StringLiteral name, int N_nodes = -1>
static void BM_mixed(benchmark::State &state)
{
    TWIST::Collocator collocator(fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", name.value), 0);
    if (N_nodes > 0) {
        if (collocator.solveWithAdaptation(1, 1e-12, N_nodes, N_nodes, 0.0, pow(0.5, 15), 100, false)) {
            throw std::runtime_error(fmt::format("{} solver failed using {} grid points", __PRETTY_FUNCTION__, N_nodes));
        }
    }
    sparse::CollocationMatrix mat;
    auto A = collocator.getBaseJacobian();
    mat.update(&A, collocator.getNNodes(), collocator.getNStages(), collocator.getNode(), 1);

    for (auto _ : state) {
        A = collocator.getBaseJacobian();
        mat.update(&A, collocator.getNNodes(), collocator.getNStages(), collocator.getNode(), 1);
        mat.factor();
    }
}

template <StringLiteral name, int N_nodes = -1>
static void BM_tree(benchmark::State &state)
{
    TWIST::Collocator collocator(fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", name.value), 0);
    if (N_nodes > 0) {
        if (collocator.solveWithAdaptation(1, 1e-12, N_nodes, N_nodes, 0.0, pow(0.5, 15), 100, false)) {
            throw std::runtime_error(fmt::format("{} solver failed using {} grid points", __PRETTY_FUNCTION__, N_nodes));
        }
    }
    DEV::Matrix mat;
    auto A = collocator.getBaseJacobian();
    const ptrdiff_t ngrid = collocator.getNNodes();
    const ptrdiff_t node = collocator.getNode();
    const ptrdiff_t ncol = collocator.getNStages();
    const ptrdiff_t nparam = 1;
    const double L = collocator.unscaledSpatialPeriod();
    const double *h = collocator.h();
    const double *y = collocator.y();
    const double *p = collocator.p();
    const int pmask[2] = { 0, (int)collocator.getContinuationParameterIndex() };
    double *V = nullptr;
    void *handle = collocator.getLibHandle();
    func_t fjac = (func_t)dlsym(handle, "fjac");
    pjac_t pjac = (pjac_t)dlsym(handle, "pjac");
    DEV::BoundaryConditions bc;
    bc.makePeriodic(node, nparam);
    make_V(V, A, nparam);

    for (auto _ : state) {
        mat.update(fjac, pjac, ngrid, node, ncol, nparam, L, h, y, p, pmask, V, &bc);
        mat.factor();
    }
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}

BENCHMARK(BM_umfpack<"fhn">)->Apply(apply_args);
BENCHMARK(BM_mixed<"fhn">)->Apply(apply_args);
BENCHMARK(BM_tree<"fhn">)->Apply(apply_args);


BENCHMARK(BM_umfpack<"tnnp">)->Apply(apply_args);
BENCHMARK(BM_mixed<"tnnp">)->Apply(apply_args);
BENCHMARK(BM_tree<"tnnp">)->Apply(apply_args);

BENCHMARK(BM_umfpack<"tnnp_full_m_p">)->Apply(apply_args);
BENCHMARK(BM_mixed<"tnnp_full_m_p">)->Apply(apply_args);
BENCHMARK(BM_tree<"tnnp_full_m_p">)->Apply(apply_args);

BENCHMARK_MAIN();
