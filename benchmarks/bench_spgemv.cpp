#include "benchmark/benchmark.h"
#include "collocator.h"
#include "linalg.h"
#include <vector>
#include <random>
#include <algorithm>

void gemv_1(const sparse::RealCSCMatrix &A, const double beta, double *y, const double alpha, const double *x)
{
    const ptrdiff_t ncols = A.ncols;
    const int64_t *Ap = A.Ap;
    const int64_t *Ai = A.Ai;
    const double *Ax = A.Ax;

    ptrdiff_t i, j;

    for (i = 0; i < ncols; i++) {
        y[i] *= beta;
    }
    for (i = 0; i < ncols; i++) {
        for (j = Ap[i]; j < Ap[i + 1]; j++) {
            y[Ai[j]] += alpha * (Ax[j] * x[i]);
        }
    }
}

void gemv_2(const sparse::RealCSCMatrix &A, const double beta, double *y, const double alpha, const double *x)
{
    const ptrdiff_t ncols = A.ncols;
    const int64_t *Ap = A.Ap;
    const int64_t *Ai = A.Ai;
    const double *Ax = A.Ax;

    ptrdiff_t i, j, k, start, stop1, stop2;

    for (i = 0; i < ncols; i++) {
        y[i] *= beta;
    }
#pragma omp parallel for private(i, j, start, stop1)
    for (i = 0; i < ncols; i++) {
        start = Ap[i];
        stop1 = Ap[i + 1];
        for (j = start; j < stop1; j++) {
#pragma omp critical
            y[Ai[j]] += alpha * (Ax[j] * x[i]);
        }
    }
}

static void BM_gemv_1(benchmark::State &state)
{
    TWIST::Collocator collocator(".cache/continuation_data/tnnp-default-sps-latest.h5", 0);
    auto A = collocator.getBaseJacobian();
    std::vector<double> x(A.ncols, 0.1);
    std::vector<double> y(A.ncols);
    for (auto _ : state) {
        gemv_1(A, 0.0, y.data(), 1.0, x.data());
    }
}

static void BM_gemv_2(benchmark::State &state)
{
    TWIST::Collocator collocator(".cache/continuation_data/tnnp-default-sps-latest.h5", 0);
    auto A = collocator.getBaseJacobian();
    std::vector<double> x(A.ncols, 0.1);
    std::vector<double> y(A.ncols);
    for (auto _ : state) {
        gemv_2(A, 0.0, y.data(), 1.0, x.data());
    }
}


static void apply_args(benchmark::internal::Benchmark *b)
{
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}


BENCHMARK(BM_gemv_1)->Apply(apply_args);
BENCHMARK(BM_gemv_2)->Apply(apply_args);

BENCHMARK_MAIN();
