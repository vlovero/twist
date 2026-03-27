#include "benchmark/benchmark.h"
#include "linalg.h"
#include <omp.h>
#include <vector>

void dlatr0(const ptrdiff_t n, double *A, const ptrdiff_t ldA)
{
    ptrdiff_t i, j;
    double temp;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            temp = A[i * ldA + j];
            A[i * ldA + j] = A[j * ldA + i];
            A[j * ldA + i] = temp;
        }
    }
}

void dlatr3(const ptrdiff_t n, double *A, const ptrdiff_t ldA)
{
    constexpr ptrdiff_t size = 24;
    ptrdiff_t block, i, j;
    double temp;
    int num_threads;

    num_threads = (n >= 256) ? (n / 64) : 1;
    num_threads = std::min(num_threads, omp_get_max_threads());

#pragma omp parallel for private(block, i, j, temp) num_threads(num_threads)
    for (block = 0; block < (n - size + 1); block += size) {
        // this pair takes care of main diagonal
        for (i = block; i < block + size; ++i) {
            for (j = i + 1; j < block + size; ++j) {
                temp = A[i * ldA + j];
                A[i * ldA + j] = A[j * ldA + i];
                A[j * ldA + i] = temp;
            }
        }
        // transpose sub-mats not on main diagonal
        for (i = block + size; i < n; ++i) {
            for (j = block; j < block + size; ++j) {
                temp = A[i * ldA + j];
                A[i * ldA + j] = A[j * ldA + i];
                A[j * ldA + i] = temp;
            }
        }
    }
    // transpose remaining along main diagonal
    for (i = block; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            temp = A[i * ldA + j];
            A[i * ldA + j] = A[j * ldA + i];
            A[j * ldA + i] = temp;
        }
    }
}


static void BM_dlatr0(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> A(n * n, 1.0);
    for (auto _ : state) {
        dlatr0(n, A.data(), n);
    }
}

static void BM_dlatr3(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> A(n * n, 1.0);
    for (auto _ : state) {
        dlatr3(n, A.data(), n);
    }
}

static void BM_dlatrn(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<double> A(n * n, 1.0);
    for (auto _ : state) {
        dlatrn(n, A.data(), n);
    }
}

// extern "C" void dimatcopy_(const char *order, const char *trans, const int *rows, const int *cols, const double *alpha, double *a, const int *lda, const int *ldb);
extern "C" void dimatcopy_k_ct(const long rows, const long cols, const double alpha, double *a, const long lda);

static void BM_imatcopy(benchmark::State &state)
{
    // void NAME( char* ORDER, char* TRANS, blasint *rows, blasint *cols, FLOAT *alpha, FLOAT *a, blasint *lda, blasint *ldb)
    int n = state.range(0);
    std::vector<double> A(n * n, 1.0);
    for (auto _ : state) {
        dimatcopy_k_ct(n, n, 1.0, A.data(), n);
    }
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr double percent_ninfs[] = { 0.0 };
    constexpr int sizes[] = { 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000, 11313 };
    // constexpr int sizes[] = { 8000 };
    // constexpr int sizes[] = { 10, 20, 30, 40 };
    for (const auto percent : percent_ninfs) {
        for (const auto size : sizes) {
            b->Args({ size, (int)(percent * size) });
        }
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}


BENCHMARK(BM_dlatr0)->Apply(apply_args);
BENCHMARK(BM_dlatr3)->Apply(apply_args);
BENCHMARK(BM_dlatrn)->Apply(apply_args);
BENCHMARK(BM_imatcopy)->Apply(apply_args);

BENCHMARK_MAIN();
