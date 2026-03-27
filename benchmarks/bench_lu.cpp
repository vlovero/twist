#include "benchmark/benchmark.h"
#include "colmat/tcolmat.h"
#include <algorithm>
#include <random>

template <typename data_t, const ptrdiff_t block_size = 32>
void getrpp(const ptrdiff_t n, const ptrdiff_t m, RP(data_t) A, const ptrdiff_t ldA, int *ipiv, int *jpiv)
{
    auto aargmax = [](const data_t *x, const ptrdiff_t nx) {
        typename type_of_data<data_t>::type xmax, tmp;
        ptrdiff_t i, imax;

        xmax = 0.0;
        imax = 0;
        for (i = 0; i < nx; i++) {
            tmp = std::abs(x[i]);
            if (xmax < tmp) {
                xmax = tmp;
                imax = i;
            }
        }
        return imax;
    };

    ptrdiff_t minnm, i, j, l, jmax, imax, ncols;
    data_t aii;

    minnm = std::min(n, m);

    for (l = 0; l < minnm; l += block_size) {
        ncols = std::min(block_size, m - l);

        for (i = l; i < l + ncols; i++) {
            imax = i + aargmax(&A[i * (ldA + 1)], n - i);
            jmax = i;

            assert(imax < n);
            assert(jmax < m);
            ipiv[i] = imax;
            jpiv[i] = jmax;

            if (imax != i) {
                // swap locally and do other swaps later
                for (j = l; j < l + ncols; j++) {
                    std::swap(A[j * ldA + i], A[j * ldA + imax]);
                }
            }
            if (jmax != i) {
                for (j = 0; j < n; j++) {
                    std::swap(A[i * ldA + j], A[jmax * ldA + j]);
                }
            }

            aii = A[i * (ldA + 1)];
            if (aii == 0.0) {
                continue;
            }

            for (j = i + 1; j < n; j++) {
                A[i * ldA + j] /= aii;
            }
            if (i < (l + ncols - 1)) {
                // DEV::linalg::ger<data_t>(n - i - 1, l + ncols - i - 1, -1.0, &A[i * ldA + i + 1], 1, &A[(i + 1) * ldA + i], ldA, &A[(i + 1) * (ldA + 1)], ldA);
            }
        }
        // swap rows before window
        DEV::linalg::laswpr(l, A, ldA, l, l + ncols, ipiv, -1);

        if ((l + ncols) == m) {
            continue;
        }

        // swap rows after window
        DEV::linalg::laswpr(m - l - ncols, &A[(l + ncols) * ldA], ldA, l, l + ncols, ipiv, -1);

        // Lower triangular and Schur stuff
        DEV::linalg::trsm<data_t>("L", "L", "N", "U", ncols, m - ncols - l, 1.0, &A[l * (ldA + 1)], ldA, &A[(l + ncols) * ldA + l], ldA);
        DEV::linalg::gemm<data_t>("N", "N", n - l - ncols, m - l - ncols, ncols, -1.0, &A[l * ldA + l + ncols], ldA, &A[(l + ncols) * ldA + l], ldA, 1.0, &A[(l + ncols) * (ldA + 1)], ldA);
    }
}


void generate_random_matrix(std::vector<double> &A, const int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(A.begin(), A.end(), [&]() { return dist(engine); });
}

static void BM_PP(benchmark::State &state)
{
    int n = state.range(0);
    int info;
    std::vector<int> ipiv(n);
    std::vector<int> jpiv(n);
    std::vector<double> A_start(n * n);
    std::vector<double> A(n * n);
    generate_random_matrix(A_start, 0);

    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        dgetrf(n, n, A.data(), n, ipiv.data(), &info);
    }
}

static void BM_WRP(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<int> ipiv(n);
    std::vector<int> jpiv(n);
    std::vector<double> A_start(n * n);
    std::vector<double> A(n * n);
    generate_random_matrix(A_start, 0);

    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        DEV::linalg::getrp3(n, n, A.data(), n, ipiv.data(), jpiv.data());
    }
}

static void BM_CP(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<int> ipiv(n);
    std::vector<int> jpiv(n);
    std::vector<double> A_start(n * n);
    std::vector<double> A(n * n);
    generate_random_matrix(A_start, 0);

    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        DEV::linalg::geluf("C", n, n, A.data(), n, ipiv.data(), jpiv.data());
    }
}

static void BM_PP2(benchmark::State &state)
{
    int n = state.range(0);
    std::vector<int> ipiv(n);
    std::vector<int> jpiv(n);
    std::vector<double> A_start(n * n);
    std::vector<double> A(n * n);
    generate_random_matrix(A_start, 0);

    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        getrpp(n, n, A.data(), n, ipiv.data(), jpiv.data());
    }
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr int sizes[] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
    for (const auto size : sizes) {
        b->Args({ size });
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_PP)->Apply(apply_args);
BENCHMARK(BM_PP2)->Apply(apply_args);
BENCHMARK(BM_CP)->Apply(apply_args);
BENCHMARK(BM_WRP)->Apply(apply_args);

BENCHMARK_MAIN();
