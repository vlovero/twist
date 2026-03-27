#include "benchmark/benchmark.h"
#include "linalg.h"
#include <vector>
#include <random>
#include <algorithm>
#include "fmt/core.h"
#include "omp.h"

extern "C" void dggev3_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
extern "C" void dggev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);

void generate_random_matrix(std::vector<double> &A, const int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(A.begin(), A.end(), [&]() { return dist(engine); });
}

static void BM_dggev3(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    omp_set_num_threads(r);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev3_("N", "N", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), NULL, &n, NULL, &n, &work_opt, &lwork, &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        dggev3_("N", "N", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), NULL, &n, NULL, &n, work, &lwork, &info);
    }
    delete[] work;
}

static void BM_dggev4(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    omp_set_num_threads(r);
    std::vector<int> jpvt(n);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev4("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, &work_opt, -1, jpvt.data(), &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev4("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

extern void dggev5(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info);

static void BM_dggev5(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    std::string nthreads = fmt::format("{}", r);
    setenv("OPENBLAS_NUM_THREADS", nthreads.c_str(), 1);
    std::vector<int> jpvt(n);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev5("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, &work_opt, -1, jpvt.data(), &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev5("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr int num_threads[] = { 1, 2, 4, 6, 8, 10, 12, 16, 20, 24 };
    constexpr int sizes[] = { 8000 };
    for (const auto num : num_threads) {
        for (const auto size : sizes) {
            b->Args({ size, num });
        }
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}


BENCHMARK(BM_dggev3)->Apply(apply_args);
BENCHMARK(BM_dggev4)->Apply(apply_args);
BENCHMARK(BM_dggev5)->Apply(apply_args);


BENCHMARK_MAIN();
