#include "benchmark/benchmark.h"
#include "linalg.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

extern "C" void dggev3_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
extern "C" void dggev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
extern void dggev5(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info);

void generate_random_matrix(std::vector<double> &A, const int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(A.begin(), A.end(), [&]() { return dist(engine); });
}

static void BM_dggev1(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
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
    dggev_("N", "N", &n, NULL, &n, NULL, &n, NULL, NULL, NULL, NULL, &n, NULL, &n, &work_opt, &lwork, &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(B.data(), 0, r * ldB * sizeof(double));
        dggev_("N", "N", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), NULL, &n, NULL, &n, work, &lwork, &info);
    }
    delete[] work;
}

static void BM_dggev3(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
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
        memset(B.data(), 0, r * ldB * sizeof(double));
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
        memset(B.data(), 0, r * ldB * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev4("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

static void BM_dggev5(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
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
        memset(B.data(), 0, r * ldB * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev5("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

static void BM_dggev3_full(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> VL(n * n);
    std::vector<double> VR(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev3_("V", "V", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), &n, VR.data(), &n, &work_opt, &lwork, &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(B.data(), 0, r * ldB * sizeof(double));
        dggev3_("V", "V", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), &n, VR.data(), &n, work, &lwork, &info);
    }
    delete[] work;
}

static void BM_dggev4_full(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    std::vector<int> jpvt(n);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> VL(n * n);
    std::vector<double> VR(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev4("V", "V", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, &work_opt, -1, jpvt.data(), &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(B.data(), 0, r * ldB * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev4("V", "V", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

static void BM_dggev5_full(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    std::vector<int> jpvt(n);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> VL(n * n);
    std::vector<double> VR(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev5("V", "V", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, &work_opt, -1, jpvt.data(), &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(B.data(), 0, r * ldB * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev5("V", "V", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}


static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr double percent_ninfs[] = { 0.0 };
    constexpr int sizes[] = { 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000 };
    // constexpr int sizes[] = { 500 };
    // constexpr int sizes[] = { 10, 20, 30, 40 };
    for (const auto percent : percent_ninfs) {
        for (const auto size : sizes) {
            b->Args({ size, (int)(percent * size) });
        }
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}


BENCHMARK(BM_dggev3)->Apply(apply_args);
BENCHMARK(BM_dggev4)->Apply(apply_args);
BENCHMARK(BM_dggev5)->Apply(apply_args);

BENCHMARK(BM_dggev3_full)->Apply(apply_args);
BENCHMARK(BM_dggev4_full)->Apply(apply_args);
BENCHMARK(BM_dggev5_full)->Apply(apply_args);

BENCHMARK_MAIN();
