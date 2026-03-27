#include "benchmark/benchmark.h"
#include "linalg.h"
#include "collocator.h"
#include <algorithm>


static void BM_umfpack(benchmark::State &state)
{
    TWIST::Collocator collocator(".cache/continuation_data/fhn-default-sps-latest.h5", 0);
    sparse::CollocationMatrix mat;
    bool failed = collocator.solveWithAdaptation(2, 1e-12, state.range(0), state.range(0), 0.0, pow(0.5, 15), 100, false);
    if (failed) {
        throw std::runtime_error(fmt::format("failed to converge with {} points", state.range(0)));
    }
    auto A = collocator.getBaseJacobian();
    mat.update(&A, collocator.getNNodes(), collocator.getNStages(), collocator.getNode(), 1);

    for (auto _ : state) {
        mat.update(&A, collocator.getNNodes(), collocator.getNStages(), collocator.getNode(), 1);
        mat.factor();
    }

    state.SetComplexityN(state.range(0));
}


static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr int sizes[] = { 8, 16, 31, 63, 127, 255, 511, 1023 };
    for (const auto size : sizes) {
        b->Args({ size });
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMicrosecond);
    b->Complexity(benchmark::oAuto);
}

BENCHMARK(BM_umfpack)->Apply(apply_args);

BENCHMARK_MAIN();
