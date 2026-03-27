#include "argparse/argparse.hpp"
#include "collocator.h"
#include "colmat/tcolmat.h"
#include "libloader.h"
#include <random>
#include <valarray>

void test_make_V(double *&V, const sparse::RealCSCMatrix &A, const ptrdiff_t nparam)
{
    int64_t i, j, k, l, stop;
    // allocate V
    V = (double *)realloc(V, A.nrows * sizeof(double));
    // zero out V
    memset(V, 0, A.nrows * sizeof(double));
    // extract last nparam rows from CSC matrix
    for (i = 0; i < A.ncols; i++) {
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

void report_dense(sparse::RealCSCMatrix &A, const std::valarray<double> &b, std::valarray<double> &x, std::valarray<double> &r)
{

    double *Ad = A.dense(true);
    std::valarray<int> ipiv(2 * b.size());
    DEV::internal::FinalBlock<double> M;
    int info;

    x = b;
    dgesv(b.size(), 1, Ad, b.size(), &ipiv[0], &x[0], b.size(), &info);
    r = b;
    A.gemv(-1, &r[0], 1, &x[0]);
    fmt::println("dense forward error   : {:.8e}", std::abs(x - 1).max());
    fmt::println("dense backward error  : {:.8e}\n", std::abs(r).max());
    free(Ad);
}

void report_umfpack(sparse::RealCSCMatrix &A, const std::valarray<double> &b, std::valarray<double> &x, std::valarray<double> &r)
{

    A.Control[UMFPACK_SYM_PIVOT_TOLERANCE] = 1;
    A.Control[UMFPACK_PIVOT_TOLERANCE] = 1;
    if (A.Symbolic) {
        exit(1);
    }
    A.solve(&b[0], &x[0]);

    r = b;
    A.gemv(-1, &r[0], 1, &x[0]);
    fmt::println("UMFPack forward error : {:.8e}", std::abs(x - 1).max());
    fmt::println("UMFPack backward error: {:.8e}\n", std::abs(r).max());
}

void report_mixed(sparse::CollocationMatrix &A, const std::valarray<double> &b, std::valarray<double> &x, std::valarray<double> &r)
{
    A.solve(&b[0], &x[0]);

    r = b;
    A.m_mat->gemv(-1, &r[0], 1, &x[0]);
    fmt::println("mixed forward error   : {:.8e}", std::abs(x - 1).max());
    fmt::println("mixed backward error  : {:.8e}\n", std::abs(r).max());
}

void report_tree(DEV::Matrix &A, const sparse::RealCSCMatrix &M, const std::valarray<double> &b, std::valarray<double> &x, std::valarray<double> &r)
{
    A.solve(&b[0], &x[0]);

    r = b;
    M.gemv(1, &r[0], -1, &x[0]);
    A.solve(&r[0], &r[0]);
    x += r;
    r = b;
    M.gemv(1, &r[0], -1, &x[0]);

    fmt::println("tree forward error    : {:.8e}", std::abs(x - 1).max());
    fmt::println("tree backward error   : {:.8e}\n", std::abs(r).max());
}

static ptrdiff_t __nrand = 0;
static std::default_random_engine engine(0);
static std::uniform_real_distribution<double> dist;

static void func_random(const double *z, const double *, double *out)
{
    for (ptrdiff_t i = 0; i < __nrand; i++) {
        uint64_t iz = *(uint64_t *)(z + i);
        engine.seed(iz);
        out[i] = dist(engine);
    }
}

static void fjac_random(const double *z, const double *, double *out)
{
    for (ptrdiff_t i = 0; i < __nrand; i++) {
        uint64_t iz = *(uint64_t *)(z + i);
        for (ptrdiff_t j = 0; j < __nrand; j++) {
            engine.seed(iz + j);
            out[i * __nrand + j] = dist(engine);
        }
    }
}

static void pjac_random(const double *z, const double *p, const int *, const int nparam, double *out)
{
    for (ptrdiff_t i = 0; i < __nrand; i++) {
        uint64_t iz = *(uint64_t *)(z + i);
        for (ptrdiff_t j = 0; j < nparam; j++) {
            uint64_t ip = *(uint64_t *)(p + j);
            engine.seed(iz & ip);
            out[i * nparam + j] = dist(engine);
        }
    }
}

struct TestColMatsArgs : public argparse::Args
{
    std::optional<std::string> &name = kwarg("n,name", "name of model to test. If not specified, random matrices will be generated");
    ptrdiff_t &nrand = kwarg("s,size", "size of system for random function").set_default(3);

    int run() override
    {
        std::string path;
        TWIST::Collocator collocator;
        sparse::RealCSCMatrix A;
        sparse::CollocationMatrix As;
        DEV::Matrix mat;
        DEV::BoundaryConditions bc;
        func_t fjac;
        pjac_t pjac;
        ptrdiff_t ngrid, ncol, node, nparam;
        double L;
        int pmask[2] = { 0 };
        double *h = nullptr, *y = nullptr, *p = nullptr, *V = nullptr;

        if (name && (name != "random")) {
            path = fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", name.value());
            TWIST::Collocator tmp(path, 0);
            void *handle = tmp.getLibHandle();
            collocator = tmp;
            collocator.setLibHandle(handle);
            tmp.setLibHandle(nullptr);
            assert(handle);
            fjac = (func_t)dlsym(handle, "fjac");
            pjac = (pjac_t)dlsym(handle, "pjac");
            assert(fjac);
            assert(pjac);
        }
        else {
            __nrand = nrand;
            std::valarray<double> t(30), y(30 * __nrand), p(1);

            // generate random data
            std::iota(&t[0], &t[t.size()], 0.0);
            t /= t.size() - 1;
            std::generate(&y[0], &y[y.size()], []() { return dist(engine); });
            p[0] = 1.0;
            diffusion_t d = { { 0, 1.0 } };

            // init collocator
            collocator = TWIST::Collocator(10, func_random, fjac_random, pjac_random, nrand, t.size(), &t[0], &y[0], 1, &p[0], d, 0.5);
            fjac = fjac_random;
            pjac = pjac_random;
        }
        nparam = 1;
        node = collocator.getNode();
        ncol = collocator.getNStages();
        ngrid = collocator.getNNodes();
        h = collocator.h();
        y = collocator.y();
        p = collocator.p();
        bc.makePeriodic(node, nparam);
        L = collocator.spatialPeriod();

        A = collocator.getBaseJacobian();
        As.update(&A, ngrid, ncol, node, nparam);
        test_make_V(V, A, nparam);

        TWIST::time_code("building mat (with allocations)", [&]() { mat.update(fjac, pjac, ngrid, node, ncol, nparam, L, h, y, p, pmask, V, &bc); });
        TWIST::time_code("building mat (no allocations)", [&]() { mat.update(fjac, pjac, ngrid, node, ncol, nparam, L, h, y, p, pmask, V, &bc); });
        puts("");

        // create test system
        std::valarray<double> b(A.nrows), x(A.nrows), r(A.nrows);
        x = 1.0;
        A.gemv(0.0, &b[0], 1.0, &x[0]);

        report_dense(A, b, x, r);
        report_umfpack(A, b, x, r);
        report_mixed(As, b, x, r);
        report_tree(mat, A, b, x, r);

        return 0;
    }
};

int main(int argc, char **argv)
{
    auto args = argparse::parse<TestColMatsArgs>(argc, argv, true);
    return args.run();
}
