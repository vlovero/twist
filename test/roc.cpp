#include "argparse/argparse.hpp"
#include "cli/load.h"
#include "collocator.h"
#include "fmt/core.h"
#include "libloader.h"
#include "numpy_like.h"
#include "serialize.h"
#include "sparse_matrix.h"
#include <fstream>

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif

#ifdef OS_WIN
#include <windows.h>

unsigned long long get_total_system_memory()
{
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
}
#else
#include <unistd.h>

unsigned long long get_total_system_memory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

std::string formatWithCommas(long long number)
{
    std::string number_str = std::to_string(number);

    for (int i = number_str.length() - 3; i > 0; i -= 3)
        number_str.insert(i, ",");

    return number_str;
}

void check_size(const int N)
{
    const size_t nbytes = N * N * sizeof(double);
    const size_t max_bytes = get_total_system_memory();

    if (max_bytes < nbytes) {
        throw std::runtime_error(fmt::format("requesting {} bytes but system only has {}", formatWithCommas(nbytes), formatWithCommas(max_bytes)));
    }
}

extern "C" int32_t LAPACKE_dgeev(int matrix_layout, char jobvl, char jobvr, int32_t n, double *a, int32_t lda, double *wr, double *wi, double *vl, int32_t ldvl, double *vr, int32_t ldvr);
extern void kron(const sparse::COOMatrix &A, const sparse::COOMatrix &B, sparse::COOMatrix &C);

namespace Collocation
{
    double compute_error(ptrdiff_t N, const TWIST::Collocator &collocator, const size_t ncol)
    {
        double smallest, a, b;
        bool failed;
        size_t i;
        std::vector<double> alphar, alphai, beta;

        const size_t node = collocator.getNode();
        void *lib = collocator.getLibHandle();
        func_t func = (func_t)dlsym(lib, "func");
        func_t fjac = (func_t)dlsym(lib, "fjac");
        pjac_t pjac = (pjac_t)dlsym(lib, "pjac");
        const size_t np = collocator.getNParam();
        const double *p = collocator.p();
        auto diffusion = collocator.getDiffusion();
        const double L = collocator.unscaledSpatialPeriod();

        // construct equispaced mesh
        std::vector<double> t = linspace(0, 1, N);
        double *y = (double *)malloc(N * node * sizeof(double));
        TWIST::ContinuationBounds bounds{};
        collocator.denseOutput(t.data(), N, y);

        TWIST::Collocator collocator_N(ncol, func, fjac, pjac, node, N, t.data(), y, np, p, diffusion, 0.5 * L);
        auto tc = collocator_N.plottablePoints();
        collocator.denseOutput(tc.data(), tc.size(), (double *)(collocator_N.y()));
        collocator.denseOutput(tc.data(), tc.size(), (double *)(collocator_N.yprev()));
        {
            double *yp = (double *)(collocator_N.yp());
            for (i = 0; i < ((N - 1) * (ncol + 1) + 1); i++) {
                func(collocator_N.y() + i * node, collocator_N.p(), &yp[i * node]);
                for (size_t j = 0; j < node; j++) {
                    yp[i * node + j] *= L;
                }
            }
        }
        // {
        //     {
        //         H5::H5File res_file(".cache/solve_res.h5", H5F_ACC_TRUNC);
        //         H5::Group group = res_file.createGroup("data");
        //         {
        //             hsize_t dims[1] = { tc.size() };
        //             H5::DataSpace dataspace(1, dims);
        //             H5::DataSet dataset = group.createDataSet("x", H5::PredType::NATIVE_DOUBLE, dataspace);
        //             dataset.write(tc.data(), H5::PredType::NATIVE_DOUBLE);
        //         }
        //         {
        //             hsize_t dims[2] = { tc.size(), node };
        //             H5::DataSpace dataspace(2, dims);
        //             H5::DataSet dataset = group.createDataSet("y", H5::PredType::NATIVE_DOUBLE, dataspace);
        //             dataset.write(collocator_N.y(), H5::PredType::NATIVE_DOUBLE);
        //         }
        //     }
        //     int code = system(PYTHON_EXEC " _interim.py");
        //     if (code) {
        //         exit(0);
        //     }
        // }
        failed = collocator_N.solveTWave(1, bounds, 0.0, std::pow(0.0, 15), 100, false);
        if (failed) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        collocator_N.spectrum(alphar, alphai, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

        smallest = std::numeric_limits<double>::infinity();
        for (i = 0; i < alphar.size(); i++) {
            if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            a = alphar[i] / beta[i];
            b = alphai[i] / beta[i];
            smallest = std::min(smallest, std::hypot(a, b));
        }

        return smallest;
    }

    int run(const char *model)
    {
        const std::string path{ fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", model) };
        TWIST::Collocator collocator(path, 0);
        collocator.solveWithAdaptation(5, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 1000, true);
        std::ofstream file;
        file.open(fmt::format("test/_col_roc_data_{}.csv", model));

        double error;
        ptrdiff_t N;
        int i;

        for (i = 0; i < 5; i++) {
            N = 32 << i;
            file << fmt::format("{:4d}", N);
            fmt::print("{:4d}", N);
            for (const int ncol : { 2, 3, 4, 5 }) {
                error = compute_error(N + 1, collocator, ncol);
                fmt::print(", {:.8e}", error);
                file << fmt::format(", {:.8e}", error);
                fflush(stdout);
            }
            puts("");
            file << '\n';
        }
        file.close();

        return 0;
    }
} // namespace Collocation

namespace FiniteDifference
{
    constexpr std::array<double, 3> COEFF_D1_ORDER2 = { -1.0 / 2, 0, 1.0 / 2 };
    constexpr std::array<double, 3> COEFF_D2_ORDER2 = { 1, -2.0, 1 };

    constexpr std::array<double, 5> COEFF_D1_ORDER4 = { 1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12 };
    constexpr std::array<double, 5> COEFF_D2_ORDER4 = { -1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12 };

    constexpr std::array<double, 7> COEFF_D1_ORDER6 = { -1.0 / 60, 3.0 / 20, -3.0 / 4, 0, 3.0 / 4, -3.0 / 20, 1.0 / 60 };
    constexpr std::array<double, 7> COEFF_D2_ORDER6 = { 1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20, 1.0 / 90 };

    constexpr std::array<double, 9> COEFF_D1_ORDER8 = { 1.0 / 280, -4.0 / 105, 1.0 / 5, -4.0 / 5, 0, 4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280 };
    constexpr std::array<double, 9> COEFF_D2_ORDER8 = { -1.0 / 560, 8.0 / 315, -1.0 / 5, 8.0 / 5, -205.0 / 72, 8.0 / 5, -1.0 / 5, 8.0 / 315, -1.0 / 560 };

    constexpr std::array<double, 11> COEFF_D1_ORDER10 = { -1.0 / 1260, 5.0 / 504, -5.0 / 84, 5.0 / 21, -5.0 / 6, 0, 5.0 / 6, -5.0 / 21, 5.0 / 84, -5.0 / 504, 1.0 / 1260 };
    constexpr std::array<double, 11> COEFF_D2_ORDER10 = { 1.0 / 3150, -5.0 / 1008, 5.0 / 126, -5.0 / 21, 5.0 / 3, -5269.0 / 1800, 5.0 / 3, -5.0 / 21, 5.0 / 126, -5.0 / 1008, 1.0 / 3150 };

    // constexpr std::array<double, 13> COEFF_D1_ORDER12 = { 1.0 / 5544, -1.0 / 385, 1.0 / 56, -5.0 / 63, 15.0 / 56, -6.0 / 7, 0, 6.0 / 7, -15.0 / 56, 5.0 / 63, -1.0 / 56, 1.0 / 385, -1.0 / 5544 };
    // constexpr std::array<double, 13> COEFF_D2_ORDER12 = { -1.0 / 16632, 2.0 / 1925, -1.0 / 112, 10.0 / 189, -15.0 / 56, 12.0 / 7, -5369.0 / 1800, 12.0 / 7, -15.0 / 56, 10.0 / 189, -1.0 / 112, 2.0 / 1925, -1.0 / 16632 };

    // constexpr std::array<double, 15> COEFF_D1_ORDER14 = { -1.0 / 24024, 7.0 / 10296, -7.0 / 1320, 7.0 / 264, -7.0 / 72, 7.0 / 24, -7.0 / 8, 0, 7.0 / 8, -7.0 / 24, 7.0 / 72, -7.0 / 264, 7.0 / 1320, -7.0 / 10296, 1.0 / 24024 };
    // constexpr std::array<double, 15> COEFF_D2_ORDER14 = { 1.0 / 84084, -7.0 / 30888, 7.0 / 3300, -7.0 / 528, 7.0 / 108, -7.0 / 24, 7.0 / 4, -266681.0 / 88200, 7.0 / 4, -7.0 / 24, 7.0 / 108, -7.0 / 528, 7.0 / 3300, -7.0 / 30888, 1.0 / 84084 };

    // constexpr std::array<double, 17> COEFF_D1_ORDER16 = { 1.0 / 102960, -8.0 / 45045, 2.0 / 1287, -56.0 / 6435, 7.0 / 198, -56.0 / 495, 14.0 / 45, -8.0 / 9, 0, 8.0 / 9, -14.0 / 45, 56.0 / 495, -7.0 / 198, 56.0 / 6435, -2.0 / 1287, 8.0 / 45045, -1.0 / 102960 };
    // constexpr std::array<double, 17> COEFF_D2_ORDER16 = { -1.0 / 411840, 16.0 / 315315, -2.0 / 3861, 112.0 / 32175, -7.0 / 396, 112.0 / 1485, -14.0 / 45, 16.0 / 9, -1077749.0 / 352800, 16.0 / 9, -14.0 / 45, 112.0 / 1485, -7.0 / 396, 112.0 / 32175, -2.0 / 3861, 16.0 / 315315, -1.0 / 411840 };

    // constexpr std::array<double, 19> COEFF_D1_ORDER18 = { -1.0 / 437580, 9.0 / 194480, -9.0 / 20020, 2.0 / 715, -9.0 / 715, 63.0 / 1430, -7.0 / 55, 18.0 / 55, -9.0 / 10, 0, 9.0 / 10, -18.0 / 55, 7.0 / 55, -63.0 / 1430, 9.0 / 715, -2.0 / 715, 9.0 / 20020, -9.0 / 194480, 1.0 / 437580 };
    // constexpr std::array<double, 19> COEFF_D2_ORDER18 = { 1.0 / 1969110, -9.0 / 777920, 9.0 / 70070, -2.0 / 2145, 18.0 / 3575, -63.0 / 2860, 14.0 / 165, -18.0 / 55, 9.0 / 5, -9778141.0 / 3175200, 9.0 / 5, -18.0 / 55, 14.0 / 165, -63.0 / 2860, 18.0 / 3575, -2.0 / 2145, 9.0 / 70070, -9.0 / 777920, 1.0 / 1969110 };

    // constexpr std::array<double, 21> COEFF_D1_ORDER20 = { 1.0 / 1847560, -5.0 / 415701, 5.0 / 38896, -15.0 / 17017, 5.0 / 1144, -12.0 / 715, 15.0 / 286, -20.0 / 143, 15.0 / 44, -10.0 / 11, 0, 10.0 / 11, -15.0 / 44, 20.0 / 143, -15.0 / 286, 12.0 / 715, -5.0 / 1144, 15.0 / 17017, -5.0 / 38896, 5.0 / 415701, -1.0 / 1847560 };
    // constexpr std::array<double, 21> COEFF_D2_ORDER20 = { -1.0 / 9237800, 10.0 / 3741309, -5.0 / 155584, 30.0 / 119119, -5.0 / 3432, 24.0 / 3575, -15.0 / 572, 40.0 / 429, -15.0 / 44, 20.0 / 11, -1968329.0 / 635040, 20.0 / 11, -15.0 / 44, 40.0 / 429, -15.0 / 572, 24.0 / 3575, -5.0 / 3432, 30.0 / 119119, -5.0 / 155584, 10.0 / 3741309, -1.0 / 9237800 };

    constexpr std::array<std::pair<const double *, const double *>, 5> fdm_methods = { std::pair<const double *, const double *>{ COEFF_D1_ORDER2.data(), COEFF_D2_ORDER2.data() }, std::pair<const double *, const double *>{ COEFF_D1_ORDER4.data(), COEFF_D2_ORDER4.data() }, std::pair<const double *, const double *>{ COEFF_D1_ORDER6.data(), COEFF_D2_ORDER6.data() }, std::pair<const double *, const double *>{ COEFF_D1_ORDER8.data(), COEFF_D2_ORDER8.data() }, std::pair<const double *, const double *>{ COEFF_D1_ORDER10.data(), COEFF_D2_ORDER10.data() } };

    double compute_error(const ptrdiff_t N, const TWIST::Collocator &collocator, const ptrdiff_t order)
    {
        const double c = collocator.waveSpeed();
        const double L = collocator.spatialPeriod();
        auto diffusion = collocator.getDiffusion();
        const double *p = collocator.p();
        const double h = L / (N - 1);
        const double fac1 = 1.0 / h;
        const double fac2 = 1.0 / (h * h);
        auto [coeffs_d1, coeffs_d2] = fdm_methods[(order / 2) - 1];
        const ptrdiff_t node = collocator.getNode() - diffusion.size();
        void *lib = collocator.getLibHandle();
        func_t fjac_ode = (func_t)dlsym(lib, "fjac_ode");

        std::vector<double> x{ linspace(0, 1, N) };
        double ywithv[collocator.getNode()];
        double ynov[node];
        double J[node * node];

        const ptrdiff_t ncoeff = order + 1;
        const ptrdiff_t start = -(ncoeff >> 1);
        const ptrdiff_t ldA = node * N;

        check_size(ldA);
        double *A = (double *)calloc(ldA * ldA, sizeof(double));
        double *wr = (double *)malloc(ldA * sizeof(double));
        double *wi = (double *)malloc(ldA * sizeof(double));

        ptrdiff_t i, j, k;
        sparse::COOMatrix D1, D2, L1, L2, D, C;
        int64_t *irow1, *irow2;
        int64_t *icol1, *icol2;
        double *data1, *data2;
        double smallest;

        L1.setNNZ(ncoeff * N);
        L1.reshape(N, N);
        L2.setNNZ(ncoeff * N);
        L2.reshape(N, N);
        C.setNNZ(node);
        C.reshape(node, node);
        D.setNNZ(diffusion.size());
        D.reshape(node, node);
        irow1 = L1.irow;
        icol1 = L1.icol;
        data1 = L1.data;
        irow2 = L2.irow;
        icol2 = L2.icol;
        data2 = L2.data;

        // create L1 and L2
        for (i = 0; i < N; i++) {
            for (j = start, k = 0; k < ncoeff; j++, k++) {
                *irow1++ = i;
                *irow2++ = i;
                *icol1++ = (N + ((i + j) % N)) % N;
                *icol2++ = (N + ((i + j) % N)) % N;
                *data1++ = coeffs_d1[k] * fac1;
                *data2++ = coeffs_d2[k] * fac2;
            }
        }
        // create C
        for (i = 0; i < node; i++) {
            C.irow[i] = i;
            C.icol[i] = i;
            C.data[i] = -c;
        }
        // create D
        i = 0;
        for (const auto &[index, coeff] : diffusion) {
            D.irow[i] = index;
            D.icol[i] = index;
            D.data[i] = coeff;
            i++;
        }

        // kron with diffusion matrix to get D2
        kron(L1, C, D1);
        kron(L2, D, D2);
        for (i = 0; i < N; i++) {
            // sample on grid
            collocator.denseOutput(&x[i], 1, ywithv);

            // delete velocities
            for (j = 0, k = 0; j < (ptrdiff_t)(node + diffusion.size()); j++, k++) {
                ynov[k] = ywithv[j];
                for (const auto &[loc, _] : diffusion) {
                    if (loc == j) {
                        j++;
                        break;
                    }
                }
            }
            // evaluate ode jacobians
            fjac_ode(ynov, p, J);
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    A[(i * node + j) * ldA + (i * node + k)] = J[j * node + k];
                }
            }
        }
        // add on generated diff mats to result
        for (i = 0; i < D1.nnz; i++) {
            j = D1.irow[i];
            k = D1.icol[i];
            A[j * ldA + k] += D1.data[i];
        }
        for (i = 0; i < D2.nnz; i++) {
            j = D2.irow[i];
            k = D2.icol[i];
            A[j * ldA + k] += D2.data[i];
        }

        // compute eigenvalues
        LAPACKE_dgeev(102, 'N', 'N', ldA, A, ldA, wr, wi, NULL, ldA, NULL, ldA);
        // find error (smallest eigenvalue)
        smallest = std::hypot(wr[0], wi[0]);
        for (i = 1; i < ldA; i++) {
            smallest = std::min(smallest, std::hypot(wr[i], wi[i]));
        }
        free(A);
        free(wr);
        free(wi);

        return smallest;
    }

    int run(const char *model)
    {
        const std::string path{ fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", model) };
        TWIST::Collocator collocator(path, 0);
        collocator.solveWithAdaptation(3, 1e-16, 2, -1lu, 0.0, std::pow(0.5, 15), 1000, true);
        std::ofstream file;
        file.open(fmt::format("test/fdm_roc_data_{}.csv", model));

        double error;
        ptrdiff_t N;
        int i;

        for (i = 0; i < 7; i++) {
            N = 32 << i;
            file << fmt::format("{:4d}", N);
            fmt::print("{:4d}", N);
            for (const int order : { 2, 4, 6, 8, 10 }) {
                error = compute_error(N, collocator, order);
                fmt::print(", {:.8e}", error);
                file << fmt::format(", {:.8e}", error);
                fflush(stdout);
            }
            puts("");
            file << '\n';
        }
        file.close();

        return 0;
    }

    void run_timer(const char *model, const int N, const int order)
    {
        double error;
        if (N < 2) {
            puts("need at least two points");
            return;
        }
        const std::string path{ fmt::format(".cache/continuation_data/{}-default-sps-16.h5", model) };
        TWIST::Collocator collocator(path, 0);
        collocator.solveWithAdaptation(3, 1e-16, 2, -1lu, 0.0, std::pow(0.5, 15), 1000, true);

        TWIST::time_code(fmt::format("FDM with {} grid points", N), [&]() { error = compute_error(N, collocator, order); });
        fmt::println("error = {:.8e}", error);
    }
} // namespace FiniteDifference

namespace Fourier
{
    double *gen_diffmat(const int nu, const ptrdiff_t n, const double L)
    {
        ptrdiff_t i, j, k;
        double *col = (double *)malloc(n * sizeof(double));
        double *A = (double *)malloc(n * n * sizeof(double));
        const double h = (2 * M_PI) / n;
        if (nu == 1) {
            col[0] = 0;
            for (k = 1; k < n; k++) {
                col[k] = 0.5 * std::pow(-1, k) / std::tan(0.5 * k * h);
                col[k] *= (2 * M_PI) / L;
            }
        }
        else {
            col[0] = -(std::pow(M_PI / h, 2) / 3 + 1.0 / 6);
            col[0] *= ((2 * M_PI) / L) * ((2 * M_PI) / L);
            for (k = 1; k < n; k++) {
                col[k] = -0.5 * std::pow(-1, k) / (std::sin(0.5 * k * h) * std::sin(0.5 * k * h));
                col[k] *= ((2 * M_PI) / L) * ((2 * M_PI) / L);
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                k = (((i - j) % n) + n) % n;
                A[i * n + j] = col[k];
            }
        }
        free(col);
        return A;
    }

    double compute_error(const ptrdiff_t N, const TWIST::Collocator &collocator)
    {
        const double c = collocator.waveSpeed();
        const double L = collocator.spatialPeriod();
        auto diffusion = collocator.getDiffusion();
        const double *p = collocator.p();

        const ptrdiff_t node = collocator.getNode() - diffusion.size();
        void *lib = collocator.getLibHandle();
        func_t fjac_ode = (func_t)dlsym(lib, "fjac_ode");

        std::vector<double> x{ linspace(0, 1, N + 1) };
        double ywithv[collocator.getNode()];
        double ynov[node];
        double J[node * node];

        const ptrdiff_t ldA = node * N;

        check_size(ldA);
        double *A = (double *)calloc(ldA * ldA, sizeof(double));
        double *wr = (double *)malloc(ldA * sizeof(double));
        double *wi = (double *)malloc(ldA * sizeof(double));

        ptrdiff_t i, j, k;
        double smallest;
        double *D1 = gen_diffmat(1, N, L);
        double *D2 = gen_diffmat(2, N, L);

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (const auto &[k, coeff] : diffusion) {
                    A[(i * node + k) * ldA + (j * node + k)] = coeff * D2[i * N + j];
                }
                for (k = 0; k < node; k++) {
                    A[(i * node + k) * ldA + (j * node + k)] -= c * D1[i * N + j];
                }
            }
        }

        for (i = 0; i < N; i++) {
            // sample on grid
            collocator.denseOutput(&x[i + 1], 1, ywithv);

            // delete velocities
            for (j = 0, k = 0; j < (ptrdiff_t)(node + diffusion.size()); j++, k++) {
                ynov[k] = ywithv[j];
                for (const auto &[loc, _] : diffusion) {
                    if (loc == j) {
                        j++;
                        break;
                    }
                }
            }
            // evaluate ode jacobians
            fjac_ode(ynov, p, J);
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    A[(i * node + j) * ldA + (i * node + k)] += J[j * node + k];
                }
            }
        }

        // compute eigenvalues
        LAPACKE_dgeev(102, 'N', 'N', ldA, A, ldA, wr, wi, NULL, ldA, NULL, ldA);
        if (0) {
            {
                H5::H5File file(".cache/solve_res.h5", H5F_ACC_TRUNC);
                H5::Group group = file.createGroup("data");
                serialize<double, 1>(group, "x", wr, { (hsize_t)ldA });
                serialize<double, 1>(group, "y", wi, { (hsize_t)ldA });
            }
            // group.close()z
            int code = system(PYTHON_EXEC " _interim.py");
            if (code) {
                exit(0);
            }
        }
        // find error (smallest eigenvalue)
        smallest = std::hypot(wr[0], wi[0]);
        for (i = 1; i < ldA; i++) {
            smallest = std::min(smallest, std::hypot(wr[i], wi[i]));
        }
        free(A);
        free(wr);
        free(wi);
        free(D1);
        free(D2);

        return smallest;
    }

    int run(const char *model)
    {
        const std::string path{ fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", model) };
        TWIST::Collocator collocator(path, 36);
        collocator.solveWithAdaptation(3, 1e-16, 2, -1lu, 0.0, std::pow(0.5, 15), 1000, true);
        // compute_error(250, collocator);
        // exit(0);
        std::ofstream file;
        file.open(fmt::format("test/for_roc_data_{}.csv", model));

        double error;
        ptrdiff_t N;
        int i;

        for (i = 0; i < 7; i++) {
            N = 32 << i;
            error = compute_error(N, collocator);
            file << fmt::format("{:4d}, {:.8e}\n", N, error);
            fmt::println("{:4d}, {:.8e}", N, error);
        }
        file.close();

        return 0;
    }
} // namespace Fourier

namespace Adaptive
{
    int run(const char *model)
    {
        const std::string path{ fmt::format(".cache/continuation_data/{}-default-sps-latest.h5", model) };
        H5::H5File h5data(path, H5F_ACC_RDONLY);
        void *lib = nullptr;
        size_t ncol_wanted = 10;
        TWIST::Collocator collocator{ load_collocator_from_h5data_and_index(h5data, 0, ncol_wanted, &lib) };
        std::ofstream file;
        std::vector<double> alphar, alphai, beta;
        double smallest, a, b;
        size_t i, N;
        bool failed;

        file.open(fmt::format("test/data/ada_roc_data_{}.csv", model));
        collocator.setLibHandle(lib);

        for (const auto geps : { 1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-4, 1e-3 }) {
            failed = collocator.solveWithAdaptation(100, geps, 2, -1lu, 0.0, pow(0.5, 15), 1000, false, true);
            if (failed) {
                smallest = std::numeric_limits<double>::quiet_NaN();
            }
            else {
                N = ((collocator.getNNodes() - 1) * (ncol_wanted + 1) + 1) * collocator.getNode();
                collocator.spectrum(alphar, alphai, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
                smallest = std::numeric_limits<double>::infinity();
                for (i = 0; i < alphar.size(); i++) {
                    if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                        continue;
                    }
                    a = alphar[i] / beta[i];
                    b = alphai[i] / beta[i];
                    smallest = std::min(smallest, std::hypot(a, b));
                }
            }
            fmt::println("{:.0e},{:.8e},{:3d}", geps, smallest, collocator.getNNodes());
            file << fmt::format("{:.0e},{:.8e},{:3d}\n", geps, smallest, collocator.getNNodes());
        }

        file.close();
        return 0;
    }
} // namespace Adaptive

enum Mode
{
    ROC = 0,
    TIME
};

enum Method
{
    collocation = 0,
    fdm,
    fourier,
    adaptive
};

struct TestTWIST : public argparse::Args
{
    Mode &mode = arg("mode", "program mode");
    Method &method = arg("method", "program method");
    std::string &model = arg("model", "which param");
    int &order = kwarg("order", "order of FDM (time mode only)").set_default(2);
    int &N = kwarg("n,num-grid", "number of grid points (timer mode only)").set_default(600);

    int run() override
    {
        if (mode == Mode::ROC) {
            switch (method) {
            case collocation:
                Collocation::run(model.c_str());
                break;
            case fdm:
                FiniteDifference::run(model.c_str());
                break;
            case fourier:
                Fourier::run(model.c_str());
                break;
            case adaptive:
                Adaptive::run(model.c_str());
                break;
            }
            return 0;
        }

        switch (method) {
        case collocation:
            break;
        case fdm:
            FiniteDifference::run_timer(model.c_str(), N, order);
            break;
        case fourier:
            break;
        case adaptive:
            break;
        }

        return 0;
    }
};

int main(int argc, char **argv)
{
    auto args = argparse::parse<TestTWIST>(argc, argv, true);
    return args.run();
}