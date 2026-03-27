#include "argparse/argparse.hpp"
#include "cli/load.h"
#include "krylov/krylov_schur.h"
#include "linalg.h"


extern void dggev5(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info);


struct TestDGGEV4Args : public argparse::Args
{
    enum class ALGS
    {
        dggev3 = 0,
        dggev4,
        dggev5
    };

    ptrdiff_t &n = arg("n", "size of pencil");
    int &seed = kwarg("s,seed", "seed for normal distribution").set_default(0);
    ALGS &alg = kwarg("a,algorithm", "which GGEV algorithm to use").set_default(ALGS::dggev4);

    int run() override
    {
        double *A, *B, *T, *S, *work, *Q, *Z, *tau, *alphar, *alphai, *beta, *R, *alphar_ans, *beta_ans;
        double dummy[1];
        int info[1], lwork, *jpvt;
        ptrdiff_t i, j;

        std::string name = std::string{ magic_enum::enum_name(alg) };

        fmt::println("Using {}", name);

        A = (double *)malloc(n * n * sizeof(double));
        B = (double *)malloc(n * n * sizeof(double));
        T = (double *)malloc(n * n * sizeof(double));
        S = (double *)malloc(n * n * sizeof(double));
        Q = (double *)malloc(n * n * sizeof(double));
        Z = (double *)malloc(n * n * sizeof(double));
        R = (double *)malloc(n * n * sizeof(double));
        tau = (double *)malloc(n * sizeof(double));
        alphar = (double *)malloc(n * sizeof(double));
        alphai = (double *)malloc(n * sizeof(double));
        beta = (double *)malloc(n * sizeof(double));
        alphar_ans = (double *)malloc(n * sizeof(double));
        beta_ans = (double *)malloc(n * sizeof(double));
        jpvt = (int *)malloc(n * sizeof(int));

        dgeqrf(n, n, Q, n, tau, dummy, -1, info);
        lwork = *dummy;
        dorgqr(n, n, n, Q, n, tau, dummy, -1, info);
        lwork = std::max<int>(lwork, *dummy);
        switch (alg) {
        case ALGS::dggev3:
            dggev3("V", "V", n, A, n, B, n, alphar, alphai, beta, Q, n, Z, n, dummy, -1, info);
            break;
        case ALGS::dggev4:
            dggev4("V", "V", n, A, n, B, n, alphar, alphai, beta, Q, n, Z, n, dummy, -1, jpvt, info);
            break;
        case ALGS::dggev5:
            dggev5("V", "V", n, A, n, B, n, alphar, alphai, beta, Q, n, Z, n, dummy, -1, jpvt, info);
            break;
        }
        lwork = std::max<int>(lwork, *dummy);
        work = (double *)malloc(lwork * sizeof(double));

        fmt::println("generating synthetic system");
        // generate test data Q and Z
        Krylov::better::generate_random_matrix(Q, n * n, seed + 0);
        Krylov::better::generate_random_matrix(Z, n * n, seed + 1);
        Krylov::better::generate_random_matrix(alphar_ans, n, seed + 2);
        Krylov::better::generate_random_matrix(beta_ans, n, seed + 3);
        dgeqrf(n, n, Q, n, tau, work, lwork, info);
        dorgqr(n, n, n, Q, n, tau, work, lwork, info);
        dgeqrf(n, n, Z, n, tau, work, lwork, info);
        dorgqr(n, n, n, Z, n, tau, work, lwork, info);

        // form A and B
        memcpy(R, Z, n * n * sizeof(double));

        // form A
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                Z[i * n + j] *= alphar_ans[j];
            }
        }
        dgemm("T", "N", n, n, n, 1.0, Q, n, Z, n, 0.0, T, n);
        memcpy(A, T, n * n * sizeof(double));

        // form B
        memcpy(Z, R, n * n * sizeof(double));
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                Z[i * n + j] *= beta_ans[j];
            }
        }
        dgemm("T", "N", n, n, n, 1.0, Q, n, Z, n, 0.0, S, n);
        memcpy(B, S, n * n * sizeof(double));
        fmt::println("finished generating synthetic system");

        // factor
        switch (alg) {
        case ALGS::dggev3:
            dggev3("V", "V", n, T, n, S, n, alphar, alphai, beta, Q, n, Z, n, work, lwork, info);
            break;
        case ALGS::dggev4:
            dggev4("V", "V", n, T, n, S, n, alphar, alphai, beta, Q, n, Z, n, work, lwork, jpvt, info);
            break;
        case ALGS::dggev5:
            dggev5("V", "V", n, T, n, S, n, alphar, alphai, beta, Q, n, Z, n, work, lwork, jpvt, info);
            break;
        }

        // compute backward errors eigenvalues (right)
        // R = AV - BVK
        // R = -BVK
        dgemm("N", "N", n, n, n, 1.0, B, n, Z, n, 0.0, R, n);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                R[i * n + j] *= -alphar[i] / beta[i];
            }
        }
        // R += AV
        dgemm("N", "N", n, n, n, 1.0, A, n, Z, n, 1.0, R, n);
        fmt::println("backward error (right) : {:.8e}", norminf(R, n * n));

        // compute backward errors eigenvalues (left)
        // R = A^TW - B^TWK
        // R = -B^TVK
        dgemm("T", "N", n, n, n, 1.0, B, n, Q, n, 0.0, R, n);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                R[i * n + j] *= -alphar[i] / beta[i];
            }
        }
        // R += A^TW
        dgemm("T", "N", n, n, n, 1.0, A, n, Q, n, 1.0, R, n);
        fmt::println("backward error (left)  : {:.8e}", norminf(R, n * n));

        // compute forward error for eigenvalues
        for (i = 0; i < n; i++) {
            alphar[i] /= beta[i];
            alphar_ans[i] /= beta_ans[i];
        }
        std::sort(alphar, alphar + n);
        std::sort(alphar_ans, alphar_ans + n);
        for (i = 0; i < n; i++) {
            alphar_ans[i] = std::abs(alphar_ans[i] - alphar[i]) / std::abs(alphar_ans[i]);
        }
        fmt::println("forward error (eigenvalues) : {:.8e}", norminf(alphar_ans, n));

        free(A);
        free(B);
        free(T);
        free(S);
        free(Q);
        free(Z);
        free(R);
        free(tau);
        free(alphar);
        free(alphai);
        free(beta);
        free(alphar_ans);
        free(beta_ans);
        free(jpvt);
        free(work);

        return 0;
    }
};

int main(int argc, char **argv)
{
    auto args = argparse::parse<TestDGGEV4Args>(argc, argv, true);
    return args.run();
}