#include "experimental/contour.h"
#include "bif_points/hopf_point.h"
#include "collocator.h"
#include "colmat/tcolmat.h"
#include "shared.h"
#include <cmath>
#include <complex>
#include <cstddef>
#include <omp.h>
#include <stack>

struct GQOp_t
{
    using complex_t = std::complex<double>;

    // these get updated every time struct called
    DEV::ComplexMatrix A;
    DEV::ComplexMatrix B;
    DEV::ComplexBoundaryConditions bc;
    DEV::ComplexBoundaryConditions zbc;

    // contants needed to generate matrices
    TWIST::Collocator *collocator;
    const double *A_rk_c;
    const double *b_rk_c;
    const PieceWiseLinearContour &contour;

    // these get updated manually
    ptrdiff_t nrhs;
    complex_t *rhs;

    GQOp_t() = delete;

    GQOp_t(TWIST::Collocator *collocator, const PieceWiseLinearContour &contour) : collocator(collocator), A_rk_c(std::get<1>(GL_methods[collocator->getNStages()])), b_rk_c(std::get<2>(GL_methods[collocator->getNStages()])), contour(contour)
    {
    }

    void operator()(const double t, complex_t *out)
    {
        double M[collocator->getNode() * collocator->getNode()], D[collocator->getNode()];
        const auto &to_delete = collocator->indicesToDelete();
        const auto &diffusion = collocator->getDiffusion();

        Hopf::setup_M_and_D(collocator->getNode(), M, D, to_delete, diffusion, collocator->spatialPeriod(), collocator->waveSpeed());
        Hopf::generate_matrices(-contour.gamma(t), collocator, A, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);

        A.solve(rhs, out, nrhs);
    }
};


void gauss_eval_n(GQOp_t &F, const ptrdiff_t nz, RP(std::complex<double>) result, RP(std::complex<double>) tmp, const double a, const double b, const ptrdiff_t nquad)
{
    double nodes[nquad], weights[nquad];
    const double diff = b - a;
    std::complex<double> fac;

    ptrdiff_t i, j;

    // setup weights and nodes for this interval
    memcpy(weights, std::get<2>(GL_methods.at(nquad)), nquad * sizeof(double));
    memcpy(nodes, std::get<3>(GL_methods.at(nquad)), nquad * sizeof(double));
    for (i = 0; i < nquad; i++) {
        nodes[i] = a + diff * nodes[i];
        weights[i] *= diff;
    }
    // puts("");
    // fmt::println("weights : {: .3e}", fmt::join(weights, weights + nquad, " "));
    // fmt::println("nodes   : {: .3e}", fmt::join(nodes, nodes + nquad, " "));

    // init to zero
    memset(result, 0, nz * sizeof(result[0]));

    // weighted sum
    for (i = 0; i < nquad; i++) {
        F(nodes[i], tmp);
        fac = F.contour.gammaPrime(nodes[i]) * weights[i];
        for (j = 0; j < nz; j++) {
            result[j] += fac * tmp[j];
        }
    }
}


void quad(GQOp_t &F, const ptrdiff_t nz, RP(std::complex<double>) result, RP(std::complex<double>) work, const ptrdiff_t lwork, const double a0, const double b0, const double tol)
{
    constexpr int nquad = 9;
    using complex_t = std::complex<double>;
    using sub_t = std::tuple<double, double, std::valarray<complex_t>>;

    ptrdiff_t i;
    complex_t *tmp1, *tmp2;
    double m, error, val, subtol, subscale;

    std::stack<sub_t> stack;

    if (lwork < (7 * nz)) {
        throw std::runtime_error("lwork not enough (7 * nz)");
    }

    std::valarray<complex_t> I0(nz);

    tmp1 = work + nz;
    tmp2 = tmp1 + nz;

    gauss_eval_n(F, nz, &I0[0], tmp1, a0, b0, nquad);
    stack.emplace(a0, b0, I0);
    memset(result, 0, nz * sizeof(complex_t));

    while (!stack.empty()) {
        std::valarray<complex_t> I_am(nz), I_mb(nz);
        auto [a, b, I_ab] = stack.top();
        stack.pop();
        m = 0.5 * (a + b);
        // pool.detach_task([&]() { gauss_eval_n(F, nz, &I_am[0], tmp1, a, m, nquad); }, BS::pr::highest);
        // pool.detach_task([&]() { gauss_eval_n(F, nz, &I_mb[0], tmp2, m, b, nquad); }, BS::pr::highest);
        // pool.wait();
        gauss_eval_n(F, nz, &I_am[0], tmp1, a, m, nquad);
        gauss_eval_n(F, nz, &I_mb[0], tmp2, m, b, nquad);

        error = 0.0;
        for (i = 0; i < nz; i++) {
            val = std::abs((I_am[i] + I_mb[i]) - I_ab[i]);
            error = std::max(error, val);
        }
        error = std::sqrt(error);
        error /= (double)((1 << nquad) - 1);
        subscale = (b - a) / (b0 - a0);
        subtol = tol * subscale;
        fmt::println("[{:.8e}, {:.8e}] : {:.8e} {:.8e}", a, b, error, subtol);

        if ((error <= subtol) || (subscale < 0.01)) {
            for (i = 0; i < nz; i++) {
                result[i] += I_am[i] + I_mb[i];
            }
        }
        else {
            stack.emplace(m, b, std::move(I_mb));
            stack.emplace(a, m, std::move(I_am));
        }
    }
}

FEASTResult feast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol)
{
    using complex_t = std::complex<double>;

    GQOp_t op(&collocator, contour);

    const double *A_rk_c = std::get<1>(GL_methods[collocator.getNStages()]);
    const double *b_rk_c = std::get<2>(GL_methods[collocator.getNStages()]);

    double beta_tol;
    int info[1], count, nconverged, ninfinite;
    complex_t dummy[1];
    int *jpvt;
    double *rwork, *errors, *serrors;
    complex_t *all_work = NULL, *quad_work, *Q, *R, *X, *Y, *MA, *MB, *W, *tau, *alpha, *beta, *lapack_work;
    DEV::ComplexMatrix A, B;
    DEV::ComplexBoundaryConditions bc, zbc;
    ptrdiff_t n, nsub, i, j, lwork, lapack_lwork;
    double M[collocator.getNode() * collocator.getNode()], D[collocator.getNode()];
    FEASTResult result;
    const auto &to_delete = collocator.indicesToDelete();
    const auto &diffusion = collocator.getDiffusion();

    Hopf::setup_M_and_D(collocator.getNode(), M, D, to_delete, diffusion, collocator.spatialPeriod(), collocator.waveSpeed());
    Hopf::generate_matrices(0.0, &collocator, A, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);
    n = A.size();

    nsub = std::min(n, std::max<ptrdiff_t>({ 2 * nstop + 1, 20, (ptrdiff_t)std::ceil(2 * std::sqrt(n)) }));

    lwork = (4 + 7) * n * nsub + 3 * nsub * nsub + 5 * nsub;
    zgeqrf(n, nsub, NULL, n, NULL, dummy, -1, info);
    lapack_lwork = dummy->real();
    zggev4("N", "V", nsub, dummy, nsub, dummy, nsub, dummy, dummy, NULL, nsub, NULL, nsub, dummy, -1, NULL, NULL, info);
    lapack_lwork = std::max<ptrdiff_t>(lapack_lwork, dummy->real());
    lwork += lapack_lwork;

    all_work = (complex_t *)malloc(lwork * sizeof(complex_t));
    rwork = (double *)malloc(8 * nsub * sizeof(double));
    jpvt = (int *)malloc(nsub * sizeof(int));

    quad_work = all_work;
    Q = quad_work + 7 * n * nsub;
    R = Q + n * nsub;
    X = R + n * nsub;
    Y = X + n * nsub;
    MA = Y + n * nsub;
    MB = MA + nsub * nsub;
    W = MB + nsub * nsub;
    errors = (double *)(W + nsub * nsub);
    serrors = errors + nsub;
    tau = (complex_t *)(errors + nsub);
    alpha = tau + nsub;
    beta = alpha + nsub;
    lapack_work = beta + nsub;

    op.nrhs = nsub;
    op.rhs = R;

    Hopf::generate_random_matrix(X, X + n * nsub, 2);

    for (count = 1; count <= 100; count++) {
        B.gemm(nsub, 0.0, R, 1.0, X);
        quad(op, nsub * n, Q, quad_work, 7 * n * nsub, 0.0, contour.arcLength(), 1e-3);
        for (i = 0; i < (n * nsub); i++) {
            Q[i] /= complex_t{ 0.0, 2 * M_PI };
        }

        // orthogonalize Q
        zgeqrf(n, nsub, Q, n, tau, lapack_work, lapack_lwork, info);
        zungqr(n, nsub, nsub, Q, n, tau, lapack_work, lapack_lwork, info);

        // form MA and MB
        A.gemm(nsub, 0.0, Y, 1.0, Q);
        DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, Q, n, Y, n, 0.0, MA, nsub);
        B.gemm(nsub, 0.0, Y, 1.0, Q);
        DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, Q, n, Y, n, 0.0, MB, nsub);
        // Krylov::better::print_forder_mat("MB", (double *)MB, 2 * nsub, 2 * nsub, nsub);

        // get subspace
        zggev4("N", "V", nsub, MA, nsub, MB, nsub, alpha, beta, NULL, nsub, W, nsub, lapack_work, lapack_lwork, rwork, jpvt, info);
        beta_tol = lapack_work[1].real();

        // get eigenvecotrs
        DEV::linalg::gemm<complex_t>("N", "N", n, nsub, nsub, 1.0, Q, n, W, nsub, 0.0, X, n);

        ninfinite = 0;
        // get residuals
        // R = -BX K
        B.gemm(nsub, 0.0, R, -1.0, X);
        // scale cols
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                ninfinite++;
                continue;
            }
            for (j = 0; j < n; j++) {
                R[i * n + j] *= (alpha[i] / beta[i]);
            }
        }
        // R += AX
        A.gemm(nsub, 1.0, R, 1.0, X);
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                memset(&R[i * n], 0, n * sizeof(complex_t));
                continue;
            }
        }

        // count how many are converged
        for (nconverged = 0, i = 0; i < nsub; i++) {
            errors[i] = norm2(&R[i * n], n);
            if (errors[i] < tol) {
                nconverged++;
            }
        }
        fmt::println("ninf = {}", ninfinite);
        nconverged -= ninfinite;

        memcpy(serrors, errors, nsub * sizeof(double));
        std::sort(serrors, serrors + nsub);
        fmt::println("{:4d} : {} | [{:.3e}]", count, nconverged, fmt::join(serrors, serrors + nstop, " "));

        if (nstop <= nconverged) {
            break;
        }
    }

    for (nconverged = 0, i = 0; i < nsub; i++) {
        if (std::abs(beta[i]) <= beta_tol) {
            continue;
        }
        if (errors[i] < tol) {
            memcpy(&X[nconverged * n], &X[i * n], n * sizeof(complex_t));
            alpha[nconverged++] = alpha[i] / beta[i];
        }
    }

    result.update(tol, n, nconverged, alpha, X);

    free(all_work);
    free(rwork);
    free(jpvt);

    return result;
}


std::complex<double> PieceWiseLinearContour::gamma(double t) const
{
    ptrdiff_t i, i1;
    double x, y, L;

    L = arcLength();
    t = std::fmod(std::fmod(t, L) + L, L);
    i = std::floor(t);
    i1 = (i + 1) % m_n;
    x = m_x[i] * (i + 1 - t) + m_x[i1] * (t - i);
    y = m_y[i] * (i + 1 - t) + m_y[i1] * (t - i);
    return { x, y };
}

std::complex<double> PieceWiseLinearContour::gammaPrime(double t) const
{
    ptrdiff_t i, i1;
    double x, y, L;

    L = arcLength();
    t = std::fmod(std::fmod(t, L) + L, L);
    i = std::floor(t);
    i1 = (i + 1) % m_n;
    x = m_x[i1] - m_x[i];
    y = m_y[i1] - m_y[i];
    return { x, y };
}

double PieceWiseLinearContour::arcLength() const
{
    return m_n;
}

void FEASTResult::clear()
{
    n = 0;
    evals.clear();
    evecs.clear();
}

void FEASTResult::update(const double tol, const ptrdiff_t n, const ptrdiff_t neig, const complex_t *k, const complex_t *V)
{
    ptrdiff_t i, j;

    clear();

    this->n = n;
    evals.reserve(2 * neig);
    evecs.reserve(2 * neig * n);

    // add base eigenvalues and vectors
    for (i = 0; i < neig; i++) {
        if (std::abs(k[i].imag()) < tol) {
            evals.emplace_back(k[i].real());
        }
        else {
            evals.emplace_back(k[i]);
        }
    }
    for (i = 0; i < (n * neig); i++) {
        evecs.emplace_back(V[i]);
    }
    // search for any cc pairs that can be added
    for (i = 0; i < neig; i++) {
        const auto ref = std::conj(evals[i]);
        auto min_loc = std::min_element(&evals[0], &evals[neig], [=](complex_t x, complex_t y) { return std::abs(ref - x) < std::abs(ref - y); });
        if (std::abs(*min_loc - ref) < (2 * tol)) {
            evals.emplace_back(ref);
            for (j = 0; j < n; j++) {
                evecs.emplace_back(std::conj(evecs[i * n + j]));
            }
        }
    }
    evecs.clear();
}

FEASTResult bfeast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol)
{
    using complex_t = std::complex<double>;
    constexpr ptrdiff_t nquad = 16;
    const int nthreads = 10;

    // quadrature and collocator stuff
    const double *b_rk = std::get<2>(GL_methods[nquad]);
    const double *c_rk = std::get<3>(GL_methods[nquad]);
    const double *A_rk_c = std::get<1>(GL_methods[collocator.getNStages()]);
    const double *b_rk_c = std::get<2>(GL_methods[collocator.getNStages()]);
    const auto &to_delete = collocator.indicesToDelete();
    const auto &diffusion = collocator.getDiffusion();

    // quadrature weights and nodes along contour
    double points[nquad];
    complex_t weights[nquad], gammas[nquad];

    // lapack stuff
    int info[1], count, nconverged, ninfinite;
    complex_t dummy[1];
    int *jpvt;
    double *rwork, *errors, *serrors, beta_tol;
    ptrdiff_t lapack_lwork;
    complex_t *lapack_work;

    // everything else
    complex_t *all_work = NULL, *Q, *R, *X, *Y, *MA, *MB, *W, *tau, *alpha, *beta;
    DEV::ComplexMatrix A, B;
    DEV::ComplexBoundaryConditions bc, zbc;
    std::array<DEV::ComplexMatrix, nquad> Ts;
    ptrdiff_t n, nsub, i, j, lwork;
    double M[collocator.getNode() * collocator.getNode()], D[collocator.getNode()];
    FEASTResult result;

    auto zgetrf = [](const int m, const int n, std::complex<double> *A, const int lda, int *ipiv, int *info) { zgetrf_(&m, &n, A, &lda, ipiv, info); };
    auto zgetrs = [](const char *trans, const int n, const int nrhs, const std::complex<double> *A, const int lda, const int *ipiv, std::complex<double> *B, const int ldb, int *info) { zgetrs_(trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, info); };
    auto zgeev = [](const char *jobvl, const char *jobvr, const int n, std::complex<double> *A, const int lda, std::complex<double> *W, std::complex<double> *VL, const int ldvl, std::complex<double> *VR, const int ldvr, std::complex<double> *work, const int lwork, double *rwork, int *info) { zgeev_(jobvl, jobvr, &n, A, &lda, W, VL, &ldvl, VR, &ldvr, work, &lwork, rwork, info); };

    Hopf::setup_M_and_D(collocator.getNode(), M, D, to_delete, diffusion, collocator.spatialPeriod(), collocator.waveSpeed());
    Hopf::generate_matrices(0.0, &collocator, A, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);
    n = A.size();
    nsub = std::min(n, std::max<ptrdiff_t>({ 2 * nquad, 2 * nstop + 1, 20, (ptrdiff_t)std::ceil(4 * std::sqrt(n)) }));
    // nsub = 10;

    lwork = (4 + (nthreads - 1)) * n * nsub + 3 * nsub * nsub + 5 * nsub;
    zgeqrf(n, nsub, NULL, n, NULL, dummy, -1, info);
    lapack_lwork = dummy->real();
    zggev4("N", "V", nsub, dummy, nsub, dummy, nsub, dummy, dummy, NULL, nsub, NULL, nsub, dummy, -1, NULL, NULL, info);
    lapack_lwork = std::max<ptrdiff_t>(lapack_lwork, dummy->real());
    lwork += lapack_lwork;

    all_work = (complex_t *)malloc(lwork * sizeof(complex_t));
    rwork = (double *)malloc(8 * nsub * sizeof(double));
    jpvt = (int *)malloc(nsub * sizeof(int));

    Q = all_work;
    R = Q + n * nsub;
    X = R + n * nsub;
    Y = X + n * nsub;
    MA = Y + n * nsub * nthreads;
    MB = MA + nsub * nsub;
    W = MB + nsub * nsub;
    errors = (double *)(W + nsub * nsub);
    serrors = errors + nsub;
    tau = (complex_t *)(errors + nsub);
    alpha = tau + nsub;
    beta = alpha + nsub;
    lapack_work = beta + nsub;

    // setup
    for (i = 0; i < nquad; i++) {
        auto &T = Ts[i];
        points[i] = c_rk[i] * contour.arcLength();
        gammas[i] = contour.gamma(points[i]);
        weights[i] = (b_rk[i] * contour.arcLength() * contour.gammaPrime(points[i])) / complex_t{ 0.0, 2 * M_PI };
        Hopf::generate_matrices(-gammas[i], &collocator, T, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);
        T.factor();
    }

    // initialize X
    Hopf::generate_random_matrix(Q, Q + n * nsub, 2);

    count = 0;
    nconverged = 0;
    omp_set_max_active_levels(1);
    while ((nconverged < nstop) && (count < 30)) {
        count++;
        ///////////////////////////////////
        B.gemm(nsub, 0.0, X, 1.0, Q);
        memset(Q, 0, n * nsub * sizeof(complex_t));
#pragma omp parallel for private(i, j) num_threads(nthreads)
        for (i = 0; i < nquad; i++) {
            auto &T = Ts[i];
            complex_t *Yi = &Y[omp_get_thread_num() * n * nsub];
            T.solve(X, Yi, nsub);
#pragma omp critical
            {
                for (j = 0; j < (n * nsub); j++) {
                    Q[j] += (points[i] * weights[i]) * Yi[j];
                }
            }
        }
        // DEV::linalg::gemv<complex_t>("N", n * nsub, nquad, 1.0, Y, n * nsub, weights, 1, 0.0, Q, 1);

        ///////////////////////////////////
        // Y[n * nsub] is U2
        B.gemm(nsub, 0.0, &Y[n * nsub], 1.0, Q);
        zgeqrf(n, nsub, &Y[n * nsub], n, tau, lapack_work, lapack_lwork, info);
        zungqr(n, nsub, nsub, &Y[n * nsub], n, tau, lapack_work, lapack_lwork, info);

        // orthogonalize Q
        zgeqrf(n, nsub, Q, n, tau, lapack_work, lapack_lwork, info);
        zungqr(n, nsub, nsub, Q, n, tau, lapack_work, lapack_lwork, info);

        // form MA and MB
        A.gemm(nsub, 0.0, Y, 1.0, Q);
        DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, &Y[n * nsub], n, Y, n, 0.0, MA, nsub);
        B.gemm(nsub, 0.0, Y, 1.0, Q);
        DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, &Y[n * nsub], n, Y, n, 0.0, MB, nsub);

        // get subspace
        TWIST::time_code("subspace", [&]() {
            zgetrf(nsub, nsub, MB, nsub, jpvt, info);
            zgetrs("N", nsub, nsub, MB, nsub, jpvt, MA, nsub, info);
            zgeev("N", "V", nsub, MA, nsub, alpha, NULL, nsub, W, nsub, lapack_work, lapack_lwork, rwork, info);
        });

        for (i = 0; i < nsub; i++) {
            beta[i] = 1.0;
        }
        beta_tol = 0.0;

        // beta_tol = lapack_work[1].real();

        // get eigenvecotrs
        DEV::linalg::gemm<complex_t>("N", "N", n, nsub, nsub, 1.0, Q, n, W, nsub, 0.0, X, n);

        ninfinite = 0;
        // get residuals
        // R = -BX K
        B.gemm(nsub, 0.0, R, -1.0, X);
        // scale cols
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                ninfinite++;
                continue;
            }
            for (j = 0; j < n; j++) {
                R[i * n + j] *= (alpha[i] / beta[i]);
            }
        }
        // R += AX
        A.gemm(nsub, 1.0, R, 1.0, X);
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                memset(&R[i * n], 0, n * sizeof(complex_t));
                continue;
            }
        }

        // count how many are converged
        for (nconverged = 0, i = 0; i < nsub; i++) {
            errors[i] = norm2(&R[i * n], n);
            if (errors[i] < tol) {
                nconverged++;
            }
        }
        fmt::println("ninf = {}", ninfinite);
        nconverged -= ninfinite;

        // print out errors
        memcpy(serrors, errors, nsub * sizeof(double));
        std::sort(serrors, serrors + nsub);
        fmt::println("{:4d} : {} | [{:.3e}]", count, nconverged, fmt::join(serrors, serrors + nstop, " "));
    }

    for (nconverged = 0, i = 0; i < nsub; i++) {
        if (std::abs(beta[i]) <= beta_tol) {
            continue;
        }
        if (errors[i] < tol) {
            memcpy(&X[nconverged * n], &X[i * n], n * sizeof(complex_t));
            alpha[nconverged++] = alpha[i] / beta[i];
        }
    }

    result.update(tol, n, nconverged, alpha, X);

    free(all_work);
    free(rwork);
    free(jpvt);

    return result;
}

FEASTResult nlfeast(TWIST::Collocator &collocator, const int nstop, const PieceWiseLinearContour &contour, const double tol)
{
    using complex_t = std::complex<double>;
    constexpr ptrdiff_t nquad = 9;
    constexpr bool testit = 0;

    const double *b_rk = std::get<2>(GL_methods[nquad]);
    const double *c_rk = std::get<3>(GL_methods[nquad]);
    const double *A_rk_c = std::get<1>(GL_methods[collocator.getNStages()]);
    const double *b_rk_c = std::get<2>(GL_methods[collocator.getNStages()]);

    double points[nquad], beta_tol;
    complex_t weights[nquad], gammas[nquad];
    int info[1], count, nconverged, ninfinite;
    complex_t dummy[1];

    int *jpvt;
    double *rwork, *errors, *serrors;
    complex_t *all_work = NULL, *Q, *R, *X, *Y, *MA, *MB, *W, *tau, *alpha, *beta, *lapack_work;
    DEV::ComplexMatrix A, B;
    DEV::ComplexBoundaryConditions bc, zbc;
    std::array<DEV::ComplexMatrix, nquad> Ts;
    ptrdiff_t n, nsub, i, j, k, lwork, lapack_lwork;
    double M[collocator.getNode() * collocator.getNode()], D[collocator.getNode()];
    FEASTResult result;

    auto main_work_block = [&]() {
        if constexpr (testit) {
            B.gemm(nsub, 0.0, &Y[n * nsub], 1.0, Q);
            zgeqrf(n, nsub, &Y[n * nsub], n, tau, lapack_work, lapack_lwork, info);
            zungqr(n, nsub, nsub, &Y[n * nsub], n, tau, lapack_work, lapack_lwork, info);

            // orthogonalize Q
            zgeqrf(n, nsub, Q, n, tau, lapack_work, lapack_lwork, info);
            zungqr(n, nsub, nsub, Q, n, tau, lapack_work, lapack_lwork, info);

            // form MA and MB
            A.gemm(nsub, 0.0, Y, 1.0, Q);
            DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, &Y[n * nsub], n, Y, n, 0.0, MA, nsub);
            B.gemm(nsub, 0.0, Y, 1.0, Q);
            DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, &Y[n * nsub], n, Y, n, 0.0, MB, nsub);
        }
        else {
            // orthogonalize Q
            zgeqrf(n, nsub, Q, n, tau, lapack_work, lapack_lwork, info);
            zungqr(n, nsub, nsub, Q, n, tau, lapack_work, lapack_lwork, info);

            // form MA and MB
            A.gemm(nsub, 0.0, Y, 1.0, Q);
            DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, Q, n, Y, n, 0.0, MA, nsub);
            B.gemm(nsub, 0.0, Y, 1.0, Q);
            DEV::linalg::gemm<complex_t>("C", "N", nsub, nsub, n, 1.0, Q, n, Y, n, 0.0, MB, nsub);
        }
        // Krylov::better::print_forder_mat("MB", (double *)MB, 2 * nsub, 2 * nsub, nsub);

        // get subspace
        zggev4("N", "V", nsub, MA, nsub, MB, nsub, alpha, beta, NULL, nsub, W, nsub, lapack_work, lapack_lwork, rwork, jpvt, info);
        beta_tol = lapack_work[1].real();

        // get eigenvecotrs
        DEV::linalg::gemm<complex_t>("N", "N", n, nsub, nsub, 1.0, Q, n, W, nsub, 0.0, X, n);

        ninfinite = 0;
        // get residuals
        // R = -BX K
        B.gemm(nsub, 0.0, R, -1.0, X);
        // scale cols
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                ninfinite++;
                continue;
            }
            for (j = 0; j < n; j++) {
                R[i * n + j] *= (alpha[i] / beta[i]);
            }
        }
        // R += AX
        A.gemm(nsub, 1.0, R, 1.0, X);
        for (i = 0; i < nsub; i++) {
            if (std::abs(beta[i]) <= beta_tol) {
                memset(&R[i * n], 0, n * sizeof(complex_t));
                continue;
            }
        }

        // count how many are converged
        for (nconverged = 0, i = 0; i < nsub; i++) {
            errors[i] = norm2(&R[i * n], n);
            if (errors[i] < tol) {
                nconverged++;
            }
        }
        fmt::println("ninf = {}", ninfinite);
        nconverged -= ninfinite;
    };

    const auto &to_delete = collocator.indicesToDelete();
    const auto &diffusion = collocator.getDiffusion();

    Hopf::setup_M_and_D(collocator.getNode(), M, D, to_delete, diffusion, collocator.spatialPeriod(), collocator.waveSpeed());
    Hopf::generate_matrices(0.0, &collocator, A, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);
    n = A.size();
    nsub = std::min(n, std::max<ptrdiff_t>({ 2 * nstop + 1, 20, (ptrdiff_t)std::ceil(2 * std::sqrt(n)) }));
    // nsub = 10;

    const int nthreads = 10;

    lwork = (4 + (nthreads - 1)) * n * nsub + 3 * nsub * nsub + 5 * nsub;
    zgeqrf(n, nsub, NULL, n, NULL, dummy, -1, info);
    lapack_lwork = dummy->real();
    zggev4("N", "V", nsub, dummy, nsub, dummy, nsub, dummy, dummy, NULL, nsub, NULL, nsub, dummy, -1, NULL, NULL, info);
    lapack_lwork = std::max<ptrdiff_t>(lapack_lwork, dummy->real());
    lwork += lapack_lwork;

    all_work = (complex_t *)malloc(lwork * sizeof(complex_t));
    rwork = (double *)malloc(8 * nsub * sizeof(double));
    jpvt = (int *)malloc(nsub * sizeof(int));

    Q = all_work;
    R = Q + n * nsub;
    X = R + n * nsub;
    Y = X + n * nsub;
    MA = Y + n * nsub * nthreads;
    MB = MA + nsub * nsub;
    W = MB + nsub * nsub;
    errors = (double *)(W + nsub * nsub);
    serrors = errors + nsub;
    tau = (complex_t *)(errors + nsub);
    alpha = tau + nsub;
    beta = alpha + nsub;
    lapack_work = beta + nsub;

    // setup
    for (i = 0; i < nquad; i++) {
        auto &T = Ts[i];
        points[i] = c_rk[i] * contour.arcLength();
        gammas[i] = contour.gamma(points[i]);
        weights[i] = (b_rk[i] * contour.arcLength() * contour.gammaPrime(points[i])) / complex_t{ 0.0, 2 * M_PI };
        Hopf::generate_matrices(-gammas[i], &collocator, T, B, &bc, &zbc, M, A_rk_c, b_rk_c, false);
        T.factor();
    }

    // initialize X
    Hopf::generate_random_matrix(X, X + n * nsub, 2);

    // init step
    memset(Q, 0, n * nsub * sizeof(complex_t));
    if constexpr (testit) {
        B.gemm(nsub, 0.0, Y, 1.0, X);
        memcpy(X, Y, n * nsub * sizeof(complex_t));
    }
    for (i = 0; i < nquad; i++) {
        auto &T = Ts[i];
        T.solve(X, Y, nsub);
        for (j = 0; j < (n * nsub); j++) {
            Q[j] += weights[i] * Y[j];
        }
    }
    // exit(0);
    count = 0;
    main_work_block();
    memcpy(serrors, errors, nsub * sizeof(double));
    std::sort(serrors, serrors + nsub);
    fmt::println("{:4d} : {} | [{:.3e}]", count, nconverged, fmt::join(serrors, serrors + nstop, " "));

    while ((nconverged < nstop) && (count < 30)) {
        count++;

        // init step
        if constexpr (testit) {
            B.gemm(nsub, 0.0, Y, 1.0, Q);
            memcpy(X, Y, n * nsub * sizeof(complex_t));
            memset(Q, 0, n * nsub * sizeof(complex_t));
            for (i = 0; i < nquad; i++) {
                auto &T = Ts[i];
                T.solve(X, Y, nsub);
                for (j = 0; j < (n * nsub); j++) {
                    Q[j] += weights[i] * Y[j];
                }
            }
        }
        else {
            memset(Q, 0, n * nsub * sizeof(complex_t));
#pragma omp parallel for private(i, j, k) num_threads(1)
            for (i = 0; i < nquad; i++) {
                auto &T = Ts[i];
                complex_t *Yi = &Y[omp_get_thread_num() * n * nsub];
                T.solve(R, Yi, nsub);
                for (j = 0; j < (n * nsub); j++) {
                    Yi[j] = X[j] - Yi[j];
                }
#pragma omp critical
                {
                    for (k = 0; k < nsub; k++) {
                        const auto scale = (std::abs(beta[i]) < beta_tol) ? 0.0 : weights[i] / (gammas[i] - (alpha[k] / beta[k]));
                        for (j = 0; j < n; j++) {
                            Q[k * n + j] += scale * Yi[k * n + j];
                        }
                    }
                }
            }
        }

        main_work_block();
        memcpy(serrors, errors, nsub * sizeof(double));
        std::sort(serrors, serrors + nsub);
        fmt::println("{:4d} : {} | [{:.3e}]", count, nconverged, fmt::join(serrors, serrors + nstop, " "));
    }

    for (nconverged = 0, i = 0; i < nsub; i++) {
        if (std::abs(beta[i]) <= beta_tol) {
            continue;
        }
        if (errors[i] < tol) {
            memcpy(&X[nconverged * n], &X[i * n], n * sizeof(complex_t));
            alpha[nconverged++] = alpha[i] / beta[i];
        }
    }

    result.update(tol, n, nconverged, alpha, X);

    free(all_work);
    free(rwork);
    free(jpvt);

    return result;
}
