#include "preprocess/drxd.h"
#include "numpy_like.h"
#include "sparse_matrix.h"

ptrdiff_t adapt(const double eps, const ptrdiff_t Nx, const double *x, const ptrdiff_t node, const RP(double) z, const RP(double), RP(double) F, double **x_opt, double **y_new)
{
    constexpr double chat = 1.0 / 12;
    double h, vi, hd, hs, hd2, hs2, theta, tmp, theta_scale;
    // double vsum1, vsum2, r2;
    ptrdiff_t i, j, a, b, N_opt, i1, i2;
    const double alpha = 1.0 / (x[Nx - 1] - x[0]);

    theta_scale = 1.0;
    // fill F with v values
    // edge 0
    // v = eps + ||z'||
    for (i = 0; i < Nx; i++) {
        vi = 0;
        hs = (i != 0) ? (x[i] - x[i - 1]) : (x[Nx - 1] - x[Nx - 2]);
        hd = (i != (Nx - 1)) ? (x[i + 1] - x[i]) : (x[1] - x[0]);
        hd2 = hd * hd;
        hs2 = hs * hs;
        i1 = ((i - 1) + Nx) % Nx;
        i2 = (i + 1) % Nx;
        // why was this j < 1??
        for (j = 0; j < 1; j++) {
            // theta_scale = std::max(theta_scale, std::abs(z[i * node + j]));
            tmp = ((hs2 * z[i2 * node + j]) + ((hd2 - hs2) * z[i * node + j]) - (hd2 * z[i1 * node + j])) / (hs * hd * (hd + hs));
            vi += tmp * tmp;
        }
        F[i] = 1.0 / (x[Nx - 1] - x[0]) + std::sqrt(vi / j);
        F[i] = std::sqrt(alpha * alpha + vi / j);
        assert(std::isfinite(F[i]));
    }

    // F = cumtrap(v)
    theta = 0;
    tmp = F[0];
    F[0] = 0;
    // r2 = 0;
    for (i = 1; i < Nx; i++) {
        h = x[i] - x[i - 1];
        theta += 0.5 * h * (tmp + F[i]);
        tmp = F[i];
        F[i] = theta;
    }
    for (i = 0; i < Nx; i++) {
        F[i] /= theta;
    }

    // compute optimal number of points
    N_opt = (theta / theta_scale) * (chat / eps) + 1;
    N_opt = std::ceil(N_opt * 1.15);
    N_opt = std::max(3l, N_opt);
    a = 0.75 * Nx;
    b = 1.25 * Nx;
    N_opt = std::min(std::max(N_opt, a), b);

    // inverse intepolate to get new mesh
    *x_opt = (double *)realloc(*x_opt, N_opt * sizeof(double));
    for (i = 0; i < N_opt; i++) {
        x_opt[0][i] = (double)i / (N_opt - 1);
    }
    interp(*x_opt, N_opt, *x_opt, F, x, Nx, 1);
    for (i = 0; i < N_opt; i++) {
        assert(std::isfinite(x_opt[0][i]));
    }

    // update new and old solutions on new mesh
    *y_new = (double *)realloc(*y_new, (N_opt * node + 1) * sizeof(double));
    interp(*y_new, N_opt, *x_opt, x, z, Nx, node);
    for (i = 0; i < (N_opt * node); i++) {
        assert(std::isfinite(y_new[0][i]));
    }
    return N_opt;
}

void solve_vanderT(const ptrdiff_t n, const double *alpha, const double *b, double *x)
{
    ptrdiff_t k, j;
    for (k = 0; k < n; k++) {
        x[k] = b[k];
    }
    for (k = 0; k < (n - 1); k++) {
        for (j = n - 1; j > k; j--) {
            x[j] -= alpha[k] * x[j - 1];
        }
    }
    for (k = n - 2; k >= 0; k--) {
        for (j = k + 1; j < n; j++) {
            x[j] /= alpha[j] - alpha[j - k - 1];
        }
        for (j = k; j < (n - 1); j++) {
            x[j] -= x[j + 1];
        }
    }
}

void kron(const sparse::COOMatrix &A, const sparse::COOMatrix &B, sparse::COOMatrix &C)
{
    const int64_t Annz = A.nnz;
    const int64_t Bnnz = B.nnz;
    int64_t i, j;

    C.reshape(A.nrow * B.nrow, A.ncol * B.ncol);
    C.setNNZ(Annz * Bnnz);

    if (Bnnz != (int64_t)1) {
        for (i = 0; i < Annz; i++) {
            for (j = 0; j < Bnnz; j++) {
                C.irow[i * Bnnz + j] = A.irow[i] * B.nrow + B.irow[j];
                C.icol[i * Bnnz + j] = A.icol[i] * B.ncol + B.icol[j];
                C.data[i * Bnnz + j] = A.data[i] * B.data[j];
            }
        }
    }
    else {
        for (i = 0; i < Annz; i++) {
            C.irow[i] = A.irow[i] * B.nrow + B.irow[0];
            C.icol[i] = A.icol[i] * B.ncol + B.icol[0];
            C.data[i] = A.data[i] * B.data[0];
        }
    }
}

void construct_L2_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &L2, const diffusion_t &diffusion, sparse::COOMatrix &tmp)
{
    ptrdiff_t i, j;
    double alpha[3] = { 0, 0, 0 };
    double coeffs[3];
    double rhs[3] = { 0, 0, 2 };
    double *data;
    int64_t *irow;
    int64_t *icol;
    sparse::COOMatrix D;

    D.setNNZ(diffusion.size());
    D.reshape(node, node);
    tmp.setNNZ(3 * Nx);
    tmp.reshape(Nx, Nx);

    data = tmp.data;
    irow = tmp.irow;
    icol = tmp.icol;

    for (i = 0; i < Nx; i++) {
        // probably not the most performant but whatever
        j = (Nx - 1 + i) % Nx;
        alpha[0] = i ? x[j] - x[j + 1] : x[j - 1] - x[j];
        alpha[2] = (i != (Nx - 1)) ? (x[(j + 2) % Nx] - x[(j + 1) % Nx]) : (x[1] - x[0]);
        solve_vanderT(3, alpha, rhs, coeffs);
        for (j = 0; j < 3; j++) {
            *data++ = coeffs[j];
            *irow++ = i;
            *icol++ = (Nx - 1 + i + j) % Nx;
        }
    }
    i = 0;
    for (const auto &[index, coeff] : diffusion) {
        D.irow[i] = index;
        D.icol[i] = index;
        D.data[i] = coeff;
        i++;
    }
    kron(tmp, D, L2);
}

void construct_DL_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &DL, const diffusion_t &diffusion, sparse::COOMatrix &tmp)
{
    ptrdiff_t i, j;
    double alpha[3] = { 0, 0, 0 };
    double coeffs[3];
    double rhs[3] = { 0, 1, 0 };
    double *data;
    int64_t *irow;
    int64_t *icol;
    sparse::COOMatrix D;

    D.setNNZ(diffusion.size());
    D.reshape(node, node);
    tmp.setNNZ(3 * Nx);
    tmp.reshape(Nx, Nx);

    data = tmp.data;
    irow = tmp.irow;
    icol = tmp.icol;

    for (i = 0; i < Nx; i++) {
        // probably not the most performant but whatever
        switch (i) {
        case 0:
            alpha[0] = -((x[Nx - 2] - x[Nx - 3]) + (x[Nx - 1] - x[Nx - 2]));
            alpha[1] = -(x[Nx - 1] - x[Nx - 2]);
            break;
        case 1:
            alpha[0] = -((x[Nx - 1] - x[Nx - 2]) + (x[i] - x[i - 1]));
            alpha[1] = -(x[i] - x[i - 1]);
            break;
        default:
            alpha[0] = -((x[i - 1] - x[i - 2]) + (x[i] - x[i - 1]));
            alpha[1] = -(x[i] - x[i - 1]);
            break;
        }
        solve_vanderT(3, alpha, rhs, coeffs);
        for (j = 0; j < 3; j++) {
            *data++ = coeffs[j];
            *irow++ = i;
            *icol++ = (Nx - 2 + i + j) % Nx;
        }
    }
    i = 0;
    for (const auto &[index, coeff] : diffusion) {
        D.irow[i] = index;
        D.icol[i] = index;
        D.data[i] = coeff;
        i++;
    }
    kron(tmp, D, DL);
}

void construct_DC_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &DC, const diffusion_t &diffusion, sparse::COOMatrix &tmp)
{
    ptrdiff_t i, j;
    double alpha[3] = { 0, 0, 0 };
    double coeffs[3];
    double rhs[3] = { 0, 1, 0 };
    double *data;
    int64_t *irow;
    int64_t *icol;
    sparse::COOMatrix D;

    D.setNNZ(diffusion.size());
    D.reshape(node, node);
    tmp.setNNZ(3 * Nx);
    tmp.reshape(Nx, Nx);

    data = tmp.data;
    irow = tmp.irow;
    icol = tmp.icol;

    for (i = 0; i < Nx; i++) {
        // probably not the most performant but whatever
        j = (Nx - 1 + i) % Nx;
        alpha[0] = i ? x[j] - x[j + 1] : x[j - 1] - x[j];
        alpha[2] = (i != (Nx - 1)) ? (x[(j + 2) % Nx] - x[(j + 1) % Nx]) : (x[1] - x[0]);
        solve_vanderT(3, alpha, rhs, coeffs);
        for (j = 0; j < 3; j++) {
            *data++ = coeffs[j];
            *irow++ = i;
            *icol++ = (Nx - 1 + i + j) % Nx;
        }
    }
    i = 0;
    for (const auto &[index, coeff] : diffusion) {
        D.irow[i] = index;
        D.icol[i] = index;
        D.data[i] = coeff;
        i++;
    }
    kron(tmp, D, DC);
}

namespace frozen
{
    void kron(const sparse::COOMatrix &A, const sparse::COOMatrix &B, sparse::COOMatrix &C)
    {
        const int64_t Annz = A.nnz;
        const int64_t Bnnz = B.nnz;
        int64_t i, j;

        C.reshape(A.nrow * B.nrow, A.ncol * B.ncol);
        C.setNNZ(Annz * Bnnz);

        if (Bnnz != (int64_t)1) {
            for (i = 0; i < Annz; i++) {
                for (j = 0; j < Bnnz; j++) {
                    C.irow[i * Bnnz + j] = A.irow[i] * B.nrow + B.irow[j];
                    C.icol[i * Bnnz + j] = A.icol[i] * B.ncol + B.icol[j];
                    C.data[i * Bnnz + j] = A.data[i] * B.data[j];
                }
            }
        }
        else {
            for (i = 0; i < Annz; i++) {
                C.irow[i] = A.irow[i] * B.nrow + B.irow[0];
                C.icol[i] = A.icol[i] * B.ncol + B.icol[0];
                C.data[i] = A.data[i] * B.data[0];
            }
        }
    }

    void construct_L2_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &L2, const diffusion_t &diffusion, sparse::COOMatrix &tmp)
    {
        ptrdiff_t i, j;
        double alpha[3] = { 0, 0, 0 };
        double coeffs[3];
        double rhs[3] = { 0, 0, 2 };
        double *data;
        int64_t *irow;
        int64_t *icol;
        sparse::COOMatrix D;

        D.setNNZ(diffusion.size());
        D.reshape(node, node);
        tmp.setNNZ(3 * Nx);
        tmp.reshape(Nx, Nx);

        data = tmp.data;
        irow = tmp.irow;
        icol = tmp.icol;

        for (i = 0; i < Nx; i++) {
            // probably not the most performant but whatever
            j = (Nx - 1 + i) % Nx;
            alpha[0] = i ? x[j] - x[j + 1] : x[j - 1] - x[j];
            alpha[2] = (i != (Nx - 1)) ? (x[(j + 2) % Nx] - x[(j + 1) % Nx]) : (x[1] - x[0]);
            solve_vanderT(3, alpha, rhs, coeffs);
            for (j = 0; j < 3; j++) {
                *data++ = coeffs[j];
                *irow++ = i;
                *icol++ = (Nx - 1 + i + j) % Nx;
            }
        }
        i = 0;
        for (const auto &[index, coeff] : diffusion) {
            D.irow[i] = index;
            D.icol[i] = index;
            D.data[i] = coeff;
            i++;
        }
        kron(tmp, D, L2);
    }

    void construct_DL_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &DL, const diffusion_t &diffusion, sparse::COOMatrix &tmp)
    {
        ptrdiff_t i, j;
        double alpha[3] = { 0, 0, 0 };
        double coeffs[3];
        double rhs[3] = { 0, 1, 0 };
        double *data;
        int64_t *irow;
        int64_t *icol;
        sparse::COOMatrix D;

        D.setNNZ(diffusion.size());
        D.reshape(node, node);
        tmp.setNNZ(3 * Nx);
        tmp.reshape(Nx, Nx);

        data = tmp.data;
        irow = tmp.irow;
        icol = tmp.icol;

        for (i = 0; i < Nx; i++) {
            // probably not the most performant but whatever
            switch (i) {
            case 0:
                alpha[0] = -((x[Nx - 2] - x[Nx - 3]) + (x[Nx - 1] - x[Nx - 2]));
                alpha[1] = -(x[Nx - 1] - x[Nx - 2]);
                break;
            case 1:
                alpha[0] = -((x[Nx - 1] - x[Nx - 2]) + (x[i] - x[i - 1]));
                alpha[1] = -(x[i] - x[i - 1]);
                break;
            default:
                alpha[0] = -((x[i - 1] - x[i - 2]) + (x[i] - x[i - 1]));
                alpha[1] = -(x[i] - x[i - 1]);
                break;
            }
            solve_vanderT(3, alpha, rhs, coeffs);
            for (j = 0; j < 3; j++) {
                *data++ = coeffs[j];
                *irow++ = i;
                *icol++ = (Nx - 2 + i + j) % Nx;
            }
        }
        i = 0;
        for (const auto &[index, coeff] : diffusion) {
            D.irow[i] = index;
            D.icol[i] = index;
            D.data[i] = coeff;
            i++;
        }
        kron(tmp, D, DL);
    }

    template <typename SDIRK_T, typename fpde_t>
    inline bool quasi_newton_sdirk(RP(double) Z, RP(double) F, sparse::RealCSCMatrix &jac, RP(double) tmp, const double tn, const size_t Nx, const size_t node, RP(double) rn, RP(double) dz, fpde_t fpde, const double h, const double *yn, const double rtol)
    {
        const double tol = std::max(10 * std::numeric_limits<double>::epsilon() / rtol, std::min(0.03, std::sqrt(rtol)));
        constexpr int MAX_ITER = 100;
        const size_t nz = Nx * node + 1;
        size_t i;
        bool converged = false;
        double norm_dz, norm_dz_prev, rate, val;
        int stage, itern;

        for (stage = 0; stage < SDIRK_T::nstages; stage++) {
            // stage predictors
            if (stage == 0) {
                memset(&Z[stage * nz], 0, nz * sizeof(double));
            }
            else if (stage == 1) {
                memcpy(&Z[nz], Z, nz * sizeof(double));
            }
            else {
                dgemv("N", nz, stage, 1.0, Z, nz, &SDIRK_T::Q[(stage - 2) * SDIRK_T::nstages], 1, 0.0, &Z[stage * nz], 1);
            }
            converged = false;
            // newton iterations
            for (itern = 0; itern < MAX_ITER; itern++) {
                // evaluate Fi
                for (i = 0; i < nz; i++) {
                    tmp[i] = yn[i] + Z[stage * nz + i];
                }
                fpde(tn + SDIRK_T::C[stage] * h, tmp, &F[stage * nz]);

                // evaluate (negative) residual
                memcpy(rn, &Z[stage * nz], nz * sizeof(double));
                dgemv("N", nz, stage + 1, h, F, nz, &SDIRK_T::A[stage * SDIRK_T::nstages], 1, -1.0, rn, 1);

                // solve for newton step
                jac.solve(rn, dz);
                norm_dz = norm2(dz, nz);

                // failure check
                if (itern) {
                    rate = norm_dz / norm_dz_prev;
                    val = std::pow(rate, MAX_ITER - itern) / (1 - rate) * norm_dz;
                    if ((rate >= 1) || (val > tol)) {
                        break;
                    }
                }

                // apply newton step
                for (i = 0; i < nz; i++) {
                    Z[stage * nz + i] += dz[i];
                }

                // convergence check
                if (norm_dz == 0) {
                    converged = true;
                    break;
                }
                else if (itern) {
                    val = rate / (1 - rate) * norm_dz;
                    if (val < tol) {
                        converged = true;
                        break;
                    }
                }
                norm_dz_prev = norm_dz;
            }
            if (!converged) {
                // fmt::println("failed on stage {}", stage);
                break;
            }

            // evaluate Fi at latest Zi for next stages
            if (stage != (SDIRK_T::nstages - 1)) {
                for (i = 0; i < nz; i++) {
                    tmp[i] = yn[i] + Z[stage * nz + i];
                }
                fpde(tn + SDIRK_T::C[stage] * h, tmp, &F[stage * nz]);
            }

            // fmt::println("SOLVE(3) : stage {:d} converged in {:d} iterations", stage, itern + 1);
        }
        // puts("");
        return converged;
    }

} // namespace frozen
