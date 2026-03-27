#include "bif_points/hopf_point.h"
#include "collocator.h"
#include "colmat/tcolmat.h"
#include "shared.h"
#include <cstddef>
#include <dlfcn.h>

namespace Hopf
{
    void locate_pairs(const std::vector<double> &wr, const std::vector<double> &wi, std::vector<ptrdiff_t> &indices)
    {
        size_t i;
        indices.clear();
        for (i = 0; i < wi.size(); i++) {
            if (wi[i] == 0) {
                continue;
            }
            if (wr[i] >= 0) {
                indices.emplace_back(i + 0);
                indices.emplace_back(i + 1);
                i++; // increment i because next one is going to be cc
            }
        }
    }

    int count_complex_pairs(const double *alphar, const double *alphai, const double *beta, const size_t N)
    {
        int npairs = 0;
        for (size_t i = 0; i < N; i++) {
            if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            if (((alphar[i] / beta[i]) > 0) && (alphai[i] != 0)) {
                npairs++;
            }
        }
        return npairs >> 1;
    }

    void get_unstable_pairs(std::vector<complex_t> &unstable, const double *alphar, const double *alphai, const double *beta, const size_t N)
    {
        unstable.clear();
        for (size_t i = 0; i < N; i++) {
            if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            if (((alphar[i] / beta[i]) > 0) && (alphai[i] != 0)) {
                unstable.emplace_back(alphar[i] / beta[i], alphai[i] / beta[i]);
            }
        }
    }

    int count_unstable(const double *alphar, const double *alphai, const double *beta, const size_t N)
    {
        int nunstable = 0;
        double tmp, dmin = std::numeric_limits<double>::infinity();
        size_t i, argmin;
        // make sure to filter out zero first
        for (i = 0; i < N; i++) {
            if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            tmp = std::hypot(alphar[i] / beta[i], alphai[i] / beta[i]);
            if (tmp < dmin) {
                argmin = i;
                dmin = tmp;
            }
        }

        for (i = 0; i < N; i++) {
            if (std::abs(beta[i]) < (N * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            if (i == argmin) {
                continue;
            }
            if ((alphar[i] / beta[i]) > 0) {
                nunstable++;
            }
        }
        return nunstable;
    }

    void setup_M_and_D(const ptrdiff_t node, double *M, double *D, const std::vector<int> &to_delete, const diffusion_t &diffusion, const double L, const double speed)
    {
        ptrdiff_t i, j, k;

        memset(M, 0, node * node * sizeof(double));
        memset(D, 0, node * sizeof(double));

        for (const auto &[k, dk] : diffusion) {
            D[k] = dk;
        }

        // setup little M for filling in blocks
        k = 0;
        for (i = 0; i < node; i++) {
            if (std::find(to_delete.begin(), to_delete.end(), i + 1) != to_delete.end()) {
                continue;
            }
            if (std::find(to_delete.begin(), to_delete.end(), i) != to_delete.end()) {
                j = to_delete[k];
                k++;
                M[j * node + (j - 1)] = -L / D[j - k];
                continue;
            }
            M[i * node + i] = L / speed;
        }
        dlatrn(node, M, node);
    }

    void add_BBlock_to_partition(const ptrdiff_t node, const ptrdiff_t nstages, complex_t *block, const ptrdiff_t ldblock, const complex_t mu, const double *M, const double *A, const double *b)
    {
        ptrdiff_t i, j, k, l;

        for (i = 0; i < nstages; i++) {
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    block[((i + 1) * node + j) * ldblock + k] += mu * b[i] * M[j * node + k];
                }
            }
        }
        for (i = 0; i < nstages; i++) {
            for (j = 0; j < nstages; j++) {
                for (k = 0; k < node; k++) {
                    for (l = 0; l < node; l++) {
                        block[((i + 1) * node + k) * ldblock + (j + 1) * node + l] += mu * A[j * nstages + i] * M[k * node + l];
                    }
                }
            }
        }
    }

    void add_BBlock_to_partition(const ptrdiff_t node, const ptrdiff_t nstages, double *block, const ptrdiff_t ldblock, const double mu, const double *M, const double *A, const double *b)
    {
        ptrdiff_t i, j, k, l;

        for (i = 0; i < nstages; i++) {
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    block[((i + 1) * node + j) * ldblock + k] += mu * b[i] * M[j * node + k];
                }
            }
        }
        for (i = 0; i < nstages; i++) {
            for (j = 0; j < nstages; j++) {
                for (k = 0; k < node; k++) {
                    for (l = 0; l < node; l++) {
                        block[((i + 1) * node + k) * ldblock + (j + 1) * node + l] += mu * A[j * nstages + i] * M[k * node + l];
                    }
                }
            }
        }
    }


    void set_BBlock_as_partition(const ptrdiff_t node, const ptrdiff_t nstages, complex_t *block, const ptrdiff_t ldblock, const complex_t mu, const double *M, const double *A, const double *b)
    {
        ptrdiff_t i, j, k, l;

        // make sure everything is zero first
        memset(block, 0, node * node * (nstages + 1) * (nstages + 2) * sizeof(complex_t));

        for (i = 0; i < nstages; i++) {
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    block[((i + 1) * node + j) * ldblock + k] = mu * b[i] * M[j * node + k];
                }
            }
        }
        for (i = 0; i < nstages; i++) {
            for (j = 0; j < nstages; j++) {
                for (k = 0; k < node; k++) {
                    for (l = 0; l < node; l++) {
                        block[((i + 1) * node + k) * ldblock + (j + 1) * node + l] = mu * A[j * nstages + i] * M[k * node + l];
                    }
                }
            }
        }
    }

    void generate_matrices(const complex_t k, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential)
    {
        complex_t mu;
        ptrdiff_t i, node, nstages;
        func_t fjac;
        pjac_t pjac;
        int pmask[2];
        void *handle;
        const double *h = collocator->h();

        handle = collocator->getLibHandle();
        fjac = (func_t)dlsym(handle, "fjac");
        pjac = (pjac_t)dlsym(handle, "pjac");

        node = collocator->getNode();
        nstages = collocator->getNStages();

        // setup_M_and_D(node, M, D, , diffusion, L, speed)
        if (!essential) {
            bc->makePeriodic(collocator->getNode(), 0);
            zbc->makeZero(collocator->getNode(), 0);
        }
        A.update(fjac, pjac, collocator->getNNodes(), node, nstages, 0, collocator->unscaledSpatialPeriod(), h, collocator->y(), collocator->p(), pmask, NULL, bc);
        B.update(fjac, pjac, collocator->getNNodes(), node, nstages, 0, collocator->unscaledSpatialPeriod(), h, collocator->y(), collocator->p(), pmask, NULL, zbc);

        for (i = 0; i < A.m_nblocks; i++) {
            mu = h[i] * k;
            add_BBlock_to_partition(node, nstages, (A.m_partitions + i)->A, A.m_partitions->ldA(), mu, M, A_rk, b_rk);
            add_BBlock_to_partition(node, nstages, (A.m_partitions_mm + i)->A, A.m_partitions->ldA(), mu, M, A_rk, b_rk);

            set_BBlock_as_partition(node, nstages, (B.m_partitions + i)->A, B.m_partitions->ldA(), h[i], M, A_rk, b_rk);
            set_BBlock_as_partition(node, nstages, (B.m_partitions_mm + i)->A, B.m_partitions->ldA(), h[i], M, A_rk, b_rk);
        }
    }

    void get_initial_eigenvector(const complex_t k, std::valarray<complex_t> &v, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk)
    {
        ptrdiff_t i;
        std::valarray<complex_t> x, y;
        complex_t mu, mu_prev;

        generate_matrices(-k, collocator, A, B, bc, zbc, M, A_rk, b_rk);
        v.resize(A.size());
        x.resize(A.size());
        y.resize(A.size());
        std::fill(&v[0], &v[A.size()], 1.0 / std::sqrt(A.size()));

        mu_prev = k;
        // apply power iteration until the method has converged
        // fmt::println("INITIALIZING EIGENPAIR");
        for (i = 0; i < 100; i++) {
            A.solve(&v[0], &y[0]);

            mu = 1.0 / inner(&v[0], &y[0], y.size());
            y /= norm2(&y[0], y.size());
            v = y;
            // fmt::println("{:4d} : {:.8e}", i, abs(mu - mu_prev));
            if (abs(mu - mu_prev) < 1e-8) {
                break;
            }
            mu_prev = mu;
        }
        mu += k;
    }

    void refine_eigenpair(complex_t &k, std::valarray<complex_t> &z, TWIST::Collocator *collocator, DEV::ComplexMatrix &A, DEV::ComplexMatrix &B, DEV::ComplexBoundaryConditions *bc, DEV::ComplexBoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential)
    {
        ptrdiff_t i;
        double norm;
        complex_t eta;
        std::valarray<complex_t> r(z.size()), x(z.size()), y(z.size()), bz(z.size());
        // fmt::println("REFINING EIGENPAIR");
        for (i = 0; i < 100; i++) {
            generate_matrices(-k, collocator, A, B, bc, zbc, M, A_rk, b_rk, essential);
            B.gemv(0.0, &bz[0], 1.0, &z[0]);
            A.gemv(0.0, &r[0], 1.0, &z[0]);
            A.solve(&r[0], &x[0]);
            A.solve(&bz[0], &y[0]);
            eta = 1.0 / inner(&z[0], &y[0], z.size());
            x = eta * y - x;
            k += eta;
            z += x;
            z /= norm2(&z[0], z.size());

            norm = std::abs(inner(&x[0], &x[0], x.size()) + inner(&eta, &eta, 1));
            norm = abs(eta);
            // fmt::println("{:4d} : {:.8e} : ({: .8e}, {: .8e})", i, norm, k.real(), k.imag());
            if (i && (norm < 1e-8)) {
                break;
            }
        }
    }

} // namespace Hopf