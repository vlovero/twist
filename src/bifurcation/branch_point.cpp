#include "bif_points/branch_point.h"
#include "bif_points/hopf_point.h"
#include "libloader.h"

namespace BranchPoint
{
    void shift_to_origin(RP(double) alphar, RP(double) alphai, RP(double) beta, const ptrdiff_t N)
    {
        const double thresh = N * std::numeric_limits<double>::epsilon();
        ptrdiff_t i, imin;
        double val, smallest = std::numeric_limits<double>::infinity();

        for (i = 0; i < N; i++) {
            if (std::abs(beta[i]) < thresh) {
                continue;
            }
            val = std::hypot(alphar[i] / beta[i], alphai[i] / beta[i]);
            if (val < smallest) {
                smallest = val;
                imin = i;
            }
        }

        for (i = 0; i < N; i++) {
            if (std::abs(beta[i]) < thresh) {
                continue;
            }
            // r = r - shift * b
            alphar[i] -= alphar[imin] * (beta[i] / beta[imin]);
        }
    }

    void get_real_unstable(std::vector<double> &unstable, const double *alphar, const double *alphai, const double *beta, const ptrdiff_t N)
    {
        const double thresh = N * std::numeric_limits<double>::epsilon();
        ptrdiff_t i;
        double real;

        unstable.clear();

        for (i = 0; i < N; i++) {
            if (std::abs(beta[i]) < thresh) {
                continue;
            }
            real = alphar[i] / beta[i];
            if ((real > 0) && (alphai[i] == 0)) {
                unstable.emplace_back(real);
            }
        }
    }

    void set_BBlock_as_partition(const ptrdiff_t node, const ptrdiff_t nstages, double *block, const ptrdiff_t ldblock, const double mu, const double *M, const double *A, const double *b)
    {
        ptrdiff_t i, j, k, l;

        // make sure everything is zero first
        memset(block, 0, node * node * (nstages + 1) * (nstages + 2) * sizeof(double));

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

    void generate_matrices(const double k, TWIST::Collocator *collocator, DEV::Matrix &A, DEV::Matrix &B, DEV::BoundaryConditions *bc, DEV::BoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential)
    {
        double mu;
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
            Hopf::add_BBlock_to_partition(node, nstages, (A.m_partitions + i)->A, A.m_partitions->ldA(), mu, M, A_rk, b_rk);
            Hopf::add_BBlock_to_partition(node, nstages, (A.m_partitions_mm + i)->A, A.m_partitions->ldA(), mu, M, A_rk, b_rk);

            set_BBlock_as_partition(node, nstages, (B.m_partitions + i)->A, B.m_partitions->ldA(), h[i], M, A_rk, b_rk);
            set_BBlock_as_partition(node, nstages, (B.m_partitions_mm + i)->A, B.m_partitions->ldA(), h[i], M, A_rk, b_rk);
        }
    }


    void get_initial_eigenvector(const double k, std::valarray<double> &v, TWIST::Collocator *collocator, DEV::Matrix &A, DEV::Matrix &B, DEV::BoundaryConditions *bc, DEV::BoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk)
    {
        ptrdiff_t i;
        std::valarray<double> x, y;
        double mu, mu_prev;

        generate_matrices(-k, collocator, A, B, bc, zbc, M, A_rk, b_rk, false);
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

    void refine_eigenpair(double &k, std::valarray<double> &z, TWIST::Collocator *collocator, DEV::Matrix &A, DEV::Matrix &B, DEV::BoundaryConditions *bc, DEV::BoundaryConditions *zbc, const double *M, const double *A_rk, const double *b_rk, const bool essential)
    {
        ptrdiff_t i;
        double norm;
        double eta;
        std::valarray<double> r(z.size()), x(z.size()), y(z.size()), bz(z.size());
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
} // namespace BranchPoint