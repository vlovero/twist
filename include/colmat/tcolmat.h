#pragma once

#include "GL/gl.h"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "krylov/bicgstab.h"
#include "linalg.h"
#include "profiling.h"
#include "shared.h"
#include "tools/helpers.h"
#include <cstddef>
#include <omp.h>
#include <stdexcept>
#include <type_traits>

extern "C" {
    void dger_(const int *m, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *a, const int *lda);
    void zger_(const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, const std::complex<double> *y, const int *incy, std::complex<double> *a, const int *lda);
}

namespace DEV
{
    namespace linalg
    {
        template <typename data_t>
        void gemv(const char *trans, const int m, const int n, const data_t alpha, const data_t *A, const int lda, const data_t *x, const int incx, const data_t beta, data_t *y, const int incy)
        {
            if ((m == 0) || (n == 0)) {
                return;
            }
            if constexpr (std::is_same_v<data_t, double>) {
                dgemv_(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zgemv_(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
            }
        }

        template <typename data_t>
        void latrn(const ptrdiff_t n, data_t *A, const ptrdiff_t ldA)
        {
            data_t temp;
            ptrdiff_t i, j;
            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    temp = A[i * ldA + j];
                    A[i * ldA + j] = A[j * ldA + i];
                    A[j * ldA + i] = temp;
                }
            }
        }

        template <typename data_t>
        void getrf(const int m, const int n, data_t *A, const int lda, int *ipiv, int *info)
        {
            if constexpr (std::is_same_v<data_t, double>) {
                dgetrf_(&m, &n, A, &lda, ipiv, info);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zgetrf_(&m, &n, A, &lda, ipiv, info);
            }
        }

        template <typename data_t>
        void trsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const data_t alpha, const RP(data_t) A, const int lda, RP(data_t) B, const int ldb)
        {
            if ((m == 0) || (n == 0)) {
                return;
            }
            if constexpr (std::is_same_v<data_t, double>) {
                dtrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                ztrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
            }
        }

        template <typename data_t>
        void gemm(const char *transa, const char *transb, const int nrow_op_a, const int ncol_op_b, const int nrow_op_b, const data_t alpha, const data_t *a, const int lda, const data_t *b, const int ldb, const data_t beta, data_t *c, const int ldc)
        {
            if constexpr (std::is_same_v<data_t, double>) {
                dgemm_(transa, transb, &nrow_op_a, &ncol_op_b, &nrow_op_b, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zgemm_(transa, transb, &nrow_op_a, &ncol_op_b, &nrow_op_b, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            }
        }

        template <typename data_t>
        void ger(const int m, const int n, const data_t alpha, const data_t *x, const int incx, const data_t *y, const int incy, data_t *a, const int lda)
        {
            if constexpr (std::is_same_v<data_t, double>) {
                dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
            }
        }

        template <typename data_t>
        void lacpy(const char *uplo, const int m, const int n, const data_t *a, const int lda, data_t *b, const int ldb)
        {
            if constexpr (std::is_same_v<data_t, double>) {
                dlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
            }
            else if constexpr (std::is_same_v<data_t, std::complex<double>>) {
                zlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
            }
        }

        template <typename data_t>
        void laswpc(const int n, RP(data_t) a, const int lda, const int k1, const int k2, const int *ipiv, const int)
        {
            int i;
            ptrdiff_t j;
            for (i = k1; i < k2; i++) {
                for (j = 0; j < n; j++) {
                    std::swap(a[i * lda + j], a[ipiv[i] * lda + j]);
                }
            }
        }

        template <typename data_t>
        void laswpr(const int n, RP(data_t) a, const int lda, const int k1, const int k2, const int *ipiv, const int)
        {
            int i;
            ptrdiff_t j;
            for (j = 0; j < n; j++) {
                for (i = k1; i < k2; i++) {
                    std::swap(a[j * lda + i], a[j * lda + ipiv[i]]);
                }
            }
        }

        template <typename data_t>
        void lacomp(const ptrdiff_t n, const ptrdiff_t m, const RP(data_t) A, const ptrdiff_t ldA, ptrdiff_t *ipiv, ptrdiff_t *jpiv)
        {
            ptrdiff_t i, j, imax, jmax;
            typename type_of_data<data_t>::type dmax, tmp;

            imax = 0;
            jmax = 0;
            dmax = 0;
            for (j = 0; j < m; j++) {
                for (i = 0; i < n; i++) {
                    tmp = std::abs(A[j * ldA + i]);
                    if (dmax < tmp) {
                        dmax = tmp;
                        imax = i;
                        jmax = j;
                    }
                }
            }
            *ipiv = imax;
            *jpiv = jmax;
        }

        template <typename data_t>
        void larook(const ptrdiff_t n, const ptrdiff_t m, const RP(data_t) A, const ptrdiff_t ldA, ptrdiff_t *ipiv, ptrdiff_t *jpiv)
        {
            bool col;
            typename type_of_data<data_t>::type dmax, tmp, refmax;
            ptrdiff_t j, imax, jmax, itmp;

            imax = 0;
            jmax = 0;
            dmax = 0.0;

            col = true;
            while (true) {
                refmax = 0.0;
                itmp = 0;
                if (col) {
                    // get max and argmax for current column
                    for (j = 0; j < n; j++) {
                        tmp = std::abs(A[jmax * ldA + j]);
                        if (refmax < tmp) {
                            itmp = j;
                            refmax = tmp;
                        }
                    }
                    // if not bigger break
                    if (refmax <= dmax) {
                        break;
                    }
                    // else update row index
                    dmax = refmax;
                    imax = itmp;
                    col = false;
                }
                else {
                    // get max and argmax for current row
                    for (j = 0; j < m; j++) {
                        tmp = std::abs(A[j * ldA + imax]);
                        if (refmax < tmp) {
                            itmp = j;
                            refmax = tmp;
                        }
                    }
                    // if not bigger break
                    if (refmax <= dmax) {
                        break;
                    }
                    // else update col index
                    dmax = refmax;
                    jmax = itmp;
                    col = true;
                }
            }
            *ipiv = imax;
            *jpiv = jmax;
        }

        template <typename data_t>
        void geluf(const char *piv, const ptrdiff_t n, const ptrdiff_t m, RP(data_t) A, const ptrdiff_t ldA, int *ipiv, int *jpiv)
        {
            ptrdiff_t minnm, i, j, k, jmax, imax;
            data_t aii, aik;
            const bool use_complete_pivoting = *piv == 'C';

            minnm = std::min(n, m);

            for (i = 0; i < minnm; i++) {
                if (use_complete_pivoting) {
                    // complete pivoting
                    lacomp(n - i, m - i, &A[i * (ldA + 1)], ldA, &imax, &jmax);
                    imax += i;
                    jmax += i;
                }
                else {
                    larook(n - i, m - i, &A[i * (ldA + 1)], ldA, &imax, &jmax);
                    imax += i;
                    jmax += i;
                }
                assert(imax < n);
                assert(jmax < m);
                ipiv[i] = imax;
                jpiv[i] = jmax;
                if (imax != i) {
                    for (j = 0; j < m; j++) {
                        std::swap(A[j * ldA + i], A[j * ldA + imax]);
                    }
                }
                if (jmax != i) {
                    for (j = 0; j < n; j++) {
                        std::swap(A[i * ldA + j], A[jmax * ldA + j]);
                    }
                }

                aii = A[i * (ldA + 1)];
                if (aii == 0.0) {
                    continue;
                }

                for (j = i + 1; j < n; j++) {
                    A[i * ldA + j] /= aii;
                }
                if (i < (minnm - 1)) {
                    for (k = i + 1; k < m; k++) {
                        aik = A[k * ldA + i];
                        for (j = i + 1; j < n; j++) {
                            A[k * ldA + j] -= A[i * ldA + j] * aik;
                        }
                    }
                }
            }
        }

        template <typename data_t, const ptrdiff_t block_size = 32>
        void getrp3(const ptrdiff_t n, const ptrdiff_t m, RP(data_t) A, const ptrdiff_t ldA, int *ipiv, int *jpiv)
        {
            ptrdiff_t minnm, i, j, l, jmax, imax, ncols;
            data_t aii;

            minnm = std::min(n, m);

            for (l = 0; l < minnm; l += block_size) {
                ncols = std::min(block_size, m - l);

                for (i = l; i < l + ncols; i++) {
                    larook(n - i, l + ncols - i, &A[i * (ldA + 1)], ldA, &imax, &jmax);
                    imax += i;
                    jmax += i;

                    assert(imax < n);
                    assert(jmax < m);
                    ipiv[i] = imax;
                    jpiv[i] = jmax;

                    if (imax != i) {
                        // swap locally and do other swaps later
                        for (j = l; j < l + ncols; j++) {
                            std::swap(A[j * ldA + i], A[j * ldA + imax]);
                        }
                    }
                    if (jmax != i) {
                        for (j = 0; j < n; j++) {
                            std::swap(A[i * ldA + j], A[jmax * ldA + j]);
                        }
                    }

                    aii = A[i * (ldA + 1)];
                    if (aii == 0.0) {
                        continue;
                    }

                    for (j = i + 1; j < n; j++) {
                        A[i * ldA + j] /= aii;
                    }
                    if (i < (l + ncols - 1)) {
                        ger<data_t>(n - i - 1, l + ncols - i - 1, -1.0, &A[i * ldA + i + 1], 1, &A[(i + 1) * ldA + i], ldA, &A[(i + 1) * (ldA + 1)], ldA);
                    }
                }
                // swap rows before window
                laswpr(l, A, ldA, l, l + ncols, ipiv, -1);

                if ((l + ncols) == m) {
                    continue;
                }

                // swap rows after window
                laswpr(m - l - ncols, &A[(l + ncols) * ldA], ldA, l, l + ncols, ipiv, -1);

                // Lower triangular and Schur stuff
                trsm<data_t>("L", "L", "N", "U", ncols, m - ncols - l, 1.0, &A[l * (ldA + 1)], ldA, &A[(l + ncols) * ldA + l], ldA);
                gemm<data_t>("N", "N", n - l - ncols, m - l - ncols, ncols, -1.0, &A[l * ldA + l + ncols], ldA, &A[(l + ncols) * ldA + l], ldA, 1.0, &A[(l + ncols) * (ldA + 1)], ldA);
            }
        }
    } // namespace linalg

    template <typename data_t>
    struct BoundaryConditionsBlocks
    {
        data_t *BC0 = nullptr;
        data_t *BC1 = nullptr;
        data_t *BCp = nullptr;

        BoundaryConditionsBlocks() = default;

        ~BoundaryConditionsBlocks()
        {
            if (BC0) {
                free(BC0);
                BC0 = nullptr;
                BC1 = nullptr;
                BCp = nullptr;
            }
        }

        void makePeriodic(const ptrdiff_t node, const ptrdiff_t nparam)
        {
            ptrdiff_t i, j;
            // realloc to be correct size
            BC0 = (data_t *)realloc(BC0, node * (2 * node + nparam) * sizeof(data_t));
            BC1 = BC0 + (node * node);
            BCp = BC1 + (node * node);

            // fill main blocks
            for (i = 0; i < node; i++) {
                for (j = 0; j < node; j++) {
                    BC0[i * node + j] = (i == j) ? +1.0 : 0.0;
                    BC1[i * node + j] = (i == j) ? -1.0 : 0.0;
                }
            }
            // fill parameter blockd
            for (i = 0; i < nparam; i++) {
                for (j = 0; j < node; j++) {
                    BCp[i * node + j] = 0.0;
                }
            }
        }

        void makePhaseShift(const double gamma, const ptrdiff_t node, const ptrdiff_t nparam, const double *y0)
        {
            static_assert(is_complex<data_t>(), "phase shift boundaries requires complex data");

            if (nparam > 1) {
                throw std::runtime_error("nparam should be 1. Not sure why it wouldn't when only varying gamma");
            }

            const double cs = std::cos(gamma);
            const double sn = std::sin(gamma);
            const data_t val1(cs, sn);
            const data_t val2(-sn, cs);

            ptrdiff_t i, j;
            // realloc to be correct size
            BC0 = (data_t *)realloc(BC0, node * (2 * node + nparam) * sizeof(data_t));
            BC1 = BC0 + (node * node);
            BCp = BC1 + (node * node);

            // fill main blocks
            for (i = 0; i < node; i++) {
                for (j = 0; j < node; j++) {
                    BC0[i * node + j] = (i == j) ? val1 : 0.0;
                    BC1[i * node + j] = (i == j) ? -1.0 : 0.0;
                }
            }
            // fill parameter blockd
            for (i = 0; i < nparam; i++) {
                for (j = 0; j < node; j++) {
                    BCp[i * node + j] = val2 * y0[i];
                }
            }
        }

        void makeZero(const ptrdiff_t node, const ptrdiff_t nparam)
        {
            // realloc to be correct size
            BC0 = (data_t *)realloc(BC0, node * (2 * node + nparam) * sizeof(data_t));
            BC1 = BC0 + (node * node);
            BCp = BC1 + (node * node);
            memset(BC0, 0, node * (2 * node + nparam) * sizeof(data_t));
        }

        // TODO projection bc
        // void setProjection(const ptrdiff_t node, const ptrdiff_t nparam, const data_t *W, const double *wr, const double *wi);
    };

    namespace internal
    {
        using namespace DEV::linalg;

        template <typename data_t>
        ptrdiff_t memory_chunk_size(const ptrdiff_t bsize, const ptrdiff_t node, const ptrdiff_t nparam)
        {
            const ptrdiff_t nrows = bsize + nparam;
            const ptrdiff_t ncols = bsize + node + nparam;
            return (nrows * ncols) * sizeof(data_t) + 2 * bsize * sizeof(int);
        }

        template <typename data_t>
        struct TrailingBlock
        {
            data_t *AL;
            data_t *AR;
            data_t *C;
            ptrdiff_t ldA;
        };

        template <typename data_t>
        struct FinalBlock
        {
            ptrdiff_t m_node = 0;
            ptrdiff_t m_nparam = 0;
            data_t *m_final = nullptr;
            int *m_ipiv = nullptr;
            int *m_jpiv = nullptr;

            FinalBlock() = default;

            FinalBlock(const ptrdiff_t node, const ptrdiff_t nparam)
            {
                reserve(node, nparam);
            }

            void reserve(const ptrdiff_t node, const ptrdiff_t nparam)
            {
                const ptrdiff_t N = 2 * node + nparam;
                m_node = node;
                m_nparam = nparam;
                m_ipiv = (int *)malloc(N * sizeof(int));
                m_jpiv = (int *)malloc(N * sizeof(int));
                m_final = (data_t *)malloc(N * N * sizeof(data_t));
            }

            ~FinalBlock()
            {
                m_node = 0;
                m_nparam = 0;
                if (m_ipiv) {
                    free(m_ipiv);
                    m_ipiv = nullptr;
                }
                if (m_jpiv) {
                    free(m_jpiv);
                    m_jpiv = nullptr;
                }
                if (m_final) {
                    free(m_final);
                    m_final = nullptr;
                }
            }

            void update(const ptrdiff_t node, const ptrdiff_t nparam, const data_t *AL, const data_t *AR, const data_t *C, const BoundaryConditionsBlocks<data_t> *bc, const data_t *BL, const data_t *BR, const data_t *D)
            {
                // assuming ldA = 2 * node and ldB/D = nparam
                ptrdiff_t i, j;
                const ptrdiff_t N = 2 * node + nparam;
                const ptrdiff_t ldA = 2 * node;

                if ((m_node != node) || (m_nparam != nparam)) {
                    reserve(node, nparam);
                }

                for (i = 0; i < m_node; i++) {
                    for (j = 0; j < m_node; j++) {
                        // AL and AR
                        m_final[i * N + j] = AL[i * ldA + j];
                        m_final[(i + m_node) * N + j] = AR[i * ldA + j];

                        // BC0 and BC1
                        m_final[i * N + j + m_node] = bc->BC0[i * m_node + j];
                        m_final[(i + m_node) * N + j + m_node] = bc->BC1[i * m_node + j];
                    }
                    // BL and BR
                    for (j = 0; j < m_nparam; j++) {
                        m_final[i * N + j + ldA] = BL[i * m_nparam + j];
                        m_final[(i + m_node) * N + j + ldA] = BR[i * m_nparam + j];
                    }
                }
                // C and BCp
                for (i = 0; i < m_nparam; i++) {
                    for (j = 0; j < m_node; j++) {
                        m_final[(i + ldA) * N + j] = C[i * ldA + j];
                        m_final[(i + ldA) * N + j + m_node] = bc->BCp[i * m_node + j];
                    }
                }
                // D
                for (i = 0; i < m_nparam; i++) {
                    for (j = 0; j < m_nparam; j++) {
                        m_final[(N - m_nparam + i) * N + (N - m_nparam + j)] = D[i * m_nparam + j];
                    }
                }
            }

            void factor()
            {
                const int N = 2 * m_node + m_nparam;
                geluf("C", N, N, m_final, N, m_ipiv, m_jpiv);
            }

            void solve(const data_t *b, data_t *x)
            {
                const ptrdiff_t N = 2 * m_node + m_nparam;

                ptrdiff_t i;
                int colperm[N];
                data_t tmp[N];

                memcpy(x, b, N * sizeof(data_t));
                // row perm
                laswpr(1, x, N, 0, N, m_ipiv, 1);
                // solve lower
                trsm<data_t>("L", "L", "N", "U", N, 1, 1.0, m_final, N, x, N);
                // solve upper
                trsm<data_t>("L", "U", "N", "N", N, 1, 1.0, m_final, N, x, N);
                // col perm
                for (i = 0; i < N; i++) {
                    colperm[i] = i;
                }
                for (i = 0; i < N; i++) {
                    std::swap(colperm[i], colperm[m_jpiv[i]]);
                }
                for (i = 0; i < N; i++) {
                    tmp[colperm[i]] = x[i];
                }
                for (i = 0; i < N; i++) {
                    x[i] = tmp[i];
                }
            }

            void solve(const data_t *b, data_t *x, const ptrdiff_t nrhs, const ptrdiff_t ldbx)
            {
                const ptrdiff_t N = 2 * m_node + m_nparam;

                ptrdiff_t i, j;
                int colperm[N];
                data_t tmp[N];

                lacpy("A", N, nrhs, b, ldbx, x, ldbx);
                // row perm
                laswpr(nrhs, x, N, 0, N, m_ipiv, 1);
                // solve lower
                trsm<data_t>("L", "L", "N", "U", N, nrhs, 1.0, m_final, N, x, ldbx);
                // solve upper
                trsm<data_t>("L", "U", "N", "N", N, nrhs, 1.0, m_final, N, x, ldbx);
                // col perm
                for (i = 0; i < N; i++) {
                    colperm[i] = i;
                }
                for (i = 0; i < N; i++) {
                    std::swap(colperm[i], colperm[m_jpiv[i]]);
                }
                for (j = 0; j < nrhs; j++) {
                    for (i = 0; i < N; i++) {
                        tmp[colperm[i]] = x[j * ldbx + i];
                    }
                    for (i = 0; i < N; i++) {
                        x[j * ldbx + i] = tmp[i];
                    }
                }
            }
        };

        struct Info
        {
            enum Flags
            {
                factored = 0,
                hasTrailing,
                hasChild
            };

            uint64_t m_mask = 0;

            Info() = default;

            void setFactored()
            {
                m_mask |= ((uint64_t)1 << Flags::factored);
            }

            void setHasTrailing()
            {
                m_mask |= ((uint64_t)1 << Flags::hasTrailing);
            }

            void setHasChild()
            {
                m_mask |= ((uint64_t)1 << Flags::hasChild);
            }

            void clearFactored()
            {
                m_mask &= ~((uint64_t)1 << Flags::factored);
            }

            void clearHasTrailing()
            {
                m_mask &= ~((uint64_t)1 << Flags::hasTrailing);
            }

            void clearHasChild()
            {
                m_mask &= ~((uint64_t)1 << Flags::hasChild);
            }

            void clearAll()
            {
                m_mask = 0;
            }

            bool getFactored() const
            {
                return m_mask & ((uint64_t)1 << Flags::factored);
            }

            bool getHasTrailing() const
            {
                return m_mask & ((uint64_t)1 << Flags::hasTrailing);
            }

            bool getHasChild() const
            {
                return m_mask & ((uint64_t)1 << Flags::hasChild);
            }
        };

        template <typename data_t>
        struct Partition
        {
            data_t *A = nullptr;
            data_t *B = nullptr;
            data_t *C = nullptr;
            data_t *D = nullptr;
            int *ipiv = nullptr;
            int *jpiv = nullptr;
            ptrdiff_t bsize = 0;
            ptrdiff_t node = 0;
            ptrdiff_t nparam = 0;

            // constructors and destructor
            Partition() = default;
            Partition(const Partition<data_t> &) = delete;

            ~Partition()
            {
                bsize = 0;
                node = 0;
                nparam = 0;
                if (A != nullptr) {
                    free(A);
                    A = nullptr;
                }
            }

            // move assignment
            Partition<data_t> &operator=(Partition<data_t> &&other)
            {
                A = other.A;
                other.A = nullptr;
                B = other.B;
                other.B = nullptr;
                C = other.C;
                other.C = nullptr;
                D = other.D;
                other.D = nullptr;
                ipiv = other.ipiv;
                other.ipiv = nullptr;
                jpiv = other.jpiv;
                other.jpiv = nullptr;
                bsize = other.bsize;
                node = other.node;
                nparam = other.nparam;

                return *this;
            }

            // [unsafe] copy assignment
            void fill(const Partition<data_t> &other)
            {
                // assumes that other.(node|nparam|bsize) is already the same
                memcpy(A, other.A, memory_chunk_size<data_t>(bsize, node, nparam));
            }

            // update method from collocator
            void update(func_t fjac, pjac_t pjac, const ptrdiff_t node, const ptrdiff_t ncol, const ptrdiff_t nparam, const double h, const double *Y, const double *p, const int *pmask, const double *prow, double *work)
            {
                TWIST_CXX_MARK_FUNCTION;
                // NOTE: h NEEDS TO BE PRESCALED BY DEFAULT SPATIAL PERIOD
                ptrdiff_t i, j, k, l, node2, bsize;
                double *jac_i;
                data_t *Aview;
                double scale, delta_ij, delta_kl;

                const _method_t &method = GL_methods[ncol];
                const double *A_rk = std::get<1>(method);
                const double *b_rk = std::get<2>(method);

                node2 = node * node;
                bsize = ldA();

                // set A, B, D to zero
                TWIST_MARK_BEGIN("empty partition");
                emptyPartition(ncol, node, nparam);
                TWIST_MARK_END("empty partition");

                // fill in A
                // step 1: evaluate jacobians and transpose
                TWIST_MARK_BEGIN("compute fjacs");
                for (i = 0; i < ncol; i++) {
                    fjac(&Y[i * node], p, &work[i * node2]);
                    // jacobians are in row major order and all of the
                    // fortran crap is in column major order
                    latrn(node, &work[i * node2], node);
                }
                TWIST_MARK_END("compute fjacs");
                // step 2: top blocks for quadrature conditions

                TWIST_MARK_BEGIN("fill in A");
                for (i = 0; i < ncol; i++) {
                    jac_i = &work[i * node2];
                    Aview = &A[(i + 1) * node * bsize];
                    scale = h * b_rk[i];
                    for (j = 0; j < node; j++) {
                        for (k = 0; k < node; k++) {
                            Aview[j * bsize + k] = scale * jac_i[j * node + k];
                        }
                    }
                }
                // step 3: fill in collocation portion
                for (i = 0; i < ncol; i++) {
                    jac_i = &work[i * node2];
                    for (j = 0; j < ncol; j++) {
                        delta_ij = (i == j) ? 1.0 : 0.0;
                        scale = h * A_rk[j * ncol + i];
                        Aview = &A[((i + 1) * node) * bsize + ((j + 1) * node)];
                        for (k = 0; k < node; k++) {
                            for (l = 0; l < node; l++) {
                                delta_kl = (k == l) ? 1.0 : 0.0;
                                Aview[k * bsize + l] = scale * jac_i[k * node + l] - delta_ij * delta_kl;
                            }
                        }
                    }
                }
                // step 4: fill in I's and -I
                for (i = 0; i < node; i++) {
                    for (j = 0; j < (ncol + 1); j++) {
                        A[i * bsize + (j * node + i)] = 1.0;
                    }
                }
                for (i = 0; i < node; i++) {
                    A[(i + bsize) * bsize + i] = -1.0;
                }
                TWIST_MARK_END("fill in A");

                // fill in C
                // evaluate parameter jacobians first and transpose them
                TWIST_MARK_BEGIN("compute pjac");
                for (i = 0; i < ncol; i++) {
                    pjac(&Y[i * node], p, pmask, nparam, &work[i * node2]);
                }
                TWIST_MARK_END("compute pjac");

                // dFk/dc = 2T hk sum(c_b[i] fc(yk[i]))
                TWIST_MARK_BEGIN("fill in C and B");
                for (i = 0; i < ncol; i++) {
                    scale = h * b_rk[i];
                    jac_i = &work[i * node2];
                    for (k = 0; k < nparam; k++) {
                        for (l = 0; l < node; l++) {
                            C[k * bsize + l] += scale * jac_i[l * nparam + k];
                        }
                    }
                }

                for (j = 0; j < ncol; j++) {
                    Aview = &C[(j + 1) * node];
                    // dHki/dc = 2T hk sum(a_ij fc(yk[j]))
                    for (i = 0; i < ncol; i++) {
                        scale = h * A_rk[j * ncol + i];
                        jac_i = &work[i * node2];
                        for (k = 0; k < nparam; k++) {
                            for (l = 0; l < node; l++) {
                                Aview[k * bsize + l] += scale * jac_i[l * nparam + k];
                            }
                        }
                    }
                }

                // fill in B
                std::copy(prow, &prow[ncol * nparam * node], &B[node * nparam]);
                TWIST_MARK_END("fill in C and B");
            }

            // update routines from parents
            void update(const Partition<data_t> *partition1, const Partition<data_t> *partition2, const data_t *prow)
            {
                // zero out A, D, and B
                memset(A, 0, memory_chunk_size<data_t>(bsize, node, nparam));

                // copy first two A's
                lacpy<data_t>("A", node, node, partition1->ALs(), partition1->ldA(), A, ldA());
                lacpy<data_t>("A", node, node, partition1->ARs(), partition1->ldA(), &A[node * ldA()], ldA());

                // copy second two A's
                lacpy<data_t>("A", node, node, partition2->ALs(), partition1->ldA(), &A[node * (ldA() + 1)], ldA());
                lacpy<data_t>("A", node, node, partition2->ARs(), partition1->ldA(), &A[ldA() * ldA() + node], ldA());

                // copy C's
                lacpy<data_t>("A", node, nparam, partition1->Cs(), partition1->ldA(), C, ldA());
                lacpy<data_t>("A", node, nparam, partition2->Cs(), partition1->ldA(), &C[node], ldA());

                // copy B
                lacpy<data_t>("A", nparam, node, prow, nparam, &B[node * nparam], nparam);
            }

            void update(const Partition<data_t> *partition1, const TrailingBlock<data_t> *trailing, const data_t *prow)
            {
                // zero out A, D, and B
                memset(A, 0, memory_chunk_size<data_t>(bsize, node, nparam));

                // copy first two A's
                lacpy<data_t>("A", node, node, partition1->ALs(), partition1->ldA(), A, ldA());
                lacpy<data_t>("A", node, node, partition1->ARs(), partition1->ldA(), &A[node * ldA()], ldA());

                // copy second two A's
                lacpy<data_t>("A", node, node, trailing->AL, trailing->ldA, &A[node * (ldA() + 1)], ldA());
                lacpy<data_t>("A", node, node, trailing->AR, trailing->ldA, &A[ldA() * ldA() + node], ldA());

                // copy C's
                lacpy<data_t>("A", node, nparam, partition1->Cs(), partition1->ldA(), C, ldA());
                lacpy<data_t>("A", node, nparam, trailing->C, partition1->ldA(), &C[node], ldA());

                // copy B
                lacpy<data_t>("A", nparam, node, prow, nparam, &B[node * nparam], nparam);
            }

            // for debugging
            void display() const
            {
                ptrdiff_t i, j;
                std::vector<data_t> tmp;

                for (i = 0; i < bsize; i++) {
                    tmp.clear();
                    for (j = 0; j < (bsize + node); j++) {
                        tmp.emplace_back(A[j * bsize + i]);
                    }
                    fmt::print("[{: .3e}] | ", fmt::join(tmp, " "));
                    tmp.clear();
                    for (j = 0; j < nparam; j++) {
                        tmp.emplace_back(C[j * bsize + i]);
                    }
                    fmt::println("[{: .3e}]", fmt::join(tmp, " "));
                }
                puts("");

                for (i = 0; i < nparam; i++) {
                    tmp.clear();
                    for (j = 0; j < (bsize + node); j++) {
                        tmp.emplace_back(B[j * nparam + i]);
                    }
                    fmt::print("[{: .3e}] | ", fmt::join(tmp, " "));
                    tmp.clear();
                    for (j = 0; j < nparam; j++) {
                        tmp.emplace_back(D[j * nparam + i]);
                    }
                    fmt::println("[{: .3e}]", fmt::join(tmp, " "));
                }
                puts("");
            }

            // realloc and zero
            void emptyPartition(const ptrdiff_t new_ncol, const ptrdiff_t new_node, const ptrdiff_t new_nparam)
            {
                ptrdiff_t total_bytes, bsize_new;
                void *buffer;

                bsize_new = new_node * (new_ncol + 1);
                if ((bsize_new == bsize) && (new_node == node) && (new_nparam == nparam)) {
                    total_bytes = memory_chunk_size<data_t>(bsize, node, nparam);
                    memset(A, 0, total_bytes);
                    return;
                }

                node = new_node;
                nparam = new_nparam;
                bsize = bsize_new;

                total_bytes = memory_chunk_size<data_t>(bsize, node, nparam);
                buffer = realloc(A, total_bytes);
                memset(buffer, 0, total_bytes);

                A = (data_t *)buffer;
                B = &A[bsize * (bsize + node)];
                C = &B[nparam * (bsize + node)];
                D = &C[bsize * nparam];
                ipiv = (int *)(&D[nparam * nparam]);
                jpiv = &ipiv[bsize];
            }

            // factor method(s)
            void condense()
            {
                TWIST_CXX_MARK_FUNCTION;
                assert(bsize != 0);
                // display();
                data_t *LU = &A[bsize * node];

                // factor window pivot window
                TWIST_MARK_BEGIN("condense (LU)");
                // geluf((bsize == (2 * node) ? "C" : "R"), bsize, bsize - node, LU, bsize, ipiv, jpiv);
                if (bsize == (2 * node)) {
                    geluf("C", bsize, bsize - node, LU, bsize, ipiv, jpiv);
                }
                else {
                    getrp3(bsize, bsize - node, LU, bsize, ipiv, jpiv);
                }
                TWIST_MARK_END("condense (LU)");
                // dgeluf("C", bsize, bsize - node, LU, bsize, ipiv, jpiv);

                TWIST_MARK_BEGIN("condense (swaps)");
                // apply permutations
                laswpr(node, A, bsize, 0, bsize - node, ipiv, 1);
                laswpr(node, &A[bsize * bsize], bsize, 0, bsize - node, ipiv, 1);
                laswpr(nparam, C, bsize, 0, bsize - node, ipiv, 1);
                laswpc(nparam, &B[node * nparam], nparam, 0, bsize - node, jpiv, 1);
                TWIST_MARK_END("condense (swaps)");

                TWIST_MARK_BEGIN("condense blas ops");
                // apply operations to A (left)
                trsm<data_t>("L", "L", "N", "U", bsize - node, node, 1.0, LU, bsize, A, bsize);
                gemm<data_t>("N", "N", node, node, bsize - node, -1.0, &LU[bsize - node], bsize, A, bsize, 1.0, &A[bsize - node], bsize);
                // apply operations to A (right)
                trsm<data_t>("L", "L", "N", "U", bsize - node, node, 1.0, LU, bsize, &A[bsize * bsize], bsize);
                gemm<data_t>("N", "N", node, node, bsize - node, -1.0, &LU[bsize - node], bsize, &A[bsize * bsize], bsize, 1.0, &A[bsize * bsize + (bsize - node)], bsize);
                // apply operations to C
                trsm<data_t>("L", "L", "N", "U", bsize - node, nparam, 1.0, LU, bsize, C, bsize);
                gemm<data_t>("N", "N", node, nparam, bsize - node, -1.0, &LU[bsize - node], bsize, C, bsize, 1.0, &C[bsize - node], bsize);
                // factor B
                trsm<data_t>("R", "U", "N", "N", nparam, bsize - node, 1.0, LU, bsize, &B[node * nparam], nparam);

                // D, V, and W are shared between partitions
                // so the condense method assumes it to be zero
                // so that after condensation is done, the first
                // subsystem can be formed by adding them all up

                // factor D
                gemm<data_t>("N", "N", nparam, nparam, bsize - node, -1.0, &B[node * nparam], nparam, C, bsize, 0.0, D, nparam);
                // factor left of B (V)
                gemm<data_t>("N", "N", nparam, node, bsize - node, -1.0, &B[node * nparam], nparam, A, bsize, 0.0, B, nparam);
                // factor right of B (W)
                gemm<data_t>("N", "N", nparam, node, bsize - node, -1.0, &B[node * nparam], nparam, &A[bsize * bsize], bsize, 0.0, &B[bsize * nparam], nparam);
                TWIST_MARK_END("condense blas ops");
            }

            // solve method(s)
            void condenseVector(RP(data_t) x, const ptrdiff_t nx, const ptrdiff_t partition_num) const
            {
                const data_t *LU = &A[bsize * node];

                data_t *Fx = &x[partition_num * bsize];
                data_t *Fd = &x[nx - nparam];

                // permute
                laswpr(1, Fx, nx, 0, bsize - node, ipiv, 1);
                // elimitate square
                trsm<data_t>("L", "L", "N", "U", bsize - node, 1, 1.0, LU, bsize, Fx, nx);
                // eliminate remaining block
                gemm<data_t>("N", "N", node, 1, bsize - node, -1.0, &LU[bsize - node], bsize, Fx, nx, 1.0, &Fx[bsize - node], nx);
                // eliminate last nparam rows
                gemm<data_t>("N", "N", nparam, 1, bsize - node, -1.0, &B[node * nparam], nparam, Fx, nx, 1.0, Fd, nx);
            }

            void condenseVector(const ptrdiff_t nrhs, RP(data_t) x, const ptrdiff_t nx, const ptrdiff_t partition_num) const
            {
                const data_t *LU = &A[bsize * node];

                data_t *Fx = &x[partition_num * bsize];
                data_t *Fd = &x[nx - nparam];

                // permute
                laswpr(nrhs, Fx, nx, 0, bsize - node, ipiv, 1);
                // elimitate square
                trsm<data_t>("L", "L", "N", "U", bsize - node, nrhs, 1.0, LU, bsize, Fx, nx);
                // eliminate remaining block
                gemm<data_t>("N", "N", node, nrhs, bsize - node, -1.0, &LU[bsize - node], bsize, Fx, nx, 1.0, &Fx[bsize - node], nx);
                // eliminate last nparam rows
                gemm<data_t>("N", "N", nparam, nrhs, bsize - node, -1.0, &B[node * nparam], nparam, Fx, nx, 1.0, Fd, nx);
            }

            void backsubstitute(RP(data_t) x, RP(data_t) y, const ptrdiff_t nx, const ptrdiff_t partition_num) const
            {
                ptrdiff_t i;
                const data_t *LU = &A[bsize * node];
                int colperm[bsize - node];

                data_t *R = &y[partition_num * bsize];
                data_t *xL = &x[partition_num * bsize];
                data_t *xC = xL + node;
                data_t *xR = xL + bsize;
                data_t *K = &x[nx - nparam];

                gemv<data_t>("N", bsize - node, node, -1.0, A, bsize, xL, 1, 1.0, R, 1);
                gemv<data_t>("N", bsize - node, node, -1.0, &A[bsize * bsize], bsize, xR, 1, 1.0, R, 1);
                gemv<data_t>("N", bsize - node, nparam, -1.0, C, bsize, K, 1, 1.0, R, 1);
                memcpy(xC, R, (bsize - node) * 1 * sizeof(data_t));
                trsm<data_t>("L", "U", "N", "N", bsize - node, 1, 1.0, LU, bsize, xC, nx);

                for (i = 0; i < (bsize - node); i++) {
                    colperm[i] = i;
                }
                for (i = 0; i < (bsize - node); i++) {
                    std::swap(colperm[i], colperm[jpiv[i]]);
                }
                for (i = 0; i < (bsize - node); i++) {
                    R[colperm[i]] = xC[i];
                }
                for (i = 0; i < (bsize - node); i++) {
                    xC[i] = R[i];
                }
            }

            void backsubstitute(const ptrdiff_t nrhs, RP(data_t) x, RP(data_t) y, const ptrdiff_t nx, const ptrdiff_t partition_num) const
            {
                // throw std::runtime_error(fmt::format("{} not implemented", __PRETTY_FUNCTION__));
                ptrdiff_t i, j;
                const data_t *LU = &A[bsize * node];
                int colperm[bsize - node];

                data_t *R = &y[partition_num * bsize];
                data_t *xL = &x[partition_num * bsize];
                data_t *xC = xL + node;
                data_t *xR = xL + bsize;
                data_t *K = &x[nx - nparam];

                // gemv<data_t>("N", bsize - node, node, -1.0, A, bsize, xL, 1, 1.0, R, 1);
                DEV::linalg::gemm<data_t>("N", "N", bsize - node, nrhs, node, -1.0, A, bsize, xL, nx, 1.0, R, nx);
                // gemv<data_t>("N", bsize - node, node, -1.0, &A[bsize * bsize], bsize, xR, 1, 1.0, R, 1);
                DEV::linalg::gemm<data_t>("N", "N", bsize - node, nrhs, node, -1.0, &A[bsize * bsize], bsize, xR, nx, 1.0, R, nx);
                // gemv<data_t>("N", bsize - node, nparam, -1.0, C, bsize, K, 1, 1.0, R, 1);
                DEV::linalg::gemm<data_t>("N", "N", bsize - node, nrhs, nparam, -1.0, C, bsize, K, nx, 1.0, R, nx);
                // memcpy(xC, R, (bsize - node) * 1 * sizeof(data_t));
                lacpy("A", bsize - node, nrhs, R, nx, xC, nx);
                trsm<data_t>("L", "U", "N", "N", bsize - node, nrhs, 1.0, LU, bsize, xC, nx);

                for (i = 0; i < (bsize - node); i++) {
                    colperm[i] = i;
                }
                for (i = 0; i < (bsize - node); i++) {
                    std::swap(colperm[i], colperm[jpiv[i]]);
                }

                for (j = 0; j < nrhs; j++) {
                    for (i = 0; i < (bsize - node); i++) {
                        R[colperm[i] + j * nx] = xC[i + j * nx];
                    }
                    for (i = 0; i < (bsize - node); i++) {
                        xC[i + j * nx] = R[i + j * nx];
                    }
                }
            }

            // method for views
            ptrdiff_t ldA() const
            {
                return bsize;
            }

            data_t *ALs() const
            {
                return &A[bsize - node];
            }

            data_t *ARs() const
            {
                return &A[bsize * bsize + bsize - node];
            }

            data_t *Cs() const
            {
                return &C[bsize - node];
            }
        };

        template <typename data_t>
        struct CollocationMatrix;

        template <typename data_t>
        struct SubsystemMatrix
        {
            Partition<data_t> *m_partitions = nullptr;
            ptrdiff_t m_nblocks = 0;
            ptrdiff_t m_cap = 0;
            ptrdiff_t m_node = 0;
            ptrdiff_t m_nparam = 0;
            ptrdiff_t m_ngrid = 0;
            BoundaryConditionsBlocks<data_t> *m_bc;
            data_t *m_prows = nullptr;
            data_t *m_D = nullptr;
            data_t *m_work = nullptr;
            SubsystemMatrix<data_t> *m_child = nullptr;
            FinalBlock<data_t> *m_final = nullptr;
            TrailingBlock<data_t> *m_trailing = nullptr;
            // bool m_factored = false;
            Info m_info;

            SubsystemMatrix() = default;

            ~SubsystemMatrix()
            {
                if (m_partitions) {
                    for (ptrdiff_t i = 0; i < m_cap; i++) {
                        (m_partitions + i)->~Partition();
                    }
                    free(m_partitions);
                    m_partitions = nullptr;
                }
                if (m_prows) {
                    free(m_prows);
                    m_prows = nullptr;
                }
                if (m_D) {
                    free(m_D);
                    m_D = nullptr;
                }
                if (m_work) {
                    free(m_work);
                    m_work = nullptr;
                }
                if (m_child) {
                    delete m_child;
                    m_child = nullptr;
                }
                if (m_final) {
                    delete m_final;
                    m_final = nullptr;
                }
                if (m_trailing) {
                    free(m_trailing);
                    m_trailing = nullptr;
                }
                m_nblocks = 0;
                m_cap = 0;
                m_node = 0;
                m_nparam = 0;
                m_ngrid = 0;
                m_info.clearAll();
            }

            // need constructors from parent ColMat and Subsystem types
            void update(const CollocationMatrix<data_t> *parent)
            {
                ptrdiff_t i;
                const data_t *parent_prows = parent->m_prows;
                const ptrdiff_t parent_stride = parent->m_partitions->ldA();

                // make sure memory has been allocated
                reserve(parent);

                // start copying everything
                // copy prows
                // collocation matrix prows has everything so the
                // stride will be nparam * node * (ncol + 1)
                for (i = 0; i < m_ngrid; i++) {
                    memcpy(&m_prows[i * m_node * m_nparam], &parent_prows[i * m_nparam * parent_stride], m_node * m_nparam * sizeof(data_t));
                }

                // copy D
                memcpy(m_D, parent->m_D, (m_nparam * m_nparam) * sizeof(data_t));

                // fill in blocks
                for (i = 0; i < m_nblocks; i++) {
                    Partition<data_t> *b1 = parent->m_partitions + (2 * i + 0);
                    Partition<data_t> *b2 = parent->m_partitions + (2 * i + 1);
                    Partition<data_t> *partition = m_partitions + i;

                    partition->update(b1, b2, &m_prows[(2 * i + 1) * m_nparam * m_node]);
                }

                m_bc = parent->m_bc;
                m_info.clearFactored();
                // NOTE: trailing already handled in reserve
            }

            void update(const SubsystemMatrix<data_t> *parent)
            {
                ptrdiff_t i, j;
                const data_t *parent_prows = parent->m_prows;

                // make sure memory has been allocated
                reserve(parent);

                // start copying everything

                // copy prows
                for (j = 0, i = 0; i < parent->m_ngrid; i += 2, j++) {
                    memcpy(&m_prows[j * m_node * m_nparam], &parent_prows[(2 * j + 0) * m_nparam * m_node], m_node * m_nparam * sizeof(data_t));
                }
                if ((2 * parent->m_nblocks + parent->hasTrailing()) & 1) {
                    memcpy(&m_prows[(m_ngrid - 1) * m_node * m_nparam], &parent_prows[(parent->m_ngrid - 1) * m_node * m_nparam], m_node * m_nparam * sizeof(data_t));
                }

                // copy D
                memcpy(m_D, parent->m_D, m_nparam * m_nparam * sizeof(data_t));

                // update blocks
                for (i = 0; i < (m_nblocks - (parent->hasTrailing() && !hasTrailing())); i++) {
                    Partition<data_t> *partition = &m_partitions[i];
                    Partition<data_t> *partition1 = &parent->m_partitions[2 * i + 0];
                    Partition<data_t> *partition2 = &parent->m_partitions[2 * i + 1];

                    partition->update(partition1, partition2, &m_prows[(2 * i + 1) * m_node * m_nparam]);
                    ;
                }

                if (hasTrailing() && parent->hasTrailing()) {
                    // just copy the reference since trailing blocks don't
                    // allocate or free memory
                    if (m_trailing == nullptr) {
                        m_trailing = (TrailingBlock<data_t> *)malloc(sizeof(TrailingBlock<data_t>));
                    }
                    m_trailing->AL = parent->m_trailing->AL;
                    m_trailing->AR = parent->m_trailing->AR;
                    m_trailing->C = parent->m_trailing->C;
                    m_trailing->ldA = parent->m_trailing->ldA;
                    m_info.setHasTrailing();
                }
                else if (hasTrailing()) {
                    // extract trailing block from last parent block
                    // NOTE: this is already handled in reserve
                    // so do nothing
                }
                else if (parent->hasTrailing()) {
                    // make last block from parents last block and trailing
                    i = m_nblocks - 1;
                    Partition<data_t> *partition = &m_partitions[i];
                    Partition<data_t> *partition1 = &parent->m_partitions[2 * i];
                    partition->update(partition1, parent->m_trailing, &m_prows[(2 * i + 1) * m_node * m_nparam]);
                }
                else {
                    // nothing to do
                }

                m_bc = parent->m_bc;
                m_info.clearFactored();
            }

            // methods for reserving memory so factor/solve methods
            void reserve(const CollocationMatrix<data_t> *parent)
            {
                ptrdiff_t i, ngrid;
                const ptrdiff_t node = parent->m_partitions->node;
                const ptrdiff_t nparam = parent->m_partitions->nparam;

                // determine number of blocks and grid points
                auto [nblocks, has_trailing] = std::div(parent->m_nblocks, 2l);
                ngrid = 2 * nblocks + 1 + has_trailing;

                if ((ngrid == m_ngrid) && (nblocks == m_nblocks) && (node == m_node) && (nparam == m_nparam) && (has_trailing == hasTrailing())) {
                    if (has_trailing) {
                        Partition<data_t> *partition = parent->m_partitions + (parent->m_nblocks - 1);
                        if (m_trailing == nullptr) {
                            m_trailing = (TrailingBlock<data_t> *)malloc(sizeof(TrailingBlock<data_t>));
                        }
                        m_trailing->AL = partition->ALs();
                        m_trailing->AR = partition->ARs();
                        m_trailing->C = partition->Cs();
                        m_trailing->ldA = partition->ldA();
                        m_info.setHasTrailing();
                    }
                    else {
                        m_info.clearHasTrailing();
                    }
                    // memory already allocated so we can quick return
                    return;
                }
                // fmt::println("({:3d}, {:1d}) : {:3d}", nblocks, has_trailing, ngrid);

                // allocate number of blocks
                if (m_cap < nblocks) {
                    // if capacity not big enough realloc
                    m_partitions = (Partition<data_t> *)realloc(m_partitions, nblocks * sizeof(Partition<data_t>));
                    for (i = m_cap; i < nblocks; i++) {
                        Partition<data_t> *partition = &m_partitions[i];
                        *partition = std::move(Partition<data_t>());
                    }
                    m_cap = nblocks;
                }
                for (i = 0; i < nblocks; i++) {
                    Partition<data_t> *partition = &m_partitions[i];
                    partition->emptyPartition(1, node, nparam);
                }

                // allocate prows
                m_prows = (data_t *)realloc(m_prows, ngrid * node * nparam * sizeof(data_t));

                // allocate D
                if ((m_D == nullptr) || (m_nparam != nparam)) {
                    m_D = (data_t *)realloc(m_D, nparam * nparam * sizeof(data_t));
                }

                // update constants
                m_nblocks = nblocks;
                m_node = node;
                m_nparam = nparam;
                m_ngrid = ngrid;

                // handle trailing
                if (has_trailing) {
                    Partition<data_t> *partition = parent->m_partitions + (parent->m_nblocks - 1);
                    if (m_trailing == nullptr) {
                        m_trailing = (TrailingBlock<data_t> *)malloc(sizeof(TrailingBlock<data_t>));
                    }
                    m_trailing->AL = partition->ALs();
                    m_trailing->AR = partition->ARs();
                    m_trailing->C = partition->Cs();
                    m_trailing->ldA = partition->ldA();
                    m_info.setHasTrailing();
                }
                else {
                    m_info.clearHasTrailing();
                }

                // update subsystem
                if (ngrid != 3) {
                    // create child
                    if (m_child == nullptr) {
                        m_child = new SubsystemMatrix<data_t>();
                    }
                    m_info.setHasChild();
                    m_child->reserve(this);
                }
                else {
                    // possible create final
                    m_info.clearHasChild();
                    if (m_final == nullptr) {
                        m_final = new FinalBlock<data_t>(m_node, m_nparam);
                    }
                    else {
                        m_final->reserve(m_node, m_nparam);
                    }
                }

                // reallocate work array
                resizeWork();
            }

            void reserve(const SubsystemMatrix<data_t> *parent)
            {
                ptrdiff_t i, ngrid;

                const ptrdiff_t node = parent->m_partitions->node;
                const ptrdiff_t nparam = parent->m_partitions->nparam;

                // determine number of blocks and grid points
                auto [nblocks, has_trailing] = std::div(((parent->m_ngrid - 1) >> 1) + parent->hasTrailing(), 2l);
                ngrid = 2 * nblocks + 1 + has_trailing;

                if ((ngrid == m_ngrid) && (nblocks == m_nblocks) && (node == m_node) && (nparam == m_nparam) && (has_trailing == hasTrailing())) {
                    // memory already allocated so we can quick return
                    // handle trailing
                    if (has_trailing) {
                        Partition<data_t> *partition = parent->m_partitions + (parent->m_nblocks - 1);
                        if (m_trailing == nullptr) {
                            m_trailing = (TrailingBlock<data_t> *)malloc(sizeof(TrailingBlock<data_t>));
                        }
                        m_trailing->AL = partition->ALs();
                        m_trailing->AR = partition->ARs();
                        m_trailing->C = partition->Cs();
                        m_trailing->ldA = partition->ldA();
                        m_info.setHasTrailing();
                    }
                    else {
                        m_info.clearHasTrailing();
                    }
                    return;
                }
                // fmt::println("({:3d}, {:1d}) : {:3d}", nblocks, has_trailing, ngrid);

                // allocate number of blocks
                if (m_cap < nblocks) {
                    // if capacity not big enough realloc
                    m_partitions = (Partition<data_t> *)realloc(m_partitions, nblocks * sizeof(Partition<data_t>));
                    for (i = m_cap; i < nblocks; i++) {
                        Partition<data_t> *partition = &m_partitions[i];
                        *partition = std::move(Partition<data_t>());
                    }
                    m_cap = nblocks;
                }
                for (i = 0; i < nblocks; i++) {
                    Partition<data_t> *partition = &m_partitions[i];
                    partition->emptyPartition(1, node, nparam);
                }

                // allocate prows
                m_prows = (data_t *)realloc(m_prows, ngrid * node * nparam * sizeof(data_t));

                // allocate D
                if ((m_D == nullptr) || (m_nparam != nparam)) {
                    m_D = (data_t *)realloc(m_D, nparam * nparam * sizeof(data_t));
                }

                // update constants
                m_nblocks = nblocks;
                m_node = node;
                m_nparam = nparam;
                m_ngrid = ngrid;

                // handle trailing
                if (has_trailing) {
                    Partition<data_t> *partition = parent->m_partitions + (parent->m_nblocks - 1);
                    if (m_trailing == nullptr) {
                        m_trailing = (TrailingBlock<data_t> *)malloc(sizeof(TrailingBlock<data_t>));
                    }
                    m_trailing->AL = partition->ALs();
                    m_trailing->AR = partition->ARs();
                    m_trailing->C = partition->Cs();
                    m_trailing->ldA = partition->ldA();
                    m_info.setHasTrailing();
                }
                else {
                    m_info.clearHasTrailing();
                }

                // update subsystem
                if (ngrid != 3) {
                    // create child
                    if (m_child == nullptr) {
                        m_child = new SubsystemMatrix<data_t>();
                    }
                    m_info.setHasChild();
                    m_child->reserve(this);
                }
                else {
                    // possible create final
                    m_info.clearHasChild();
                    if (m_final == nullptr) {
                        m_final = new FinalBlock<data_t>(m_node, m_nparam);
                    }
                    else {
                        m_final->reserve(m_node, m_nparam);
                    }
                }

                // reallocate work array
                resizeWork();
            }

            void resizeWork(const ptrdiff_t nrhs = 1)
            {
                m_work = (data_t *)realloc(m_work, nrhs * 3 * (m_node * m_ngrid + m_nparam) * sizeof(data_t));
            }

            void updateJustBC(const BoundaryConditionsBlocks<data_t> *bc)
            {
                memcpy(m_bc->BC0, bc->BC0, m_node * (2 * m_node + m_nparam) * sizeof(data_t));

                if (m_ngrid == 3) {
                    m_final->update(m_node, m_nparam, m_partitions->ALs(), m_partitions->ARs(), m_partitions->Cs(), bc, m_prows, &m_prows[2 * m_nparam * m_node], m_D);
                    m_final->factor();
                }
                else {
                    m_child->updateJustBC(bc);
                }
            }

            // factor and solve methods
            void factor()
            {
                Partition<data_t> *block;
                ptrdiff_t i, j, k;
                data_t *prow0, *prow1;

                if (m_info.getFactored()) {
                    return;
                }

#pragma omp parallel for private(i, block) num_threads(std::min(2, omp_get_num_threads()))
                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];
                    block->condense();
                }

                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];

                    prow0 = &m_prows[2 * (i + 0) * m_node * m_nparam];
                    prow1 = &m_prows[2 * (i + 1) * m_node * m_nparam];

                    // update parameter rows
                    for (j = 0; j < m_node; j++) {
                        for (k = 0; k < m_nparam; k++) {
                            prow0[j * m_nparam + k] += block->B[(j + 0 * m_node) * m_nparam + k];
                            prow1[j * m_nparam + k] += block->B[(j + 2 * m_node) * m_nparam + k];
                        }
                    }
                    // update D block
                    for (j = 0; j < m_nparam * m_nparam; j++) {
                        m_D[j] += block->D[j];
                    }
                }

                if (m_ngrid == 3) {
                    // update and factor
                    m_final->update(m_node, m_nparam, m_partitions->ALs(), m_partitions->ARs(), m_partitions->Cs(), m_bc, m_prows, &m_prows[2 * m_nparam * m_node], m_D);
                    m_final->factor();
                }
                else {
                    m_child->update(this);
                    m_child->factor();
                }

                // m_factored = true;
                m_info.setFactored();
            }

            bool hasTrailing() const
            {
                return m_info.getHasTrailing();
            }

            void solve(const data_t *b, data_t *x)
            {
                ptrdiff_t i;
                data_t *y, *v, *w;
                Partition<data_t> *block;
                const ptrdiff_t N = m_node * m_ngrid + m_nparam;
                const ptrdiff_t Ns = m_info.getHasChild() ? (m_node * (m_child->m_ngrid) + m_nparam) : (2 * m_node + m_nparam);

                y = m_work;
                v = y + N;
                w = v + N;

                memcpy(x, b, N * sizeof(data_t));

                factor();

                // condense rhs and form rhs to subsystem
                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];
                    block->condenseVector(x, N, i);
                    memcpy(&v[i * m_node], &x[(2 * i + 1) * m_node], m_node * sizeof(data_t));
                }
                // handle trailing block
                if (hasTrailing()) {
                    memcpy(&v[m_nblocks * m_node], &x[N - 2 * m_node - m_nparam], (2 * m_node + m_nparam) * sizeof(data_t));
                }
                else {
                    memcpy(&v[m_nblocks * m_node], &x[N - m_node - m_nparam], (m_node + m_nparam) * sizeof(data_t));
                }

                // preserve reduced rhs for backsub later
                memcpy(y, x, N * sizeof(data_t));

                // solve child or final block
                if (m_ngrid != 3) {
                    m_child->solve(v, w);
                }
                else {
                    m_final->solve(v, w);
                }

                // get subsytem solution
                for (i = 0; i < m_nblocks; i++) {
                    memcpy(&x[2 * i * m_node], &w[i * m_node], m_node * sizeof(data_t));
                }
                // get right end point and parameters
                if (!hasTrailing()) {
                    memcpy(&x[N - m_nparam - m_node], &w[Ns - m_node - m_nparam], (m_node + m_nparam) * sizeof(data_t));
                }
                else {
                    // handle for trailing
                    memcpy(&x[N - m_nparam - 2 * m_node], &w[Ns - 2 * m_node - m_nparam], (2 * m_node + m_nparam) * sizeof(data_t));
                }

                // backsub to get remaining solution
                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];
                    block->backsubstitute(x, y, N, i);
                }
            }

            void solve(const data_t *b, data_t *x, const ptrdiff_t nrhs)
            {
                ptrdiff_t i;
                data_t *y, *v, *w;
                Partition<data_t> *block;
                const ptrdiff_t N = m_node * m_ngrid + m_nparam;
                const ptrdiff_t Ns = m_info.getHasChild() ? (m_node * (m_child->m_ngrid) + m_nparam) : (2 * m_node + m_nparam);

                resizeWork(nrhs); // make sure workspace is large enough for nrhs

                y = m_work;
                v = y + N * nrhs;
                w = v + N * nrhs;

                memcpy(x, b, nrhs * N * sizeof(data_t));

                factor();

                // condense rhs and form rhs to subsystem
                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];
                    block->condenseVector(nrhs, x, N, i);
                    lacpy("A", m_node, nrhs, &x[(2 * i + 1) * m_node], N, &v[i * m_node], Ns);
                }
                // handle trailing block
                if (hasTrailing()) {
                    lacpy("A", 2 * m_node + m_nparam, nrhs, &x[N - 2 * m_node - m_nparam], N, &v[m_nblocks * m_node], Ns);
                }
                else {
                    lacpy("A", m_node + m_nparam, nrhs, &x[N - m_node - m_nparam], N, &v[m_nblocks * m_node], Ns);
                }

                // preserve reduced rhs for backsub later
                memcpy(y, x, nrhs * N * sizeof(data_t));

                // solve child or final block
                if (m_ngrid != 3) {
                    m_child->solve(v, w, nrhs);
                }
                else {
                    m_final->solve(v, w, nrhs, Ns);
                }

                // get subsytem solution
                for (i = 0; i < m_nblocks; i++) {
                    // memcpy(&x[2 * i * m_node], &w[i * m_node], m_node * sizeof(data_t));
                    lacpy("A", m_node, nrhs, &w[i * m_node], Ns, &x[2 * i * m_node], N);
                }
                // get right end point and parameters
                if (!hasTrailing()) {
                    lacpy("A", m_node + m_nparam, nrhs, &w[Ns - m_nparam - m_node], Ns, &x[N - m_nparam - m_node], N);
                }
                else {
                    // handle for trailing
                    lacpy("A", 2 * m_node + m_nparam, nrhs, &w[Ns - m_nparam - 2 * m_node], Ns, &x[N - m_nparam - 2 * m_node], N);
                }

// backsub to get remaining solution
#pragma omp parallel for private(i, block)
                for (i = 0; i < m_nblocks; i++) {
                    block = &m_partitions[i];
                    block->backsubstitute(nrhs, x, y, N, i);
                }
            }
        };

        template <typename data_t>
        struct CollocationMatrix
        {
            ptrdiff_t m_nblocks = 0;
            ptrdiff_t m_cap = 0;
            Partition<data_t> *m_partitions = nullptr;
            Partition<data_t> *m_partitions_mm = nullptr;
            BoundaryConditionsBlocks<data_t> *m_bc = nullptr;
            SubsystemMatrix<data_t> *m_S = nullptr;
            data_t *m_V_mm = nullptr;
            data_t *m_prows = nullptr;
            data_t *m_D = nullptr;
            data_t *m_work = nullptr;
            Info m_info;

            CollocationMatrix() = default;

            ~CollocationMatrix()
            {
                ptrdiff_t i;
                if (m_partitions) {
                    for (i = 0; i < m_cap; i++) {
                        (m_partitions + i)->~Partition();
                    }
                    free(m_partitions);
                    m_partitions = NULL;
                }
                if (m_partitions_mm) {
                    for (i = 0; i < m_cap; i++) {
                        (m_partitions_mm + i)->~Partition();
                    }
                    free(m_partitions_mm);
                    m_partitions_mm = NULL;
                }
                if (m_work) {
                    free(m_work);
                    m_work = nullptr;
                }
                if (m_prows) {
                    free(m_prows);
                    m_prows = nullptr;
                }
                if (m_D) {
                    free(m_D);
                    m_D = nullptr;
                }
                if (m_S) {
                    delete m_S;
                    m_S = nullptr;
                }
                if (m_V_mm) {
                    free(m_V_mm);
                    m_V_mm = nullptr;
                }
                m_nblocks = 0;
                m_cap = 0;
                m_info.clearAll();
            }

            void updateJustBC(const BoundaryConditionsBlocks<data_t> *bc)
            {
                const ptrdiff_t node = m_partitions->node;
                const ptrdiff_t nparam = m_partitions->nparam;
                memcpy(m_bc->BC0, bc->BC0, node * (2 * node + nparam) * sizeof(data_t));
                m_S->updateJustBC(bc);
            }

            void update(func_t fjac, pjac_t pjac, const ptrdiff_t ngrid, const ptrdiff_t node, const ptrdiff_t ncol, const ptrdiff_t nparam, const double L, const double *h, const double *y, const double *p, const int *pmask, const double *V, BoundaryConditionsBlocks<data_t> *bc)
            {
                // throw std::runtime_error("CollocationMatrix::update not implemented");
                ptrdiff_t i, nblocks;
                const double *Yi;
                const double *prowi;

                nblocks = ngrid - 1;

                // reallocate memory if needed
                if ((nblocks != m_nblocks) || (node != m_partitions->node) || (ncol != ((m_partitions->bsize / node) - 1)) || (nparam != m_partitions->nparam)) {
                    // blocks
                    if (m_cap < nblocks) {
                        m_partitions = (Partition<data_t> *)realloc(m_partitions, nblocks * sizeof(Partition<data_t>));
                        m_partitions_mm = (Partition<data_t> *)realloc(m_partitions_mm, nblocks * sizeof(Partition<data_t>));
                        for (i = m_cap; i < nblocks; i++) {
                            m_partitions[i] = std::move(Partition<data_t>());
                            m_partitions_mm[i] = std::move(Partition<data_t>());
                        }
                        m_cap = nblocks;
                    }
                    m_nblocks = nblocks;
                    for (i = 0; i < nblocks; i++) {
                        Partition<data_t> *partition = &m_partitions[i];
                        Partition<data_t> *partition_mm = &m_partitions_mm[i];
                        partition->emptyPartition(ncol, node, nparam);
                        partition_mm->emptyPartition(ncol, node, nparam);
                    }

                    // prows
                    m_prows = (data_t *)realloc(m_prows, nparam * (nblocks * node * (ncol + 1) + node) * sizeof(data_t));
                    // D
                    m_D = (data_t *)realloc(m_D, nparam * nparam * sizeof(data_t));
                    m_V_mm = (data_t *)realloc(m_V_mm, nparam * size() * sizeof(data_t));
                    // work
                    resizeWork();
                }

                m_bc = bc;
                if (nparam != 0) {
                    std::copy(V, &V[(size() - nparam) * nparam], m_prows);
                    std::copy(&V[(size() - nparam) * nparam], &V[size() * nparam], m_D);
                    std::copy(V, &V[size() * nparam], m_V_mm);
                }

// go through and update each block
#pragma omp parallel for private(i, Yi, prowi) if (node > 5)
                for (i = 0; i < m_nblocks; i++) {
                    double *local_work = ((double *)m_work) + (omp_get_thread_num() * ncol * node * node);
                    Partition<data_t> *partition = &m_partitions[i];
                    Partition<data_t> *partition_mm = &m_partitions_mm[i];
                    Yi = &y[i * node * (ncol + 1) + node];
                    prowi = &V[i * nparam * (node * (ncol + 1)) + node * nparam];
                    partition->update(fjac, pjac, node, ncol, nparam, L * h[i], Yi, p, pmask, prowi, local_work);
                    partition_mm->fill(*partition);
                }

                // factored = false;
                m_info.clearFactored();
            }

            void factor()
            {
                TWIST_CXX_MARK_FUNCTION;
                ptrdiff_t i, j, k, node, nparam, bsize;
                data_t *prow1, *prow2;
                OpenBLASThreadContext __openblas_context(1);

                if (m_info.getFactored()) {
                    return;
                }

                node = m_partitions->node;
                nparam = m_partitions->nparam;
                bsize = m_partitions->ldA();

                TWIST_MARK_BEGIN("condense");

#pragma omp parallel for private(i)
                for (i = 0; i < m_nblocks; i++) {
                    Partition<data_t> *partition = &m_partitions[i];
                    partition->condense();
                    // partition->display();
                }
                TWIST_MARK_END("condense");

                TWIST_MARK_BEGIN("update overlap");
                // need to benchmark this under openmp critical section
                for (i = 0; i < m_nblocks; i++) {
                    // update prows and D
                    Partition<data_t> *partition = &m_partitions[i];
                    prow1 = &m_prows[(i + 0) * nparam * bsize];
                    prow2 = &m_prows[(i + 1) * nparam * bsize];
                    for (j = 0; j < node; j++) {
                        for (k = 0; k < nparam; k++) {
                            prow1[j * nparam + k] += partition->B[(j + 0 * bsize) * nparam + k];
                            prow2[j * nparam + k] += partition->B[(j + 1 * bsize) * nparam + k];
                        }
                    }
                    for (j = 0; j < (nparam * nparam); j++) {
                        m_D[j] += partition->D[j];
                    }
                }
                TWIST_MARK_END("update overlap");

                if (m_S == nullptr) {
                    m_S = new SubsystemMatrix<data_t>();
                }
                TWIST_MARK_BEGIN("main S update");
                m_S->update(this);
                TWIST_MARK_END("main S update");
                TWIST_MARK_BEGIN("main S factor");
                m_S->factor();
                TWIST_MARK_END("main S factor");

                // factored = true;
                m_info.setFactored();
            }

            void resizeWork(const ptrdiff_t nrhs = 1)
            {
                ptrdiff_t lwork = getLWork(nrhs);
                ptrdiff_t update_lwork = omp_get_max_threads() * m_partitions->node * m_partitions->ldA();
                lwork = std::max(lwork, update_lwork);
                // lwork = std::max(lwork, omp_get_max_threads() * ncol * node * std::max(node, nparam));

                // work needs to be 2 * N + 2 * M
                m_work = (data_t *)realloc(m_work, lwork * sizeof(data_t));
            }

            ptrdiff_t size() const
            {
                const ptrdiff_t nparam = m_partitions->nparam;
                const ptrdiff_t node = m_partitions->node;
                const ptrdiff_t bsize = m_partitions->bsize;

                return bsize * m_nblocks + node + nparam;
            }

            void solve(const data_t *b, data_t *x)
            {
                TWIST_CXX_MARK_FUNCTION;
                ptrdiff_t i;
                data_t *v, *w, *y;
                int niters;

                const ptrdiff_t nparam = m_partitions->nparam;
                const ptrdiff_t node = m_partitions->node;
                const ptrdiff_t bsize = m_partitions->bsize;
                const ptrdiff_t N = size();
                const ptrdiff_t Ns = (m_nblocks + 1) * node + nparam;

                OpenBLASThreadContext __openblas_context(1);

                v = m_work;
                w = v + N;
                y = w + N;

                // factor if not already
                factor();

                // copy rhs into x
                memcpy(x, b, N * sizeof(data_t));

                // condense x
                for (i = 0; i < m_nblocks; i++) {
                    const Partition<data_t> &partition = m_partitions[i];
                    partition.condenseVector(x, N, i);
                    // copy condensed rhs node section into subsystem RHS
                    memcpy(&v[i * node], &x[(i + 1) * bsize - node], node * sizeof(data_t));
                }
                // don't forget last node and parameter RHS
                memcpy(&v[Ns - node - nparam], &x[N - node - nparam], (node + nparam) * sizeof(data_t));
                // copy current RHS into backup vector y
                memcpy(y, x, N * sizeof(data_t));

                // solve subsystem Sw = v
                m_S->solve(v, w);

                // NOTE: sometimes when nparam > 1, solver will need 1 iteration of BiCGStab
                // This is likely a bug in the code somewhere, but I wasn't able to track it
                // down and this is chdeap and it seems (for my tests) to have a universal
                // iteration bound of one
                auto A_Op = [this](const data_t *in, data_t *out, size_t) {
                    sgemv(0.0, out, 1.0, in);
                    return 0;
                };
                auto M_Op = [this](const data_t *in, data_t *out, size_t) {
                    m_S->solve(in, out);
                    return 0;
                };
                Krylov::bicgstab<decltype(A_Op), decltype(M_Op), data_t>(A_Op, M_Op, Ns, w, v, &m_work[3 * N], 10, Ns * std::pow(0.5, 52), std::pow(0.5, 104), std::pow(0.5, 104), &niters, false);

                // copy results into x
                for (i = 0; i < (m_nblocks + 1); i++) {
                    memcpy(&x[i * bsize], &w[i * node], node * sizeof(data_t));
                }
                memcpy(&x[N - nparam], &w[Ns - nparam], nparam * sizeof(data_t));

                // back substitution on final blocks and inverse column swaps
#pragma omp parallel for private(i) firstprivate(x, y, N)
                for (i = 0; i < m_nblocks; i++) {
                    const Partition<data_t> &partition = m_partitions[i];
                    partition.backsubstitute(x, y, N, i);
                }
            }

            ptrdiff_t getLWork(const ptrdiff_t nrhs = 1) const
            {
                const ptrdiff_t nparam = m_partitions->nparam;
                const ptrdiff_t node = m_partitions->node;
                const ptrdiff_t M = size();
                const ptrdiff_t N = (m_nblocks + 1) * node + nparam;
                if (nrhs > 1) {
                    using fake_t = int (*)(const data_t *, data_t *, size_t, size_t);
                    int lwork = -1;
                    Krylov::block_bicgstab<fake_t, fake_t, data_t>(NULL, NULL, size(), nrhs, NULL, NULL, NULL, lwork, 0, 0.0, 0.0, NULL, false);
                    return nrhs * (10 * N + 3 * M) + lwork;
                }
                return nrhs * (10 * N + 3 * M);
            }

            void solve(const data_t *b, data_t *x, const ptrdiff_t nrhs)
            {
                ptrdiff_t i;
                data_t *v, *w, *y;
                int niters;

                const ptrdiff_t nparam = m_partitions->nparam;
                const ptrdiff_t node = m_partitions->node;
                const ptrdiff_t bsize = m_partitions->bsize;
                const ptrdiff_t N = size();
                const ptrdiff_t Ns = (m_nblocks + 1) * node + nparam;
                int lwork = getLWork(nrhs) - 3 * N * nrhs;
                OpenBLASThreadContext __openblas_context(1);

                if (nrhs == 1) {
                    // block bicgstab will probably fail for nrhs = 1
                    return solve(b, x);
                }

                resizeWork(nrhs);

                v = m_work;
                w = v + N * nrhs;
                y = w + N * nrhs;

                // factor if not already
                factor();

                // copy rhs into x
                memcpy(x, b, nrhs * N * sizeof(data_t));

                // condense x
                for (i = 0; i < m_nblocks; i++) {
                    const Partition<data_t> &partition = m_partitions[i];
                    partition.condenseVector(nrhs, x, N, i);
                    // copy condensed rhs node section into subsystem RHS
                    // memcpy(&v[i * node], &x[(i + 1) * bsize - node], node * sizeof(data_t));
                    lacpy("A", node, nrhs, &x[(i + 1) * bsize - node], N, &v[i * node], Ns);
                }
                // don't forget last node and parameter RHS
                // memcpy(&v[N - node - nparam], &x[M - node - nparam], (node + nparam) * sizeof(data_t));
                lacpy("A", node + nparam, nrhs, &x[N - node - nparam], N, &v[Ns - node - nparam], Ns);
                // copy current RHS into backup vector y
                memcpy(y, x, nrhs * N * sizeof(data_t));

                // solve subsystem Sw = v
                m_S->solve(v, w, nrhs);

                // NOTE: sometimes when nparam > 1, solver will need 1 iteration of BiCGStab
                // This is likely a bug in the code somewhere, but I wasn't able to track it
                // down and this is chdeap and it seems (for my tests) to have a universal
                // iteration bound of one
                auto A_Op = [this](const data_t *in, data_t *out, size_t, size_t nrhs) {
                    // sgemv(0.0, out, 1.0, in);
                    sgemm(nrhs, 0.0, out, 1.0, in);
                    return 0;
                };
                auto M_Op = [this](const data_t *in, data_t *out, size_t, size_t nrhs) {
                    m_S->solve(in, out, nrhs);
                    return 0;
                };
                // block_bicgstab<decltype(A_Op), decltype(M_Op), data_t>(A_Op, M_Op, N, nrhs, w, v, &m_work[3 * M], lwork, 10, N * std::pow(0.5, 52), std::pow(0.5, 104), std::pow(0.5, 104), &niters, true);
                for (i = 0; i < nrhs; i += 32) {
                    Krylov::block_bicgstab<decltype(A_Op), decltype(M_Op), data_t>(A_Op, M_Op, Ns, std::min<int>(32, nrhs - i), &w[i * Ns], &v[i * Ns], &m_work[3 * N * nrhs], lwork, 10, Ns * pow(0.5, 52), pow(0.5, 104), &niters, false);
                }
                // block_bicgstab<decltype(A_Op), decltype(M_Op), data_t>(A_Op, M_Op, Ns, nrhs, w, v, &m_work[3 * N * nrhs], lwork, 10, Ns * pow(0.5, 52), pow(0.5, 104), &niters, true);

                // copy results into x
                for (i = 0; i < (m_nblocks + 1); i++) {
                    // memcpy(&x[i * bsize], &w[i * node], node * sizeof(data_t));
                    lacpy("A", node, nrhs, &w[i * node], Ns, &x[i * bsize], N);
                }
                // memcpy(&x[M - nparam], &w[N - nparam], nparam * sizeof(data_t));
                lacpy("A", nparam, nrhs, &w[Ns - nparam], Ns, &x[N - nparam], N);

                // back substitution on final blocks and inverse column swaps
#pragma omp parallel for private(i)
                for (i = 0; i < m_nblocks; i++) {
                    const Partition<data_t> &partition = m_partitions[i];
                    // partition.backsubstitute(x, y, M, i);
                    partition.backsubstitute(nrhs, x, y, N, i);
                }
            }

            void gemv(const data_t beta, data_t *y, const data_t alpha, const data_t *x) const
            {
                Partition<data_t> *partition;
                int nrowA, ncolA, ncolC, node;
                ptrdiff_t i, N;

                node = m_partitions->node;
                nrowA = m_partitions->bsize;
                ncolA = nrowA + node;
                ncolC = m_partitions->nparam;

                N = size();

                // main blocks
                for (i = 0; i < m_nblocks; i++) {
                    partition = &m_partitions_mm[i];
                    DEV::linalg::gemv<data_t>("N", nrowA, ncolA, alpha, partition->A, nrowA, &x[i * nrowA], 1, beta, &y[i * nrowA], 1);
                    DEV::linalg::gemv<data_t>("N", nrowA, ncolC, alpha, partition->C, nrowA, &x[N - ncolC], 1, 1.0, &y[i * nrowA], 1);
                }

                // boundary conditions
                DEV::linalg::gemv<data_t>("N", node, node, alpha, m_bc->BC0, node, x, 1, beta, &y[N - node - ncolC], 1);
                DEV::linalg::gemv<data_t>("N", node, node, alpha, m_bc->BC1, node, &x[N - node - ncolC], 1, 1.0, &y[N - node - ncolC], 1);
                DEV::linalg::gemv<data_t>("N", node, ncolC, alpha, m_bc->BCp, node, &x[N - ncolC], 1, 1.0, &y[N - node - ncolC], 1);

                // last rows
                DEV::linalg::gemv<data_t>("N", ncolC, N, alpha, m_V_mm, ncolC, x, 1, beta, &y[N - ncolC], 1);
            }

            void gemm(const ptrdiff_t nrhs, const data_t beta, data_t *y, const data_t alpha, const data_t *x) const
            {
                Partition<data_t> *partition;
                int nrowA, ncolA, ncolC, node;
                ptrdiff_t i, N, bc_idx;

                node = m_partitions->node;
                nrowA = m_partitions->bsize;
                ncolA = nrowA + node;
                ncolC = m_partitions->nparam;

                N = size();
                bc_idx = N - node - ncolC;

                // main blocks
                for (i = 0; i < m_nblocks; i++) {
                    partition = &m_partitions_mm[i];
                    DEV::linalg::gemm<data_t>("N", "N", nrowA, nrhs, ncolA, alpha, partition->A, nrowA, &x[i * nrowA], N, beta, &y[i * nrowA], N);
                    DEV::linalg::gemm<data_t>("N", "N", nrowA, nrhs, ncolC, alpha, partition->C, nrowA, &x[N - ncolC], N, 1.0, &y[i * nrowA], N);
                }

                // boundary conditions
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, m_bc->BC0, node, x, N, beta, &y[bc_idx], N);
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, m_bc->BC1, node, &x[bc_idx], N, 1.0, &y[bc_idx], N);
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, ncolC, alpha, m_bc->BCp, node, &x[N - ncolC], N, 1.0, &y[bc_idx], N);

                // last rows
                DEV::linalg::gemm<data_t>("N", "N", ncolC, nrhs, N, alpha, m_V_mm, ncolC, x, N, beta, &y[N - ncolC], N);
            }

            void sgemv(const data_t beta, data_t *y, const data_t alpha, const data_t *x) const
            {
                Partition<data_t> *partition;
                const data_t *x0, *x1, *xp;
                int node, ldX, nparam;
                ptrdiff_t i, N;

                node = m_partitions->node;
                ldX = m_partitions->ldA();
                nparam = m_partitions->nparam;

                N = (m_nblocks + 1) * node + nparam;
                xp = &x[N - nparam];

                // main blocks
                for (i = 0; i < m_nblocks; i++) {
                    partition = &m_partitions[i];
                    x0 = &x[(i + 0) * node];
                    x1 = &x[(i + 1) * node];

                    DEV::linalg::gemv<data_t>("N", node, node, alpha, partition->ALs(), ldX, x0, 1, beta, &y[i * node], 1);
                    DEV::linalg::gemv<data_t>("N", node, node, alpha, partition->ARs(), ldX, x1, 1, 1.0, &y[i * node], 1);
                    DEV::linalg::gemv<data_t>("N", node, nparam, alpha, partition->Cs(), ldX, xp, 1, 1.0, &y[i * node], 1);
                }

                // boundary conditions
                DEV::linalg::gemv<data_t>("N", node, node, alpha, m_bc->BC0, node, x, 1, beta, &y[N - node - nparam], 1);
                DEV::linalg::gemv<data_t>("N", node, node, alpha, m_bc->BC1, node, &x[N - node - nparam], 1, 1.0, &y[N - node - nparam], 1);
                DEV::linalg::gemv<data_t>("N", node, nparam, alpha, m_bc->BCp, node, &x[N - nparam], 1, 1.0, &y[N - node - nparam], 1);

                // last rows
                for (i = 0; i < nparam; i++) {
                    y[N - nparam + i] *= beta;
                }
                for (i = 0; i < (m_nblocks + 1); i++) {
                    DEV::linalg::gemv<data_t>("N", nparam, node, alpha, &m_prows[i * ldX * nparam], nparam, &x[i * node], 1, 1.0, &y[N - nparam], 1);
                }
                DEV::linalg::gemv<data_t>("N", nparam, nparam, alpha, m_D, nparam, &x[N - nparam], 1, 1.0, &y[N - nparam], 1);
            }

            void sgemm(const ptrdiff_t nrhs, const data_t beta, data_t *y, const data_t alpha, const data_t *x) const
            {
                Partition<data_t> *partition;
                const data_t *x0, *x1, *xp;
                int node, ldX, nparam;
                ptrdiff_t i, N;

                node = m_partitions->node;
                ldX = m_partitions->ldA();
                nparam = m_partitions->nparam;

                N = (m_nblocks + 1) * node + nparam;
                xp = &x[N - nparam];

                // main blocks
                for (i = 0; i < m_nblocks; i++) {
                    partition = &m_partitions[i];
                    x0 = &x[(i + 0) * node];
                    x1 = &x[(i + 1) * node];

                    DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, partition->ALs(), ldX, x0, N, beta, &y[i * node], N);
                    DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, partition->ARs(), ldX, x1, N, 1.0, &y[i * node], N);
                    DEV::linalg::gemm<data_t>("N", "N", node, nrhs, nparam, alpha, partition->Cs(), ldX, xp, N, 1.0, &y[i * node], N);
                }

                // boundary conditions
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, m_bc->BC0, node, x, N, beta, &y[N - node - nparam], N);
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, node, alpha, m_bc->BC1, node, &x[N - node - nparam], N, 1.0, &y[N - node - nparam], N);
                DEV::linalg::gemm<data_t>("N", "N", node, nrhs, nparam, alpha, m_bc->BCp, node, &x[N - nparam], N, 1.0, &y[N - node - nparam], N);

                // last rows
                for (i = 0; i < (m_nblocks + 1); i++) {
                    data_t current_beta = (i == 0) ? beta : 1.0;
                    DEV::linalg::gemm<data_t>("N", "N", nparam, nrhs, node, alpha, &m_prows[i * ldX * nparam], nparam, &x[i * node], N, current_beta, &y[N - nparam], N);
                }
                DEV::linalg::gemm<data_t>("N", "N", nparam, nrhs, nparam, alpha, m_D, nparam, &x[N - nparam], N, 1.0, &y[N - nparam], N);
            }
        };
    } // namespace internal

    typedef internal::CollocationMatrix<double> Matrix;
    typedef internal::CollocationMatrix<std::complex<double>> ComplexMatrix;

    typedef BoundaryConditionsBlocks<double> BoundaryConditions;
    typedef BoundaryConditionsBlocks<std::complex<double>> ComplexBoundaryConditions;
} // namespace DEV