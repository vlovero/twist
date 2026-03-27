#include "collocation_matrix.h"

inline void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const double alpha, const RP(double) A, const int lda, RP(double) B, const int ldb)
{
    if ((m == 0) || (n == 0)) {
        return;
    }
    dtrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void dlaswpc(const int n, RP(double) a, const int lda, const int k1, const int k2, const int *ipiv, const int)
{
    int i;
    ptrdiff_t j;
    for (i = k1; i < k2; i++) {
        for (j = 0; j < n; j++) {
            std::swap(a[i * lda + j], a[ipiv[i] * lda + j]);
        }
    }
}

void dlaswpr(const int n, RP(double) a, const int lda, const int k1, const int k2, const int *ipiv, const int)
{
    int i;
    ptrdiff_t j;
    for (j = 0; j < n; j++) {
        for (i = k1; i < k2; i++) {
            std::swap(a[j * lda + i], a[j * lda + ipiv[i]]);
        }
    }
}

void dlarook(const ptrdiff_t n, const ptrdiff_t m, const RP(double) A, const ptrdiff_t ldA, ptrdiff_t *ipiv, ptrdiff_t *jpiv)
{
    int i;
    double dmax, tmp;
    ptrdiff_t c0, r0, c1, r1, j;

    r0 = 0;
    c0 = 0;

    // find max value on first column
    r1 = 0;
    dmax = 0.0;
    for (j = 0; j < n; j++) {
        tmp = std::abs(A[j]);
        if (dmax < tmp) {
            dmax = tmp;
            r1 = j;
        }
    }

    // find max value on r1'st row
    c1 = 0;
    dmax = 0.0;
    for (j = 0; j < m; j++) {
        tmp = std::abs(A[j * ldA + r1]);
        if (dmax < tmp) {
            dmax = tmp;
            c1 = j;
        }
    }

    if (c0 == c1) {
        *ipiv = r1;
        *jpiv = c1;
        return;
    }

    r0 = r1;
    c0 = c1;

    i = 1;
    while (true) {
        if (i & 1) {
            r1 = 0;
            dmax = 0.0;
            for (j = 0; j < n; j++) {
                tmp = std::abs(A[c0 * ldA + j]);
                if (dmax < tmp) {
                    dmax = tmp;
                    r1 = j;
                }
            }
            i++;
            if (r0 == r1) {
                break;
            }
        }
        else {
            c1 = 0;
            dmax = 0.0;
            for (j = 0; j < m; j++) {
                tmp = std::abs(A[j * ldA + r0]);
                if (dmax < tmp) {
                    dmax = tmp;
                    c1 = j;
                }
            }
            i++;
            if (c0 == c1) {
                break;
            }
        }
        r0 = r1;
        c0 = c1;
    }

    *ipiv = r1;
    *jpiv = c1;
}

void dgeluf(const ptrdiff_t n, const ptrdiff_t m, RP(double) A, const ptrdiff_t ldA, int *ipiv, int *jpiv)
{
    ptrdiff_t minnm, i, j, k, jmax, imax;
    double tmp1, tmp2, aii, aik;

    minnm = std::min(n, m);

    for (i = 0; i < minnm; i++) {
        constexpr bool use_complete_pivoting = false;
        if constexpr (use_complete_pivoting) {
            // complete pivoting
            imax = i;
            jmax = i;
            tmp1 = 0.0;
            for (k = i; k < m; k++) {
                for (j = i; j < n; j++) {
                    tmp2 = std::abs(A[ldA * k + j]);
                    if (tmp1 < tmp2) {
                        tmp1 = tmp2;
                        imax = j;
                        jmax = k;
                    }
                }
            }
        }
        else {
            dlarook(n - i, m - i, &A[i * (ldA + 1)], ldA, &imax, &jmax);
            imax += i;
            jmax += i;
        }
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
        if (aii == 0) {
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

namespace sparse
{
    ColMatPartition &ColMatPartition::operator=(ColMatPartition &&other)
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
        ncol = other.ncol;
        node = other.node;
        nparam = other.nparam;

        return *this;
    }

    ColMatPartition::ColMatPartition(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam)
    {
        update(mat, partition_num, ncol, node, nparam);
    }

    void ColMatPartition::update(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam)
    {
        ptrdiff_t i, j, k, col_start, col_stop, row_stop, start, stop, ldA;

        const double *Ax = mat->Ax;
        const int64_t *Ap = mat->Ap;
        const int64_t *Ai = mat->Ai;
        const ptrdiff_t M = mat->nrows;

        emptyPartition(ncol, node, nparam);
        ldA = node * (ncol + 1);

        col_start = partition_num * ldA;
        row_stop = (partition_num + 1) * ldA;
        col_stop = (partition_num + 1) * ldA + node;

        // fill in A and B together since they share columns
        for (i = col_start; i < col_stop; i++) {
            start = Ap[i];
            stop = Ap[i + 1];

            for (j = start; j < stop; j++) {
                k = Ai[j];
                // fill in A
                if ((col_start <= k) && (k < row_stop)) {
                    A[(i - col_start) * ldA + (k - col_start)] = Ax[j];
                }
                if (((i - col_start) < node) || (row_stop <= i)) {
                    // these columns are zero in B so skip
                    continue;
                }
                // fill in B
                if ((M - nparam) <= k) {
                    B[(i - col_start) * nparam + (k + nparam - M)] = Ax[j];
                }
            }
        }

        // fill in C
        for (i = M - nparam; i < M; i++) {
            // pjac columns are dense in construction of mat by collocator
            start = Ap[i] + partition_num * (node * (ncol + 1));
            stop = Ap[i + 1];

            for (j = start; j < stop; j++) {
                k = Ai[j];
                // fill in A
                if ((col_start <= k) && (k < row_stop)) {
                    C[(i + nparam - M) * ldA + (k - col_start)] = Ax[j];
                }
                else if (row_stop <= k) {
                    break;
                }
            }
        }
    }

    void ColMatPartition::update(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num)
    {
        ptrdiff_t i, j, k, col_start, col_stop, row_stop, start, stop, ldA;

        const double *Ax = mat->Ax;
        const int64_t *Ap = mat->Ap;
        const int64_t *Ai = mat->Ai;
        const ptrdiff_t M = mat->nrows;

        ldA = node * (ncol + 1);

        col_start = partition_num * ldA;
        row_stop = (partition_num + 1) * ldA;
        col_stop = (partition_num + 1) * ldA + node;

        memset(A, 0, ldA * (ldA + node) * sizeof(double));
        memset(B, 0, nparam * (ldA + node) * sizeof(double));
        memset(C, 0, nparam * ldA * sizeof(double));
        memset(D, 0, nparam * nparam * sizeof(double));

        // fill in A and B together since they share columns
        for (i = col_start; i < col_stop; i++) {
            start = Ap[i];
            stop = Ap[i + 1];

            for (j = start; j < stop; j++) {
                k = Ai[j];
                // fill in A
                if ((col_start <= k) && (k < row_stop)) {
                    A[(i - col_start) * ldA + (k - col_start)] = Ax[j];
                }
                if (((i - col_start) < node) || (row_stop <= i)) {
                    // these columns are zero in B so skip
                    continue;
                }
                // fill in B
                if ((M - nparam) <= k) {
                    // if (!std::isfinite(Ax[j])) {
                    //     fmt::println("partition {} got an invalid entry", partition_num);
                    //     // double *AA = mat->dense();
                    //     // for (int ii = 0; ii < mat->ncols; ii++) {
                    //     //     fmt::println("{: .8e}", AA[(mat->nrows - 1) * mat->ncols]);
                    //     // }
                    //     // free(AA);
                    //     exit(0);
                    // }
                    B[(i - col_start) * nparam + (k + nparam - M)] = Ax[j];
                }
            }
        }

        // fill in C
        for (i = M - nparam; i < M; i++) {
            // pjac columns are dense in construction of mat by collocator
            start = Ap[i] + partition_num * (node * (ncol + 1));
            stop = Ap[i + 1];

            for (j = start; j < stop; j++) {
                k = Ai[j];
                // fill in A
                if ((col_start <= k) && (k < row_stop)) {
                    C[(i + nparam - M) * ldA + (k - col_start)] = Ax[j];
                }
                else if (row_stop <= k) {
                    break;
                }
            }
        }
    }

    void ColMatPartition::display() const
    {
        ptrdiff_t i, j;
        const ptrdiff_t bsize = node * (ncol + 1);
        std::vector<double> tmp;

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

    ColMatPartition::~ColMatPartition()
    {
        ncol = 0;
        node = 0;
        nparam = 0;
        if (A != nullptr) {
            free(A);
            A = nullptr;
        }
        if (B != nullptr) {
            free(B);
            B = nullptr;
        }
        if (C != nullptr) {
            free(C);
            C = nullptr;
        }
        if (D != nullptr) {
            free(D);
            D = nullptr;
        }
        if (ipiv != nullptr) {
            free(ipiv);
            ipiv = nullptr;
        }
        if (jpiv != nullptr) {
            free(jpiv);
            jpiv = nullptr;
        }
    }

    void ColMatPartition::emptyPartition(const ptrdiff_t new_ncol, const ptrdiff_t new_node, const ptrdiff_t new_nparam)
    {
        ptrdiff_t bsize;

        ncol = new_ncol;
        node = new_node;
        nparam = new_nparam;
        bsize = node * (ncol + 1);

        A = (double *)realloc(A, bsize * (bsize + node) * sizeof(double));
        B = (double *)realloc(B, nparam * (bsize + node) * sizeof(double));
        C = (double *)realloc(C, nparam * bsize * sizeof(double));
        D = (double *)realloc(D, nparam * nparam * sizeof(double));
        // V = (double *)realloc(V, nparam * node * sizeof(double));
        // W = (double *)realloc(W, nparam * node * sizeof(double));
        ipiv = (int32_t *)realloc(ipiv, bsize * sizeof(int));
        jpiv = (int32_t *)realloc(jpiv, bsize * sizeof(int));
        memset(A, 0, bsize * (bsize + node) * sizeof(double));
        memset(B, 0, nparam * (bsize + node) * sizeof(double));
        memset(C, 0, nparam * bsize * sizeof(double));
        memset(D, 0, nparam * nparam * sizeof(double));
        // memset(V, 0, nparam * node * sizeof(double));
        // memset(W, 0, nparam * node * sizeof(double));
        memset(ipiv, 0, bsize * sizeof(int));
        memset(jpiv, 0, bsize * sizeof(int));
    }

    void ColMatPartition::addPartitionToCOO(sparse::COOMatrix &mat, const ptrdiff_t partition_num) const
    {
        double *data;
        int64_t *irow, *icol;
        ptrdiff_t i, j, k;
        const ptrdiff_t bsize = node * (ncol + 1);
        const ptrdiff_t nnz = mat.nnz;
        const ptrdiff_t N = mat.ncol;
        const ptrdiff_t partition_nnz = 2 * node * node + // main blocks
                                        nparam * bsize +  // parameter cols
                                        nparam * nparam + // D
                                        nparam * node +   // V
                                        nparam * node;    // W

        mat.setNNZ(nnz + partition_nnz);
        data = mat.data + nnz;
        irow = mat.irow + nnz;
        icol = mat.icol + nnz;
        k = 0;

        // add blocks
        for (i = 0; i < node; i++) {
            for (j = 0; j < node; j++) {
                data[k] = A[i * bsize + (bsize - node + j)];
                irow[k] = node * partition_num + j;
                icol[k] = node * partition_num + i;
                k++;
            }
        }
        for (i = 0; i < node; i++) {
            for (j = 0; j < node; j++) {
                data[k] = A[(i + bsize) * bsize + (bsize - node + j)];
                irow[k] = node * partition_num + j;
                icol[k] = node * partition_num + i + node;
                k++;
            }
        }
        // add cols
        for (i = 0; i < nparam; i++) {
            for (j = 0; j < node; j++) {
                data[k] = C[i * bsize + (bsize - node + j)];
                irow[k] = node * partition_num + j;
                icol[k] = N - nparam + i;
                k++;
            }
        }
        // add D
        for (i = 0; i < nparam; i++) {
            for (j = 0; j < nparam; j++) {
                data[k] = D[i * nparam + j];
                irow[k] = N - nparam + j;
                icol[k] = N - nparam + i;
                k++;
            }
        }
        // add V
        for (i = 0; i < node; i++) {
            for (j = 0; j < nparam; j++) {
                data[k] = B[i * nparam + j];
                irow[k] = N - nparam + j;
                icol[k] = node * partition_num + i;
                k++;
            }
        }
        // add W
        for (i = 0; i < node; i++) {
            for (j = 0; j < nparam; j++) {
                data[k] = B[(i + bsize) * nparam + j];
                irow[k] = N - nparam + j;
                icol[k] = node * partition_num + i + node;
                k++;
            }
        }
    }

    void ColMatPartition::condenseVector(RP(double) x, const ptrdiff_t nx, const ptrdiff_t partition_num) const
    {
        const ptrdiff_t bsize = node * (ncol + 1);
        const double *LU = &A[bsize * node];

        double *Fx = &x[partition_num * bsize];
        double *Fd = &x[nx - nparam];

        // permute
        dlaswpr(1, Fx, nx, 0, bsize - node, ipiv, 1);
        // elimitate square
        dtrsm("L", "L", "N", "U", bsize - node, 1, 1.0, LU, bsize, Fx, nx);
        // eliminate remaining block
        dgemm("N", "N", node, 1, bsize - node, -1.0, &LU[bsize - node], bsize, Fx, nx, 1.0, &Fx[bsize - node], nx);
        // eliminate last nparam rows
        dgemm("N", "N", nparam, 1, bsize - node, -1.0, &B[node * nparam], nparam, Fx, nx, 1.0, Fd, nx);
    }

    void ColMatPartition::condense()
    {
        const ptrdiff_t bsize = node * (ncol + 1);
        double *LU = &A[bsize * node];

        dgeluf(bsize, bsize - node, LU, bsize, ipiv, jpiv);
        // fmt::println("ipiv = [{:2d}]", fmt::join(std::vector<int>(ipiv, ipiv + bsize - node), " "));
        // fmt::println("jpiv = [{:2d}]\n", fmt::join(std::vector<int>(jpiv, jpiv + bsize - node), " "));

        // apply permutations
        dlaswpr(node, A, bsize, 0, bsize - node, ipiv, 1);
        dlaswpr(node, &A[bsize * bsize], bsize, 0, bsize - node, ipiv, 1);
        dlaswpr(nparam, C, bsize, 0, bsize - node, ipiv, 1);
        dlaswpc(nparam, &B[node * nparam], nparam, 0, bsize - node, jpiv, 1);

        // apply operations to A (left)
        dtrsm("L", "L", "N", "U", bsize - node, node, 1.0, LU, bsize, A, bsize);
        dgemm("N", "N", node, node, bsize - node, -1.0, &LU[bsize - node], bsize, A, bsize, 1.0, &A[bsize - node], bsize);
        // apply operations to A (right)
        dtrsm("L", "L", "N", "U", bsize - node, node, 1.0, LU, bsize, &A[bsize * bsize], bsize);
        dgemm("N", "N", node, node, bsize - node, -1.0, &LU[bsize - node], bsize, &A[bsize * bsize], bsize, 1.0, &A[bsize * bsize + (bsize - node)], bsize);
        // apply operations to C
        dtrsm("L", "L", "N", "U", bsize - node, nparam, 1.0, LU, bsize, C, bsize);
        dgemm("N", "N", node, nparam, bsize - node, -1.0, &LU[bsize - node], bsize, C, bsize, 1.0, &C[bsize - node], bsize);
        // factor B
        dtrsm("R", "U", "N", "N", nparam, bsize - node, 1.0, LU, bsize, &B[node * nparam], nparam);

        // D, V, and W are shared between partitions
        // so the condense method assumes it to be zero
        // so that after condensation is done, the first
        // subsystem can be formed by adding them all up

        // factor D
        dgemm("N", "N", nparam, nparam, bsize - node, -1.0, &B[node * nparam], nparam, C, bsize, 0.0, D, nparam);
        // factor left of B (V)
        dgemm("N", "N", nparam, node, bsize - node, -1.0, &B[node * nparam], nparam, A, bsize, 0.0, B, nparam);
        // factor right of B (W)
        dgemm("N", "N", nparam, node, bsize - node, -1.0, &B[node * nparam], nparam, &A[bsize * bsize], bsize, 0.0, &B[bsize * nparam], nparam);
    }

    void ColMatPartition::backsubstitute(RP(double) x, RP(double) y, const ptrdiff_t nx, const ptrdiff_t partition_num) const
    {
        ptrdiff_t i;
        const ptrdiff_t bsize = node * (ncol + 1);
        const double *LU = &A[bsize * node];
        int colperm[bsize - node];

        double *R = &y[partition_num * bsize];
        double *xL = &x[partition_num * bsize];
        double *xC = xL + node;
        double *xR = xL + bsize;
        double *K = &x[nx - nparam];

        dgemv("N", bsize - node, node, -1.0, A, bsize, xL, 1, 1.0, R, 1);
        dgemv("N", bsize - node, node, -1.0, &A[bsize * bsize], bsize, xR, 1, 1.0, R, 1);
        dgemv("N", bsize - node, nparam, -1.0, C, bsize, K, 1, 1.0, R, 1);
        memcpy(xC, R, (bsize - node) * 1 * sizeof(double));
        dtrsm("L", "U", "N", "N", bsize - node, 1, 1.0, LU, bsize, xC, nx);

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

    CollocationMatrix::CollocationMatrix(const sparse::RealCSCMatrix *mat, const ptrdiff_t ngrid, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam) : m_mat(mat), m_nblocks(ngrid - 1), m_cap(ngrid - 1), factored(false)
    {
        ptrdiff_t i, N;

        if (((((mat->ncols - nparam) / node) - 1) / (ncol + 1) + 1) != ngrid) {
            throw std::runtime_error("CollocationMatrix::update received invalid partition sizes for provided matrix");
        }

        m_partitions = (ColMatPartition *)malloc(m_nblocks * sizeof(ColMatPartition));
        memset(m_partitions, 0, m_nblocks * sizeof(ColMatPartition));
        for (i = 0; i < m_nblocks; i++) {
            m_partitions[i].update(m_mat, i, ncol, node, nparam);
            // m_partitions[i] = std::move(ColMatPartition(m_mat, i, ncol, node, nparam));
        }

        // set base size
        N = node * ngrid + nparam;
        m_S_coo.reshape(N, N);
        // set base memory
        m_S_coo.setNNZ(m_nblocks * 2 * node * node + 2 * nparam * N);
        m_S_coo.setNNZ(0);
        // add bc
        extractBC();
        // get base param equations
        extractParameterEquations();
        // get base D
        extractD();
    }

    void CollocationMatrix::update(const sparse::RealCSCMatrix *mat, const ptrdiff_t ngrid, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam)
    {
        ptrdiff_t i, N, new_cap;

        if (((((mat->ncols - nparam) / node) - 1) / (ncol + 1) + 1) != ngrid) {
            throw std::runtime_error("CollocationMatrix::update received invalid partition sizes for provided matrix");
        }

        m_mat = mat;
        factored = false;

        // make sure memory get reallocated if there's a structure change
        if (m_partitions && ((m_partitions->ncol != ncol) || (m_partitions->node != node) || (m_partitions->nparam != nparam))) {
            for (i = 0; i < m_nblocks; i++) {
                (m_partitions + i)->emptyPartition(ncol, node, nparam);
            }
        }

        // check if we need to grow or not
        if (m_cap < (ngrid - 1)) {
            // fmt::println("growing");
            new_cap = ngrid - 1;
            m_partitions = (ColMatPartition *)realloc(m_partitions, new_cap * sizeof(ColMatPartition));
            for (i = m_cap; i < new_cap; i++) {
                memset(&m_partitions[i], 0, sizeof(ColMatPartition));
                m_partitions[i].update(m_mat, i, ncol, node, nparam);
            }
            for (i = 0; i < m_cap; i++) {
                (m_partitions + i)->update(m_mat, i);
            }
            m_cap = new_cap;
            m_nblocks = new_cap;
        }
        else {
            m_nblocks = ngrid - 1;
            for (i = 0; i < m_nblocks; i++) {
                (m_partitions + i)->update(m_mat, i);
            }
        }

        // set base size
        N = node * ngrid + nparam;
        m_S_coo.reshape(N, N);
        // set base memory
        m_S_coo.setNNZ(m_nblocks * 2 * node * node + 2 * nparam * N);
        m_S_coo.setNNZ(0);
        // zero out anything else
        for (i = 0; i < m_S_coo.cap; i++) {
            m_S_coo.data[i] = 0;
            m_S_coo.irow[i] = 0;
            m_S_coo.icol[i] = 0;
        }
        // add bc
        extractBC();
        // get base param equations
        extractParameterEquations();
        // get base D
        extractD();
    }

    CollocationMatrix::~CollocationMatrix()
    {
        ptrdiff_t i;
        if (m_partitions) {
            for (i = 0; i < m_cap; i++) {
                (m_partitions + i)->~ColMatPartition();
            }
            free(m_partitions);
            m_partitions = NULL;
        }
        if (m_work) {
            free(m_work);
            m_work = nullptr;
        }
        m_nblocks = 0;
        m_cap = 0;
        factored = false;
    }

    void CollocationMatrix::extractBC()
    {
        // for now just manually add periodic bc
        ptrdiff_t i, l, nnz, nnz_new;

        const ptrdiff_t nparam = m_partitions->nparam;
        const ptrdiff_t node = m_partitions->node;
        const ptrdiff_t N = m_S_coo.nrow;

        nnz = m_S_coo.nnz;
        nnz_new = nnz + 2 * node;
        m_S_coo.setNNZ(nnz_new);

        for (l = 0, i = 0; i < node; i++) {
            m_S_coo.data[nnz + l] = 1.0;
            m_S_coo.irow[nnz + l] = N - nparam - node + i;
            m_S_coo.icol[nnz + l] = i;
            l++;
        }

        for (i = 0; i < node; i++) {
            m_S_coo.data[nnz + l] = -1.0;
            m_S_coo.irow[nnz + l] = N - nparam - node + i;
            m_S_coo.icol[nnz + l] = N - nparam - node + i;
            l++;
        }
    }

    void CollocationMatrix::extractParameterEquations()
    {
        ptrdiff_t ii, i, j, k, l, nnz, nnz_new, start, stop, i0, i1;

        const ptrdiff_t nparam = m_partitions->nparam;
        const ptrdiff_t ncol = m_partitions->ncol;
        const ptrdiff_t node = m_partitions->node;
        ;
        const ptrdiff_t M = m_mat->nrows;
        const ptrdiff_t N = m_S_coo.nrow;
        const double *Ax = m_mat->Ax;
        const int64_t *Ap = m_mat->Ap;
        const int64_t *Ai = m_mat->Ai;

        nnz = m_S_coo.nnz;
        nnz_new = nnz + nparam * node * (m_nblocks + 1);
        m_S_coo.setNNZ(nnz_new);

        for (l = 0, ii = 0; ii < (m_nblocks + 1); ii++) {
            i0 = ii * node * (ncol + 1);
            i1 = i0 + node;

            for (i = i0; i < i1; i++) {
                start = Ap[i];
                stop = Ap[i + 1];

                for (j = stop - 1; j >= start; j--) {
                    k = Ai[j];
                    if (k < (M - nparam)) {
                        break;
                    }
                    m_S_coo.data[nnz + l] = Ax[j];
                    m_S_coo.irow[nnz + l] = N - nparam + k - M + nparam;
                    m_S_coo.icol[nnz + l] = ii * node + i - i0;
                    l++;
                }
            }
        }
    }

    void CollocationMatrix::extractD()
    {
        ptrdiff_t i, j, k, l, nnz, nnz_new, start, stop;

        const ptrdiff_t nparam = m_partitions->nparam;
        const ptrdiff_t ncol = m_partitions->ncol;
        const ptrdiff_t node = m_partitions->node;
        // const ptrdiff_t nblock = m_partitions.size();
        const ptrdiff_t M = m_mat->nrows;
        const ptrdiff_t N = m_S_coo.nrow;
        const double *Ax = m_mat->Ax;
        const int64_t *Ap = m_mat->Ap;
        const int64_t *Ai = m_mat->Ai;

        nnz = m_S_coo.nnz;
        nnz_new = nnz + nparam * nparam;
        m_S_coo.setNNZ(nnz_new);

        for (l = 0, i = M - nparam; i < M; i++) {
            start = Ap[i] + m_nblocks * node * (ncol + 1);
            stop = Ap[i + 1];
            for (j = start; j < stop; j++) {
                k = Ai[j];
                if (k < (M - nparam)) {
                    continue;
                }
                m_S_coo.data[nnz + l] = Ax[j];
                m_S_coo.irow[nnz + l] = N - nparam + k - M + nparam;
                m_S_coo.icol[nnz + l] = N - nparam + i - M + nparam;
                l++;
            }
        }
    }

    void CollocationMatrix::factor()
    {
        ptrdiff_t i;

        if (factored) {
            return;
        }

        // factor each partition
#pragma omp parallel for private(i)
        for (i = 0; i < m_nblocks; i++) {
            ColMatPartition &partition = m_partitions[i];
            partition.condense();
        }

        // add each partition to COO matrix
        for (i = 0; i < m_nblocks; i++) {
            ColMatPartition &partition = m_partitions[i];
            partition.addPartitionToCOO(m_S_coo, i);
        }

        // update UMFPack controls to a more efficient block
        // size for this subsystem
        m_S_csc.Control[UMFPACK_BLOCK_SIZE] = 4 * m_partitions->node;
        m_S_csc.Control[UMFPACK_SYM_PIVOT_TOLERANCE] = 1;
        m_S_csc.Control[UMFPACK_PIVOT_TOLERANCE] = 1;
        m_S_csc.updateFromCOO(m_S_coo);

        m_S_csc.factor();

        factored = true;
    }

    void CollocationMatrix::resizeWork()
    {
        const ptrdiff_t M = m_mat->nrows;
        const ptrdiff_t N = m_S_coo.nrow;
        // work needs to be 2 * N + 2 * M
        m_work = (double *)realloc(m_work, (2 * N + 3 * M) * sizeof(double));
    }

    void CollocationMatrix::solveWork(const RP(double) b, RP(double) x)
    {
        ptrdiff_t i;
        double *v, *w, *y;

        const ptrdiff_t M = m_mat->nrows;
        const ptrdiff_t N = m_S_coo.nrow;
        const ptrdiff_t nparam = m_partitions->nparam;
        const ptrdiff_t ncol = m_partitions->ncol;
        const ptrdiff_t node = m_partitions->node;
        const ptrdiff_t bsize = node * (ncol + 1);

        v = m_work;
        w = v + N;
        y = w + N;

        // factor if not already
        factor();

        // copy rhs into x
        memcpy(x, b, M * sizeof(double));

        // condense x
        for (i = 0; i < m_nblocks; i++) {
            const ColMatPartition &partition = m_partitions[i];
            partition.condenseVector(x, M, i);
            // copy condensed rhs node section into subsystem RHS
            memcpy(&v[i * node], &x[(i + 1) * bsize - node], node * sizeof(double));
        }
        // don't forget last node and parameter RHS
        memcpy(&v[N - node - nparam], &x[M - node - nparam], (node + nparam) * sizeof(double));
        // copy current RHS into backup vector y
        memcpy(y, x, M * sizeof(double));

        // solve subsystem Sw = v
        m_S_csc.solve(v, w);

        // copy results into x
        for (i = 0; i < (m_nblocks + 1); i++) {
            memcpy(&x[i * bsize], &w[i * node], node * sizeof(double));
        }
        memcpy(&x[M - nparam], &w[N - nparam], nparam * sizeof(double));

        // back substitution on final blocks and inverse column swaps
#pragma omp parallel for private(i)
        for (i = 0; i < m_nblocks; i++) {
            const ColMatPartition &partition = m_partitions[i];
            partition.backsubstitute(x, y, M, i);
        }
    }

    void CollocationMatrix::solve(const RP(double) b, RP(double) x)
    {
        ptrdiff_t i;
        double *r, *delta;
        const ptrdiff_t M = m_mat->nrows;
        const ptrdiff_t N = m_S_coo.nrow;

        resizeWork();

        // solveWork uses 2N + M work so
        r = &m_work[2 * N + M];
        delta = r + M;

        solveWork(b, x);

        // take single refinement step
        for (int j = 0; j < 0; j++) {
            memcpy(r, b, M * sizeof(double));
            m_mat->gemv(1.0, r, -1.0, x);
            fmt::println("{} ||b - Ax|| = {:.8e}", j, norm2(r, M) / M);
            solveWork(r, delta);
            for (i = 0; i < M; i++) {
                x[i] += delta[i];
            }
        }
    }
} // namespace sparse