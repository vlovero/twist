#include "linalg.h"
#include "shared.h"
#include <omp.h>

void zlatrn(const ptrdiff_t n, std::complex<double> *A, const ptrdiff_t ldA)
{
    constexpr ptrdiff_t block_size = 16;
    std::complex<double> buff1[block_size * block_size];
    std::complex<double> temp;
    ptrdiff_t i, j, ii, jj, nn, r, m1, m2, k, kk;
    int num_threads;

    if (n <= 1) {
        return;
    }
    else if (n <= block_size) {
        for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
                temp = A[i * ldA + j];
                A[i * ldA + j] = A[j * ldA + i];
                A[j * ldA + i] = temp;
            }
        }
        return;
    }

    num_threads = (n >= 256) ? (n / 64) : 1;
    num_threads = std::min(omp_get_max_threads(), num_threads);
    (void)num_threads; // silence unused variable warning
    // num_threads = 1;

    r = n % block_size;
    nn = n - r;

    if (r == 0) {
#pragma omp parallel for private(i, j, buff1, ii, jj, k, kk, temp) firstprivate(A, ldA, n, nn) num_threads(num_threads)
        for (i = 0; i < nn; i += block_size) {
            // transpose along main diagonal
            for (ii = i, k = 0; ii < i + block_size; ii++, k++) {
                for (jj = i + k + 1; jj < i + block_size; jj++) {
                    temp = A[ii * ldA + jj];
                    A[ii * ldA + jj] = A[jj * ldA + ii];
                    A[jj * ldA + ii] = temp;
                }
            }
            // do out of place transposes on non-overlapping blocks
            for (j = i + block_size; j < nn; j += block_size) {
                for (ii = i, k = 0; ii < i + block_size; ii++, k++) {
                    for (jj = j, kk = 0; jj < j + block_size; jj++, kk++) {
                        buff1[kk * block_size + k] = A[ii * ldA + jj];
                    }
                }
                for (ii = i; ii < i + block_size; ii++) {
                    for (jj = j; jj < j + block_size; jj++) {
                        A[ii * ldA + jj] = A[jj * ldA + ii];
                    }
                }
                for (jj = j, kk = 0; jj < j + block_size; jj++, kk++) {
                    for (ii = i, k = 0; ii < i + block_size; ii++, k++) {
                        A[jj * ldA + ii] = buff1[kk * block_size + k];
                    }
                }
            }
        }
        return;
    }

    // when size is not aligned with block size
    // combine remainder with block size, then
    // split that larger block into two smaller
    // blocks that are less optimal but still more
    // cache efficient than unblocked code

    m1 = (r + block_size) / 2;
    m2 = block_size - m1 + r;

    // transpose first diagonal
    for (i = 0; i < m1; i++) {
        for (j = i + 1; j < m1; j++) {
            temp = A[i * ldA + j];
            A[i * ldA + j] = A[j * ldA + i];
            A[j * ldA + i] = temp;
        }
    }

    // transpose corners
    for (i = 0; i < m1; i++) {
        for (j = 0; j < m2; j++) {
            buff1[j * m1 + i] = A[i * ldA + j + m1];
        }
    }
    for (i = 0; i < m2; i++) {
        for (j = 0; j < m1; j++) {
            A[j * ldA + i + m1] = A[(m1 + i) * ldA + j];
        }
    }
    for (j = 0; j < m2; j++) {
        for (i = 0; i < m1; i++) {
            A[(j + m1) * ldA + i] = buff1[j * m1 + i];
        }
    }

    // transpose second diagonal
    for (i = m1; i < m1 + m2; i++) {
        for (j = i + 1; j < m1 + m2; j++) {
            temp = A[i * ldA + j];
            A[i * ldA + j] = A[j * ldA + i];
            A[j * ldA + i] = temp;
        }
    }

#pragma omp parallel for private(i, j, buff1, ii, jj, k, kk, temp) firstprivate(m1, m2, A, ldA, n) num_threads(num_threads)
    for (j = m1 + m2; j < n; j += block_size) {
        // top blocks
        for (i = 0; i < m1; i++) {
            for (jj = 0; jj < block_size; jj++) {
                buff1[jj * m1 + i] = A[i * ldA + jj + j];
            }
        }
        for (i = 0; i < block_size; i++) {
            for (jj = 0; jj < m1; jj++) {
                A[jj * ldA + i + j] = A[(j + i) * ldA + jj];
            }
        }
        for (jj = 0; jj < block_size; jj++) {
            for (i = 0; i < m1; i++) {
                A[(jj + j) * ldA + i] = buff1[jj * m1 + i];
            }
        }

        // lower blocks
        for (ii = 0; ii < m2; ii++) {
            for (jj = 0; jj < block_size; jj++) {
                buff1[jj * m2 + ii] = A[(m1 + ii) * ldA + jj + j];
            }
        }
        for (ii = 0; ii < block_size; ii++) {
            for (jj = 0; jj < m2; jj++) {
                A[(m1 + jj) * ldA + j + ii] = A[(j + ii) * ldA + m1 + jj];
            }
        }
        for (jj = 0; jj < block_size; jj++) {
            for (ii = 0; ii < m2; ii++) {
                A[(j + jj) * ldA + m1 + ii] = buff1[jj * m2 + ii];
            }
        }
    }

    return zlatrn(n - m1 - m2, &A[(m1 + m2) * ldA + (m1 + m2)], ldA);
}

void zlarev(const char *which, const ptrdiff_t n, std::complex<double> *A, ptrdiff_t ldA)
{
    std::complex<double> *col1;
    std::complex<double> *col2;
    std::complex<double> a, b, c, d;
    bool odd;
    ptrdiff_t i, j, n2;

    if (*which == 'B') {
        n2 = n / 2;
        odd = n & 1;
        for (i = 0; i < n2; i++) {
            col1 = &A[i * ldA];
            col2 = &A[(n - 1 - i) * ldA];
            for (j = 0; j < n2; j++) {
                a = col1[j];
                b = col2[j];
                c = col1[n - 1 - j];
                d = col2[n - 1 - j];
                col1[j] = d;
                col2[j] = c;
                col1[n - 1 - j] = b;
                col2[n - 1 - j] = a;
            }
            if (odd) {
                std::swap(col1[n2], col2[n2]);
            }
        }
        if (odd) {
            i = n2;
            col1 = &A[i * ldA];
            for (j = 0; j < n2; j++) {
                std::swap(col1[j], col1[n - 1 - j]);
            }
        }
    }
    else if (*which == 'R') {
        for (i = 0; i < n; i++) {
            for (j = 0; j < (n / 2); j++) {
                std::swap(A[i * ldA + j], A[i * ldA + (n - 1 - j)]);
            }
        }
    }
    else if (*which == 'C') {
        for (i = 0; i < (n / 2); i++) {
            for (j = 0; j < n; j++) {
                std::swap(A[i * ldA + j], A[(n - 1 - i) * ldA + j]);
            }
        }
    }
}

void zggprp(const bool wantq, const bool wantz, const int n, std::complex<double> *A, const int ldA, std::complex<double> *B, const int ldB, std::complex<double> *Q, const int ldQ, std::complex<double> *Z, const int ldZ, std::complex<double> *work, const int lwork, double *rwork, int *jpvt, int *info)
{
    double tol, normb;
    std::complex<double> dummy[1];
    int ierr, workneeded;
    ptrdiff_t i, j;

    workneeded = 0;
    zgeqp3(n, n, B, ldB, jpvt, NULL, dummy, -1, rwork, &ierr);
    workneeded = std::max(workneeded, (int)((*dummy).real()));
    zunmqr("R", "N", n, n, n, B, ldB, NULL, A, ldA, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)((*dummy).real()));
    zungqr(n, n, n, Z, ldZ, NULL, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)((*dummy).real()));
    if (lwork == -1) {
        *info = 0;
        *work = workneeded;
        return;
    }

    normb = zlange("F", n, n, B, ldB, rwork);
    tol = std::max(1.2 * n, normb) * std::numeric_limits<double>::epsilon();

    // B = B^T
    zlatrn(n, B, ldB);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B[i * ldB + j] = std::conj(B[i * ldB + j]);
        }
    }

    // B = B[:, ::-1]
    zlarev("C", n, B, ldB);

    // RRQR(B)
    memset(jpvt, 0, sizeof(int) * n);
    zgeqp3(n, n, B, ldB, jpvt, work, &work[n], lwork - n, rwork, &ierr);

    // A = A[::-1]
    zlarev("R", n, A, ldA);
    // A = A[p]
    zlapmr(1, n, n, A, ldA, jpvt);
    // A = AQ
    zunmqr("R", "N", n, n, n, B, ldB, work, A, ldA, &work[n], lwork - n, &ierr);
    if (wantq) {
        // initialize Q = J P J
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                Q[i * ldQ + j] = 0;
            }
        }
        for (i = 0; i < n; i++) {
            j = n - jpvt[n - i - 1];
            Q[i * ldQ + j] = 1;
        }
    }
    if (wantz) {
        // copy B into Z
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                Z[i * ldZ + j] = B[i * ldB + j];
            }
        }
        zungqr(n, n, n, Z, ldZ, work, &work[n], lwork - n, &ierr);
        zlarev("C", n, Z, ldZ);
    }
    // A = A[::-1][:, ::-1]
    zlarev("B", n, A, ldA);
    // B = B.T[::-1][:, ::-1]
    for (i = 0; i < (n - 1); i++) {
        for (j = i + 1; j < n; j++) {
            B[i * ldB + j] = 0;
        }
    }
    zlatrn(n, B, ldB);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B[i * ldB + j] = std::conj(B[i * ldB + j]);
        }
    }
    zlarev("B", n, B, ldB);
    work[0] = normb;
    work[1] = tol;
}

void zgghd4(const char *compq, const char *compz, const ptrdiff_t n, const int ilo, const int ihi, std::complex<double> *A, const ptrdiff_t ldA, std::complex<double> *B, const ptrdiff_t ldB, std::complex<double> *Q, const ptrdiff_t ldQ, std::complex<double> *Z, const ptrdiff_t ldZ, std::complex<double> *work, int lwork, double *rwork, int *info)
{
    int workneeded, ierr;
    std::complex<double> dummy[1];
    double norma, tol, tmp, norm;
    int niter, ncols, jobq, jobz;
    std::complex<double> *X;
    ptrdiff_t i, j;
    ptrdiff_t k, col_len;
    const double eps = std::numeric_limits<double>::epsilon();

    if (*compq == 'I') {
        jobq = 1;
    }
    else if (*compq == 'V') {
        jobq = 2;
    }
    else {
        jobq = 0;
    }

    if (*compz == 'I') {
        jobz = 1;
    }
    else if (*compz == 'V') {
        jobz = 2;
    }
    else {
        jobz = 0;
    }

    // arg checks here

    // workpace for iterative portion
    workneeded = 2 * (n + 1);
    zgehrd(n, 1, n, NULL, ldA, NULL, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zunmhr("L", "C", n, n, 1, n, NULL, ldA, NULL, NULL, n, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)(*dummy).real());

    zgerqf(n, n, NULL, n, NULL, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zunmrq("R", "C", n, n, n, NULL, n, NULL, NULL, n, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    workneeded += n * (n + 1);

    if (lwork == -1) {
        *work = workneeded;
        *info = 0;
        return;
    }

    assert(lwork >= workneeded);
    X = work;
    work += n * n;
    lwork -= n * n;

    if (jobq == 1) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                tmp = (i == j) ? 1.0 : 0.0;
                Q[i * ldQ + j] = tmp;
            }
        }
    }
    if (jobz == 1) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                tmp = (i == j) ? 1.0 : 0.0;
                Z[i * ldZ + j] = tmp;
            }
        }
    }

    norma = zlange("F", n, n, A, ldA, rwork);
    tol = eps * norma;
    niter = 0;
    k = ilo;
    while (k < n) {
        niter++;
        ncols = ihi - k;
        // copy A into X
        for (size_t i = k, ii = 0; i < (size_t)n; i++, ii++) {
            for (size_t j = k, jj = 0; j < (size_t)n; j++, jj++) {
                X[ii * ncols + jj] = A[i * ldA + j];
            }
        }
        ztrsm("R", "U", "N", "N", ncols, ncols, 1.0, &B[k * ldB + k], ldB, X, ncols);
        // _, Qc = hessenberg(X, calc_q=True)
        zgehrd(ncols, 1, ncols, X, ncols, work, &work[ncols], lwork - ncols, &ierr);
        // A[k:, k:] = Qc.T @ A[k:, k:]
        zunmhr("L", "C", ncols, ncols, 1, ncols, X, ncols, work, &A[k * ldA + k], ldA, &work[ncols], lwork - ncols, &ierr);
        // B[k:, k:] = Qc.T @ B[k:, k:]
        zunmhr("L", "C", ncols, ncols, 1, ncols, X, ncols, work, &B[k * ldB + k], ldB, &work[ncols], lwork - ncols, &ierr);
        if (jobq) {
            zunmhr("R", "N", n, ncols, 1, ncols, X, ncols, work, &Q[k * ldQ], ldQ, &work[ncols], lwork - ncols, &ierr);
        }
        // B[k:, k:], Zc = rq(B[k:, k:])
        zgerqf(ncols, ncols, &B[k * ldB + k], ldB, work, &work[ncols], lwork - ncols, &ierr);
        // A[:, k:] = A[:, k:] @ Zc.T
        zunmrq("R", "C", ihi, ncols, ncols, &B[k * ldB + k], ldB, work, &A[k * ldA], ldA, &work[ncols], lwork - ncols, &ierr);
        // B[:k, k:] = B[:k, k:] @ Zc.T
        zunmrq("R", "C", k, ncols, ncols, &B[k * ldB + k], ldB, work, &B[k * ldA], ldB, &work[ncols], lwork - ncols, &ierr);
        if (jobz) {
            zunmrq("R", "C", n, ncols, ncols, &B[k * ldB + k], ldB, work, &Z[k * ldA], ldZ, &work[ncols], lwork - ncols, &ierr);
        }

        for (i = 0; i < (n - 1); i++) {
            memset(&B[i * ldB + i + 1], 0, (n - 1 - i) * sizeof(std::complex<double>));
        }

        while (1) {
            if (k >= n) {
                break;
            }
            col_len = std::max(0l, n - 2 - k);
            norm = 0.0;
            for (i = 0; i < col_len; i++) {
                norm = std::max(norm, std::abs(A[k * ldA + k + 2 + i]));
            }
            if (norm > tol) {
                break;
            }
            memset(&A[k * ldA + k + 2], 0, col_len * sizeof(std::complex<double>));
            k++;
        }
        if (niter == 50) {
            break;
        }
    }
}

void zggev4(const char *jobvl, const char *jobvr, const int n, std::complex<double> *A, const int ldA, std::complex<double> *B, const int ldB, std::complex<double> *alpha, std::complex<double> *beta, std::complex<double> *vl, const int ldvl, std::complex<double> *vr, const int ldvr, std::complex<double> *work, const int lwork, double *rwork, int *jpvt, int *info)
{
    int workneeded, ninfinite, in;
    double normb, tol, eps, tmp;
    std::complex<double> dummy[1];
    bool compvl, compvr;
    char jobqz, jobvec;
    ptrdiff_t i, j, k;

    compvl = *jobvl == 'V';
    compvr = *jobvr == 'V';
    jobqz = (compvl || compvr) ? 'S' : 'E';
    if (compvl && compvr) {
        jobvec = 'B';
    }
    else if (compvl) {
        jobvec = 'L';
    }
    else if (compvr) {
        jobvec = 'R';
    }
    else {
        jobvec = 'N';
    }
    eps = std::numeric_limits<double>::epsilon();

    // argument checks

    // workspace queries
    workneeded = 0;
    zggprp(compvl, compvr, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, dummy, -1, rwork, jpvt, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zgghd4(jobvl, jobvr, n, 1, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, dummy, -1, rwork, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zgeqrf(n, n, A, ldA, NULL, dummy, -1, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zunmqr("L", "C", n, n, n, A, ldA, NULL, NULL, n, dummy, -1, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zgerqf(n, n, A, ldA, NULL, dummy, -1, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zunmrq("L", "C", n, n, n, A, ldA, NULL, NULL, n, dummy, -1, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());
    zlaqz0(&jobqz, jobvl, jobvr, n, 1, n, A, ldA, B, ldB, alpha, beta, vl, ldvl, vr, ldvr, dummy, -1, rwork, 0, info);
    workneeded = std::max(workneeded, (int)(*dummy).real());

    if (lwork == -1) {
        *info = 0;
        *work = workneeded;
        return;
    }

    // Step 1: Preprocessing
    zggprp(compvl, compvr, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, work, lwork, rwork, jpvt, info);
    normb = (*work).real();
    ninfinite = 0;
    tol = normb * eps;
    for (k = 0; k < n; k++) {
        if (tol < std::abs(B[k * ldB + k])) {
            break;
        }
        ninfinite++;
        memset(&B[ldB * k], 0, ninfinite * sizeof(std::complex<double>));
    }
    k = ninfinite;
    // Step 2: Deflation
    if (ninfinite) {
        // QR portion
        // Q, A[:, :k] = np.linalg.qr(A[:, :k], mode="complete")
        zgeqrf(n, k, A, ldA, work, &work[k], lwork - k, info);
        // A[:, k:] = Q.T @ A[:, k:]
        zunmqr("L", "C", n, n - k, k, A, ldA, work, &A[k * ldA], ldA, &work[n], lwork - n, info);
        // B[:, k:] = Q.T @ B[:, k:]
        zunmqr("L", "C", n, n - k, k, A, ldA, work, &B[k * ldB], ldB, &work[n], lwork - n, info);
        if (compvl) {
            // Q = QA @ Q
            zunmqr("R", "N", n, n, k, A, ldA, work, vl, ldvl, &work[n], lwork - n, info);
        }
        // RQ portion
        // B[k:, k:], Q = rq(B[k:, k:])
        zgerqf(n - k, n - k, &B[k * ldB + k], ldB, work, &work[n - k], lwork - n + k, info);
        // A[:, k:] = A[:, k:] @ Q.T
        zunmrq("R", "C", n, n - k, n - k, &B[k * ldB + k], ldB, work, &A[k * ldA], ldA, &work[n - k], lwork - n + k, info);
        // B[:k, k:] = B[:k, k:] @ Q.T
        zunmrq("R", "C", k, n - k, n - k, &B[k * ldB + k], ldB, work, &B[k * ldB], ldB, &work[n - k], lwork - n + k, info);
        if (compvr) {
            // Z[:, k:] = Z[:, k:] @ QB.T
            zunmrq("R", "C", n, n - k, n - k, &B[k * ldB + k], ldB, work, &vr[k * ldvr], ldvr, &work[n - k], lwork - n + k, info);
        }

        for (i = 0; i < k; i++) {
            memset(&A[i * ldA + i + 1], 0, (n - i - 1) * sizeof(std::complex<double>));
        }
        for (i = k; i < n; i++) {
            memset(&B[i * ldB + i + 1], 0, (n - i - 1) * sizeof(std::complex<double>));
        }
    }
    // Step 3: Hessenberg-Triangular Reduction
    zgghd4(jobvl, jobvr, n, k, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, work, lwork, rwork, info);

    // Step 4: QZ
    // perform qz on remaining portion of A and B
    if (ninfinite != n) {
        zlaqz0(&jobqz, jobvl, jobvr, n, ninfinite + 1, n, A, ldA, B, ldB, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, 0, info);
    }
    // Step 5: Eigenvectors
    if (compvl || compvr) {
        ztgevc(&jobvec, "B", NULL, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, n, &in, work, rwork, info);
    }
    if (compvl) {
        for (i = 0; i < n; i++) {
            tmp = 0.0;
            for (j = 0; j < n; j++) {
                tmp += vl[i * ldvl + j].real() * vl[i * ldvl + j].real();
                tmp += vl[i * ldvl + j].imag() * vl[i * ldvl + j].imag();
            }
            tmp = 1.0 / std::sqrt(tmp);
            for (j = 0; j < n; j++) {
                vl[i * ldvl + j] *= tmp;
            }
        }
    }
    if (compvr) {
        for (i = 0; i < n; i++) {
            tmp = 0.0;
            for (j = 0; j < n; j++) {
                tmp += vr[i * ldvr + j].real() * vr[i * ldvr + j].real();
                tmp += vr[i * ldvr + j].imag() * vr[i * ldvr + j].imag();
            }
            tmp = 1.0 / std::sqrt(tmp);
            for (j = 0; j < n; j++) {
                vr[i * ldvr + j] *= tmp;
            }
        }
    }
    work[0] = ninfinite;
    work[1] = tol;
}