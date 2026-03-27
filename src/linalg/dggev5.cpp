#include "fmt/core.h"
#include "linalg.h"
#include "shared.h"
#include <omp.h>

extern void dgeqp4(int *m, int *n, double *A, int *lda, int *jpvt, double *tau, double *work, int *lwork, int *info);
extern void dlatrn(const ptrdiff_t n, double *A, const ptrdiff_t ldA);
void dgghd4(const char *compq, const char *compz, const ptrdiff_t n, const int ilo, const int ihi, double *A, const ptrdiff_t ldA, double *B, const ptrdiff_t ldB, double *Q, const ptrdiff_t ldQ, double *Z, const ptrdiff_t ldZ, double *work, int lwork, int *info);
void dlarev(const char *which, const ptrdiff_t n, double *A, ptrdiff_t ldA);

void dgeqp4(int m, int n, double *A, int lda, int *jpvt, double *tau, double *work, int lwork, int *info)
{
    dgeqp4(&m, &n, A, &lda, jpvt, tau, work, &lwork, info);
}

void _dggprp(const bool wantq, const bool wantz, const int n, double *A, const int ldA, double *B, const int ldB, double *Q, const int ldQ, double *Z, const int ldZ, double *work, const int lwork, int *jpvt, int *info)
{
    double tol, normb, dummy[1];
    int ierr, workneeded;
    ptrdiff_t i, j;

    workneeded = 0;
    dgeqp4(n, n, B, ldB, jpvt, NULL, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)*dummy);
    dormqr("R", "N", n, n, n, B, ldB, NULL, A, ldA, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)*dummy);
    dorgqr(n, n, n, Z, ldZ, NULL, dummy, -1, &ierr);
    workneeded = std::max(workneeded, (int)*dummy);
    if (lwork == -1) {
        *info = 0;
        *work = workneeded;
        return;
    }

    normb = dlange("F", n, n, B, ldB, work);
    tol = std::max(1.2 * n, normb) * std::numeric_limits<double>::epsilon();

    // B = B^T
    dlatrn(n, B, ldB);

    // B = B[:, ::-1]
    dlarev("C", n, B, ldB);

    // RRQR(B)
    memset(jpvt, 0, sizeof(int) * n);
    dgeqp4(n, n, B, ldB, jpvt, work, &work[n], lwork - n, &ierr);

    // A = A[::-1]
    dlarev("R", n, A, ldA);
    // A = A[p]
    dlapmr(1, n, n, A, ldA, jpvt);
    // A = AQ
    dormqr("R", "N", n, n, n, B, ldB, work, A, ldA, &work[n], lwork - n, &ierr);
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
        dorgqr(n, n, n, Z, ldZ, work, &work[n], lwork - n, &ierr);
        dlarev("C", n, Z, ldZ);
    }
    // A = A[::-1][:, ::-1]
    dlarev("B", n, A, ldA);
    // B = B.T[::-1][:, ::-1]
    for (i = 0; i < (n - 1); i++) {
        for (j = i + 1; j < n; j++) {
            B[i * ldB + j] = 0;
        }
    }
    dlatrn(n, B, ldB);
    dlarev("B", n, B, ldB);
    work[0] = normb;
    work[1] = tol;
}


void dggev5(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info)
{
    int workneeded, ninfinite, in;
    double dummy[1], tol, tmp;
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

    // argument checks

    // workspace queries
    workneeded = 0;
    _dggprp(compvl, compvr, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, dummy, -1, jpvt, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dgghd4(jobvl, jobvr, n, 1, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, dummy, -1, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dgeqrf(n, n, A, ldA, NULL, dummy, -1, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dormqr("L", "T", n, n, n, A, ldA, NULL, NULL, n, dummy, -1, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dgerqf(n, n, A, ldA, NULL, dummy, -1, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dormrq("L", "T", n, n, n, A, ldA, NULL, NULL, n, dummy, -1, info);
    workneeded = std::max(workneeded, (int)*dummy);
    dlaqz0(&jobqz, jobvl, jobvr, n, 1, n, A, ldA, B, ldB, alphar, alphai, beta, vl, ldvl, vr, ldvr, dummy, -1, 0, info);
    workneeded = std::max(workneeded, (int)*dummy);

    if (lwork == -1) {
        *info = 0;
        *work = workneeded;
        return;
    }

    // Step 1: Preprocessing
    // TIME_CODE("preprocessing (dggev5)", _dggprp(compvl, compvr, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, work, lwork, jpvt, info););
    _dggprp(compvl, compvr, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, work, lwork, jpvt, info);
    tol = work[1];
    ninfinite = 0;
    for (k = 0; k < n; k++) {
        if (tol < std::abs(B[k * ldB + k])) {
            break;
        }
        ninfinite++;
        memset(&B[ldB * k], 0, ninfinite * sizeof(double));
    }
    k = ninfinite;
    // Step 2: Deflation
    if (ninfinite) {
        // QR portion
        // Q, A[:, :k] = np.linalg.qr(A[:, :k], mode="complete")
        dgeqrf(n, k, A, ldA, work, &work[k], lwork - k, info);
        // A[:, k:] = Q.T @ A[:, k:]
        dormqr("L", "T", n, n - k, k, A, ldA, work, &A[k * ldA], ldA, &work[n], lwork - n, info);
        // B[:, k:] = Q.T @ B[:, k:]
        dormqr("L", "T", n, n - k, k, A, ldA, work, &B[k * ldB], ldB, &work[n], lwork - n, info);
        if (compvl) {
            // Q = QA @ Q
            dormqr("R", "N", n, n, k, A, ldA, work, vl, ldvl, &work[n], lwork - n, info);
        }
        // RQ portion
        // B[k:, k:], Q = rq(B[k:, k:])
        dgerqf(n - k, n - k, &B[k * ldB + k], ldB, work, &work[n - k], lwork - n + k, info);
        // A[:, k:] = A[:, k:] @ Q.T
        dormrq("R", "T", n, n - k, n - k, &B[k * ldB + k], ldB, work, &A[k * ldA], ldA, &work[n - k], lwork - n + k, info);
        // B[:k, k:] = B[:k, k:] @ Q.T
        dormrq("R", "T", k, n - k, n - k, &B[k * ldB + k], ldB, work, &B[k * ldB], ldB, &work[n - k], lwork - n + k, info);
        if (compvr) {
            // Z[:, k:] = Z[:, k:] @ QB.T
            dormrq("R", "T", n, n - k, n - k, &B[k * ldB + k], ldB, work, &vr[k * ldvr], ldvr, &work[n - k], lwork - n + k, info);
        }

        for (i = 0; i < k; i++) {
            memset(&A[i * ldA + i + 1], 0, (n - i - 1) * sizeof(double));
        }
        for (i = k; i < n; i++) {
            memset(&B[i * ldB + i + 1], 0, (n - i - 1) * sizeof(double));
        }
    }

    // add small pertubation to B if needed
    for (i = k; i < n; i++) {
        if (tol < std::abs(B[i * ldB + i])) {
            break;
        }
        B[i * ldB + i] = tol;
    }

    // Step 3: Hessenberg-Triangular Reduction
    dgghd4(jobvl, jobvr, n, k, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, work, lwork, info);

    // Step 4: QZ
    // perform qz on remaining portion of A and B
    if (ninfinite != n) {
        dlaqz0(&jobqz, jobvl, jobvr, n, ninfinite + 1, n, A, ldA, B, ldB, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, 0, info);
    }
    // Step 5: Eigenvectors
    if (compvl || compvr) {
        dtgevc(&jobvec, "B", NULL, n, A, ldA, B, ldB, vl, ldvl, vr, ldvr, n, &in, work, info);
    }
    if (compvl) {
        for (i = 0; i < n; i++) {
            tmp = 0.0;
            if (alphai[i] != 0) {
                // complex case
                for (j = 0; j < n; j++) {
                    tmp += vl[(i + 0) * ldvl + j] * vl[(i + 0) * ldvl + j];
                    tmp += vl[(i + 1) * ldvl + j] * vl[(i + 1) * ldvl + j];
                }
                tmp = 1.0 / std::sqrt(tmp);
                for (j = 0; j < n; j++) {
                    vl[(i + 0) * ldvl + j] *= tmp;
                    vl[(i + 1) * ldvl + j] *= tmp;
                }
                i++;
            }
            else if (alphai[i] == 0) {
                // real case
                for (j = 0; j < n; j++) {
                    tmp += vl[i * ldvl + j] * vl[i * ldvl + j];
                }
                tmp = 1.0 / std::sqrt(tmp);
                for (j = 0; j < n; j++) {
                    vl[i * ldvl + j] *= tmp;
                }
            }
        }
    }
    if (compvr) {
        for (i = 0; i < n; i++) {
            tmp = 0.0;
            if (alphai[i] != 0) {
                // complex case
                for (j = 0; j < n; j++) {
                    tmp += vr[(i + 0) * ldvr + j] * vr[(i + 0) * ldvr + j];
                    tmp += vr[(i + 1) * ldvr + j] * vr[(i + 1) * ldvr + j];
                }
                tmp = 1.0 / std::sqrt(tmp);
                for (j = 0; j < n; j++) {
                    vr[(i + 0) * ldvr + j] *= tmp;
                    vr[(i + 1) * ldvr + j] *= tmp;
                }
                i++;
            }
            else if (alphai[i] == 0) {
                // real case
                for (j = 0; j < n; j++) {
                    tmp += vr[i * ldvr + j] * vr[i * ldvr + j];
                }
                tmp = 1.0 / std::sqrt(tmp);
                for (j = 0; j < n; j++) {
                    vr[i * ldvr + j] *= tmp;
                }
            }
        }
    }
    *work = ninfinite;
}
