#include "linalg.h"

void dgeev(const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info)
{
    dgeev_(jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

void dgetrf(const int m, const int n, double *A, const int lda, int *ipiv, int *info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

void dgetrs(const char *trans, const int n, const int nrhs, const double *A, const int lda, const int *ipiv, double *B, const int ldb, int *info)
{
    dgetrs_(trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

void dgemv(const char *trans, const int m, const int n, const double alpha, const double *A, const int lda, const double *x, const int incx, const double beta, double *y, const int incy)
{
    dgemv_(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

void dgemm(const char *transa, const char *transb, const int nrow_op_a, const int ncol_op_b, const int nrow_op_b, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc)
{
    dgemm_(transa, transb, &nrow_op_a, &ncol_op_b, &nrow_op_b, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void dgesv(const int n, const int nrhs, double *A, const int ldA, int *ipiv, double *b, const int ldb, int *info)
{
    dgesv_(&n, &nrhs, A, &ldA, ipiv, b, &ldb, info);
}

void dggev3(const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *B, const int ldb, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info)
{
    dggev3_(jobvl, jobvr, &n, A, &lda, B, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

void dgesdd(const char *jobz, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, double *work, const int lwork, int *iwork, int *info)
{
    dgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, iwork, info);
}

void dtgevc(const char *side, const char *howmny, const int *select, const int n, const double *s, const int lds, const double *p, const int ldp, double *vl, const int ldvl, double *vr, const int ldvr, const int mm, int *m, double *work, int *info)
{
    dtgevc_(side, howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, m, work, info);
}

double dlange(const char *norm, const int m, const int n, const double *A, const int ldA, double *work)
{
    return dlange_(norm, &m, &n, A, &ldA, work);
}

void dlapmr(const int32_t forwrd, const int32_t m, const int32_t n, double *X, const int32_t ldx, int32_t *K)
{
    dlapmr_(&forwrd, &m, &n, X, &ldx, K);
}

void dlaqz0(const char *job, const char *compq, const char *compz, const int n, const int ilo, const int ihi, double *A, const int lda, double *B, const int ldb, double *alphar, double *alphai, double *beta, double *Q, const int ldq, double *Z, const int ldz, double *work, const int lwork, const int rec, int *info)
{
    dlaqz0_(job, compq, compz, &n, &ilo, &ihi, A, &lda, B, &ldb, alphar, alphai, beta, Q, &ldq, Z, &ldz, work, &lwork, &rec, info);
}

void dggbal(const char *job, const int n, double *A, const int ldA, double *B, const int ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info)
{
    dggbal_(job, &n, A, &ldA, B, &ldB, ilo, ihi, lscale, rscale, work, info);
}

void dorgqr(const int m, const int n, const int k, double *A, const int lda, const double *tau, double *work, const int lwork, int *info)
{
    dorgqr_(&m, &n, &k, A, &lda, tau, work, &lwork, info);
}

void dorghr(const int n, const int ilo, const int ihi, double *A, const int lda, const double *tau, double *work, const int lwork, int *info)
{
    dorghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, info);
}

void dgeqp3(const int32_t m, const int32_t n, double *A, const int32_t lda, int32_t *JPVT, double *tau, double *work, const int32_t lwork, int32_t *info)
{
    dgeqp3_(&m, &n, A, &lda, JPVT, tau, work, &lwork, info);
}

void dgehrd(const int n, const int ilo, const int ihi, double *A, const int lda, double *tau, double *work, const int lwork, int *info)
{
    dgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, info);
}

void dormhr(const char *side, const char *trans, const int m, const int n, const int ilo, const int ihi, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info)
{
    dormhr_(side, trans, &m, &n, &ilo, &ihi, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void dgeqrf(const int m, const int n, double *A, const int lda, double *tau, double *work, const int lwork, int *info)
{
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}

void dormqr(const char *side, const char *trans, const int m, const int n, const int k, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info)
{
    dormqr_(side, trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void dgerqf(const int m, const int n, double *A, const int lda, double *tau, double *work, const int lwork, int *info)
{
    dgerqf_(&m, &n, A, &lda, tau, work, &lwork, info);
}

void dormrq(const char *side, const char *trans, const int m, const int n, const int k, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info)
{
    dormrq_(side, trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const double alpha, const double *A, const int lda, double *B, const int ldb)
{
    dtrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void dlacpy(const char *uplo, const int m, const int n, const double *a, const int lda, double *b, const int ldb)
{
    dlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}

void dgees(const char *jobvs, const char *sort, int (*select)(double *, double *), const int n, double *a, const int lda, int *sdim, double *wr, double *wi, double *vs, const int ldvs, double *work, const int lwork, int *bwork, int *info)
{
    dgees_(jobvs, sort, select, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work, &lwork, bwork, info);
}

void dtrevc3(const char *side, const char *howmny, const int *select, const int n, const double *t, const int ldt, double *vl, const int ldvl, double *vr, const int ldvr, const int mm, int *m, double *work, const int lwork, int *info)
{
    dtrevc3_(side, howmny, select, &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, m, work, &lwork, info);
}

void dtrsen(const char *job, const char *compq, const int *select, const int n, double *t, const int ldt, double *q, const int ldq, double *wr, double *wi, int *m, double *s, double *sep, double *work, const int lwork, int *iwork, const int liwork, int *info)
{
    dtrsen_(job, compq, select, &n, t, &ldt, q, &ldq, wr, wi, m, s, sep, work, &lwork, iwork, &liwork, info);
}


void ztgevc(const char *side, const char *howmny, const int *select, const int n, const std::complex<double> *s, const int lds, const std::complex<double> *p, const int ldp, std::complex<double> *vl, const int ldvl, std::complex<double> *vr, const int ldvr, const int mm, int *m, std::complex<double> *work, double *rwork, int *info)
{
    ztgevc_(side, howmny, select, &n, s, &lds, p, &ldp, vl, &ldvl, vr, &ldvr, &mm, m, work, rwork, info);
}

double zlange(const char *norm, const int m, const int n, const std::complex<double> *A, const int ldA, double *work)
{
    return zlange_(norm, &m, &n, A, &ldA, work);
}

void zlapmr(const int32_t forwrd, const int32_t m, const int32_t n, std::complex<double> *X, const int32_t ldx, int32_t *K)
{
    zlapmr_(&forwrd, &m, &n, X, &ldx, K);
}

void zlaqz0(const char *job, const char *compq, const char *compz, const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, std::complex<double> *B, const int ldb, std::complex<double> *alpha, std::complex<double> *beta, std::complex<double> *Q, const int ldq, std::complex<double> *Z, const int ldz, std::complex<double> *work, const int lwork, double *rwork, int rec, int *info)
{
    zlaqz0_(job, compq, compz, &n, &ilo, &ihi, A, &lda, B, &ldb, alpha, beta, Q, &ldq, Z, &ldz, work, &lwork, rwork, &rec, info);
}

void zggbal(const char *job, const int n, std::complex<double> *A, const int ldA, std::complex<double> *B, const int ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info)
{
    zggbal_(job, &n, A, &ldA, B, &ldB, ilo, ihi, lscale, rscale, work, info);
}

void zungqr(const int m, const int n, const int k, std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info)
{
    zungqr_(&m, &n, &k, A, &lda, tau, work, &lwork, info);
}

void zunghr(const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info)
{
    zunghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, info);
}

void zgeqp3(const int32_t m, const int32_t n, std::complex<double> *A, const int32_t lda, int32_t *JPVT, std::complex<double> *tau, std::complex<double> *work, const int32_t lwork, double *rwork, int32_t *info)
{
    zgeqp3_(&m, &n, A, &lda, JPVT, tau, work, &lwork, rwork, info);
}

void zgehrd(const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info)
{
    zgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, info);
}

void zunmhr(const char *side, const char *trans, const int m, const int n, const int ilo, const int ihi, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info)
{
    zunmhr_(side, trans, &m, &n, &ilo, &ihi, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void zgeqrf(const int m, const int n, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info)
{
    zgeqrf_(&m, &n, A, &lda, tau, work, &lwork, info);
}

void zunmqr(const char *side, const char *trans, const int m, const int n, const int k, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info)
{
    zunmqr_(side, trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void zgerqf(const int m, const int n, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info)
{
    zgerqf_(&m, &n, A, &lda, tau, work, &lwork, info);
}

void zunmrq(const char *side, const char *trans, const int m, const int n, const int k, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info)
{
    zunmrq_(side, trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, info);
}

void ztrsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const std::complex<double> alpha, const std::complex<double> *A, const int lda, std::complex<double> *B, const int ldb)
{
    ztrsm_(side, uplo, transa, diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

void zlacpy(const char *uplo, const int m, const int n, const std::complex<double> *a, const int lda, std::complex<double> *b, const int ldb)
{
    zlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}