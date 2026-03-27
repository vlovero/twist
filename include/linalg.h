#ifndef TWIST_LINALG_H
#define TWIST_LINALG_H

#include <complex>

extern "C" {
    int openblas_get_num_threads();
    void openblas_set_num_threads_(int *);
    // BLAS/LAPACK routines used for the the new dggev4 routine
    // BLAS ROUTINES
    void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const double *alpha, const double *A, const int *lda, double *B, const int *ldb);

    // LAPACK ROUTINES
    // Hessenberg
    void dgehrd_(const int *n, const int *ilo, const int *ihi, double *A, const int *lda, double *tau, double *work, const int *lwork, int *info);
    void dormhr_(const char *side, const char *trans, const int *m, const int *n, const int *ilo, const int *ihi, const double *A, const int *lda, const double *tau, double *C, const int *ldc, double *work, const int *lwork, int *info);
    void dorghr_(const int *n, const int *ilo, const int *ihi, double *A, const int *lda, const double *tau, double *work, const int *lwork, int *info);
    // RQ
    void dgerqf_(const int *m, const int *n, double *A, const int *lda, double *tau, double *work, const int *lwork, int *info);
    void dormrq_(const char *side, const char *trans, const int *m, const int *n, const int *k, const double *A, const int *lda, const double *tau, double *C, const int *ldc, double *work, const int *lwork, int *info);
    // QR
    void dgeqrf_(const int *m, const int *n, double *A, const int *lda, double *tau, double *work, const int *lwork, int *info);
    void dormqr_(const char *side, const char *trans, const int *m, const int *n, const int *k, const double *A, const int *lda, const double *tau, double *C, const int *ldc, double *work, const int *lwork, int *info);
    void dorgqr_(const int *m, const int *n, const int *k, double *A, const int *lda, const double *tau, double *work, const int *lwork, int *info);
    void dgeqp3_(const int32_t *m, const int32_t *n, double *A, const int32_t *lda, int32_t *JPVT, double *tau, double *work, const int32_t *lwork, int32_t *info);

    // permute
    void dlapmr_(const int32_t *forwrd, const int32_t *m, const int32_t *n, double *X, const int32_t *ldx, int32_t *K);

    // balance
    void dggbal_(const char *job, const int *n, double *A, const int *ldA, double *B, const int *ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info);

    // qz and eig
    void dlaqz0_(const char *wants, const char *wantq, const char *wantz, const int *n, const int *ilo, const int *ihi, double *A, const int *lda, double *B, const int *ldb, double *alphar, double *alphai, double *beta, double *Q, const int *ldq, double *Z, const int *ldz, double *work, const int *lwork, const int *rec, int *info);
    void dtgevc_(const char *side, const char *howmny, const int *select, const int *n, const double *s, const int *lds, const double *p, const int *ldp, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);

    // misc
    double dlange_(const char *norm, const int *m, const int *n, const double *A, const int *ldA, double *work);

    // common routines used throughout the program
    void dgemv_(const char *trans, const int *m, const int *n, const double *alpha, const double *A, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *, const int *, const double *, const int *, const double *, double *, const int *);
    void dgetrf_(const int *m, const int *n, double *A, const int *lda, int *ipiv, int *info);
    void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *A, const int *lda, const int *ipiv, double *B, const int *ldb, int *info);
    void dgesdd_(const char *jobz, const int *m, const int *n, double *A, const int *lda, double *S, double *U, const int *ldu, double *VT, const int *ldvt, double *work, const int *lwork, int *iwork, int *info);
    void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *A, const int *lda, double *wr, double *wi, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void dgesv_(const int *n, const int *nrhs, double *A, const int *ldA, int *ipiv, double *b, const int *ldb, int *info);
    void dlapmt_(const int32_t *forwrd, const int32_t *m, const int32_t *n, double *X, const int32_t *ldx, int32_t *K);
    void dtrtri_(const char *uplo, const char *diag, const int *n, double *A, const int *ldA, int *info);
    void dggev3_(const char *jobvl, const char *jobvr, const int *n, double *A, const int *lda, double *B, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void dlacpy_(const char *uplo, const int *m, const int *n, const double *a, const int *lda, double *b, const int *ldb);
    void dgees_(const char *jobvs, const char *sort, int (*select)(double *, double *), const int *n, double *a, const int *lda, int *sdim, double *wr, double *wi, double *vs, const int *ldvs, double *work, const int *lwork, int *bwork, int *info);
    void dtrevc3_(const char *side, const char *howmny, const int *select, const int *n, const double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, const int *lwork, int *info);
    void dtrsen_(const char *job, const char *compq, const int *select, const int *n, double *t, const int *ldt, double *q, const int *ldq, double *wr, double *wi, int *m, double *s, double *sep, double *work, const int *lwork, int *iwork, const int *liwork, int *info);
}

double dlange(const char *norm, const int m, const int n, const double *A, const int ldA, double *work);
void dgeev(const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info);
void dgetrf(const int m, const int n, double *A, const int lda, int *ipiv, int *info);
void dgetrs(const char *trans, const int n, const int nrhs, const double *A, const int lda, const int *ipiv, double *B, const int ldb, int *info);
void dgesdd(const char *jobz, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, double *work, const int lwork, int *iwork, int *info);
void dgemv(const char *trans, const int m, const int n, const double alpha, const double *A, const int lda, const double *x, const int incx, const double beta, double *y, const int incy);
void dgemm(const char *transa, const char *transb, const int nrow_op_a, const int ncol_op_b, const int nrow_op_b, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc);
void dgesv(const int n, const int nrhs, double *A, const int ldA, int *ipiv, double *b, const int ldb, int *info);
void dggev3(const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *B, const int ldb, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info);
void dgesdd(const char *jobz, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, double *work, const int lwork, int *iwork, int *info);
void dtgevc(const char *side, const char *howmny, const int *select, const int n, const double *s, const int lds, const double *p, const int ldp, double *vl, const int ldvl, double *vr, const int ldvr, const int mm, int *m, double *work, int *info);
void dlapmr(const int32_t forwrd, const int32_t m, const int32_t n, double *X, const int32_t ldx, int32_t *K);
void dlaqz0(const char *job, const char *compq, const char *compz, const int n, const int ilo, const int ihi, double *A, const int lda, double *B, const int ldb, double *alphar, double *alphai, double *beta, double *Q, const int ldq, double *Z, const int ldz, double *work, const int lwork, const int rec, int *info);
void dggbal(const char *job, const int n, double *A, const int ldA, double *B, const int ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info);
void dorgqr(const int m, const int n, const int k, double *A, const int lda, const double *tau, double *work, const int lwork, int *info);
void dorghr(const int n, const int ilo, const int ihi, double *A, const int lda, const double *tau, double *work, const int lwork, int *info);
void dgeqp3(const int32_t m, const int32_t n, double *A, const int32_t lda, int32_t *JPVT, double *tau, double *work, const int32_t lwork, int32_t *info);
void dgehrd(const int n, const int ilo, const int ihi, double *A, const int lda, double *tau, double *work, const int lwork, int *info);
void dormhr(const char *side, const char *trans, const int m, const int n, const int ilo, const int ihi, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info);
void dgeqrf(const int m, const int n, double *A, const int lda, double *tau, double *work, const int lwork, int *info);
void dormqr(const char *side, const char *trans, const int m, const int n, const int k, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info);
void dgerqf(const int m, const int n, double *A, const int lda, double *tau, double *work, const int lwork, int *info);
void dormrq(const char *side, const char *trans, const int m, const int n, const int k, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info);
void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const double alpha, const double *A, const int lda, double *B, const int ldb);
void dlacpy(const char *uplo, const int m, const int n, const double *a, const int lda, double *b, const int ldb);
void dgees(const char *jobvs, const char *sort, int (*select)(double *, double *), const int n, double *a, const int lda, int *sdim, double *wr, double *wi, double *vs, const int ldvs, double *work, const int lwork, int *bwork, int *info);
void dtrevc3(const char *side, const char *howmny, const int *select, const int n, const double *t, const int ldt, double *vl, const int ldvl, double *vr, const int ldvr, const int mm, int *m, double *work, const int lwork, int *info);
void dtrsen(const char *job, const char *compq, const int *select, const int n, double *t, const int ldt, double *q, const int ldq, double *wr, double *wi, int *m, double *s, double *sep, double *work, const int lwork, int *iwork, const int liwork, int *info);

void dlatrn(const ptrdiff_t n, double *A, const ptrdiff_t ldA);
void dggev4(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info);

// complex routines
extern "C" {
    // BLAS ROUTINES
    void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, std::complex<double> *B, const int *ldb);
    void zgemm_(const char *, const char *, const int *, const int *, const int *, const std::complex<double> *, const std::complex<double> *, const int *, const std::complex<double> *, const int *, const std::complex<double> *, std::complex<double> *, const int *);
    void zgemv_(const char *trans, const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);

    // LAPACK ROUTINES
    // Hessenberg
    void zgehrd_(const int *n, const int *ilo, const int *ihi, std::complex<double> *A, const int *lda, std::complex<double> *tau, std::complex<double> *work, const int *lwork, int *info);
    void zunmhr_(const char *side, const char *trans, const int *m, const int *n, const int *ilo, const int *ihi, const std::complex<double> *A, const int *lda, const std::complex<double> *tau, std::complex<double> *C, const int *ldc, std::complex<double> *work, const int *lwork, int *info);
    void zunghr_(const int *n, const int *ilo, const int *ihi, std::complex<double> *A, const int *lda, const std::complex<double> *tau, std::complex<double> *work, const int *lwork, int *info);
    // RQ
    void zgerqf_(const int *m, const int *n, std::complex<double> *A, const int *lda, std::complex<double> *tau, std::complex<double> *work, const int *lwork, int *info);
    void zunmrq_(const char *side, const char *trans, const int *m, const int *n, const int *k, const std::complex<double> *A, const int *lda, const std::complex<double> *tau, std::complex<double> *C, const int *ldc, std::complex<double> *work, const int *lwork, int *info);
    // QR
    void zgeqrf_(const int *m, const int *n, std::complex<double> *A, const int *lda, std::complex<double> *tau, std::complex<double> *work, const int *lwork, int *info);
    void zunmqr_(const char *side, const char *trans, const int *m, const int *n, const int *k, const std::complex<double> *A, const int *lda, const std::complex<double> *tau, std::complex<double> *C, const int *ldc, std::complex<double> *work, const int *lwork, int *info);
    void zungqr_(const int *m, const int *n, const int *k, std::complex<double> *A, const int *lda, const std::complex<double> *tau, std::complex<double> *work, const int *lwork, int *info);
    void zgeqp3_(const int *m, const int *n, std::complex<double> *A, const int *lda, int *JPVT, std::complex<double> *tau, std::complex<double> *work, const int *lwork, double *rwork, int *info);

    // permute
    void zlapmr_(const int *forwrd, const int *m, const int *n, std::complex<double> *X, const int *ldx, int32_t *K);

    // balance
    void zggbal_(const char *job, const int *n, std::complex<double> *A, const int *ldA, std::complex<double> *B, const int *ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info);

    // qz and eig
    void zlaqz0_(const char *wants, const char *wantq, const char *wantz, const int *n, const int *ilo, const int *ihi, std::complex<double> *A, const int *lda, std::complex<double> *B, const int *ldb, std::complex<double> *alpha, std::complex<double> *beta, std::complex<double> *Q, const int *ldq, std::complex<double> *Z, const int *ldz, std::complex<double> *work, const int *lwork, double *rwork, const int *rec, int *info);
    void ztgevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<double> *s, const int *lds, const std::complex<double> *p, const int *ldp, std::complex<double> *vl, const int *ldvl, std::complex<double> *vr, const int *ldvr, const int *mm, int *m, std::complex<double> *work, double *rwork, int *info);

    // misc
    double zlange_(const char *norm, const int *m, const int *n, const std::complex<double> *A, const int *ldA, double *work);

    void zgetrf_(const int *m, const int *n, std::complex<double> *A, const int *lda, int *ipiv, int *info);
    void zgetrs_(const char *trans, const int *n, const int *nrhs, const std::complex<double> *A, const int *lda, const int *ipiv, std::complex<double> *B, const int *ldb, int *info);
    void zgeev_(const char *jobvl, const char *jobvr, const int *n, std::complex<double> *A, const int *lda, std::complex<double> *W, std::complex<double> *VL, const int *ldvl, std::complex<double> *VR, const int *ldvr, std::complex<double> *work, const int *lwork, double *rwork, int *info);
    void zlacpy_(const char *uplo, const int *m, const int *n, const std::complex<double> *a, const int *lda, std::complex<double> *b, const int *ldb);
}

double zlange(const char *norm, const int m, const int n, const std::complex<double> *A, const int ldA, double *work);
void ztgevc(const char *side, const char *howmny, const int *select, const int n, const std::complex<double> *s, const int lds, const std::complex<double> *p, const int ldp, std::complex<double> *vl, const int ldvl, std::complex<double> *vr, const int ldvr, const int mm, int *m, std::complex<double> *work, double *rwork, int *info);
void zlapmr(const int32_t forwrd, const int32_t m, const int32_t n, std::complex<double> *X, const int32_t ldx, int32_t *K);
void zlaqz0(const char *job, const char *compq, const char *compz, const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, std::complex<double> *B, const int ldb, std::complex<double> *alpha, std::complex<double> *beta, std::complex<double> *Q, const int ldq, std::complex<double> *Z, const int ldz, std::complex<double> *work, const int lwork, double *rwork, int rec, int *info);
void zggbal(const char *job, const int n, std::complex<double> *A, const int ldA, std::complex<double> *B, const int ldB, int *ilo, int *ihi, double *lscale, double *rscale, double *work, int *info);
void zungqr(const int m, const int n, const int k, std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info);
void zunghr(const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info);
void zgeqp3(const int32_t m, const int32_t n, std::complex<double> *A, const int32_t lda, int32_t *JPVT, std::complex<double> *tau, std::complex<double> *work, const int32_t lwork, double *rwork, int32_t *info);
void zgehrd(const int n, const int ilo, const int ihi, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info);
void zunmhr(const char *side, const char *trans, const int m, const int n, const int ilo, const int ihi, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info);
void zgeqrf(const int m, const int n, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info);
void zunmqr(const char *side, const char *trans, const int m, const int n, const int k, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info);
void zgerqf(const int m, const int n, std::complex<double> *A, const int lda, std::complex<double> *tau, std::complex<double> *work, const int lwork, int *info);
void zunmrq(const char *side, const char *trans, const int m, const int n, const int k, const std::complex<double> *A, const int lda, const std::complex<double> *tau, std::complex<double> *C, const int ldc, std::complex<double> *work, const int lwork, int *info);
void ztrsm(const char *side, const char *uplo, const char *transa, const char *diag, const int m, const int n, const std::complex<double> alpha, const std::complex<double> *A, const int lda, std::complex<double> *B, const int ldb);
void zlacpy(const char *uplo, const int m, const int n, const std::complex<double> *a, const int lda, std::complex<double> *b, const int ldb);

void zlatrn(const ptrdiff_t n, std::complex<double> *A, const ptrdiff_t ldA);
void zggev4(const char *jobvl, const char *jobvr, const int n, std::complex<double> *A, const int ldA, std::complex<double> *B, const int ldB, std::complex<double> *alpha, std::complex<double> *beta, std::complex<double> *vl, const int ldvl, std::complex<double> *vr, const int ldvr, std::complex<double> *work, const int lwork, double *rwork, int *jpvt, int *info);

#endif // TWIST_LINALG_H