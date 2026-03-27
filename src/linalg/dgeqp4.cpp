#include "linalg.h"
#include "shared.h"
#include <cmath>
#include <random>

extern "C" {
    void dswap_(const int *N, double *DX, const int *INCX, double *DY, const int *INCY);
    int ilaenv_(const int *ISPEC, const char *NAME, const char *OPTS, const int *N1, const int *N2, const int *N3, const int *N4);
    void dgeqpf_(const int *M, const int *N, double *A, const int *LDA, int *JPVT, double *TAU, double *WORK, int *INFO);
    void xerbla_(const char *SRNAME, const int *INFO);
    void dtrmm_(const char *SIDE, const char *UPLO, const char *TRANS, const char *DIAG, const int *M, const int *N, const double *ALPHA, const double *A, const int *LDA, double *B, const int *LDB);
    void dlarfb_(const char *SIDE, const char *TRANS, const char *DIRECTION, const char *VORDER, const int *M, const int *N, const int *K, const double *V, const int *LDV, const double *T, const int *LDT, double *C, const int *LDC, double *WORK, const int *LDWORK);
    void dlarf_(const char *SIDE, const int *M, const int *N, const double *V, const int *INCV, const double *TAU, double *C, const int *LDC, double *WORK);
    void dlarfg_(const int *N, double *ALPHA, double *X, const int *INCX, double *TAU);
    void dlarft_(const char *DIRECTION, const char *VORDER, const int *M, const int *N, const double *V, const int *LDV, const double *TAU, double *T, const int *LDT);
    int idamax_(const int *N, const double *DX, const int *INCX);
    double dlamch_(const char *CMACH);
    double dnrm2_(const int *N, const double *DX, const int *INCX);
}

// /* [x] */ double NoFLA_Normal_random_number(double mu, double sigma);
// /* [x] */ int NoFLA_Normal_random_matrix(int m_A, int n_A, double *buff_A, int ldim_A);
// /* [x] */ int NoFLA_HQRRP_WY_blk_var4(int m_A, int n_A, double *buff_A, int ldim_A, int *buff_jpvt, double *buff_tau, int nb_alg, int pp, int panel_pivoting);
// /* [x] */ int NoFLA_Downdate_Y(int m_U11, int n_U11, double *buff_U11, int ldim_U11, int m_U21, int n_U21, double *buff_U21, int ldim_U21, int m_A12, int n_A12, double *buff_A12, int ldim_A12, int m_T, int n_T, double *buff_T, int ldim_T, int m_Y2, int n_Y2, double *buff_Y2, int ldim_Y2, int
// m_G1, int n_G1, double *buff_G1, int ldim_G1, int m_G2, int n_G2, double *buff_G2, int ldim_G2);
// /* [x] */ int NoFLA_Apply_Q_WY_lhfc_blk_var4(int m_U, int n_U, double *buff_U, int ldim_U, int m_T, int n_T, double *buff_T, int ldim_T, int m_B, int n_B, double *buff_B, int ldim_B);
// /* [x] */ int NoFLA_Apply_Q_WY_rnfc_blk_var4(int m_U, int n_U, double *buff_U, int ldim_U, int m_T, int n_T, double *buff_T, int ldim_T, int m_B, int n_B, double *buff_B, int ldim_B);
// /* [x] */ int NoFLA_QRPmod_WY_unb_var4(int pivoting, int num_stages, int m_A, int n_A, double *buff_A, int ldim_A, int *buff_p, double *buff_t, int pivot_B, int m_B, double *buff_B, int ldim_B, int pivot_C, int m_C, double *buff_C, int ldim_C, int build_T, double *buff_T, int ldim_T);
// /* [x] */ int NoFLA_QRP_compute_norms(int m_A, int n_A, double *buff_A, int ldim_A, double *buff_d, double *buff_e);
// /* [x] */ int NoFLA_QRP_downdate_partial_norms(int m_A, int n_A, double *buff_d, int st_d, double *buff_e, int st_e, double *buff_wt, int st_wt, double *buff_A, int ldim_A);
// /* [x] */ int NoFLA_QRP_pivot_G_B_C(int j_max_col, int m_G, double *buff_G, int ldim_G, int pivot_B, int m_B, double *buff_B, int ldim_B, int pivot_C, int m_C, double *buff_C, int ldim_C, int *buff_p, double *buff_d, double *buff_e);

#define THRESHOLD_FOR_DGEQPF 1
#define THRESHOLD_FOR_DGEQP3 1


void generate_random_matrix(RP(double) first, RP(double) last, const int seed)
{
    std::default_random_engine engine(seed);
    std::normal_distribution<double> dist;
    std::generate(first, last, [&]() { return dist(engine); });
}

int NoFLA_Apply_Q_WY_lhfc_blk_var4(int m_U, int n_U, double *buff_U, int ldim_U, int m_T, int n_T, double *buff_T, int ldim_T, int m_B, int n_B, double *buff_B, int ldim_B, double *work)
{
    (void)m_U;
    (void)m_T;
    (void)n_T;
    // LWORK : n_B * n_U
    // It applies the transpose of a block transformation Q to a matrix B from
    // the left:
    //   B := Q' * B
    // where:
    //   Q = I - U * T' * U'.
    //
    double *buff_W;
    int ldim_W;

    // Create auxiliary object.
    buff_W = work;
    ldim_W = std::max(1, n_B);

    // Apply the block transformation.
    dlarfb_("L", "T", "F", "C", &m_B, &n_B, &n_U, buff_U, &ldim_U, buff_T, &ldim_T, buff_B, &ldim_B, buff_W, &ldim_W);

    return 0;
}

int NoFLA_Apply_Q_WY_rnfc_blk_var4(int m_U, int n_U, double *buff_U, int ldim_U, int m_T, int n_T, double *buff_T, int ldim_T, int m_B, int n_B, double *buff_B, int ldim_B, double *work)
{
    (void)m_U;
    (void)m_T;
    (void)n_T;
    // LWORK : m_B * n_U
    // It applies a block transformation Q to a matrix B from the right:
    //   B = B * Q
    // where:
    //   Q = I - U * T' * U'.
    //
    double *buff_W;
    int ldim_W;

    // Create auxiliary object.
    //// FLA_Obj_create_conf_to( FLA_TRANSPOSE, B1, & W );
    buff_W = work;
    ldim_W = std::max(1, m_B);

    // Apply the block transformation.
    dlarfb_("R", "N", "F", "C", &m_B, &n_B, &n_U, buff_U, &ldim_U, buff_T, &ldim_T, buff_B, &ldim_B, buff_W, &ldim_W);

    return 0;
}

int NoFLA_QRP_compute_norms(int m_A, int n_A, double *buff_A, int ldim_A, double *buff_d, double *buff_e)
{
    //
    // It computes the column norms of matrix A. The norms are stored into
    // vectors d and e.
    //
    int j, i_one = 1;

    // Main loop.
    for (j = 0; j < n_A; j++) {
        *buff_d = dnrm2_(&m_A, buff_A, &i_one);
        *buff_e = *buff_d;
        buff_A += ldim_A;
        buff_d++;
        buff_e++;
    }

    return 0;
}

int NoFLA_QRP_downdate_partial_norms(int m_A, int n_A, double *buff_d, int st_d, double *buff_e, int st_e, double *buff_wt, int st_wt, double *buff_A, int ldim_A)
{
    //
    // It updates (downdates) the column norms of matrix A. It uses Drmac's method.
    //
    int j, i_one = 1;
    double *ptr_d, *ptr_e, *ptr_wt, *ptr_A;
    double temp, temp2, temp5, tol3z;

    /*
  *
  *           Update partial column norms
  *
            DO 30 J = I + 1, N
               IF( WORK( J ).NE.ZERO ) THEN
  *
  *                 NOTE: The following 4 lines follow from the analysis in
  *                 Lapack Working Note 176.
  *
                  TEMP = ABS( A( I, J ) ) / WORK( J )
                  TEMP = MAX( ZERO, ( ONE+TEMP )*( ONE-TEMP ) )
                  TEMP2 = TEMP*( WORK( J ) / WORK( N+J ) )**2
                  IF( TEMP2 .LE. TOL3Z ) THEN
                     IF( M-I.GT.0 ) THEN
                        WORK( J ) = DNRM2( M-I, A( I+1, J ), 1 )
                        WORK( N+J ) = WORK( J )
                     ELSE
                        WORK( J ) = ZERO
                        WORK( N+J ) = ZERO
                     END IF
                  ELSE
                     WORK( J ) = WORK( J )*SQRT( TEMP )
                  END IF
               END IF
   30       CONTINUE
    */

    // Some initializations.
    tol3z = std::sqrt(dlamch_("Epsilon"));
    ptr_d = buff_d;
    ptr_e = buff_e;
    ptr_wt = buff_wt;
    ptr_A = buff_A;

    // Main loop.
    for (j = 0; j < n_A; j++) {
        if (*ptr_d != 0.0) {
            temp = std::abs(*ptr_wt) / *ptr_d;
            temp = std::max(0.0, (1.0 + temp) * (1 - temp));
            temp5 = *ptr_d / *ptr_e;
            temp2 = temp * temp5 * temp5;
            if (temp2 <= tol3z) {
                if (m_A > 0) {
                    *ptr_d = dnrm2_(&m_A, ptr_A, &i_one);
                    *ptr_e = *ptr_d;
                }
                else {
                    *ptr_d = 0.0;
                    *ptr_e = 0.0;
                }
            }
            else {
                *ptr_d = *ptr_d * std::sqrt(temp);
            }
        }
        ptr_A += ldim_A;
        ptr_d += st_d;
        ptr_e += st_e;
        ptr_wt += st_wt;
    }

    return 0;
}

int NoFLA_QRP_pivot_G_B_C(int j_max_col, int m_G, double *buff_G, int ldim_G, int pivot_B, int m_B, double *buff_B, int ldim_B, int pivot_C, int m_C, double *buff_C, int ldim_C, int *buff_p, double *buff_d, double *buff_e)
{
    //
    // It pivots matrix G, pivot vector p, and norms vectors d and e.
    // Matrices B and C are optionally pivoted.
    //
    int ival, i_one = 1;
    double *ptr_g1, *ptr_g2, *ptr_b1, *ptr_b2, *ptr_c1, *ptr_c2;

    // Swap columns of G, pivots, and norms.
    if (j_max_col != 0) {

        // Swap full column 0 and column "j_max_col" of G.
        ptr_g1 = &buff_G[0 + 0 * ldim_G];
        ptr_g2 = &buff_G[0 + j_max_col * ldim_G];
        dswap_(&m_G, ptr_g1, &i_one, ptr_g2, &i_one);

        // Swap full column 0 and column "j_max_col" of B.
        if (pivot_B) {
            ptr_b1 = &buff_B[0 + 0 * ldim_B];
            ptr_b2 = &buff_B[0 + j_max_col * ldim_B];
            dswap_(&m_B, ptr_b1, &i_one, ptr_b2, &i_one);
        }

        // Swap full column 0 and column "j_max_col" of C.
        if (pivot_C) {
            ptr_c1 = &buff_C[0 + 0 * ldim_C];
            ptr_c2 = &buff_C[0 + j_max_col * ldim_C];
            dswap_(&m_C, ptr_c1, &i_one, ptr_c2, &i_one);
        }

        // Swap element 0 and element "j_max_col" of pivot vector "p".
        ival = buff_p[j_max_col];
        buff_p[j_max_col] = buff_p[0];
        buff_p[0] = ival;

        // Copy norms of column 0 to column "j_max_col".
        buff_d[j_max_col] = buff_d[0];
        buff_e[j_max_col] = buff_e[0];
    }

    return 0;
}

int NoFLA_QRPmod_WY_unb_var4(int pivoting, int num_stages, int m_A, int n_A, double *buff_A, int ldim_A, int *buff_p, double *buff_t, int pivot_B, int m_B, double *buff_B, int ldim_B, int pivot_C, int m_C, double *buff_C, int ldim_C, int build_T, double *buff_T, int ldim_T, double *work)
{
    // LWORK : 3 * n_A
    // It computes an unblocked QR factorization of matrix A with or without
    // pivoting. Matrices B and C are optionally pivoted, and matrix T is
    // optionally built.
    //
    // Arguments:
    // "pivoting": If pivoting==1, then QR factorization with pivoting is used.
    // "numstages": It tells the number of columns that are factorized.
    //   If "num_stages" is negative, the whole matrix A is factorized.
    //   If "num_stages" is positive, only the first "num_stages" are factorized.
    // "pivot_B": if "pivot_B" is true, matrix "B" is pivoted too.
    // "pivot_C": if "pivot_C" is true, matrix "C" is pivoted too.
    // "build_T": if "build_T" is true, matrix "T" is built.
    //
    int j, mn_A, m_a21, m_A22, n_A22, n_dB, idx_max_col, i_one = 1, n_house_vector, m_rest;
    double *buff_d, *buff_e, *buff_workspace, diag;

    // Some initializations.
    mn_A = std::min(m_A, n_A);

    // Set the number of stages, if needed.
    if (num_stages < 0) {
        num_stages = mn_A;
    }

    // Create auxiliary vectors.
    buff_d = work;
    buff_e = &buff_d[n_A];
    buff_workspace = &buff_e[n_A];

    if (pivoting == 1) {
        // Compute initial norms of A into d and e.
        NoFLA_QRP_compute_norms(m_A, n_A, buff_A, ldim_A, buff_d, buff_e);
    }

    // Main Loop.
    for (j = 0; j < num_stages; j++) {
        n_dB = n_A - j;
        m_a21 = m_A - j - 1;
        m_A22 = m_A - j - 1;
        n_A22 = n_A - j - 1;

        if (pivoting == 1) {
            // Obtain the index of the column with largest 2-norm.
            idx_max_col = idamax_(&n_dB, &buff_d[j], &i_one) - 1;

            // Swap columns of A, B, C, pivots, and norms vectors.
            NoFLA_QRP_pivot_G_B_C(idx_max_col, m_A, &buff_A[0 + j * ldim_A], ldim_A, pivot_B, m_B, &buff_B[0 + j * ldim_B], ldim_B, pivot_C, m_C, &buff_C[0 + j * ldim_C], ldim_C, &buff_p[j], &buff_d[j], &buff_e[j]);
        }

        // Compute tau1 and u21 from alpha11 and a21 such that tau1 and u21
        // determine a Householder transform H such that applying H from the
        // left to the column vector consisting of alpha11 and a21 annihilates
        // the entries in a21 (and updates alpha11).
        n_house_vector = m_a21 + 1;
        dlarfg_(&n_house_vector, &buff_A[j + j * ldim_A], &buff_A[std::min(m_A - 1, j + 1) + j * ldim_A], &i_one, &buff_t[j]);

        // / a12t \ =  H / a12t \
        // \ A22  /      \ A22  /
        //
        // where H is formed from tau1 and u21.
        diag = buff_A[j + j * ldim_A];
        buff_A[j + j * ldim_A] = 1.0;
        m_rest = m_A22 + 1;
        dlarf_("Left", &m_rest, &n_A22, &buff_A[j + j * ldim_A], &i_one, &buff_t[j], &buff_A[j + (j + 1) * ldim_A], &ldim_A, buff_workspace);
        buff_A[j + j * ldim_A] = diag;

        if (pivoting == 1) {
            // Update partial column norms.
            NoFLA_QRP_downdate_partial_norms(m_A22, n_A22, &buff_d[j + 1], 1, &buff_e[j + 1], 1, &buff_A[j + (j + 1) * ldim_A], ldim_A, &buff_A[(j + 1) + std::min(n_A - 1, (j + 1)) * ldim_A], ldim_A);
        }
    }

    // Build T.
    if (build_T) {
        dlarft_("Forward", "Columnwise", &m_A, &num_stages, buff_A, &ldim_A, buff_t, buff_T, &ldim_T);
    }

    return 0;
}

int NoFLA_Downdate_Y(int m_U11, int n_U11, RP(double) buff_U11, int ldim_U11, int m_U21, int n_U21, RP(double) buff_U21, int ldim_U21, int m_A12, int n_A12, RP(double) buff_A12, int ldim_A12, int m_T, int n_T, RP(double) buff_T, int ldim_T, int m_Y2, int n_Y2, RP(double) buff_Y2, int ldim_Y2, int m_G1, int n_G1, RP(double) buff_G1, int ldim_G1, int m_G2, int n_G2, RP(double) buff_G2, int ldim_G2, RP(double) work)
{
    (void)n_U21;
    (void)n_A12;
    (void)m_G2;
    // LWORK : m_B * n_B
    // It downdates matrix Y, and updates matrix G.
    // Only Y2 of Y is updated.
    // Only G1 and G2 of G are updated.
    //
    // Y2 = Y2 - ( G1 - ( G1*U11 + G2*U21 ) * T11 * U11' ) * R12.
    //
    int i, j;
    double *buff_B;
    double d_one = 1.0;
    double d_minus_one = -1.0;
    int m_B = m_G1;
    int n_B = n_G1;
    int ldim_B = m_G1;

    // Create object B.
    buff_B = work;

    // B = G1.
    dlacpy_("A", &m_G1, &n_G1, buff_G1, &ldim_G1, buff_B, &ldim_B);

    // B = B * U11.
    dtrmm_("R", "L", "N", "U", &m_B, &n_B, &d_one, buff_U11, &ldim_U11, buff_B, &ldim_B);

    // B = B + G2 * U21.
    dgemm_("N", "N", &m_B, &n_B, &m_U21, &d_one, buff_G2, &ldim_G2, buff_U21, &ldim_U21, &d_one, buff_B, &ldim_B);

    // B = B * T11.
    // Used dtrmm instead of dtrsm because of using compact WY instead of UT.
    dtrmm_("R", "U", "N", "N", &m_B, &n_B, &d_one, buff_T, &ldim_T, buff_B, &ldim_B);

    // B = - B * U11^H.
    dtrmm_("R", "L", "C", "U", &m_B, &n_B, &d_minus_one, buff_U11, &ldim_U11, buff_B, &ldim_B);

    // B = G1 + B.
    //// FLA_Axpy( FLA_ONE, G1, B );
    for (j = 0; j < n_B; j++) {
        for (i = 0; i < m_B; i++) {
            buff_B[i + j * ldim_B] += buff_G1[i + j * ldim_G1];
        }
    }

    // Y2 = Y2 - B * R12.
    dgemm_("N", "N", &m_Y2, &n_Y2, &m_A12, &d_minus_one, buff_B, &ldim_B, buff_A12, &ldim_A12, &d_one, buff_Y2, &ldim_Y2);

    // GR = GR * Q
    NoFLA_Apply_Q_WY_rnfc_blk_var4(m_U11 + m_U21, n_U11, buff_U11, ldim_U11, m_T, n_T, buff_T, ldim_T, m_G1, n_G1 + n_G2, buff_G1, ldim_G1, &buff_B[m_B * n_B]);

    return 0;
}

int NoFLA_HQRRP_WY_blk_var4(int m_A, int n_A, RP(double) buff_A, int ldim_A, int *buff_jpvt, RP(double) buff_tau, int nb_alg, int pp, int panel_pivoting, RP(double) work)
{
    // work size = ((nb + pp) * (3 * n_A + m_A)) + max((3 * n_A), (nb * n_A), (nb + pp) * nb)
    int b, j, last_iter, mn_A, m_Y, n_Y, ldim_Y, m_V, n_V, ldim_V, m_W, n_W, ldim_W, n_VR, m_AB1, n_AB1, ldim_T1_T, m_A11, n_A11, m_A12, n_A12, m_A21, n_A21, m_A22, m_G, n_G, ldim_G;
    double *buff_Y, *buff_V, *buff_W, *buff_VR, *buff_YR, *buff_s, *buff_sB, *buff_s1, *buff_AR, *buff_AB1, *buff_A01, *buff_Y1, *buff_T1_T, *buff_A11, *buff_A21, *buff_A12, *buff_Y2, *buff_G, *buff_G1, *buff_G2, *workspace;
    int *buff_p, *buff_pB, *buff_p1;
    double d_zero = 0.0;
    double d_one = 1.0;

    // Check arguments.
    if (m_A < 0) {
        fprintf(stderr, "ERROR in NoFLA_HQRRP_WY_blk_var4: m_A is < 0.\n");
    }
    if (n_A < 0) {
        fprintf(stderr, "ERROR in NoFLA_HQRRP_WY_blk_var4: n_A is < 0.\n");
    }
    if (ldim_A < std::max(1, m_A)) {
        fprintf(stderr, "ERROR in NoFLA_HQRRP_WY_blk_var4: ldim_A is < max( 1, m_A ).\n");
    }


    // Some initializations.
    mn_A = std::min(m_A, n_A);
    buff_p = buff_jpvt;
    buff_s = buff_tau;

    // Quick return.
    if (mn_A == 0) {
        return 0;
    }

    m_Y = nb_alg + pp;
    n_Y = n_A;
    ldim_Y = m_Y;

    m_V = nb_alg + pp;
    n_V = n_A;
    ldim_V = m_V;

    m_W = nb_alg;
    n_W = n_A;
    ldim_W = m_W;

    m_G = nb_alg + pp;
    n_G = m_A;
    ldim_G = m_G;

    buff_Y = work;
    buff_V = &buff_Y[n_Y * m_Y];
    buff_W = &buff_V[n_V * m_V];
    buff_G = &buff_W[n_W * m_W];
    workspace = &buff_G[n_G * m_G];

    // Initialize matrices G and Y.
    generate_random_matrix(buff_G, &buff_G[n_G * m_G], 69);
    dgemm_("N", "N", &m_Y, &n_Y, &m_A, &d_one, buff_G, &ldim_G, buff_A, &ldim_A, &d_zero, buff_Y, &ldim_Y);

    // Main Loop.
    for (j = 0; j < mn_A; j += nb_alg) {
        b = std::min(nb_alg, std::min(n_A - j, m_A - j));

        // Check whether it is the last iteration.
        last_iter = (((j + nb_alg >= m_A) || (j + nb_alg >= n_A)) ? 1 : 0);

        // Some initializations for the iteration of this loop.
        n_VR = n_V - j;
        buff_VR = &buff_V[0 + j * ldim_V];
        buff_YR = &buff_Y[0 + j * ldim_Y];
        buff_pB = &buff_p[j];
        buff_sB = &buff_s[j];
        buff_AR = &buff_A[0 + j * ldim_A];

        m_AB1 = m_A - j;
        n_AB1 = b;
        buff_AB1 = &buff_A[j + j * ldim_A];
        buff_p1 = &buff_p[j];
        buff_s1 = &buff_s[j];
        buff_A01 = &buff_A[0 + j * ldim_A];
        buff_Y1 = &buff_Y[0 + j * ldim_Y];
        buff_T1_T = &buff_W[0 + j * ldim_W];
        ldim_T1_T = ldim_W;

        buff_A11 = &buff_A[j + j * ldim_A];
        m_A11 = b;
        n_A11 = b;

        buff_A21 = &buff_A[std::min(m_A - 1, j + nb_alg) + j * ldim_A];
        m_A21 = std::max(0, m_A - j - b);
        n_A21 = b;

        buff_A12 = &buff_A[j + std::min(n_A - 1, j + b) * ldim_A];
        m_A12 = b;
        n_A12 = std::max(0, n_A - j - b);

        m_A22 = std::max(0, m_A - j - b);

        buff_Y2 = &buff_Y[0 + std::min(n_Y - 1, j + b) * ldim_Y];
        buff_G1 = &buff_G[0 + j * ldim_G];
        buff_G2 = &buff_G[0 + std::min(n_G - 1, j + b) * ldim_G];

        if (last_iter == 0) {
            // Compute QRP of YR, and apply permutations to matrix AR.
            // A copy of YR is made into VR, and permutations are applied to YR.
            dlacpy_("A", &m_V, &n_VR, buff_YR, &ldim_Y, buff_VR, &ldim_V);
            NoFLA_QRPmod_WY_unb_var4(1, b, m_V, n_VR, buff_VR, ldim_V, buff_pB, buff_sB, 1, m_A, buff_AR, ldim_A, 1, m_Y, buff_YR, ldim_Y, 0, buff_Y, ldim_Y, workspace);
        }

        // Compute QRP of panel AB1 = [ A11; A21 ].
        // Apply same permutations to A01 and Y1, and build T1_T.
        NoFLA_QRPmod_WY_unb_var4(panel_pivoting, -1, m_AB1, n_AB1, buff_AB1, ldim_A, buff_p1, buff_s1, 1, j, buff_A01, ldim_A, 1, m_Y, buff_Y1, ldim_Y, 1, buff_T1_T, ldim_W, workspace);

        // Update the rest of the matrix.
        if ((j + b) < n_A) {
            // Apply the Householder transforms associated with AB1 = [ A11; A21 ]
            // and T1_T to [ A12; A22 ]:
            //   / A12 \ := QB1' / A12 \
            //   \ A22 /         \ A22 /
            // where QB1 is formed from AB1 and T1_T.
            NoFLA_Apply_Q_WY_lhfc_blk_var4(m_A11 + m_A21, n_A11, buff_A11, ldim_A, b, b, buff_T1_T, ldim_W, m_A12 + m_A22, n_A12, buff_A12, ldim_A, workspace);
        }

        // Downdate matrix Y.
        if (!last_iter) {
            NoFLA_Downdate_Y(m_A11, n_A11, buff_A11, ldim_A, m_A21, n_A21, buff_A21, ldim_A, m_A12, n_A12, buff_A12, ldim_A, b, b, buff_T1_T, ldim_T1_T, m_Y, std::max(0, n_Y - j - b), buff_Y2, ldim_Y, m_G, b, buff_G1, ldim_G, m_G, std::max(0, n_G - j - b), buff_G2, ldim_G, workspace);
        }
    }

    return 0;
}

void dgeqp4(int *m, int *n, double *A, int *lda, int *jpvt, double *tau, double *work, int *lwork, int *info)
{
    //
    // This routine is plug compatible with LAPACK's routine dgeqp3.
    // It computes the new HQRRP while keeping the same header as LAPACK's dgeqp3.
    // It uses dgeqpf or dgeqp3 for small matrices. The thresholds are defined in
    // constants THRESHOLD_FOR_DGEQPF and THRESHOLD_FOR_DGEQP3.
    //
    int INB = 1;
    int i_one = 1, i_minus_one = -1, m_A, n_A, mn_A, ldim_A, lquery, nb, num_factorized_fixed_cols, minus_info, iws, lwkopt, j, k, num_fixed_cols, n_rest, itmp;
    int *previous_jpvt;

    // Some initializations.
    m_A = *m;
    n_A = *n;
    mn_A = std::min(m_A, n_A);
    ldim_A = *lda;

    // Check input arguments.
    *info = 0;
    lquery = (*lwork == -1);
    if (m_A < 0) {
        *info = -1;
    }
    else if (n_A < 0) {
        *info = -2;
    }
    else if (ldim_A < std::max(1, m_A)) {
        *info = -4;
    }

    if (*info == 0) {
        if (mn_A == 0) {
            iws = 1;
            lwkopt = 1;
        }
        else {
            iws = 3 * n_A + 1;
            nb = ilaenv_(&INB, "DGEQRF", " ", &m_A, &n_A, &i_minus_one, &i_minus_one);
            lwkopt = 2 * n_A + (n_A + 1) * nb;
            lwkopt = std::max(lwkopt, ((64 + 10) * (3 * n_A + m_A)) + ((3 * n_A) + (64 * n_A) + (64 + 10) * 64));
        }
        work[0] = (double)lwkopt;

        if ((*lwork < iws) && (!lquery)) {
            *info = -8;
        }
    }

    if (*info != 0) {
        minus_info = -*info;
        xerbla_("DGEQP3", &minus_info);
        return;
    }
    else if (lquery) {
        return;
    }

    // Quick return if possible.
    if (mn_A == 0) {
        return;
    }

    // Use LAPACK's DGEQPF or DGEQP3 for small matrices.
    if (mn_A < THRESHOLD_FOR_DGEQPF) {
        // Call to LAPACK routine.
        dgeqpf_(m, n, A, lda, jpvt, tau, work, info);
        return;
    }
    else if (mn_A < THRESHOLD_FOR_DGEQP3) {
        //// printf( "Calling dgeqp3\n" );
        dgeqp3_(m, n, A, lda, jpvt, tau, work, lwork, info);
        return;
    }

    // Move initial columns up front.
    num_fixed_cols = 0;
    for (j = 0; j < n_A; j++) {
        if (jpvt[j] != 0) {
            if (j != num_fixed_cols) {
                //// printf( "Swapping columns: %d %d \n", j, num_fixed_cols );
                dswap_(&m_A, &A[0 + j * ldim_A], &i_one, &A[0 + num_fixed_cols * ldim_A], &i_one);
                jpvt[j] = jpvt[num_fixed_cols];
                jpvt[num_fixed_cols] = j + 1;
            }
            else {
                jpvt[j] = j + 1;
            }
            num_fixed_cols++;
        }
        else {
            jpvt[j] = j + 1;
        }
    }

    // Factorize fixed columns at the front.
    num_factorized_fixed_cols = std::min(m_A, num_fixed_cols);
    if (num_factorized_fixed_cols > 0) {
        dgeqrf_(&m_A, &num_factorized_fixed_cols, A, &ldim_A, tau, work, lwork, info);
        if (*info != 0) {
            fprintf(stderr, "ERROR in dgeqrf: Info: %d \n", *info);
        }
        iws = std::max(iws, (int)work[0]);
        if (num_factorized_fixed_cols < n_A) {
            n_rest = n_A - num_factorized_fixed_cols;
            dormqr_("Left", "Transpose", &m_A, &n_rest, &num_factorized_fixed_cols, A, &ldim_A, tau, &A[0 + num_factorized_fixed_cols * ldim_A], &ldim_A, work, lwork, info);
            if (*info != 0) {
                fprintf(stderr, "ERROR in dormqr: Info: %d \n", *info);
            }

            iws = std::max(iws, (int)work[0]);
        }
    }

    // Create intermediate jpvt vector.
    previous_jpvt = (int *)malloc(n_A * sizeof(int));

    // Save a copy of jpvt vector.
    if (num_factorized_fixed_cols > 0) {
        // Copy vector.
        for (j = 0; j < n_A; j++) {
            previous_jpvt[j] = jpvt[j];
        }
    }

    // Factorize free columns at the bottom with default values:
    // nb_alg = 64, pp = 10, panel_pivoting = 1.
    if (num_factorized_fixed_cols < mn_A) {
        *info = NoFLA_HQRRP_WY_blk_var4(m_A - num_factorized_fixed_cols, n_A - num_factorized_fixed_cols, &A[num_factorized_fixed_cols + num_factorized_fixed_cols * ldim_A], ldim_A, &jpvt[num_factorized_fixed_cols], &tau[num_factorized_fixed_cols], 64, 10, 1, work);
    }

    // Pivot block above factorized block by NoFLA_HQRRP.
    if (num_factorized_fixed_cols > 0) {
        // Pivot block above factorized block.
        for (j = num_factorized_fixed_cols; j < n_A; j++) {
            //// printf( "%% Processing j: %d \n", j );
            for (k = j; k < n_A; k++) {
                if (jpvt[j] == previous_jpvt[k]) {
                    //// printf( "%%   Found j: %d  k: %d \n", j, k );
                    break;
                }
            }
            // Swap vector previous_jpvt and block above factorized block.
            if (k != j) {
                // Swap elements in previous_jpvt.
                //// printf( "%%   Swapping  j: %d  k: %d \n", j, k );
                itmp = previous_jpvt[j];
                previous_jpvt[j] = previous_jpvt[k];
                previous_jpvt[k] = itmp;

                // Swap columns in block above factorized block.
                dswap_(&num_factorized_fixed_cols, &A[0 + j * ldim_A], &i_one, &A[0 + k * ldim_A], &i_one);
            }
        }
    }

    // Remove intermediate jpvt vector.
    free(previous_jpvt);

    // Return workspace length required.
    work[0] = iws;
    return;
}