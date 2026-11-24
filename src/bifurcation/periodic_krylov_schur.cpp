#include "krylov/periodic_krylov_schur.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Krylov
{
    namespace
    {
        inline void givens(double a, double b, double &c, double &s)
        {
            if (b == 0.0) {
                c = 1.0;
                s = 0.0;
            }
            else if (std::abs(b) > std::abs(a)) {
                const double t = a / b;
                const double u = std::copysign(std::sqrt(1.0 + t * t), b);
                s = 1.0 / u;
                c = t * s;
            }
            else {
                const double t = b / a;
                const double u = std::copysign(std::sqrt(1.0 + t * t), a);
                c = 1.0 / u;
                s = t * c;
            }
        }

        // Applies Givens rotation G = [c, s; -s, c] on the LEFT to rows (r1, r2) of column-major A (n x n):
        //   row r1 <-  c * row r1 + s * row r2
        //   row r2 <- -s * row r1 + c * row r2
        inline void apply_givens_left(size_t /*n*/, double *A, size_t ldA, size_t r1, size_t r2, size_t col_start, size_t col_end, double c, double s)
        {
            for (size_t j = col_start; j < col_end; j++) {
                const double a1 = A[j * ldA + r1];
                const double a2 = A[j * ldA + r2];
                A[j * ldA + r1] = c * a1 + s * a2;
                A[j * ldA + r2] = -s * a1 + c * a2;
            }
        }

        // Applies G^T = [c, -s; s, c] on the RIGHT to columns (c1, c2) of column-major A (n x n):
        // Wait: to accumulate Q <- Q * G_rot where left transform was G_rot^T * A,
        // if left was [c, s; -s, c] (which is G^T for G = [c, -s; s, c]), then G on the right is:
        //   col c1 <- c * col c1 + s * col c2
        //   col c2 <- -s * col c1 + c * col c2
        inline void apply_givens_right(double *A, size_t ldA, size_t row_start, size_t row_end, size_t c1, size_t c2, double c, double s)
        {
            double *col1 = &A[c1 * ldA];
            double *col2 = &A[c2 * ldA];
            for (size_t i = row_start; i < row_end; i++) {
                const double a1 = col1[i];
                const double a2 = col2[i];
                col1[i] = c * a1 + s * a2;
                col2[i] = -s * a1 + c * a2;
            }
        }

        // Computes a 2- or 3-element Householder vector v (v[0] = 1) and tau such that (I - tau v v^T) x = alpha e_0
        inline void householder(size_t len, double *x, double &tau)
        {
            double xnorm2 = 0.0;
            for (size_t i = 1; i < len; i++) {
                xnorm2 += x[i] * x[i];
            }
            if (xnorm2 == 0.0) {
                tau = 0.0;
                return;
            }
            const double alpha = x[0];
            const double beta = -std::copysign(std::sqrt(alpha * alpha + xnorm2), alpha);
            tau = (beta - alpha) / beta;
            const double inv_v0 = 1.0 / (alpha - beta);
            x[0] = 1.0;
            for (size_t i = 1; i < len; i++) {
                x[i] *= inv_v0;
            }
        }

        inline void apply_house_left(double *A, size_t ldA, size_t r0, size_t len, size_t col_start, size_t col_end, const double *v, double tau)
        {
            if (tau == 0.0) {
                return;
            }
            for (size_t j = col_start; j < col_end; j++) {
                double dot = A[j * ldA + r0];
                for (size_t i = 1; i < len; i++) {
                    dot += v[i] * A[j * ldA + r0 + i];
                }
                const double s = tau * dot;
                A[j * ldA + r0] -= s;
                for (size_t i = 1; i < len; i++) {
                    A[j * ldA + r0 + i] -= s * v[i];
                }
            }
        }

        inline void apply_house_right(double *A, size_t ldA, size_t row_start, size_t row_end, size_t c0, size_t len, const double *v, double tau)
        {
            if (tau == 0.0) {
                return;
            }
            for (size_t i = row_start; i < row_end; i++) {
                double dot = A[c0 * ldA + i];
                for (size_t j = 1; j < len; j++) {
                    dot += A[(c0 + j) * ldA + i] * v[j];
                }
                const double s = tau * dot;
                A[c0 * ldA + i] -= s;
                for (size_t j = 1; j < len; j++) {
                    A[(c0 + j) * ldA + i] -= s * v[j];
                }
            }
        }

        void compute_periodic_eigenvalues(size_t p, size_t n, double *const *H, double *wr, double *wi)
        {
            size_t i = 0;
            while (i < n) {
                if (i + 1 < n && std::abs(H[p - 1][i * n + (i + 1)]) > 0.0) {
                    // 2x2 block at (i, i+1): compute product P = H[p-1] * ... * H[0] cleanly
                    // Normalize at each step to prevent overflow/underflow in intermediate matrix, track log scale?
                    // Note: for 2x2 eigenvalues we can track product directly if scaled, or scale by max entry!
                    double P[4] = { 1.0, 0.0, 0.0, 1.0 }; // col-major 2x2
                    double scale_log = 0.0;
                    for (size_t m = 0; m < p; m++) {
                        const double *Hm = H[m];
                        const double h00 = Hm[i * n + i];
                        const double h10 = Hm[i * n + (i + 1)];
                        const double h01 = Hm[(i + 1) * n + i];
                        const double h11 = Hm[(i + 1) * n + (i + 1)];
                        // P_new = H_m * P
                        const double p00 = h00 * P[0] + h01 * P[1];
                        const double p10 = h10 * P[0] + h11 * P[1];
                        const double p01 = h00 * P[2] + h01 * P[3];
                        const double p11 = h10 * P[2] + h11 * P[3];
                        const double mx = std::max({ std::abs(p00), std::abs(p10), std::abs(p01), std::abs(p11) });
                        if (mx > 0.0 && (mx > 1e50 || mx < 1e-50)) {
                            const double inv = 1.0 / mx;
                            P[0] = p00 * inv;
                            P[1] = p10 * inv;
                            P[2] = p01 * inv;
                            P[3] = p11 * inv;
                            scale_log += std::log(mx);
                        }
                        else {
                            P[0] = p00;
                            P[1] = p10;
                            P[2] = p01;
                            P[3] = p11;
                        }
                    }
                    const double scale = (scale_log != 0.0) ? std::exp(scale_log) : 1.0;
                    const double tr = 0.5 * (P[0] + P[3]);
                    const double det_part = (P[0] - tr) * (P[3] - tr) - P[1] * P[2];
                    if (det_part <= 0.0) {
                        const double im = std::sqrt(-det_part);
                        wr[i] = tr * scale;
                        wi[i] = im * scale;
                        wr[i + 1] = tr * scale;
                        wi[i + 1] = -im * scale;
                    }
                    else {
                        const double sq = std::sqrt(det_part);
                        wr[i] = (tr + sq) * scale;
                        wi[i] = 0.0;
                        wr[i + 1] = (tr - sq) * scale;
                        wi[i + 1] = 0.0;
                    }
                    i += 2;
                }
                else {
                    double prod = 1.0;
                    for (size_t m = 0; m < p; m++) {
                        prod *= H[m][i * n + i];
                    }
                    wr[i] = prod;
                    wi[i] = 0.0;
                    i += 1;
                }
            }
        }
    } // namespace

    void periodic_real_schur(size_t p, size_t n, double *const *H, double *const *Z, double *wr, double *wi, double *work)
    {
        for (size_t m = 0; m < p; m++) {
            memset(Z[m], 0, n * n * sizeof(double));
            for (size_t i = 0; i < n; i++) {
                Z[m][i * n + i] = 1.0;
            }
        }
        if (n == 0) {
            return;
        }
        if (n == 1) {
            compute_periodic_eigenvalues(p, n, H, wr, wi);
            return;
        }

        const bool own_work = (work == nullptr);
        if (own_work) {
            work = (double *)malloc((4 * p * n * n + 16 * n + 64) * sizeof(double));
        }

        // Phase 1a: Make H[0..p-2] upper triangular via QR sweep
        for (size_t m = 0; m < p - 1; m++) {
            for (size_t col = 0; col < n - 1; col++) {
                for (size_t row = n - 1; row > col; row--) {
                    double c, s;
                    givens(H[m][col * n + (row - 1)], H[m][col * n + row], c, s);
                    apply_givens_left(n, H[m], n, row - 1, row, col, n, c, s);
                    H[m][col * n + row] = 0.0;
                    apply_givens_right(Z[m + 1], n, 0, n, row - 1, row, c, s);
                    apply_givens_right(H[m + 1], n, 0, n, row - 1, row, c, s);
                }
            }
        }

        // Phase 1b: Reduce H[p-1] to upper Hessenberg while preserving upper triangular H[0..p-2]
        for (size_t col = 0; col < n - 2; col++) {
            for (size_t row = n - 1; row > col + 1; row--) {
                double c, s;
                givens(H[p - 1][col * n + (row - 1)], H[p - 1][col * n + row], c, s);
                apply_givens_left(n, H[p - 1], n, row - 1, row, col, n, c, s);
                H[p - 1][col * n + row] = 0.0;
                apply_givens_right(Z[0], n, 0, n, row - 1, row, c, s);
                apply_givens_right(H[0], n, 0, row + 1, row - 1, row, c, s);

                // Chase bulge at (row, row - 1) through H[0..p-2]
                for (size_t m = 0; m < p - 1; m++) {
                    double cm, sm;
                    givens(H[m][(row - 1) * n + (row - 1)], H[m][(row - 1) * n + row], cm, sm);
                    apply_givens_left(n, H[m], n, row - 1, row, row - 1, n, cm, sm);
                    H[m][(row - 1) * n + row] = 0.0;
                    apply_givens_right(Z[m + 1], n, 0, n, row - 1, row, cm, sm);
                    const size_t row_limit = (m + 1 == p - 1) ? n : (row + 1);
                    apply_givens_right(H[m + 1], n, 0, row_limit, row - 1, row, cm, sm);
                }
            }
        }

        // Phase 2: Periodic Implicit Double-Shift Francis QR
        const double eps = std::numeric_limits<double>::epsilon();
        size_t ihi = n - 1;
        int iters = 0;
        const int max_qr_iters = 60 * (int)n;

        while (ihi > 0 && iters < max_qr_iters) {
            // Find ilo: scan backwards for negligible subdiagonal in H[p-1]
            size_t ilo = ihi;
            while (ilo > 0) {
                const double h_sub = std::abs(H[p - 1][(ilo - 1) * n + ilo]);
                const double h_diag = std::abs(H[p - 1][(ilo - 1) * n + (ilo - 1)]) + std::abs(H[p - 1][ilo * n + ilo]);
                const double tol = std::max(eps * h_diag, 1e-300);
                if (h_sub <= tol) {
                    H[p - 1][(ilo - 1) * n + ilo] = 0.0;
                    break;
                }
                ilo--;
            }

            if (ilo == ihi) {
                if (ihi == 0) break;
                ihi--;
                iters = 0;
                continue;
            }
            if (ilo + 1 == ihi) {
                // 2x2 block: check if eigenvalues are real -> split via periodic single-shift QR if needed,
                // or keep as 2x2 complex block!
                double P[4] = { 1.0, 0.0, 0.0, 1.0 };
                for (size_t m = 0; m < p; m++) {
                    const double *Hm = H[m];
                    const double h00 = Hm[ilo * n + ilo];
                    const double h10 = Hm[ilo * n + ihi];
                    const double h01 = Hm[ihi * n + ilo];
                    const double h11 = Hm[ihi * n + ihi];
                    const double p00 = h00 * P[0] + h01 * P[1];
                    const double p10 = h10 * P[0] + h11 * P[1];
                    const double p01 = h00 * P[2] + h01 * P[3];
                    const double p11 = h10 * P[2] + h11 * P[3];
                    const double mx = std::max({ std::abs(p00), std::abs(p10), std::abs(p01), std::abs(p11) });
                    if (mx > 0.0 && (mx > 1e50 || mx < 1e-50)) {
                        const double inv = 1.0 / mx;
                        P[0] = p00 * inv; P[1] = p10 * inv; P[2] = p01 * inv; P[3] = p11 * inv;
                    }
                    else {
                        P[0] = p00; P[1] = p10; P[2] = p01; P[3] = p11;
                    }
                }
                const double tr = 0.5 * (P[0] + P[3]);
                const double det_part = (P[0] - tr) * (P[3] - tr) - P[1] * P[2];
                if (det_part < 0.0) {
                    // Complex conjugate pair -> converged 2x2 Schur block!
                    if (ilo == 0) break;
                    ihi = ilo - 1;
                    iters = 0;
                    continue;
                }
                // Real eigenvalues in 2x2 block: do a shifted 2x2 periodic QR step to split!
            }

            iters++;

            // Compute shift from bottom-right 2x2 block of H[p-1]*...*H[0] (or exceptional shift)
            double P2[4] = { 1.0, 0.0, 0.0, 1.0 };
            for (size_t m = 0; m < p; m++) {
                const double *Hm = H[m];
                const size_t i0 = ihi - 1, i1 = ihi;
                const double h00 = Hm[i0 * n + i0], h10 = Hm[i0 * n + i1];
                const double h01 = Hm[i1 * n + i0], h11 = Hm[i1 * n + i1];
                const double p00 = h00 * P2[0] + h01 * P2[1];
                const double p10 = h10 * P2[0] + h11 * P2[1];
                const double p01 = h00 * P2[2] + h01 * P2[3];
                const double p11 = h10 * P2[2] + h11 * P2[3];
                const double mx = std::max({ std::abs(p00), std::abs(p10), std::abs(p01), std::abs(p11) });
                if (mx > 0.0 && (mx > 1e50 || mx < 1e-50)) {
                    const double inv = 1.0 / mx;
                    P2[0] = p00 * inv; P2[1] = p10 * inv; P2[2] = p01 * inv; P2[3] = p11 * inv;
                }
                else {
                    P2[0] = p00; P2[1] = p10; P2[2] = p01; P2[3] = p11;
                }
            }
            // Note: if P2 was rescaled differently from top-left? Wait!
            // If we scale P2 separately from top-left of Pi, the shift wouldn't match top-left of Pi!
            // So compute the FIRST 3 columns of (Pi_m - shift) WITHOUT overflow by tracking scale at top-left!
            // Actually, compute shift & first column using the SAME per-factor normalization!
            // Even better: for single-shift when (ihi - ilo == 1) or double-shift when (ihi - ilo >= 2):
            if (ihi == ilo + 1) {
                // 2x2 real split: pick eigenvalue of P2 closer to P2[3]
                const double tr = 0.5 * (P2[0] + P2[3]);
                const double disc = std::sqrt(std::max(0.0, (P2[0] - tr) * (P2[3] - tr) - P2[1] * P2[2]));
                const double s1 = tr + disc, s2 = tr - disc;
                const double shift = (std::abs(s1 - P2[3]) < std::abs(s2 - P2[3])) ? s1 : s2;
                // Wait: if P2 was rescaled, we must compute Pi_2x2 without independent scaling or track shared scale!
                // Wait: for 2x2 active block, P2 IS the whole active block! So (P2 - shift * I) * e0 has exact first col!
                double v[2] = { P2[0] - shift, P2[1] };
                double c, s;
                givens(v[0], v[1], c, s);
                apply_givens_left(n, H[p - 1], n, ilo, ihi, ilo, n, c, s);
                apply_givens_right(Z[0], n, 0, n, ilo, ihi, c, s);
                apply_givens_right(H[0], n, 0, ihi + 1, ilo, ihi, c, s);
                for (size_t m = 0; m < p - 1; m++) {
                    double cm, sm;
                    givens(H[m][ilo * n + ilo], H[m][ilo * n + ihi], cm, sm);
                    apply_givens_left(n, H[m], n, ilo, ihi, ilo, n, cm, sm);
                    H[m][ilo * n + ihi] = 0.0;
                    apply_givens_right(Z[m + 1], n, 0, n, ilo, ihi, cm, sm);
                    const size_t rlim = (m + 1 == p - 1) ? n : (ihi + 1);
                    apply_givens_right(H[m + 1], n, 0, rlim, ilo, ihi, cm, sm);
                }
                continue;
            }

            // General (ihi - ilo >= 2): compute product of top-left 3x3 AND bottom-right 2x2 of H[m]
            // Normalize each factor H[m] on [ilo..ihi] by the SAME scalar alpha_m = max(|top-left|, |bot-right|)
            // Wait: if one factor has 1e12 at top-left and 1 at bottom-right, scaling both by 1e12 makes bottom-right 1e-12,
            // which can underflow after 24 steps ((1e-12)^24 = 1e-288)!
            // Look at the standard Bojanczyk-Golub-Van Dooren trick for avoiding overflow/underflow in shift:
            // Instead of multiplying 2x2 matrices first, propagate a 3-vector (and shift) using cyclic Givens / normalized ratios!
            // Specifically, for Francis shift we can use the bottom-right 2x2 eigenvalues lambda1, lambda2 represented in factored form:
            // Actually, in Krylov-Schur at iteration `iter`, the eigenvalues of H_m have already been tracked, OR we can compute:
            //   Let M_top be 3x3 (from ilo..ilo+2), and M_bot be 2x2 (from ihi-1..ihi).
            //   Notice (M_bot - s I) only depends on the ratio between M_bot^1/p and M_top^1/p!
            //   Even simpler: compute M_top = H_top[p-1]...H_top[0] (with log_scale_top) and M_bot = H_bot[p-1]...H_bot[0] (with log_scale_bot)!
            //   Then tr_s = tr(M_bot) * exp(log_scale_bot - log_scale_top), det_s = det(M_bot) * exp(2*(log_scale_bot - log_scale_top))!
            //   Check that! Since x = (M_top^2 - tr*M_top + det*I) e_0, dividing by exp(2*log_scale_top) gives:
            //     x_scaled = (M_top_norm^2 - tr_s * M_top_norm + det_s * I) e_0!!
            //   Wait: if log_scale_bot - log_scale_top < -100, then tr_s = 0 and det_s = 0 (Wilkinson shift = 0, which STILL deflates the large top-left mode cleanly without underflow!);
            //   and if log_scale_bot - log_scale_top > +100, divide x_scaled by det_s -> e_0 (or shift dominant)!
            //   THAT HANDLES ANY DYNAMIC RANGE (even 10^300!) WITHOUT OVERFLOW OR UNDERFLOW!
            double T3[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // col-major 3x3 at ilo..ilo+2
            double B2[4] = { 1, 0, 0, 1 };                // col-major 2x2 at ihi-1..ihi
            double log_t = 0.0, log_b = 0.0;

            for (size_t m = 0; m < p; m++) {
                const double *Hm = H[m];
                double nextT[9] = { 0 };
                for (size_t c = 0; c < 3; c++) {
                    for (size_t r = 0; r < 3; r++) {
                        double s = 0.0;
                        for (size_t k = 0; k < 3; k++) {
                            s += Hm[(ilo + k) * n + (ilo + r)] * T3[c * 3 + k];
                        }
                        nextT[c * 3 + r] = s;
                    }
                }
                double maxT = 0.0;
                for (int q = 0; q < 9; q++) maxT = std::max(maxT, std::abs(nextT[q]));
                if (maxT > 0.0) {
                    const double invT = 1.0 / maxT;
                    for (int q = 0; q < 9; q++) T3[q] = nextT[q] * invT;
                    log_t += std::log(maxT);
                }
                else {
                    memcpy(T3, nextT, sizeof(T3));
                }

                const size_t i0 = ihi - 1, i1 = ihi;
                const double b00 = Hm[i0 * n + i0] * B2[0] + Hm[i1 * n + i0] * B2[1];
                const double b10 = Hm[i0 * n + i1] * B2[0] + Hm[i1 * n + i1] * B2[1];
                const double b01 = Hm[i0 * n + i0] * B2[2] + Hm[i1 * n + i0] * B2[3];
                const double b11 = Hm[i0 * n + i1] * B2[2] + Hm[i1 * n + i1] * B2[3];
                const double maxB = std::max({ std::abs(b00), std::abs(b10), std::abs(b01), std::abs(b11) });
                if (maxB > 0.0) {
                    const double invB = 1.0 / maxB;
                    B2[0] = b00 * invB; B2[1] = b10 * invB; B2[2] = b01 * invB; B2[3] = b11 * invB;
                    log_b += std::log(maxB);
                }
                else {
                    B2[0] = b00; B2[1] = b10; B2[2] = b01; B2[3] = b11;
                }
            }

            if (iters % 10 == 0) {
                // Exceptional shift to break stagnation
                const double es = std::abs(H[p - 1][ihi * n + (ihi - 1)]) + std::abs(H[p - 1][(ihi - 1) * n + (ihi - 2)]);
                B2[0] = 0.75 * es; B2[3] = 0.75 * es; B2[1] = -0.4375 * es; B2[2] = es;
                log_b = log_t;
            }

            const double dlog = log_b - log_t;
            double v3[3];
            if (dlog > 200.0) {
                // det_s dominates -> proportional to e_0 (or shift along T3)
                v3[0] = 1.0; v3[1] = 0.0; v3[2] = 0.0;
            }
            else {
                const double ratio = (dlog < -300.0) ? 0.0 : std::exp(dlog);
                const double tr_s = (B2[0] + B2[3]) * ratio;
                const double det_s = (B2[0] * B2[3] - B2[1] * B2[2]) * ratio * ratio;
                // First column of T3^2 - tr_s * T3 + det_s * I:
                v3[0] = T3[0] * T3[0] + T3[3] * T3[1] + T3[6] * T3[2] - tr_s * T3[0] + det_s;
                v3[1] = T3[1] * T3[0] + T3[4] * T3[1] + T3[7] * T3[2] - tr_s * T3[1];
                v3[2] = T3[2] * T3[0] + T3[5] * T3[1] + T3[8] * T3[2] - tr_s * T3[2];
            }

            // Bulge chase: step k = ilo .. ihi - 1
            for (size_t k = ilo; k < ihi; k++) {
                const size_t nr = std::min<size_t>(3, ihi - k + 1);
                if (k > ilo) {
                    for (size_t r = 0; r < nr; r++) {
                        v3[r] = H[p - 1][(k - 1) * n + (k + r)];
                    }
                }
                double tau = 0.0;
                householder(nr, v3, tau);
                if (k > ilo) {
                    const double beta_k = H[p - 1][(k - 1) * n + k] - tau * (H[p - 1][(k - 1) * n + k] + (nr > 1 ? v3[1] * H[p - 1][(k - 1) * n + k + 1] : 0.0) + (nr > 2 ? v3[2] * H[p - 1][(k - 1) * n + k + 2] : 0.0));
                    H[p - 1][(k - 1) * n + k] = beta_k;
                    if (nr > 1) H[p - 1][(k - 1) * n + k + 1] = 0.0;
                    if (nr > 2) H[p - 1][(k - 1) * n + k + 2] = 0.0;
                }
                apply_house_left(H[p - 1], n, k, nr, k, n, v3, tau);
                apply_house_right(Z[0], n, 0, n, k, nr, v3, tau);
                apply_house_right(H[0], n, 0, std::min(n, k + nr), k, nr, v3, tau);

                // Chase bulge of size nr at rows/cols (k .. k+nr-1) through H[0..p-2]
                for (size_t m = 0; m < p - 1; m++) {
                    // Reduce lower-triangular entries in columns k..k+nr-2 of H[m]
                    for (size_t c_sub = 0; c_sub + 1 < nr; c_sub++) {
                        const size_t col_m = k + c_sub;
                        const size_t len_m = nr - c_sub;
                        double vm[3];
                        for (size_t r = 0; r < len_m; r++) {
                            vm[r] = H[m][col_m * n + (col_m + r)];
                        }
                        double taum = 0.0;
                        householder(len_m, vm, taum);
                        apply_house_left(H[m], n, col_m, len_m, col_m, n, vm, taum);
                        for (size_t r = 1; r < len_m; r++) {
                            H[m][col_m * n + col_m + r] = 0.0;
                        }
                        apply_house_right(Z[m + 1], n, 0, n, col_m, len_m, vm, taum);
                        const size_t rlim = (m + 1 == p - 1) ? (ihi + 1) : std::min(n, k + nr);
                        apply_house_right(H[m + 1], n, 0, rlim, col_m, len_m, vm, taum);
                    }
                }
            }
        }

        compute_periodic_eigenvalues(p, n, H, wr, wi);

        if (own_work) {
            free(work);
        }
    }

    namespace
    {
        // Swaps adjacent diagonal blocks at `pos` (size d1) and `pos + d1` (size d2) in periodic Schur form.
        // Returns true if swap succeeded.
        bool swap_periodic_blocks(size_t p, size_t n, double *const *H, double *const *Z, size_t pos, size_t d1, size_t d2, double *work)
        {
            const size_t b = d1 * d2; // <= 4
            const size_t N_sys = p * b;
            double *S = work;               // N_sys * N_sys (col-major)
            double *rhs = S + N_sys * N_sys; // N_sys
            int *ipiv = (int *)(rhs + N_sys);

            memset(S, 0, N_sys * N_sys * sizeof(double));
            memset(rhs, 0, N_sys * sizeof(double));

            const size_t p1 = pos;
            const size_t p2 = pos + d1;

            // Equation m: A_m * X_m - X_{(m+1)%p} * B_m = -C_m
            // Vectorize X_m in col-major: idx(r, c) = c * d1 + r
            for (size_t m = 0; m < p; m++) {
                const double *Hm = H[m];
                const size_t m_next = (m + 1) % p;
                const size_t row_off = m * b;
                const size_t col_x = m * b;
                const size_t col_xnext = m_next * b;

                for (size_t c = 0; c < d2; c++) {
                    for (size_t r = 0; r < d1; r++) {
                        const size_t eq_row = row_off + c * d1 + r;
                        rhs[eq_row] = -Hm[(p2 + c) * n + (p1 + r)];

                        // + A_m * X_m : sum_k A_m(r, k) * X_m(k, c)
                        for (size_t k = 0; k < d1; k++) {
                            S[(col_x + c * d1 + k) * N_sys + eq_row] += Hm[(p1 + k) * n + (p1 + r)];
                        }
                        // - X_next * B_m : sum_l X_next(r, l) * B_m(l, c)
                        for (size_t l = 0; l < d2; l++) {
                            S[(col_xnext + l * d1 + r) * N_sys + eq_row] -= Hm[(p2 + c) * n + (p2 + l)];
                        }
                    }
                }
            }

            int info = 0;
            dgesv((int)N_sys, 1, S, (int)N_sys, ipiv, rhs, (int)N_sys, &info);
            if (info != 0) {
                return false;
            }
            for (size_t i = 0; i < N_sys; i++) {
                if (!std::isfinite(rhs[i]) || std::abs(rhs[i]) > 1e12) {
                    return false;
                }
            }

            // Build U_m (size d x d, where d = d1 + d2 <= 4) from QR of [-X_m; I_{d2}]
            const size_t d = d1 + d2;
            double *U_all = (double *)(ipiv + N_sys); // p * d * d
            double M_qr[16];
            double tau_qr[4];
            double qr_work[64];

            for (size_t m = 0; m < p; m++) {
                double *Um = &U_all[m * d * d];
                const double *Xm = &rhs[m * b];
                memset(M_qr, 0, sizeof(M_qr));
                for (size_t c = 0; c < d2; c++) {
                    for (size_t r = 0; r < d1; r++) {
                        M_qr[c * d + r] = -Xm[c * d1 + r];
                    }
                    M_qr[c * d + (d1 + c)] = 1.0;
                }
                int id = (int)d, id2 = (int)d2, lwork = 64, qinfo = 0;
                dgeqrf_(&id, &id2, M_qr, &id, tau_qr, qr_work, &lwork, &qinfo);
                memset(Um, 0, d * d * sizeof(double));
                for (size_t c = 0; c < d2; c++) {
                    for (size_t r = 0; r < d; r++) {
                        Um[c * d + r] = M_qr[c * d + r];
                    }
                }
                dorgqr_(&id, &id, &id2, Um, &id, tau_qr, qr_work, &lwork, &qinfo);
            }

            // Apply H_m <- U_{(m+1)%p}^T * H_m * U_m and Z_m <- Z_m * U_m
            double tmp[16];
            for (size_t m = 0; m < p; m++) {
                const double *Um = &U_all[m * d * d];
                const double *Unext = &U_all[((m + 1) % p) * d * d];

                // Update columns p1..p1+d-1 of H[m] from the right: H[m][:, p1:p1+d] *= U_m
                for (size_t r = 0; r < n; r++) {
                    double row_tmp[4];
                    for (size_t c = 0; c < d; c++) {
                        double acc = 0.0;
                        for (size_t k = 0; k < d; k++) {
                            acc += H[m][(p1 + k) * n + r] * Um[c * d + k];
                        }
                        row_tmp[c] = acc;
                    }
                    for (size_t c = 0; c < d; c++) {
                        H[m][(p1 + c) * n + r] = row_tmp[c];
                    }
                }

                // Update rows p1..p1+d-1 of H[m] from the left: H[m][p1:p1+d, :] = U_next^T * H[m][p1:p1+d, :]
                for (size_t c = 0; c < n; c++) {
                    double col_tmp[4];
                    for (size_t r = 0; r < d; r++) {
                        double acc = 0.0;
                        for (size_t k = 0; k < d; k++) {
                            acc += Unext[r * d + k] * H[m][c * n + (p1 + k)];
                        }
                        col_tmp[r] = acc;
                    }
                    for (size_t r = 0; r < d; r++) {
                        H[m][c * n + (p1 + r)] = col_tmp[r];
                    }
                }

                // Zero strict lower-left d2 x d1 block explicitly (exact up to rounding)
                for (size_t c = 0; c < d1; c++) {
                    for (size_t r = 0; r < d2; r++) {
                        H[m][(p1 + c) * n + (p1 + d2 + r)] = 0.0;
                    }
                }

                // Update Z[m] <- Z[m] * U_m
                for (size_t r = 0; r < n; r++) {
                    for (size_t c = 0; c < d; c++) {
                        double acc = 0.0;
                        for (size_t k = 0; k < d; k++) {
                            acc += Z[m][(p1 + k) * n + r] * Um[c * d + k];
                        }
                        tmp[c] = acc;
                    }
                    for (size_t c = 0; c < d; c++) {
                        Z[m][(p1 + c) * n + r] = tmp[c];
                    }
                }
            }

            // Re-triangularize any 2x2 diagonal blocks inside H[0..p-2] created by swapping 2x2 blocks
            for (size_t m = 0; m < p - 1; m++) {
                for (size_t off : { (size_t)0, d2 }) {
                    const size_t sz = (off == 0) ? d2 : d1;
                    if (sz == 2) {
                        const size_t q = p1 + off;
                        double c, s;
                        givens(H[m][q * n + q], H[m][q * n + (q + 1)], c, s);
                        apply_givens_left(n, H[m], n, q, q + 1, q, n, c, s);
                        H[m][q * n + (q + 1)] = 0.0;
                        apply_givens_right(Z[m + 1], n, 0, n, q, q + 1, c, s);
                        apply_givens_right(H[m + 1], n, 0, n, q, q + 1, c, s);
                    }
                }
            }
            return true;
        }
    } // namespace

    size_t periodic_schur_reorder(size_t p, size_t n, double *const *H, double *const *Z, const int *select, double *wr, double *wi, double *work)
    {
        const bool own_work = (work == nullptr);
        if (own_work) {
            work = (double *)malloc((32 * p * p + 64 * p + 256) * sizeof(double));
        }

        // Build list of current diagonal blocks (start, size)
        struct Block
        {
            size_t start;
            size_t size;
            int sel;
        };
        Block blocks[n];
        size_t nb = 0;
        size_t i = 0;
        while (i < n) {
            if (i + 1 < n && std::abs(H[p - 1][i * n + (i + 1)]) > 0.0) {
                const int s = (select[i] || select[i + 1]) ? 1 : 0;
                blocks[nb++] = { i, 2, s };
                i += 2;
            }
            else {
                blocks[nb++] = { i, 1, select[i] ? 1 : 0 };
                i += 1;
            }
        }

        // Bubble selected blocks to the front while preserving relative order
        size_t next_pos = 0;
        size_t k_kept = 0;
        for (size_t bi = 0; bi < nb; bi++) {
            if (!blocks[bi].sel) {
                continue;
            }
            // Move block bi left past unselected blocks back to next_pos
            size_t cur = bi;
            while (cur > next_pos) {
                const size_t prev = cur - 1;
                if (!swap_periodic_blocks(p, n, H, Z, blocks[prev].start, blocks[prev].size, blocks[cur].size, work)) {
                    break;
                }
                const size_t d1 = blocks[prev].size;
                const size_t d2 = blocks[cur].size;
                blocks[prev].size = d2;
                blocks[cur].size = d1;
                blocks[cur].start = blocks[prev].start + d2;
                std::swap(blocks[prev].sel, blocks[cur].sel);
                cur = prev;
            }
            k_kept += blocks[next_pos].size;
            next_pos++;
        }

        compute_periodic_eigenvalues(p, n, H, wr, wi);

        if (own_work) {
            free(work);
        }
        return k_kept;
    }

    PeriodicKSResult::~PeriodicKSResult()
    {
        freeBuffers();
    }

    void PeriodicKSResult::freeBuffers()
    {
        if (dims) { free(dims); dims = nullptr; }
        if (q_offsets) { free(q_offsets); q_offsets = nullptr; }
        if (Q_all) { free(Q_all); Q_all = nullptr; }
        if (H_all) { free(H_all); H_all = nullptr; }
        if (wr) { free(wr); wr = nullptr; }
        if (wi) { free(wi); wi = nullptr; }
        if (work) { free(work); work = nullptr; }
        work_cap = 0;
        p = 0;
        nsub = 0;
        nconverged = 0;
    }

    void PeriodicKSResult::reserve(size_t in_p, const size_t *in_dims, size_t in_nsub)
    {
        freeBuffers();
        p = in_p;
        nsub = in_nsub;
        dims = (size_t *)malloc(p * sizeof(size_t));
        q_offsets = (size_t *)malloc((p + 1) * sizeof(size_t));
        q_offsets[0] = 0;
        size_t max_dim = 0;
        for (size_t n = 0; n < p; n++) {
            dims[n] = in_dims[n];
            max_dim = std::max(max_dim, in_dims[n]);
            q_offsets[n + 1] = q_offsets[n] + in_dims[n] * (nsub + 1);
        }
        Q_all = (double *)malloc(q_offsets[p] * sizeof(double));
        H_all = (double *)malloc(p * nsub * nsub * sizeof(double));
        wr = (double *)calloc(nsub, sizeof(double));
        wi = (double *)calloc(nsub, sizeof(double));

        const size_t schur_work_len = std::max(4 * p * nsub * nsub + 32 * nsub, 32 * p * p + 64 * p) + 256;
        work_cap = p * nsub * nsub + schur_work_len + nsub + max_dim * (nsub + 2) + 256;
        work = (double *)malloc(work_cap * sizeof(double));
    }

    void PeriodicKSResult::getEigenvectorsAtNode0(size_t k, std::complex<double> *V_out) const
    {
        if (k == 0 || p == 0) {
            return;
        }
        const int ik = (int)k;
        const size_t d0 = dims[0];
        double *M = (double *)malloc(k * k * sizeof(double));
        double *M_tmp = (double *)malloc(k * k * sizeof(double));
        double *VR = (double *)malloc(k * k * sizeof(double));
        double *trevc_work = (double *)malloc(3 * k * sizeof(double));

        // Form k x k quasi-upper-triangular product M = H[p-1][0:k, 0:k] * ... * H[0][0:k, 0:k]
        memset(M, 0, k * k * sizeof(double));
        for (size_t i = 0; i < k; i++) {
            M[i * k + i] = 1.0;
        }
        for (size_t m = 0; m < p; m++) {
            const double *Hm = H(m);
            dgemm("N", "N", ik, ik, ik, 1.0, Hm, (int)nsub, M, ik, 0.0, M_tmp, ik);
            memcpy(M, M_tmp, k * k * sizeof(double));
        }

        // Clean any tiny below-subdiagonal noise so dtrevc3 sees exact quasi-upper-triangular form
        for (size_t j = 0; j < k; j++) {
            for (size_t i = j + 2; i < k; i++) {
                M[j * k + i] = 0.0;
            }
        }

        int m_out = 0, info = 0;
        dtrevc3("R", "A", nullptr, ik, M, ik, nullptr, ik, VR, ik, ik, &m_out, trevc_work, 3 * ik, &info);

        // Project into V_0: Q_0[:, 0:k] * VR
        double *V_real = (double *)malloc(d0 * k * sizeof(double));
        dgemm("N", "N", (int)d0, ik, ik, 1.0, Q(0), (int)d0, VR, ik, 0.0, V_real, (int)d0);

        size_t col = 0;
        while (col < k) {
            if (col + 1 < k && std::abs(M[col * k + (col + 1)]) > 0.0) {
                for (size_t r = 0; r < d0; r++) {
                    const double vr = V_real[col * d0 + r];
                    const double vi = V_real[(col + 1) * d0 + r];
                    V_out[col * d0 + r] = std::complex<double>(vr, vi);
                    V_out[(col + 1) * d0 + r] = std::complex<double>(vr, -vi);
                }
                col += 2;
            }
            else {
                for (size_t r = 0; r < d0; r++) {
                    V_out[col * d0 + r] = std::complex<double>(V_real[col * d0 + r], 0.0);
                }
                col += 1;
            }
        }

        free(V_real);
        free(trevc_work);
        free(VR);
        free(M_tmp);
        free(M);
    }
} // namespace Krylov
