#include "modulating/floquet.h"
#include "linalg.h"
#include "tools/helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

namespace TWIST
{
    namespace
    {
        // Computes exact spatial derivative dY/d(xi_hat) of a collocation vector Y on `slice`
        // at every collocation point (nodes and stages) using the Lagrange polynomial on {0, c_1, ..., c_sx}.
        void eval_spatial_derivative(const SpatialSlice &slice, const double *Y, double *dY_dxi)
        {
            const size_t nsub = slice.nSub();
            const size_t m = slice.node;
            const size_t s = slice.nstages;
            const size_t n_basis = s + 1;

            double nodes[n_basis];
            nodes[0] = 0.0;
            for (size_t j = 0; j < s; j++) {
                nodes[j + 1] = slice.c_c[j];
            }

            // Precompute differentiation matrix D_gl[l, j] = phi'_j(nodes[l]) for l=0..s, j=0..s
            double D_gl[n_basis * n_basis];
            for (size_t l = 0; l < n_basis; l++) {
                const double th = nodes[l];
                for (size_t j = 0; j < n_basis; j++) {
                    double den = 1.0;
                    for (size_t q = 0; q < n_basis; q++) {
                        if (q != j) den *= (nodes[j] - nodes[q]);
                    }
                    double num = 0.0;
                    for (size_t k = 0; k < n_basis; k++) {
                        if (k == j) continue;
                        double prod = 1.0;
                        for (size_t q = 0; q < n_basis; q++) {
                            if (q != j && q != k) prod *= (th - nodes[q]);
                        }
                        num += prod;
                    }
                    D_gl[l * n_basis + j] = num / den;
                }
            }

            for (size_t k = 0; k < nsub; k++) {
                const size_t stride_k = k * n_basis * m;
                const double inv_hk = 1.0 / slice.h[k];
                for (size_t l = 0; l < n_basis; l++) {
                    double *out = &dY_dxi[stride_k + l * m];
                    memset(out, 0, m * sizeof(double));
                    for (size_t j = 0; j < n_basis; j++) {
                        const double w = inv_hk * D_gl[l * n_basis + j];
                        const double *src = &Y[stride_k + j * m];
                        for (size_t d = 0; d < m; d++) {
                            out[d] += w * src[d];
                        }
                    }
                }
            }
            // Right endpoint xi = 1 matches xi = 0
            memcpy(&dY_dxi[nsub * n_basis * m], dY_dxi, m * sizeof(double));
        }
    } // namespace

    FloquetSpectrum FloquetAnalyzer::compute(size_t nt, size_t st, const SpatialSlice *slices, const SparseDenseOutputOp *transfers, const SubintervalCondenser *condensers, const double *y_stages, const double *dtau, double T, const double *p, size_t neig, size_t nsub, double tol, double unstable_tol, bool verbose)
    {
        omp_set_max_active_levels(1);
        OpenBLASThreadContext __openblas_context(1);

        FloquetSpectrum spec;

        // Build local condensers/transfers only if caller didn't supply them
        SubintervalCondenser *local_conds = nullptr;
        SparseDenseOutputOp *local_trans = nullptr;
        if (condensers == nullptr) {
            local_conds = new SubintervalCondenser[nt];
            size_t offset = 0;
            for (size_t n = 0; n < nt; n++) {
                local_conds[n].factor(&slices[n], st, &y_stages[offset], p, dtau[n] * T, true);
                offset += st * slices[n].n_space;
            }
            condensers = local_conds;
        }
        if (transfers == nullptr) {
            local_trans = new SparseDenseOutputOp[nt];
            for (size_t n = 0; n < nt; n++) {
                const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
                local_trans[n].build(slices[prev], slices[n]);
            }
            transfers = local_trans;
        }

        // Spaces V_0 .. V_{nt-1}: V_n lives on slices[(n - 1 + nt) % nt]
        size_t dims[nt];
        double *weights[nt];
        for (size_t n = 0; n < nt; n++) {
            const size_t src_slice_idx = (n == 0) ? (nt - 1) : (n - 1);
            dims[n] = slices[src_slice_idx].n_space;
            weights[n] = (double *)malloc(dims[n] * sizeof(double));
            slices[src_slice_idx].fillQuadratureWeights(weights[n], true);
        }

        auto apply_op = [condensers, transfers](size_t n, const double *v_in, double *v_out) {
            condensers[n].applyCondensedC(v_in, transfers[n], v_out, true);
        };

        Krylov::periodic_krylov_schur(apply_op, nt, dims, weights, neig, nsub, m_ks_result, tol, verbose);

        const size_t k_out = std::min(neig, m_ks_result.nconverged > 0 ? m_ks_result.nconverged : neig);
        spec.mu_real.resize(k_out);
        spec.mu_imag.resize(k_out);
        for (size_t i = 0; i < k_out; i++) {
            spec.mu_real[i] = m_ks_result.wr[i];
            spec.mu_imag[i] = m_ks_result.wi[i];
        }

        // Tag the two trivial Floquet multipliers at mu = +1 via L2 overlap with dY/dxi and dY/dtau at tau = 1 (node 0 = M_{nt-1})
        const size_t d0 = dims[0];
        const size_t last_idx = nt - 1;
        size_t last_stage_off = 0;
        for (size_t n = 0; n < last_idx; n++) {
            last_stage_off += st * slices[n].n_space;
        }
        const double *y_last_stages = &y_stages[last_stage_off];
        const double *y_end_last = &y_last_stages[(st - 1) * d0];

        double *v_xi = (double *)malloc(d0 * sizeof(double));
        double *v_tau = (double *)malloc(d0 * sizeof(double));
        double *y_prev_trans = (double *)malloc(d0 * sizeof(double));

        eval_spatial_derivative(slices[last_idx], y_end_last, v_xi);

        // dY/dtau at tau = 1 (stage st - 1 of last element)
        const size_t prev_of_last = (last_idx == 0) ? (nt - 1) : (last_idx - 1);
        size_t prev_stage_off = 0;
        for (size_t n = 0; n < prev_of_last; n++) {
            prev_stage_off += st * slices[n].n_space;
        }
        const double *y_prev_end = &y_stages[prev_stage_off + (st - 1) * slices[prev_of_last].n_space];
        transfers[last_idx].apply(y_prev_end, y_prev_trans);

        const RadauDGMethod &rm = getRadauDG(st);
        const double inv_dtau_last = 1.0 / dtau[last_idx];
        const double *d_row = &rm.D_tau[(st - 1) * (st + 1)];
        for (size_t i = 0; i < d0; i++) {
            double val = d_row[0] * y_prev_trans[i];
            for (size_t j = 0; j < st; j++) {
                val += d_row[j + 1] * y_last_stages[j * d0 + i];
            }
            v_tau[i] = val * inv_dtau_last;
        }

        // Orthonormalize span(v_xi, v_tau) in weights[0] inner product
        const double *w0 = weights[0];
        const double nrm_xi = Krylov::internal::weighted_norm(d0, w0, v_xi);
        if (nrm_xi > 1e-14) {
            for (size_t i = 0; i < d0; i++) v_xi[i] /= nrm_xi;
        }
        const double proj_xt = Krylov::internal::weighted_dot(d0, w0, v_xi, v_tau);
        for (size_t i = 0; i < d0; i++) v_tau[i] -= proj_xt * v_xi[i];
        const double nrm_tau = Krylov::internal::weighted_norm(d0, w0, v_tau);
        const bool has_tau = (nrm_tau > 1e-12);
        if (has_tau) {
            for (size_t i = 0; i < d0; i++) v_tau[i] /= nrm_tau;
        }

        std::complex<double> *V0 = (std::complex<double> *)malloc(d0 * k_out * sizeof(std::complex<double>));
        m_ks_result.getEigenvectorsAtNode0(k_out, V0);

        // Score candidates near mu = +1
        double scores[k_out];
        for (size_t j = 0; j < k_out; j++) {
            const std::complex<double> mu(spec.mu_real[j], spec.mu_imag[j]);
            const double dist1 = std::abs(mu - std::complex<double>(1.0, 0.0));
            if (dist1 > 0.25) {
                scores[j] = -1.0;
                continue;
            }
            double norm2_v = 0.0;
            std::complex<double> p_xi(0.0, 0.0), p_tau(0.0, 0.0);
            const std::complex<double> *vj = &V0[j * d0];
            for (size_t i = 0; i < d0; i++) {
                norm2_v += w0[i] * std::norm(vj[i]);
                p_xi += w0[i] * v_xi[i] * vj[i];
                if (has_tau) {
                    p_tau += w0[i] * v_tau[i] * vj[i];
                }
            }
            const double overlap = (norm2_v > 0.0) ? ((std::norm(p_xi) + std::norm(p_tau)) / norm2_v) : 0.0;
            // Combine high overlap and closeness to +1
            scores[j] = overlap - 0.1 * dist1;
        }

        // Pick top 2 trivial modes
        int idx1 = -1, idx2 = -1;
        for (size_t j = 0; j < k_out; j++) {
            if (scores[j] < 0.0) continue;
            if (idx1 == -1 || scores[j] > scores[idx1]) {
                idx2 = idx1;
                idx1 = (int)j;
            }
            else if (idx2 == -1 || scores[j] > scores[idx2]) {
                idx2 = (int)j;
            }
        }
        // Fallback: if overlap didn't match (e.g. A -> 0 so v_tau = 0), pick closest to +1
        if (idx1 == -1 || idx2 == -1) {
            for (size_t j = 0; j < k_out; j++) {
                if ((int)j == idx1) continue;
                const double d = std::abs(std::complex<double>(spec.mu_real[j], spec.mu_imag[j]) - std::complex<double>(1.0, 0.0));
                if (idx1 == -1) {
                    idx1 = (int)j;
                }
                else if (idx2 == -1 || d < std::abs(std::complex<double>(spec.mu_real[idx2], spec.mu_imag[idx2]) - std::complex<double>(1.0, 0.0))) {
                    idx2 = (int)j;
                }
            }
        }

        spec.trivial_idx[0] = idx1;
        spec.trivial_idx[1] = idx2;
        double terr = 0.0;
        if (idx1 >= 0) {
            terr = std::max(terr, std::abs(std::complex<double>(spec.mu_real[idx1], spec.mu_imag[idx1]) - std::complex<double>(1.0, 0.0)));
        }
        if (idx2 >= 0) {
            terr = std::max(terr, std::abs(std::complex<double>(spec.mu_real[idx2], spec.mu_imag[idx2]) - std::complex<double>(1.0, 0.0)));
        }
        spec.trivial_err = terr;

        spec.n_unstable = 0;
        spec.max_nontrivial_mag = 0.0;
        for (size_t j = 0; j < k_out; j++) {
            if ((int)j == idx1 || (int)j == idx2) {
                continue;
            }
            const double mag = std::hypot(spec.mu_real[j], spec.mu_imag[j]);
            spec.max_nontrivial_mag = std::max(spec.max_nontrivial_mag, mag);
            if (mag > 1.0 + unstable_tol) {
                spec.n_unstable++;
            }
        }

        free(V0);
        free(y_prev_trans);
        free(v_tau);
        free(v_xi);
        for (size_t n = 0; n < nt; n++) {
            free(weights[n]);
        }
        if (local_conds) delete[] local_conds;
        if (local_trans) delete[] local_trans;

        return spec;
    }

    void FloquetAnalyzer::computeBloch(size_t nt, size_t st, const SpatialSlice *slices, const SparseDenseOutputOp *transfers, const double *y_stages, const double *dtau, double T, const double *p, double gamma, std::vector<double> &out_mu_real, std::vector<double> &out_mu_imag, size_t neig, size_t nsub, double tol)
    {
        if (std::abs(gamma) < 1e-14) {
            FloquetSpectrum spec = compute(nt, st, slices, transfers, nullptr, y_stages, dtau, T, p, neig, nsub, tol);
            out_mu_real = std::move(spec.mu_real);
            out_mu_imag = std::move(spec.mu_imag);
            return;
        }

        // Twisted spatial BC e^{i*gamma} y(0) = y(1) formulated as real 2*n_space system (Re, Im):
        // Uses DEV::ComplexMatrix with ComplexBoundaryConditions::makePhaseShift(gamma, node, 0, nullptr)!
        using complex_t = std::complex<double>;
        const RadauDGMethod &rm = getRadauDG(st);

        // Build per-element complex matrices for the complex-shift Hairer-Wanner solves:
        // Since A_inv has complex eigenvalues alpha +/- i*beta and spatial BC is complex e^{i*gamma},
        // treating Re(v) and Im(v) as 2 real components per stage decouples into `st` independent
        // ComplexMatrix solves for each (lambda_r, gamma):
        // Wait: (A_inv x I) has real eigenvalues gamma_r and complex conjugate pairs alpha_r +/- i*beta_r.
        // Because the spatial operator A_gamma has complex BCs, A_gamma(z) is C-linear!
        // Specifically, in C^n_space, the stage equation is:
        //   B_n * (Z_{n,j} - w_in) - dtau_T * sum_l a^R_{j,l} * A_gamma * Z_{n,l} = 0!
        // Using A_inv = V_c * diag(lambda_1, ..., lambda_st) * V_c^(-1) over C, it decouples into `st`
        // scalar ComplexMatrix solves (A_gamma - (lambda_r / dtau_T) * B_n) z_hat_r = b_hat_r in C^n_space!
        // For real representation in PeriodicKS (which uses real matrices), stack [Re(v); Im(v)] in R^(2*n_space)!

        // Pre-compute complex eigenvalues/vectors of A_inv via the real blocks T_s and (alpha, +/- beta):
        // Note: each complex pair column (v_R, v_I) in T_s corresponds to v_R + i v_I with eigenvalue alpha + i beta
        // and v_R - i v_I with eigenvalue alpha - i beta!
        struct ComplexCondenser
        {
            const SpatialSlice *slice = nullptr;
            size_t st = 0;
            size_t n_space = 0;
            double dtau_T = 1.0;
            DEV::ComplexBoundaryConditions bc;
            DEV::ComplexMatrix mats[4]; // one per eigenvalue lambda_k of (A^R)^(-1)
            complex_t Vc[16];
            complex_t Vc_inv[16];
            complex_t lam[4];
        };

        std::vector<ComplexCondenser> cconds(nt);
        std::vector<SparseDenseOutputOp> local_trans;
        if (transfers == nullptr) {
            local_trans.resize(nt);
            for (size_t n = 0; n < nt; n++) {
                const size_t prev = (n == 0) ? (nt - 1) : (n - 1);
                local_trans[n].build(slices[prev], slices[n]);
            }
            transfers = local_trans.data();
        }

        size_t offset = 0;
        for (size_t n = 0; n < nt; n++) {
            ComplexCondenser &cc = cconds[n];
            cc.slice = &slices[n];
            cc.st = st;
            cc.n_space = slices[n].n_space;
            cc.dtau_T = dtau[n] * T;
            cc.bc.makePhaseShift(gamma, (ptrdiff_t)slices[n].node, 0, nullptr);
            // Set BCp unused (nparam = 0)

            // Stage-averaged state
            std::vector<double> y_avg(cc.n_space, 0.0);
            for (size_t l = 0; l < st; l++) {
                for (size_t i = 0; i < cc.n_space; i++) {
                    y_avg[i] += y_stages[offset + l * cc.n_space + i] / (double)st;
                }
            }
            offset += st * cc.n_space;

            // Build Vc from T_s
            size_t col = 0;
            for (size_t r = 0; r < rm.n_real; r++) {
                cc.lam[col] = complex_t(rm.real_eigs[r], 0.0);
                for (size_t i = 0; i < st; i++) {
                    cc.Vc[i * st + col] = complex_t(rm.T[i * st + col], 0.0);
                }
                col++;
            }
            for (size_t r = 0; r < rm.n_complex; r++) {
                cc.lam[col] = complex_t(rm.complex_re[r], rm.complex_im[r]);
                cc.lam[col + 1] = complex_t(rm.complex_re[r], -rm.complex_im[r]);
                for (size_t i = 0; i < st; i++) {
                    const double vr = rm.T[i * st + col];
                    const double vi = rm.T[i * st + col + 1];
                    cc.Vc[i * st + col] = complex_t(vr, vi);
                    cc.Vc[i * st + col + 1] = complex_t(vr, -vi);
                }
                col += 2;
            }
            // Compute Vc_inv via Gaussian elimination on st x st
            complex_t aug[4][8];
            for (size_t i = 0; i < st; i++) {
                for (size_t j = 0; j < st; j++) {
                    aug[i][j] = cc.Vc[i * st + j];
                    aug[i][j + st] = (i == j) ? complex_t(1.0, 0.0) : complex_t(0.0, 0.0);
                }
            }
            for (size_t c_idx = 0; c_idx < st; c_idx++) {
                size_t piv = c_idx;
                for (size_t r = c_idx + 1; r < st; r++) {
                    if (std::abs(aug[r][c_idx]) > std::abs(aug[piv][c_idx])) piv = r;
                }
                for (size_t j = 0; j < 2 * st; j++) std::swap(aug[c_idx][j], aug[piv][j]);
                const complex_t d = aug[c_idx][c_idx];
                for (size_t j = c_idx; j < 2 * st; j++) aug[c_idx][j] /= d;
                for (size_t r = 0; r < st; r++) {
                    if (r != c_idx) {
                        const complex_t f = aug[r][c_idx];
                        for (size_t j = c_idx; j < 2 * st; j++) aug[r][j] -= f * aug[c_idx][j];
                    }
                }
            }
            for (size_t i = 0; i < st; i++) {
                for (size_t j = 0; j < st; j++) {
                    cc.Vc_inv[i * st + j] = aug[i][j + st];
                }
            }

            DEV::Matrix base_real;
            DEV::BoundaryConditions real_bc;
            real_bc.makePeriodic((ptrdiff_t)slices[n].node, 0);
            int pmask_dummy[2] = { 0, 0 };
            base_real.update(slices[n].fjac, slices[n].pjac, (ptrdiff_t)slices[n].nnodes, (ptrdiff_t)slices[n].node, (ptrdiff_t)slices[n].nstages, 0, slices[n].unscaled_spatial_period, slices[n].h, y_avg.data(), p, pmask_dummy, nullptr, &real_bc);

            for (size_t k = 0; k < st; k++) {
                cc.mats[k].copyPartitionsFrom(base_real, &cc.bc, false);
                cc.mats[k].addMassBlockSparse(slices[n].m_nnz, slices[n].m_rows, slices[n].m_cols, slices[n].m_vals, slices[n].c_A, slices[n].c_b, slices[n].h, -cc.lam[k] / cc.dtau_T, false);
                cc.mats[k].factor();
            }
        }

        std::vector<size_t> dims(nt);
        std::vector<double *> weights(nt);
        size_t max_n = 0;
        for (size_t n = 0; n < nt; n++) {
            const size_t src_idx = (n == 0) ? (nt - 1) : (n - 1);
            const size_t ns = slices[src_idx].n_space;
            max_n = std::max(max_n, ns);
            dims[n] = 2 * ns;
            weights[n] = (double *)malloc(2 * ns * sizeof(double));
            slices[src_idx].fillQuadratureWeights(weights[n], true);
            memcpy(weights[n] + ns, weights[n], ns * sizeof(double));
        }

        std::vector<double> tmp_r(max_n), tmp_i(max_n), bw_r(max_n), bw_i(max_n), mwork(max_n);
        std::vector<complex_t> b_hat(max_n), z_hat(max_n * st);

        auto apply_bloch = [&](size_t n, const double *v_in, double *v_out) {
            ComplexCondenser &cc = cconds[n];
            const size_t n_in = cc.n_space;
            const size_t src_idx = (n == 0) ? (nt - 1) : (n - 1);
            const size_t n_prev = slices[src_idx].n_space;
            transfers[n].apply(v_in, tmp_r.data());
            transfers[n].apply(v_in + n_prev, tmp_i.data());
            cc.slice->applyMass(tmp_r.data(), bw_r.data(), 1.0, 0.0, mwork.data());
            cc.slice->applyMass(tmp_i.data(), bw_i.data(), 1.0, 0.0, mwork.data());

            // RHS in stage space is 1_st x B*w, so after Vc_inv * A_inv * 1_st:
            for (size_t k = 0; k < st; k++) {
                complex_t coeff(0.0, 0.0);
                for (size_t l = 0; l < st; l++) {
                    coeff += cc.Vc_inv[k * st + l] * rm.A_inv_ones[l];
                }
                coeff /= (-cc.dtau_T);
                for (size_t i = 0; i < n_in; i++) {
                    b_hat[i] = coeff * complex_t(bw_r[i], bw_i[i]);
                }
                cc.mats[k].solve(b_hat.data(), &z_hat[k * n_in]);
            }

            // Last stage out = sum_k Vc[st - 1, k] * z_hat[k]
            for (size_t i = 0; i < n_in; i++) {
                complex_t z_last(0.0, 0.0);
                for (size_t k = 0; k < st; k++) {
                    z_last += cc.Vc[(st - 1) * st + k] * z_hat[k * n_in + i];
                }
                v_out[i] = z_last.real();
                v_out[n_in + i] = z_last.imag();
            }
        };

        Krylov::PeriodicKSResult res;
        Krylov::periodic_krylov_schur(apply_bloch, nt, dims.data(), weights.data(), 2 * neig, std::max<size_t>(2 * nsub, 2 * neig + 4), res, tol, false);

        const size_t k_out = std::min(2 * neig, res.nconverged > 0 ? res.nconverged : 2 * neig);
        out_mu_real.assign(res.wr, res.wr + k_out);
        out_mu_imag.assign(res.wi, res.wi + k_out);

        for (size_t n = 0; n < nt; n++) {
            free(weights[n]);
        }
    }
} // namespace TWIST
