#include "collocator.h"

double computePhaseCondition(size_t nnodes, size_t nstages, size_t node, const RP(double) h, const RP(double) b, const RP(double) Y, const RP(double) Yold, const RP(double) Ypold)
{
    // I know this isn'double the best code practice
    // but it matches the math notation...
    const size_t n = nnodes - 1;
    const size_t s = nstages;
    const size_t m = node;

    /*
        Y has shape (n * (s + 1) + 1, m)
        need to view first n * (s + 1) rows as "3d" with shape
        (n, s + 1, m) then indexing will be
        [k, i, j] -> ((m * (s + 1)) * k) + ((m) * i) + (j)
    */

    double phase_condition, sum1, sum2, val;
    size_t i, j, k, stride_k, stride_i, l;

    phase_condition = 0.0;
    for (k = 0; k < n; k++) {
        sum1 = 0.0;
        stride_k = m * (s + 1) * k;

        for (i = 0; i < s; i++) {
            sum2 = 0.0;
            // + 1 to skip node and jump to collocation point
            stride_i = m * (i + 1);
            for (j = 0; j < m; j++) {
                l = stride_k + stride_i + j;
                val = Ypold[l] * (Y[l] - Yold[l]);
                sum2 += val;
            }
            sum1 += b[i] * sum2;
        }
        phase_condition += h[k] * sum1;
    }

    return phase_condition;
}

namespace TWIST
{
    class HopfCollocator
    {
        std::vector<Collocator> m_collocators;
        double *m_t = nullptr;
        double *m_h = nullptr;
        uint64_t m_nt;
        double m_temporal_period;

        HopfCollocator(const Collocator &collocator, const double *psi_real, const double *psi_imag, const double omega, const double ds)
        {
            uint64_t i, j, N;
            double dt, sn, cs;

            m_temporal_period = (2 * M_PI) / omega;
            m_nt = 50;
            m_t = (double *)malloc(m_nt * sizeof(double));
            m_h = (double *)malloc((m_nt - 1) * sizeof(double));

            dt = m_temporal_period / (m_nt - 1);
            m_collocators.reserve(m_nt);
            N = collocator.getNode() * ((collocator.getNNodes() - 1) * (collocator.getNStages() + 1) + 1);

            for (i = 0; i < m_nt; i++) {
                m_t[i] = (i * dt) / m_temporal_period;
            }
            for (i = 1; i < m_nt; i++) {
                m_h[i - 1] = m_t[i] - m_t[i - 1];
            }

            for (i = 0; i < m_nt; i++) {
                m_collocators.emplace_back(collocator);
                Collocator &collocator_curr = m_collocators.back();
                double *y = collocator_curr.y();

                sn = -std::sin(omega * i * dt);
                cs = std::cos(omega * i * dt);
                for (j = 0; j < N; j++) {
                    // Re((cs + isn) * (pr + i pi)) = cs pr - sn pi
                    y[j] += ds * 2 * (psi_real[j] * cs - psi_imag[j] * sn);
                }
            }
        }

        ~HopfCollocator()
        {
            if (m_t) {
                free(m_t);
                m_t = nullptr;
            }
            if (m_h) {
                free(m_h);
                m_h = nullptr;
            }
        }

        void evaluate_residual()
        {
        }
    };
} // namespace TWIST