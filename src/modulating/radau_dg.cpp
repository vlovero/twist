#include "modulating/radau_dg.h"
#include <array>
#include <stdexcept>

namespace TWIST
{
    namespace
    {
        template <size_t ST>
        constexpr std::array<double, ST * ST> make_p_in(const double *T_inv, const double *A_inv)
        {
            std::array<double, ST * ST> out{};
            for (size_t i = 0; i < ST; i++) {
                for (size_t j = 0; j < ST; j++) {
                    double s = 0.0;
                    for (size_t k = 0; k < ST; k++) {
                        s += T_inv[i * ST + k] * A_inv[k * ST + j];
                    }
                    out[i * ST + j] = s;
                }
            }
            return out;
        }

        template <size_t ST>
        constexpr std::array<double, ST> make_gamma_ones(const double *T_inv, const double *A_inv_ones)
        {
            std::array<double, ST> out{};
            for (size_t i = 0; i < ST; i++) {
                double s = 0.0;
                for (size_t k = 0; k < ST; k++) {
                    s += T_inv[i * ST + k] * A_inv_ones[k];
                }
                out[i] = s;
            }
            return out;
        }
        // =====================================================================
        // dG(1) : 2-stage Radau IIA (order 3)
        // =====================================================================
        constexpr double DG1_c[2] = { 1.0 / 3.0, 1.0 };
        constexpr double DG1_b[2] = { 0.75, 0.25 };
        constexpr double DG1_A[4] = {
            5.0 / 12.0, -1.0 / 12.0,
            0.75,        0.25
        };
        constexpr double DG1_A_inv[4] = {
             1.5, 0.5,
            -4.5, 2.5
        };
        constexpr double DG1_A_inv_ones[2] = { 2.0, -2.0 };
        constexpr double DG1_T[4] = {
            0.10540925533894599, -0.29814239699997197,
            0.94868329805051377,  0.0
        };
        constexpr double DG1_T_inv[4] = {
             0.0,                 1.0540925533894598,
            -3.3541019662496843,  0.372677996249965
        };
        constexpr double DG1_complex_re[1] = { 2.0 };
        constexpr double DG1_complex_im[1] = { 1.4142135623730951 };
        constexpr double DG1_D_tau[6] = {
            -2.0,  1.5, 0.5,
             2.0, -4.5, 2.5
        };
        constexpr double DG1_err_weights[3] = { 3.0, -4.5, 1.5 };

        // =====================================================================
        // dG(2) : 3-stage Radau IIA (order 5, default)
        // =====================================================================
        constexpr double DG2_c[3] = {
            0.1550510257216822, 0.64494897427831777, 1.0
        };
        constexpr double DG2_b[3] = {
            0.37640306270046725, 0.51248582618842164, 0.1111111111111111
        };
        constexpr double DG2_A[9] = {
            0.19681547722366041, -0.065535425850198392, 0.023770974348220151,
            0.39442431473908729,  0.29207341166522849, -0.041548752125997929,
            0.37640306270046725,  0.51248582618842164,  0.1111111111111111
        };
        constexpr double DG2_A_inv[9] = {
             3.2247448713915889,  1.1678400846904056, -0.25319726474218085,
            -3.5678400846904057,  0.77525512860841095, 1.0531972647421808,
             5.5319726474218083, -7.5319726474218083,  5.0
        };
        constexpr double DG2_A_inv_ones[3] = {
             4.139387691339814,
            -1.7393876913398141,
             3.0
        };
        constexpr double DG2_T[9] = {
            0.091232394870892949, -0.12845806217830097,  0.027308654751321444,
            0.24171793270710701,   0.18563595103095704, -0.34824890439657519,
            0.96604818261509284,   0.90940351764686178, -1.6653345369377348e-16
        };
        constexpr double DG2_T_inv[9] = {
             4.325579890063155,  0.33919925181580934, 0.5417705399358751,
            -4.5950103671960703, -0.36032719733585594, 0.52410568603675911,
             0.55296974905815477, -2.828147131551269,  0.65541774719600099
        };
        constexpr double DG2_real_eigs[1] = { 3.6378342527444953 };
        constexpr double DG2_complex_re[1] = { 2.6810828736277523 };
        constexpr double DG2_complex_im[1] = { 3.0504301992474105 };
        constexpr double DG2_D_tau[12] = {
            -4.139387691339814,  3.2247448713915889,  1.1678400846904056, -0.25319726474218085,
             1.7393876913398141, -3.5678400846904057, 0.77525512860841095, 1.0531972647421808,
            -3.0,                5.5319726474218083, -7.5319726474218083,  5.0
        };
        constexpr double DG2_err_weights[4] = {
            -10.0, 15.580782047249224, -8.9141153805825581, 3.333333333333333
        };

        // =====================================================================
        // dG(3) : 4-stage Radau IIA (order 7)
        // =====================================================================
        constexpr double DG3_c[4] = {
            0.088587959512703943, 0.40946686444073471, 0.787659461760847, 1.0
        };
        constexpr double DG3_b[4] = {
            0.22046221117676837, 0.38819346884317191, 0.32884431998005975, 0.0625
        };
        constexpr double DG3_A[16] = {
            0.11299947932315618, -0.040309220723522207, 0.025802377420336392, -0.0099046765072664245,
            0.23438399574740026,  0.2068925739353589,  -0.047857128048540719,  0.016047422806516273,
            0.21668178462325033,  0.4061232638673733,   0.18903651817005634,  -0.02418210489983294,
            0.22046221117676837,  0.38819346884317191,  0.32884431998005975,   0.0625
        };
        constexpr double DG3_A_inv[16] = {
             5.6441078759500893,  1.9235072770547126, -0.58590148210381632,  0.17387835257424572,
            -5.0492146383914092,  1.2211000288946918,  1.7546809887608368,  -0.43479146121258139,
             3.4924661586254375, -3.9845178957824965,  0.63479209515521873,  1.822137598434254,
            -6.9234882564454541,  6.5952376696281441, -12.17174941318269,   8.5
        };
        constexpr double DG3_A_inv_ones[4] = {
             7.155592023475231,
            -2.5082250819484617,
             1.9648779564324137,
            -4.0
        };
        constexpr double DG3_T[16] = {
             0.056065097932870886,  0.039375655283653964, -0.011063268554737167, -0.033290881388153185,
            -0.16981361104118367,  -0.025736324388209414,  0.043521415936430469, -0.030245407010078426,
             0.25436928992926566,  -0.37783408841409721,   0.31632861438586646,  -0.11838730248767126,
             0.87083001076563071,  -1.1102230246251565e-16, 0.93908523433465785,   8.3266726846886741e-17
        };
        constexpr double DG3_T_inv[16] = {
             2.0444069061846153, -4.2054701630153204,  0.49951344175938406, 0.050725367805411196,
             5.7820493987745207,  2.2318302619956887, -2.1961150385497623,  0.7044408272574707,
            -1.8958139506723697,  3.8998053567819917, -0.46320747037737547, 1.0178275543701063,
           -19.126385881492421,  -5.7386558571716781, -1.6023489864043057,  0.58037641280297381
        };
        constexpr double DG3_complex_re[2] = { 3.2128068968715349, 4.7871931031284669 };
        constexpr double DG3_complex_im[2] = { 4.7730874332766433, 1.5674764168952051 };
        constexpr double DG3_D_tau[20] = {
            -7.155592023475231,  5.6441078759500893,  1.9235072770547126, -0.58590148210381632,  0.17387835257424572,
             2.5082250819484617,-5.0492146383914092,  1.2211000288946918,  1.7546809887608368, -0.43479146121258139,
            -1.9648779564324137, 3.4924661586254375, -3.9845178957824965,  0.63479209515521873,  1.822137598434254,
             4.0,               -6.9234882564454541,  6.5952376696281441,-12.17174941318269,    8.5
        };
        constexpr double DG3_err_weights[5] = {
            35.0, -55.21381739209685, 34.078680832035786, -22.614863439938929, 8.7499999999999964
        };

        constexpr auto DG1_P_in = make_p_in<2>(DG1_T_inv, DG1_A_inv);
        constexpr auto DG1_gamma_ones = make_gamma_ones<2>(DG1_T_inv, DG1_A_inv_ones);
        constexpr auto DG2_P_in = make_p_in<3>(DG2_T_inv, DG2_A_inv);
        constexpr auto DG2_gamma_ones = make_gamma_ones<3>(DG2_T_inv, DG2_A_inv_ones);
        constexpr auto DG3_P_in = make_p_in<4>(DG3_T_inv, DG3_A_inv);
        constexpr auto DG3_gamma_ones = make_gamma_ones<4>(DG3_T_inv, DG3_A_inv_ones);

        constexpr RadauDGMethod METHOD_DG1 = {
            2, DG1_c, DG1_b, DG1_A, DG1_A_inv, DG1_A_inv_ones,
            DG1_T, DG1_T_inv, 0, 1, nullptr, DG1_complex_re, DG1_complex_im,
            DG1_D_tau, DG1_err_weights, 1.0 / 6.0,
            DG1_P_in.data(), DG1_gamma_ones.data()
        };
        constexpr RadauDGMethod METHOD_DG2 = {
            3, DG2_c, DG2_b, DG2_A, DG2_A_inv, DG2_A_inv_ones,
            DG2_T, DG2_T_inv, 1, 1, DG2_real_eigs, DG2_complex_re, DG2_complex_im,
            DG2_D_tau, DG2_err_weights, 1.0 / 120.0,
            DG2_P_in.data(), DG2_gamma_ones.data()
        };
        constexpr RadauDGMethod METHOD_DG3 = {
            4, DG3_c, DG3_b, DG3_A, DG3_A_inv, DG3_A_inv_ones,
            DG3_T, DG3_T_inv, 0, 2, nullptr, DG3_complex_re, DG3_complex_im,
            DG3_D_tau, DG3_err_weights, 1.0 / 5040.0,
            DG3_P_in.data(), DG3_gamma_ones.data()
        };
    } // namespace

    const RadauDGMethod &getRadauDG(size_t st)
    {
        switch (st) {
        case 2:
            return METHOD_DG1;
        case 3:
            return METHOD_DG2;
        case 4:
            return METHOD_DG3;
        default:
            throw std::invalid_argument("Unsupported Radau dG stage count (supported: 2, 3, 4)");
        }
    }

    void radauDenseWeights(size_t st, double theta, double *weights_out)
    {
        const RadauDGMethod &m = getRadauDG(st);
        const size_t n = st + 1;
        double nodes[5];
        nodes[0] = 0.0;
        for (size_t j = 0; j < st; j++) {
            nodes[j + 1] = m.c[j];
        }
        for (size_t j = 0; j < n; j++) {
            double prod = 1.0;
            for (size_t k = 0; k < n; k++) {
                if (k != j) {
                    prod *= (theta - nodes[k]) / (nodes[j] - nodes[k]);
                }
            }
            weights_out[j] = prod;
        }
    }
} // namespace TWIST
