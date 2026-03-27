#pragma once

#include <cstdint>
#include <thread>

#include "fmt/core.h"
#include "indicators/progress_bar.hpp"
#include "linalg.h"
#include "numpy_like.h"
#include "python/plotting.h"
#include "shared.h"
#include "sparse_matrix.h"


namespace Tables
{
    // Q matrices are used for 3rd order predictors
    // when computing the stages
    struct DIRK5
    {
        static constexpr int nstages = 5;
        static constexpr int error_order = 4;
        static constexpr std::array<double, nstages * nstages> A = { 1. / 4, 0, 0, 0, 0, 1. / 2.0, 1. / 4.0, 0, 0, 0, 17. / 50, -1. / 25, 1. / 4, 0, 0, 371. / 1360, -137. / 2720, 15. / 544, 1. / 4, 0, 25. / 24, -49. / 48, 125. / 16, -85. / 12, 1. / 4 };
        static constexpr std::array<double, nstages> B = { 25. / 24, -49. / 48, 125. / 16, -85. / 12, 1. / 4 };
        static constexpr std::array<double, nstages> C = { 1. / 4, 3. / 4, 11. / 20, 1. / 2, 1 };
        static constexpr std::array<double, nstages> E = { 3.8333333333333393, 1.416666666666666, -31.25, 28.333333333333332, 1.0 };
        static constexpr std::array<double, (nstages - 2) * nstages> Q = { 0.8800000000000006, 0.44, 0.0, 0.0, 0.0, 0.4776661884265906, 0.15059780009564766, 0.4866092778574844, 0.0, 0.0, -1.3964663990212065, 2.262968292899715, -0.2331843822627853, -0.4397164193499028, 0.0 };
    };

    struct DIRK865
    {
        static constexpr int nstages = 8;
        static constexpr int error_order = 6;
        static constexpr std::array<double, nstages * nstages> A = { 0.477264457385826, 0, 0, 0, 0, 0, 0, 0, -0.197052588415002, 0.476363428459584, 0, 0, 0, 0, 0, 0, -0.0347674430372966, 0.633051807335483, 0.193634310075028, 0, 0, 0, 0, 0, 0.0967797668578702, -0.193533526466535, -0.000207622945800473, 0.159572204849431, 0, 0, 0, 0, 0.162527231819875, -0.249672513547382, -0.0459079972041795, 0.36579476400859, 0.255752838307699, 0, 0, 0, -0.00707603197171262, 0.846299854860295, 0.344020016925018, -0.0720926054548865, -0.215492331980875, 0.104341097622161, 0, 0, 0.00176857935179744, 0.0779960013127515, 0.303333277564557, 0.213160806732836, 0.351769320319038, -0.381545894386538, 0.433517909105558, 0, 0, 0.22732353410559, 0.308415837980118, 0.157263419573007, 0.243551137152275, -0.120953626732831, -0.0802678473399899, 0.264667545261832 };
        static constexpr std::array<double, nstages> B = { 0, 0.22732353410559, 0.308415837980118, 0.157263419573007, 0.243551137152275, -0.120953626732831, -0.0802678473399899, 0.264667545261832 };
        static constexpr std::array<double, nstages> C = { 0.477264457385826, 0.279310840044582, 0.791918674373215, 0.0626108222949662, 0.488494323384602, 1, 1, 1 };
        static constexpr std::array<double, nstages> E = { 0.1549994616331405, 0.37424253743468133, -0.17247510925480963, 0.4365767355351293, -0.30375820816114457, -0.15085397406288115, -0.10011046454568737, 0.3300946989907722 };
        static constexpr std::array<double, (nstages - 2) * nstages> Q = { 1.2596870692450772, 0.6828048949463623, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10363964046330282, 0.2627468778179861, 0.048851339353163284, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1605631595387555, 0.864411908087465, 0.3879653462239415, 0.26271486491057144, 0.0, 0.0, 0.0, 0.0, 1.0679523984047115, -0.5164816194827835, 0.9926296489070142, -0.19789484774129712, -0.2848106862945633, 0.0, 0.0, 0.0, 0.3040114754415494, -0.1470255972194526, 0.28256952705198213, -0.0563342567833371, -0.0810763823287142, 0.715332372589192, 0.0, 0.0, 0.1772318183342294, -0.08571259982549881, 0.16473164709499227, -0.03284159833018621, -0.04726569825435262, 0.4170226039105448, 0.41702260391054413, 0.0 };
    };

    struct DIRK965
    {
        static constexpr int nstages = 9;
        static constexpr int error_order = 6;
        static constexpr std::array<double, nstages * nstages> A
        = { 0.218127781944908, 0, 0, 0, 0, 0, 0, 0, 0, -0.0903514856119419, 0.218127781944908, 0, 0, 0, 0, 0, 0, 0, 0.172952039138937, -0.35365501036282, 0.218127781944908, 0, 0, 0, 0, 0, 0, 0.511999875919193, 0.0289640332201925, -0.0144030945657094, 0.218127781944908, 0, 0, 0, 0, 0, 0.00465303495506782, -0.075635818766597, 0.217273030786712, -0.0206519428725472, 0.218127781944908, 0, 0, 0, 0, 0.896145501762472, 0.139267327700498, -0.186920979752805, 0.0672971012371723, -0.350891963442176, 0.218127781944908, 0, 0, 0, 0.552959701885751, -0.439360579793662, 0.333704002325091, -0.0339426520778416, -0.151947445912595, 0.0213825661026943, 0.218127781944908, 0, 0, 0.631360374036476, 0.724733619641466, -0.432170625425258, 0.598611382182477, -0.709087197034345, -0.483986685696934, 0.378391562905131, 0.218127781944908, 0, 0, -0.15504452530869, 0.194518478660789, 0.63515640279203, 0.81172278664173, 0.110736108691585, -0.495304692414479, -0.319912341007872, 0.218127781944908 };
        static constexpr std::array<double, nstages> B = { 0, -0.15504452530869, 0.194518478660789, 0.63515640279203, 0.81172278664173, 0.110736108691585, -0.495304692414479, -0.319912341007872, 0.218127781944908 };
        static constexpr std::array<double, nstages> C = { 0.218127781944908, 0.127776296332966, 0.0374248107210239, 0.744688596518583, 0.343766086047543, 0.783024769450069, 0.500923374474345, 0.925980212553918, 1 };
        static constexpr std::array<double, nstages> E = { -2.2993473150488936696, -2.1961739352594720209, 3.0092040338747536588, 2.1301646175095974023, 2.966781758146975978, -0.79142815210528860703, -2.391004906898235749, 1.1239089231802195101, -1.1599359734988115012 };
        static constexpr std::array<double, (nstages - 2) * nstages> Q = { -0.050252531694160114, 0.378679656440353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.715677094920489, -4.984610466513976, -2.225050355008739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.814493740666634, 0.9466458546430688, 0.3175451544893768, 0.04466216428721537, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01956273251471241, -0.17139679215691267, -0.06588755520467823, 1.1323263668742003, -0.11666846728320834, 0.0, 0.0, 0.0, 0.0, 0.28913289488713945, 0.3078763430051077, 0.10205684783114627, 0.17999789234854327, 0.5673935499055155, 0.08378235939049805, 0.0, 0.0, 0.0, -0.18760578820614304, -0.3467790930148002, -0.1218905219857048, 0.7398713510957944, -0.48674221988677446, 0.9583611184700155, -0.23615320265460657, 0.0, 0.0, -0.14080326535328258, -0.20945628078486558, -0.07224111189207992, 0.25920881711597454, -0.32432728649114645, 0.3692531738139112, -0.21899745717201646, 0.8630958701641663, 0.0 };
    };

    struct DIRK1175
    {
        static constexpr int nstages = 11;
        static constexpr int error_order = 6;
        static constexpr std::array<double, nstages * nstages> A = { 0.200252661187742, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.082947368165267, 0.200252661187742, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.483452690540751, 0, 0.200252661187742, 0, 0, 0, 0, 0, 0, 0, 0, 0.771076453481321, -0.22936926341842, 0.289733373208823, 0.200252661187742, 0, 0, 0, 0, 0, 0, 0, 0.0329683054968892, -0.162397421903366, 0.000951777538562805, 0, 0.200252661187742, 0, 0, 0, 0, 0, 0, 0.265888743485945, 0.606743151103931, 0.173443800537369, -0.0433968261546912, -0.385211017224481, 0.200252661187742, 0, 0, 0, 0, 0, 0.220662294551146, -0.0465078507657608, -0.0333111995282464, 0.011801580836998, 0.169480801030105, -0.0167974432139385, 0.200252661187742, 0, 0, 0, 0, 0.323099728365267, 0.0288371831672575, -0.0543404318773196, 0.0137765831431662, 0.0516799019060702, -0.0421359763835713, 0.181297932037826, 0.200252661187742, 0, 0, 0, -0.164226696476538, 0.187552004946792, 0.0628674420973025, -0.0108886582703428, -0.0117628641717889,
            0.0432176880867965, -0.0315206836275473, -0.0846007021638797, 0.200252661187742, 0, 0, 0.651428598623771, -0.10208078475356, 0.198305701801888, -0.0117354096673789, -0.0440385966743686, -0.0358364455795087, -0.0075408087654097, 0.160320941654639, 0.017940248694499, 0.200252661187742, 0, 0, -0.266259448580236, -0.615982357748271, 0.561474126687165, 0.266911112787025, 0.219775952207137, 0.387847665451514, 0.612483137773236, 0.330027015806089, -0.6965298655714, 0.200252661187742 };
        static constexpr std::array<double, nstages> B = { 0, -0.266259448580236, -0.615982357748271, 0.561474126687165, 0.266911112787025, 0.219775952207137, 0.387847665451514, 0.612483137773236, 0.330027015806089, -0.6965298655714, 0.200252661187742 };
        static constexpr std::array<double, nstages> C = { 0.200252661187742, 0.117305293022475, 0.683705351728493, 1.03169322445947, 0.0717753223198282, 0.817720512935815, 0.505580844098045, 0.702467581546438, 0.190890191608537, 1.02701610652231, 1 };
        static constexpr std::array<double, nstages> E = { 0.23150141408409325, 0.47963013735289145, -1.0761860073176586, -0.8861687694005007, 0.003124069703761789, 0.6754965614194881, 1.0640284761180319, -0.3355192599993853, -0.8985576255200071, 1.1411109623828555, -0.3078396184231458 };
        static constexpr std::array<double, (nstages - 2) * nstages> Q = { 6.8284271247461925, -5.8284271247461925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.4316616208603135, -3.3279299749595173, 2.7921731652231387, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11202557831032876, 0.0994659724655204, 0.18522832569233366, -0.08623453015106688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21483772161377457, 0.16282953730768754, 0.5179636337321731, 0.3819462182921097, 0.10326790224940695, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31819746277085575, 0.28149433890545866, 0.5321189454708477, -0.22477376604936922, 0.18159212439084274, 0.3227158212945191, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14289972708141974, 0.11656463671480713, 0.29639188210682815, 0.09217472908989044, 0.07455416712919126, 0.24610547555425036, 0.3082404709601835, 0.0, 0.0, 0.0, 0.0, 0.10752671501547338, 0.10074393341412603, 0.14705966328199507, -0.186121296421769, 0.06535603349509692, 0.05133729526634966, 0.2110974908878318, 0.13611607265498904,
            0.0, 0.0, 0.0, -0.08577346898736071, -0.11241026655730268, 0.0694771644856625, 0.7766578224665154, -0.07489540250224308, 0.2881627833276264, -0.11714366802205348, 0.0960468798047597, -0.15663944300849694, 0.0, 0.0, -0.031246351709363185, -0.04687351932844594, 0.05983545592465053, 0.3990432986102437, -0.0314908008508404, 0.1658083915975086, -0.03320151566065356, 0.0728120932706327, -0.06419693428075542, 0.3931032875176652, 0.0 };
    };
} // namespace Tables

ptrdiff_t adapt(const double eps, const ptrdiff_t Nx, const double *x, const ptrdiff_t node, const RP(double) z, const RP(double) y, RP(double) F, double **x_opt, double **y_new);
void solve_vanderT(const ptrdiff_t n, const double *alpha, const double *b, double *x);
void construct_L2_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &L2, const diffusion_t &diffusion, sparse::COOMatrix &tmp);
void construct_DL_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &DL, const diffusion_t &diffusion, sparse::COOMatrix &tmp);
void construct_DC_from_mesh(const ptrdiff_t node, const ptrdiff_t Nx, const double *x, sparse::COOMatrix &DC, const diffusion_t &diffusion, sparse::COOMatrix &tmp);

template <typename SDIRK_T, typename fpde_t>
inline bool quasi_newton_sdirk(RP(double) Z, RP(double) F, sparse::RealCSCMatrix &jac, RP(double) tmp, const double tn, const size_t Nx, const size_t node, RP(double) rn, RP(double) dz, fpde_t fpde, const double h, const double *yn, const double rtol)
{
    const double tol = std::max(10 * std::numeric_limits<double>::epsilon() / rtol, std::min(0.03, std::sqrt(rtol)));
    constexpr int MAX_ITER = 100;
    const size_t nz = Nx * node;
    size_t i;
    bool converged = false;
    double norm_dz, norm_dz_prev, rate, val;
    int stage, itern;

    for (stage = 0; stage < SDIRK_T::nstages; stage++) {
        // stage predictors
        if (stage == 0) {
            memset(&Z[stage * nz], 0, nz * sizeof(double));
        }
        else if (stage == 1) {
            memcpy(&Z[nz], Z, nz * sizeof(double));
        }
        else {
            dgemv("N", nz, stage, 1.0, Z, nz, &SDIRK_T::Q[(stage - 2) * SDIRK_T::nstages], 1, 0.0, &Z[stage * nz], 1);
        }
        converged = false;
        // newton iterations
        for (itern = 0; itern < MAX_ITER; itern++) {
            // evaluate Fi
            for (i = 0; i < nz; i++) {
                tmp[i] = yn[i] + Z[stage * nz + i];
            }
            fpde(tn + SDIRK_T::C[stage] * h, tmp, &F[stage * nz]);

            // evaluate (negative) residual
            memcpy(rn, &Z[stage * nz], nz * sizeof(double));
            dgemv("N", nz, stage + 1, h, F, nz, &SDIRK_T::A[stage * SDIRK_T::nstages], 1, -1.0, rn, 1);

            // solve for newton step
            jac.solve(rn, dz);
            norm_dz = norm2(dz, nz);

            // failure check
            if (itern) {
                rate = norm_dz / norm_dz_prev;
                val = std::pow(rate, MAX_ITER - itern) / (1 - rate) * norm_dz;
                if ((rate >= 1) || (val > tol)) {
                    break;
                }
            }

            // apply newton step
            for (i = 0; i < nz; i++) {
                Z[stage * nz + i] += dz[i];
            }

            // convergence check
            if (norm_dz == 0) {
                converged = true;
                break;
            }
            else if (itern) {
                val = rate / (1 - rate) * norm_dz;
                if (val < tol) {
                    converged = true;
                    break;
                }
            }
            norm_dz_prev = norm_dz;
        }
        if (!converged) {
            // fmt::println("failed on stage {}", stage);
            break;
        }

        // evaluate Fi at latest Zi for next stages
        if (stage != (SDIRK_T::nstages - 1)) {
            for (i = 0; i < nz; i++) {
                tmp[i] = yn[i] + Z[stage * nz + i];
            }
            fpde(tn + SDIRK_T::C[stage] * h, tmp, &F[stage * nz]);
        }

        // fmt::println("SOLVE(3) : stage {:d} converged in {:d} iterations", stage, itern + 1);
    }
    // puts("");
    return converged;
}

// std::tuple<double *, double *, ptrdiff_t, double> rxdiff_simulate1(func_t func_ode, func_t fjac_ode, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t node, const double *y, const double *p, const double tf, const double xeps, const double atol, const double rtol, const bool estimate_wave_speed, const bool adapt_mesh, 0, NULL, 0, NULL, NULL, false)
// std::tuple<double *, double *, ptrdiff_t, double> rxdiff_simulate(func_t func_ode, func_t fjac_ode, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t node, const double *y, const double *p, const double tf, const double xeps, const double atol, const double rtol, const bool estimate_wave_speed, const bool adapt_mesh, const ptrdiff_t nteval, const double *teval, const ptrdiff_t nxeval, const double *xeval, double *yeval, const bool animate = false)


/*
    NOTE: add arguments
        const ptrdiff_t nteval,
        const double *teval,
        const ptrdiff_t Nx_save,
        double *yeval
*/
template <typename SDIRK_T>
std::tuple<double *, double *, ptrdiff_t, double> rxdiff_simulate(func_t func_ode, func_t fjac_ode, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t node, const double *y, const double *p, const double tf, const double xeps, const double atol, const double rtol, const bool estimate_wave_speed, const bool adapt_mesh, const ptrdiff_t nteval, const double *teval, const ptrdiff_t nxeval, const double *xeval, double *yeval, const bool animate = false)
{
    using namespace indicators;
    int prog = 0;
    ProgressBar pbar{ option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[simulating] " }, option::ShowPercentage(true), option::ShowElapsedTime(true), option::ShowRemainingTime(true) };

    ptrdiff_t i, nz, Nx_new, nnz;
    // current state
    double *xn = nullptr;
    double *yn = nullptr;

    // next state
    double *xn1 = nullptr;
    double *yn1 = nullptr;

    double tn = 0.0;
    double h = 1e-2;

    sparse::COOMatrix jacpde_data, tmp_coo;
    sparse::RealCSCMatrix D2, J0;
    double factor, tn1;
    double *work = nullptr;
    double *fn = nullptr;
    double *rn = nullptr;
    double *dk = nullptr;
    double *tmp = nullptr;
    double *Z = nullptr;
    double *F = nullptr;
    double *yn1_old;
    double error, val;
    bool converged = false;
    bool realloc_data = true;
    double *bdiags_data = nullptr;
    int64_t *bdiags_rows = nullptr;
    int64_t *bdiags_cols = nullptr;

    // stuff for wave speed estimation
    double wave_speed = std::numeric_limits<double>::infinity();
    const double L = x[Nx - 1] - x[0];
    bool first_point_found = false;
    size_t peek_index;
    double first_time_point, first_space_point;
    double second_time_point, second_space_point;

    // stuff for evaluation points
    const bool saving_checkpoints = nteval != 0;
    // ptrdiff_t i_prev = 0;
    ptrdiff_t i_next = 1;

    // stuff for animation
    int animation_thread_code = INT_MAX;
    std::thread *animation_thread = nullptr;
    FILE *animation_file = nullptr;

    auto fpde = [&](double, const double *z, double *out) {
        ptrdiff_t i, j;
        double tmp[node];
        D2.gemv(0.0, out, 1.0, z);

        for (i = 0; i < Nx; i++) {
            func_ode(&z[i * node], p, tmp);
            for (j = 0; j < node; j++) {
                out[i * node + j] += tmp[j];
            }
        }
    };

    auto fjacpde = [&](double, const double *z) {
        ptrdiff_t i, j, k;
        for (i = 0; i < Nx; i++) {
            fjac_ode(&z[i * node], p, &bdiags_data[i * node * node]);
            for (j = 0; j < node; j++) {
                for (k = 0; k < node; k++) {
                    bdiags_rows[(i * node * node) + (j * node) + k] = i * node + j;
                    bdiags_cols[(i * node * node) + (j * node) + k] = i * node + k;
                }
            }
        }
        J0.updateFromCOO(jacpde_data);
    };

    // setup initial states
    nz = Nx * node;
    xn = (double *)malloc(Nx * sizeof(double));
    memcpy(xn, x, Nx * sizeof(double));
    yn = (double *)malloc(nz * sizeof(double));
    memcpy(yn, y, nz * sizeof(double));

    pbar.print_progress();

    if (saving_checkpoints) {
        interp(yeval, nxeval, xeval, xn, yn, Nx, node);
    }

    if (animate) {
        animation_file = fopen(".cache/animation.dat", "wb");
        fflush(animation_file);


        animation_thread = new std::thread([&] { animation_thread_code = run_python_script(python::animate::get_script(), {}); });


        fseek(animation_file, 0, SEEK_SET);
        double dNx = Nx;
        double dnode = node;
        fwrite(&dNx, 1, sizeof(double), animation_file);
        fwrite(&dnode, 1, sizeof(double), animation_file);
        fwrite(xn, 1, Nx * sizeof(double), animation_file);
        fwrite(yn, 1, nz * sizeof(double), animation_file);
        fflush(animation_file);
        // give python time to boot and create window
        std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
    }

    while (tn < tf) {
        if ((tn + h) > tf) {
            h = tf - tn;
        }
        // fmt::println("tn = {:.15g}, h = {:.8e}", tn, h);
        tn1 = tn + h;

        {
            int prog_new = (int)(100 * ((tn1 - 0) / (tf - 0)));
            int diff = std::max(prog_new - prog, 0);
            if (diff) {
                pbar.set_progress(prog_new);
                prog = prog_new;
            }
        }

        nz = Nx * node;
        // reallocate memory for current mesh size
        if (realloc_data) {
            // update jacobian data
            nnz = Nx * node * node + 3 * Nx * diffusion.size();
            jacpde_data.setNNZ(nnz);
            jacpde_data.reshape(nz, nz);
            // update work
            work = (double *)realloc(work, nz * (4 + 2 * SDIRK_T::nstages) * sizeof(double));
            fn = work;
            rn = fn + nz;
            dk = rn + nz;
            tmp = dk + nz;
            Z = tmp + nz;
            F = Z + SDIRK_T::nstages * nz;
            realloc_data = false;
        }

        // construct jacobians
        if ((tn == 0) || (adapt_mesh)) {
            // construct_diff_operator(xn, ...)
            construct_L2_from_mesh(node, Nx, xn, jacpde_data, diffusion, tmp_coo);
            D2.updateFromCOO(jacpde_data);
            jacpde_data.setNNZ(nnz); // restore correct nnz for data buffer
            // realloc is called when creating 2nd derivative operator
            bdiags_data = jacpde_data.data + 3 * Nx * diffusion.size();
            bdiags_rows = jacpde_data.irow + 3 * Nx * diffusion.size();
            bdiags_cols = jacpde_data.icol + 3 * Nx * diffusion.size();
        }

        fpde(tn, yn, fn);
        fjacpde(tn, yn);
        J0.scale(-SDIRK_T::A[0] * h);
        J0.addIdentity();

        converged = quasi_newton_sdirk<SDIRK_T>(Z, F, J0, tmp, tn, Nx, node, rn, dk, fpde, h, yn, rtol);
        if (!converged) {
            h *= 0.5;
            continue;
        }

        // get yn1(old mesh)
        yn1_old = rn;
        for (i = 0; i < nz; i++) {
            yn1_old[i] = yn[i] + Z[(SDIRK_T::nstages - 1) * nz + i];
        }
        // compute error
        dgemv("N", nz, SDIRK_T::nstages, 1.0, Z, nz, &SDIRK_T::E[0], 1, 0.0, dk, 1);

        error = 0.0;
        for (i = 0; i < nz; i++) {
            val = dk[i] / (atol + rtol * std::max(std::abs(yn[i]), std::abs(yn1_old[i])));
            error += val * val;
        }
        error = std::sqrt(error / nz);

        factor = error ? 0.9 * std::pow(error, -1.0 / SDIRK_T::error_order) : 10.0;
        factor = std::min(10.0, std::max(0.2, factor));
        h *= factor;
        h = std::max(1e-3, h);

        if ((error > 1) && (h > 1e-3)) {
            // bad step go back to top with updated step size
            continue;
        }

        if (saving_checkpoints) {
            // dk <- f(tn1, yn1)
            fpde(tn1, yn1_old, dk);
            // f0*t + t**3*(f0 + f1 + 2*y0 - 2*y1) + t**2*(-2*f0 - f1 - 3*y0 + 3*y1) + y0
            while ((i_next < nteval) && (teval[i_next] <= tn1)) {
                const double theta = (teval[i_next] - tn) / (tn1 - tn);
                const double h00 = 1 + theta * theta * (-3 + theta * 2);
                const double h10 = (tn1 - tn) * (theta * (1 + theta * (-2 + theta)));
                const double h01 = theta * theta * (3 - 2 * theta);
                const double h11 = (tn1 - tn) * (theta * theta * (theta - 1));
                // compute interpolant at teval[i_next] on current grid
                for (i = 0; i < node; i++) {
                    tmp[i] = (1 - theta) * yn[i] + theta * yn1_old[i];
                }
                for (i = node; i < (nz - node); i++) {
                    tmp[i] = (h00 * yn[i]) + (h10 * fn[i]) + (h01 * yn1_old[i]) + (h11 * dk[i]);
                }
                for (i = nz - node; i < nz; i++) {
                    tmp[i] = (1 - theta) * yn[i] + theta * yn1_old[i];
                }
                // interpolate interpolant on desired grid
                interp(&yeval[i_next * nxeval * node], nxeval, xeval, xn, tmp, Nx, node);
                i_next++;
            }
        }

        // otherwise adapt mesh (if option is set)
        if (adapt_mesh) {
            Nx_new = adapt(xeps, Nx, xn, node, yn1_old, yn, tmp, &xn1, &yn1);
        }
        else {
            if (tn == 0) {
                yn1 = (double *)realloc(yn1, nz * sizeof(double));
            }
            memcpy(yn1, yn1_old, nz * sizeof(double));
            xn1 = xn;
            Nx_new = Nx;
        }

        // update old with new
        tn = tn1;

        if (estimate_wave_speed) {
            peek_index = argmax(yn1_old, nz, node) / node;
            if ((!first_point_found) && ((0.9 * L) <= xn[peek_index]) && (xn[peek_index] < (0.99 * L))) {
                first_space_point = xn[peek_index];
                first_time_point = tn1;
                first_point_found = true;
            }
            else if (first_point_found && ((0.01 * L) < xn[peek_index]) && (xn[peek_index] <= (0.1 * L))) {
                second_space_point = xn[peek_index];
                second_time_point = tn1;
                wave_speed = std::min(wave_speed, std::abs((second_space_point - first_space_point) / (second_time_point - first_time_point)));
                first_point_found = false;
            }
        }
        // swap pointers to avoid extra copies
        std::swap(xn, xn1);
        std::swap(yn, yn1);
        realloc_data = Nx_new != Nx;
        Nx = Nx_new;

        if (animate && (animation_thread_code == INT_MAX)) {
            // fclose(animation_file);
            // fclose(fopen(".cache/animation.dat", "wb"));
            // animation_file = fopen(".cache/animation.dat", "wb");
            // animation_file = freopen(".cache/animation.dat", "wb", animation_file);
            fseek(animation_file, 0, SEEK_SET);
            double dNx = Nx;
            double dnode = node;
            fwrite(&dNx, 1, sizeof(double), animation_file);
            fwrite(&dnode, 1, sizeof(double), animation_file);
            fwrite(xn, 1, Nx * sizeof(double), animation_file);
            fwrite(yn, 1, Nx * node * sizeof(double), animation_file);
            fflush(animation_file);
        }
    }


    free(work);
    free(yn1);
    if (adapt_mesh) {
        free(xn1);
    }

    if (animate) {
        animation_thread->join();
        delete animation_thread;
        fclose(animation_file);
    }

    return { xn, yn, Nx, wave_speed };
}

void rxdiff_simulate_frozen(func_t func_ode, func_t fjac_ode, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t node, double *y, const double *p, const double tf, const double c0, std::vector<double> &cvalues);
std::tuple<double *, double *, ptrdiff_t, double> rxdiff_simulate_iif2(func_t func, func_t fjac, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t Ny, const double *y, const double *p, const double tf, const double xeps, const double atol, const double rtol, const bool estimate_wave_speed, const ptrdiff_t nteval, const double *teval, const ptrdiff_t nxeval, const double *xeval, double *yeval, const bool animate);