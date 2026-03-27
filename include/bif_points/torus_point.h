#pragma once

#include "cli/colors.h"
#include "collocator.h"
#include <valarray>
#include <vector>

namespace Hopf
{
    typedef struct
    {
        double a, b, c, d, e;
        double fa, fb, fc;
    } brent_state_t;

    template <typename func_t>
    int brent_init(void *vstate, func_t f, double *root, double x_lower, double x_upper)
    {
        brent_state_t *state = (brent_state_t *)vstate;

        double f_lower, f_upper;

        *root = 0.5 * (x_lower + x_upper);

        f_lower = f(x_lower);
        f_upper = f(x_upper);
        // exit(0);

        state->a = x_lower;
        state->fa = f_lower;

        state->b = x_upper;
        state->fb = f_upper;

        state->c = x_upper;
        state->fc = f_upper;

        state->d = x_upper - x_lower;
        state->e = x_upper - x_lower;

        if ((f_lower < 0.0 && f_upper < 0.0) || (f_lower > 0.0 && f_upper > 0.0)) {
            fmt::println("need opposite signs");
            exit(0);
        }

        return 0;
    }

    template <typename func_t>
    int brent_iterate(void *vstate, func_t f, double *root, double *x_lower, double *x_upper)
    {
        brent_state_t *state = (brent_state_t *)vstate;

        double tol, m;

        int ac_equal = 0;

        double a = state->a, b = state->b, c = state->c;
        double fa = state->fa, fb = state->fb, fc = state->fc;
        double d = state->d, e = state->e;

        if ((fb < 0 && fc < 0) || (fb > 0 && fc > 0)) {
            ac_equal = 1;
            c = a;
            fc = fa;
            d = b - a;
            e = b - a;
        }

        if (std::abs(fc) < std::abs(fb)) {
            ac_equal = 1;
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        tol = 0.5 * std::numeric_limits<double>::epsilon() * std::abs(b);
        m = 0.5 * (c - b);

        if (fb == 0) {
            *root = b;
            *x_lower = b;
            *x_upper = b;

            return 0;
        }

        if (std::abs(m) <= tol) {
            *root = b;

            if (b < c) {
                *x_lower = b;
                *x_upper = c;
            }
            else {
                *x_lower = c;
                *x_upper = b;
            }

            return 0;
        }

        if (std::abs(e) < tol || std::abs(fa) <= std::abs(fb)) {
            d = m; /* use bisection */
            e = m;
        }
        else {
            double p, q, r; /* use inverse cubic interpolation */
            double s = fb / fa;

            if (ac_equal) {
                p = 2 * m * s;
                q = 1 - s;
            }
            else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                q = (q - 1) * (r - 1) * (s - 1);
            }

            if (p > 0) {
                q = -q;
            }
            else {
                p = -p;
            }

            if (2 * p < std::min(3 * m * q - std::abs(tol * q), std::abs(e * q))) {
                e = d;
                d = p / q;
            }
            else {
                /* interpolation failed, fall back to bisection */

                d = m;
                e = m;
            }
        }

        a = b;
        fa = fb;

        if (std::abs(d) > tol) {
            b += d;
        }
        else {
            b += (m > 0 ? +tol : -tol);
        }

        fb = f(b);
        // SAFE_FUNC_CALL(f, b, &fb);

        state->a = a;
        state->b = b;
        state->c = c;
        state->d = d;
        state->e = e;
        state->fa = fa;
        state->fb = fb;
        state->fc = fc;

        /* Update the best estimate of the root and bounds on each
           iteration */

        *root = b;

        if ((fb < 0 && fc < 0) || (fb > 0 && fc > 0)) {
            c = a;
        }

        if (b < c) {
            *x_lower = b;
            *x_upper = c;
        }
        else {
            *x_lower = c;
            *x_upper = b;
        }

        return 0;
    }

    void locate_pairs(const std::vector<double> &wr, const std::vector<double> &wi, std::vector<ptrdiff_t> &indices)
    {
        size_t i;
        for (i = 0; i < wi.size(); i++) {
            if (wi[i] == 0) {
                continue;
            }
            if (wr[i] >= 0) {
                indices.emplace_back(i + 0);
                indices.emplace_back(i + 1);
                i++; // increment i because next one is going to be cc
            }
        }
    }

    double func(const double alpha, TWIST::Collocator *collocator, const ptrdiff_t target_npairs)
    {
        fmt::println("evaluating function with alpha = {:.15f}", alpha);
        const ptrdiff_t pindex = collocator->getContinuationParameterIndex();
        const double alpha_backup = collocator->p()[pindex];

        ptrdiff_t i, j, npairs, min_index, index;
        double min_distance, distance;
        std::complex<double> tmp;
        std::vector<double> wr, wi, beta;
        std::vector<std::complex<double>> mu;
        std::vector<ptrdiff_t> indices, next_largest;

        // make sure solution is valid
        collocator->p()[pindex] = alpha;
        collocator->solveWithAdaptation(0, 1e-12, 20, -1lu, 0.0, pow(0.5, 15), 100, false);

        // compute eigenvalues
        collocator->spectrum(wr, wi, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
        for (i = 0; i < wr.size(); i++) {
            if (std::abs(beta[i]) < (wr.size() * std::numeric_limits<double>::epsilon())) {
                continue;
            }
            mu.emplace_back(wr[i] / beta[i], wi[i] / beta[i]);
        }
        // sort for later
        std::sort(mu.begin(), mu.end(), [](const std::complex<double> &a, const std::complex<double> &b) { return a.real() > b.real(); });

        // check which indices are unstable
        for (i = 0; i < mu.size(); i++) {
            if (mu[i].imag() == 0) {
                continue;
            }
            else if (mu[i].real() > 0) {
                indices.emplace_back(i);
            }
        }
        assert((indices.size() & 1) == 0);
        npairs = indices.size() >> 1;

        /*
            if (npairs < tartget) {
                havent reeach pair count

            }
        */
        // get next closest complex pair of eigenvalues if alpha is before bifurcation point
        // fmt::println("[before check] indices.size() = {}", indices.size());
        if (npairs < target_npairs) {
            fmt::println("current number of pairs ({}) is less than target ({})", npairs, target_npairs);
            if (indices.size()) {
                fmt::println("found unstable modes");
                index = indices.back();
                min_index = -1;
                min_distance = std::numeric_limits<double>::infinity();
                for (i = 0; i < mu.size(); i++) {
                    if ((i == index) || (mu[i].imag() == 0)) {
                        continue;
                    }
                    distance = std::abs(mu[i] - mu[index]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        min_index = i;
                    }
                }
                tmp = mu[min_index];
            }
            else {
                for (i = 0; i < mu.size(); i++) {
                    if (mu[i].imag() == 0) {
                        continue;
                    }
                    min_index = i;
                    tmp = mu[i];
                    break;
                }
            }
            for (i = 0; i < indices.size(); i++) {
                mu[i] = mu[indices[i]];
            }
            mu[i] = tmp;
            mu[i + 1] = std::conj(tmp);
            mu.resize(indices.size() + 2);
        }
        else {
            fmt::println("current number of pairs ({}) is at least target ({})", npairs, target_npairs);
            for (i = 0; i < indices.size(); i++) {
                mu[i] = mu[indices[i]];
            }
            mu.resize(indices.size());
        }

        // fmt::println("indices.size() = {}", indices.size());
        // for (auto val : mu) {
        //     fmt::println("{:.8e}{:+.8e}", val.real(), val.imag());
        // }

        // compute test function
        tmp = 1.0;
        for (i = 0; i < mu.size(); i++) {
            for (j = i + 1; j < mu.size(); j++) {
                tmp *= (mu[i] + mu[j]);
            }
        }

        collocator->p()[pindex] = alpha_backup;

        return tmp.real();
    }

    void locate(TWIST::Collocator *collocator, double alpha_lo, double alpha_hi, ptrdiff_t target_npairs)
    {
        double alpha_star;
        brent_state_t state;
        auto r = [&](double alpha) { return func(alpha, collocator, target_npairs); };
        brent_init(&state, r, &alpha_star, alpha_lo, alpha_hi);
        fmt::println("|R| = {: .8e} [{:.12e}, {:.12e}]", state.fb, alpha_lo, alpha_hi);
        while ((std::abs(state.fb) > 1e-8) || (std::abs(alpha_hi - alpha_lo) > 1e-8)) {
            brent_iterate(&state, r, &alpha_star, &alpha_lo, &alpha_hi);
            fmt::println("|R| = {: .8e} [{:.12e}, {:.12e}]", state.fb, alpha_lo, alpha_hi);
        }
    }
}; // namespace Hopf

int count_pairs(const std::vector<double> &wr, const std::vector<double> &wi)
{
    size_t i;
    int npairs = 0;
    for (i = 0; i < wi.size(); i++) {
        if (wi[i] == 0) {
            continue;
        }
        if (wr[i] >= 0) {
            npairs++;
            fmt::println("{: .8e}{:+.8e}i", wr[i], wi[i]);
            // i++; // increment i because next one is going to be cc
        }
    }
    puts("");
    return npairs / 2;
}

/*
def Hopf_bialternate_det(ev):
    n = len(ev)
    d = 1
    for i in range(1, n):
        evi = ev[i]
        for j in range(i):
            d *= evi + ev[j]
    return d.real
*/

double phi(TWIST::Collocator *collocator, std::vector<double> &wr, std::vector<double> &wi, std::vector<double> &beta, const int pivot)
{
    size_t i, j;
    double min_real, max_real;
    std::vector<ptrdiff_t> indices;
    int npairs;

    wr.clear();
    wi.clear();
    beta.clear();
    // values.clear();

    collocator->spectrum(wr, wi, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

    npairs = 0;
    min_real = +std::numeric_limits<double>::infinity();
    max_real = -std::numeric_limits<double>::infinity();
    for (i = 0; i < wr.size(); i++) {
        // skip infinite eigenvalues
        if (std::abs(beta[i]) < (wr.size() * std::numeric_limits<double>::epsilon())) {
            continue;
        }
        // skip real only want complex for this
        if (wi[i] == 0) {
            continue;
        }

        if (wr[i] > 0) {
            npairs++;
            indices.emplace_back(i);
            min_real = std::min(min_real, wr[i]);
        }
        else {
            max_real = std::max(max_real, wr[i]);
        }
    }
    npairs >>= 1;

    if (npairs > pivot) {
        return min_real;
    }
    return max_real;
}


void locate_torus_with_secant(TWIST::Collocator *collocator1, TWIST::Collocator *collocator2, const int pivot)
{
    double alpha1, alpha2, phi1, phi2, rslope, alpha3, tol, delta, alpha_lo, alpha_hi, damp;
    std::vector<double> wr, wi, beta;
    ptrdiff_t param_index = collocator1->getContinuationParameterIndex();

    alpha1 = collocator1->getContinuationParameterValue();
    alpha2 = collocator2->getContinuationParameterValue();
    phi1 = phi(collocator1, wr, wi, beta, pivot);
    tol = 1e-8;

    alpha_lo = std::min(alpha1, alpha2);
    alpha_hi = std::max(alpha1, alpha2);

    while ((std::abs(alpha1 - alpha2) > tol) || (std::abs(phi1) > tol)) {
        phi2 = phi(collocator2, wr, wi, beta, pivot);
        rslope = (alpha2 - alpha1) / (phi2 - phi1);
        damp = 1.0;
        while (true) {
            alpha3 = alpha2 - damp * rslope * phi2;
            if ((alpha_lo <= alpha3) && (alpha3 <= alpha_hi)) {
                break;
            }
            damp *= 0.5;
        }
        // fmt::println("[{: .15f}, {: .15f}] [{: .15f}, {: .15f}]", alpha1, alpha2, phi1, phi2);
        fmt::println("|\u03d5| = {:.15e}", std::abs(phi2));

        phi1 = phi2;
        alpha1 = alpha2;
        alpha2 = alpha3;
        collocator1->copyCollocatorDataIntoSelf(*collocator2);
        ((double *)collocator2->p())[param_index] = alpha2;
        collocator2->solveWithAdaptation(0, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 100, false);
    }
    fmt::println("[{: .15f}, {: .15f}] [{: .15f}, {: .15f}]", alpha1, alpha2, phi1, phi2);
}


#if 1
void locate_torus_with_bisection(TWIST::Collocator *collocator1, TWIST::Collocator *collocator2)
{
    double s1, s2, s3;
    int npairs1, npairs2, npairs3, which;
    std::vector<double> wr1, wi1, wr2, wi2, beta;
    ptrdiff_t param_index;

    const ptrdiff_t nsub = 100;
    const double sigma = 0.2;

    param_index = collocator1->getContinuationParameterIndex();

    s1 = collocator1->getContinuationParameterValue();
    s2 = collocator2->getContinuationParameterValue();

    // collocator1->generateSubspace(nsub, wr1, wi1, sigma);
    collocator1->spectrum(wr1, wi1, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
    // collocator2->generateSubspace(nsub, wr2, wi2, sigma);
    collocator2->spectrum(wr2, wi2, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

    npairs1 = count_pairs(wr1, wi1);
    npairs2 = count_pairs(wr2, wi2);

    which = 1;
    fmt::println("should need {} iterations", (int)std::ceil(std::log2(std::abs(s2 - s1) / 1e-8)));
    while (std::abs(s1 - s2) > 1e-8) {
        s3 = 0.5 * (s1 + s2);
        fmt::println("it. {:2d} {}", which++, std::abs(s2 - s1));

        ((double *)collocator2->p())[param_index] = s3;
        collocator2->solveWithAdaptation(10, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 1000, false);

        wr2.clear();
        wi2.clear();
        beta.clear();
        // collocator2->generateSubspace(nsub, wr2, wi2, sigma);
        collocator2->spectrum(wr2, wi2, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

        npairs3 = count_pairs(wr2, wi2);
        if (npairs3 > npairs1) {
            // still too big
            s2 = s3;
        }
        else {
            s1 = s3;
        }
    }

    ((double *)collocator2->p())[param_index] = s3;
    collocator2->solveWithAdaptation(10, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 1000, false);
    wr2.clear();
    wi2.clear();
    beta.clear();
    // collocator2->generateSubspace(nsub, wr2, wi2, sigma);
    collocator2->spectrum(wr2, wi2, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
    for (size_t i = 0; i < wr2.size(); i++) {
        fmt::println(COLOR_GREEN "{:.3e}{:+.3e}i" COLOR_RESET, wr2[i], wi2[i]);
    }
}
#endif

std::tuple<int, double, double> get_single_crossing_interval(TWIST::Collocator *collocator1, TWIST::Collocator *collocator2)
{
    int npairs1, npairs2, npairs3;
    std::vector<double> wr1, wi1, wr2, wi2, beta;
    double param_lo, param_hi, param_mid;
    ptrdiff_t param_index;

    param_index = collocator1->getContinuationParameterIndex();

    collocator1->spectrum(wr1, wi1, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
    collocator2->spectrum(wr2, wi2, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);

    npairs1 = count_pairs(wr1, wi1);
    npairs2 = count_pairs(wr2, wi2);

    if (npairs2 <= npairs1) {
        puts("collocator1 must have fewer pairs than collocator2");
        exit(1);
    }

    param_lo = collocator1->getContinuationParameterValue();
    param_hi = collocator2->getContinuationParameterValue();

    // perform bisection method to find interval where only
    // ONE pair of eigenvalues crosses imaginary axis
    while ((npairs2 - npairs1) > 1) {
        fmt::println("[{:.6f}, {:.6f}] ({}, {}) {:.8e}", param_lo, param_hi, npairs1, npairs2, abs(param_hi - param_lo));
        param_mid = 0.5 * (param_lo + param_hi);
        // set value and solve
        ((double *)collocator2->p())[param_index] = param_mid;
        collocator2->solveWithAdaptation(10, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 1000, false);

        wr2.clear();
        wi2.clear();
        collocator2->spectrum(wr2, wi2, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
        npairs3 = count_pairs(wr2, wi2);

        if ((npairs3 - npairs1) > 1) {
            param_hi = param_mid;
        }
        else if (npairs3 <= npairs1) {
            param_lo = param_mid;
        }
        npairs2 = npairs3;
    }
    fmt::println("finished braketing with\n[{:.6f}, {:.6f}] ({}, {}) {:.8e}", param_lo, param_hi, npairs1, npairs2, abs(param_hi - param_lo));

    // collocator2 will have a parameter value where ONE pair
    // of cc eigenvaues crosses over. So collocator1 and
    // collocator2 can now be used in a secant iteration
    // for finding the hopf point

    // return upper bound for number of pairs
    // this will be needed when trying to find several
    // crossings in other routines
    return { npairs2, param_lo, param_hi };
}

void locate_all_torus_points(TWIST::Collocator *collocator1, TWIST::Collocator *collocator2, std::vector<TWIST::Collocator> &solutions)
{
    std::vector<double> wr, wi, beta;
    int npairs1, npairs2, i;
    double s_lower;
    ptrdiff_t param_index;

    solutions.clear();
    param_index = collocator1->getContinuationParameterIndex();

    collocator1->spectrum(wr, wi, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
    npairs1 = count_pairs(wr, wi);
    fmt::println("npairs1 = {}", npairs1);
    wr.clear();
    wi.clear();

    collocator2->spectrum(wr, wi, beta, TWIST::SpectrumStrategy::shiftAndInvert, false, NULL);
    npairs2 = count_pairs(wr, wi);
    fmt::println("npairs2 = {}", npairs2);

    // want collocator1 to have fewer pairs
    if (npairs2 < npairs1) {
        std::swap(collocator1, collocator2);
        std::swap(npairs1, npairs2);
    }

    for (i = 0; i < (npairs2 - npairs1); i++) {
        // on exit collocator_tmp will have npairs1 + 1 pairs
        TWIST::Collocator collocator_tmp{ *collocator2 };
        auto [target, alpha_lo, alpha_hi] = get_single_crossing_interval(collocator1, &collocator_tmp);
        // get value where npairs1 + 1 exists for later
        s_lower = collocator_tmp.getContinuationParameterValue();

        // refine. On exit collocator_tmp will be appx at the point
        // where the solution transitions from npairs1 -> npairs1 + 1
        collocator_tmp.copyCollocatorDataIntoSelf(*collocator1);
        fmt::println("starting brent with interval [{:.15f} {:.15f}]", alpha_hi, alpha_lo);
        Hopf::locate(&collocator_tmp, alpha_hi, alpha_lo, target);
        // locate_torus_with_secant(collocator1, &collocator_tmp, npairs1);
        // push to make copy
        solutions.push_back(collocator_tmp);

        // update collocator1 to have new lower bound for next iteration
        // in for loop
        ((double *)collocator1->p())[param_index] = s_lower;
        collocator1->solveWithAdaptation(10, 1e-12, 20, -1lu, 0.0, std::pow(0.5, 15), 1000, false);
    }
}
