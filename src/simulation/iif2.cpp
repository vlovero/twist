#include "preprocess/drxd.h"
#include "shared.h"
#include "thread_pool.h"
#include <algorithm>
#include <alloca.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <omp.h>


#define PARALLEL_NEWTON 1

// these are copied from FFTW header
#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3)     /* NO_EXHAUSTIVE is default */
#define FFTW_PRESERVE_INPUT (1U << 4) /* cancels FFTW_DESTROY_INPUT */
#define FFTW_PATIENT (1U << 5)        /* IMPATIENT is default */
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_WISDOM_ONLY (1U << 21)

#define LOCAL_NEWTON_MAX_ITER 30

using complex_t = std::complex<double>;
using fftw_complex = complex_t;

BS::light_thread_pool pool(2);
#if PARALLEL_NEWTON
BS::light_thread_pool newton_pool(2);
#endif

// I want to use the overloaded operators and exp function
// from std::complex. fftw_complex is just a double[2] so
// the memory layout is identical to std::complex and
// C's _Complex. To use std::complex I can't include
// fftw header so extern everything I'll be using.

extern "C" {
    struct __plan;
    typedef struct __plan *fftw_plan;

    void fftw_destroy_plan(fftw_plan p);
    fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned int flags);
    void fftw_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out);
    fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, unsigned int flags);
    void fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out);
    void *fftw_malloc(size_t n);
    void fftw_free(void *p);
}

// Context and work structs to simplify the time stepping
// and integrtion loop. It's not the most efficient to use
// different memory blocks, but it made the code so much
// nicer and easier to write

struct RFFTContext
{
    fftw_plan plan;

    RFFTContext() = delete;

    RFFTContext(const int N, const int NF, const diffusion_t &diffusion)
    {
        plan = fftw_plan_many_dft_r2c(1, &N, diffusion.size(), NULL, NULL, 1, N, NULL, NULL, 1, NF, FFTW_ESTIMATE);
    }

    void run(double *input, fftw_complex *output) const
    {
        fftw_execute_dft_r2c(plan, input, output);
    }

    ~RFFTContext()
    {
        fftw_destroy_plan(plan);
    }
};

struct IRFFTContext
{
    fftw_plan plan;

    IRFFTContext() = delete;

    IRFFTContext(const int N, const int NF, const diffusion_t &diffusion)
    {
        plan = fftw_plan_many_dft_c2r(1, &N, diffusion.size(), NULL, NULL, 1, NF, NULL, NULL, 1, N, FFTW_ESTIMATE);
    }

    void run(fftw_complex *input, double *output) const
    {
        fftw_execute_dft_c2r(plan, input, output);
    }

    ~IRFFTContext()
    {
        fftw_destroy_plan(plan);
    }
};

struct IIF2Work
{
    double *b;
    double *bdiff;
    fftw_complex *tmp;
    bool *converged;

    IIF2Work() = delete;

    IIF2Work(const int Nx, const int Ny, const int Ndiff)
    {
        b = (double *)fftw_malloc(Nx * Ny * sizeof(double));
        bdiff = (double *)fftw_malloc(Nx * Ndiff * sizeof(double));
        tmp = (fftw_complex *)fftw_malloc((Nx / 2 + 1) * Ndiff * sizeof(fftw_complex));
        converged = (bool *)malloc(Nx);
    }

    ~IIF2Work()
    {
        fftw_free(b);
        fftw_free(bdiff);
        fftw_free(tmp);
        free(converged);
    }
};

void local_newton(const ptrdiff_t i, const double hdt, const double *p, func_t func, func_t fjac, RP(bool) converged, const ptrdiff_t Nx, const ptrdiff_t Ny, const double *b, RP(double) zn1, RP(double) fn1)
{
    ptrdiff_t j, k;
    double norm_ri, norm_dx;
    double zn1_i[Ny], fn1_i[Ny], b_i[Ny], r_i[Ny], J_i[Ny * Ny];
    int ipiv[Ny], info[1];

    // newton
    converged[i] = 0;

    // read into contiguous buffer
    for (j = 0; j < Ny; j++) {
        zn1_i[j] = zn1[j * Nx + i];
        b_i[j] = b[j * Nx + i];
    }
    // get jacobian and scale it plus add Id
    fjac(zn1_i, p, J_i);
    for (j = 0; j < Ny; j++) {
        for (k = 0; k < Ny; k++) {
            J_i[j * Ny + k] = ((j == k) ? 1.0 : 0.0) - hdt * J_i[j * Ny + k];
        }
    }
    // factor it
    dgetrf(Ny, Ny, J_i, Ny, ipiv, info);

    for (j = 0; j < LOCAL_NEWTON_MAX_ITER; j++) {
        /*
        rn = hdt * f(zn1_i, p, fn1_i) + b_i - zn1_i
        */
        func(zn1_i, p, fn1_i);
        for (k = 0; k < Ny; k++) {
            r_i[k] = hdt * fn1_i[k] + b_i[k] - zn1_i[k];
        }
        norm_ri = norm2(r_i, Ny);
        dgetrs("T", Ny, 1, J_i, Ny, ipiv, r_i, Ny, info);
        norm_dx = norm2(r_i, Ny);
        for (k = 0; k < Ny; k++) {
            zn1_i[k] += r_i[k];
        }
        if ((norm_dx < 1e-8) || (norm_ri < 1e-8)) {
            converged[i] = 1;
            break;
        }
    }

    if (converged[i]) {
        // if converged save fn1 for later steps
        for (j = 0; j < Ny; j++) {
            fn1[j * Nx + i] = fn1_i[j];
            zn1[j * Nx + i] = zn1_i[j];
        }
    }
}

void step_iif2_fftw(bool *step_converged, RP(double) zn1, RP(double) fn1, func_t func, func_t fjac, const double *p, const ptrdiff_t Nx, const ptrdiff_t Ny, const diffusion_t &diffusion, const double *zn, const double *fn, const double dt, const fftw_complex *omega, const RFFTContext *rfft, const IRFFTContext *irfft, IIF2Work *work)
{
    const ptrdiff_t Nz = Nx * Ny;
    const ptrdiff_t NxF = (Nx / 2) + 1;
    const double hdt = 0.5 * dt;

    ptrdiff_t i, j;
    double *b = nullptr, *bdiff = nullptr;
    bool *converged = nullptr;
    fftw_complex *tmp = nullptr;

    b = work->b;
    bdiff = work->bdiff;
    tmp = work->tmp;
    converged = work->converged;

    for (i = 0; i < Nz; i++) {
        b[i] = zn[i] + hdt * fn[i];
    }

    // do only diffusive

    // copy diffusive b into buffer
    i = 0;
    for (const auto &[index, _] : diffusion) {
        memcpy(&bdiff[i * Nx], &b[index * Nx], Nx * sizeof(double));
        i++;
    }

    // rfft buffer
    rfft->run(bdiff, tmp);
    // apply spectrum of integrating factor matrix
    j = 0;
    for (const auto &[_, coeff] : diffusion) {
        for (i = 0; i < NxF; i++) {
            auto val = std::exp(coeff * dt * omega[i]);
            tmp[j * NxF + i] *= val;
        }
        j++;
    }

    // irfft buffer
    irfft->run(tmp, bdiff);
    for (i = 0; i < (ptrdiff_t)(Nx * diffusion.size()); i++) {
        bdiff[i] /= Nx;
    }

    // update diffusive rows of b
    i = 0;
    for (const auto &[index, _] : diffusion) {
        memcpy(&b[index * Nx], &bdiff[i * Nx], Nx * sizeof(double));
        i++;
    }


    // zn is the initial guess
    memcpy(zn1, zn, Nz * sizeof(double));

#if PARALLEL_NEWTON
    newton_pool.detach_loop(0l, Nx, [=](const ptrdiff_t i) { local_newton(i, hdt, p, func, fjac, converged, Nx, Ny, b, zn1, fn1); });
    newton_pool.wait();
#else
    for (i = 0; i < Nx; i++) {
        local_newton(i, hdt, p, func, fjac, converged, Nx, Ny, b, zn1, fn1);
    }
#endif
    // pool.wait();

    for (i = 0; i < Nx; i++) {
        if (converged[i] == 0) {
            *step_converged = false;
            return;
        }
    }

    *step_converged = true;
}

void setup_omega_iif2_fftw(const ptrdiff_t Nx, const double *x, complex_t **omega)
{
    ptrdiff_t i;
    const ptrdiff_t Nf = ((Nx - 1) / 2) + 1;
    double *col = (double *)fftw_malloc(Nx * sizeof(double));
    const double L = x[Nx - 1] - x[0];
    const double h = (2 * M_PI) / (Nx - 1);
    const double scale = std::pow((2 * M_PI) / L, 2);

    *omega = (complex_t *)fftw_malloc(Nf * sizeof(complex_t));

    // I can never remember the correct fft frequencies on [0, L]
    // so construct differentiation matrix column instead
    // then take fft of that
    if (Nx & 1) {
        col[0] = -(std::pow((M_PI / h), 2) / 3.0 + (1.0 / 6.0));
        col[0] *= scale;
        for (i = 1; i < Nx - 1; i++) {
            col[i] = -0.5 * std::pow(-1, i) / std::pow(std::sin(i * h * 0.5), 2);
            col[i] *= scale;
        }
    }
    else {
        col[0] = -(std::pow(M_PI / h, 2) / 3.0 + (1.0 / 12.0));
        col[0] *= scale;
        for (i = 1; i < (Nx - 1); i++) {
            const double t = 0.5 * i * h;
            const double c = std::cos(t);
            const double s = std::sin(t);
            col[i] = -0.5 * std::pow(-1, i) * c / (s * s);
            col[i] *= scale;
        }
    }

    RFFTContext rfft(Nx - 1, Nf, diffusion_t(1));
    rfft.run(col, *omega);

    fftw_free(col);
}

std::tuple<double *, double *, ptrdiff_t, double> rxdiff_simulate_iif2(func_t func, func_t fjac, const diffusion_t &diffusion, ptrdiff_t Nx, const double *x, ptrdiff_t Ny, const double *y, const double *p, const double tf, const double xeps, const double atol, const double rtol, const bool estimate_wave_speed, const ptrdiff_t nteval, const double *teval, const ptrdiff_t nxeval, const double *xeval, double *yeval, const bool animate)
{
    using namespace indicators;
    ProgressBar pbar(option::BarWidth{ 50 }, option::Start{ "|" }, option::Fill{ "█" }, option::Lead{ "█" }, option::Remainder{ "-" }, option::End{ "|" }, option::PrefixText{ "[simulating] " }, option::ShowPercentage(true), option::ShowElapsedTime(true), option::ShowRemainingTime(true));

    ptrdiff_t i, j, peak_index;
    bool *converged = (bool *)alloca(2 * sizeof(bool));
    double tn, tn1, h, norm_error, tmp, factor;
    double wave_speed;
    double first_time_point, first_space_point, second_time_point, second_space_point;
    bool first_point_found;
    double *yn, *yn1, *yn12, *yn2, *fn, *fn1, *fn12, *fn2, *rwork, *error;
    complex_t *omega;
    void *work;
    int prog;

    ptrdiff_t Nz = Nx * Ny;
    const double L = x[Nx - 1] - x[0];

    // animation stuff
    int animation_thread_code = INT_MAX;
    std::thread *animation_thread = nullptr;
    FILE *animation_file = nullptr;

    // stuff for evaluation points
    const bool saving_checkpoints = nteval != 0;
    // ptrdiff_t i_prev = 0;
    ptrdiff_t i_next = 1;

    // fft stuff
    RFFTContext rfft(Nx - 1, ((Nx - 1) / 2) + 1, diffusion);
    IRFFTContext irfft(Nx - 1, ((Nx - 1) / 2) + 1, diffusion);
    IIF2Work work1(Nx - 1, Ny, diffusion.size());
    IIF2Work work2(Nx - 1, Ny, diffusion.size());

    // unused but here for syntax compatibility with sdirk methods
    (void)xeps;
    (void)xeval;

    work = malloc(10 * Nz * sizeof(double));
    yn = (double *)work;
    yn1 = yn + Nz;
    yn12 = yn1 + Nz;
    yn2 = yn12 + Nz;
    fn = yn2 + Nz;
    fn1 = fn + Nz;
    fn12 = fn1 + Nz;
    fn2 = fn12 + Nz;

    // no need to allocate these two
    rwork = fn12;
    error = fn12;
    prog = 0;
    pbar.print_progress();

    if (saving_checkpoints) {
        memcpy(yeval, y, Nx * Ny * sizeof(double));
    }

    setup_omega_iif2_fftw(Nx, x, &omega);

    Nx -= 1;
    Nz -= Ny;

    // copy init state into local workspace
    // NOTE: this routine works with the transposed system
    for (i = 0; i < Nx; i++) {
        func(&y[i * Ny], p, rwork);
        for (j = 0; j < Ny; j++) {
            fn[j * Nx + i] = rwork[j];
            yn[j * Nx + i] = y[i * Ny + j];
        }
    }

    tn = 0;
    h = std::max(1.4e-8, 0.1 * std::sqrt(std::max(atol, rtol)));
    wave_speed = std::numeric_limits<double>::infinity();
    first_point_found = false;

    if (animate) {
        animation_file = fopen(".cache/animation.dat", "wb");
        fflush(animation_file);

        animation_thread = new std::thread([&] { animation_thread_code = run_python_script(python::animate::get_script(), {}); });
        // give python time to boot and create window
        std::this_thread::sleep_for(std::chrono::duration<double>(2.0));

        fseek(animation_file, 0, SEEK_SET);
        double dNx = Nx;
        double dnode = 1;
        fwrite(&dNx, 1, sizeof(double), animation_file);
        fwrite(&dnode, 1, sizeof(double), animation_file);
        fwrite(x, 1, Nx * sizeof(double), animation_file);
        fwrite(yn, 1, Nx * sizeof(double), animation_file);
        fflush(animation_file);
        // allow for initial drawing to finish before entering integration loop
        std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
    }

    while (tn < tf) {
        if (tf < (tn + h)) {
            h = tf - tn;
        }
        tn1 = tn + h;

        {
            int prog_new = (int)(100 * ((tn1 - 0) / (tf - 0)));
            int diff = std::max(prog_new - prog, 0);
            if (diff) {
                pbar.set_progress(prog_new);
                prog = prog_new;
            }
        }

        auto rfft_addr = &rfft;
        auto irfft_addr = &irfft;
        auto work1_addr = &work1;
        auto work2_addr = &work2;


        pool.detach_task([=]() { step_iif2_fftw(converged, yn1, fn1, func, fjac, p, Nx, Ny, diffusion, yn, fn, h, omega, rfft_addr, irfft_addr, work1_addr); }, BS::pr::highest);
        pool.detach_task([=]() { step_iif2_fftw(&converged[1], yn12, fn12, func, fjac, p, Nx, Ny, diffusion, yn, fn, 0.5 * h, omega, rfft_addr, irfft_addr, work2_addr); }, BS::pr::highest);
        pool.wait();

        if (!(converged[0])) {
            h *= 0.5;
            continue;
        }
        // continue;

        step_iif2_fftw(converged, yn2, fn2, func, fjac, p, Nx, Ny, diffusion, yn12, fn12, 0.5 * h, omega, rfft_addr, irfft_addr, work2_addr);

        norm_error = 0.0;
        for (i = 0; i < (Nx * Ny); i++) {
            error[i] = (yn2[i] - yn1[i]) / 3.0;
            tmp = error[i] / (atol + rtol * std::max(std::abs(yn[i]), std::abs(yn2[i])));
            norm_error += tmp * tmp;
        }
        norm_error = std::sqrt(norm_error / (Nx * Ny));
        factor = norm_error ? 0.9 * std::pow(norm_error, -1.0 / 3.0) : 10.0;
        h *= factor;
        if (1.0 < norm_error) {
            continue;
        }

        if (saving_checkpoints) {
            // throw std::runtime_error("IIF2 checkpoints not implemented");
            // dk <- f(tn1, yn1)
            // f0*t + t**3*(f0 + f1 + 2*y0 - 2*y1) + t**2*(-2*f0 - f1 - 3*y0 + 3*y1) + y0
            while ((i_next < nteval) && (teval[i_next] <= tn1)) {
                double *yeval_view = &yeval[i_next * nxeval * Ny];
                // 1. Interpolate in time
                const double theta = (teval[i_next] - tn) / (tn1 - tn);
                const double h00 = 1 + theta * theta * (-3 + theta * 2);
                const double h10 = (tn1 - tn) * (theta * (1 + theta * (-2 + theta)));
                const double h01 = theta * theta * (3 - 2 * theta);
                const double h11 = (tn1 - tn) * (theta * theta * (theta - 1));
                // compute interpolant at teval[i_next] on current grid
                for (i = 0; i < Nx; i++) {
                    for (j = 0; j < Ny; j++) {
                        yeval_view[i * Ny + j] = (h00 * yn[j * Nx + i]) + (h10 * fn[j * Nx + i]) + (h01 * yn2[j * Nx + i]) + (h11 * fn2[j * Nx + i]);
                    }
                }
                // FFT doesn't use last endpoint so make sure to include it
                for (i = 0; i < Ny; i++) {
                    yeval_view[Nx * Ny + i] = yeval_view[i];
                }

                // 2. Interpolate in space (interpolate interpolant on desired grid)
                // interp(&yeval[i_next * nxeval * Ny], nxeval, xeval, x, yn12, Nx + 1, Ny);
                i_next++;
            }
        }

        for (i = 0; i < Nz; i++) {
            yn[i] = yn2[i] + error[i];
            fn[i] = fn2[i];
        }

        tn = tn1;
        if (estimate_wave_speed) {
            // estimate speed of traveling wave by computing average velocity at endpoints
            peak_index = argmax(yn, Nx, 1);
            if ((!first_point_found) && ((0.9 * L) <= x[peak_index]) && (x[peak_index] < (0.99 * L))) {
                first_space_point = x[peak_index];
                first_time_point = tn1;
                first_point_found = true;
            }
            else if (first_point_found && ((0.01 * L) < x[peak_index]) && (x[peak_index] <= (0.1 * L))) {
                second_space_point = x[peak_index];
                second_time_point = tn1;
                wave_speed = std::min(wave_speed, std::abs((second_space_point - first_space_point) / (second_time_point - first_time_point)));
                first_point_found = false;
            }
        }

        if (animate && (animation_thread_code == INT_MAX)) {
            fseek(animation_file, 0, SEEK_SET);
            double dNx = Nx;
            double dnode = 1;
            fwrite(&dNx, 1, sizeof(double), animation_file);
            fwrite(&dnode, 1, sizeof(double), animation_file);
            fwrite(x, 1, Nx * sizeof(double), animation_file);
            fwrite(yn, 1, Nx * sizeof(double), animation_file);
            fflush(animation_file);
        }
    }

    // actually allocate memory for these two pointers
    rwork = (double *)malloc((Nz + Ny) * sizeof(double));
    error = (double *)malloc((Nx + 1) * sizeof(double));
    // fill in with final state to be returned
    for (i = 0; i < Nx; i++) {
        error[i] = x[i];
        for (j = 0; j < Ny; j++) {
            rwork[i * Ny + j] = yn[j * Nx + i];
        }
    }
    error[i] = x[i];
    for (j = 0; j < Ny; j++) {
        rwork[i * Ny + j] = yn[j * Nx];
    }


    fftw_free(omega);
    free(work);
    if (animate) {
        animation_thread->join();
        delete animation_thread;
        fclose(animation_file);
    }

    return { error, rwork, Nx + 1, wave_speed };
}