#include "cli/colors.h"
#include "libloader.h"
#include "linalg.h"
#include "serialize.h"
#include "shared.h"
#include <csignal>
#include <filesystem>


#define MAX_FREE (1)

static volatile sig_atomic_t sigint_received;

static void handler(int)
{
    sigint_received = 1;
}

static ptrdiff_t get_lwork(const ptrdiff_t nsys)
{
    const ptrdiff_t N = nsys + MAX_FREE;
    ptrdiff_t total;
    int info[1];
    double dummy[1];

    total = N * (5 * 4 * N);
    dgeev("V", "V", nsys, NULL, nsys, NULL, NULL, NULL, nsys, NULL, nsys, dummy, -1, info);
    total += (ptrdiff_t)*dummy + nsys * nsys;

    return total;
}

struct ContinuationContext
{
    func_t m_func;
    func_t m_fjac;
    pjac_t m_pjac;
    ptrdiff_t m_nsys = 0;
    ptrdiff_t m_np = 0;
    int m_pmask[MAX_FREE];
    int m_nparam = 0;
    double *m_curr = nullptr;
    double *m_prev = nullptr;
    double *m_wr = nullptr;
    double *m_wi = nullptr;
    double *m_vl = nullptr;
    double *m_vr = nullptr;
    double *m_work = nullptr;
    double *m_p = nullptr;
    int *m_ipiv = nullptr;
    H5::H5File m_serialization_file;
    int m_serialization_index = -1;

    enum FixedPointType : int
    {
        regular = 0,
        hopf,
        saddle,
        branch
    };

    ContinuationContext(func_t func, func_t fjac, pjac_t pjac, const ptrdiff_t nsys, const double *z0, const ptrdiff_t np, const double *p) : m_func(func), m_fjac(fjac), m_pjac(pjac), m_nsys(nsys), m_nparam(0), m_curr((double *)malloc(get_lwork(nsys) * sizeof(double)))
    {
        const ptrdiff_t n = Nmax();
        m_prev = m_curr + n;
        m_wr = m_prev + n;
        m_wi = m_wr + n;
        m_vl = m_wi + n;
        m_vr = m_vl + n * n;
        m_work = m_vr + n * n;
        m_p = (double *)malloc(np * sizeof(double));
        m_np = np;
        m_ipiv = (int *)malloc(n * sizeof(int));

        memcpy(m_curr, z0, nsys * sizeof(double));
        memcpy(m_p, p, np * sizeof(double));
    }

    ~ContinuationContext()
    {
        if (m_curr) {
            free(m_curr);
            m_curr = nullptr;
        }
        if (m_ipiv) {
            free(m_ipiv);
            m_ipiv = nullptr;
        }
        if (m_p) {
            free(m_p);
            m_p = nullptr;
        }
    }

    ptrdiff_t Nmax() const
    {
        return m_nsys + MAX_FREE;
    }

    ptrdiff_t N() const
    {
        return m_nsys + m_nparam;
    }

    void setFreeParam(int index)
    {
        m_pmask[0] = index;
        m_nparam = 1;
    }

    void constructJac(const ptrdiff_t n, RP(double) A0, RP(double) M, RP(double) v) const
    {
        ptrdiff_t i;
        m_fjac(m_curr, m_p, A0);
        dlatrn(m_nsys, A0, m_nsys);
        dlacpy("A", m_nsys, m_nsys, A0, m_nsys, M, n);
        m_pjac(m_curr, m_p, m_pmask, m_nparam, &M[n * m_nsys]);
        for (i = 0; i < n; i++) {
            M[i * n + n - 1] = v[i];
        }
    }

    void generateNullspace(const ptrdiff_t n, RP(double) A0, RP(double) M, RP(double) v, const bool init_v) const
    {
        ptrdiff_t i;
        int info[1];

        if (init_v) {
            for (i = 0; i < n; i++) {
                v[i] = (double)rand() / RAND_MAX;
            }
            normalize(v, n);
        }
        constructJac(n, A0, M, v);

        memset(v, 0, (n - 1) * sizeof(double));
        v[n - 1] = 1;
        dgetrf(n, n, M, n, m_ipiv, info);
        dgetrs("N", n, 1, M, n, m_ipiv, v, n, info);
        normalize(v, n);
    }

    void spectrum()
    {
        int info[1];
        const ptrdiff_t N = Nmax();
        const ptrdiff_t lwork = get_lwork(m_nsys) - N * (5 * 4 * N);
        double *A = &m_work[N * (5 * 4 * N)];

        m_fjac(m_curr, m_p, A);
        dlatrn(m_nsys, A, m_nsys);
        dgeev("V", "V", m_nsys, A, m_nsys, m_wr, m_wi, m_vl, m_nsys, m_vr, m_nsys, A + m_nsys * m_nsys, lwork - m_nsys * m_nsys, info);
    }

    double freeParam() const
    {
        return m_p[m_pmask[0]];
    }

    bool step(TWIST::ContinuationBounds &bounds, const double tol, const bool first_step, int *niters)
    {
        const double parmin = bounds.parmin;
        const double parmax = bounds.parmax;
        double &ds = bounds.ds;
        const ptrdiff_t n = N();
        int niter, info[1];
        bool converged;
        double norm_r, norm_dx, norm0;
        double ftol, xtol;
        ptrdiff_t i;
        double *A0 = m_work;
        double *M = A0 + n * n;
        double *v = M + n * n;
        double *r = v + n;

        // generate nullspace
        generateNullspace(n, A0, M, v, first_step);

        // check if ds is going to step out of bounds
        if (const double tmp = m_p[m_pmask[0]] + ds * v[n - 1]; tmp < parmin) {
            ds = (parmin - m_p[m_pmask[0]]) / v[n - 1];
        }
        else if (tmp > parmax) {
            ds = (parmax - m_p[m_pmask[0]]) / v[n - 1];
        }

        m_curr[n - 1] = m_p[m_pmask[0]];
        memcpy(m_prev, m_curr, n * sizeof(double));
        norm0 = norm2(m_curr, n);

        if (tol <= 0) {
            ftol = std::numeric_limits<double>::epsilon() * n;
            xtol = norm0 * ftol;
        }
        else {
            ftol = tol;
            xtol = norm0 * ftol;
        }

        converged = false;

        // Newton
        for (niter = 0; niter < 100; niter++) {
            m_func(m_curr, m_p, r);
            r[n - 1] = inner(m_curr, v, n) - inner(m_prev, v, n) - ds;
            norm_r = norm2(r, n);

            constructJac(n, A0, M, v);
            dgetrf(n, n, M, n, m_ipiv, info);

            dgetrs("N", n, 1, M, n, m_ipiv, r, n, info);
            norm_dx = norm2(r, n);
            for (i = 0; i < n; i++) {
                m_curr[i] -= r[i];
            }

            // fmt::println("{} {:.8e} {:.8e}", niter, norm_r, norm_dx);
            if ((norm_r < ftol) || (norm_dx < xtol)) {
                *niters = niter + 1;
                converged = true;
                break;
            }
            // generateNullspace(n, A0, M, v, false);
            // fmt::println("{: .4e}", fmt::join(m_wr, m_wr + n - 1, " "));
            // fmt::println("{: .4e}", fmt::join(v, v + n, " "));
        }
        if (converged) {
            m_p[m_pmask[0]] = m_curr[n - 1];
            memcpy(m_prev, m_curr, n * sizeof(double));
            spectrum();
            // fmt::println("{: .8e}", fmt::join(m_wr, m_wr + m_nsys, " "));
            // fmt::println("{: .8e}", fmt::join(m_wi, m_wi + m_nsys, " "));
        }

        return converged;
    }

    void serialize(const std::string &name, const std::string &parameter_set, const std::string &parameter_name, const std::string &tag, const std::string &prefix, const FixedPointType &type)
    {
        if (m_serialization_index == -1) {
            // create file
            auto file_path = std::filesystem::path(prefix) / "local_continuation_data";
            std::filesystem::create_directories(file_path);
            file_path /= fmt::format("{}-{}-{}-{}.h5", name, parameter_set, parameter_name, tag);
            m_serialization_file = H5::H5File(file_path.c_str(), H5F_ACC_TRUNC);

            // meta data
            H5::Group group(m_serialization_file.createGroup("meta_data"));
            const std::string program_argv{ get_program_argv() };
            const std::string cwd = std::filesystem::current_path();
            ::serialize<char, 1>(group, "directory", cwd.c_str(), { cwd.size() });
            ::serialize<char, 1>(group, "cmdline", program_argv.c_str(), { program_argv.size() });

            // basic info (do I need it??)
        }
        m_serialization_index++;

        H5::Group group = m_serialization_file.createGroup(fmt::format("{}", m_serialization_index));
        ::serialize<double, 1>(group, "y", m_curr, { (hsize_t)m_nsys });
        ::serialize<double, 1>(group, "p", m_p, { (hsize_t)m_np });
        ::serialize<double, 1>(group, "wr", m_wr, { (hsize_t)m_nsys });
        ::serialize<double, 1>(group, "wi", m_wi, { (hsize_t)m_nsys });
        ::serialize<double, 2>(group, "vl", m_vl, { (hsize_t)m_nsys, (hsize_t)m_nsys });
        ::serialize<double, 2>(group, "vr", m_vr, { (hsize_t)m_nsys, (hsize_t)m_nsys });
        ::serialize<int>(group, "type", type);
    }
};

void update_ds(double &ds, const double, const double dsmax, const int niters, const bool failed)
{
    if (failed) {
        ds *= 0.5;
    }
    else if (niters < 2) {
        ds *= 2;
    }
    else if (niters < 5) {
        ds *= 1.1;
    }
    else if (niters < 7) {
    }
    else {
        ds *= 0.5;
    }
    ds = TWIST::sign(ds) * std::min(std::abs(ds), dsmax);
}

void local_continuation_loop(TWIST::ContinuationBounds &bounds, void *handle, const int pindex, const ptrdiff_t ny, const double *y, const ptrdiff_t np, const double *p, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix)
{
    int niters;
    bool converged;
    double tmp;

    func_t func = (func_t)dlsym(handle, "func");
    func_t fjac = (func_t)dlsym(handle, "fjac");
    pjac_t pjac = (pjac_t)dlsym(handle, "pjac");

    double &ds = bounds.ds;
    const double dsmin = bounds.dsmin;
    const double dsmax = bounds.dsmax;
    const double parmin = bounds.parmin;
    const double parmax = bounds.parmax;
    // auto &&[ds, dsmin, dsmax, parmin, parmax, geps, min_nodes_adapt, max_nodes_adapt, allow_mesh_adaptation] = bounds;

    ContinuationContext context(func, fjac, pjac, ny, y, np, p);
    // LocationBar bar(parmin, parmax);

    context.setFreeParam(pindex);
    signal(SIGINT, &handler);

    // refine init
    tmp = ds;
    ds = 0.0;
    converged = context.step(bounds, 0.0, true, &niters);
    if (!converged) {
        fmt::println("couldn't get initial fixed point");
        return;
    }
    ds = tmp;
    context.serialize("fhn", "default", "c", "delete", ".cache", ContinuationContext::regular);
    context.serialize(name, parameter_set, parameter, tag, prefix, ContinuationContext::regular);

    sigint_received = 0;
    while (!sigint_received) {
        if ((context.freeParam() < parmin) || (parmax < context.freeParam())) {
            fmt::println("out of bounds");
            break;
        }
        else if (std::abs(ds) < parmin) {
            fmt::println("ds too small");
            break;
        }
        converged = context.step(bounds, 0.0, false, &niters);
        update_ds(ds, dsmin, dsmax, niters, !converged);
        // fmt::println("ds = {} | {}", ds, context.freeParam());
        // test functions here

        // bar.update(context.freeParam(), fmt::format(" | {:4d} |", niters));
    }
    puts("");
    if (sigint_received) {
        fmt::println(COLOR_YELLOW "\n The continuation loop has been interrupted by ctrl+C" COLOR_RESET);
    }
    sigint_received = 0;
    signal(SIGINT, SIG_DFL);
}
