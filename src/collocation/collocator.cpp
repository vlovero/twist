#include "collocator.h"
#include "bif_points/branch_point.h"
#include "bif_points/hopf_point.h"
#include "colmat/tcolmat.h"
#include "fmt/base.h"
#include "krylov/krylov_schur.h"
#include "libloader.h"
#include "linalg.h"
#include "numpy_like.h"
#include "profiling.h"
#include "serialize.h"
#include "shared.h"
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <valarray>

// TODO: start adding in pulse features
#define TEST_PULSE 0

extern void dggev5(const char *jobvl, const char *jobvr, const int n, double *A, const int ldA, double *B, const int ldB, double *alphar, double *alphai, double *beta, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *jpvt, int *info);

static const char *__no_use_col_mat = std::getenv("NO_USE_COL_MAT");
static const char *__no_use_tree_mat = std::getenv("NO_USE_TREE_MAT");

inline bool use_tree_mat()
{
    return (__no_use_tree_mat == nullptr) || (*__no_use_tree_mat == '0');
}

inline bool use_col_mat()
{
    return (__no_use_col_mat == nullptr) || (*__no_use_col_mat == '0');
}

namespace TWIST
{
#if TEST_PULSE
    namespace InDev
    {
        class PulseCollocator : Collocator
        {
        public:
            using Collocator::Collocator;

            void g(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, RP(T) out, const RP(T) yhat, const RP(T) p, RP(T) work, const int lwork, const T eps0, const T eps1) const
            {
                TWIST_CXX_MARK_FUNCTION;

                /*
                    pulse bc + phase function + projection error
                    phase is the same

                    bc needs to be changed to:
                    Es(yhat, alpha)^T(y[0] - yhat) = 0
                    Eu(yhat, alpha)^T(y[L] - yhat) = 0

                    // compute schur with sorting
                    // call eig on upper triangular to get Vl
                    // left multiply schur vectors on Vl to get [Es | Eu]
                */

                size_t i;
                const T *y0 = Y;
                const T *yn = &Y[getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 0) - m_node];

                double *A, *wr, *wi, *Q;
                int info[1], sdim, bwork[m_node], lapack_lwork, local_lwork;
                double dummy[2];

                A = work;
                Q = work + m_node * m_node;
                wr = work + m_node * m_node;
                wi = work + m_node;
                local_lwork = 2 * m_node * (1 + m_node);
                lapack_lwork = lwork - local_lwork;

                // compute A
                m_fjac(yhat, p, A);
                dlatrn(m_node, A, m_node);

                // verify lwork is big enough
                dgees("V", "S", [](double *re, double *) -> int { return (*re) < 0.0; }, m_node, A, m_node, &sdim, wr, wi, Q, m_node, dummy, -1, bwork, info);
                dtrevc3("L", "B", NULL, m_node, A, m_node, Q, m_node, NULL, m_node, m_node, bwork, &dummy[1], -1, info);
                if (lapack_lwork < std::max<int>(dummy[0], dummy[1])) {
                    throw std::runtime_error("[PulseCollocator::g] fix LAPACK lwork");
                }

                // get left eigenspace sorted as Q = [Es | Eu]
                dgees("V", "S", [](double *re, double *) -> int { return (*re) < 0.0; }, m_node, A, m_node, &sdim, wr, wi, Q, m_node, work + local_lwork, lapack_lwork, bwork, info);
                dtrevc3("L", "B", NULL, m_node, A, m_node, Q, m_node, NULL, m_node, m_node, bwork, work + local_lwork, lapack_lwork, info);

                // sanity check to make sure sorting was done correctly
                for (i = 0; i < (size_t)sdim; i++) {
                    if (0.0 < wr[i]) {
                        throw std::logic_error(fmt::format("dgees said stable subspace had size {} but there was a positive eigenvalue", sdim));
                    }
                }
                for (; i < m_node; i++) {
                    if (wr[i] < 0) {
                        throw std::logic_error(fmt::format("dgees said unstable subspace had size {} but there was a negative eigenvalue", m_node - sdim));
                    }
                }

                // compute projection BC
                for (i = 0; i < m_node; i++) {
                    wi[i] = y0[i] - yhat[i];
                }
                dummy[0] = inner(wi, wi, m_node);
                dgemv("T", m_node, sdim, 1.0, Q, m_node, wi, 1, 0.0, out, 1);

                for (i = 0; i < m_node; i++) {
                    wi[i] = yn[i] - yhat[i];
                }
                dummy[1] = inner(wi, wi, m_node);
                dgemv("T", m_node, sdim, 1.0, &Q[m_node * sdim], m_node, wi, 1, 0.0, &out[sdim], 1);

                // compute phase condition
                out[m_node] = computePhaseCondition(Y, Yold, Ypold);
                out[m_node + 1] = dummy[0] - eps0 * eps0;
                out[m_node + 2] = dummy[1] - eps1 * eps1;

                // continuation parameter stuff
                for (i = 3; i < (size_t)m_nparam; i++) {
                    out[m_node + i] = 0.0;
                }
            }
        };

    } // namespace InDev
#endif

    Collocator::Collocator(const Collocator &collocator) : c_nstages(collocator.c_nstages), c_A(collocator.c_A), c_b(collocator.c_b), c_c(collocator.c_c), c_P(collocator.c_P), c_chat(collocator.c_chat), m_func(collocator.m_func), m_fjac(collocator.m_fjac), m_pjac(collocator.m_pjac), m_node(collocator.m_node), m_nnodes(collocator.m_nnodes), m_np(collocator.m_np), m_nparam(collocator.m_nparam), m_diffusion(collocator.m_diffusion), m_spatial_period(collocator.m_spatial_period)
    {
        for (size_t i = 0; i < max_nparam; i++) {
            m_pmask[i] = collocator.m_pmask[i];
        }
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        const size_t lwork = getLWork(m_nnodes, c_nstages, m_node);
        createCopy(m_state_curr, collocator.m_state_curr, N);
        createCopy(m_state_prev, collocator.m_state_prev, N);
        createCopy(m_state_next, collocator.m_state_next, N);
        createCopy(m_nullspacev, collocator.m_nullspacev, N);
        createCopy(m_p, collocator.m_p, m_np);
        createCopy(m_ypold, collocator.m_ypold, N);
        createCopy(m_t, collocator.m_t, m_nnodes);
        createCopy(m_h, collocator.m_h, m_nnodes - 1);
        createCopy(m_work, collocator.m_work, lwork);
    }

    Collocator &Collocator::operator=(const Collocator &other)
    {
        // I don't know any other ways of doing this...
        *((size_t *)(&c_nstages)) = other.c_nstages;
        c_A = other.c_A;
        c_b = other.c_b;
        c_c = other.c_c;
        c_P = other.c_P;
        c_chat = other.c_chat;
        m_func = other.m_func;
        m_fjac = other.m_fjac;
        m_pjac = other.m_pjac;
        m_node = other.m_node;
        m_nnodes = other.m_nnodes;
        m_np = other.m_np;
        m_nparam = other.m_nparam;
        m_diffusion = other.m_diffusion;
        m_spatial_period = other.m_spatial_period;
        // m_jac_csc = other.m_jac_csc;
        // m_jac_coo = other.m_jac_coo;
        m_serialization_index = other.m_serialization_index;
        m_serialization_file = other.m_serialization_file;

        for (size_t i = 0; i < max_nparam; i++) {
            m_pmask[i] = other.m_pmask[i];
        }
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        const size_t lwork = getLWork(m_nnodes, c_nstages, m_node);
        createCopy(m_state_curr, other.m_state_curr, N);
        createCopy(m_state_prev, other.m_state_prev, N);
        createCopy(m_state_next, other.m_state_next, N);
        createCopy(m_nullspacev, other.m_nullspacev, N);
        createCopy(m_p, other.m_p, m_np);
        createCopy(m_ypold, other.m_ypold, N);
        createCopy(m_t, other.m_t, m_nnodes);
        createCopy(m_h, other.m_h, m_nnodes - 1);
        createCopy(m_work, other.m_work, lwork);

        return *this;
    }

    Collocator::Collocator(const size_t ncol, func_t func, func_t fjac, pjac_t pjac, size_t node, size_t nnodes, const T *t, const T *y0, size_t np, const T *p, const std::vector<std::pair<int, T>> &diffusion, const T spatial_period) : c_nstages(std::get<0>(GL_methods[ncol])), c_A(std::get<1>(GL_methods[ncol])), c_b(std::get<2>(GL_methods[ncol])), c_c(std::get<3>(GL_methods[ncol])), c_P(std::get<4>(GL_methods[ncol])), c_chat(std::get<5>(GL_methods[ncol])), m_func(func), m_fjac(fjac), m_pjac(pjac), m_node(node), m_nnodes(nnodes), m_np(np), m_diffusion(diffusion), m_spatial_period(spatial_period)
    {
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        size_t k, lwork;

        // allocate data
        m_state_prev = (T *)malloc(N * sizeof(T));
        m_state_curr = (T *)malloc(N * sizeof(T));
        m_state_next = (T *)malloc(N * sizeof(T));
        m_nullspacev = (T *)malloc(N * sizeof(T));
        m_h = (T *)malloc((nnodes - 1) * sizeof(T));
        m_ypold = (T *)malloc(N * sizeof(T));
        lwork = getLWork(m_nnodes, c_nstages, m_node);
        m_work = (T *)malloc(lwork * sizeof(T));

        // init state data
        createCopy(m_p, p, m_np);
        createCopy(m_t, t, m_nnodes);
        diff(m_t, m_nnodes, m_h);
        setupLinearInterpolationAtColloationPoints(m_state_curr, y0);
        memcpy(m_state_prev, m_state_curr, N * sizeof(T));

        // init ypold
        for (k = 0; k < ((c_nstages + 1) * (m_nnodes - 1) + 1); k++) {
            m_func(&m_state_curr[k * m_node], m_p, &m_ypold[k * m_node]);
        }
        m_pmask[0] = 0;
        m_pmask[1] = -1;
        m_serialization_index = -1;
    }

    Collocator::Collocator(const std::string &h5data_path, const int solution_index) : c_nstages(std::get<0>(GL_methods[extract_nstages(h5data_path)])), c_A(std::get<1>(GL_methods[c_nstages])), c_b(std::get<2>(GL_methods[c_nstages])), c_c(std::get<3>(GL_methods[c_nstages])), c_P(std::get<4>(GL_methods[c_nstages])), c_chat(std::get<5>(GL_methods[c_nstages]))
    {
        H5::H5File file(h5data_path, H5F_ACC_RDONLY);
        size_t lwork;
        char *model_name = nullptr;
        int *diffusion_indices = nullptr;
        double *diffusion_coeffs = nullptr;
        int ndiff;

        if (file.exists("basic_info")) {
            H5::Group group = file.openGroup("basic_info");
            ::deserialize(group, "node", m_node);
            ::deserialize(group, "np", m_np);
            ::deserialize(group, "nparam", m_nparam);
            ::deserialize(group, "spatial_period", m_spatial_period);
            ::deserialize(group, "pmask", m_pmask);
            ::deserialize(group, "name", model_name);
            ::deserialize(group, "diffusion_indices", diffusion_indices);
            ::deserialize(group, "diffusion_coeffs", diffusion_coeffs);
            ::deserialize(group, "ndiff", ndiff);
            std::string lib_path{ fmt::format(".cache/models/lib/{}.so", model_name) };
            if (!std::filesystem::exists(lib_path)) {
                fmt::println("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", model_name);
                exit(1);
            }
            m_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
            for (int i = 0; i < ndiff; i++) {
                m_diffusion.emplace_back(std::pair<int, double>{ diffusion_indices[i], diffusion_coeffs[i] });
            }
            m_func = (func_t)dlsym(m_handle, "func");
            m_fjac = (func_t)dlsym(m_handle, "fjac");
            m_pjac = (pjac_t)dlsym(m_handle, "pjac");
            free(model_name);
            free(diffusion_indices);
            free(diffusion_coeffs);
        }
        else {
            throw std::runtime_error(fmt::format("basic_info is not present. {} was not created by TWIST or has been corruputed", h5data_path));
        }
        {
            H5::Group group = file.openGroup(fmt::format("{}", solution_index));
            ::deserialize(group, "nnodes", m_nnodes);
            ::deserialize(group, "p", m_p);
            ::deserialize(group, "state_curr", m_state_curr);
            ::deserialize(group, "state_prev", m_state_prev);
            ::deserialize(group, "state_next", m_state_next);
            ::deserialize(group, "nullspacev", m_nullspacev);
            ::deserialize(group, "ypold", m_ypold);
            ::deserialize(group, "t", m_t);
            ::deserialize(group, "h", m_h);
        }
        lwork = getLWork(m_nnodes, c_nstages, m_node);
        m_work = (T *)malloc(lwork * sizeof(T));
    }


    Collocator::Collocator(const H5::H5File &file, const int solution_index) : c_nstages(std::get<0>(GL_methods[extract_nstages(file)])), c_A(std::get<1>(GL_methods[c_nstages])), c_b(std::get<2>(GL_methods[c_nstages])), c_c(std::get<3>(GL_methods[c_nstages])), c_P(std::get<4>(GL_methods[c_nstages])), c_chat(std::get<5>(GL_methods[c_nstages]))
    {
        size_t lwork;
        char *model_name = nullptr;
        int *diffusion_indices = nullptr;
        double *diffusion_coeffs = nullptr;
        int ndiff;

        if (file.exists("basic_info")) {
            H5::Group group = file.openGroup("basic_info");
            ::deserialize(group, "node", m_node);
            ::deserialize(group, "np", m_np);
            ::deserialize(group, "nparam", m_nparam);
            ::deserialize(group, "spatial_period", m_spatial_period);
            ::deserialize(group, "pmask", m_pmask);
            ::deserialize(group, "name", model_name);
            ::deserialize(group, "diffusion_indices", diffusion_indices);
            ::deserialize(group, "diffusion_coeffs", diffusion_coeffs);
            ::deserialize(group, "ndiff", ndiff);
            std::string lib_path{ fmt::format(".cache/models/lib/{}.so", model_name) };
            if (!std::filesystem::exists(lib_path)) {
                fmt::println("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", model_name);
                exit(1);
            }
            m_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
            for (int i = 0; i < ndiff; i++) {
                m_diffusion.emplace_back(std::pair<int, double>{ diffusion_indices[i], diffusion_coeffs[i] });
            }
            m_func = (func_t)dlsym(m_handle, "func");
            m_fjac = (func_t)dlsym(m_handle, "fjac");
            m_pjac = (pjac_t)dlsym(m_handle, "pjac");
            free(model_name);
            free(diffusion_indices);
            free(diffusion_coeffs);
        }
        else {
            throw std::runtime_error(fmt::format("basic_info is not present. {} was not created by TWIST or has been corruputed", file.getFileName()));
        }
        {
            H5::Group group = file.openGroup(fmt::format("{}", solution_index));
            ::deserialize(group, "nnodes", m_nnodes);
            ::deserialize(group, "p", m_p);
            ::deserialize(group, "state_curr", m_state_curr);
            ::deserialize(group, "state_prev", m_state_prev);
            ::deserialize(group, "state_next", m_state_next);
            ::deserialize(group, "nullspacev", m_nullspacev);
            ::deserialize(group, "ypold", m_ypold);
            ::deserialize(group, "t", m_t);
            ::deserialize(group, "h", m_h);
        }
        lwork = getLWork(m_nnodes, c_nstages, m_node);
        m_work = (T *)malloc(lwork * sizeof(T));
    }

    Collocator::~Collocator()
    {
        if (m_state_curr) {
            free(m_state_curr);
            m_state_curr = nullptr;
        }
        if (m_state_prev) {
            free(m_state_prev);
            m_state_prev = nullptr;
        }
        if (m_state_next) {
            free(m_state_next);
            m_state_next = nullptr;
        }
        if (m_nullspacev) {
            free(m_nullspacev);
            m_nullspacev = nullptr;
        }
        if (m_p) {
            free(m_p);
            m_p = nullptr;
        }
        if (m_ypold) {
            free(m_ypold);
            m_ypold = nullptr;
        }
        if (m_t) {
            free(m_t);
            m_t = nullptr;
        }
        if (m_h) {
            free(m_h);
            m_h = nullptr;
        }
        if (m_work) {
            free(m_work);
            m_work = nullptr;
        }
        if (m_handle) {
            dlclose(m_handle);
            m_handle = nullptr;
        }
    }

    // helper routine to generate a vector of plottable points
    std::vector<T> Collocator::generatePlottablePoints(const RP(T) tref, size_t nt, const RP(T) href) const
    {
        std::vector<T> t;
        t.reserve(nt * (c_nstages + 1) + 1);

        for (size_t k = 0; k < (nt - 1); k++) {
            T tk = tref[k];
            T hk = href[k];
            t.push_back(tk);
            for (size_t i = 0; i < c_nstages; i++) {
                t.push_back(tk + c_c[i] * hk);
            }
        }
        t.push_back(tref[nt - 1]);
        return t;
    }

    std::vector<T> Collocator::plottablePoints() const
    {
        return generatePlottablePoints(m_t, m_nnodes, m_h);
    }

    // static functions to me moved to their own file
    // based off provided domain points, evaluate interval sizes.
    size_t Collocator::getLWork(size_t nnodes, size_t nstages, size_t node)
    {
        size_t lwork = 0;
        size_t N = getNumberOfUnknowns(nnodes, nstages, node, max_nparam);
        lwork += N; // main/param block doesn't need this much but the code is cleaner later
        lwork += N; // rhs
        lwork += N; // rhs1
        lwork += N; // yp
        lwork += N; // ypnext
        lwork += N; // delta
        lwork += N; // deltab
        lwork += N; // dV
        lwork += N; // Rk
        lwork += N; // vold
        lwork += N; // phase condition (colmat)
        lwork += N; // nullspace (colmat)

        return lwork;
    }

    void Collocator::diff(const T *x, const size_t nx, T *dx)
    {
        for (size_t i = 0; i < (nx - 1); i++) {
            dx[i] = x[i + 1] - x[i];
        }
    }

    size_t Collocator::jacobianMainBlockNNZ(size_t node, size_t nstages)
    {
        return (node * nstages) * ((nstages + 1) * node) + node * (nstages + 2);
        size_t m = node * (nstages + 1);
        return (m + node) * m;
    }

    size_t Collocator::jacobianParamBlockNNZ(size_t node, size_t nstages, size_t nparam)
    {
        return (node * (nstages + 1)) * nparam;
    }

    size_t Collocator::jacobianNNZ(size_t nnodes, size_t node, size_t nstages, size_t nparam)
    {
        size_t n = nnodes - 1;
        size_t dim = getNumberOfUnknowns(nnodes, nstages, node, nparam);
        size_t nnz = n * (jacobianMainBlockNNZ(node, nstages) + jacobianParamBlockNNZ(node, nstages, nparam));
        nnz += 2 * node;
        if (nparam > 0) {
            nnz += n * node * nstages;
            nnz += dim * (nparam - 1);
        }
        return nnz;
    }

    // helper routine to create a copy of an array
    template <typename data_t>
    void Collocator::createCopy(data_t *&dst, const data_t *src, size_t n)
    {
        if (dst != nullptr) {
            free(dst);
        }
        dst = (data_t *)malloc(n * sizeof(data_t));
        assert(dst);

        memcpy(dst, src, n * sizeof(data_t));
    }

    void Collocator::scaleVector(T scale, RP(T) x, size_t nx)
    {
        for (size_t i = 0; i < nx; i++) {
            x[i] *= scale;
        }
    }

    size_t Collocator::getNumberOfUnknowns(size_t nnodes, size_t nstages, size_t node, size_t nparam)
    {
        return (nnodes - 1) * node * (nstages + 1) + node + nparam;
    }

    size_t Collocator::getMainJBlockNNZ(size_t, size_t s, size_t m)
    {
        return (m * s) * ((s + 1) * m) + m * (s + 2);
    }

    std::tuple<size_t, size_t, size_t> Collocator::getJacobianNNZ(size_t, size_t, size_t, size_t)
    {
        return { 0, 0, 0 };
    }

    size_t Collocator::searchsorted(const T *arr, size_t size, T key)
    {
        TWIST_CXX_MARK_FUNCTION;

        size_t low = 0;
        size_t high = size;

        while (low < high) {
            size_t mid = low + (high - low) / 2;

            if (arr[mid] <= key) {
                low = mid + 1; // Search in the upper half
            }
            else {
                high = mid; // Search in the lower half
            }
        }
        return low;
    }

    double Collocator::solutionNorm(const bool include_velocities) const
    {
        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;

        /*
            Y has shape (n * (s + 1) + 1, m)
            need to view first n * (s + 1) rows as "3d" with shape
            (n, s + 1, m) then indexing will be
            [k, i, j] -> ((m * (s + 1)) * k) + ((m) * i) + (j)
        */

        T solution_norm, sum1, sum2, val;
        size_t i, j, k, stride_k, stride_i, l;
        const auto &to_delete = indicesToDelete();

        solution_norm = 0.0;
        for (k = 0; k < n; k++) {
            sum1 = 0.0;
            stride_k = m * (s + 1) * k;

            for (i = 0; i < s; i++) {
                sum2 = 0.0;
                // + 1 to skip node and jump to collocation point
                stride_i = m * (i + 1);
                for (j = 0; j < m; j++) {
                    if ((!include_velocities) && (std::find(to_delete.begin(), to_delete.end(), j) != to_delete.end())) {
                        continue;
                    }
                    l = stride_k + stride_i + j;
                    val = m_state_curr[l];
                    sum2 += val * val;
                }
                sum1 += c_b[i] * sum2;
            }
            solution_norm += m_h[k] * sum1;
        }

        return std::sqrt(solution_norm * spatialPeriod());
    }

    T Collocator::computePhaseCondition(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold) const
    {
        TWIST_CXX_MARK_FUNCTION;

        // I know this isn't the best code practice
        // but it matches the math notation from my
        // original notes...
        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;

        /*
            Y has shape (n * (s + 1) + 1, m)
            need to view first n * (s + 1) rows as "3d" with shape
            (n, s + 1, m) then indexing will be
            [k, i, j] -> ((m * (s + 1)) * k) + ((m) * i) + (j)
        */

        T phase_condition, sum1, sum2, val;
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
                sum1 += c_b[i] * sum2;
            }
            phase_condition += m_h[k] * sum1;
        }

        return phase_condition;
    }

    void Collocator::g(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, RP(T) out) const
    {
        TWIST_CXX_MARK_FUNCTION;

        size_t i;
        const T *y0 = Y;
        const T *yn = &Y[getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 0) - m_node];

        for (i = 0; i < m_node; i++) {
            out[i] = y0[i] - yn[i];
        }

        out[m_node] = computePhaseCondition(Y, Yold, Ypold);
        for (int i = 1; i < m_nparam; i++) {
            out[m_node + i] = 0;
        }
    }

    void Collocator::evaluateResidual(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) residual, RP(T) yp) const
    {
        TWIST_CXX_MARK_FUNCTION;

        const size_t n = m_nnodes - 1;
        const size_t m = m_node;
        const size_t s = c_nstages;

        const T scale = 2 * m_spatial_period;

        T hk, fac;

        size_t i, j, k, stride_k, stride_i, l, stride_k1;

        m_func(Y, p, yp);
        scaleVector(scale, yp, m);

        for (k = 0; k < n; k++) {
            hk = m_h[k];
            stride_k = m * (s + 1) * k;
            stride_k1 = m * (s + 1) * (k + 1);

            const T *yk = &Y[stride_k];
            const T *yk1 = &Y[stride_k1];
            T *Fk = &residual[stride_k];

            // Fk = yk - yk1 + hk Sum_i(b_i f(y^k_i))
            for (l = 0; l < m; l++) {
                Fk[l] = yk[l] - yk1[l];
            }

            for (i = 0; i < s; i++) {
                stride_i = (i + 1) * m;
                // stage value
                const T *yk_i = &Y[stride_k + stride_i];
                // stage/derivative
                T *Sk_i = &yp[stride_k + stride_i];

                // evaluate S^k_i
                m_func(yk_i, p, Sk_i);
                scaleVector(scale, Sk_i, m);

                // add to Fk
                fac = hk * c_b[i];
                for (l = 0; l < m; l++) {
                    Fk[l] += fac * Sk_i[l];
                }
            }
            // evaluate f(y_{k + 1}) for yp vector
            m_func(yk1, p, &yp[stride_k1]);
            scaleVector(scale, &yp[stride_k1], m);

            // H^k_i = yk - yk_i + hk Sum_j(a_ij S^k_j)
            for (i = 0; i < s; i++) {
                stride_i = (i + 1) * m;

                const T *yki = &Y[stride_k + stride_i];
                T *Hk_i = &residual[stride_k + stride_i];

                // set data to yk - yk_i
                for (l = 0; l < m; l++) {
                    Hk_i[l] = (yk[l] - yki[l]);
                }

                // for each j add on 2T hk a_ij S^k_j)
                for (j = 0; j < s; j++) {
                    T *Sk_j = &yp[stride_k + (j + 1) * m];
                    // yp has already been scaled
                    fac = hk * c_A[i * s + j];

                    for (l = 0; l < m; l++) {
                        // stages should be stored in yp now
                        Hk_i[l] += fac * Sk_j[l];
                    }
                }
            }
        }

        g(Y, Yold, Ypold, &residual[n * m * (s + 1)]);
    }

    void Collocator::setupLinearInterpolationAtColloationPoints(T *state, const T *y0) const
    {
        size_t i, j, k, shift;
        T slope, z;
        for (k = 0; k < (m_nnodes - 1); k++) {
            for (j = 0; j < m_node; j++) {
                state[k * (c_nstages + 1) * m_node + j] = y0[k * m_node + j];
            }
            for (i = 1; i < (c_nstages + 1); i++) {
                z = c_c[i - 1] * m_h[k];
                // z = m_h[k] * (c_c[i - 1] - 1);
                shift = (k * (c_nstages + 1) * m_node) + (i * m_node);
                for (j = 0; j < m_node; j++) {
                    slope = (y0[(k + 1) * m_node + j] - y0[k * m_node + j]) / m_h[k];
                    state[shift + j] = y0[k * m_node + j] + slope * z;
                    // state[shift + j] = y0[(k + 1) * m_node + j] + slope * z;
                }
            }
        }
        for (j = 0; j < m_node; j++) {
            state[k * (c_nstages + 1) * m_node + j] = y0[k * m_node + j];
        }
    }

    void Collocator::generateMainBlock(RP(T) computed_jacs, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, int64_t rowoffset, int64_t coloffset, T hk, const RP(T) yk, const RP(T) p) const
    {
        const int64_t s = c_nstages;
        const int64_t m = m_node;

        const int64_t skip = m * m;

        int64_t i, j, k, l;

        T *jac_tmp, delta, fac, eta;

        const T scale = 2 * m_spatial_period;

        // dFk/dyk = I
        for (i = 0; i < m; i++) {
            *irows++ = rowoffset + i;
            *icols++ = coloffset + i;
            *data++ = 1.0;
        }

        // dFk/dyk,i = 2 T hk bi J(y_k,i)
        for (i = 0; i < s; i++) {
            jac_tmp = &computed_jacs[i * skip];
            m_fjac(&yk[i * m], p, jac_tmp);
            fac = hk * c_b[i] * scale;
            for (j = 0; j < m; j++) {
                for (k = 0; k < m; k++) {
                    *data++ = fac * jac_tmp[j * m + k];
                    if constexpr (true) {
                        // row major jacobians
                        *irows++ = rowoffset + j;
                        *icols++ = coloffset + k + (i + 1) * m;
                    }
                    else {
                        // col major jacobians
                        *irows++ = rowoffset + k;
                        *icols++ = coloffset + j + (i + 1) * m;
                    }
                }
            }
        }

        // dFk/dyk+1 = -I
        for (i = 0; i < m; i++) {
            *irows++ = rowoffset + i;
            *icols++ = coloffset + i + (s + 1) * m;
            *data++ = -1.0;
        }

        for (i = 0; i < s; i++) {
            // dHki/dyk = I
            for (j = 0; j < m; j++) {
                *irows++ = rowoffset + j + (i + 1) * m;
                *icols++ = coloffset + j;
                *data++ = 1.0;
            }

            for (j = 0; j < s; j++) {
                jac_tmp = &computed_jacs[j * skip];
                delta = (i == j) ? -1.0 : 0.0;
                fac = hk * c_A[i * s + j] * scale;
                // dHki/dyk,j = 2T hk aij J(y_k,j) - delta_ij I
                for (k = 0; k < m; k++) {
                    for (l = 0; l < m; l++) {
                        eta = (k == l) ? 1.0 : 0.0;
                        *data++ = fac * jac_tmp[k * m + l] + delta * eta;
                        if constexpr (true) {
                            // row major
                            *irows++ = rowoffset + k + (i + 1) * m;
                            *icols++ = coloffset + l + (j + 1) * m;
                        }
                        else {
                            // row major
                            *irows++ = rowoffset + l + (i + 1) * m;
                            *icols++ = coloffset + k + (j + 1) * m;
                        }
                    }
                }
            }
        }
    }

    void Collocator::generateParamBlock(RP(T) computed_jacs, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, int64_t rowoffset, int64_t coloffset, T hk, const RP(T) yk, const RP(T) p) const
    {
        const int64_t m = m_node;
        const int64_t s = c_nstages;
        const int64_t nnz = m * (s + 1) * m_nparam;
        const int64_t skip = m * m_nparam;

        int64_t i, j, k, l;
        T *jac_tmp, *block;
        T fac;

        const T scale = 2 * m_spatial_period;

        memset(data, 0, nnz * sizeof(T));

        // dFk/dc = 2T hk sum(c_b[i] fc(yk[i]))

        for (i = 0; i < s; i++) {
            jac_tmp = &computed_jacs[i * skip];
            m_pjac(&yk[i * m], const_cast<T *>(p), m_pmask, m_nparam, jac_tmp);
            fac = hk * c_b[i] * scale;
            for (j = 0; j < m; j++) {
                for (k = 0; k < m_nparam; k++) {
                    // data[j * m_nparam + k] += fac * jac_tmp[k * m + j];
                    data[j * m_nparam + k] += fac * jac_tmp[j * m_nparam + k];
                }
            }
        }
        // update indexing for dFk/dc
        for (j = 0; j < m; j++) {
            for (k = 0; k < m_nparam; k++) {
                *irows++ = rowoffset + j;
                *icols++ = coloffset + k;
            }
        }

        for (i = 0; i < s; i++) {
            block = &data[(i + 1) * skip];
            // dHki/dc = 2T hk sum(a_ij fc(yk[j]))

            for (j = 0; j < s; j++) {
                jac_tmp = &computed_jacs[j * skip];
                fac = hk * c_A[i * s + j] * scale;
                for (k = 0; k < m; k++) {
                    for (l = 0; l < m_nparam; l++) {
                        // block[k * m_nparam + l] += fac * jac_tmp[l * m + k]; // k * m_nparam + l];
                        block[k * m_nparam + l] += fac * jac_tmp[k * m_nparam + l];
                    }
                }
            }
            // update indexing for dHki/dc
            for (j = 0; j < m; j++) {
                for (k = 0; k < m_nparam; k++) {
                    *irows++ = rowoffset + j + (i + 1) * m;
                    *icols++ = coloffset + k;
                }
            }
        }
    }

    void Collocator::gjac(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, RP(T) work, const RP(T) v) const
    {
        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        T tmp;
        size_t i, j, k, l, stride_k, stride_i;

        // get rid of compiler warnings
        // leave in for when projection bc are added
        (void)Y;
        (void)Yold;
        (void)p;
        (void)work;

        // boundary conditions
        for (i = 0; i < m_node; i++) {
            *irows++ = N - m_nparam - m_node + i;
            *icols++ = i;
            *data++ = +1.0;
        }

        for (i = 0; i < m_node; i++) {
            *irows++ = N - m_nparam - m_node + i;
            *icols++ = N - m_nparam - m_node + i;
            *data++ = -1.0;
        }
        if (m_nparam == 0) {
            return;
        }
        // last row phase condition
        for (k = 0; k < n; k++) {
            stride_k = m * (s + 1) * k;

            for (i = 0; i < s; i++) {
                // + 1 to skip node and jump to collocation point
                stride_i = m * (i + 1);
                for (j = 0; j < m; j++) {
                    l = stride_k + stride_i + j;
                    tmp = m_h[k] * Ypold[l] * c_b[i];
                    *data++ = tmp;
                    *irows++ = N - m_nparam;
                    *icols++ = l;
                }
            }
        }

        if (m_nparam == 2) {
            // V vector stuff goes here
            for (i = 0; i < N; i++) {
                *data++ = v[i];
                *irows++ = N - 1;
                *icols++ = i;
            }
        }
    }

    void Collocator::generateJacobianColmat(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v)
    {
        TWIST_CXX_MARK_FUNCTION;

        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        // puts(__PRETTY_FUNCTION__);

        T tmp;
        size_t i, j, k, l, stride_k, stride_i;
        double *V;

        // get rid of compiler warnings
        (void)Yold;
        (void)work;

        V = &m_work[10 * N];

        // clear memory regardless of access later
        memset(V, 0, 2 * N * sizeof(double));

        // make bc blocks
        m_bc.makePeriodic(m_node, m_nparam);

        if (1 <= m_nparam) {
            // last row phase condition
            for (k = 0; k < n; k++) {
                stride_k = m * (s + 1) * k;
                for (i = 0; i < s; i++) {
                    // + 1 to skip node and jump to collocation point
                    stride_i = m * (i + 1);
                    for (j = 0; j < m; j++) {
                        l = stride_k + stride_i + j;
                        tmp = m_h[k] * Ypold[l] * c_b[i];
                        V[m_nparam * l] = tmp;
                    }
                }
            }
        }
        if (m_nparam == 2) {
            // fill in nullspace if in continuation mode
            for (i = 0; i < N; i++) {
                V[m_nparam * i + 1] = v[i];
            }
        }

        // m_colmat.~CollocationMatrix();
        m_colmat.update(m_fjac, m_pjac, m_nnodes, m_node, c_nstages, m_nparam, unscaledSpatialPeriod(), m_h, Y, p, m_pmask, V, &m_bc);
    }

    void Collocator::generateJacobianColmat(DEV::Matrix *colmat, const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v)
    {
        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        // puts(__PRETTY_FUNCTION__);

        T tmp;
        size_t i, j, k, l, stride_k, stride_i;
        double *V;

        // ger rid of compiler warnings
        (void)Yold;
        (void)work;

        V = &m_work[10 * N];

        // clear memory regardless of access later
        memset(V, 0, 2 * N * sizeof(double));

        // make bc blocks
        m_bc.makePeriodic(m_node, m_nparam);

        if (1 <= m_nparam) {
            // last row phase condition
            for (k = 0; k < n; k++) {
                stride_k = m * (s + 1) * k;
                for (i = 0; i < s; i++) {
                    // + 1 to skip node and jump to collocation point
                    stride_i = m * (i + 1);
                    for (j = 0; j < m; j++) {
                        l = stride_k + stride_i + j;
                        tmp = m_h[k] * Ypold[l] * c_b[i];
                        V[m_nparam * l] = tmp;
                    }
                }
            }
        }
        if (m_nparam == 2) {
            // fill in nullspace if in continuation mode
            for (i = 0; i < N; i++) {
                V[m_nparam * i + 1] = v[i];
            }
        }

        // m_colmat.~CollocationMatrix();
        colmat->update(m_fjac, m_pjac, m_nnodes, m_node, c_nstages, m_nparam, unscaledSpatialPeriod(), m_h, Y, p, m_pmask, V, &m_bc);
    }

    void Collocator::generateJacobian(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v)
    {
        const size_t n = m_nnodes - 1;
        const size_t s = c_nstages;
        const size_t m = m_node;
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);
        const size_t nnz_pblock = jacobianParamBlockNNZ(m_node, c_nstages, m_nparam);
        const size_t nnz_block = jacobianMainBlockNNZ(m_node, c_nstages);
        const size_t nnz_jac = jacobianNNZ(m_nnodes, m_node, c_nstages, m_nparam);

        size_t k, rowoffset, coloffset, r, stride_k, shift;
        int64_t *irows, *icols;
        T *data;
        m_jac_coo.setNNZ(nnz_jac);
        m_jac_coo.reshape(N, N);
        irows = m_jac_coo.irow;
        icols = m_jac_coo.icol;
        data = m_jac_coo.data;

        r = m * (s + 1);

        // BEGIN NEW parallelizable construction (old method)
        // #pragma omp parallel for private(k, shift, stride_k, coloffset, rowoffset, __work) num_threads(2)
        for (k = 0; k < n; k++) {
            shift = k * (nnz_block + nnz_pblock);
            stride_k = r * k;
            // coloffset = rowoffset;

            const T *yk = &Y[stride_k + m];
            rowoffset = k * r;
            coloffset = rowoffset;
            // + m is becuase i only need stage values not node
            generateMainBlock(work, &irows[shift], &icols[shift], &data[shift], rowoffset, coloffset, m_h[k], yk, p);
            shift += nnz_block;
            coloffset = N - m_nparam;
            generateParamBlock(work, &irows[shift], &icols[shift], &data[shift], rowoffset, coloffset, m_h[k], yk, p);
        }
        shift = n * (nnz_block + nnz_pblock);
        // END NEW parallelizable construction (old method)

        gjac(Y, Yold, Ypold, p, irows + shift, icols + shift, data + shift, work, v);

        // TIME_CODE("updating csc",
        m_jac_csc.updateFromCOO(m_jac_coo);
        // );
        m_jac_csc.Control[UMFPACK_BLOCK_SIZE] = m_node * (c_nstages + 1);

        if (false || use_col_mat()) {
            m_collocation_matrix.update(&m_jac_csc, m_nnodes, c_nstages, m_node, m_nparam);
        }
    }

    T Collocator::predict(T ds, const T parmin, const T parmax)
    {
        size_t i;
        int k;
        T tmp;
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        assert(m_nparam == 2);

        k = m_pmask[1];
        tmp = m_p[k] + ds * m_nullspacev[N - 1];
        if (tmp < parmin) {
            ds = TWIST::sign(ds) * abs((parmin - m_p[k]) / m_nullspacev[N - 1]);
        }
        else if (parmax < tmp) {
            ds = TWIST::sign(ds) * abs((parmax - m_p[k]) / m_nullspacev[N - 1]);
        }

        for (i = 0; i < N; i++) {
            m_state_curr[i] += ds * m_nullspacev[i];
        }
        // fmt::println("dk = {: .8e}", ds * m_nullspacev[N - 1]);
        for (i = 0; i < (size_t)m_nparam; i++) {
            m_p[m_pmask[i]] = m_state_curr[N - m_nparam + i];
        }
        return ds;
    }

    void Collocator::computeMonitorFunctionData(RP(T) F, RP(T) v, T &theta, T &err_exp) const
    {
        size_t i, k;
        T tmp[m_node];
        T vk;
        const T *Sk;
        const size_t n = m_nnodes - 1;
        const T fac = tgamma(c_nstages + 2);
        err_exp = 1.0 / (c_nstages + 1);

        F[0] = 0.0;
        for (k = 0; k < n; k++) {
            Sk = &m_ypold[m_node * (c_nstages + 1) * k + m_node];

            dgemv("N", m_node, c_nstages, 1.0, Sk, m_node, &c_P[c_nstages - 1], c_nstages, 0.0, tmp, 1);
            vk = m_h[k] * fac * norm2(tmp, m_node);
            vk = std::pow(vk, err_exp);
            v[k] = vk;
            F[k + 1] = F[k] + vk;
        }
        theta = F[n];
        for (i = 0; i < m_nnodes; i++) {
            F[i] /= theta;
        }

#if ENABLE_SHOW_MONITOR_FUNCTION
        if (static const char *env = std::getenv("SHOW_MONITOR_FUNCTION"); env && (*env != '0')) {
            {
                H5::H5File res_file(".cache/solve_res.h5", H5F_ACC_TRUNC);
                H5::Group group = res_file.createGroup("data");
                ::serialize<double, 1>(group, "x", m_t, { m_nnodes });
                ::serialize<double, 2>(group, "y", F, { m_nnodes, 1 });
            }
#error "update with correct call to run_python_script;
            int code = run_python_script;
            if (code) {
                exit(0);
            }
        }
#endif
    }

    T *Collocator::monitorFunctionInfo(const T eps, const size_t min_nnodes, const size_t max_nnodes, size_t &nnodes_opt, const bool small_change) const
    {
        T *F, *v, *t_opt;
        T theta, err_exp, r1, r2, tmp;
        size_t i;

        const T leading_error_scale = c_chat[0];

        F = m_work;
        v = m_work + m_nnodes + 1;
        computeMonitorFunctionData(F, v, theta, err_exp);
        nnodes_opt = theta * std::pow(leading_error_scale / eps, err_exp) + 1;

        r1 = 0.0;
        r2 = 0.0;
        for (i = 0; i < (m_nnodes - 1); i++) {
            tmp = m_h[i] * std::pow((leading_error_scale / eps) * v[i], err_exp);
            r1 = std::max(r1, tmp);
            r2 += tmp;
        }
        r2 = ((int)r2) + 1;
        nnodes_opt = std::min(nnodes_opt, std::max(nnodes_opt, (size_t)r2) / 2) + 1;
        nnodes_opt = std::ceil(nnodes_opt * 1.15);
        if (small_change) {
            nnodes_opt = std::min(nnodes_opt, (size_t)(1.2 * m_nnodes));
            nnodes_opt = std::max(nnodes_opt, (size_t)(0.8 * m_nnodes));
        }
        nnodes_opt = std::max(min_nnodes, nnodes_opt);
        nnodes_opt = std::min(max_nnodes, nnodes_opt);
        // tree solver needs at least three nodes or the program will crash
        nnodes_opt = std::max(3lu, nnodes_opt);

        t_opt = (T *)malloc(nnodes_opt * sizeof(T));
        for (i = 0; i < nnodes_opt; i++) {
            t_opt[i] = (T)i / (nnodes_opt - 1);
        }
        interp(t_opt, nnodes_opt, t_opt, F, m_t, m_nnodes, 1);

        return t_opt;
    }

    void Collocator::adaptGrid(const T eps, const size_t min_nnodes, const size_t max_nnodes, const bool small_change)
    {
        TWIST_CXX_MARK_FUNCTION;

        size_t nnodes_new, k;
        T *h_new;
        T *t_new;
        T *state_curr_new;
        T *state_prev_new;
        T *state_next_new;
        T *nullspacev_new;
        T *ypold_new;
        T *work_new;
        t_new = monitorFunctionInfo(eps, min_nnodes, max_nnodes, nnodes_new, small_change);
        const size_t N_new = getNumberOfUnknowns(nnodes_new, c_nstages, m_node, max_nparam);
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        const size_t lwork_new = getLWork(nnodes_new, c_nstages, m_node);

        h_new = (T *)malloc((nnodes_new - 1) * sizeof(T));
        diff(t_new, nnodes_new, h_new);

        TWIST_MARK_BEGIN("allocations");
        auto tc_new = generatePlottablePoints(t_new, nnodes_new, h_new);
        auto tc = plottablePoints();

        // allocate updated mesh
        state_curr_new = (T *)malloc(N_new * sizeof(T));
        state_prev_new = (T *)malloc(N_new * sizeof(T));
        state_next_new = (T *)malloc(N_new * sizeof(T));
        nullspacev_new = (T *)malloc(N_new * sizeof(T));
        ypold_new = (T *)malloc(N_new * sizeof(T));
        work_new = (T *)malloc(lwork_new * sizeof(T));

        TWIST_MARK_END("allocations");

        // get solution on new mesh
        denseOutput(tc_new.data(), tc_new.size(), state_curr_new);
        memcpy(state_prev_new, state_curr_new, N_new * sizeof(T));

        TWIST_MARK_BEGIN("new y prime");
        for (k = 0; k < ((c_nstages + 1) * (nnodes_new - 1) + 1); k++) {
            m_func(&state_curr_new[k * m_node], m_p, &ypold_new[k * m_node]);
            // scaleVector(2 * m_spatial_period, &ypold_new[k * m_node], m_node);
        }
        TWIST_MARK_END("new y prime");

        // interpolate nullspace on new mesh
        TWIST_MARK_BEGIN("interpolate");
        interp(nullspacev_new, tc_new.size(), tc_new.data(), tc.data(), m_nullspacev, tc.size(), m_node);
        for (k = 0; k < max_nparam; k++) {
            nullspacev_new[N_new - max_nparam + k] = m_nullspacev[N - max_nparam + k];
        }
        TWIST_MARK_END("interpolate");

        free(m_h);
        m_h = h_new;
        free(m_t);
        m_t = t_new;
        free(m_state_curr);
        m_state_curr = state_curr_new;
        free(m_state_prev);
        m_state_prev = state_prev_new;
        free(m_state_next);
        m_state_next = state_next_new;
        free(m_nullspacev);
        m_nullspacev = nullspacev_new;
        free(m_ypold);
        m_ypold = ypold_new;
        free(m_work);
        m_work = work_new;

        m_nnodes = nnodes_new;
    }

    void Collocator::fillBBlockGGEV(RP(T) block, const size_t ldblock, const RP(T) M, const T hk) const
    {
        size_t i, j, k, l, kk, ll;

        for (i = 0; i < c_nstages; i++) {
            for (j = 0; j < m_node; j++) {
                for (k = (i + 1) * m_node, kk = 0; k < ((i + 2) * m_node); k++, kk++) {
                    block[j * ldblock + k] = hk * c_b[i] * M[j * m_node + kk];
                }
            }
        }
        for (i = 0; i < c_nstages; i++) {
            for (j = 0; j < c_nstages; j++) {
                for (k = (i + 1) * m_node, kk = 0; k < ((i + 2) * m_node); k++, kk++) {
                    for (l = (j + 1) * m_node, ll = 0; l < ((j + 2) * m_node); l++, ll++) {
                        block[k * ldblock + l] = hk * c_A[i * c_nstages + j] * M[kk * m_node + ll];
                    }
                }
            }
        }
    }

    std::vector<int> Collocator::indicesToDelete() const
    {
        int offset;
        std::vector<int> to_delete;
        to_delete.reserve(m_diffusion.size());

        offset = 1;
        for (const auto &[k, _] : m_diffusion) {
            to_delete.push_back(k + offset);
            offset++;
        }
        return to_delete;
    }

    void Collocator::fillSparseBBlock(const RP(T) M, const T hk, std::vector<T> &B_data, std::vector<int64_t> &B_rows, std::vector<int64_t> &B_cols, const size_t row_offset) const
    {
        size_t i, j, k, l, kk, ll;
        T val;

        for (i = 0; i < c_nstages; i++) {
            for (j = 0; j < m_node; j++) {
                for (k = (i + 1) * m_node, kk = 0; k < ((i + 2) * m_node); k++, kk++) {
                    val = hk * c_b[i] * M[j * m_node + kk];
                    if (val) {
                        B_data.emplace_back(val);
                        B_rows.emplace_back(row_offset + j);
                        B_cols.emplace_back(row_offset + k);
                    }
                    // block[j * ldblock + k] = val;
                }
            }
        }
        for (i = 0; i < c_nstages; i++) {
            for (j = 0; j < c_nstages; j++) {
                for (k = (i + 1) * m_node, kk = 0; k < ((i + 2) * m_node); k++, kk++) {
                    for (l = (j + 1) * m_node, ll = 0; l < ((j + 2) * m_node); l++, ll++) {
                        val = hk * c_A[i * c_nstages + j] * M[kk * m_node + ll];
                        if (val) {
                            B_data.emplace_back(val);
                            B_rows.emplace_back(row_offset + k);
                            B_cols.emplace_back(row_offset + l);
                        }
                        // block[k * ldblock + l] = val;
                    }
                }
            }
        }
    }

    void Collocator::setupSparseABPencilForSpectrum(sparse::RealCSCMatrix &AA, sparse::RealCSCMatrix &B, const T sigma)
    {
        sparse::RealCSCMatrix A, BB;
        size_t row_offset, r, k, j, i;
        T M[m_node * m_node];
        T D[m_node];
        auto to_delete = indicesToDelete();
        std::vector<T> B_data;
        std::vector<int64_t> B_rows, B_cols;

        m_nparam = 0;

        generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        A.updateFromCOO(m_jac_coo);
        B_data.reserve(m_jac_coo.nnz);
        B_rows.reserve(m_jac_coo.nnz);
        B_cols.reserve(m_jac_coo.nnz);

        memset(M, 0, m_node * m_node * sizeof(T));
        memset(D, 0, m_node * sizeof(T));

        for (const auto &[k, dk] : m_diffusion) {
            D[k] = dk;
        }

        // setup little M for filling in blocks
        k = 0;
        for (i = 0; i < m_node; i++) {
            if (std::find(to_delete.begin(), to_delete.end(), i + 1) != to_delete.end()) {
                continue;
            }
            if (std::find(to_delete.begin(), to_delete.end(), i) != to_delete.end()) {
                j = to_delete[k];
                k++;
                M[j * m_node + (j - 1)] = -((2 * m_spatial_period) * m_p[m_np - 1]) / D[j - k];
                continue;
            }
            M[i * m_node + i] = ((2 * m_spatial_period) * m_p[m_np - 1]) / m_p[0];
        }

        row_offset = 0;
        r = m_node * (c_nstages + 1);
        for (k = 0; k < (m_nnodes - 1); k++) {
            fillSparseBBlock(M, m_h[k], B_data, B_rows, B_cols, row_offset);
            row_offset += r;
        }

        B.updateFromCOO(A.nrows, A.ncols, B_data.size(), B_data.data(), B_rows.data(), B_cols.data());
        BB.updateFromCOO(A.nrows, A.ncols, B_data.size(), B_data.data(), B_rows.data(), B_cols.data());
        // B. updateCSC(A.nrows, A.ncols, B_rows.data(), B_cols.data(), B_data.data(), B_data.size());
        // BB.updateCSC(A.nrows, A.ncols, B_rows.data(), B_cols.data(), B_data.data(), B_data.size());
        {
            std::vector<T> A_data;
            std::vector<int64_t> A_rows, A_cols;
            A_data.reserve(m_jac_coo.nnz + B_data.size());
            A_rows.reserve(m_jac_coo.nnz + B_data.size());
            A_cols.reserve(m_jac_coo.nnz + B_data.size());
            for (int64_t j = 0; j < A.ncols; ++j) {
                int64_t start = A.Ap[j];
                int64_t end = A.Ap[j + 1];
                // Iterate through non-zero elements in column j
                for (int64_t k = start; k < end; ++k) {
                    int64_t i = A.Ai[k]; // Row index of non-zero element
                    // printf("(%4d, %4d) : %d\n", i, j, k);
                    A_data.emplace_back(A.Ax[k]);
                    A_rows.emplace_back(i);
                    A_cols.emplace_back(j);
                }
            }
            for (int64_t j = 0; j < B.ncols; ++j) {
                int64_t start = B.Ap[j];
                int64_t end = B.Ap[j + 1];
                // Iterate through non-zero elements in column j
                for (int64_t k = start; k < end; ++k) {
                    int64_t i = B.Ai[k]; // Row index of non-zero element
                    // printf("(%4d, %4d) : %d\n", i, j, k);
                    A_data.emplace_back(-sigma * B.Ax[k]);
                    A_rows.emplace_back(i);
                    A_cols.emplace_back(j);
                }
            }
            AA.updateFromCOO(A.nrows, A.ncols, A_data.size(), A_data.data(), A_rows.data(), A_cols.data());
            // AA.updateCSC(A.nrows, A.ncols, A_rows.data(), A_cols.data(), A_data.data(), A_data.size());
        }
    }

    std::tuple<sparse::RealCSCMatrix, sparse::RealCSCMatrix, std::vector<T>, double *, ptrdiff_t> Collocator::setupForSubspace(const ptrdiff_t k, const T sigma)
    {
        sparse::RealCSCMatrix A, B;
        double *tmp;
        std::vector<T> v1;
        ptrdiff_t m;

        auto generate_random_matrix = [](std::vector<double> &A, const int seed) {
            std::default_random_engine engine(seed);
            // std::uniform_real_distribution dist(0, 1);
            std::normal_distribution<double> dist;
            std::generate(A.begin(), A.end(), [&]() { return dist(engine); });
        };

        setupSparseABPencilForSpectrum(A, B, sigma);
        m = std::max<ptrdiff_t>({ 2 * k + 1, (ptrdiff_t)std::ceil(2 * std::sqrt(A.nrows)), 20l });
        tmp = (double *)malloc(A.nrows * sizeof(double));

        // tmp will need to be freed
        v1.resize(A.nrows);
        generate_random_matrix(v1, 0);

        return { A, B, v1, tmp, m };
    }

    void Collocator::generateSubspace(const ptrdiff_t k, std::vector<T> &wr, std::vector<T> &wi, const T sigma)
    {
        // Cayley transform (A + sB)(A - sB)^{-1}
        ptrdiff_t i, nsub;
        double mu;
        double *tmp;
        DEV::Matrix A1, A2;
        DEV::BoundaryConditions bc;

        // need to construct M
        double M[m_node * m_node], D[m_node];
        Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), waveSpeed());
        // construct A's
        bc.makePeriodic(m_node, 0);
        A1.update(m_fjac, m_pjac, m_nnodes, m_node, c_nstages, 0, unscaledSpatialPeriod(), m_h, m_state_curr, m_p, m_pmask, NULL, &bc);
        A2.update(m_fjac, m_pjac, m_nnodes, m_node, c_nstages, 0, unscaledSpatialPeriod(), m_h, m_state_curr, m_p, m_pmask, NULL, &bc);
        for (i = 0; i < A1.m_nblocks; i++) {
            mu = m_h[i] * sigma;
            // construct A + sB
            Hopf::add_BBlock_to_partition(m_node, c_nstages, (A1.m_partitions + i)->A, (A1.m_partitions + i)->ldA(), +mu, M, c_A, c_b);
            // construct A - sB
            Hopf::add_BBlock_to_partition(m_node, c_nstages, (A2.m_partitions + i)->A, (A1.m_partitions + i)->ldA(), -mu, M, c_A, c_b);

            Hopf::add_BBlock_to_partition(m_node, c_nstages, (A1.m_partitions_mm + i)->A, (A1.m_partitions + i)->ldA(), +mu, M, c_A, c_b);
            Hopf::add_BBlock_to_partition(m_node, c_nstages, (A2.m_partitions_mm + i)->A, (A1.m_partitions + i)->ldA(), -mu, M, c_A, c_b);
        }
        // make M(x) = (A + sB)solve(A - sB, x)
        tmp = (double *)malloc(A1.size() * sizeof(double));
        constexpr int ipow = 1;
        auto Mop = [&](const double *input, double *output) {
            // A1.gemv(0.0, tmp, 1.0, input);
            // A2.solve(tmp, output);
            A2.solve(input, tmp);
            A1.gemv(0.0, output, 1.0, tmp);

            for (int ii = 0; ii < (ipow - 1); ii++) {
                A2.solve(output, tmp);
                A1.gemv(0.0, output, 1.0, tmp);
            }
        };
        nsub = std::max<ptrdiff_t>({ 2 * k + 1, (ptrdiff_t)std::ceil(2 * std::sqrt(A1.size())), 20l });

        Krylov::better::KSResult result;
        Krylov::better::krylov_schur(Mop, A1.size(), k, nsub, result, 1e-8);
        ptrdiff_t nc = result.nconverged;
        fmt::println("nc = {}", nc);
        wr.resize(nc);
        wi.resize(nc);
        result.getEigenvalues(wr.data(), wi.data());
        for (i = 0; i < nc; i++) {
            const std::complex<double> theta = std::pow(std::complex<double>{ wr[i], wi[i] }, 1.0 / ipow);
            // fmt::println("{:.16e}", theta);
            const std::complex<double> z = sigma * (theta + 1.0) / (theta - 1.0);
            wr[i] = z.real();
            wi[i] = (wi[i] == 0.0) ? 0.0 : z.imag();
        }

        free(tmp);
    }

    int Collocator::setupABPencilForSpectrum(T **A, T **B, const bool include_c)
    {
        size_t N, row_offset, r, k, j, i;
        T M[m_node * m_node];
        T D[m_node];
        auto to_delete = indicesToDelete();

        m_nparam = include_c;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        *A = m_jac_csc.dense();
        *B = (T *)calloc(N * N, sizeof(T));

        memset(M, 0, m_node * m_node * sizeof(T));
        memset(D, 0, m_node * sizeof(T));

        for (const auto &[k, dk] : m_diffusion) {
            D[k] = dk;
        }

        // setup little M for filling in blocks
        k = 0;
        for (i = 0; i < m_node; i++) {
            if (std::find(to_delete.begin(), to_delete.end(), i + 1) != to_delete.end()) {
                continue;
            }
            if (std::find(to_delete.begin(), to_delete.end(), i) != to_delete.end()) {
                j = to_delete[k];
                k++;
                M[j * m_node + (j - 1)] = -(2 * m_spatial_period * m_p[m_np - 1]) / D[j - k];
                continue;
            }
            M[i * m_node + i] = (2 * m_spatial_period * m_p[m_np - 1]) / m_p[0];
        }

        row_offset = 0;
        r = m_node * (c_nstages + 1);
        T *BB = *B;
        for (k = 0; k < (m_nnodes - 1); k++) {
            fillBBlockGGEV(&BB[row_offset * N + row_offset], N, M, m_h[k]);
            row_offset += r;
        }

        return N;
    }

    void Collocator::copyCollocatorDataIntoSelf(const Collocator &other)
    {
        // I don't know any other ways of doing this...
        if (c_nstages != other.c_nstages) {
            puts("cannot copy collocator with different number of stages");
            exit(1);
        }
        m_func = other.m_func;
        m_fjac = other.m_fjac;
        m_pjac = other.m_pjac;
        m_node = other.m_node;
        m_nnodes = other.m_nnodes;
        m_np = other.m_np;
        m_nparam = other.m_nparam;
        m_diffusion = other.m_diffusion;
        m_spatial_period = other.m_spatial_period;

        for (size_t i = 0; i < max_nparam; i++) {
            m_pmask[i] = other.m_pmask[i];
        }
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        const size_t lwork = getLWork(m_nnodes, c_nstages, m_node);
        createCopy(m_state_curr, other.m_state_curr, N);
        createCopy(m_state_prev, other.m_state_prev, N);
        createCopy(m_state_next, other.m_state_next, N);
        createCopy(m_nullspacev, other.m_nullspacev, N);
        createCopy(m_p, other.m_p, m_np);
        createCopy(m_ypold, other.m_ypold, N);
        createCopy(m_t, other.m_t, m_nnodes);
        createCopy(m_h, other.m_h, m_nnodes - 1);
        createCopy(m_work, other.m_work, lwork);
    }

    Collocator Collocator::doubleMesh() const
    {
        ptrdiff_t i;
        const ptrdiff_t nnodes_mid = 2 * m_nnodes - 1;
        double *tmid = (double *)malloc(sizeof(double) * nnodes_mid);
        double *ymid = (double *)malloc(sizeof(double) * nnodes_mid * m_node);

        for (i = 0; i < (ptrdiff_t)(m_nnodes - 1); i++) {
            tmid[2 * i + 0] = m_t[i];
            tmid[2 * i + 1] = 0.5 * (m_t[i] + m_t[i + 1]);
        }
        tmid[2 * m_nnodes - 2] = m_t[m_nnodes - 1];

        denseOutput(tmid, nnodes_mid, ymid);
        return Collocator(c_nstages, m_func, m_fjac, m_pjac, m_node, nnodes_mid, tmid, ymid, m_np, m_p, m_diffusion, m_spatial_period);
    }

    double Collocator::getDirection() const
    {
        return TWIST::sign(m_nullspacev[getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 2) - 1]);
    }

    void Collocator::performWaveSpeedCorrections(const bool)
    {
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 1);
        double c_tmp;
        double hh, dc, norm0, normm1, normp1;
        int niter = 0;
        double *rhs = m_work;
        double *yp = m_work + N;

        m_nparam = 1;
        while (1) {
            c_tmp = m_p[0];
            hh = std::abs(c_tmp) * std::pow(0.5, 26);

            evaluateResidual(m_state_curr, m_state_prev, m_ypold, m_p, rhs, yp);
            norm0 = norm2(rhs, N);
            norm0 *= norm0;

            m_p[0] = c_tmp + hh;
            evaluateResidual(m_state_curr, m_state_prev, m_ypold, m_p, rhs, yp);
            normp1 = norm2(rhs, N);
            normp1 *= normp1;

            m_p[0] = c_tmp - hh;
            evaluateResidual(m_state_curr, m_state_prev, m_ypold, m_p, rhs, yp);
            normm1 = norm2(rhs, N);
            normm1 *= normm1;
            dc = -((normp1 - normm1) / (2 * hh)) / ((normm1 - 2 * norm0 + normp1) / (hh * hh));
            if (!std::isfinite(dc)) {
                break;
            }
            fmt::println("{:4d} | {:.6e} | {:.6e} | {:.6e} |", ++niter, norm0, std::abs(dc), c_tmp);
            m_p[0] = c_tmp + 0.05 * dc;
            if (std::abs(dc) < 1e-5) {
                break;
            }
        }
        fmt::println("final corrected wave speed : {}", m_p[0]);
    }

    bool Collocator::locateBranchPoint(double &k)
    {
        const double sqeps = std::sqrt(std::numeric_limits<double>::epsilon());
        double k0, k1, R0, R1, t, delta;
        int niter;
        ptrdiff_t N;
        std::valarray<double> v;
        Collocator backup(*this);
        bool converged;
        DEV::Matrix A, B;
        DEV::BoundaryConditions bc, zbc;
        double M[m_node * m_node], D[m_node];

        ContinuationBounds bounds = {
            .ds = 1e-4,
            .dsmin = 0.0,
            .dsmax = 1e-8,
            .parmin = -std::numeric_limits<T>::infinity(),
            .parmax = +std::numeric_limits<T>::infinity(),
            .geps = 1e-12,
            .min_nodes_adapt = (int)m_nnodes,
            .max_nodes_adapt = (int)m_nnodes,
            .allow_mesh_adaptation = 0,
        };

        m_nparam = 2;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);
        v.resize(N - 2);
        k0 = k;

        generateNullspace();
        delta = 0;
        converged = false;

        Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), waveSpeed());
        BranchPoint::get_initial_eigenvector(k, v, this, A, B, &bc, &zbc, M, c_A, c_b);

        for (niter = 0; niter < 100; niter++) {
            // init for newton
            t = bounds.ds;
            copyCollocatorDataIntoSelf(backup);
            solveTWave(2, bounds, 0.0, pow(0.5, 15), 100, false);

            // get first residual
            Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), waveSpeed());
            BranchPoint::refine_eigenpair(k0, v, this, A, B, &bc, &zbc, NULL, c_A, c_b, false);
            R0 = k0;

            // print stuff out
            fmt::println("{:4d} : {:.8e} : {: .8e} : {: .8e}", niter, std::abs(R0), t, delta);
            if (std::abs(R0) < 1e-8) {
                converged = true;
                break;
            }

            // update ds and resolve
            delta = TWIST::sign(t) * sqeps * (1 + std::abs(t));

            // perform step with perturbed step size
            bounds.ds = t + delta;
            copyCollocatorDataIntoSelf(backup);
            solveTWave(2, bounds, 0.0, pow(0.5, 15), 10, false);
            // get first residual
            Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), waveSpeed());
            k1 = k0;
            BranchPoint::refine_eigenpair(k1, v, this, A, B, &bc, &zbc, NULL, c_A, c_b, false);
            R1 = k0;

            // get newton step
            delta = R0 * delta / (R1 - R0);

            // apply correction
            bounds.ds = t - delta;
        }
        puts("");

        return converged;
    }

    bool Collocator::locateSaddleNode()
    {
        /*
            NOTE: this method (likely) only works after a saddle point has been crossed
        */
        const double sqeps = std::sqrt(std::numeric_limits<double>::epsilon());
        double R0, R1, t, delta;
        bool converged = false;
        int niter;
        ptrdiff_t N;

        ContinuationBounds bounds = {
            .ds = 1e-4,
            .dsmin = 0.0,
            .dsmax = 1e-8,
            .parmin = -std::numeric_limits<T>::infinity(),
            .parmax = +std::numeric_limits<T>::infinity(),
            .geps = 1e-12,
            .min_nodes_adapt = (int)m_nnodes,
            .max_nodes_adapt = (int)m_nnodes,
            .allow_mesh_adaptation = 0,
        };

        Collocator backup(*this);

        m_nparam = 2;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        // make sure there's a nullspace
        generateNullspace();
        delta = 0;

        for (niter = 0; niter < 100; niter++) {

            // perform continuation step with current step size
            t = bounds.ds;
            copyCollocatorDataIntoSelf(backup);
            solveTWave(2, bounds, 0.0, pow(0.5, 15), 10, false);
            R0 = m_nullspacev[N - 1];

            fmt::println("{:4d} : {:.8e} : {: .8e} : {: .8e}", niter, std::abs(R0), t, delta);

            if (std::abs(R0) < 1e-8) {
                converged = true;
                break;
            }

            delta = TWIST::sign(t) * sqeps * (1 + std::abs(t));

            // perform step with perturbed step size
            bounds.ds = t + delta;
            copyCollocatorDataIntoSelf(backup);
            solveTWave(2, bounds, 0.0, pow(0.5, 15), 10, false);
            R1 = m_nullspacev[N - 1];

            // get newton step
            delta = R0 * delta / (R1 - R0);

            // apply correction
            bounds.ds = t - delta;
        }
        puts("");

        return converged;
    }

    void Collocator::generateEssentialSpectrumBranch(const std::complex<double> lambda0, std::vector<std::complex<double>> &results)
    {
        using complex_t = std::complex<T>;
        constexpr int nsteps = 100;

        int i;
        double M[m_node * m_node], D[m_node];
        // bool converged = false;
        double gamma;
        complex_t k;
        std::valarray<complex_t> v;

        Collocator backup(*this);
        DEV::ComplexMatrix A, B;
        DEV::ComplexBoundaryConditions bc, zbc;

        // initialize everything (k, M, D, bc, v);
        results.clear();
        results.emplace_back(lambda0);
        k = lambda0;
        Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), m_p[0]);
        bc.makePeriodic(m_node, 0);
        zbc.makeZero(m_node, 0);
        // get initial eigenvector
        Hopf::get_initial_eigenvector(lambda0, v, this, A, B, &bc, &zbc, M, c_A, c_b);

        for (i = 0; i < nsteps; i++) {
            // set gamma and refine
            gamma = (double)i * (2.0 * M_PI / (nsteps - 1));
            // fmt::println("gamma = {}", gamma);
            bc.makePhaseShift(gamma, m_node, 0, m_state_curr);
            Hopf::refine_eigenpair(k, v, this, A, B, &bc, &zbc, M, c_A, c_b, true);
            results.emplace_back(k);
        }

        k = lambda0;
        bc.makePeriodic(m_node, 0);
        Hopf::get_initial_eigenvector(lambda0, v, this, A, B, &bc, &zbc, M, c_A, c_b);

        for (i = 0; i < nsteps; i++) {
            // set gamma and refine
            gamma = (double)i * (-2.0 * M_PI / (nsteps - 1));
            // fmt::println("gamma = {}", gamma);
            bc.makePhaseShift(gamma, m_node, 0, m_state_curr);
            Hopf::refine_eigenpair(k, v, this, A, B, &bc, &zbc, M, c_A, c_b, true);
            results.emplace_back(k);
        }
    }

    bool Collocator::locateTorusPoint(ContinuationBounds &bounds, std::vector<std::complex<double>> &unstable_modes, const ptrdiff_t index)
    {
        // TODO: clean up code by moving the lambda expressions into the Hopf namespace
        // where all of the old routines live
        using complex_t = std::complex<T>;

        double M[m_node * m_node], D[m_node];
        bool converged = false;
        double t, dp, pdot, ds_lo, ds_hi, ds, R0;
        complex_t k;
        std::valarray<complex_t> v;
        ptrdiff_t N;

        Collocator backup(*this);
        DEV::ComplexMatrix A, B;
        DEV::ComplexBoundaryConditions bc, zbc;

        Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), m_p[0]);

        k = unstable_modes[index];
        // fmt::println("init k = {}", k.real());
        Hopf::get_initial_eigenvector(k, v, this, A, B, &bc, &zbc, M, c_A, c_b);

        generateNullspace();
        if (getContinuationParameterValue() == bounds.parmax) {
            bounds.ds = -std::abs(bounds.ds);
        }
        else {
            Hopf::refine_eigenpair(k, v, this, A, B, &bc, &zbc, M, c_A, c_b);
        }

        dp = bounds.parmax - bounds.parmin;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 2);
        pdot = m_nullspacev[N - 1];
        // set max step to be size of the interval
        // this will cause bisection method to converge
        // to a bifurcation point that occured in the
        // interval or get pushed to a boundary (because
        // the crossing happend during another continuation
        // step)
        ds_hi = std::abs(dp / pdot);
        ds_lo = 0.0;

        t = bounds.ds;
        for (; (ds_hi - ds_lo) > 1e-15;) {
            ds = TWIST::sign(t) * 0.5 * (ds_lo + ds_hi);
            bounds.ds = ds;
            copyCollocatorDataIntoSelf(backup);
            solveTWave(2, bounds, 0.0, pow(0.5, 15), 10, false);
            Hopf::setup_M_and_D(m_node, M, D, indicesToDelete(), m_diffusion, spatialPeriod(), m_p[0]);
            Hopf::refine_eigenpair(k, v, this, A, B, &bc, &zbc, M, c_A, c_b);
            R0 = k.real();
            // fmt::println("{:4d} : {: .12e} : {: .12e} : {: .12e}", niter++, R0, ds_lo, ds_hi);
            if (std::abs(R0) < 1e-10) {
                converged = true;
                break;
            }
            else if (!std::isfinite(R0)) {
                break;
            }
            if (R0 > 0) {
                // step too small
                ds_lo = std::abs(ds);
            }
            else {
                ds_hi = std::abs(ds);
            }
        }
        if ((ds_hi != std::abs(dp / pdot)) && (ds_lo != 0.0)) {
            // bisection detected at least one crossing
            // but refinement errors likely didn't allow
            // the residuals to go below the tolerance
            // so we'll assume it converged
            converged = true;
        }

        // puts("");
        if (converged) {
            unstable_modes[index] = k;
            fmt::println("error = {: .8e}", std::abs(k.real()));
            fmt::println("omega = {: .8e}", k.imag());
            fmt::println("alpha = {: .8e}", m_p[m_pmask[1]]);
        }


        return converged;
    }

    bool Collocator::solveTWave(const int nparam, ContinuationBounds &bounds, T tol, const T min_damp, const int max_iter, const bool verbose)
    {
        TWIST_CXX_MARK_FUNCTION;

        size_t i;
        int num_iter, stall_count;
        T norm_rhs, norm_delta, norm_deltab, norm_rhs1, damp, norm_Rk, norm_dV;
        T ds, dsmin, dsmax, parmin, parmax;
        T *rhs, *rhs1, *delta, *deltab, *dV, *yp, *ypnext, *jacwork, *Rk;
        T time_jac, time_solve, r1, r2, rate, damp_average, scale;
        T ftol, xtol;
        bool cond1, cond2, cond3, failed_to_converge;
        std::vector<T> norm_hist_rhs, damping_terms_used;

        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, nparam);
        const bool performing_continuation = nparam > 1;

        m_nparam = nparam;

        assert(m_nparam <= max_nparam);

        failed_to_converge = true;

        if (performing_continuation) {
            ds = bounds.ds;
            dsmin = bounds.dsmin;
            dsmax = bounds.dsmax;
            parmin = bounds.parmin;
            parmax = bounds.parmax;
        }

        if (tol <= 0) {
            tol = N * std::numeric_limits<T>::epsilon();
            if (verbose) {
                fmt::println("\nUsing default tolerance of {:.6e}", tol);
            }
            ftol = tol;
            xtol = tol;
            if (0 & verbose) {
                ftol = N * std::numeric_limits<T>::epsilon();
                xtol = norminf(m_state_curr, N) * std::numeric_limits<T>::epsilon();
                fmt::println("xtol = {:.6e}", xtol);
                fmt::println("ftol = {:.6e}", ftol);
            }
        }
        else {
            ftol = tol;
            xtol = tol;
        }

        rhs = &m_work[0 * N];
        rhs1 = &m_work[1 * N];
        delta = &m_work[2 * N];
        deltab = &m_work[3 * N];
        dV = &m_work[4 * N];
        yp = &m_work[5 * N];
        ypnext = &m_work[6 * N];
        jacwork = &m_work[7 * N];
        Rk = &m_work[8 * N];

        for (i = 0; i < (size_t)m_nparam; i++) {
            m_state_curr[N - m_nparam + i] = m_p[m_pmask[i]];
        }

        if (performing_continuation) {
            for (i = 0; i < ((c_nstages + 1) * (m_nnodes - 1) + 1); i++) {
                m_func(&m_state_curr[i * m_node], m_p, &m_ypold[i * m_node]);
                scaleVector(2 * m_spatial_period, &m_ypold[i * m_node], m_node);
            }
            memcpy(m_state_prev, m_state_curr, N * sizeof(T));
            // if (std::abs(ds * m_nullspacev[N - 1]) < 0.1 * dsmin) {
            //     ds = TWIST::sign(ds) * abs(0.1 * dsmin / m_nullspacev[N - 1]);
            // }
            ds = predict(ds, parmin, parmax);
        }

        evaluateResidual(m_state_curr, m_state_prev, m_ypold, m_p, rhs, yp);

        norm_rhs = norm2(rhs, N);
        norm_hist_rhs.push_back(norm_rhs);
        if (!std::isfinite(norm_rhs)) {
            fmt::println("{} {} {} {}", m_nnodes, c_nstages, m_node, nparam);
            fmt::println("N = {}", N);
            for (i = 0; i < N; i++) {
                fmt::println("{:5d} : {: .8e}", i, rhs[i]);
            }
            fmt::println("init norm not finite");
            exit(1);
        }

        if (norm_rhs < tol) {
            failed_to_converge = false;
            bounds.ds = ds;
            if (use_tree_mat()) {
                generateJacobianColmat(m_state_curr, m_state_prev, m_ypold, m_p, jacwork, m_nullspacev);
            }
            else {
                generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, jacwork, m_nullspacev);
            }
            return failed_to_converge;
        }

        if (verbose) {
            constexpr std::string_view header_format_str = "{:^5s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} | {:^8s} | {:^12s} | {:^8s} |";
            auto header = fmt::format(header_format_str, "iter", "||F||", "||dy||", "damp", "c", "J (sec)", "|phase|", "nodes", "solve(sec)", "rate");
            puts(header.c_str());
            fmt::print("{:-^{}s}\n", "", header.size() + 1);
            fmt::println("{:5d} | {:12.6e} | {:^12s} | {:^12s} | {:12.6e} | {:^12s} | {:12.6e} | {:8d} | {:^12s} | {: 8.4f} |", 0, norm_rhs, "-", "-", m_p[m_pmask[0]], "-", std::abs(rhs[N - m_nparam]), m_nnodes, "-", 0.0);
        }

        rate = 0.0;
        damp = 1.0;
        for (num_iter = 0; num_iter < max_iter; num_iter++) {
            if ((num_iter == 0) || (aargmax(rhs, N, 1) == (N - m_nparam))) {
                rhs[N - m_nparam] = 0.0;
            }
            TWIST_MARK_BEGIN("jacobian main work");
            {
                auto start = std::chrono::high_resolution_clock::now();
                if (use_tree_mat()) {
                    generateJacobianColmat(m_state_curr, m_state_prev, m_ypold, m_p, jacwork, m_nullspacev);
                }
                else {
                    generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, jacwork, m_nullspacev);
                }
                auto stop = std::chrono::high_resolution_clock::now();
                time_jac = std::chrono::duration<T>(stop - start).count();
            }
            {
                auto start = std::chrono::high_resolution_clock::now();

                if (use_tree_mat()) {
                    m_colmat.solve(rhs, delta);
                }
                else if (false || use_col_mat()) {
                    m_collocation_matrix.solve(rhs, delta);
                }
                else {
                    m_jac_csc.solve(rhs, delta);
                }
                auto stop = std::chrono::high_resolution_clock::now();
                time_solve = std::chrono::duration<T>(stop - start).count();
            }
            TWIST_MARK_END("jacobian main work");

            if (performing_continuation) {
                // throw std::runtime_error("Need to implement CollocationMatrix::gemv");
                // m_jac_csc.gemv(0.0, Rk, 1.0, m_nullspacev);

                if (use_tree_mat()) {
                    m_colmat.gemv(0.0, Rk, 1.0, m_nullspacev);
                }
                else {
                    m_jac_csc.gemv(0.0, Rk, 1.0, m_nullspacev);
                }
                Rk[N - 1] = 0;
                // use ar
                // fmt::println("rhs[N - 1] = {:.8e}", rhs[N - 1]);
                if (use_tree_mat()) {
                    m_colmat.solve(Rk, dV);
                }
                else if (false || use_col_mat()) {
                    m_collocation_matrix.solve(Rk, dV);
                }
                else {
                    m_jac_csc.solve(Rk, dV);
                }
            }
            norm_delta = norm2(delta, N);
            if (!std::isfinite(norm_delta)) {
#ifndef NDEBUG
                puts("delta had nonfinite norm");
#endif
                failed_to_converge = true;
                return failed_to_converge;
            }

            TWIST_MARK_BEGIN("newton loop");
            // damped newton work loop
            damp = std::min(2 * damp, 1.0);
            while (min_damp < damp) {
                for (i = 0; i < N; i++) {
                    m_state_next[i] = m_state_curr[i] - damp * delta[i];
                }
                for (i = 0; i < (size_t)m_nparam; i++) {
                    m_p[m_pmask[i]] = m_state_next[N - m_nparam + i];
                }
                evaluateResidual(m_state_next, m_state_curr, m_ypold, m_p, rhs1, ypnext);
                // if (performing_continuation) {
                //     rhs1[N - 1] = (inner(m_state_curr, m_nullspacev, N) - inner(m_state_prev, m_nullspacev, N)) - ds;
                // }

                if ((num_iter == 0) || (aargmax(rhs, N, 1) == (N - m_nparam))) {
                    rhs1[N - m_nparam] = 0.0;
                }

                if (use_tree_mat()) {
                    m_colmat.solve(rhs1, deltab);
                }
                else if (false || use_col_mat()) {
                    m_collocation_matrix.solve(rhs1, deltab);
                }
                else {
                    m_jac_csc.solve(rhs1, deltab);
                }

                norm_rhs1 = norm2(rhs1, N);
                norm_deltab = norm2(deltab, N);

                cond1 = norm_deltab <= ((1 - 0.5 * damp) * norm_delta);
                cond2 = norm_rhs1 < norm_rhs;
                if (cond1 && cond2) {
                    break;
                }
                damp *= 0.5;
            }
            TWIST_MARK_END("newton loop");

            TWIST_MARK_BEGIN("state updates");
            if (num_iter) {
                r1 = norm_rhs1 / norm_rhs;
                r2 = norm_rhs / norm_hist_rhs[num_iter - 1];
                if (r2 != 1.0) {
                    rate = std::log(r1) / std::log(r2);
                }
                else {
                    rate = std::numeric_limits<T>::quiet_NaN();
                }
            }
            else {
                rate = 0.0;
            }

            memcpy(m_state_prev, m_state_curr, N * sizeof(double));
            memcpy(m_ypold, yp, (N - m_nparam) * sizeof(double));
            memcpy(m_state_curr, m_state_next, N * sizeof(double));
            if (performing_continuation) {
                // fmt::println("||dV|| = {:.8e}", norm2(dV, N));
                for (i = 0; i < N; i++) {
                    m_nullspacev[i] -= damp * dV[i];
                }
                // fmt::println("scaling by 1 / {:.8e}", norm2(m_nullspacev, N));
                // scaleVector(1.0 / norm2(m_nullspacev, N), m_nullspacev, N);
                normalize(m_nullspacev, N);
            }
            memcpy(yp, ypnext, (N - m_nparam) * sizeof(double));
            memcpy(rhs, rhs1, N * sizeof(double));
            norm_rhs = norm_rhs1;
            norm_hist_rhs.push_back(norm_rhs);
            damping_terms_used.push_back(damp);
            TWIST_MARK_END("state updates");

            if (verbose) {
                fmt::println("{:5d} | {:12.6e} | {:12.6e} | {:12.6e} | {:12.6e} | {:12.6e} | {:12.6e} | {:8d} | {: 12.5e} | {: 8.4f} |", num_iter + 1, norm_rhs, norm_delta, (damp > min_damp ? damp : 2 * damp), m_p[m_pmask[0]], time_jac, abs(rhs[N - m_nparam]), m_nnodes, time_solve, rate);
            }
            if ((norm_rhs < tol) || (norm_delta < tol)) {
                // if ((norm_rhs < ftol) || (norm_delta < xtol)) {
                memcpy(m_ypold, yp, (N - m_nparam) * sizeof(double));
                failed_to_converge = false;
                break;
            }

            cond1 = (norm_rhs / tol - 1) < 2.0;
            cond2 = (norm_delta / tol - 1) < 2.0;
            cond3 = damp <= min_damp;

            if ((cond1 || cond2) && cond3) {
                stall_count++;
            }
            else {
                stall_count = 0;
            }

            if (stall_count == 3) {
                failed_to_converge = false;
                break;
            }
        }
        if (failed_to_converge && verbose) {
            printf("failed to converge in %d iterations\n", max_iter);
        }
        if (performing_continuation && (!failed_to_converge)) {
            int nullspace_count;
            double norm_Rk0 = norm2(Rk, N);
            for (nullspace_count = 0; nullspace_count < 10; nullspace_count++) {
                // m_jac_csc.matvec(Rk, m_nullspacev);
                // throw std::runtime_error("need to implement gemv again");
                // m_jac_csc.gemv(0.0, Rk, 1.0, m_nullspacev);
                if (use_tree_mat()) {
                    m_colmat.gemv(0.0, Rk, 1.0, m_nullspacev);
                }
                else {
                    m_jac_csc.gemv(0.0, Rk, 1.0, m_nullspacev);
                }
                Rk[N - 1] = 0;
                norm_Rk = norm2(Rk, N);

                if (use_tree_mat()) {
                    m_colmat.solve(Rk, dV);
                }
                else if (false || use_col_mat()) {
                    m_collocation_matrix.solve(Rk, dV);
                }
                else {
                    m_jac_csc.solve(Rk, dV);
                }

                norm_dV = norm2(dV, N);
                for (i = 0; i < N; i++) {
                    m_nullspacev[i] -= dV[i];
                }
                scaleVector(1.0 / norm2(m_nullspacev, N), m_nullspacev, N);
                if ((norm_dV <= tol) || (norm_Rk <= tol)) {
                    break;
                }
            }
            if (nullspace_count == 10) {
                fmt::println("\n nullspace failed to converge ({:.8e},{:.8e} > {:.8e}) [{:.8e}] {} -> manual update", norm_dV, norm_Rk, tol, norm_Rk0, ds == 0);
                double dir = TWIST::sign(m_nullspacev[N - 1]);

                generateNullspace();
                if (dir != getDirection()) {
                    flipNullspaceDirection();
                }
                // failed_to_converge = true;
            }

            if (!failed_to_converge && (bounds.allow_mesh_adaptation)) {
                // adapt mesh here
                adaptGrid(bounds.geps, bounds.min_nodes_adapt, bounds.max_nodes_adapt, true);

                // NOTE: get rid of this crap and replace with method boolean argument
                if (solveTWave(1, bounds, 0, min_damp, max_iter, verbose)) {
                    // set ds to be zero to allow for parameter to be corrected
                    // after grid adaptation perturbs computed solution.  This
                    // should prevent the following corrector solve from failing
                    // in the same way natural continuatin fails.
                    double dsold = bounds.ds;
                    double sign_old = getDirection();
                    bounds.ds = 0;
                    bounds.allow_mesh_adaptation = 0;
                    failed_to_converge = solveTWave(2, bounds, 0, min_damp, max_iter, verbose);
                    if (getDirection() != sign_old) {
                        flipNullspaceDirection();
                    }
                    bounds.ds = dsold;
                    bounds.allow_mesh_adaptation = 1;
                }

                if (failed_to_converge) {
                    fmt::println("\ngrid adaptation failed during continuation");
                }
            }
            // scale accordingy to the convergence rate
            damp_average = std::accumulate(damping_terms_used.begin(), damping_terms_used.end(), 0.0) / damping_terms_used.size();
            scale = 6.0 / std::pow(std::max(1, num_iter), damp_average);
            if (!failed_to_converge) {
                ds *= scale;
                ds = TWIST::sign(ds) * std::min(dsmax, std::max(abs(ds), dsmin));
            }
            else {
                ds *= 0.1;
            }
            bounds.ds = ds;
        }

        return failed_to_converge;
    }

    void Collocator::denseOutput(const RP(T) x, size_t nx, RP(T) res) const
    {
        TWIST_CXX_MARK_FUNCTION;

        T *K;
        T q[c_nstages + 1];
        T z[c_nstages + 1];
        size_t i, j, index, stride;
        T xi, theta, acc, hi;
        for (i = 0; i < nx; i++) {
            xi = std::fmod(x[i], 1.0);

            index = searchsorted(m_t, m_nnodes - 1, xi) - 1;
            stride = m_node * (c_nstages + 1) * index;

            TWIST_MARK_BEGIN("polynomial evaluation");

            hi = m_h[index];
            theta = (xi - m_t[index]) / hi;
            // K = m_ypold + stride;
            // skip first one for new interpolant
            // also changed all c_nstages + 1 to c_nstages + 0 below
            K = m_ypold + stride + m_node;

            // z = cumprod(theta)
            acc = 1;
            for (j = 0; j < c_nstages; j++) {
                acc *= theta;
                z[j] = acc;
            }
            // res = yk + hk * K.TPz
            memcpy(&res[i * m_node], &m_state_curr[stride], m_node * sizeof(T));
            dgemv("T", c_nstages, c_nstages, 1.0, &c_P[0], c_nstages, z, 1, 0.0, q, 1);
            dgemv("N", m_node, c_nstages, hi, K, m_node, q, 1, 1.0, &res[i * m_node], 1);
            TWIST_MARK_END("polynomial evaluation");
        }
    }

    void Collocator::denseDerivativeOutput(const RP(T) x, size_t nx, RP(T) res) const
    {
        T *K;
        T q[c_nstages + 1];
        T z[c_nstages + 1];
        size_t i, j, index, stride;
        T xi, theta, acc, hi;
        for (i = 0; i < nx; i++) {
            xi = std::fmod(x[i], 1.0);

            index = searchsorted(m_t, m_nnodes, xi) - 1;
            stride = m_node * (c_nstages + 1) * index;

            hi = m_h[index];
            theta = (xi - m_t[index]) / hi;
            K = m_ypold + stride + m_node;

            // z = cumprod(theta)
            acc = 1;
            for (j = 0; j < c_nstages; j++) {
                z[j] = acc * (j + 1) / (spatialPeriod() * hi);
                acc *= theta;
            }
            // res = yk + hk * K.TPz
            memcpy(&res[i * m_node], K - m_node, m_node * sizeof(T));
            dgemv("T", c_nstages, c_nstages, 1.0, &c_P[0], c_nstages, z, 1, 0.0, q, 1);
            dgemv("N", m_node, c_nstages, hi, K, m_node, q, 1, 0.0, &res[i * m_node], 1);
        }
    }

    void Collocator::denseNuthDerivativeOutput(const size_t nu, const RP(T) x, size_t nx, RP(T) res) const
    {
        const int inu = nu;
        T *K;
        T q[c_nstages + 1];
        T z[c_nstages + 1];
        size_t i, j, k, index, stride;
        T xi, theta, coeff, coeff0, hi;
        for (i = 0; i < nx; i++) {
            xi = std::fmod(x[i], 1.0);

            index = searchsorted(m_t, m_nnodes, xi) - 1;
            stride = m_node * (c_nstages + 1) * index;

            hi = m_h[index];
            theta = (xi - m_t[index]) / hi;
            K = m_ypold + stride + m_node;
            coeff0 = std::pow(spatialPeriod() * hi, -inu);
            for (j = 0; j < c_nstages; j++) {
                if (j < (nu - 1)) {
                    z[j] = 0.0;
                    continue;
                }
                coeff = coeff0;
                for (k = 0; k < nu; k++) {
                    coeff *= (j + 1 - k);
                }
                z[j] = coeff * std::pow(theta, j + 1 - nu);
            }

            memcpy(&res[i * m_node], K - m_node, m_node * sizeof(T));
            dgemv("T", c_nstages, c_nstages, 1.0, &c_P[0], c_nstages, z, 1, 0.0, q, 1);
            dgemv("N", m_node, c_nstages, hi, K, m_node, q, 1, 0.0, &res[i * m_node], 1);
        }
    }

    bool Collocator::solveWithAdaptation(const int nadapt, T geps, size_t min_nodes, size_t max_nodes, T tol, const T min_damp, const int max_iter, const bool verbose, const bool return_on_fail)
    {
        TWIST_CXX_MARK_FUNCTION;

        ContinuationBounds bounds;
        bool failed = solveTWave(1, bounds, tol, min_damp, max_iter, verbose);

        for (int i = 0; i < nadapt; i++) {
            adaptGrid(geps, min_nodes, max_nodes);
            auto start = std::chrono::high_resolution_clock::now();
            failed = solveTWave(1, bounds, tol, min_damp, max_iter, verbose);
            auto stop = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(stop - start).count();
            if (verbose) {
                fmt::println("solve took {:.6e} sec", duration);
            }
            if (return_on_fail && failed) {
                break;
            }
        }
        return failed;
    }

    void Collocator::dumpEig(const std::string &path)
    {
        T *A, *B, *Vr, *work = nullptr;
        T dummy[1];
        int N, info, lwork;
        int *jpvt;
        A = nullptr;
        B = nullptr;
        Vr = nullptr;

        std::vector<T> alphar, alphai, beta;

        N = setupABPencilForSpectrum(&A, &B);
        jpvt = (int *)calloc(N, sizeof(int));
        Vr = (double *)malloc(N * N * sizeof(double));
        alphar.resize(N);
        alphai.resize(N);
        beta.resize(N);

        // auto start = std::chrono::high_resolution_clock::now();
        size_t i;
        auto dgetrf = [](const int m, const int n, double *A, const int lda, int *ipiv, int *info) { dgetrf_(&m, &n, A, &lda, ipiv, info); };
        auto dgetrs = [](const char *trans, const int n, const int nrhs, const double *A, const int lda, const int *ipiv, double *B, const int ldb, int *info) { dgetrs_(trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, info); };
        auto dgeev = [](const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info) { dgeev_(jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info); };

        // A -= 0.1 * B
        for (i = 0; i < (size_t)(N * N); i++) {
            A[i] -= 0.1 * B[i];
        }

        // B = A^{-1}B
        dlatrn(N, B, N);
        dgetrf(N, N, A, N, jpvt, &info);
        dgetrs("T", N, N, A, N, jpvt, B, N, &info);

        // eigvals(B)
        dgeev("N", "V", N, B, N, alphar.data(), alphai.data(), NULL, N, Vr, N, dummy, -1, &info);
        lwork = *dummy;
        work = (T *)realloc(work, lwork * sizeof(T));
        dgeev("N", "V", N, B, N, alphar.data(), alphai.data(), NULL, N, Vr, N, work, lwork, &info);

        for (i = 0; i < (size_t)N; i++) {
            double a = alphar[i];
            double b = alphai[i];
            double m = std::hypot(a, b);
            if ((b == 0) && (m >= (N * std::numeric_limits<double>::epsilon()))) {
                alphar[i] = (1.0 / a) + 0.1;
                beta[i] = 1.0;
            }
            else if (m < (N * std::numeric_limits<double>::epsilon())) {
                alphar[i] = std::numeric_limits<double>::infinity();
                alphai[i] = 0.0;
                beta[i] = 0.0;
            }
            else {
                alphar[i] = (a / (m * m)) + 0.1;
                alphai[i] = (-b / (m * m));
                beta[i] = 1.0;
            }
        }
        H5::H5File file(path, H5F_ACC_TRUNC);
        H5::Group group{ file.createGroup("data") };
        auto xi = plottablePoints();
        ::serialize<double, 1>(group, "alphar", alphar.data(), { (hsize_t)N });
        ::serialize<double, 1>(group, "alphai", alphai.data(), { (hsize_t)N });
        ::serialize<double, 1>(group, "beta", beta.data(), { (hsize_t)N });
        ::serialize<double, 1>(group, "xi", xi.data(), { (hsize_t)xi.size() });
        ::serialize<double, 3>(group, "Vr", Vr, { (hsize_t)2, (hsize_t)N, (hsize_t)N });

        free(jpvt);
        free(work);
        free(A);
        free(B);
        free(Vr);
    }

    void Collocator::spectrum(std::vector<T> &alphar, std::vector<T> &alphai, std::vector<T> &beta, SpectrumStrategy strategy, const bool eigenfunctions, T **V, const bool include_c)
    {
        T *A, *B, *work = nullptr;
        T *fake = nullptr;
        T dummy[1];
        int N, info, lwork;
        int *jpvt;
        A = nullptr;
        B = nullptr;

        N = setupABPencilForSpectrum(&A, &B, include_c);
        if (eigenfunctions) {
            *V = (T *)malloc(N * N * sizeof(T));
        }
        else {
            V = &fake;
        }
        jpvt = (int *)calloc(N, sizeof(int));
        alphar.resize(N);
        alphai.resize(N);
        beta.resize(N);

        if (strategy == SpectrumStrategy::ggev4) {
            dggev4("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, dummy, -1, jpvt, &info);
            lwork = *dummy;
            work = (T *)malloc(lwork * sizeof(T));
            dggev4("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, work, lwork, jpvt, &info);
        }
        else if (strategy == SpectrumStrategy::shiftAndInvert) {
            // auto start = std::chrono::high_resolution_clock::now();
            size_t i;
            double threshold;
            // auto dgetrf = [](const int m, const int n, double *A, const int lda, int *ipiv, int *info) { dgetrf_(&m, &n, A, &lda, ipiv, info); };
            // auto dgetrs = [](const char *trans, const int n, const int nrhs, const double *A, const int lda, const int *ipiv, double *B, const int ldb, int *info) { dgetrs_(trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, info); };
            // auto dgeev = [](const char *jobvl, const char *jobvr, const int n, double *A, const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr, const int ldvr, double *work, const int lwork, int *info) { dgeev_(jobvl, jobvr, &n, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info); };

            // A -= 0.1 * B
            for (i = 0; i < (size_t)(N * N); i++) {
                A[i] -= 0.1 * B[i];
            }

            // B = A^{-1}B
            dlatrn(N, B, N);
            // transpose of A is not the most efficient
            // but solving the transposed system ("T" in dgetrs) leads
            // to more rounding errors that let some infinite eigenvalues
            // get through
            dlatrn(N, A, N);
            dgetrf(N, N, A, N, jpvt, &info);
            dgetrs("N", N, N, A, N, jpvt, B, N, &info);
            threshold = M_SQRT2 * std::max<double>(N, norm2(B, N * N)) * std::numeric_limits<double>::epsilon();
            // eigvals(B)
            dgeev("N", eigenfunctions ? "V" : "N", N, B, N, alphar.data(), alphai.data(), NULL, N, *V, N, dummy, -1, &info);
            lwork = *dummy;
            work = (T *)realloc(work, lwork * sizeof(T));
            dgeev("N", eigenfunctions ? "V" : "N", N, B, N, alphar.data(), alphai.data(), NULL, N, *V, N, work, lwork, &info);
            if (info != 0) {
                exit(1);
                fmt::println("dgeev returned code {}", info);
                for (i = 0; i < N; i++) {
                    for (size_t j = 0; j < N; j++) {
                        if (!std::isfinite(B[i * N + j])) {
                            fmt::println("nonfinite entry in matrix at index ({:5d}, {:5d})", j, i);
                        }
                    }
                }
            }

            for (i = 0; i < (size_t)N; i++) {
                double a = alphar[i];
                double b = alphai[i];
                double m = std::hypot(a, b);
                // fmt::println("{:5d} {:.8e} {:.8e}", i, m, threshold);
                if ((b == 0) && (m >= threshold)) {
                    alphar[i] = (1.0 / a) + 0.1;
                    beta[i] = 1.0;
                }
                else if (m < threshold) {
                    alphar[i] = std::numeric_limits<double>::infinity();
                    alphai[i] = 0.0;
                    beta[i] = 0.0;
                }
                else {
                    alphar[i] = (a / (m * m)) + 0.1;
                    alphai[i] = (-b / (m * m));
                    beta[i] = 1.0;
                }
            }
            // auto stop = std::chrono::high_resolution_clock::now();
            // double total = std::chrono::duration<double>(stop - start).count();
            // fmt::println("shift and invert + dgeev took {} sec", total);
        }
        else if (strategy == SpectrumStrategy::ggev3) {
            dggev3("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, dummy, -1, &info);
            lwork = *dummy;
            work = (T *)malloc(lwork * sizeof(T));
            dggev3("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, work, lwork, &info);
        }
        else if (strategy == SpectrumStrategy::ggev5) {
            dggev5("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, dummy, -1, jpvt, &info);
            lwork = *dummy;
            work = (T *)malloc(lwork * sizeof(T));
            dggev5("N", eigenfunctions ? "V" : "N", N, A, N, B, N, alphar.data(), alphai.data(), beta.data(), NULL, N, *V, N, work, lwork, jpvt, &info);
        }

        free(jpvt);
        free(work);
        free(A);
        free(B);
    }

    void Collocator::essentialSpectrum(const double gamma, std::vector<T> &alphar, std::vector<T> &alphai, std::vector<T> &beta, SpectrumStrategy)
    {
        sparse::RealCSCMatrix sA, sB;
        std::complex<T> *A, *B, *work = nullptr;
        std::complex<T> dummy[1];
        int N, info, lwork;
        int *jpvt;
        std::vector<std::complex<double>> w;
        std::vector<double> rwork;
        A = nullptr;
        B = nullptr;

        setupSparseABPencilForSpectrum(sA, sB, 0.1);
        {
            A = (std::complex<double> *)calloc(sA.nrows * sA.ncols, sizeof(std::complex<double>));
            // printf("(%lld, %lld)\n", nrows, ncols);
            for (int64_t j = 0; j < sA.ncols; ++j) {
                int64_t start = sA.Ap[j];
                int64_t end = sA.Ap[j + 1];
                // Iterate through non-zero elements in column j
                for (int64_t k = start; k < end; ++k) {
                    int64_t i = sA.Ai[k]; // Row index of non-zero element
                    // printf("(%4d, %4d) : %d\n", i, j, k);
                    A[i * sA.ncols + j] = sA.Ax[k]; // Store value in dense matrix
                }
            }
        }
        {
            B = (std::complex<double> *)calloc(sB.nrows * sB.ncols, sizeof(std::complex<double>));
            // printf("(%lld, %lld)\n", nrows, ncols);
            for (int64_t j = 0; j < sB.ncols; ++j) {
                int64_t start = sB.Ap[j];
                int64_t end = sB.Ap[j + 1];
                // Iterate through non-zero elements in column j
                for (int64_t k = start; k < end; ++k) {
                    int64_t i = sB.Ai[k]; // Row index of non-zero element
                    // printf("(%4d, %4d) : %d\n", i, j, k);
                    B[i * sB.ncols + j] = sB.Ax[k]; // Store value in dense matrix
                }
            }
        }

        N = sA.nrows;
        jpvt = (int *)calloc(N, sizeof(int));
        w.resize(N);
        alphar.resize(N);
        alphai.resize(N);
        beta.resize(N);
        rwork.resize(2 * N);

        size_t i;
        double threshold;
        auto zgetrf = [](const int m, const int n, std::complex<double> *A, const int lda, int *ipiv, int *info) { zgetrf_(&m, &n, A, &lda, ipiv, info); };
        auto zgetrs = [](const char *trans, const int n, const int nrhs, const std::complex<double> *A, const int lda, const int *ipiv, std::complex<double> *B, const int ldb, int *info) { zgetrs_(trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, info); };
        auto zgeev = [](const char *jobvl, const char *jobvr, const int n, std::complex<double> *A, const int lda, std::complex<double> *W, std::complex<double> *VL, const int ldvl, std::complex<double> *VR, const int ldvr, std::complex<double> *work, const int lwork, double *rwork, int *info) { zgeev_(jobvl, jobvr, &n, A, &lda, W, VL, &ldvl, VR, &ldvr, work, &lwork, rwork, info); };

        for (i = 0; i < m_node; i++) {
            A[(N - m_node + i - m_nparam) * N + i] *= std::exp(std::complex<double>{ 0.0, 2 * M_PI * gamma });
        }
        // for (i = 0; i < m_node; i++) {
        //     for (size_t j = 0; j < m_node; j++) {
        //         fmt::println("({:4d}, {:4d}) : {: .3e}{:+.3e}i", i, j, A[(N - m_node + i) * N + j].real(), A[(N - m_node + i) * N + j].imag());
        //     }
        // }
        // B = A^{-1}B
        zlatrn(N, B, N);
        zgetrf(N, N, A, N, jpvt, &info);
        zgetrs("T", N, N, A, N, jpvt, B, N, &info);
        threshold = M_SQRT2 * std::max<double>(N, norm2(B, N * N)) * std::numeric_limits<double>::epsilon();
        // eigvals(B)

        zgeev("N", "N", N, B, N, w.data(), NULL, N, NULL, N, dummy, -1, rwork.data(), &info);
        lwork = (*dummy).real();
        work = (std::complex<T> *)realloc(work, lwork * sizeof(std::complex<T>));
        zgeev("N", "N", N, B, N, w.data(), NULL, N, NULL, N, work, lwork, rwork.data(), &info);

        for (i = 0; i < (size_t)N; i++) {
            double a = w[i].real();
            double b = w[i].imag();
            double m = std::hypot(a, b);
            // fmt::println("{:5d} {:.8e} {:.8e}", i, m, threshold);
            if ((b == 0) && (m >= threshold)) {
                alphar[i] = (1.0 / a) + 0.1;
                beta[i] = 1.0;
            }
            else if (m < threshold) {
                alphar[i] = std::numeric_limits<double>::infinity();
                alphai[i] = 0.0;
                beta[i] = 0.0;
            }
            else {
                alphar[i] = (a / (m * m)) + 0.1;
                alphai[i] = (-b / (m * m));
                beta[i] = 1.0;
            }
        }

        free(jpvt);
        free(work);
        free(A);
        free(B);
    }

    void Collocator::serialize(const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, const SolutionTypes solution_type, const FileType file_type)
    {
        /*
        // sizes of things
        std::vector<std::pair<int, T>> m_diffusion;
        */
        hsize_t N;
        H5::Group group;
        if (m_serialization_index == -1) {
            m_serialization_index = 0;
            std::vector<int> diffusion_indices;
            std::vector<double> diffusion_coeffs;
            const int ndiff = m_diffusion.size();
            diffusion_indices.reserve(ndiff);
            diffusion_coeffs.reserve(ndiff);

            for (const auto &[index, coeff] : m_diffusion) {
                diffusion_indices.emplace_back(index);
                diffusion_coeffs.emplace_back(coeff);
            }

            std::filesystem::path p_prefix(prefix);
            if (prefix != "") {
                std::filesystem::create_directories(prefix);
            }
            std::filesystem::path p_name(fmt::format("{}-{}-{}-{}.h5", name, parameter_set, parameter, tag));
            std::filesystem::path p_fpath = p_prefix / p_name;
            m_serialization_file = H5::H5File(p_fpath.c_str(), H5F_ACC_TRUNC);
            {
                H5::Group group(m_serialization_file.createGroup("meta_data"));
                const std::string program_argv{ get_program_argv() };
                const std::string cwd = std::filesystem::current_path();
                ::serialize<int>(group, "file_type", file_type);
                ::serialize<char, 1>(group, "directory", cwd.c_str(), { cwd.size() });
                ::serialize<char, 1>(group, "cmdline", program_argv.c_str(), { program_argv.size() });
            }
            H5::Group group(m_serialization_file.createGroup("basic_info"));
            ::serialize<size_t>(group, "node", m_node);
            ::serialize<size_t>(group, "np", m_np);
            ::serialize<size_t>(group, "nparam", m_nparam);
            ::serialize<size_t>(group, "nstages", c_nstages);
            ::serialize<double>(group, "spatial_period", m_spatial_period);
            ::serialize<int, 1>(group, "pmask", m_pmask, { (hsize_t)max_nparam });
            ::serialize<char, 1>(group, "name", name.c_str(), { name.size() });
            ::serialize<int, 1>(group, "diffusion_indices", diffusion_indices.data(), { (hsize_t)ndiff });
            ::serialize<double, 1>(group, "diffusion_coeffs", diffusion_coeffs.data(), { (hsize_t)ndiff });
            ::serialize<int>(group, "ndiff", ndiff);
            ::serialize<double, 2>(group, "P", c_P, { (hsize_t)c_nstages, (hsize_t)c_nstages });
        }
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, max_nparam);
        group = m_serialization_file.createGroup(fmt::format("{}", m_serialization_index));
        m_serialization_index++;

        ::serialize<size_t>(group, "nnodes", m_nnodes);
        ::serialize<double, 1>(group, "p", m_p, { m_np });
        ::serialize<double, 1>(group, "state_curr", m_state_curr, { N });
        ::serialize<double, 1>(group, "state_prev", m_state_prev, { N });
        ::serialize<double, 1>(group, "state_next", m_state_next, { N });
        ::serialize<double, 1>(group, "nullspacev", m_nullspacev, { N });
        ::serialize<double, 1>(group, "ypold", m_ypold, { N });
        ::serialize<double, 1>(group, "t", m_t, { m_nnodes });
        ::serialize<double, 1>(group, "h", m_h, { m_nnodes - 1 });
        ::serialize<uint64_t>(group, "stype", solution_type);
        ::serialize<double>(group, "norm_wv", solutionNorm(true));
        ::serialize<double>(group, "norm_nv", solutionNorm(false));
    }

    void Collocator::setContinuationParameter(const int parnum)
    {
        m_pmask[1] = parnum;
    }

    T Collocator::getContinuationParameterValue() const
    {
        return m_p[m_pmask[1]];
    }

    size_t Collocator::getNNodes() const
    {
        return m_nnodes;
    }

    size_t Collocator::getNStages() const
    {
        return c_nstages;
    }

    size_t Collocator::getNode() const
    {
        return m_node;
    }

    void Collocator::flipNullspaceDirection()
    {
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 2);
        size_t i;

        for (i = 0; i < N; i++) {
            m_nullspacev[i] = -m_nullspacev[i];
        }
    }

    sparse::RealCSCMatrix Collocator::getBaseJacobian()
    {
        m_nparam = 1;
        generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        return m_jac_csc;
    }

    sparse::COOMatrix &Collocator::getBaseJacobianCOO()
    {
        m_nparam = 1;
        generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        return m_jac_coo;
    }

    void Collocator::getParameterColumn(std::vector<T> &col)
    {
        const size_t N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 1);
        const size_t block_size = (c_nstages + 1) * m_node;
        int64_t irow[block_size];
        int64_t icol[block_size];
        size_t k, stride_k;

        col.clear();
        col.resize(N);

        std::swap(m_pmask[0], m_pmask[1]);
        for (k = 0; k < (m_nnodes - 1); k++) {
            stride_k = block_size * k;
            const T *yk = &m_state_curr[stride_k + m_node];
            generateParamBlock(m_work, irow, icol, &col[stride_k], 0, 0, m_h[k], yk, m_p);
        }
        std::swap(m_pmask[0], m_pmask[1]);
    }

    void Collocator::getParameterColumn(double *col)
    {
        const size_t block_size = (c_nstages + 1) * m_node;
        int64_t irow[block_size];
        int64_t icol[block_size];
        size_t k, stride_k;

        std::swap(m_pmask[0], m_pmask[1]);
        for (k = 0; k < (m_nnodes - 1); k++) {
            stride_k = block_size * k;
            const T *yk = &m_state_curr[stride_k + m_node];
            generateParamBlock(m_work, irow, icol, &col[stride_k], 0, 0, m_h[k], yk, m_p);
        }
        std::swap(m_pmask[0], m_pmask[1]);
    }

    void Collocator::generateNullspace()
    {
        size_t i, N;
        assert((0 < m_pmask[1]) && (m_pmask[1] < (int)m_np));
        m_nparam = 2;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);
        double *r = m_work;
        double *d = m_work + N;

        // generate random last row for matrix
        std::default_random_engine engine(0);
        std::normal_distribution<double> dist;
        std::generate(m_nullspacev, m_nullspacev + N, [&]() { return dist(engine); });
        normalize(m_nullspacev, N);

        if (use_tree_mat()) {
            generateJacobianColmat(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        }
        else {
            generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        }

        // create RHS
        memset(r, 0, (N - 1) * sizeof(T));
        r[N - 1] = 1;

        // x = A^{-1}r
        if (use_tree_mat()) {
            m_colmat.solve(r, m_nullspacev);
        }
        else if (false || use_col_mat()) {
            m_collocation_matrix.solve(r, m_nullspacev);
        }
        else {
            m_jac_csc.solve(r, m_nullspacev);
        }
        normalize(m_nullspacev, N);
        // m_jac_csc.matvec(d, m_nullspacev);
        // throw std::runtime_error("need to implement gemv here too");
        if (use_tree_mat()) {
            m_colmat.gemv(0.0, d, 1.0, m_nullspacev);
        }
        else {
            m_jac_csc.gemv(0.0, d, 1.0, m_nullspacev);
        }
        r[N - 1] = d[N - 1];

        if (use_tree_mat()) {
            m_colmat.solve(r, m_nullspacev);
        }
        else if (false || use_col_mat()) {
            m_collocation_matrix.solve(r, m_nullspacev);
        }
        else {
            m_jac_csc.solve(r, m_nullspacev);
        }
        normalize(m_nullspacev, N);

        // make sure we're going in the desired direction on
        // the first continuation step
        if (TWIST::sign(m_nullspacev[N - 1]) != 1) {
            for (i = 0; i < N; i++) {
                m_nullspacev[i] = -m_nullspacev[i];
            }
        }
    }

    T *Collocator::p()
    {
        return m_p;
    }

    const T *Collocator::p() const
    {
        return m_p;
    }

    size_t Collocator::getNParam() const
    {
        return m_np;
    }

    T *Collocator::t()
    {
        return m_t;
    }

    T *Collocator::h()
    {
        return m_h;
    }

    T *Collocator::y()
    {
        return m_state_curr;
    }

    T *Collocator::yprev()
    {
        return m_state_prev;
    }

    T *Collocator::yp()
    {
        return m_ypold;
    }

    size_t Collocator::NUnknowns() const
    {
        return getNumberOfUnknowns(m_nnodes, c_nstages, m_node, 0);
    }

    diffusion_t Collocator::getDiffusion() const
    {
        return m_diffusion;
    }

    double Collocator::waveSpeed() const
    {
        return *m_p;
    }

    double Collocator::spatialPeriod() const
    {
        return m_p[m_np - 1] * (2 * m_spatial_period);
    }

    double Collocator::unscaledSpatialPeriod() const
    {
        return 2 * m_spatial_period;
    }

    void *Collocator::getLibHandle() const
    {
        return m_handle;
    }

    void Collocator::clearLibHandle()
    {
        m_handle = nullptr;
    }

    void Collocator::setLibHandle(void *handle)
    {
        m_handle = handle;
    }

    std::pair<double, double> Collocator::getJacobianDet()
    {
        return m_jac_csc.det();
    }

    ptrdiff_t Collocator::getContinuationParameterIndex() const
    {
        return m_pmask[1];
    }

    void Collocator::residualForBifurcation(double *r)
    {
        m_nparam = 1;
        evaluateResidual(m_state_curr, m_state_prev, m_ypold, m_p, r, m_work);
    }

    T *Collocator::nullspace()
    {
        return m_nullspacev;
    }

    void Collocator::setupSparsePencilB(sparse::COOMatrix &B)
    {
        sparse::COOMatrix BB;
        size_t N, row_offset, r, k, j, i;
        T M[m_node * m_node];
        T D[m_node];
        auto to_delete = indicesToDelete();
        std::vector<T> B_data;
        std::vector<int64_t> B_rows, B_cols;

        m_nparam = 1;
        N = getNumberOfUnknowns(m_nnodes, c_nstages, m_node, m_nparam);

        generateJacobian(m_state_curr, m_state_prev, m_ypold, m_p, m_work, m_nullspacev);
        B_data.reserve(m_jac_coo.nnz);
        B_rows.reserve(m_jac_coo.nnz);
        B_cols.reserve(m_jac_coo.nnz);

        memset(M, 0, m_node * m_node * sizeof(T));
        memset(D, 0, m_node * sizeof(T));

        for (const auto &[k, dk] : m_diffusion) {
            D[k] = dk;
        }

        // setup little M for filling in blocks
        k = 0;
        for (i = 0; i < m_node; i++) {
            if (std::find(to_delete.begin(), to_delete.end(), i + 1) != to_delete.end()) {
                continue;
            }
            if (std::find(to_delete.begin(), to_delete.end(), i) != to_delete.end()) {
                j = to_delete[k];
                k++;
                M[j * m_node + (j - 1)] = -((2 * m_spatial_period) * m_p[m_np - 1]) / D[j - k];
                continue;
            }
            M[i * m_node + i] = ((2 * m_spatial_period) * m_p[m_np - 1]) / m_p[0];
        }

        row_offset = 0;
        r = m_node * (c_nstages + 1);
        for (k = 0; k < (m_nnodes - 1); k++) {
            fillSparseBBlock(M, m_h[k], B_data, B_rows, B_cols, row_offset);
            row_offset += r;
        }
        BB.data = B_data.data();
        BB.irow = B_rows.data();
        BB.icol = B_cols.data();
        BB.nnz = B_data.size();
        BB.cap = B_data.size();
        BB.nrow = N;
        BB.ncol = N;

        B.setNNZ(0);
        B += BB;

        BB.data = 0;
        BB.irow = 0;
        BB.icol = 0;
        BB.nnz = 0;
        BB.cap = 0;
        BB.nrow = 0;
        BB.ncol = 0;
    }

    void Collocator::setSerializationFileAndIndex(H5::H5File &file, const int index)
    {
        m_serialization_file = file;
        m_serialization_index = index;
    }
} // namespace TWIST
