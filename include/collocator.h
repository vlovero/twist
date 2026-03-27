#pragma once

#include <vector>

#include "GL/gl.h"
#include "H5Cpp.h"
#include "collocation_matrix.h"
#include "colmat/tcolmat.h"
#include "shared.h"
#include "sparse_matrix.h"


size_t extract_nstages(const H5::H5File &file);
size_t extract_nstages(const std::string &h5data_path);


namespace TWIST
{
    class Collocator
    {
    protected:
        // collocator stuff
        const size_t c_nstages = 0;
        const T *c_A = nullptr;
        const T *c_b = nullptr;
        const T *c_c = nullptr;
        const T *c_P = nullptr;
        const T *c_chat = nullptr;

        // functions
        func_t m_func = nullptr;
        func_t m_fjac = nullptr;
        pjac_t m_pjac = nullptr;

        // sizes of things
        size_t m_node = 0;
        size_t m_nnodes = 0;
        size_t m_np = 0;
        int m_nparam = 0;
        std::vector<std::pair<int, T>> m_diffusion;

        // states
        T *m_state_curr = nullptr;
        T *m_state_prev = nullptr;
        T *m_state_next = nullptr;
        T *m_nullspacev = nullptr;
        T *m_p = nullptr;
        T *m_ypold = nullptr;
        int m_pmask[max_nparam];

        // grid info
        T m_spatial_period;
        T *m_t = nullptr;
        T *m_h = nullptr;

        // misc
        T *m_work = nullptr;
        sparse::RealCSCMatrix m_jac_csc;
        sparse::COOMatrix m_jac_coo;
        int m_serialization_index = -1;
        H5::H5File m_serialization_file;
        void *m_handle = nullptr;
        sparse::CollocationMatrix m_collocation_matrix;
        DEV::Matrix m_colmat;
        DEV::BoundaryConditions m_bc;

    public:
        Collocator() = default;
        Collocator &operator=(const Collocator &other);
        Collocator(const Collocator &collocator);
        Collocator(const size_t ncol, func_t func, func_t fjac, pjac_t pjac, size_t node, size_t nnodes, const T *t, const T *y0, size_t np, const T *p, const std::vector<std::pair<int, T>> &diffusion, const T spatial_period);
        Collocator(const std::string &h5data_path, const int solution_index);
        Collocator(const H5::H5File &file, const int solution_index);
        ~Collocator();
        std::vector<T> generatePlottablePoints(const RP(T) tref, size_t nt, const RP(T) href) const;
        std::vector<T> plottablePoints() const;

    protected:
        static size_t getLWork(size_t nnodes, size_t nstages, size_t node);
        static void diff(const T *x, const size_t nx, T *dx);
        static size_t jacobianMainBlockNNZ(size_t node, size_t nstages);
        static size_t jacobianParamBlockNNZ(size_t node, size_t nstages, size_t nparam);
        static size_t jacobianNNZ(size_t nnodes, size_t node, size_t nstages, size_t nparam);

        template <typename data_t>
        void createCopy(data_t *&dst, const data_t *src, size_t n);

        static void scaleVector(T scale, RP(T) x, size_t nx);
        static size_t getNumberOfUnknowns(size_t nnodes, size_t nstages, size_t node, size_t nparam);
        static size_t getMainJBlockNNZ(size_t n, size_t s, size_t m);
        static std::tuple<size_t, size_t, size_t> getJacobianNNZ(size_t n, size_t s, size_t m, size_t np);
        static size_t searchsorted(const T *arr, size_t size, T key);

    protected:
        T computePhaseCondition(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold) const;
        void g(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, RP(T) out) const;
        void evaluateResidual(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) residual, RP(T) yp) const;
        void setupLinearInterpolationAtColloationPoints(T *state, const T *y0) const;
        void generateMainBlock(RP(T) computed_jacs, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, int64_t rowoffset, int64_t coloffset, T hk, const RP(T) yk, const RP(T) p) const;
        void generateParamBlock(RP(T) computed_jacs, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, int64_t rowoffset, int64_t coloffset, T hk, const RP(T) yk, const RP(T) p) const;
        void gjac(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(int64_t) irows, RP(int64_t) icols, RP(T) data, RP(T) work, const RP(T) v = nullptr) const;
        void generateJacobian(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v = nullptr);
        void generateJacobianColmat(const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v = nullptr);
        T predict(T ds, const T parmin, const T parmax);
        void computeMonitorFunctionData(RP(T) F, RP(T) v, T &theta, T &err_exp) const;
        T *monitorFunctionInfo(const T eps, const size_t min_nnodes, const size_t max_nnodes, size_t &nnodes_opt, const bool small_change = false) const;
        void adaptGrid(const T eps, const size_t min_nnodes, const size_t max_nnodes, const bool small_change = false);
        void fillBBlockGGEV(RP(T) block, const size_t ldblock, const RP(T) M, const T hk) const;
        void fillSparseBBlock(const RP(T) M, const T hk, std::vector<T> &B_data, std::vector<int64_t> &B_rows, std::vector<int64_t> &B_cols, const size_t row_offset) const;

    public:
        std::vector<int> indicesToDelete() const;
        void setupSparseABPencilForSpectrum(sparse::RealCSCMatrix &A, sparse::RealCSCMatrix &B, const T sigma);
        int setupABPencilForSpectrum(T **A, T **B, const bool include_c = false);
        std::tuple<sparse::RealCSCMatrix, sparse::RealCSCMatrix, std::vector<T>, double *, ptrdiff_t> setupForSubspace(const ptrdiff_t k, const T sigma);
        void generateSubspace(const ptrdiff_t k, std::vector<T> &wr, std::vector<T> &wi, const T sigma);
        void copyCollocatorDataIntoSelf(const Collocator &other);
        Collocator doubleMesh() const;
        double getDirection() const;
        void performWaveSpeedCorrections(const bool verbose);
        bool solveTWave(const int nparam, ContinuationBounds &bounds, T tol, const T min_damp, const int max_iter, const bool verbose);
        void denseOutput(const RP(T) x, size_t nx, RP(T) res) const;
        void denseDerivativeOutput(const RP(T) x, size_t nx, RP(T) res) const;
        void denseNuthDerivativeOutput(const size_t nu, const RP(T) x, size_t nx, RP(T) res) const;
        bool solveWithAdaptation(const int nadapt, T geps, size_t min_nodes, size_t max_nodes, T tol, const T min_damp, const int max_iter, const bool verbose, const bool return_on_fail = false);
        void spectrum(std::vector<T> &alphar, std::vector<T> &alphai, std::vector<T> &beta, SpectrumStrategy strategy, const bool eigenfunctions, T **V, const bool include_c = false);
        void essentialSpectrum(const double gamma, std::vector<T> &alphar, std::vector<T> &alphai, std::vector<T> &beta, SpectrumStrategy strategy);
        void serialize(const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix = "", const SolutionTypes solution_type = SolutionTypes::regular, const FileType file_type = FileType::continuation);
        void setContinuationParameter(const int parnum);
        T getContinuationParameterValue() const;
        size_t getNNodes() const;
        size_t getNode() const;
        size_t getNStages() const;
        void flipNullspaceDirection();
        void generateNullspace();
        T *p();
        const T *p() const;
        size_t getNParam() const;
        T *t();
        T *y();
        T *h();
        T *yprev();
        T *yp();
        size_t NUnknowns() const;
        diffusion_t getDiffusion() const;
        double waveSpeed() const;
        double solutionNorm(const bool include_velocities) const;
        double spatialPeriod() const;
        double unscaledSpatialPeriod() const;
        void *getLibHandle() const;
        void clearLibHandle();
        void setLibHandle(void *handle);
        std::pair<double, double> getJacobianDet();
        ptrdiff_t getContinuationParameterIndex() const;
        sparse::RealCSCMatrix getBaseJacobian();
        sparse::COOMatrix &getBaseJacobianCOO();
        void getParameterColumn(std::vector<T> &col);
        void getParameterColumn(double *col);
        void residualForBifurcation(double *r);
        T *nullspace();
        void dumpEig(const std::string &path);
        void setupSparsePencilB(sparse::COOMatrix &B);
        void setSerializationFileAndIndex(H5::H5File &file, const int index);

        bool locateSaddleNode();
        bool locateTorusPoint(ContinuationBounds &bounds, std::vector<std::complex<double>> &unstable_modes, const ptrdiff_t index);
        bool locateBranchPoint(double &k);
        void generateJacobianColmat(DEV::Matrix *colmat, const RP(T) Y, const RP(T) Yold, const RP(T) Ypold, const RP(T) p, RP(T) work, RP(T) v);
        void generateEssentialSpectrumBranch(const std::complex<double> lambda0, std::vector<std::complex<double>> &results);
    };
} // namespace TWIST
