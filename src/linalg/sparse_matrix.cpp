#include "sparse_matrix.h"
#include "fmt/core.h"

namespace sparse
{
    COOMatrix::~COOMatrix()
    {
        nrow = ncol = nnz = cap = 0;

        if (irow) {
            free(irow);
            irow = nullptr;
        }
        if (icol) {
            free(icol);
            icol = nullptr;
        }
        if (data) {
            free(data);
            data = nullptr;
        }
    }

    void COOMatrix::reshape(size_t new_nrow, size_t new_ncol)
    {
        nrow = new_nrow;
        ncol = new_ncol;
    }

    void COOMatrix::setNNZ(size_t new_nnz)
    {
        void *tmp;
        if (new_nnz <= (size_t)cap) {
            nnz = new_nnz;
            return;
        }

        tmp = realloc(irow, new_nnz * sizeof(int64_t));
        assert(tmp);
        irow = (int64_t *)tmp;
        memset(&irow[nnz], 0, (new_nnz - nnz) * sizeof(int64_t));

        tmp = realloc(icol, new_nnz * sizeof(int64_t));
        assert(tmp);
        icol = (int64_t *)tmp;
        memset(&icol[nnz], 0, (new_nnz - nnz) * sizeof(int64_t));

        tmp = realloc(data, new_nnz * sizeof(double));
        assert(tmp);
        data = (double *)tmp;
        memset(&data[nnz], 0, (new_nnz - nnz) * sizeof(double));

        cap = new_nnz;
        nnz = new_nnz;
    }

    COOMatrix &COOMatrix::operator+=(const COOMatrix &other)
    {
        ptrdiff_t nnz_old = nnz;
        nrow = std::max(nrow, other.nrow);
        ncol = std::max(ncol, other.ncol);
        setNNZ(nnz + other.nnz);
        memcpy(&data[nnz_old], other.data, other.nnz * sizeof(double));
        memcpy(&irow[nnz_old], other.irow, other.nnz * sizeof(int64_t));
        memcpy(&icol[nnz_old], other.icol, other.nnz * sizeof(int64_t));
        return *this;
    }

    COOMatrix &COOMatrix::operator=(const COOMatrix &&other)
    {
        this->~COOMatrix();
        nrow = std::move(other.nrow);
        ncol = std::move(other.ncol);
        nnz = std::move(other.nnz);
        cap = std::move(other.cap);
        data = std::move(other.data);
        irow = std::move(other.irow);
        icol = std::move(other.icol);
        return *this;
    }

    COOMatrix &COOMatrix::operator=(const COOMatrix &other)
    {
        this->~COOMatrix();
        nrow = other.nrow;
        ncol = other.ncol;
        nnz = other.nnz;
        cap = other.cap;
        data = (double *)malloc(cap * sizeof(double));
        memcpy(data, other.data, cap * sizeof(double));
        irow = (int64_t *)malloc(cap * sizeof(int64_t));
        memcpy(irow, other.irow, cap * sizeof(int64_t));
        icol = (int64_t *)malloc(cap * sizeof(int64_t));
        memcpy(icol, other.icol, cap * sizeof(int64_t));
        return *this;
    }

    void COOMatrix::scale(const double value)
    {
        for (ptrdiff_t i = 0; i < nnz; i++) {
            data[i] *= value;
        }
    }

    void COOMatrix::addIdentity()
    {
        auto nnz_old = nnz;
        setNNZ(nnz_old + nrow);
        for (ptrdiff_t i = 0; i < nrow; i++) {
            irow[nnz_old + i] = i;
            icol[nnz_old + i] = i;
            data[nnz_old + i] = 1.0;
        }
    }

    double *COOMatrix::dense(const bool forder) const
    {
        ptrdiff_t i, j, k;
        double *A = (double *)calloc(nrow * ncol, sizeof(double));
        if (forder) {
            for (i = 0; i < nnz; i++) {
                j = irow[i];
                k = icol[i];
                A[j * ncol + k] += data[i];
            }
        }
        else {
            for (i = 0; i < nnz; i++) {
                j = irow[i];
                k = icol[i];
                A[k * nrow + j] += data[i];
            }
        }
        return A;
    }

    void COOMatrix::display() const
    {
        ptrdiff_t i;
        int ndigit = std::ceil(std::log10(nrow));
        for (i = 0; i < nnz; i++) {
            fmt::println("({:{}d}, {:{}d}) : {: .15g}", irow[i], ndigit, icol[i], ndigit, data[i]);
        }
    }

    void COOMatrix::offsetIndices(const int64_t row_offset, const int64_t col_offset)
    {
        for (int64_t i = 0; i < nnz; i++) {
            if (nrow <= (irow[i] + row_offset)) {
                throw std::runtime_error("row index went beyond bounds");
            }
            if (ncol <= (icol[i] + col_offset)) {
                throw std::runtime_error("col index went beyond bounds");
            }
            irow[i] += row_offset;
            icol[i] += col_offset;
        }
    }

    RealCSCMatrix::RealCSCMatrix()
    {
        initStuff();
    }

    // RealCSCMatrix::RealCSCMatrix(const COOMatrix &matrix)
    // {
    //     initStuff();
    // }

    RealCSCMatrix::~RealCSCMatrix()
    {
        nrows = 0;
        ncols = 0;

        if (Ap) {
            free(Ap);
            Ap = nullptr;
        }
        if (Ai) {
            free(Ai);
            Ai = nullptr;
        }
        if (Ax) {
            free(Ax);
            Ax = nullptr;
        }
        if (Symbolic) {
            umfpack_dl_free_symbolic(&Symbolic);
            Symbolic = nullptr;
        }
        if (Numeric) {
            umfpack_dl_free_numeric(&Numeric);
            Numeric = nullptr;
        }
    }

    void RealCSCMatrix::initStuff()
    {
        // not doing this was the source to so much pain
        // and confusion...
        memset(Control, 0, UMFPACK_CONTROL * sizeof(double));
        memset(Info, 0, UMFPACK_INFO * sizeof(double));
        umfpack_dl_defaults(Control);

        // 1 -> only print out errors
        Control[UMFPACK_PRL] = 1;
        Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_BEST;
        Control[UMFPACK_PIVOT_TOLERANCE] = 1;
        Control[UMFPACK_SYM_PIVOT_TOLERANCE] = 1;
        umfpack_dl_report_control(Control);
    }

    bool RealCSCMatrix::alreadyAllocated() const
    {
        return (Ax != nullptr) && (Ai != nullptr) && (Ap != nullptr);
    }

    void RealCSCMatrix::updateFromCOO(const COOMatrix &matrix, const bool same_sparsity_structure)
    {
        void *tmp;
        int status;

        if (!(alreadyAllocated() && same_sparsity_structure && (nrows == matrix.nrow) && (ncols == matrix.ncol))) {
            // reallocate everything
            nrows = matrix.nrow;
            ncols = matrix.ncol;

            tmp = realloc(Ap, (ncols + 1) * sizeof(int64_t));
            assert(tmp);
            Ap = (int64_t *)tmp;

            tmp = realloc(Ai, matrix.nnz * sizeof(int64_t));
            assert(tmp);
            Ai = (int64_t *)tmp;

            tmp = (double *)realloc(Ax, matrix.nnz * sizeof(double));
            assert(tmp);
            Ax = (double *)tmp;

            if (Symbolic) {
                umfpack_dl_free_symbolic(&Symbolic);
                Symbolic = nullptr;
            }
        }

        if (Numeric) {
            umfpack_dl_free_numeric(&Numeric);
            Numeric = nullptr;
        }
        status = umfpack_dl_triplet_to_col(nrows, ncols, matrix.nnz, matrix.irow, matrix.icol, matrix.data, Ap, Ai, Ax, NULL);
        if (status < 0) {
            matrix.display();
            umfpack_dl_report_status(Control, status);
            throw std::runtime_error("umfpack_dl_triplet_to_col failed");
        }
    }

    void RealCSCMatrix::updateFromCOO(const int64_t nrow, const int64_t ncol, const ptrdiff_t nnz, double *data, int64_t *irow, int64_t *icol, const bool same_sparsity_structure)
    {
        COOMatrix mat;
        mat.nrow = nrow;
        mat.ncol = ncol;
        mat.nnz = nnz;
        mat.cap = nnz;
        mat.data = data;
        mat.irow = irow;
        mat.icol = icol;

        updateFromCOO(mat, same_sparsity_structure);

        mat.nrow = 0;
        mat.ncol = 0;
        mat.nnz = 0;
        mat.cap = 0;
        mat.data = 0;
        mat.irow = 0;
        mat.icol = 0;
    }

    double *RealCSCMatrix::dense(const bool forder) const
    {
        ptrdiff_t i, j, k, start, end;
        double *A = (double *)calloc(nrows * ncols, sizeof(double));
        if (forder) {
            for (j = 0; j < ncols; j++) {
                start = Ap[j];
                end = Ap[j + 1];
                for (k = start; k < end; k++) {
                    i = Ai[k];
                    A[j * nrows + i] = Ax[k];
                }
            }
        }
        else {
            for (j = 0; j < ncols; j++) {
                start = Ap[j];
                end = Ap[j + 1];
                for (k = start; k < end; k++) {
                    i = Ai[k];
                    A[i * ncols + j] = Ax[k];
                }
            }
        }

        return A;
    }

    void RealCSCMatrix::factor()
    {
        int status;
        // if no symbolic decomp do that
        if (!Symbolic) {
            status = umfpack_dl_symbolic(nrows, ncols, Ap, Ai, Ax, &Symbolic, Control, Info);
            if (status < 0) {
                Control[UMFPACK_PRL] = 6;
                umfpack_dl_report_info(Control, Info);
                umfpack_dl_report_status(Control, status);
                throw std::runtime_error("umfpack_dl_symbolic failed");
            }
        }

        if (!Numeric) {
            status = umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
            if (status < 0) {
                Control[UMFPACK_PRL] = 6;
                umfpack_dl_report_info(Control, Info);
                umfpack_dl_report_status(Control, status);
                throw std::runtime_error("umfpack_dl_numeric failed");
            }
        }
    }

    void RealCSCMatrix::solve(const double *b, double *x, const int solve_type)
    {
        int status;

        factor();

        status = umfpack_dl_solve(solve_type, Ap, Ai, Ax, x, b, Numeric, Control, Info);
        if (status < 0) {
            Control[UMFPACK_PRL] = 6;
            umfpack_dl_report_info(Control, Info);
            umfpack_dl_report_status(Control, status);
            throw std::runtime_error("umfpack_dl_solve failed");
        }
    }

    // perform y = beta * y + alpha * A * x
    void RealCSCMatrix::gemv(const double beta, double *y, const double alpha, const double *x) const
    {
        ptrdiff_t i, j;
        double fac;
        if (beta == 0) {
            memset(y, 0, ncols * sizeof(double));
        }
        else {
            for (i = 0; i < ncols; i++) {
                y[i] *= beta;
            }
        }
        for (i = 0; i < ncols; i++) {
            fac = x[i] * alpha;
            for (j = Ap[i]; j < Ap[i + 1]; j++) {
                y[Ai[j]] += fac * Ax[j];
            }
        }
    }

    void RealCSCMatrix::gemv(const std::complex<double> beta, std::complex<double> *y, const std::complex<double> alpha, const std::complex<double> *x) const
    {
        ptrdiff_t i, j;
        std::complex<double> fac;
        if (beta == 0.0) {
            memset(y, 0, ncols * sizeof(std::complex<double>));
        }
        else {
            for (i = 0; i < ncols; i++) {
                y[i] *= beta;
            }
        }
        for (i = 0; i < ncols; i++) {
            fac = x[i] * alpha;
            for (j = Ap[i]; j < Ap[i + 1]; j++) {
                y[Ai[j]] += fac * Ax[j];
            }
        }
    }

    std::pair<double, double> RealCSCMatrix::det()
    {
        double Mx, Ex;
        factor();
        int status = umfpack_dl_get_determinant(&Mx, &Ex, Numeric, Info);
        if (status < 0) {
            Control[UMFPACK_PRL] = 6;
            umfpack_dl_report_info(Control, Info);
            umfpack_dl_report_status(Control, status);
            throw std::runtime_error("umfpack_dl_get_determinant failed");
        }
        return std::pair<double, double>{ Mx, Ex };
    }

    int64_t RealCSCMatrix::nnz() const
    {
        return Ap[ncols];
    }

    RealCSCMatrix &RealCSCMatrix::operator=(const RealCSCMatrix &other)
    {
        this->~RealCSCMatrix();
        initStuff();
        nrows = other.nrows;
        ncols = other.ncols;
        Ap = (int64_t *)malloc((other.ncols + 1) * sizeof(int64_t));
        Ai = (int64_t *)malloc((other.nnz() + 1) * sizeof(int64_t));
        Ax = (double *)malloc((other.nnz() + 1) * sizeof(double));
        Symbolic = nullptr;
        Numeric = nullptr;
        memcpy(Ap, other.Ap, (ncols + 1) * sizeof(int64_t));
        memcpy(Ai, other.Ai, other.nnz() * sizeof(int64_t));
        memcpy(Ax, other.Ax, other.nnz() * sizeof(double));
        return *this;
    }

    RealCSCMatrix::RealCSCMatrix(const RealCSCMatrix &mat) : nrows(mat.nrows), ncols(mat.ncols), Ap((int64_t *)malloc((mat.ncols + 1) * sizeof(int64_t))), Ai((int64_t *)malloc((mat.nnz() + 1) * sizeof(int64_t))), Ax((double *)malloc((mat.nnz() + 1) * sizeof(double))), Symbolic(nullptr), Numeric(nullptr)
    {
        initStuff();
        memcpy(Ap, mat.Ap, (ncols + 1) * sizeof(int64_t));
        memcpy(Ai, mat.Ai, mat.nnz() * sizeof(int64_t));
        memcpy(Ax, mat.Ax, mat.nnz() * sizeof(double));
    }

    void RealCSCMatrix::scale(const double scale)
    {
        for (int64_t i = 0; i < nnz(); i++) {
            Ax[i] *= scale;
        }
    }

    void RealCSCMatrix::addIdentity()
    {
        int64_t i, j;
        for (i = 0; i < ncols; i++) {
            for (j = Ap[i]; j < Ap[i + 1]; j++) {
                if (Ai[j] == i) {
                    Ax[j] += 1.0;
                    break;
                }
            }
        }
    }

    ComplexCSCMatrix::ComplexCSCMatrix()
    {
        initStuff();
    }

    ComplexCSCMatrix::ComplexCSCMatrix(const COOMatrix &A_real, const COOMatrix &A_imag)
    {
        void *tmp;
        assert(A_real.nrow == A_imag.nrow);
        assert(A_real.ncol == A_imag.ncol);
        assert(A_real.nnz == A_imag.nnz);
        for (ptrdiff_t i = 0; i < A_real.nnz; i++) {
            assert(A_real.irow[i] == A_imag.irow[i]);
            assert(A_real.icol[i] == A_imag.icol[i]);
        }
        nrows = A_real.nrow;
        ncols = A_real.ncol;

        tmp = realloc(Ap, (ncols + 1) * sizeof(int64_t));
        assert(tmp);
        Ap = (int64_t *)tmp;

        tmp = realloc(Ai, A_real.nnz * sizeof(int64_t));
        assert(tmp);
        Ai = (int64_t *)tmp;

        tmp = (double *)realloc(Ax, A_real.nnz * sizeof(double));
        assert(tmp);
        Ax = (double *)tmp;

        tmp = (double *)realloc(Az, A_real.nnz * sizeof(double));
        assert(tmp);
        Az = (double *)tmp;

        umfpack_zl_triplet_to_col(nrows, ncols, A_real.nnz, A_real.irow, A_real.icol, A_real.data, A_imag.data, Ap, Ai, Ax, Az, NULL);
        initStuff();
    }

    ComplexCSCMatrix::~ComplexCSCMatrix()
    {
        nrows = 0;
        ncols = 0;

        if (Ap) {
            free(Ap);
            Ap = nullptr;
        }
        if (Ai) {
            free(Ai);
            Ai = nullptr;
        }
        if (Ax) {
            free(Ax);
            Ax = nullptr;
        }
        if (Az) {
            free(Az);
            Az = nullptr;
        }
        if (Symbolic) {
            umfpack_dl_free_symbolic(&Symbolic);
            Symbolic = nullptr;
        }
        if (Numeric) {
            umfpack_dl_free_numeric(&Numeric);
            Numeric = nullptr;
        }
    }

    void ComplexCSCMatrix::initStuff()
    {
        // not doing this was the source to so much pain
        // and confusion...
        memset(Control, 0, UMFPACK_CONTROL * sizeof(double));
        memset(Info, 0, UMFPACK_INFO * sizeof(double));
        umfpack_zl_defaults(Control);

        // 1 -> only print out errors
        Control[UMFPACK_PRL] = 1;
        Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_BEST;
        umfpack_zl_report_control(Control);
    }

    bool ComplexCSCMatrix::alreadyAllocated() const
    {
        return (Ax != nullptr) && (Az != nullptr) && (Ai != nullptr) && (Ap != nullptr);
    }

    void ComplexCSCMatrix::factor()
    {
        int status;
        // if no symbolic decomp do that
        if (!Symbolic) {
            status = umfpack_zl_symbolic(nrows, ncols, Ap, Ai, Ax, Az, &Symbolic, Control, Info);
            if (status < 0) {
                Control[UMFPACK_PRL] = 6;
                umfpack_zl_report_info(Control, Info);
                umfpack_zl_report_status(Control, status);
                throw std::runtime_error("umfpack_zl_symbolic failed");
            }
        }

        if (!Numeric) {
            status = umfpack_zl_numeric(Ap, Ai, Ax, Az, Symbolic, &Numeric, Control, Info);
            if (status < 0) {
                Control[UMFPACK_PRL] = 6;
                umfpack_zl_report_info(Control, Info);
                umfpack_zl_report_status(Control, status);
                throw std::runtime_error("umfpack_zl_numeric failed");
            }
        }
    }

    void ComplexCSCMatrix::solve(const double *br, const double *bi, double *xr, double *xi, const int solve_type)
    {
        int status;

        factor();
        status = umfpack_zl_solve(solve_type, Ap, Ai, Ax, Az, xr, xi, br, bi, Numeric, Control, Info);
        if (status < 0) {
            Control[UMFPACK_PRL] = 6;
            umfpack_zl_report_info(Control, Info);
            umfpack_zl_report_status(Control, status);
            throw std::runtime_error("umfpack_zl_solve failed");
        }
    }

    void ComplexCSCMatrix::gemv(const std::complex<double> beta, double *yr, double *yi, const std::complex<double> alpha, const double *xr, const double *xi) const
    {
        ptrdiff_t i, j;
        std::complex<double> tmp;

        for (i = 0; i < ncols; i++) {
            tmp = beta * std::complex<double>{ yr[i], yi[i] };
            yr[i] = tmp.real();
            yi[i] = tmp.imag();
        }
        for (i = 0; i < ncols; i++) {
            for (j = Ap[i]; j < Ap[i + 1]; j++) {
                tmp = alpha * (std::complex<double>{ Ax[j], Az[j] } * std::complex<double>{ xr[i], xi[i] });
                yr[Ai[j]] += tmp.real();
                yi[Ai[j]] += tmp.imag();
            }
        }
    }

    int64_t ComplexCSCMatrix::nnz() const
    {
        return Ap[ncols];
    }

    ComplexCSCMatrix::ComplexCSCMatrix(const ComplexCSCMatrix &mat) : nrows(mat.nrows), ncols(mat.ncols), Ap((int64_t *)malloc((mat.ncols + 1) * sizeof(int64_t))), Ai((int64_t *)malloc((mat.nnz() + 1) * sizeof(int64_t))), Ax((double *)malloc((mat.nnz() + 1) * sizeof(double))), Az((double *)malloc((mat.nnz() + 1) * sizeof(double))), Symbolic(nullptr), Numeric(nullptr)
    {
        initStuff();
        memcpy(Ap, mat.Ap, (ncols + 1) * sizeof(int64_t));
        memcpy(Ai, mat.Ai, mat.nnz() * sizeof(int64_t));
        memcpy(Ax, mat.Ax, mat.nnz() * sizeof(double));
        memcpy(Az, mat.Az, mat.nnz() * sizeof(double));
    };
} // namespace sparse
