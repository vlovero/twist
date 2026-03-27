#pragma once

#include "shared.h"
#include "suitesparse/umfpack.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace sparse
{
    struct COOMatrix
    {
        ptrdiff_t nrow = 0;
        ptrdiff_t ncol = 0;
        ptrdiff_t nnz = 0;
        ptrdiff_t cap = 0;
        double *data = nullptr;
        int64_t *irow = nullptr;
        int64_t *icol = nullptr;

        COOMatrix() = default;
        ~COOMatrix();                             // done
        void reshape(size_t nrows, size_t ncols); // done
        void setNNZ(size_t nnz);                  // done
        void scale(const double value);           // done
        void addIdentity();                       // done;
        double *dense(const bool forder = false) const;
        void display() const;
        void offsetIndices(const int64_t row_offset, const int64_t col_offset);
        COOMatrix &operator+=(const COOMatrix &other); // done
        COOMatrix &operator=(const COOMatrix &&other);
        COOMatrix &operator=(const COOMatrix &other);
    };

    struct RealCSCMatrix
    {
        int64_t nrows = 0;
        int64_t ncols = 0;
        int64_t *Ap = nullptr;
        int64_t *Ai = nullptr;
        double *Ax = nullptr;
        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        void *Symbolic = nullptr;
        void *Numeric = nullptr;

        RealCSCMatrix();
        RealCSCMatrix(const COOMatrix &mat);
        RealCSCMatrix(const RealCSCMatrix &mat);
        ~RealCSCMatrix();
        void initStuff();
        void updateFromCOO(const COOMatrix &mat, const bool same_sparsity_structure = true);
        void updateFromCOO(const int64_t nrow, const int64_t ncol, const ptrdiff_t nnz, double *data, int64_t *irow, int64_t *icol, const bool same_sparsity_structure = true);
        bool alreadyAllocated() const;
        double *dense(const bool forder = false) const;
        void factor();
        void solve(const double *b, double *x, const int solve_type = UMFPACK_A);
        void gemv(const double beta, double *y, const double alpha, const double *x) const;
        void gemv(const std::complex<double> beta, std::complex<double> *y, const std::complex<double> alpha, const std::complex<double> *x) const;
        std::pair<double, double> det();
        RealCSCMatrix &operator=(const RealCSCMatrix &other);
        int64_t nnz() const;
        void scale(const double scale);
        void addIdentity();
    };

    struct ComplexCSCMatrix
    {
        int64_t nrows = 0;
        int64_t ncols = 0;
        int64_t *Ap = nullptr;
        int64_t *Ai = nullptr;
        double *Ax = nullptr;
        double *Az = nullptr;
        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        void *Symbolic = nullptr;
        void *Numeric = nullptr;

        ComplexCSCMatrix();
        ~ComplexCSCMatrix();
        ComplexCSCMatrix(const COOMatrix &A_real, const COOMatrix &A_imag);
        ComplexCSCMatrix(const ComplexCSCMatrix &mat);

        void initStuff();
        bool alreadyAllocated() const;

        void factor();
        void solve(const double *br, const double *bi, double *xr, double *xi, const int solve_type = UMFPACK_A);
        void gemv(const std::complex<double> beta, double *yr, double *yi, const std::complex<double> alpha, const double *xr, const double *xi) const;
        int64_t nnz() const;
    };
} // namespace sparse