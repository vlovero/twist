#pragma once

#include "fmt/core.h"
#include "fmt/ranges.h"
#include "linalg.h"
#include "sparse_matrix.h"

void dlaswpc(const int n, RP(double) a, const int lda, const int k1, const int k2, const int *ipiv, const int incx);
void dlaswpr(const int n, RP(double) a, const int lda, const int k1, const int k2, const int *ipiv, const int incx);
void dlarook(const ptrdiff_t n, const ptrdiff_t m, const RP(double) A, const ptrdiff_t ldA, ptrdiff_t *ipiv, ptrdiff_t *jpiv);
void dgeluf(const ptrdiff_t n, const ptrdiff_t m, RP(double) A, const ptrdiff_t ldA, int *ipiv, int *jpiv);

namespace sparse
{
    struct ColMatPartition
    {
        double *A = nullptr;
        double *B = nullptr;
        double *C = nullptr;
        double *D = nullptr;
        int *ipiv = nullptr;
        int *jpiv = nullptr;
        ptrdiff_t ncol = 0;
        ptrdiff_t node = 0;
        ptrdiff_t nparam = 0;

        ColMatPartition() = default;
        ColMatPartition(const ColMatPartition &) = delete;
        ColMatPartition &operator=(ColMatPartition &&other);
        ColMatPartition(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam);
        void update(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam);
        void update(const sparse::RealCSCMatrix *mat, const ptrdiff_t partition_num);
        void display() const;
        ~ColMatPartition();
        void emptyPartition(const ptrdiff_t new_ncol, const ptrdiff_t new_node, const ptrdiff_t new_nparam);
        void addPartitionToCOO(sparse::COOMatrix &mat, const ptrdiff_t partition_num) const;
        void condenseVector(RP(double) x, const ptrdiff_t nx, const ptrdiff_t partition_num) const;
        void condense();
        void backsubstitute(RP(double) x, RP(double) y, const ptrdiff_t nx, const ptrdiff_t partition_num) const;
    };

    struct CollocationMatrix
    {
        const sparse::RealCSCMatrix *m_mat;
        ColMatPartition *m_partitions = nullptr;
        ptrdiff_t m_nblocks = 0;
        ptrdiff_t m_cap = 0;
        sparse::COOMatrix m_S_coo;
        sparse::RealCSCMatrix m_S_csc;
        double *m_work = nullptr;
        bool factored = 0;

        CollocationMatrix() = default;
        CollocationMatrix(const sparse::RealCSCMatrix *mat, const ptrdiff_t ngrid, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam);
        ~CollocationMatrix();
        void update(const sparse::RealCSCMatrix *mat, const ptrdiff_t ngrid, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam);
        void update(const RP(double) Z, const RP(double) U, const RP(double) V, func_t fjac, pjac_t pjac, const ptrdiff_t ngrid, const ptrdiff_t ncol, const ptrdiff_t node, const ptrdiff_t nparam, const int *pmask);
        void extractBC();
        void extractParameterEquations();
        void extractD();
        void factor();
        void solveWork(const RP(double) b, RP(double) x);
        void solve(const RP(double) b, RP(double) x);
        void resizeWork();
    };
} // namespace sparse