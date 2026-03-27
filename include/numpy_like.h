#ifndef TWIST_NUMPY_LIKE_H
#define TWIST_NUMPY_LIKE_H

#include "shared.h"


void interp(double *y, const size_t nx, const double *x, const double *xp, const double *fp, const ptrdiff_t nxp, const ptrdiff_t nfp);
std::vector<double> linspace(double a, double b, size_t n);
size_t argmax(const RP(double) x, const size_t nx, const size_t incx);
size_t aargmax(const RP(double) x, const size_t nx, const size_t incx);
void roll(int shift, double *x, const int nx);
double std_deviation(const double *x, const size_t nx, const size_t incx);
double mean(const double *x, const size_t nx, const size_t incx);

#endif // TWIST_NUMPY_LIKE_H