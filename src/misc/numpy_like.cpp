#include "numpy_like.h"
#include <cstdio>

size_t searchsorted(const double *arr, size_t size, double key)
{
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

void interp(double *y, const double x, const double *xp, const double *fp, ptrdiff_t length, ptrdiff_t nfp)
{
    // ptrdiff_t low, high, mid, i;
    ptrdiff_t low, i;
    double x0, x1, y0, y1;
    low = searchsorted(xp, length - 1, x) - 1;

    // Linear interpolation within the interval
    assert((low + 1) < length);
    x0 = xp[low];
    x1 = xp[low + 1];
    for (i = 0; i < nfp; i++) {
        y0 = fp[nfp * (low + 0) + i];
        y1 = fp[nfp * (low + 1) + i];
        y[i] = y0 + ((y1 - y0) / (x1 - x0)) * (x - x0);
    }
}

void interp(double *y, const size_t nx, const double *x, const double *xp, const double *fp, const ptrdiff_t nxp, const ptrdiff_t nfp)
{
    for (size_t i = 0; i < nx; i++) {
        interp(&y[i * nfp], x[i], xp, fp, nxp, nfp);
    }
}

std::vector<double> linspace(double a, double b, size_t n)
{
    std::vector<double> x;
    x.resize(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = (double)i * (b - a) / (n - 1);
    }
    return x;
}

size_t argmax(const RP(double) x, const size_t nx, const size_t incx)
{
    size_t i;
    double dmax = *x;
    double imax = 0;
    for (i = incx; i < nx; i += incx) {
        if (x[i] > dmax) {
            dmax = x[i];
            imax = i;
        }
    }
    return imax;
}

size_t aargmax(const RP(double) x, const size_t nx, const size_t incx)
{
    size_t i;
    double dmax = std::abs(*x);
    double imax = 0;
    for (i = incx; i < nx; i += incx) {
        if (std::abs(x[i]) > dmax) {
            dmax = std::abs(x[i]);
            imax = i;
        }
    }
    return imax;
}

void roll(int shift, double *x, const int nx)
{
    // [0, 1, 2, 3, 4, 5] +2 -> [4, 5, 0, 1, 2, 3]
    // [0, 1, 2, 3, 4, 5] -2 -> [2, 3, 4, 5, 0, 1]
    shift %= nx;
    if (shift == 0) {
        return;
    }
    if (shift > 0) {
        double *buff = new double[shift];
        memcpy(buff, &x[nx - shift], shift * sizeof(double));
        memmove(&x[shift], x, (nx - shift) * sizeof(double));
        memcpy(x, buff, shift * sizeof(double));
        delete[] buff;
    }
    else {
        shift = -shift;
        double *buff = new double[shift];
        memcpy(buff, x, shift * sizeof(double));
        memcpy(x, &x[shift], (nx - shift) * sizeof(double));
        memcpy(&x[nx - shift], buff, shift * sizeof(double));
        delete[] buff;
    }
}

double std_deviation(const double *x, const size_t nx, const size_t incx)
{
    size_t i;
    double acc = 0.0, val;
    const double mu = mean(x, nx, incx);
    for (i = 0; i < nx; i++) {
        val = x[i * incx] - mu;
        acc += (val * val - acc) / (i + 1);
    }
    return std::sqrt(acc) / mu;
}

double mean(const double *x, const size_t nx, const size_t incx)
{
    // numerically stable mean
    double acc = 0.0;
    for (size_t i = 0; i < nx; i++) {
        acc += (x[i * incx] - acc) / (i + 1);
    }
    return acc;
}
