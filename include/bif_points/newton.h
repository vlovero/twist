#pragma once

#include "fmt/core.h"
#include "shared.h"


namespace Newton
{
    using residual_t = void (*)(double *, const double *, void *);
    using construct_t = void *(*)(double *, void *);
    using solver_t = void (*)(void *, double *, double *, void *);
    using callback_t = void (*)(double *, double *, void *);

    /*
    def newton(f, jac, x, tol, maxiter, min_damp):
        r = f(x) // evaluate_residuals(r, x, ...)
        damp = 1.0
        if ||r|| < tol:
            return

        for _ in range(maxiter):
            J = jac(x) // construct_linear_system(J, x, ...)
            dx = J.solve(r) // solve_linear_system(J, dx, r, ...)

            damp = min(2 * damp, 1.0)
            while damp > min_damp:
                x1 = x - damp * dx
                r1 = f(x1) // evalute_residual(r1, x1, ...)
                dxb = J.solve(r1) // solve_linear_system(J, dxb, r1)
                if ||dxb|| <= (1 - damp/2) * ||dx||:
                    break
                damp *= 0.5

            r[:] = r1
            x[:] = x1 // update_state(x, x1, ...)
            if ||r|| < tol or ||dx|| < tol:
                break
    */

    bool solve(const ptrdiff_t n, double *x0, residual_t residual, construct_t construct_jac, solver_t solve_system, callback_t callback, double tol, int maxiter, void *data, const bool verbose = true);
} // namespace Newton