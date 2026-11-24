#include "modulating/hopf_continuation.h"
#include "cli/colors.h"
#include "continuation.h"
#include "fmt/core.h"
#include <algorithm>
#include <cmath>
#include <signal.h>

static volatile sig_atomic_t hopf_sigint_received = 0;

static void hopf_sigint_handler(int)
{
    hopf_sigint_received = 1;
}

void hopf_continuation_work_loop(TWIST::HopfBranchSolver &solver, TWIST::HopfContinuationBounds &bounds, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, bool verbose, int max_steps)
{
    signal(SIGINT, &hopf_sigint_handler);
    hopf_sigint_received = 0;

    const double parmin = bounds.parmin;
    const double parmax = bounds.parmax;
    double ds = std::abs(bounds.ds);

    const double alpha0 = solver.continuationParam();
    double alpha = alpha0;
    double alpha_prev = alpha0;
    bool classified_branch = false;
    bool already_failed = false;

    LocationBar pbar(parmin, parmax);

    // Serialize initial point (step 0: Hopf point at A=0, or resumed point)
    solver.serialize(name, parameter_set, parameter, tag, prefix, TWIST::SolutionTypes::torusPoint);

    fmt::println("Starting modulating wave continuation at {} = {:.6e}, c = {:.6f}, T = {:.6f}", parameter, alpha0, solver.waveSpeed(), solver.period());

    for (int step = 1; step <= max_steps && !hopf_sigint_received; step++) {
        if (!((parmin <= alpha) && (alpha <= parmax))) {
            fmt::println("\n** out of bounds [{} \u2209 ({}, {})] **", alpha, parmin, parmax);
            break;
        }

        // First step off the Hopf point uses moderate initial amplitude step so Newton stays in Hopf basin
        double step_ds = ds;
        if (step == 1 && solver.oscillationAmplitude() < 1e-8) {
            step_ds = std::min(ds, 0.05);
        }

        solver.predict(step_ds);
        const bool converged = solver.solveNewton(step_ds, 1e-10, 1e-4, 25, verbose);

        if (!converged) {
            solver.restoreAcceptedStep();
            ds *= 0.5;
            if (ds < bounds.dsmin) {
                if (already_failed) {
                    fmt::println("\n** breaking because ds ({:.3e}) < dsmin ({:.3e}) **", ds, bounds.dsmin);
                    break;
                }
                ds = bounds.dsmin;
                already_failed = true;
            }
            if (verbose) {
                fmt::println("  [HopfCont] step {} failed, reducing ds -> {:.3e}", step, ds);
            }
            continue;
        }

        already_failed = false;

        // Adapt space-time mesh if enabled (skip on tiny amplitude step 1 unless error exceeds geps)
        if (bounds.allow_mesh_adaptation && (step >= 2 || solver.oscillationAmplitude() > 1e-3)) {
            solver.adaptSpaceTimeMesh(bounds, step_ds, true, verbose);
        }

        solver.updatePhaseConditions();
        solver.computeTangent();
        solver.acceptStep();

        alpha = solver.continuationParam();
        const double delta_alpha = alpha - alpha_prev;
        alpha_prev = alpha;
        const double amp = solver.oscillationAmplitude();

        if (!classified_branch && step >= 2 && std::abs(alpha - alpha0) > 1e-10) {
            classified_branch = true;
            const char *dir_str = (alpha > alpha0) ? "alpha > alpha_Hopf" : "alpha < alpha_Hopf";
            fmt::println("\n[HopfBranch] Branch emerging toward {} (alpha={:.6e}, A={:.3e}, T={:.6f})", dir_str, alpha, amp, solver.period());
        }

        pbar.update(alpha, fmt::format(" | {}={: .5e} | A={:.3e} | T={:.5f} | c={:.5f} | {}x{} | ds={:.2e} |",
            parameter, alpha, amp, solver.period(), solver.waveSpeed(), solver.nt(), solver.maxSpatialNodes(), ds));

        solver.serialize(name, parameter_set, parameter, tag, prefix, TWIST::SolutionTypes::regular);

        // Check branch reconnection (A -> 0 after moving away from Hopf point)
        if (step > 3 && amp < 1e-8) {
            fmt::println("\n** modulating wave branch reconnected to steady wave (A = {:.3e}) **", amp);
            break;
        }

        // Adaptive step size control
        if (step >= 2 && std::abs(delta_alpha) < 1e-12 * (1.0 + std::abs(alpha)) && amp < 1e-9) {
            fmt::println("\n** step change too small **");
            break;
        }
        ds = std::min(bounds.dsmax, 1.25 * ds);
    }

    puts("");
    if (hopf_sigint_received) {
        fmt::println(COLOR_YELLOW "\n The Hopf continuation loop has been interrupted by ctrl+C" COLOR_RESET);
    }
    hopf_sigint_received = 0;
    signal(SIGINT, SIG_DFL);
}
