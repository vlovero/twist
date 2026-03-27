#include <signal.h>
#include <sys/ioctl.h>

#include "cli/colors.h"
#include "collocator.h"
#include "continuation.h"
#include "fmt/core.h"
#include "shared.h"

static volatile sig_atomic_t sigint_received;

static void handler(int)
{
    sigint_received = 1;
}

LocationBar::LocationBar(double a, double b) : m_a(a), m_b(b), m_pos(0)
{
    ptrdiff_t total, width;
    m_nwhite = 101;
    total = m_nwhite + 25 + 40;
    struct winsize w;
    ioctl(fileno(stdout), TIOCGWINSZ, &w);
    width = std::min<ptrdiff_t>(1024, w.ws_col);
    if (total > (width)) {
        m_nwhite -= total - width;
    }
    m_nwhite = std::max<ptrdiff_t>(m_nwhite, 1);
    m_pbar = fmt::format("\r{: .3e} [{:{}s}] {:.3e}", m_a, "", m_nwhite, m_b);
    m_offset = m_pbar.find('[', 0) + 1;
    m_prev_checkpoint = std::chrono::high_resolution_clock::now();
}

void LocationBar::update(const double y, const std::string &tail)
{
    m_curr_checkpoint = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(m_curr_checkpoint - m_prev_checkpoint).count();

    // don't let io take up too many resources
    if (elapsed < 5e-2) {
        return;
    }
    m_prev_checkpoint = m_curr_checkpoint;

    if (m_a == m_b) {
        fmt::print("{}{}", m_pbar, tail);
        fflush(stdout);
        return;
    }
    int new_pos = std::round((m_nwhite - 1) * (y - m_a) / (m_b - m_a));
    new_pos = std::max(0, new_pos);
    new_pos = std::min((int)(m_nwhite - 1), new_pos);
    m_pbar[m_pos + m_offset] = ' ';
    m_pbar[new_pos + m_offset] = '+';

    fmt::print("{}{}", m_pbar, tail);
    fflush(stdout);

    m_pos = new_pos;
}


void continuation_work_loop(TWIST::Collocator &collocator, int parnum, TWIST::ContinuationBounds &bounds, const std::string &name, const std::string &parameter_set, const std::string &parameter, const std::string &tag, const std::string &prefix, const bool get_nullspace)
{
    bool need_to_break = false;
    bool failed_to_converge = false;
    bool already_failed = false;
    auto &&[ds, dsmin, dsmax, parmin, parmax, geps, min_nodes_adapt, max_nodes_adapt, allow_mesh_adaptation] = bounds;
    LocationBar pbar(parmin, parmax);
    double alpha, alpha_prev, delta_alpha;
    size_t nnodes;

    signal(SIGINT, &handler);

    collocator.setContinuationParameter(parnum);
    alpha = collocator.getContinuationParameterValue();
    alpha_prev = alpha;
    if (get_nullspace) {
        collocator.generateNullspace();
    }

    // serialize initial solution
    collocator.serialize(name, parameter_set, parameter, tag, prefix);

    sigint_received = 0;
    while (!sigint_received) {
        if (!((parmin <= alpha) && (alpha <= parmax))) {
            fmt::println("\n** out of bounds [{} \u2209 ({}, {})] **", alpha, parmin, parmax);
            break;
        }
        if (need_to_break) {
            break;
        }

        TWIST::Collocator backup{ collocator };
        failed_to_converge = collocator.solveTWave(2, bounds, 0.0, pow(0.5, 15), 100, false);
        // failed_to_converge = collocator.arclengthContinuationStep(2, bounds, 0.0, pow(0.5, 15), 20, false);

        if ((!already_failed) && failed_to_converge) {
            puts("\nfailed");
            // inside collocator.solveTwave, ds is scaled by 0.1 when the
            // root-finder fails to converge so no need to do it here.
            collocator.copyCollocatorDataIntoSelf(backup);
            bounds.ds = TWIST::sign(ds) * bounds.dsmin;
            // collocator.solveWithAdaptation(1, bounds.geps, bounds.min_nodes_adapt, bounds.max_nodes_adapt, 0.0, pow(0.5, 15), 100, true);
            if (std::abs(bounds.ds) < (1e3 * std::numeric_limits<double>::epsilon())) {
                need_to_break = true;
                puts("\n** breaking becuase ds is too small (likely at parmin/max) **");
                break;
            }
            already_failed = true;
            continue;
        }

        alpha = collocator.getContinuationParameterValue();
        delta_alpha = alpha - alpha_prev;
        alpha_prev = alpha;
        nnodes = collocator.getNNodes();
        if (!need_to_break) {
            // print(f" | {continuer.p[continuer.pmask[1]]: .6e} | {niter:3d} | {nnodes:4d} | {scale:.2e}", end="")
            pbar.update(alpha, fmt::format(" | {: .6e} | {:4d} | {:.3e} |", alpha, nnodes, std::abs(bounds.ds)));

            const bool ds_too_small = std::abs(bounds.ds) < (1e3 * std::numeric_limits<double>::epsilon());
            const bool alpha_change_too_small = std::abs(delta_alpha) < (1e3 * std::numeric_limits<double>::epsilon() * std::max(std::abs(alpha), std::abs(alpha_prev)));

            if (!failed_to_converge) {
                collocator.serialize(name, parameter_set, parameter, tag, prefix);
                already_failed = false;
            }

            if (failed_to_converge) {
                need_to_break = true;
                puts("\n** breaking becuase Newton iteration failed to converge **");
                break;
            }
            else if (ds_too_small) {
                need_to_break = true;
                puts("\n** breaking becuase ds is too small (likely at parmin/max) **");
                break;
            }
            else if (alpha_change_too_small) {
                need_to_break = true;
                puts("\n** breaking becuase change is continuation parameter is too small (likely at parmin/max) **");
                break;
            }
            else {
                need_to_break = false;
                already_failed = false;
            }
        }
    }
    // while (parmin <= continuer.p[continuer.pmask[1]] <= parmax) and not need_to_break
    puts("");
    if (sigint_received) {
        fmt::println(COLOR_YELLOW "\n The continuation loop has been interrupted by ctrl+C" COLOR_RESET);
    }
    sigint_received = 0;
    signal(SIGINT, SIG_DFL);
}
