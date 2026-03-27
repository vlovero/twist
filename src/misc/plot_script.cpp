#include <fstream>
#include <iterator>
#include <string_view>

static const std::string_view default_script = R"TWIST(
import h5py
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from argparse import ArgumentParser


def setup_rcparams(rc_file=None):
    if rc_file is not None:
        mpl.rc_file(rc_file, use_default_template=True)
        return

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["blue"])
    plt.rcParams["axes.labelsize"] = "x-large"

    tex_packages = ["bm", "amsmath"]
    preamble = [rf"\usepackage{{{p}}}" for p in tex_packages]
    preamble = "\n".join(preamble)
    plt.rcParams.update({"text.latex.preamble": preamble})


def grid_axes(*axes):
    for ax in axes:
        ax.grid(True)


def legend_axes(*axes, **kwargs):
    for ax in axes:
        ax.legend(**kwargs)


def set_tick_size(*axes, size, dox=True, doy=True):
    for ax in axes:
        if dox:
            ax.xaxis.set_tick_params(labelsize=size)
        if doy:
            ax.yaxis.set_tick_params(labelsize=size)


def subplots_centered(nrows, ncols, figsize, nax, row2center=-1, **kwargs):
    if not (nax < nrows * ncols):
        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, **kwargs)
        return fig, axes.ravel()

    fig = plt.figure(figsize=figsize)
    axs = []

    m = nax % ncols
    m = range(1, ncols + 1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m * ncols)
    row2center = -1

    for i in range(0, nax):
        row = i // ncols
        col = i % ncols

        if row == (row2center % nrows):  # center only last row
            off = int(m * (ncols - nax % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m * col + off : m * (col + 1) + off])
        axs.append(ax)

    return fig, np.array(axs)


def plot_after_solve_init(dense=False, names=()):
    node = file["data/node"][...][0]
    ncol = file["data/ncol"][...][0]
    nrows = int((node // 3) + (node % 3 != 0))
    fig, axes = subplots_centered(nrows, 3, (10, (7.25 / 3) * nrows), node)

    if names:
        tex_names = []
        for name in names:
            if name.startswith("$"):
                index = name.rfind("$") + 1
                tmp = "_".join(f"{{{item}}}" for item in name[index:].split("_"))
                tex_names.append(f"{name[: index - 1]}{tmp}$")
                continue

            name = "_".join(f"{{{item}}}" for item in name.split("_"))
            tex_names.append(f"${name}$")
        names = tex_names
    else:
        names = [None] * node

    tc = file["data/tc"][...]
    y = file["data/y"][...].reshape(-1, node)
    if not dense:
        for name, ax, yi in zip(names, axes, y.T):
            ax.plot(tc, yi)
            ax.set_ylabel(name)
            ax.set_xlabel(r"$\xi$")
            ax.grid()
    else:
        td = file["data/td"][...]
        yd = file["data/yd"][...].reshape(-1, node)
        for name, ax, yi, ydi in zip(names, axes, y.T, yd.T):
            ax.plot(td, ydi, color="orange")
            ax.scatter(tc[:: ncol + 1], yi[:: ncol + 1], c="k", s=6, zorder=13)
            ax.set_ylabel(name)
            ax.set_xlabel(r"$\xi$")
            ax.grid()

    fig.tight_layout()


def get_eigenvalues(file, solution_id):
    alphar = file[f"{solution_id}/alphar"][...]
    alphai = file[f"{solution_id}/alphai"][...]
    beta = file[f"{solution_id}/beta"][...]
    where = abs(beta) > (len(beta) * pow(2, -52))
    x = alphar[where] / beta[where]
    y = alphai[where] / beta[where]
    return x, y


def plot_spectrum(solution_id="data", file=None, ax=None, color="k", annotate=True, set_lim=True):
    if file is None:
        file = globals()["file"]
    x, y = get_eigenvalues(file, solution_id)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    imin = np.hypot(x, y).argmin()
    print(np.hypot(x[imin], y[imin]))
    sc = ax.scatter(x, y, c=color, s=6, zorder=13)
    if set_lim:
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        ax.set_xlim(x.min() - 0.05 * dx, x.max() + 0.05 * dx)
        ax.set_ylim(y.min() - 0.05 * dy, y.max() + 0.05 * dy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if annotate:
            ax.annotate(
                r"Re$(\lambda)$",
                xy=(1, 0),
                xycoords=("axes fraction", "data"),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=14,
            )
            ax.annotate(
                r"Im$(\lambda)$",
                xy=(0, 1),
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_position("zero")
    return sc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--solve-init", action="store_true", default=False)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--dense", action="store_true", default=False)
    parser.add_argument("--spectrum", action="store_true", default=False)
    args = parser.parse_args()

    try:
        file = h5py.File(".cache/solve_res.h5")
    except (FileNotFoundError, OSError):
        file = None

    setup_rcparams()

    if args.solve_init:
        plot_after_solve_init(dense=args.dense, names=args.names)
    if args.spectrum:
        plot_spectrum()
    if any((args.solve_init, args.dense, args.spectrum)):
        plt.show()

)TWIST";

static std::string plot_script;

namespace python::plot
{
    void set_script(const std::string &script_path)
    {
        std::ifstream inputFile(script_path);
        plot_script = std::string((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
    }

    const std::string_view get_script()
    {
        return plot_script.empty() ? default_script : plot_script;
    }
} // namespace python::plot
