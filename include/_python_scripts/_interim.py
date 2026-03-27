import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib import style
from argparse import ArgumentParser


if (style_name := os.environ.get("PLOT_STYLE", None)) is not None:
    style.use(style_name)

USE_THICC = os.environ.get("THICC") not in (None, "0")

SC = "k"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

if USE_THICC:
    plt.rcParams["axes.labelsize"] = 32
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["lines.linewidth"] = 3.5
else:
    plt.rcParams["axes.labelsize"] = "x-large"

tex_packages = ["bm", "amsmath"]
preamble = [rf"\usepackage{{{p}}}" for p in tex_packages]
preamble = "\n".join(preamble)
plt.rcParams.update({"text.latex.preamble": preamble})


def plot_kymo(data):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.imshow(data, cmap="jet", origin="lower", aspect="auto", interpolation="bicubic")
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel("space")
    ax.set_ylabel("time")

    if USE_THICC:
        x = 0.03
        ax.annotate(
            "+",  # Empty text, as we just want an arrow
            xytext=(x, 0.3),  # Arrow tip position (figure fraction)
            xycoords="figure fraction",
            xy=(x, 0.1),  # Arrow tail position (figure fraction)
            textcoords="figure fraction",
            clip_on=False,
            fontsize=20,
            fontweight="bold",
        )
        ax.annotate(
            "",
            xytext=(x + 0.01, 0.3),
            xycoords="figure fraction",
            xy=(x + 0.01, 1.5 * x),
            textcoords="figure fraction",
            arrowprops=dict(
                arrowstyle="<-",
                lw=3.5,
            ),
            clip_on=False,
        )
        ax.annotate(
            "+",
            xytext=(0.2, x),
            xycoords="figure fraction",
            xy=(x, x),
            textcoords="figure fraction",
            clip_on=False,
            fontsize=20,
            fontweight="bold",
        )
        ax.annotate(
            "",
            xytext=(0.2, x + 2e-2),
            xycoords="figure fraction",
            xy=(x + 7e-3, x + 2e-2),
            textcoords="figure fraction",
            arrowprops=dict(
                arrowstyle="<-",
                lw=3.5,
            ),
            clip_on=False,
        )

    plt.show()


def plot_kymo_with_start(x, data):
    # data = data * 120 - 87
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2, layout="constrained")

    ax1.plot(x, data[0], "b")
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    ax1.set_xlabel("space", fontsize=24)
    ax1.set_ylabel("Voltage", fontsize=24)

    im = ax2.imshow(data, cmap="jet", origin="lower", aspect="auto", interpolation="bicubic")
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    ax2.set_xlabel("space", fontsize=24)
    ax2.set_ylabel("time", fontsize=24)
    fig.colorbar(im, ax=ax2)
    plt.show()


def plot_standard(x, y):
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 4))
    ax.scatter(x, y, c="k", s=6, zorder=13)
    ax.grid()
    plt.show()


def plot_time_AP(x, y):
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 4))
    ax.plot(x, np.full_like(x, y[0, 0]), "--k")
    ax.plot(x, y[:, 0], "b")
    ax.set_xlabel("$t$")
    ax.set_ylabel("wave speed")
    plt.show()


def plot_space_AP(x, y, xlabel):
    y = y.reshape(x.shape[0], -1)
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 4))
    ax.plot(x, y[:, 0], "b")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"${xlabel}$")
    ax.set_ylabel("$V$")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", default=".cache/solve_res.h5", nargs="?")
    parser.add_argument("--space-ap-xlabel", default="t")
    args = parser.parse_args()
    file = h5py.File(args.file)
    x = file["data/x"][...]
    y = file["data/y"][...]

    if y.ndim == 3:
        plot_kymo(y[:, :, 0])
        # plot_kymo_with_start(x, y[:, :, 0])
    elif y.ndim == 1:
        plot_standard(x, y)
    elif y.shape[1] == 1:
        plot_time_AP(x, y)
    else:
        plot_space_AP(x, y, args.space_ap_xlabel)
