#include <fstream>
#include <iterator>
#include <string_view>

static const std::string_view default_script = R"TWIST(
import os
import h5py
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from pprint import pprint


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
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            "blue",
            "orange",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "purple",
            "pink",
            "brown",
            "orange",
            "teal",
            "coral",
            "lightblue",
            "lime",
            "lavender",
            "turquoise",
            "darkgreen",
            "tan",
            "salmon",
            "gold",
        ]
    )
    tex_packages = ["bm", "amsmath"]
    plt.rcParams["text.latex.preamble"] = rf"\usepackage{{{'.'.join(tex_packages)}}}"


def kymo_plot(ax, args, Y, L, T):
    im = ax.imshow(Y[:, :, 0], cmap="jet", origin="lower", aspect="auto", interpolation="bicubic", extent=[0, L, 0, T])
    if args.hide_ticks:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    
    if args.colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if args.colorbar_label is not None:
            cbar.set_label(args.colorbar_label, labelpad=15)
    
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)


def starting_profile_plot(ax, args, x, Y, sysvars):
    im = ax.plot(x, Y[0, :, 0])
    if args.hide_ticks:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    
    ax.set_xlabel(args.xlabel)
    if args.profile_ylabel is None:
        ax.set_ylabel(f"${sysvars[0]}$")
    else:
        ax.set_ylabel(f"{args.profile_ylabel}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5_data")
    parser.add_argument("index", type=int)
    parser.add_argument("sim_data")
    parser.add_argument("--rc-file", default=None)
    parser.add_argument("--colorbar", action="store_true", default=False)
    parser.add_argument("--colorbar-label", default=None)
    parser.add_argument("--apply-transforms", action="store_true", default=False)
    parser.add_argument("--hide-ticks", action="store_true", default=False)
    parser.add_argument("--xlabel", default="space")
    parser.add_argument("--ylabel", default="time")
    parser.add_argument("--save-fig", default=None)
    parser.add_argument("--with-initial-profile", action="store_true", default=False)
    parser.add_argument("--profile-ylabel", default=None)

    # need: labels

    args = parser.parse_args()

    setup_rcparams(args.rc_file)

    with h5py.File(args.sim_data) as sim_data, h5py.File(args.h5_data) as h5_data:
        # get meta data from h5 file for transforms
        meta_data = h5_data["meta_data"]
        cmdline = "".join(map(chr, meta_data["cmdline"][...]))
        spec_path = cmdline.split()[2]
        spec_path = os.path.join("".join(map(chr, meta_data["directory"][...])), spec_path)
        with open(spec_path) as spec_file:
            spec = json.load(spec_file)
        
        sysvars = list(spec["system"].keys())
        # get transforms here
        if args.apply_transforms and ("transforms" in spec):
            transforms = []
            transforms_dict = spec["transforms"]
            for varname, (shift, scale) in transforms_dict.items():
                transforms.append((sysvars.index(varname), shift, scale))
        else:
            transforms = None
        
        # get simulation data
        t = sim_data["t"][...]
        x = sim_data["x"][...]
        Y = sim_data["Y"][...]
        T = t[-1]
        L = x[-1]
        if transforms is not None:
            for index, shift, scale in transforms:
                Y[:, :, index] = Y[:, :, index] * scale - shift
    
    # plot all of the data
    figsize = plt.rcParams["figure.figsize"] if not args.with_initial_profile else (13, 6)
    fig, axes = plt.subplots(figsize=figsize, layout="constrained", ncols=1 + args.with_initial_profile)
    axes = np.atleast_1d(axes)
    
    kymo_plot(axes[-1], args, Y, L, T)
    
    if args.with_initial_profile:
        starting_profile_plot(axes[0], args, x, Y, sysvars)
    
    if args.save_fig is None:
        plt.show()
    else:
        fig.savefig(args.save_fig)

)TWIST";

static std::string plot_script;

namespace python::simviz
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
} // namespace python::simviz
