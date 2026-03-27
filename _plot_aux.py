import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from json import load
from argparse import ArgumentParser
from pprint import pprint

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["axes.labelsize"] = "x-large"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["blue", "red", "black", "magenta", "cyan", "green", "yellow",
                                                    "black", "purple", "pink", "brown", "orange", "teal",
                                                    "coral", "lightblue", "lime", "lavender", "turquoise",
                                                    "darkgreen", "tan", "salmon", "gold"])


def subplots_centered(nrows, ncols, figsize, nax, row2center=-1, **kwargs):
    if not (nax < nrows * ncols):
        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, **kwargs, layout="constrained")
        axes = np.atleast_1d(axes)
        return fig, axes.ravel()

    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = []

    m = nax % ncols
    m = range(1, ncols + 1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m * ncols, figure=fig)
    row2center = -1

    for i in range(0, nax):
        row = i // ncols
        col = i % ncols

        if row == (row2center % nrows):  # center only last row
            off = int(m * (ncols - nax % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m * col + off: m * (col + 1) + off])
        axs.append(ax)
    return fig, np.array(axs)


def eval_aux(json_data, cont_data, index, aux_map):
    # get base solution data
    node = int(cont_data["basic_info/node"][0])
    ncol = int(cont_data["basic_info/nstages"][0])
    didx = cont_data["basic_info/diffusion_indices"][...]
    gauss = (0.5 * (np.polynomial.legendre.leggauss(ncol)[0] + 1))
    y = cont_data[f"{index}/state_curr"][:-2]
    p = cont_data[f"{index}/p"][1:-1]
    t = cont_data[f"{index}/t"][...]
    t = np.append(np.append(t[:-1, None], t[:-1, None] + np.diff(t)
                            [:, None] * gauss, axis=1).ravel(), t[-1])
    y = y.reshape(t.shape[0], node)

    # load json data
    system = json_data["system"]
    params = json_data["params"]
    system_vars = list(system.keys())
    diffusion = json_data["diffusion"]
    to_delete = np.array(didx) + np.arange(1, len(didx) + 1)
    y = np.delete(y, to_delete, axis=1)
    if "transforms" in json_data:
        transforms = json_data["transforms"]
        for var, (shift, scale) in transforms.items():
            k = system_vars.index(var)
            y[:, k] = y[:, k] * scale - shift

    # start computing everything
    local_stuff = {}
    for name in dir(np):
        local_stuff[name] = getattr(np, name)
    for i, name in enumerate(params):
        local_stuff[name] = p[i]
    if "constants" in json_data:
        local_stuff.update(**json_data["constants"])
    for i, var in enumerate(system_vars):
        local_stuff[var] = y[:, i]

    for aux_name, aux_expr in aux_map.items():
        local_stuff[aux_name] = np.atleast_1d(eval(aux_expr, globals(), local_stuff))
        if len(local_stuff[aux_name]) == 1:
            local_stuff[aux_name] = np.full_like(t, local_stuff[aux_name][0])
    return t, local_stuff


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-i", "--indices", type=int, nargs="+")
    parser.add_argument("-w", "--which", nargs="+")
    parser.add_argument("-l", "--labels", nargs="+", default=None)
    parser.add_argument("--no-labels", action="store_true", default=False)
    args = parser.parse_args()

    path = args.path
    indices = args.indices
    which_aux = args.which
    labels = args.labels

    with h5py.File(path) as file:
        nsols = len(file) - 2
        cmdline = "".join(map(chr, file["meta_data/cmdline"]))[:-1]
        json_path = cmdline.split()[2]

        naux = len(which_aux)
        nrows = int((naux // 3) + (naux % 3 != 0))
        fig, axes = subplots_centered(nrows, min(3, naux), (10, (7 / 3) * nrows), naux)

        indices = [((index % nsols) + nsols) % nsols for index in indices]
        if labels is not None:
            if len(labels) != len(indices):
                f"{len(indices)} indices were specified but {len(labels)} labels were"
                exit(1)
        else:
            labels = [f"soln. {index}" for index in indices]

        with open(json_path) as json_file:
            json_data = load(json_file)
            aux_map = json_data["aux"]

            for j, index in enumerate(indices):
                t, aux_data = eval_aux(json_data, file, index, aux_map)
                for i, aux_name in enumerate(which_aux):
                    axes[i].plot(t, aux_data[aux_name], label=labels[j])
            for ax, aux_name in zip(axes, which_aux):
                one, *two = aux_name.split("_")
                if two:
                    two = "{" + "_".join(two) + "}"
                    aux_name = f"${one}_{two}$"
                else:
                    aux_name = f"${one}$"

                ax.set_ylabel(aux_name)
                ax.set_xlabel(r"$\xi$")
                ax.grid()
                if not args.no_labels:
                    ax.legend()

            plt.show()
