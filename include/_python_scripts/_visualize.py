import os
import h5py
import warnings
import json
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from argparse import ArgumentParser
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.collections import LineCollection

SADDLE_NODE = 1
BRANCH_POINT = 2
TORUS_POINT = 3

_RENDERING = True

def _set_rendering(do_rendering):
    global _RENDERING
    _RENDERING = do_rendering


class CollocatorSpline:
    __slots__ = ("t", "y", "K", "P")
    
    def __init__(self, t, y, K, P):
        self.P = P
        self.t = t
        self.y = y[::len(self.P) + 1]
        self.K = K[:-1].reshape(-1, len(self.P) + 1, y.shape[1])[:, 1:, :]
    
    def __call__(self, x):
        s = len(self.P)
        t = self.t
        y = self.y
        P = self.P
        mask = np.searchsorted(self.t[:-1], x, side="right") - 1
        h = t[mask + 1] - t[mask]
        theta = (x - t[mask]) / h
        K = self.K[mask]
        z = np.tile(theta, (s, 1)).T.cumprod(axis=1).reshape(-1, s, 1)
        tmp = np.matmul(P, z)
        res = y[mask] + h[:, None] * np.matmul(K.transpose(0, 2, 1), tmp).reshape(-1, self.y.shape[1])
        return res
        

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


def print_rcparams():
    for key, value in plt.rcParams.items():
        if isinstance(value, (list, tuple)):
            print(f"{key}: {', '.join(map(str, value))}")
        elif isinstance(value, str) and value.startswith("#"):
            print(f"{key}: {value[1:]}")
        elif value is None:
            continue
        else:
            print(f"{key}: {value}")
    

def gen_l2_norms(data: h5py.File):
    n = len(data) - 2
    ncol = int(data["basic_info/nstages"][0])
    node = int(data["basic_info/node"][0])
    L0 = 2 * data["basic_info/spatial_period"][0]
    gauss, weights = leggauss(ncol)
    gauss = 0.5 * (gauss + 1)
    norms = []

    for i in range(n):
        sol = data[f"{i}"]
        if "norm_nv" in sol:
            norms.append(sol["norm_nv"][0])
            continue

        ngrid = int(sol["nnodes"][0])
        L = sol["p"][-1] * L0
        N = (ngrid - 1) * (ncol + 1) + 1
        indices = np.array(sorted(list(set(range(N)) - set(range(0, N, ncol + 1)))))

        h2 = (0.5 * L) * sol["h"][:]
        h2 = h2[:, None]
        y = sol["state_curr"][:-2].reshape(N, node)
        y = np.delete(y, 1, axis=1)

        integrand = (y[indices] * y[indices]).sum(axis=1)
        norm = np.sqrt(((h2 * integrand.reshape(ngrid - 1, ncol)) * weights).sum())

        norms.append(norm)

    return np.array(norms)


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


def index_apd_t_start(v, x, t, min_slope=0):
    vmax = v.max()
    vmin = v.min()
    thresh = vmin * (1 - t) + vmax * t
    thresh = 0.09
    dist_lo = np.inf
    j = -1
    for i in range(v.shape[0] - 1):
        dist = abs(v[i] - thresh)
        m = (v[i + 1] - v[i]) / (x[i + 1] - x[i])
        if (dist < dist_lo) and (m > min_slope):
            dist_lo = dist
            j = i
    return j


def index_apd_t_end(v, x, t, i0):
    vmax = v.max()
    vmin = v.min()
    thresh = vmin * (1 - t) + vmax * t
    thresh = 0.09
    dist_lo = np.inf
    j = -1
    for i in range(i0, v.shape[0] - 1):
        dist = abs(v[i] - thresh)
        m = (v[i + 1] - v[i]) / (x[i + 1] - x[i])
        if (dist < dist_lo) and (m < 0):
            dist_lo = dist
            j = i
    if (j == -1) or (j == v.shape[0] - 2):
        for i in range(i0 - 1):
            dist = abs(v[i] - thresh)
            m = (v[i + 1] - v[i]) / (x[i + 1] - x[i])
            if (dist < dist_lo) and (m < 0):
                dist_lo = dist
                j = i
    return j


def compute_apd_t(v, x, t, min_slope=0):
    vmax = v.max()
    vmin = v.min()
    thresh = vmin * (1 - t) + vmax * t
    thresh = 0.09

    i0 = index_apd_t_start(v, x, t, min_slope)
    i1 = index_apd_t_end(v, x, t, i0)
    # plt.figure()
    # plt.plot(x, v)
    # plt.plot(x[[i0, i1]], v[[i0, i1]], "ok")
    # plt.show()

    if v[i0] == thresh:
        x0s = x[i0]
    elif v[i0] < thresh:
        x0s = (thresh - v[i0]) * (x[i0 + 1] - x[i0]) / (v[i0 + 1] - v[i0]) + x[i0]
    else:
        x0s = (thresh - v[i0]) * (x[i0 - 1] - x[i0]) / (v[i0 - 1] - v[i0]) + x[i0]

    if v[i1] == thresh:
        x1s = x[i1]
    elif v[i1] < thresh:
        x1s = (thresh - v[i1]) * (x[i1 - 1] - x[i1]) / (v[i1 - 1] - v[i1]) + x[i1]
    else:
        x1s = (thresh - v[i1]) * (x[i1 + 1] - x[i1]) / (v[i1 + 1] - v[i1]) + x[i1]

    if x1s < x0s:
        x1s += (x[-1] - x[0]) - x0s
        return x1s

    return x1s - x0s


def compute_apd(v, vp, x, r, _index, _cs):
    thresh = os.environ.get("APD_THRESH", None)
    N = int(os.environ.get("APD_SAMPLE", 10000))
    if thresh is None:
        vmin = v.min()
        vmax = v.max()
        thresh = vmin * (1 - r) + vmax * r
    else:
        thresh = float(thresh)
    
    xdense = np.linspace(*x[[0, -1]], N)
    vdense = _cs(xdense)
    
    mask = vdense >= thresh
    ishift = mask.argmin()
    if ishift:
        vdense2 = np.roll(vdense, -ishift - 1)
        mask = np.roll(mask, -ishift - 1)
    
    istart = mask.argmax()
    istop = N - 1 - mask[::-1].argmax()
    
    return xdense[istop] - xdense[istart]


def gen_apds(data: h5py.File, apd_value):
    n = len(data) - 2
    ncol = int(data["basic_info/nstages"][0])
    node = int(data["basic_info/node"][0])
    L0 = 2 * data["basic_info/spatial_period"][0]
    P = data["basic_info/P"][...]
    gauss, _ = leggauss(ncol)
    gauss = 0.5 * (gauss + 1)
    APDs = []

    for i in range(n):
        sol = data[f"{i}"]
        ngrid = int(sol["nnodes"][0])
        L = sol["p"][-1] * L0
        N = (ngrid - 1) * (ncol + 1) + 1

        h2 = (0.5 * L) * sol["h"][:]
        h2 = h2[:, None]
        y = sol["state_curr"][:-2].reshape(N, node)

        t = sol["t"][...] * L
        t = np.append(np.append(t[:-1, None], t[:-1, None] + np.diff(t)[:, None] * gauss, axis=1).ravel(), t[-1])
        sp = CollocatorSpline(sol["t"][...], y, sol["ypold"][:-2].reshape(y.shape), P)
        _cs = lambda x: sp(x / L)[..., 0]
        APD = compute_apd(y[:, 0], y[:, 1], t, 1 - apd_value * 1e-2, _index=i, _cs=_cs)
        # print(f"{i:4d} : {APD:.8f}")
        APDs.append(APD / sol["p"][0])

    return np.array(APDs)


class FileIndexPartition:
    def __init__(self, h5_files):
        partitions = []
        offset = 0
        for i, file in enumerate(h5_files):
            nsols = len(file) - 2
            partitions.append((offset + i, nsols + offset + i))
            offset += nsols
        self.partitions = partitions

    def index(self, solution_index, throw=True):
        for i, (lo, hi) in enumerate(self.partitions):
            if lo <= solution_index < hi:
                return i
        if throw:
            raise RuntimeError("figure out")
        return -1


class DispersionCurve:
    def __init__(
        self,
        continuation_data,
        fignum=None,
        do_l2_norms=False,
        do_apds=False,
        apd_value=90,
        individual=False,
        no_stability=False,
        dispersion_mode="default",
        actual_spatial_period=False,
        spatial_period_units=None,
        wave_speed_units=None,
        wave_profile_units=None,
        center_wave=False,
        dispersion_labels=None,
        hide_bifurcation_points=False,
        fixed_solutions=None,
        starting_extents=None,
        dashed_unstable=False,
        apply_transforms=False,
        unstable_color="black",
        stable_color="red",
        stability_legend=False,
    ):
        self.cpaths = np.array([continuation_data], dtype=object).ravel()
        self.continuation_data = tuple(h5py.File(file) for file in self.cpaths)
        self.fip = FileIndexPartition(self.continuation_data)
        cmdline = "".join(map(chr, self.continuation_data[0]["meta_data/cmdline"][:]))
        self.profile_index = 0
        with open(cmdline.split()[2]) as spec_file:
            partial = r"\frac{\partial}{\partial \xi} "
            spec = json.load(spec_file)
            names = np.array(list(spec["system"].keys()), object)
            diff_idx = self.continuation_data[0]["basic_info/diffusion_indices"][:]
            names = np.insert(names, diff_idx + 1, [partial + name for name in names[diff_idx]])
            self.names = tuple(f"${{{name.replace('_', '}_{')}}}$" for name in names)

            if apply_transforms and ("transforms" in spec):
                transforms = []
                transforms_dict = spec["transforms"]
                sysvars = list(spec["system"].keys())
                for varname, (shift, scale) in transforms_dict.items():
                    transforms.append((sysvars.index(varname), shift, scale))
            else:
                transforms = None
            self.transforms = transforms

        if all(("0/spectrum" not in file) for file in self.continuation_data) or no_stability:
            self.stability_data = None
        else:
            self.stability_data = True

        if self.stability_data is None:
            if not individual:
                fig, (self.ax1, self.ax2) = plt.subplots(num=fignum, figsize=(9, 6), layout="constrained", nrows=2)
                self.figs = (fig,)
            else:
                fig1, self.ax1 = plt.subplots(layout="constrained")
                fig2, self.ax2 = plt.subplots(layout="constrained")
                self.figs = (fig1, fig2)
            self.spectrum_ax = None
        else:
            if not individual:
                fig = plt.figure(figsize=(12, 6), layout="constrained")
                self.figs = (fig,)
                gs = gridspec.GridSpec(2, 2, figure=fig)
                self.ax1 = fig.add_subplot(gs[0, 0])
                self.ax2 = fig.add_subplot(gs[1, 0])
                self.spectrum_ax = fig.add_subplot(gs[:, 1])
            else:
                fig1, self.ax1 = plt.subplots(layout="constrained")
                fig2, self.ax2 = plt.subplots(layout="constrained")
                fig3, self.spectrum_ax = plt.subplots(layout="constrained")
                self.figs = (fig1, fig2, fig3)

        self.do_l2_norms = do_l2_norms
        self.do_apds = do_apds
        self.apd_value = apd_value
        self.dispersion_mode, *self.dispersion_mode_colors = dispersion_mode.split(":")

        self.solution_index = 0
        self.initialized = False
        self.all_plots = []
        self.shift_activated = False
        self.digits_entered = []
        self.shown_solutions = [0]
        self.actual_spatial_period = actual_spatial_period
        self.spatial_period_units = spatial_period_units
        self.wave_speed_units = wave_speed_units
        self.wave_profile_units = wave_profile_units
        self.center_wave = center_wave
        self.dispersion_labels = dispersion_labels
        self.hide_bifurcation_points = hide_bifurcation_points
        self.dashed_unstable = dashed_unstable
        self.unstable_color = unstable_color
        self.stable_color = stable_color
        self.stability_legend = stability_legend

        if fixed_solutions is not None:
            self.fixed_solutions = tuple(int(index) for index in fixed_solutions)
        else:
            self.fixed_solutions = None

        self._init_dispersion()
        if self.fixed_solutions is not None:
            for index in self.fixed_solutions:
                index = ((index % self.nsolutions) + self.nsolutions) % self.nsolutions
                self._init_solution_plot(index)
            if len(self.fixed_solutions) == 0:
                for fig in self.figs[1:]:
                    plt.close(fig)
                self.figs = self.figs[:1]
        else:
            self._init_solution_plot()
            self._init_listeners()
        
        if starting_extents is not None:
            self._set_starting_extents(starting_extents)

        self.initialized = True
    
    def _set_starting_extents(self, extents):
        for item in extents:
            index, *extent = item.split(":")
            index = int(index)
            assert len(extent) == 4, "extent syntax is index:xlo:xhi:ylo:yhi"
            xlo, xhi, ylo, yhi = map(eval, extent)
            if index == 0:
                self.ax1.set_xlim(xlo, xhi)
                self.ax1.set_ylim(ylo, yhi)
            elif index == 1:
                self.ax2.set_xlim(xlo, xhi)
                self.ax2.set_ylim(ylo, yhi)
            elif (index == 2) and (self.spectrum_ax is not None):
                self.spectrum_ax.set_xlim(xlo, xhi)
                self.spectrum_ax.set_ylim(ylo, yhi)

    def reset(self):
        self.ax1.cla()
        self.ax2.cla()
        if self.stability_data is not None:
            self.spectrum_ax.cla()
        self.all_plots.clear()

        self.initialized = False
        self._init_dispersion()
        self._init_solution_plot(self.solution_index)
        self.initialized = True

    def _default_dispersion_colors(self, ax, alpha, c, stability):
        points = np.c_[alpha, c].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = ListedColormap([self.unstable_color, self.stable_color])
        bounds = [0, 0.5, 1]
        norm = BoundaryNorm(bounds, cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm, picker=10)
        lc.set_array(np.array(stability))
        self.disp_line = ax.add_collection(lc)
        lc.set_linewidth(2)
    
        if self.dashed_unstable:
            alpha_copy = alpha.copy()
            c_copy = c.copy()
            mask = (stability == 1) | (stability == -1)
            alpha_copy[mask] = np.nan
            c_copy[mask] = np.nan
            ax.plot(alpha_copy, c_copy, linestyle=(3, (3, 3)), color=ax.get_facecolor(), linewidth=lc.get_linewidth() * 1.05)
        
        if (stability != -1).any() and self.stability_legend:
            ax.plot([], [], color=self.stable_color, label="stable")
            ax.plot([], [], color=self.unstable_color, label="unstable")
            ax.legend()

    def _multi_colored_dispersion_colors(self, ax, alpha, c, stability):
        if not self.dispersion_mode_colors:
            raise RuntimeError("need to specify color list")
        colors = self.dispersion_mode_colors[0].split(",")
        if len(colors) < len(self.cpaths):
            raise RuntimeError(f"need to specify at least {len(self.cpaths)} colors")

        points = np.c_[alpha, c].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = ListedColormap(colors)
        N = max(2, len(self.cpaths))
        bounds = np.arange(N)
        bounds = np.insert(bounds.astype(float), bounds[1:], 0.5 * (bounds[:-1] + bounds[1:]))
        stability = [self.fip.index(i, False) for i in range(len(c))]
        norm = BoundaryNorm(bounds, max(2, cmap.N))
        lc = LineCollection(segments, cmap=cmap, norm=norm, picker=10)
        lc.set_array(np.array(stability))
        self.disp_line = ax.add_collection(lc)
        lc.set_linewidth(2)

        if self.dispersion_labels is None:
            return
        if len(self.dispersion_labels) != len(self.cpaths):
            print(self.dispersion_labels)
            raise RuntimeError(f"expected {len(self.dispersion_labels)} labels")

        for i, label in enumerate(self.dispersion_labels):
            ax.plot([], [], color=cmap(float(i)), label=label)
        ax.legend()

    def _color_mapped_dispersion_colors(self, ax, alpha, c, stability):
        if not self.dispersion_mode_colors:
            raise RuntimeError("need to specify color map")
        cmap = plt.get_cmap(self.dispersion_mode_colors[0])
        points = np.c_[alpha, c].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        N = len(self.cpaths)
        bounds = np.arange(N)
        bounds = np.insert(bounds.astype(float), bounds[1:], 0.5 * (bounds[:-1] + bounds[1:]))
        stability = [self.fip.index(i, False) for i in range(len(c))]
        lc = LineCollection(segments, cmap=cmap, picker=10, norm=Normalize(0, N - 1, clip=True))
        lc.set_array(np.array(stability))
        self.disp_line = ax.add_collection(lc)
        lc.set_linewidth(2)

        if self.dispersion_labels is None:
            return
        if len(self.dispersion_labels) != len(self.cpaths):
            print(self.dispersion_labels)
            raise RuntimeError(f"expected {len(self.dispersion_labels)} labels")

        for i, label in enumerate(self.dispersion_labels):
            _color = i / max(1, len(self.dispersion_labels) - 1)
            ax.plot([], [], color=cmap(_color), label=label)
        ax.legend()

    def _init_dispersion(self, ax=None):
        if ax is None:
            ax = self.ax1

        param = os.path.basename(self.cpaths[0]).split("-")[2]
        if param == "GNa":
            param = "g_Na"
        elif param == "eps":
            param = r"\varepsilon"
        if param == "sps":
            if self.actual_spatial_period:
                label = r"$\Lambda$"
                if self.spatial_period_units is not None:
                    label = f"{label} ({self.spatial_period_units})"
                ax.set_xlabel(label)
            else:
                ax.set_xlabel(r"$\Lambda / \Lambda_0$")
        else:
            param = "_".join(f"{{{item}}}" for item in param.split("_"))
            param = f"${param}$"
            ax.set_xlabel(param)
        if self.wave_speed_units is not None:
            ax.set_ylabel(f"wave speed ({self.wave_speed_units})")
        else:
            ax.set_ylabel("wave speed")

        c_all = []
        alpha_all = []
        stability_all = []
        solution_types = []
        for file in self.continuation_data:
            nsolutions = len(file) - 2
            pmask = file["basic_info/pmask"][...]

            stability = []
            c, alpha = np.empty((2, nsolutions))
            for k in range(nsolutions):
                c[k], alpha[k] = file[f"{k}/p"][...][pmask]
                solution_types.append(file[f"{k}/stype"][0])
                if self.stability_data and (f"{k}/spectrum" in file):
                    evals = get_eigenvalues(file, f"{k}/spectrum")
                    evals = evals[0] + 1j * evals[1]
                    evals = evals - evals[abs(evals).argmin()].real
                    stable = (evals.real <= 0).all()
                    stability.append(stable)
                else:
                    stability.append(-1)

            if self.do_l2_norms:
                c = gen_l2_norms(file)
                ax.set_ylabel(r"$\|u(\xi)\|_2$")
            elif self.do_apds:
                c = gen_apds(file, self.apd_value)
                if self.wave_speed_units is not None:
                    ax.set_ylabel(f"APD{self.apd_value} ({self.wave_speed_units})")
                else:
                    ax.set_ylabel(f"APD{self.apd_value}")
                    

            if c_all:
                c_all.append(np.nan)
                alpha_all.append(np.nan)
                stability_all.append(False)
                solution_types.append(-1)

            c_all.extend(c)
            alpha_all.extend(alpha)
            stability_all.extend(stability)

        # set bounds
        self.nsolutions = len(c_all)
        c = self.c = np.array(c_all)
        alpha = self.alpha = np.array(alpha_all)
        if (param == "sps") and self.actual_spatial_period:
            self.alpha *= file["basic_info/spatial_period"][0] * 2
        c_min = c[np.isfinite(c)].min()
        c_max = c[np.isfinite(c)].max()
        alpha_min = alpha[np.isfinite(alpha)].min()
        alpha_max = alpha[np.isfinite(alpha)].max()
        stability = np.array(stability_all)
        dx = alpha_max - alpha_min
        dy = c_max - c_min
        ax.set_xlim(alpha_min - 0.05 * dx, alpha_max + 0.05 * dx)
        ax.set_ylim(c_min - 0.05 * dy, c_max + 0.05 * dy)

        # set colors
        if self.dispersion_mode == "default":
            self._default_dispersion_colors(ax, alpha, c, stability)
        elif self.dispersion_mode == "multi":
            self._multi_colored_dispersion_colors(ax, alpha, c, stability)
        elif self.dispersion_mode == "cmap":
            self._color_mapped_dispersion_colors(ax, alpha, c, stability)
        else:
            raise RuntimeError(f"dispersion mode '{self.dispersion_mode}' is not valid")

        if not self.hide_bifurcation_points:
            solution_types = np.array(solution_types)
            ax.scatter(alpha[solution_types == SADDLE_NODE], c[solution_types == SADDLE_NODE], c="violet", zorder=13, marker="s")
            ax.scatter(alpha[solution_types == TORUS_POINT], c[solution_types == TORUS_POINT], c="limegreen", zorder=13, marker="o")
            ax.scatter(alpha[solution_types == BRANCH_POINT], c[solution_types == BRANCH_POINT], c="violet", zorder=13, marker="^")

    def _init_solution_plot(self, index=0):
        ax1 = self.ax1
        ax2 = self.ax2
        file = self.continuation_data[self.fip.index(index)]
        index -= self.fip.partitions[self.fip.index(index)][0]

        s = int(file["basic_info/nstages"][...][0])
        self.node = int(file["basic_info/node"][...][0])
        self.gauss = 0.5 * (np.polynomial.legendre.leggauss(s)[0] + 1)

        (self.sol_dot,) = ax1.plot(self.alpha[:1], self.c[:1], "o")
        (self.sol_line,) = ax2.plot([], [])
        self.all_plots.append(self.sol_dot)
        self.all_plots.append(self.sol_line)
        if self.actual_spatial_period:
            label = "$x$"
            if self.spatial_period_units is not None:
                label = f"{label} ({self.spatial_period_units})"
            ax2.set_xlabel(label)
        else:
            ax2.set_xlabel(r"$\xi$")
        if self.stability_data is not None:
            self.spectrum_points = self.spectrum_ax.scatter
            self.spectrum_sc = plot_spectrum(
                f"{index}/spectrum", file, ax=self.spectrum_ax, color=self.sol_line.get_color(), set_lim=not self.initialized
            )
        self._plot_solution_at_index(index)

    def _update_inactive_solutions(self):
        node = self.node

        for index, line in zip(self.shown_solutions[:-1], self.ax2.get_lines()):
            file = self.continuation_data[self.fip.index(index)]
            t = file[f"{index}/t"][...]
            t = np.append(np.append(t[:-1, None], t[:-1, None] + np.diff(t)[:, None] * self.gauss, axis=1).ravel(), t[-1])
            y = file[f"{index}/state_curr"][self.profile_index :: node][: len(t)]
            line.set_data(t, y)

    def _plot_solution_at_index(self, index):
        node = self.node
        index = ((index % self.nsolutions) + self.nsolutions) % self.nsolutions
        file = self.continuation_data[self.fip.index(index)]
        sol_index = index - self.fip.partitions[self.fip.index(index)][0]

        t = file[f"{sol_index}/t"][...]
        t = np.append(np.append(t[:-1, None], t[:-1, None] + np.diff(t)[:, None] * self.gauss, axis=1).ravel(), t[-1])
        if self.actual_spatial_period:
            t *= file["basic_info/spatial_period"][0] * 2
            if file["basic_info/pmask"][1] == (file["basic_info/np"][0] - 1):
                t *= self.alpha[index] / t[-1]
            if self.center_wave:
                t -= t[-1] * 0.5
        y = file[f"{sol_index}/state_curr"][self.profile_index :: node][: len(t)]
        # apply transforms
        if self.transforms is not None:
            for index, shift, scale in self.transforms:
                if index == self.profile_index:
                    y = y * scale - shift
                    break

        self.sol_line.set_data(t, y)
        self.sol_dot.set_data(self.alpha[[index]], self.c[[index]])

        self.ax2.relim()
        self.ax2.autoscale_view(True, True, True)
        ylabel = self.names[self.profile_index]
        if self.wave_profile_units is not None:
            ylabel = f"{ylabel} ({self.wave_profile_units})"
        self.ax2.set_ylabel(ylabel)

        if self.stability_data is not None:
            if f"{sol_index}/spectrum" in file:
                x, y = get_eigenvalues(file, f"{sol_index}/spectrum")
            else:
                x, y = [], []
            self.spectrum_sc.set_offsets(np.c_[x, y])
            self.spectrum_ax.relim()
            self.ax2.autoscale_view(True, True, True)

        if self.initialized:
            self.ax1.draw_artist(self.ax1.patch)
            self.ax2.draw_artist(self.ax2.patch)
            self.ax1.draw_artist(self.disp_line)
            self.ax2.draw_artist(self.ax2.yaxis)

            for i in range(0, len(self.all_plots) - 2, 2):
                self.ax1.draw_artist(self.all_plots[i + 0])
                self.ax2.draw_artist(self.all_plots[i + 1])

            self.ax1.draw_artist(self.sol_dot)
            self.ax2.draw_artist(self.sol_line)

            if self.stability_data is not None:
                self.spectrum_ax.draw_artist(self.spectrum_ax.patch)
                self.spectrum_ax.draw_artist(self.spectrum_ax.xaxis)
                self.spectrum_ax.draw_artist(self.spectrum_ax.yaxis)
                self.spectrum_ax.draw_artist(self.spectrum_ax.spines["left"])
                self.spectrum_ax.draw_artist(self.spectrum_ax.spines["bottom"])
                self.spectrum_ax.draw_artist(self.spectrum_sc)

            for fig in self.figs:
                if _RENDERING:
                    fig.canvas.update()
                    fig.canvas.flush_events()
            if _RENDERING:
                plt.pause(1e-4)
        else:
            for fig in self.figs:
                if _RENDERING:
                    fig.canvas.draw_idle()

    def on_pick(self, event):
        self.solution_index = event.ind[0]
        self.shown_solutions[-1] = self.solution_index
        self._plot_solution_at_index(self.solution_index)
        for fig in self.figs:
            if _RENDERING:
                fig.canvas.draw_idle()

    def show(self):
        plt.show()

    def on_press(self, event):
        if (event.key == "up") or (event.key == "n"):
            self.solution_index += 1
            self.solution_index = ((self.solution_index % self.nsolutions) + self.nsolutions) % self.nsolutions
            if not np.isfinite(self.c[self.solution_index]):
                self.solution_index += 1
            self.shown_solutions[-1] = self.solution_index
            self._plot_solution_at_index(self.solution_index)
        elif (event.key == "down") or (event.key == "b"):
            self.solution_index -= 1
            self.solution_index = ((self.solution_index % self.nsolutions) + self.nsolutions) % self.nsolutions
            if not np.isfinite(self.c[self.solution_index]):
                self.solution_index -= 1
            self.shown_solutions[-1] = self.solution_index
            self._plot_solution_at_index(self.solution_index)
        elif event.key == "tab":
            self.shown_solutions.append(self.solution_index)
            self._init_solution_plot(self.solution_index)
            self._plot_solution_at_index(self.solution_index)
        elif event.key == "escape":
            self.reset()
        elif event.key == "i":
            file_index = self.fip.index(self.solution_index)
            index = self.solution_index - self.fip.partitions[file_index][0]
            print()
            print(f"file                   : {self.cpaths[file_index]}")
            print(f"solution index         : {index}")
            print(f"parameter value        : {self.alpha[self.solution_index]}")
            if self.stability_data is not None:
                wr, wi = self.spectrum_sc.get_offsets().T
                sizes = np.hypot(wr, wi)
                if sizes.size:
                    wr -= wr[sizes.argmin()]
                    mask = wr > 0
                    print(f"unstable eigenvalues   : {wr[mask] + 1j * wi[mask]!s}")
                    print(f"# unstable eigenvalues : {mask.sum()}")
                    print(f"approximate error      : {sizes.min()}")
        elif event.key == "shift":
            if self.shift_activated and self.digits_entered:
                self.profile_index = int("".join(self.digits_entered)) % len(self.names)
                self.digits_entered.clear()
                self._update_inactive_solutions()
                self._plot_solution_at_index(self.solution_index)
            else:
                self.shift_activated = True
                self.digits_entered.clear()
        elif event.key.isdigit():
            self.digits_entered.append(event.key)

    def _init_listeners(self):
        self.figs[0].canvas.mpl_connect("pick_event", self.on_pick)
        for fig in self.figs:
            fig.canvas.mpl_connect("key_press_event", self.on_press)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--rc-file", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--dispersion-mode", default="default")
    parser.add_argument("--l2", default=False, action="store_true")
    parser.add_argument("--apd", default=False, action="store_true")
    parser.add_argument("--individual", default=False, action="store_true")
    parser.add_argument("--no-stability", default=False, action="store_true")
    parser.add_argument("--apd-value", default=90, type=int)
    parser.add_argument("--actual-spatial-period", default=False, action="store_true")
    parser.add_argument("--center-wave", default=False, action="store_true")
    parser.add_argument("--spatial-period-units", default=None)
    parser.add_argument("--wave-speed-units", default=None)
    parser.add_argument("--wave-profile-units", default=None)
    parser.add_argument("--dispersion-labels", nargs="*", default=None)
    parser.add_argument("--hide-bifurcation-points", default=False, action="store_true")
    parser.add_argument("--fixed-solutions", nargs="*", default=None)
    parser.add_argument("--print-rc-file", default=False, action="store_true")
    parser.add_argument("--starting-extents", nargs="*", default=None)
    parser.add_argument("--dashed-unstable", default=False, action="store_true")
    parser.add_argument("--apply-transforms", default=False, action="store_true")
    parser.add_argument("--unstable-color", default="black")
    parser.add_argument("--stable-color", default="red")
    parser.add_argument("--stability-legend", default=False, action="store_true")

    args = parser.parse_args()

    setup_rcparams(args.rc_file)
    
    if args.print_rc_file:
        print_rcparams()
    
    if not args.files:
        exit()
    
    curve = DispersionCurve(
        args.files,
        do_l2_norms=args.l2,
        do_apds=args.apd,
        apd_value=args.apd_value,
        individual=args.individual,
        no_stability=args.no_stability,
        dispersion_mode=args.dispersion_mode,
        actual_spatial_period=args.actual_spatial_period,
        spatial_period_units=args.spatial_period_units,
        wave_speed_units=args.wave_speed_units,
        wave_profile_units=args.wave_profile_units,
        center_wave=args.center_wave,
        dispersion_labels=args.dispersion_labels,
        hide_bifurcation_points=args.hide_bifurcation_points,
        fixed_solutions=args.fixed_solutions,
        starting_extents=args.starting_extents,
        dashed_unstable=args.dashed_unstable,
        apply_transforms=args.apply_transforms,
        stable_color=args.stable_color,
        unstable_color=args.unstable_color,
        stability_legend=args.stability_legend,
    )
    if args.save is not None:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        if args.individual and (len(curve.figs) > 1):
            base, ext = os.path.splitext(args.save)
            for i, fig in enumerate(curve.figs):
                fig.savefig(f"{base}-{i}{ext}")
        else:
            curve.figs[0].savefig(args.save)

    else:
        curve.show()
