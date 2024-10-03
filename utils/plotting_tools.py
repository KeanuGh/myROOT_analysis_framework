from typing import Sequence, TypedDict
from warnings import warn

import ROOT
import boost_histogram as bh
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import pandas as pd  # type: ignore
from matplotlib import ticker
from matplotlib.colors import LogNorm  # type: ignore

from src.histogram import Histogram1D
from utils.variable_names import variable_data

# set plot style
plt.style.use([hep.style.ATLAS])


class PlotOpts(TypedDict):
    """Per-histogram options for plotting"""

    vals: list[str | Histogram1D | ROOT.TH1]
    hists: list[Histogram1D]
    datasets: list[str]
    systematics: list[str]
    selections: list[str]
    labels: list[str]
    colours: list[str]


# ===============================
# ===== HISTOGRAM VARIABLES =====
# ===============================
def get_axis_labels(
    var_name: str | Sequence[str], lepton: str | None = "lepton", diff_xs: bool = False
) -> tuple[str | None, str | None]:
    """Gets label for corresponding variable in axis labels dictionary"""
    if not isinstance(var_name, str):
        new_varname = ""
        # pick first one that appears in variable data dictionary
        for var in var_name:
            if var[:4] == "cut_":  # cut out prefix if it exists
                var = var[4:]

            if isinstance(var, str) and (var in variable_data):
                new_varname = var
                break

        if new_varname:
            var_name = new_varname
        else:
            var_name = var_name[0]

    if var_name[:4] == "cut_":  # cut out prefix if it exists
        var_name = var_name[4:]
    # get name and units from data dictionary
    try:
        name = variable_data[var_name]["name"]
        units = variable_data[var_name]["units"]
    except (KeyError, TypeError):
        return var_name, "Entries"

    # just the symbol(s) in latex math format if it exists
    symbol = name.split("$")[1] if len(name.split("$")) > 1 else name

    # convert x label string for correct lepton if applicable
    xlabel = name + (f" [{units}]" if units else "")
    try:
        xlabel %= lepton
    except TypeError:
        pass

    if variable_data[var_name]["tag"] == "truth":
        # weird but it does the job
        if diff_xs:
            ylabel = (
                r"$\frac{d\sigma}{d"
                + symbol
                + r"}$ [pb"
                + ((f" {units}" + "$^{-1}$]") if units else "]")
            )
        else:
            ylabel = r"$d\sigma$ [pb]"
    else:
        ylabel = "Entries"

    return xlabel, ylabel


def get_axis(
    bins: Sequence[float | int] | tuple[int, float, float], logbins: bool = False
) -> bh.axis.Axis:
    """
    Returns the correct type of boost-histogram axis based on the input bins.

    :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                 In the first case returns an axis of type Regular(), otherwise of type Variable().
                 Raises error if not formatted in one of these ways.
    :param logbins: whether logarithmic binnins
    """
    transform = bh.axis.transform.log if logbins else None
    if isinstance(bins, tuple):
        if len(bins) != 3:
            raise ValueError("Tuple of bins should be formatted like (n_bins, start, stop).")
        return bh.axis.Regular(*bins, transform=transform)
    elif isinstance(bins, list):
        if transform is not None:
            warn("Transforms cannot be passed to variable bin types")
        return bh.axis.Variable(bins)
    else:
        raise TypeError(
            f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. Got {bins}"
        )


def set_axis_options(
    per_hist_vars: PlotOpts,
    ax: plt.Axes,
    ratio_ax: plt.Axes | None = None,
    title: str = "",
    ratio_label: str = "",
    scale_by_bin_width: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    logx: bool = False,
    logy: bool = False,
    x_axlim: tuple[float, float] | None = None,
    y_axlim: tuple[float, float] | None = None,
) -> None:
    """Set axis options for plot"""

    # let mplhep handle the easy stuff
    hep.atlas.label(italic=(True, True, False), ax=ax, loc=0, llabel="Internal", rlabel=title)

    # get axis labels from variable names if possible
    # it'll error out when plotting if the histogram edges aren't equal
    bin_range = (
        per_hist_vars["hists"][0].bin_edges[0],
        per_hist_vars["hists"][0].bin_edges[-1],
    )
    if (
        all(isinstance(val, str) for val in per_hist_vars["vals"])
        and len(val_name := set(per_hist_vars["vals"])) == 1
    ):
        val_name = next(iter(val_name))
        if val_name in variable_data:
            _xlabel, _ylabel = get_axis_labels(val_name, diff_xs=scale_by_bin_width)
            if not xlabel:
                xlabel = _xlabel
            if not ylabel:
                ylabel = _ylabel

    # Main plot yaxis options
    if y_axlim:
        ax.set_ylim(*y_axlim)
    if logy:
        ax.semilogy()
    else:
        # if there are no negative yvalues, set limit at origin
        ax.set_ylim(bottom=0)
    if ylabel:
        ax.set_ylabel(ylabel)

    if x_axlim:
        ax.set_xlim(*x_axlim)
    else:
        ax.set_xlim(*bin_range)

    # set some xaxis options only on bottom xaxis (ie ratio plot if it exists else regular plot)
    axis_ = ratio_ax if (ratio_ax is not None) else ax
    if logx:
        axis_.semilogx()
        axis_.xaxis.set_minor_formatter(ticker.LogFormatter())
        axis_.xaxis.set_major_formatter(ticker.LogFormatter())
    if xlabel:
        axis_.set_xlabel(xlabel)

    if ratio_ax is not None:
        if len(per_hist_vars["hists"]) > 2:  # don't show legend if there's only two plots
            ratio_ax.legend(fontsize=10, loc=1)
        ratio_ax.set_ylabel(ratio_label)

        if x_axlim:
            ratio_ax.set_xlim(*x_axlim)
        else:
            ratio_ax.set_xlim(*bin_range)

        # just in case (I do not trust matplotlib)
        ax.set_xticklabels([])
        ax.set_xlabel("")


# ===============================
# ==== BUILDING HISTOGRAMS ======
# ===============================
def histplot_2d(
    var_x: pd.Series,
    var_y: pd.Series,
    xbins: Sequence[float | int],
    ybins: Sequence[float | int],
    weights: pd.Series,
    ax: plt.axes,
    fig: plt.figure,
    n_threads: int = -1,
    is_z_log: bool = True,
    is_square: bool = True,
) -> bh.Histogram:
    """
    Plots and prints out 2d histogram. Does not support axis transforms (yet!)

    :param var_x: pandas series of var to plot on x-axis
    :param var_y: pandas series of var to plot on y-axis
    :param xbins: tuple of bins in x (n_bins, start, stop) or list of bin edges
    :param ybins: tuple of bins in y (n_bins, start, stop) or list of bin edges
    :param weights: series of weights to apply to axes
    :param ax: axis to plot on
    :param fig: figure to plot on (for colourbar)
    :param n_threads: number of threads for filling
    :param is_z_log: whether z-axis should be scaled logarithmically
    :param is_square: whether to set square aspect ratio
    :return: histogram
    """
    # setup and fill histogram
    hist_2d = bh.Histogram(get_axis(xbins), get_axis(ybins))
    hist_2d.fill(var_x, var_y, weight=weights, threads=n_threads)

    if is_z_log:
        norm = LogNorm()
    else:
        norm = None

    # setup mesh differently depending on bh storage
    if hasattr(hist_2d.view(), "value"):
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().value.T, norm=norm)
    else:
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().T, norm=norm)

    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    if is_square:  # square aspect ratio
        ax.set_aspect(1 / ax.get_data_ratio())

    return hist_2d
