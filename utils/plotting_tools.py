from dataclasses import dataclass
from typing import Tuple, Sequence, List, Dict
from warnings import warn

import ROOT  # type: ignore
import boost_histogram as bh
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from numpy.typing import ArrayLike

from utils.variable_names import variable_data

# set plot style
plt.style.use([hep.style.ATLAS])

default_mass_bins = [
    130,
    140.3921,
    151.6149,
    163.7349,
    176.8237,
    190.9588,
    206.2239,
    222.7093,
    240.5125,
    259.7389,
    280.5022,
    302.9253,
    327.1409,
    353.2922,
    381.5341,
    412.0336,
    444.9712,
    480.5419,
    518.956,
    560.4409,
    605.242,
    653.6246,
    705.8748,
    762.3018,
    823.2396,
    889.0486,
    960.1184,
    1036.869,
    1119.756,
    1209.268,
    1305.936,
    1410.332,
    1523.072,
    1644.825,
    1776.311,
    1918.308,
    2071.656,
    2237.263,
    2416.107,
    2609.249,
    2817.83,
    3043.085,
    3286.347,
    3549.055,
    3832.763,
    4139.151,
    4470.031,
    4827.361,
    5213.257,
]


@dataclass
class PlottingOptionItem:
    bins: Tuple[int, float, float] | List[float]
    weight_col: str


default_plotting_options: Dict[str, PlottingOptionItem]


# ===============================
# ===== HISTOGRAM VARIABLES =====
# ===============================
def get_TH1_bins(
    bins: List[float] | Tuple[int, float, float] | bh.axis.Axis, logbins: bool = False
) -> Tuple[int, list | ArrayLike] | Tuple[int, float, float]:
    """Format bins for TH1 constructor"""
    if isinstance(bins, bh.axis.Axis):
        return bins.size, bins.edges

    elif hasattr(bins, "__iter__"):
        if logbins:
            if not isinstance(bins, tuple) or (len(bins) != 3):
                raise ValueError(
                    "Must pass tuple of (nbins, xmin, xmax) as bins to calculate logarithmic bins"
                )
            return bins[0], np.geomspace(bins[1], bins[2], bins[0] + 1)  # type: ignore

        if len(bins) == 3 and isinstance(bins, tuple):
            return bins
        else:
            return len(bins) - 1, np.array(bins)

    raise ValueError("Bins should be list of bin edges or tuple like (nbins, xmin, xmax)")


def get_axis_labels(
    var_name: str | Sequence[str], lepton: str | None = "lepton", diff_xs: bool = False
) -> Tuple[str | None, str | None]:
    """Gets label for corresponding variable in axis labels dictionary"""
    if not isinstance(var_name, str):
        new_varname = ""
        # pick first one that appears in variable data dictionary
        for var in var_name:
            if var[:4] == "cut_":  # cut out prefix if it exists
                var = var[4:]

            if var in variable_data:
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
    except KeyError:
        warn(
            f"Axis labels for {var_name} not found in label lookup dictionary. "
            f"Falling back to default",
            UserWarning,
        )
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
    bins: Sequence[float | int] | Tuple[int, float, float], logbins: bool = False
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
    axis: plt.axes,
    var_name: Sequence[str] | str,
    strict_xax_lims: bool = True,
    strict_yax_lims: bool = False,
    lepton: str = "",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    logx: bool = False,
    logy: bool = False,
    label: bool = True,
    diff_xs: bool = False,
) -> plt.axes:
    """
    Sets my default axis options

    :param axis: axis to change
    :param var_name: name of variable being plotted
    :param strict_xax_lims: whether to force the x-axis limits to the data limits
    :param strict_yax_lims: whether to force the y-axis limits to the data limits
    :param lepton: name of lepton for axis label
    :param xlabel: x label
    :param ylabel: y label
    :param title: plot title
    :param logx: bool whether to set log axis
    :param logy: whether to set logarithmic bins where appropriate
    :param label: whether to add ATLAS label
    :param diff_xs: set yaxis label to differential cross-section
    :return: changed axis (also works in place)
    """
    if logx:
        axis.semilogx()
    if logy:
        axis.semilogy()

    # restrict axes to extent of data. if statement is there to skip blank/guide lines
    if strict_xax_lims:
        xmin = min(
            [
                min(line.get_xdata(), default=np.inf)
                for line in axis.lines
                if len(line.get_xdata()) > 2
            ]
        )
        xmax = max(
            [
                max(line.get_xdata(), default=-np.inf)
                for line in axis.lines
                if len(line.get_xdata()) > 2
            ]
        )
        axis.set_xlim(xmin, xmax)

    if strict_yax_lims:
        ymin = min(
            [
                min(line.get_ydata(), default=np.inf)
                for line in axis.lines
                if len(line.get_xdata()) > 2
            ]
        )
        ymax = max(
            [
                max(line.get_ydata(), default=-np.inf)
                for line in axis.lines
                if len(line.get_xdata()) > 2
            ]
        )
        axis.set_ylim(ymin, ymax)

    # set axis labels
    _xlabel, _ylabel = get_axis_labels(var_name, lepton, diff_xs)
    axis.set_xlabel(xlabel if xlabel else _xlabel)
    axis.set_ylabel(ylabel if ylabel else _ylabel)
    if label:
        hep.atlas.label(italic=(True, True), ax=axis, loc=0, llabel="Internal", rlabel=title)

    return axis


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
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().value.T, norm=norm)  # type: ignore
    else:
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().T, norm=norm)

    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    if is_square:  # square aspect ratio
        ax.set_aspect(1 / ax.get_data_ratio())

    return hist_2d
