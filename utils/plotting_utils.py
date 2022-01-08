from typing import Tuple, Optional, Union, List
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from matplotlib.colors import LogNorm

import src.config as config
from utils.axis_labels import labels_xs

# set plot style
plt.style.use([hep.style.ATLAS])


# TODO: migrate all these to use new histogram

# ===============================
# ===== HISTOGRAM VARIABLES =====
# ===============================
def get_axis_labels(var_name: str, lepton: str = 'lepton') -> Tuple[Optional[str], Optional[str]]:
    """Gets label for corresponding variable in axis labels dictionary"""
    if var_name in labels_xs:
        xlabel = labels_xs[var_name]['xlabel']
        ylabel = labels_xs[var_name]['ylabel']

        # convert x label string for correct lepton if applicable
        try:
            xlabel %= lepton
        except TypeError:
            pass

    else:
        warn(f"Axis labels for {var_name} not found in label lookup dictionary. "
             f"They will be left blank.", UserWarning)
        xlabel = None
        ylabel = None
    return xlabel, ylabel


def get_axis(bins: Union[List[float], Tuple[int, float, float]],
             logbins: bool = False,
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
            f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. Got {bins}")


def set_axis_options(
        axis: plt.axes,
        var_name: str,
        bins: Union[tuple, list],
        lepton: str = None,
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        logx: bool = False,
        logy: bool = False,
) -> plt.axes:
    """
    Sets my default axis options

    :param axis: axis to change
    :param var_name: name of variable being plotted
    :param bins: tuple of bins in y (n_bins, start, stop) or list of bin edge
    :param lepton: name of lepton for axis label
    :param xlabel: x label
    :param ylabel: y label
    :param title: plot title
    :param logx: bool whether to set log axis
    :param logy: whether to set logarithmic bins where appropriate
    :return: changed axis (also works in place)
    """
    # get axis labels (x label, y label)
    _xlabel, _ylabel = get_axis_labels(str(var_name), lepton)

    if logy:
        axis.semilogy()

    if isinstance(bins, tuple):
        axis.set_xlim(bins[1], bins[2])
    elif isinstance(bins, list):
        axis.set_xlim(bins[0], bins[-1])
    else:
        raise TypeError("Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. "
                        f"Given input was {bins}")
    if logx:
        axis.semilogx()

    # set axis labels
    axis.set_xlabel(xlabel if xlabel else _xlabel)
    axis.set_ylabel(ylabel if ylabel else _ylabel)
    hep.atlas.label(italic=(True, True), ax=axis, loc=0, llabel='Internal', rlabel=title)

    return axis


# ===============================
# ==== BUILDING HISTOGRAMS ======
# ===============================
def histplot_2d(
        var_x: pd.Series, var_y: pd.Series,
        xbins: Union[tuple, list], ybins: Union[tuple, list],
        weights: pd.Series,
        ax: plt.axes,
        fig: plt.figure,
        n_threads: int = config.n_threads,
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
    if hasattr(hist_2d.view(), 'value'):
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().value.T, norm=norm)
    else:
        mesh = ax.pcolormesh(*hist_2d.axes.edges.T, hist_2d.view().T, norm=norm)

    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    if is_square:  # square aspect ratio
        ax.set_aspect(1 / ax.get_data_ratio())

    return hist_2d
