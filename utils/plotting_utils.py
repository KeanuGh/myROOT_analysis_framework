from typing import Tuple, List
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from matplotlib.colors import LogNorm

import src.config as config
from utils.variable_names import variable_data

# set plot style
plt.style.use([hep.style.ATLAS])


# ===============================
# ===== HISTOGRAM VARIABLES =====
# ===============================
def get_axis_labels(var_name: str | List[str], lepton: str = 'lepton') -> Tuple[str | None, str | None]:
    """Gets label for corresponding variable in axis labels dictionary"""
    if isinstance(var_name, list):
        # If a list is passed, check if all variables have the same name, otherwise throw warning and pick first
        try:
            names = [variable_data[var]['name'] for var in var_name]
        except KeyError:
            warn(f"Axis labels for {var_name} not found in label lookup dictionary. "
                 f"Falling back to default", UserWarning)
            return str(var_name), 'Entries'

        name_set = set(names)
        if len(name_set) > 1:
            w = f"Variables have different names: {names}. Picking {names[0]} as axis label"
            warn(w)
        var_name = var_name[0]

    # get name and units from data dictionary
    try:
        name = variable_data[var_name]['name']
        units = variable_data[var_name]['units']
    except KeyError:
        warn(f"Axis labels for {var_name} not found in label lookup dictionary. "
             f"Falling back to default", UserWarning)
        return var_name, 'Entries'

    # just the symbol(s) in latex math format if it exists
    symbol = name.split('$')[1] if len(name.split('$')) > 1 else name

    # convert x label string for correct lepton if applicable
    xlabel = name + (f" [{units}]" if units else '')
    try:
        xlabel %= lepton
    except TypeError:
        pass

    if variable_data[var_name]['tag'] == 'truth':
        # weird but it does the job
        ylabel = r'$\frac{d\sigma}{d' + symbol + r'}$ [fb' + (f' {units}' + '$^{-1}$]' if units else ']')
    else:
        ylabel = 'Entries'

    return xlabel, ylabel


def get_axis(bins: List[float] | Tuple[int, float, float], logbins: bool = False) -> bh.axis.Axis:
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
        var_name: List[str] | str,
        bins: tuple | list,
        lepton: str = None,
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        logx: bool = False,
        logy: bool = False,
        label: bool = True
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
    :param label: whether to add ATLAS label
    :return: changed axis (also works in place)
    """
    if logx: axis.semilogx()
    if logy: axis.semilogy()

    if isinstance(bins, tuple):  axis.set_xlim(bins[1], bins[2])
    elif isinstance(bins, list): axis.set_xlim(bins[0], bins[-1])
    else:
        raise TypeError("Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. "
                        f"Given input was {bins}")

    # set axis labels
    _xlabel, _ylabel = get_axis_labels(var_name, lepton)
    axis.set_xlabel(xlabel if xlabel else _xlabel)
    axis.set_ylabel(ylabel if ylabel else _ylabel)
    axis.minorticks_on()
    if label:
        hep.atlas.label(italic=(True, True), ax=axis, loc=0, llabel='Internal', rlabel=title)

    return axis


# ===============================
# ==== BUILDING HISTOGRAMS ======
# ===============================
def histplot_2d(
        var_x: pd.Series, var_y: pd.Series,
        xbins: tuple | list, ybins: tuple | list,
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
