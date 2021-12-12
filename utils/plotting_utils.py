import logging
import pickle as pkl
from typing import Tuple, Optional, Union, Iterable, List
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

import src.config as config
from utils.axis_labels import labels_xs
from utils.phys_vars import get_luminosity

logger = logging.getLogger('analysis')

# set plot style
plt.style.use([hep.style.ATLAS,
               {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
               {'errorbar.capsize': 5},
               {'axes.labelsize': 23},
               {'axes.labelpad': 23},
               ])


# TODO: OOP-ify all this

# ===============================
# ========== SCALINGS ===========
# ===============================
def scale_to_crosssection(hist: bh.Histogram, luminosity) -> bh.Histogram:
    """
    Scales histogram to cross-section. Currently undefined for multidimensional histograms.
    Also rescales weight if able
    """
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    factor = (luminosity * hist.axes[0].widths)
    hist /= factor
    return hist


def scale_by_bin_widths(hist: bh.Histogram) -> bh.Histogram:
    """
    Divides number of entries in bins by bin width. Also rescales weight if able.
    """
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    factor = hist.axes[0].widths
    hist /= factor
    return hist


def scale_hist(scaling: str, hist: bh.Histogram,
               lumi: Optional[float] = None, df: Optional[pd.DataFrame] = None) -> bh.Histogram:
    """Scales histogram by option either 'xs': cross-section, 'widths': by bin-width, or None"""
    if scaling == 'xs':
        if not lumi and not df:
            raise Exception("Must supply either luminosity or dataframe")
        elif not lumi:
            lumi = get_luminosity(df)
        hist = scale_to_crosssection(hist, luminosity=lumi)
    elif scaling == 'widths':
        hist = scale_by_bin_widths(hist)
    elif not scaling:
        pass
    else:
        raise ValueError("Scaling currently supports cross-section 'xs', by bin-width 'widths' or None")
    return hist


# ===============================
# ===== HISTOGRAM VARIABLES =====
# ===============================
def get_sumw2(hist: bh.Histogram) -> np.array:
    """Returns numpy array of sum of weights squared in oost-histogram histogram"""
    # variances are sum of weights squared in each histogram bin
    return hist.view().variance


def get_root_sumw2(hist: bh.Histogram) -> np.array:
    """Returns numpy array of root of sum of weights squared in boost-histogram histogram"""
    # variances are sum of weights squared in each histogram bin
    return np.sqrt(hist.view().variance)


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


def getbins(var_to_plot) -> Tuple[bool, Optional[tuple]]:
    """
    Returns special bins if variable input requires it. Returns None if not a special variable
    :param var_to_plot: variable to choose bins from
    :return: tuple (n_bins, start, stop) of bins or None
    """
    is_logbins = not any(map(var_to_plot.__contains__, config.not_log))
    if not is_logbins:
        # set bins for special variables
        sp_var = [sp_var for sp_var in config.not_log if sp_var in var_to_plot]
        if len(sp_var) != 1:
            raise Exception(f"Expected one matching variable for special binning. Got {sp_var}")
        return is_logbins, config.special_binning[sp_var[0]]
    else:
        return is_logbins, None


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
        raise TypeError(f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. Got {bins}")


def set_fig_1d_axis_options(axis: plt.axes,
                            var_name: str,
                            bins: Union[tuple, list],
                            lepton: str = None,
                            scaling: Optional[str] = None,
                            is_logbins: bool = True,
                            logy: bool = False,
                            logx: bool = False
                            ) -> plt.axes:
    """
    Sets my default axis options

    :param axis: axis to change
    :param var_name: name of variable being plotted
    :param bins: tuple of bins in y (n_bins, start, stop) or list of bin edge
    :param lepton: name of lepton for axis label
    :param scaling: scaling used in plot
    :param is_logbins: whether bins are logarithmic
    :param logx: bool whether to set log axis
    :param logy: whether to set logarithmic bins where appropriate
    :return: changed axis (also works in place)
    """
    # get axis labels (x label, y label)
    xlabel, ylabel = get_axis_labels(str(var_name), lepton)

    if logy:
        # log y axis, unless plotting Bjorken X
        if 'PDFinfo_X' not in var_name:
            axis.semilogy()

    # apply axis limits
    if is_logbins:  # set axis edge at 0
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
    axis.set_xlabel(xlabel)
    if scaling == 'xs':
        axis.set_ylabel(ylabel)
    elif scaling == 'widths':
        axis.set_ylabel('Entries / bin width')
    elif scaling is None:
        axis.set_ylabel('Entries')

    return axis


# ===============================
# ==== BUILDING HISTOGRAMS ======
# ===============================
def histplot_1d(var_x: pd.Series,
                bins: Union[tuple, list],
                fig_axis: plt.axes,
                yerr: Optional[Union[str, Iterable]] = 'sumw2',
                weights: pd.Series = None,
                lumi: Optional[float] = None,
                scaling: Optional[str] = None,
                label: Optional[str] = None,
                is_logbins: bool = True,
                n_threads: int = config.n_threads,
                **kwargs
                ) -> bh.Histogram:
    """
    Plots variable as histogram onto given figure axis

    :param var_x: 1D series (or iterable) of variable to plot
    :param weights: 1D series (or iterable) w/ same dimensions as var_x. '1' if not provided.
    :param bins: tuple of bins in y (n_bins, start, stop) or list of bin edges
    :param fig_axis: figure axis to plot onto
    :param yerr:| 1D series (or iterable) of y-errors.
                | if 'SumW2' (case-insensitive), calculates sumw2 errors
    :param lumi: luminostity of sample if scaling by cross-section
    :param scaling: | type of scaling applied to histogram. either:
                    | - 'xs': cross-section
                    | - 'widths': by bin-width,
                    | -  None: No scaling
    :param label: optional label for histogram (eg for legend)
    :param is_logbins: whether it should be binned logarithmically
    :param n_threads: multithreading: number of threads used to fill histogram
    :param kwargs: keyword arguments to be passed to plotting function (eg color, linewidth...)
    :return: histogram object
    """

    hist = bh.Histogram(get_axis(bins, is_logbins), storage=bh.storage.Weight())
    hist.fill(var_x, weight=weights, threads=n_threads)
    if scaling:
        hist = scale_hist(scaling, hist, lumi)

    # global_scale
    if isinstance(yerr, str):
        if yerr.lower() == 'sumw2':
            yerr = get_root_sumw2(hist)  # get sum of weights squared
        else:
            raise ValueError("possible y error string value(s): 'sumw2'")

    # plot
    hep.histplot(hist.view().value, bins=hist.axes[0].edges,
                 ax=fig_axis, yerr=yerr, label=label, **kwargs)

    return hist


def histplot_2d(var_x: pd.Series, var_y: pd.Series,
                xbins: Union[tuple, list], ybins: Union[tuple, list],
                weights: pd.Series,
                ax: plt.axes,
                fig: plt.figure,
                n_threads: int = config.n_threads,
                is_z_log: bool = True,
                # is_x_log: bool = True, is_y_log: bool = True,  # Perhaps add this in later
                is_square: bool = True,
                ) -> bh.Histogram:
    """
    Plots and prints out 2d histogram. Does not support axis transforms (yet!)

    :param var_x: pandas series of var to plot on x axis
    :param var_y: pandas series of var to plot on y axis
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


# ===============================
# ===== PLOTTING FUNCTIONS ======
# ===============================
def plot_1d_hist(
        df: pd.DataFrame,
        x: str,
        bins: Union[tuple, list],
        fig: plt.Figure,
        ax: plt.Axes,
        x_label: str = None,
        y_label: str = None,
        yerr: Union[str, Iterable] = None,
        scaling: str = None,
        is_logbins: bool = None,
        lepton: str = None,
        log_x: bool = False,
        log_y: bool = False,
        legend_label: str = None,
        weight_col: str = 'weight_mc',
        ) -> plt.figure:
    """Simple plotting function"""

    # check if variable needs to be specially binned
    if is_logbins is not None:
        _, alt_bins = getbins(x)
    else:
        is_logbins, alt_bins = getbins(x)
    if alt_bins:
        bins = alt_bins

    lumi = get_luminosity(df) if scaling else None

    histplot_1d(var_x=df[x], weights=df[weight_col],
                bins=bins, fig_axis=ax,
                yerr=yerr,
                scaling=scaling,
                lumi=lumi,
                label=legend_label,
                is_logbins=is_logbins,
                color='k', linewidth=2)

    ax = set_fig_1d_axis_options(axis=ax, var_name=x, bins=bins,
                                 scaling=scaling, is_logbins=is_logbins,
                                 logy=log_y, logx=log_x, lepton=lepton)
    if y_label: ax.set_ylabel(y_label)
    if x_label: ax.set_xlabel(x_label)

    return fig


def ratio_plot_1d(
        p: pd.Series,
        q: pd.Series,
        x_label: str,
        y_label: str,
        title: str,
        bins: Union[tuple, list],
        out_dir: str = None,
        is_logbins: bool = False,
        weight: pd.Series = None,
        log_x: bool = False,
        log_y: bool = False,
        to_file: bool = True,
        y_lim: tuple = None,
        **kwargs
        ) -> plt.figure:
    """Plot of bin-wise ratio between two variables"""

    ax_transform = bh.axis.transform.log if is_logbins else None

    hp = bh.Histogram(get_axis(bins, ax_transform))
    hp.fill(p, threads=config.n_threads, weight=weight)
    hq = bh.Histogram(get_axis(bins, ax_transform))
    hq.fill(q, threads=config.n_threads, weight=weight)

    hp /= hq

    fig, ax = plt.subplots()
    hep.histplot(hp.view(), bins=hp.axes[0].edges, ax=ax, label=title, **kwargs)
    ax = set_fig_1d_axis_options(axis=ax, var_name='', bins=bins,
                                 scaling=None, is_logbins=is_logbins,
                                 logy=log_y, logx=log_x)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if y_lim:
        ax.set_ylim(y_lim)
    hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=title)

    if to_file:
        hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=title)
        out_png_file = out_dir + f"{title}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        logger.info(f"Figure saved to {out_png_file}")

    return fig


# OLD FUNCTION
def plot_mass_slices(df: pd.DataFrame,
                     lepton: str,
                     xvar: str,
                     xbins: Union[tuple, list] = (100, 300, 10000),
                     logbins: bool = True,
                     logx: bool = False,
                     id_col: str = 'DSID',
                     weight_col: str = 'total_event_weight',
                     plot_label: str = '',
                     plot_path: str = None,
                     hist_path: str = None,
                     inclusive_dataset: pd.DataFrame = None,
                     to_pkl: bool = False
                     ) -> None:
    """Plots all mass slices as well as inclusive sample (in this case just all given slices together)"""
    fig, ax = plt.subplots()

    hists = dict()  # dictionary to hold mass slice histograms

    for dsid, mass_slice in df.groupby(id_col):
        hist = histplot_1d(var_x=mass_slice[xvar], weights=mass_slice[weight_col],
                           bins=xbins, yerr=None, fig_axis=ax, label=str(dsid),
                           is_logbins=logbins, scaling='widths')
        if to_pkl:
            hists[dsid] = hist

    if inclusive_dataset is not None:
        hist_inc = histplot_1d(var_x=inclusive_dataset[xvar], weights=inclusive_dataset[weight_col],
                               bins=xbins, fig_axis=ax, yerr=None, is_logbins=logbins,
                               scaling='widths', color='k', linewidth=2, label='Inclusive')
    else:
        hist_inc = histplot_1d(var_x=df[xvar], weights=df[weight_col],
                               bins=xbins, fig_axis=ax, yerr=None, is_logbins=logbins,
                               scaling='widths', color='k', linewidth=2, label='Inclusive')
    if to_pkl:
        hists['inclusive'] = hist_inc

    ax.legend(fontsize=10, ncol=2, loc='upper right')
    ax.semilogy()
    if logx:
        plt.semilogx()
    hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label)

    xlabel, ylabel = get_axis_labels(xvar, lepton)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(130, 11000)

    name = f"{xvar}_mass_slices_full"
    if to_pkl:
        with open(hist_path + plot_label + '_' + name + '.pkl', 'wb') as f:
            pkl.dump(hists, f)
            logger.info(f"Saved pickle file to {f.name}")
    path = plot_path + plot_label + '_' + name + '.png'
    fig.savefig(path, bbox_inches='tight')
    logger.info(f"Figure saved to {path}")
