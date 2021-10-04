import pickle as pkl
from typing import Tuple, Optional, List, OrderedDict, Union, Iterable
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

import analysis.config as config
from utils.axis_labels import labels_xs
from utils.dataframe_utils import get_luminosity, cut_on_cutgroup

# set plot style
plt.style.use([hep.style.ATLAS,
               {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
               {'errorbar.capsize': 5},
               {'axes.labelsize': 23},
               {'axes.labelpad': 23},
               ])


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
               lumi: Optional = None, df: Optional[pd.DataFrame] = None) -> bh.Histogram:
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


def get_axis(bins: Union[tuple, list],
             transform: bh.axis.transform = None
             ) -> bh.axis.Axis:
    """
    Returns the correct type of boost-histogram axis based on the input bins.

    :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                 In the first case returns an axis of type Regular(), otherwise of type Variable().
                 Raises error if not formatted in one of these ways.
    :param transform: transform to pass to axis constructor
    """
    if isinstance(bins, tuple):
        if len(bins) != 3:
            raise ValueError("Tuple of bins should be formatted like (n_bins, start, stop).")
        return bh.axis.Regular(*bins, transform=transform)
    elif isinstance(bins, list):
        if transform is not None:
            warn("Transforms cannot be passed to variable bin types")
        return bh.axis.Variable(bins)
    else:
        raise TypeError("Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges.")


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

    # axis transformation
    if is_logbins:
        ax_transform = bh.axis.transform.log
    else:
        ax_transform = None

    hist = bh.Histogram(get_axis(bins, ax_transform), storage=bh.storage.Weight())
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
        x: Union[pd.Series, List[pd.Series]],
        bins: Union[tuple, list],
        title: str = None,
        filename: str = None,
        x_label: str = None,
        y_label: str = None,
        yerr: Union[str, Iterable] = None,
        weights: pd.Series = None,
        is_logbins: bool = False,
        log_x: bool = False,
        log_y: bool = False,
        to_file: bool = False,
        legend_label: Union[str, List[str]] = None,
        legend: bool = False,
        # to_pkl: bool = False,
) -> plt.figure:
    """Simple plotting function"""
    fig, ax = plt.subplots()

    if isinstance(x, list):
        if legend and not legend_label:
            raise ValueError("Legend labels required for making legend with multiple variables.")

        for i, s in enumerate(x):
            label = legend_label[i] if legend else None

            histplot_1d(var_x=s, weights=weights,
                        bins=bins, fig_axis=ax,
                        yerr=yerr,
                        scaling=None,
                        label=label,
                        is_logbins=is_logbins,
                        linewidth=2)
    elif isinstance(x, pd.Series):
        histplot_1d(var_x=x, weights=weights,
                    bins=bins, fig_axis=ax,
                    yerr=yerr,
                    scaling=None,
                    label=legend_label,
                    is_logbins=is_logbins,
                    color='k', linewidth=2)
    else:
        raise ValueError("Plotting value should be a Series or a list of Series")

    if isinstance(legend_label, list):
        var_name = x[0].name
    else:
        var_name = x.name
    ax = set_fig_1d_axis_options(axis=ax, var_name=var_name, bins=bins,
                                 scaling=None, is_logbins=is_logbins,
                                 logy=log_y, logx=log_x)
    if y_label: ax.set_ylabel(y_label)
    if x_label: ax.set_xlabel(x_label)
    fig.tight_layout()
    if legend:
        plt.legend()

    # save figure
    if to_file:
        if not title and not filename:
            raise ValueError("Must provide either a title or a filename")
        hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=title)
        out_png_file = config.plot_dir + f"{filename if filename else title}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        print(f"Figure saved to {out_png_file}")
    return fig


def ratio_plot_1d(
        p: pd.Series,
        q: pd.Series,
        x_label: str,
        y_label: str,
        title: str,
        bins: Union[tuple, list],
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

    if to_file:
        hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=title)
        out_png_file = config.plot_dir + f"{title}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        print(f"Figure saved to {out_png_file}")

    return fig


def plot_1d_overlay_and_acceptance_cutgroups(
        df: pd.DataFrame,
        lepton: str,
        var_to_plot: str,
        cutgroups: OrderedDict[str, List[str]],
        bins: Union[tuple, list],
        weight_col: str = 'weight',
        lumi: Optional[float] = None,
        scaling: Optional[str] = None,
        log_x: bool = False,
        to_pkl: bool = False,
        plot_width=7,
        plot_height=7,
        plot_label: Optional[str] = None,
) -> None:
    """Plots overlay of cutgroups and acceptance (ratio) plots"""
    # TODO: write docs for this function
    fig, (fig_ax, accept_ax) = plt.subplots(2, 1,
                                            figsize=(plot_width * 1.2, plot_height),
                                            gridspec_kw={'height_ratios': [3, 1]})

    if to_pkl:
        hists = dict()

    # check if variable needs to be specially binned
    is_logbins, alt_bins = getbins(var_to_plot)
    if alt_bins:
        bins = alt_bins

    # INCLUSIVE PLOT
    # ================
    h_inclusive = histplot_1d(var_x=df[var_to_plot], weights=df[weight_col],
                              bins=bins, fig_axis=fig_ax,
                              yerr='sumw2',
                              lumi=lumi, scaling=scaling,
                              label='Inclusive',
                              is_logbins=is_logbins,
                              color='k', linewidth=2)
    if to_pkl:
        hists['inclusive'] = h_inclusive

    # PLOT CUTS
    # ================
    for cutgroup in cutgroups.keys():
        print(f"    - generating cutgroup '{cutgroup}'")
        cut_df = cut_on_cutgroup(df, cutgroups, cutgroup)
        weight_cut = cut_df[weight_col]
        var_cut = cut_df[var_to_plot]

        h_cut = histplot_1d(var_x=var_cut, weights=weight_cut,
                            bins=bins, fig_axis=fig_ax,
                            lumi=lumi, scaling=scaling,
                            label=cutgroup,
                            is_logbins=is_logbins)
        if to_pkl:
            hists[cutgroup] = h_cut

        # RATIO PLOT
        # ================
        hep.histplot(h_cut.view().value / h_inclusive.view().value,
                     bins=h_cut.axes[0].edges, ax=accept_ax, label=cutgroup,
                     color=fig_ax.get_lines()[-1].get_color())

    # AXIS FORMATTING
    # ==================
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)

    # figure plot
    fig_ax = set_fig_1d_axis_options(axis=fig_ax, var_name=var_to_plot, bins=bins,
                                     scaling=scaling, is_logbins=is_logbins,
                                     logy=True, logx=log_x, lepton=lepton)
    fig_ax.legend()
    fig_ax.axes.xaxis.set_visible(False)

    # ratio plot
    set_fig_1d_axis_options(axis=accept_ax, var_name=var_to_plot, bins=bins, is_logbins=is_logbins, lepton=lepton)
    accept_ax.set_ylabel("Acceptance")

    # save figure
    if to_pkl:
        with open(config.pkl_hist_dir + plot_label + '_' + var_to_plot + '_1d_cutgroups_ratios.pkl', 'wb') as f:
            pkl.dump(hists, f)
            print(f"Saved pickle file to {f.name}")
    hep.atlas.label(llabel="Internal", loc=0, ax=fig_ax, rlabel=plot_label)
    out_png_file = config.plot_dir + f"{var_to_plot}_{str(scaling)}.png"
    fig.savefig(out_png_file, bbox_inches='tight')
    print(f"Figure saved to {out_png_file}")
    fig.clf()  # clear for next plot


def plot_2d_cutgroups(df: pd.DataFrame,
                      lepton: str,
                      x_var: str, y_var: str,
                      xbins: Union[tuple, list], ybins: Union[tuple, list],
                      cutgroups: OrderedDict[str, List[str]],
                      plot_label: str = '',
                      is_logz: bool = True,
                      to_pkl: bool = False,
                      ) -> None:
    """
    Runs over cutgroups in dictrionary and plots 2d histogram for each group

    :param df: Input dataframe
    :param lepton: name of lepton for axis labels
    :param x_var: column in dataframe to plot on x axis
    :param y_var: column in dataframe to plot on y axis
    :param xbins: binning in x
    :param ybins: binning in y
    :param cutgroups: ordered dictionary of cutgroups
    :param plot_label: plot title
    :param is_logz: whether display z-axis logarithmically
    :param to_pkl: whether to save histograms as pickle file
    """
    if to_pkl:
        hists = dict()

    # INCLUSIVE
    fig, ax = plt.subplots(figsize=(7, 7))
    weight_cut = df['weight']
    x_vars = df[x_var]
    y_vars = df[y_var]

    out_path = config.plot_dir + f"2d_{x_var}-{y_var}_inclusive.png"
    hist = histplot_2d(
        var_x=x_vars, var_y=y_vars,
        xbins=xbins, ybins=ybins,
        ax=ax, fig=fig,
        weights=weight_cut,
        is_z_log=is_logz,
    )
    if to_pkl:
        hists['inclusive'] = hist

    # get axis labels
    xlabel, _ = get_axis_labels(str(x_var), lepton)
    ylabel, _ = get_axis_labels(str(y_var), lepton)

    hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - inclusive', loc=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(out_path, bbox_inches='tight')
    print(f"printed 2d histogram to {out_path}")
    plt.close(fig)

    for cutgroup in cutgroups:
        print(f"    - generating cutgroup '{cutgroup}'")
        fig, ax = plt.subplots(figsize=(7, 7))

        cut_df = cut_on_cutgroup(df, cutgroups, cutgroup)
        weight_cut = cut_df['weight']
        x_vars = cut_df[x_var]
        y_vars = cut_df[y_var]

        out_path = config.plot_dir + f"2d_{x_var}-{y_var}_{cutgroup}.png"
        hist = histplot_2d(
            var_x=x_vars, var_y=y_vars,
            xbins=xbins, ybins=ybins,
            ax=ax, fig=fig,
            weights=weight_cut,
            is_z_log=is_logz,
        )
        if to_pkl:
            hists[cutgroup] = hist

        # get axis labels
        xlabel, _ = get_axis_labels(str(x_var), lepton)
        ylabel, _ = get_axis_labels(str(y_var), lepton)

        hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - ' + cutgroup, loc=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.savefig(out_path, bbox_inches='tight')
        print(f"printed 2d histogram to {out_path}")
        plt.close(fig)

    if to_pkl:
        with open(config.pkl_hist_dir + plot_label + f"_{x_var}-{y_var}_2d.pkl", 'wb') as f:
            pkl.dump(hists, f)
            print(f"Saved pickle file to {f.name}")


def plot_mass_slices(df: pd.DataFrame,
                     lepton: str,
                     xvar: str,
                     xbins: Union[tuple, list] = (100, 300, 10000),
                     logbins: bool = True,
                     logx: bool = False,
                     id_col: str = 'DSID',
                     weight_col: str = 'weight',
                     plot_label: str = '',
                     to_pkl: bool = False
                     ) -> None:
    """Plots all mass slices as well as inclusive sample (in this case just all given slices together)"""
    fig, ax = plt.subplots()

    if to_pkl:
        hists = dict()  # dictionary to hold mass slice histograms

    for dsid, mass_slice in df.groupby(id_col):
        hist = histplot_1d(mass_slice[xvar], mass_slice[weight_col], xbins, ax, yerr=None, label=dsid,
                           is_logbins=logbins, scaling='widths')
        if to_pkl:
            hists[dsid] = hist

    hist_inc = histplot_1d(df[xvar], df[weight_col], xbins, ax, yerr=None, is_logbins=logbins, scaling='widths',
                           c='k', linewidth=2, label='Inclusive')
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

    name = f"{xvar}_mass_slices_full"
    if to_pkl:
        with open(config.pkl_hist_dir + plot_label + '_' + name + '.pkl', 'wb') as f:
            pkl.dump(hists, f)
            print(f"Saved pickle file to {f.name}")
    path = config.plot_dir + name + '.png'
    fig.savefig(path, bbox_inches='tight')
    print(f"Figure saved to {path}")
