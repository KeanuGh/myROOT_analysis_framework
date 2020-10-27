from datetime import datetime
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from typing import Tuple, Optional, List, OrderedDict
from utils.axis_labels import labels_xs
from warnings import warn
import pandas as pd

from utils.dataframe_utils import get_luminosity


def scale_to_crosssection(hist: bh.Histogram, luminosity) -> None:
    """Scales histogram to cross-section. Currently undefined for multidimensional histograms"""
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    hist /= luminosity
    hist /= hist.axes[0].widths


def scale_by_bin_widths(hist: bh.Histogram) -> None:
    """Divides number of entries in bins by bin width"""
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    hist /= hist.axes[0].widths


def get_sumw2_1d(hist: bh.Histogram) -> np.array:
    """Returns numpy array of sum of weights squared in 1d boost-histogram histogram"""
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    # variances are sum of weights squared in each histogram bin
    return np.array([hist[idx].variance for idx, _ in enumerate(hist.axes[0])])


def get_root_sumw2_1d(hist: bh.Histogram) -> np.array:
    """Returns numpy array of root of sum of weights squared in 1d boost-histogram histogram"""
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    # variances are sum of weights squared in each histogram bin
    return np.array([np.sqrt(hist[idx].variance) for idx, _ in enumerate(hist.axes[0])])


def get_axis_labels(var_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Gets label for corresponting variable in axis labels dictionary"""
    if var_name in labels_xs:
        xlabel = labels_xs[var_name]['xlabel']
        ylabel = labels_xs[var_name]['ylabel']
    else:
        warn(f"Axis labels for {var_name} not found in in label lookup dictionary."
             f"They will be left blank.", UserWarning)
        xlabel = None
        ylabel = None
    return xlabel, ylabel


def plot_overlay_and_acceptance(var_name: str,
                                df: pd.DataFrame,
                                cutgroups: OrderedDict[str, List[str]],
                                dir_path: str,
                                cut_label: str,
                                lepton: str,
                                lumi: Optional[float] = None,
                                scaling: Optional[str] = None,
                                not_log: Optional[List[str]] = None,
                                plot_width=10,
                                plot_height=10,
                                n_bins: int = 30,
                                binrange: tuple = (1, 500),
                                eta_binrange: tuple = (-20, 20),
                                n_threads: int = 1,
                                ) -> None:
    if not_log is None:
        not_log = []
    print(f"Generating histogram for {var_name}...")
    fig, (fig_ax, accept_ax) = plt.subplots(1, 2)

    # whether or not bins should be logarithmic bins
    is_logbins = not any(map(var_name.__contains__, not_log))

    # get axis labels (x label, y label)
    xlabel, ylabel = get_axis_labels(var_name)

    # convert x label string for correct lepton if applicable
    try:
        xlabel = xlabel % lepton
    except TypeError:
        pass

    # INCLUSIVE PLOT
    # ================
    # setup inclusive histogram
    if is_logbins:
        h_inclusive = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log),
                                   storage=bh.storage.Weight())
    else:
        h_inclusive = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange),
                                   storage=bh.storage.Weight())

    h_inclusive.fill(df[var_name], weight=df['weight'], threads=n_threads)  # fill

    # scale
    if scaling == 'cs':
        if not lumi:
            lumi = get_luminosity(df)
        scale_to_crosssection(h_inclusive, luminosity=lumi)
    elif scaling == 'widths':
        scale_by_bin_widths(h_inclusive)
    elif scaling is not None:
        warn(f"Warning: Scaling value {scaling} unknown")

    yerr = get_root_sumw2_1d(h_inclusive)  # get sum of weights squared

    # plot
    hep.histplot(h_inclusive.view().value, bins=h_inclusive.axes[0].edges,
                 ax=fig_ax, yerr=yerr, label='Inclusive')

    # PLOT CUTS
    # ================
    for cutgroup in cutgroups.keys():
        print(f"    - generating cutgroup '{cutgroup}'")
        # get column names for boolean columns in dataframe containing the cuts
        cut_rows = [cut_name + cut_label for cut_name in cutgroups[cutgroup]]

        # setup
        if is_logbins:
            h_cut = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log),
                                 storage=bh.storage.Weight())
        else:
            h_cut = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange),
                                 storage=bh.storage.Weight())

        # perform cut and fill
        cut_df = df[df[cut_rows].any(1)]
        h_cut.fill(cut_df[var_name], weight=cut_df['weight'], threads=n_threads)  # fill

        # scale
        if scaling == 'cs':
            scale_to_crosssection(h_inclusive, luminosity=lumi)
        elif scaling == 'widths':
            scale_by_bin_widths(h_inclusive)

        cut_yerr = get_root_sumw2_1d(h_cut)  # get sum of weights squared

        # plot
        hep.histplot(h_cut.view().value, bins=h_cut.axes[0].edges,
                     ax=fig_ax, yerr=cut_yerr, label=cutgroup)

        # RATIO PLOT
        # ================
        hep.histplot(h_cut.view().value / h_inclusive.view().value,
                     bins=h_cut.axes[0].edges, ax=accept_ax, label=cutgroup)

    # log y axis, unless plotting Bjorken X
    if 'PDFinfo_X' not in var_name:
        fig_ax.semilogy()

    # apply axis options
    if is_logbins:  # set axis edge at 0
        fig_ax.set_xlim(*binrange)
    else:
        fig_ax.set_xlim(*eta_binrange)
    fig_ax.set_xlabel(xlabel)

    if scaling == 'xs':
        fig_ax.set_ylabel(ylabel)
    elif scaling == 'widths':
        fig_ax.set_ylabel('Entries / bin width')
    elif scaling is None:
        fig_ax.set_ylabel('Entries')

    fig_ax.legend()
    hep.box_aspect(fig_ax)  # makes just the main figure a square (& too small)

    # ratio plot
    if is_logbins:  # set axis edge at 0
        accept_ax.set_xlim(*binrange)
    else:
        accept_ax.set_xlim(*eta_binrange)
    accept_ax.set_xlabel(xlabel)
    accept_ax.set_ylabel("Acceptance")
    accept_ax.legend()
    hep.box_aspect(accept_ax)

    # set dimensions manually
    fig.set_figheight(plot_height)
    fig.set_figwidth(plot_width * 2)

    # save figure
    hep.atlas.label(data=False, ax=fig_ax, paper=False, year=datetime.now().year)
    out_png_file = dir_path + f"{var_name}_XS.png"
    fig.savefig(out_png_file)
    print(f"Figure saved to {out_png_file}")
    plt.clf()  # clear for next plot
