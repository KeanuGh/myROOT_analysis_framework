from collections.abc import Sequence
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
from matplotlib import ticker

from models.plotting import (
    Hist2dOpts as Hist2dOpts,
)
from models.plotting import (
    PlotKwargs as PlotKwargs,
)
from models.plotting import (
    PlotOpts as PlotOpts,
)
from models.plotting import (
    ProfileOpts as ProfileOpts,
)
from utils.variable_names import variable_data

# set plot style
plt.style.use([hep.style.ATLAS])


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
            warn("Transforms cannot be passed to variable bin types", stacklevel=2)
        return bh.axis.Variable(bins)
    else:
        raise TypeError(
            f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. Got {bins}"
        )


def set_axis_options(
    ax: plt.Axes,
    per_hist_vars: PlotOpts | None = None,
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
    label_params: dict | None = None,
    font_size: int = 16,
) -> None:
    """Set axis options for plot"""

    # let mplhep handle the easy stuff
    if label_params is None:
        label_params = {}
    set_hep_label(ax=ax, title=title, **label_params)

    # get axis labels from variable names if possible
    # it'll error out when plotting if the histogram edges aren't equal
    if per_hist_vars:
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
        ax.set_xlim(*bin_range)
        if ratio_ax:
            ratio_ax.set_xlim(*bin_range)

    if logx:
        ax.semilogx()
    if logy:
        ax.semilogy()
    else:
        # if there are no negative yvalues, set limit at origin
        ax.set_ylim(bottom=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=font_size)

    # Main plot axis options
    if y_axlim:
        ax.set_ylim(*y_axlim)
    if x_axlim:
        ax.set_xlim(*x_axlim)
        if ratio_ax:
            ratio_ax.set_xlim(*x_axlim)

    ax.tick_params(axis="both", which="both", labelsize=font_size)
    ax.get_xaxis().get_offset_text().set_fontsize(font_size)
    ax.get_yaxis().get_offset_text().set_fontsize(font_size)

    # set some xaxis options only on bottom xaxis (ie ratio plot if it exists else regular plot)
    axis_ = ratio_ax if (ratio_ax is not None) else ax
    if logx:
        axis_.semilogx()
        axis_.xaxis.set_minor_formatter(ticker.LogFormatter())
        axis_.xaxis.set_major_formatter(ticker.LogFormatter())
        axis_.tick_params(axis="both", which="both", labelsize=font_size)
    if xlabel:
        axis_.set_xlabel(xlabel, fontsize=font_size)

    if ratio_ax is not None:
        # if len(per_hist_vars["hists"]) > 2:  # don't show legend if there's only two plots
        #     ratio_ax.legend(fontsize=10, loc=1)
        ratio_ax.set_ylabel(ratio_label, fontsize=font_size)
        ratio_ax.tick_params(axis="both", which="both", labelsize=font_size)
        ratio_ax.get_xaxis().get_offset_text().set_fontsize(font_size)
        ratio_ax.get_yaxis().get_offset_text().set_fontsize(font_size)

        # i hate matplotlib i hate matplotlib i hate matplotlib i hate matplotlib i hate maplotlib i
        if logx:
            ax.semilogx()
            ax.xaxis.set_minor_formatter(
                ticker.LogFormatter(labelOnlyBase=True, minor_thresholds=(0, 0))
            )
            ax.xaxis.set_major_formatter(
                ticker.LogFormatter(labelOnlyBase=True, minor_thresholds=(0, 0))
            )
        ax.xaxis.set_ticklabels([])


def set_hep_label(ax: plt.Axes, title: str = "", **label_params) -> None:
    """
    Sets the mplhep label on axis.
    """
    label_args = dict(
        fontstyle=("italic", "italic", "normal", "normal"),
        ax=ax,
        loc=0,
        llabel="",
        rlabel=title,
    )
    if label_params:
        label_args.update(label_params)
    hep.atlas.label(**label_args)
