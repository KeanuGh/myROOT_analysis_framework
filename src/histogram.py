from __future__ import annotations

import copy
import logging
from typing import List, Tuple, Any, overload, Type

import ROOT
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike

from src.logger import get_logger
from utils.AtlasStyle import set_atlas_style
from utils.ROOT_utils import load_ROOT_settings
from utils.context import redirect_stdout

# settings
load_ROOT_settings()
set_atlas_style()  # set ATLAS plotting style to ROOT plots
plt.style.use(hep.style.ATLAS)  # set atlas-style plots in matplotilb
np.seterr(invalid="ignore")  # ignore division by zero errors


# TODO: 2D hist
class Histogram1D(bh.Histogram, family=None):
    """
    Wrapper around boost-histogram histogram for 1D
    """

    __slots__ = "logger", "name", "TH1"

    @overload
    def __init__(
        self,
        th1: Type[ROOT.TH1],
        logger: logging.Logger = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        bins: List[float] | np.ndarray,
        logger: logging.Logger = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        bins: Tuple[int, float, float],
        logbins: bool = False,
        logger: logging.Logger = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        var: ArrayLike = None,
        bins: List[float] | np.ndarray | Tuple[int, float, float] | bh.axis.Axis = (10, 0, 10),
        weight: ArrayLike | int = None,
        logbins: bool = False,
        logger: logging.Logger = None,
        name: str = "",
        title: str = "",
        **kwargs,
    ) -> None:
        ...

    def __init__(
        self,
        var: ArrayLike = None,
        bins: List[float] | np.ndarray | Tuple[int, float, float] | bh.axis.Axis = (10, 0, 10),
        weight: ArrayLike | float = None,
        logbins: bool = False,
        logger: logging.Logger = None,
        name: str = "",
        title: str = "",
        th1: Type[ROOT.TH1] = None,
        **kwargs,
    ) -> None:
        """
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param var: iterable of values to fill histogram
        :param weight: iterable of weights corresponding to each variable value or value to weight by
        :param logbins: whether logarithmic binnings
        :param logger: logger object to pass messages to. Creates new logger object that logs to console if None
        :param name: name of histogram
        :param title: histogram title
        :param th1: create a new histogram from a TH1
        :param kwargs: keyword arguments to pass to boost_histogram.Histogram()
        """
        if logger is None:
            logger = get_logger()
        self.logger = logger

        self.logger.debug("Checking variables..")
        # check length of var and weight
        if hasattr(weight, "__len__") and (len(var) != len(weight)):
            raise ValueError(
                f"Weight and value arrays are of different lengths! {len(weight)}, {len(var)}"
            )

        # check for invalid entries (nan or inf in weights or values)
        if weight is not None:
            inv_bool = np.logical_xor(
                np.logical_xor(np.isnan(var), np.isnan(weight)),
                np.logical_xor(np.isinf(var), np.isinf(weight)),
            )
            if n_inv := inv_bool.sum():
                self.logger.error(f"{n_inv} invalid entries in histogram!")

        if th1:
            # create from TH1
            self.logger.info(f"Creating histogram from TH1: '{th1}'...")
            edges = [th1.GetBinLowEdge(i + 1) for i in range(th1.GetNbinsX() + 1)]
            super().__init__(bh.axis.Variable(edges), storage=bh.storage.Weight())

            # set vars
            self.TH1 = th1.Clone()
            if title:
                self.TH1.SetTitle(title)
            if name:
                self.TH1.SetName(name)
                self.name = name
            else:
                self.name = th1.GetName()

            # fill
            for idx, _ in np.ndenumerate(self.view(flow=True)):
                self.view(flow=True).value[idx] = th1.GetBinContent(*idx)  # bin value
                self.view(flow=True).variance[idx] = th1.GetBinError(*idx) ** 2

            if var is not None:
                self.Fill(var, weight=weight)

        else:
            self.logger.debug(f"Initialising histogram {name}...")

            # TH1
            self.TH1 = ROOT.TH1F(name, title, *self.__get_TH1_bins(bins))
            self.name = name

            # get axis
            axis = bins if isinstance(bins, bh.axis.Axis) else self.__gen_axis(bins, logbins)
            super().__init__(axis, storage=bh.storage.Weight(), **kwargs)

            if var is not None:
                self.Fill(var, weight=weight)

    def Fill(self, var: ArrayLike, weight: ArrayLike | int | float = None) -> Histogram1D:
        self.logger.debug(f"Filling histogram {self.name} with {len(var)} events..")
        super().fill(var, weight=weight, threads=0)

        # fill vector with weight in order to zip
        if isinstance(weight, (int, float)):
            weight = np.full(len(var), weight)
        elif weight is None:
            weight = np.ones(len(var))

        # TODO: got to be a faster way than this
        for v, w in zip(var, weight):
            self.TH1.Fill(v, w)
        return self

    def __copy__(self) -> Histogram1D:
        new = self._new_hist(copy.copy(self._hist))
        new.TH1 = self.TH1.Clone()
        new.name = self.name
        return new

    def __deepcopy__(self, memo: Any) -> Histogram1D:
        new = self._new_hist(copy.deepcopy(self._hist), memo=memo)
        new.TH1 = self.TH1.Clone()
        new.name = self.name
        return new

    def __truediv__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        result = self.copy()
        return result.__itruediv__(other)

    def __itruediv__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        if isinstance(other, Histogram1D):
            # Scale variances based on ROOT method. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l02929
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            clsq = c1 * c1
            variance = (
                (self.view(flow=True).variance * clsq) + (other.view(flow=True).variance * c0 * c0)
            ) / (clsq * clsq)

            self.view(flow=True).value = self.view(flow=True).value / other.view(flow=True).value
            self.view(flow=True).variance = variance
            self.TH1.Divide(other.TH1)
            return self
        else:
            # scale TH1 properly
            if hasattr(other, "__iter__"):
                for i, val in enumerate(other):
                    self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) / val)
                    self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) / val)
            else:
                self.TH1.Scale(1 / other)
            # let boost-histogram handle return
            return self._compute_inplace_op("__itruediv__", other)

    def __mul__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow multiplying weighted histograms so implement that here"""
        result = self.copy()
        return result.__imul__(other)

    def __imul__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow multiplying weighted histograms so implement that here"""
        if isinstance(other, Histogram1D):
            # Scale variances based on ROOT method. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l06116
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            variance = (
                self.view(flow=True).variance * c1 * c1 + other.view(flow=True).variance * c0 * c0
            )

            self.view(flow=True).value.__imul__(other.view(flow=True).value)
            self.view(flow=True).variance = variance
            self.TH1.Multiply(other.TH1)
            return self
        else:
            # scale TH1 properly
            if hasattr(other, "__iter__"):
                for i, val in enumerate(other):
                    self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) * val)
                    self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) * val)
            else:
                self.TH1.Scale(other)
            return self._compute_inplace_op("__imul__", other)

    @staticmethod
    @overload
    def __gen_axis(bins: List[float] | ArrayLike) -> bh.axis.Axis:
        ...

    @staticmethod
    @overload
    def __gen_axis(bins: Tuple[int, float, float], logbins: bool = False) -> bh.axis.Axis:
        ...

    @staticmethod
    def __gen_axis(
        bins: List[float] | ArrayLike | Tuple[int, float, float], logbins: bool = False
    ) -> bh.axis.Axis:
        """
        Returns the correct type of boost-histogram axis based on the input bins.

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
        :param logbins: whether logarithmic bins. Silently ignored if using variable bins.
        """
        if len(bins) == 3 and isinstance(bins, tuple):
            return bh.axis.Regular(*bins, transform=bh.axis.transform.log if logbins else None)
        else:
            return bh.axis.Variable(bins)

    @staticmethod
    def __get_TH1_bins(
        bins: List[float] | Tuple[int, float, float] | bh.axis.Axis
    ) -> Tuple[int, list | np.ndaray] | Tuple[int, float, float]:
        """Format bins for TH1 constructor"""
        if isinstance(bins, bh.axis.Axis):
            return bins.size, bins.edges

        elif hasattr(bins, "__iter__"):
            if len(bins) == 3 and isinstance(bins, tuple):
                return bins
            else:
                return len(bins) - 1, np.array(bins)

        raise ValueError("Bins should be list of bin edges or tuple like (nbins, xmin, xmax)")

    # Variables
    # ===================
    @property
    def n_bins(self) -> int:
        """Get number of bins"""
        return self.axes[0].size

    @property
    def extent(self) -> int:
        return self.axes[0].extent

    @property
    def bin_widths(self) -> np.array:
        """get bin widths"""
        return self.axes[0].widths

    @property
    def bin_range(self) -> np.array:
        """get histogram range"""
        return self.bin_edges[-1] - self.bin_edges[0]

    @property
    def bin_edges(self) -> np.array:
        """get bin edges"""
        return self.axes[0].edges

    @property
    def bin_centres(self) -> np.array:
        """get bin centres"""
        return self.axes[0].centers

    @property
    def n_entries(self) -> float:
        """Get number of entries"""
        return self.TH1.GetEntries()

    def eff_entries(self, flow: bool = False) -> float:
        """get effective number of entries"""
        return self.bin_sum(flow) * self.bin_sum(flow) / sum(self.sumw2(flow))

    def sumw2(self, flow: bool = False) -> np.array:
        """get squared sum of weights"""
        return self.view(flow).variance

    def root_sumw2(self, flow: bool = False) -> np.array:
        """get squared sum of weights"""
        return np.sqrt(self.sumw2(flow))

    def bin_values(self, flow: bool = False) -> np.array:
        """get bin values"""
        return self.values(flow)

    @property
    def areas(self) -> np.array:
        """get bin areas"""
        return self.bin_values() * self.bin_widths

    @property
    def integral(self) -> float:
        """get integral of histogram"""
        return sum(self.areas)

    def bin_sum(self, flow: bool = False) -> float:
        """Get sum of bin contents"""
        return sum(self.bin_values(flow))

    # Stats
    # ===================
    @property
    def mean(self) -> float:
        """Return mean value of histogram - ROOT"""
        return self.TH1.GetMean()

    @property
    def mean_error(self) -> float:
        """Return standard error of mean - ROOT"""
        return self.TH1.GetMeanError()

    @property
    def std(self) -> float:
        """Return standard deviation - ROOT"""
        return self.TH1.GetStdDev()

    @property
    def std_error(self) -> float:
        """Return standard deviation error - ROOT"""
        return self.TH1.GetStdDevError()

    # Scaling
    # ===================
    def normalised(self) -> Histogram1D:
        """Return histogram normalised to unity"""
        return self.normalised_to(1.0)

    def normalised_to(self, factor: float) -> Histogram1D:
        """Return histogram normalised to factor"""
        return self.scaled(factor / self.bin_sum())

    def scaled(self, scale_factor: float) -> Histogram1D:
        """Return rescaled histogram"""
        return self.copy() * scale_factor

    def normalise(self) -> None:
        """Normalise histogram to unity inplace"""
        self.normalise_to(1.0)

    def normalise_to(self, factor: float) -> None:
        """Normalise histogram to factor inplace"""
        self.scale(factor / self.bin_sum())

    def scale(self, scale_factor: float) -> None:
        """Return rescaled histogram"""
        self.__imul__(scale_factor)

    # Fitting
    # ===================
    def chi_square(self, other: ROOT.TF1 | ROOT.TH1 | Histogram1D) -> Tuple[float, float]:
        """Perform chi-squared test. Retun chi2 per degree of freedom, pvalue"""
        h1 = self.TH1
        if isinstance(other, ROOT.TH1):
            return h1.Chi2Test(other, "WWCHI2/NDF"), h1.Chi2Test(other, "WW")
        elif isinstance(other, Histogram1D):
            h2 = other.TH1
            return h1.Chi2Test(h2, "WWCHI2/NDF"), h1.Chi2Test(h2, "WW")
        elif isinstance(other, ROOT.TF1):
            return h1.Chisquare(other, "WWCHI2/NDF"), h1.Chisquare(other, "WW")
        else:
            raise TypeError(f"{type(other)} is an incorrect type for a chi-square test")

    # Conversion
    # ===================
    def to_TH1(self, name: str = "name", title: str = "title") -> ROOT.TH1F:
        """Convert Histogram to ROOT TH1F"""
        if self.TH1 is None:
            h_root = ROOT.TH1F(name, title, self.n_bins, self.bin_edges)
        else:
            h_root = self.TH1

        # fill TH1
        for idx, bin_cont in np.ndenumerate(self.view(flow=True)):
            h_root.SetBinContent(*idx, bin_cont[0])  # bin value
            h_root.SetBinError(*idx, np.sqrt(bin_cont[1]))  # root sum of weights

        self.TH1 = h_root

        return h_root

    # Plotting
    # ===================
    def plot(
        self,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool | None = True,
        w2: bool = False,
        normalise: float | bool = False,
        scale_by_bin_width: bool = False,
        stats_box: bool = False,
        out_filename: str = None,
        show: bool = False,
        **kwargs,
    ) -> Histogram1D:
        """
        Plot histogram on axis ax

        :param ax: matplotlib Axes object to plot on. Will create new axis if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param w2: Whether to do a poissonian interval error calculation based on weights
        :param normalise: Normalisation value:
                          - int or float
                          - True for normalisation of unity
        :param scale_by_bin_width: whether to scale histogram by bin widths
        :param stats_box: whether to add a stats box to the plot
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :param out_filename: provide filename to print. If not given, nothing is saved
        :param show: whether to display the plot (plt.show())
        :return: matplotlib axis object with plot
        """
        hist = self.copy()

        if scale_by_bin_width:
            self.logger.debug(f"Scaling histogram {self.name} by bin width...")
            hist /= hist.bin_widths

        # normalise/scale
        if not normalise:
            self.logger.debug(f"Plotting histogram {self.name}...")
        elif normalise is True:
            self.logger.debug(f"Plotting histogram {self.name} normalised to unity...")
            hist.normalise()
        else:
            self.logger.debug(
                f"Plotting normalised histogram {self.name} normalised to {normalise}..."
            )
            hist.normalise_to(normalise)

        # set error
        if yerr is True:
            yerr = hist.root_sumw2()
        elif yerr and not isinstance(yerr, bool) and not hasattr(yerr, "__len__"):
            raise TypeError(f"yerr should be a bool or iterable of values. Got {yerr}")

        if not ax:
            _, ax = plt.subplots()

        hep.histplot(hist, ax=ax, yerr=yerr, w2=hist.sumw2() if w2 else None, **kwargs)

        if stats_box:
            # dumb workaround to avoid the stats boxes from overlapping eachother
            box_xpos, box_ypos = (0.75, 0.55)
            for artist in ax.get_children():
                if isinstance(artist, plt.Text):
                    if r"$\mu=" in artist.get_text():
                        box_ypos = 0.25

            textstr = "\n".join(
                (
                    hist.name,
                    r"$\mu=%.2f\pm%.2f$" % (hist.mean, hist.mean_error),
                    # r'$\sigma=%.2f\pm%.2f$' % (self.std, self.std_error),
                    r"$\mathrm{Entries}: %.0f$" % hist.n_entries,
                    r"$\mathrm{Bin sum}: %.2f$" % hist.bin_sum(True),
                    r"$\mathrm{Integral}: %.2f$" % hist.integral,
                )
            )
            ax.text(x=box_xpos, y=box_ypos, s=textstr, transform=ax.transAxes, fontsize="x-small")

        if out_filename:
            plt.savefig(out_filename, bbox_inches="tight")
            self.logger.info(f"image saved in {out_filename}")

        if show:
            plt.show()

        return hist

    def Rplot(
        self,
        normalise: float | bool = False,
        plot_option: str = "",
        stats_box: bool = False,
        out_filename: str = False,
    ) -> None:
        """
        ROOT plot

        :param normalise: whether to normalise
        :param plot_option: option(s) to pass to Draw()
        :param stats_box: whether to print stats box
        :param out_filename: filename to print if necessary
        """
        if normalise:
            h = self.normalised_to(normalise)
        else:
            h = self.copy()

        if stats_box:
            h.TH1.SetStats(True)

        c = ROOT.TCanvas()
        h.TH1.Draw("E Hist" + plot_option)

        if out_filename:
            c.Print(out_filename)

    def plot_ratio(
        self,
        other: Histogram1D,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool | str = True,
        normalise: bool = False,
        label: str = None,
        fit: bool = False,
        name: str = "",
        out_filename: str = None,
        yax_lim: float = False,
        display_stats: bool = True,
        color: str = "k",
        **kwargs,
    ) -> Histogram1D:
        """
        Plot (and properly format) ratio between this histogram and another.

        :param other: Other histogram
        :param ax: Axis to plot on. will create new if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - 'carry', carries fractional error from original histogram
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param normalise: Whether histograms are normalised before taking ratio
        :param label: Legend label
        :param fit: whether to fit to a 0-degree polynomial and display line, chi-square and p-value
        :param name: histogram name
        :param out_filename: provide filename to print. If not given, nothing is saved
        :param yax_lim: limit y-axis to 1 +/- {yax_lim}
        :param display_stats: whether to display the fit parameters on the plot
        :param color: plot colour
        :param kwargs: Args to pass to ax.errorbar()
        :return: axis object with plot
        """
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Bins do not match!")
        if not ax:
            _, ax = plt.subplots()

        # create ratio histogram
        if normalise:
            h_ratio = other.normalised() / self.normalised()
        else:
            h_ratio = other / self

        # set name
        if name:
            h_ratio.name = name

        if yerr is True:
            yerr = h_ratio.root_sumw2()
        elif yerr == "carry":
            yerr = (self.root_sumw2() / self.bin_values()) * h_ratio.bin_values()
            for i in range(h_ratio.TH1.GetNbinsX()):
                h_ratio.TH1.SetBinError(i + 1, yerr[i])

        if fit:
            self.logger.info("Performing fit on ratio..")

            with redirect_stdout() as fit_output:
                fit_results = h_ratio.TH1.Fit("pol0", "VFSN")

            self.logger.debug(
                f"ROOT fit output:\n"
                f"==========================================================================\n"
                f"{fit_output.getvalue()}"
                f"=========================================================================="
            )
            c = fit_results.Parameters()[0]
            err = fit_results.Errors()[0]

            # display fit line
            col = "r" if color == "k" else color
            ax.fill_between(
                [self.bin_edges[0], self.bin_edges[-1]], [c - err], [c + err], color=col, alpha=0.3
            )
            ax.axhline(c, color=col, linewidth=1.0)

            if display_stats:
                textstr = "\n".join(
                    (
                        r"$\chi^2=%.3f$" % fit_results.Chi2(),
                        r"$\mathrm{NDF}=%.3f$" % fit_results.Ndf(),
                        r"$c=%.2f\pm%.3f$" % (c, err),
                    )
                )
                # dumb workaround to avoid the stats boxes from overlapping eachother
                loc = "upper left"
                for artist in ax.get_children():
                    if isinstance(artist, AnchoredText):
                        if r"$\chi^2=" in artist.get_children()[0].get_text():
                            loc = "lower left"
                stats_box = AnchoredText(
                    textstr, loc=loc, frameon=False, prop=dict(fontsize="x-small")
                )
                ax.add_artist(stats_box)

        ax.axhline(1.0, linestyle="--", linewidth=1.0, c="k")
        ax.errorbar(
            h_ratio.bin_centres,
            h_ratio.bin_values(),
            xerr=h_ratio.bin_widths / 2,
            yerr=yerr,
            linestyle="None",
            label=label,
            **kwargs,
            c=color,
        )
        ax.grid(visible=True, which="both", axis="y")

        if yax_lim:
            ax.set_ylim(1 - yax_lim, 1 + yax_lim)
        else:
            # I don't know why the matplotlib automatic yaxis scaling doesn't work, but I've done it for them here
            ymax = np.max(np.ma.masked_invalid(h_ratio.bin_values() + (yerr / 2)))
            ymin = np.min(np.ma.masked_invalid(h_ratio.bin_values() - (yerr / 2)))
            vspace = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - vspace, ymax + vspace)

        if out_filename:
            plt.savefig(out_filename, bbox_inches="tight")
            self.logger.info(f"image saved in {out_filename}")

        return h_ratio
