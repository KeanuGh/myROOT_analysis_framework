from __future__ import annotations

import copy
import logging
from copy import deepcopy
from typing import Any, overload

import ROOT
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike

from src.logger import get_logger
from utils import ROOT_utils
from utils.AtlasUtils import set_atlas_style
from utils.context import redirect_stdout

# settings
if ROOT.PyConfig.StartGUIThread:  # prevents a ROOT crash from settings being applied multiple times
    ROOT_utils.load_ROOT_settings()
    set_atlas_style()  # set ATLAS plotting style to ROOT plots
    plt.style.use(hep.style.ATLAS)  # set atlas-style plots in matplotilb
    np.seterr(invalid="ignore", divide="ignore")  # ignore division by zero errors


class Histogram1D(bh.Histogram, family=None):
    """
    Wrapper around boost-histogram histogram for 1D
    """

    __slots__ = "logger", "name", "TH1"

    @overload
    def __init__(
        self,
        th1: ROOT.TH1,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        var: ROOT.TH1,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        bins: list[float] | np.ndarray,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        bins: tuple[int, float, float],
        logbins: bool = False,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        var: ArrayLike | None | ROOT.TH1 = None,
        bins: list[float] | np.ndarray | tuple[int, float, float] | bh.axis.Axis = (10, 0, 10),
        weight: ArrayLike | int | None = None,
        logbins: bool = False,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
        **kwargs,
    ) -> None:
        ...

    def __init__(
        self,
        var: ArrayLike | None | ROOT.TH1 = None,
        bins: list[float] | np.ndarray | tuple[int, float, float] | bh.axis.Axis = (10, 0, 10),
        weight: ArrayLike | float | None = None,
        logbins: bool = False,
        logger: logging.Logger | None = None,
        name: str = "",
        title: str = "",
        th1: ROOT.TH1 | None = None,
        **kwargs,
    ) -> None:
        """
        :type bins: object
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param var: iterable of values to fill histogram
        :param weight: iterable of weights corresponding to each variable value or value to weight by
        :param logbins: whether logarithmic binnings
        :param logger: logger object to pass messages to. Creates new logger object that logs to console if None
        :param name: name of histogram
        :param title: title of histogram
        :param th1: create a new histogram from a TH1
        :param kwargs: keyword arguments to pass to boost_histogram.Histogram()
        """
        if logger is None:
            logger = get_logger()
        self.logger = logger

        if isinstance(var, ROOT.TH1):
            th1 = var

        if th1:
            # if passed histogram is a TH2, project along the X-axis
            if isinstance(th1, ROOT.TH2):
                th1 = th1.ProfileX()

            # create from TH1
            edges = ROOT_utils.get_th1_bin_edges(th1)
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
            if self.TH1.GetEntries() > 0:
                for idx, _ in np.ndenumerate(self.view(flow=True)):
                    self.view(flow=True).value[idx] = th1.GetBinContent(*idx)  # bin value
                    self.view(flow=True).variance[idx] = th1.GetBinError(*idx) ** 2

            if var is not None:
                self.Fill(var, weight=weight)

        else:
            self.logger.debug("Checking variables..")
            # check length of var and weight
            if hasattr(weight, "__len__") and (len(var) != len(weight)):
                raise ValueError(
                    f"Weight and value arrays are of different lengths! {len(weight)}, {len(var)}"
                )

            if hasattr(var, "dtype") and var.dtype.name == "object":
                # try and turn numerical
                var = var.astype("float64")

            # check for invalid entries (nan or inf in weights or values)
            if weight is not None:
                try:
                    inv_bool = np.logical_xor(
                        np.logical_xor(np.isnan(var), np.isnan(weight)),
                        np.logical_xor(np.isinf(var), np.isinf(weight)),
                    )
                    if n_inv := inv_bool.sum():
                        self.logger.error(f"{n_inv} invalid entries in histogram!")
                except TypeError:
                    self.logger.debug("Skipped invalid event count due to type error")

            self.logger.debug(f"Initialising histogram {name}...")

            # TH1
            self.TH1 = ROOT.TH1F(name, title, *ROOT_utils.get_TH1_bin_args(bins, logbins=logbins))
            self.name = name

            # get axis
            axis = bins if isinstance(bins, bh.axis.Axis) else self.__gen_axis(bins, logbins)
            super().__init__(axis, storage=bh.storage.Weight(), **kwargs)

            if var is not None:
                self.Fill(var, weight=weight)

    def Fill(self, var: ArrayLike, weight: ArrayLike | int | float | None = None) -> Histogram1D:
        """Fill boost-histogram and TH1. Only accepts numpy array and derivatives or scalar weight values."""
        super().fill(var, weight=weight, threads=0)

        # fastest way to multifill a TH1 in pyROOT I've found is with an RDataFrame
        rdf_dict = dict(x=var.values if isinstance(var, pd.Series) else var)
        if weight is not None:
            if isinstance(weight, (int, float)):
                weight = np.full(len(var), weight)  # type: ignore
            rdf_dict["w"] = weight.values if isinstance(weight, pd.Series) else weight

        # catch weird type errors
        try:
            rdf = ROOT.RDF.FromNumpy(rdf_dict)
            self.TH1 = rdf.Fill(self.TH1, list(rdf_dict.keys())).GetPtr()

        except RuntimeError as e:
            if "Object not convertible" in str(e):
                raise RuntimeError(
                    f"Cannot convert object of type '{type(rdf_dict['x'])}' to 'AsRVec'"
                ) from e
            raise e

        return self

    def __copy__(self) -> Histogram1D:
        new = self._new_hist(copy.copy(self._hist))
        new.TH1 = self.TH1.Clone()
        new.name = self.name
        new.logger = self.logger
        return new

    def __deepcopy__(self, memo: Any) -> Histogram1D:
        new = self._new_hist(copy.deepcopy(self._hist), memo=memo)
        new.TH1 = self.TH1.Clone()
        new.name = self.name
        new.logger = self.logger
        return new

    def __truediv__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        result = self.copy()
        return result.__itruediv__(other)

    def __itruediv__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        if isinstance(other, Histogram1D):
            # Scale variances based on ROOT method. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l02929
            c0 = self.view(flow=True).value  # type: ignore
            c1 = other.view(flow=True).value  # type: ignore
            c0sq = c0 * c0
            c1sq = c1 * c1
            variance = (
                (self.view(flow=True).variance * c1sq)  # type: ignore
                + (other.view(flow=True).variance * c0sq)  # type: ignore
            ) / (c1sq * c1sq)

            self.view(flow=True).value = (
                self.view(flow=True).value / other.view(flow=True).value
            )  # type: ignore
            self.view(flow=True).variance = variance  # type: ignore
            self.TH1.Divide(other.TH1)
            return self

        # scale TH1 properly
        if hasattr(other, "__iter__"):
            if len(other) != self.TH1.GetNbinsX():
                raise ValueError("Number of bins don't match!")
            for i, val in enumerate(other):
                self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) / val)
                self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) / val)
        else:
            self.TH1.Scale(1 / other)  # type: ignore
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
            c0 = self.view(flow=True).value  # type: ignore
            c1 = other.view(flow=True).value  # type: ignore
            variance = (
                self.view(flow=True).variance * c1 * c1 + other.view(flow=True).variance * c0 * c0
            )

            self.view(flow=True).value.__imul__(other.view(flow=True).value)  # type: ignore
            self.view(flow=True).variance = variance  # type: ignore
            self.TH1.Multiply(other.TH1)
            return self

        # scale TH1 properly
        if hasattr(other, "__iter__"):
            if len(other) != self.TH1.GetNbinsX():
                raise ValueError("Number of bins don't match!")
            for i, val in enumerate(other):
                self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) * val)
                self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) * val)
        else:
            self.TH1.Scale(other)
        return self._compute_inplace_op("__imul__", other)

    def __add__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """Need to handle adding together TH1s"""
        result = self.copy()
        return result.__iadd__(other)

    def __iadd__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """Handle adding TH1s"""
        if isinstance(other, Histogram1D):
            self.TH1.Add(other.TH1)
        else:
            # scale TH1 properly
            if hasattr(other, "__iter__"):
                if len(other) != self.TH1.GetNbinsX():
                    raise ValueError("Number of bins don't match!")
                for i, val in enumerate(other):
                    self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) + val)
                    self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) + val)
            else:
                for i in range(self.TH1.GetNbinsX() + 2):
                    self.TH1.SetBinContent(i, self.TH1.GetBinContent(i) + other)

        return self._compute_inplace_op("__iadd__", other)

    def __sub__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """Need to handle adding together TH1s"""
        result = self.copy()
        return result.__isub__(other)

    def __isub__(self, other: bh.Histogram | "np.typing.NDArray[Any]" | float) -> Histogram1D:
        """Handle adding TH1s"""
        if isinstance(other, Histogram1D):
            self.TH1.Add(other.TH1, -1)
        else:
            # scale TH1 properly
            if hasattr(other, "__iter__"):
                if len(other) != self.TH1.GetNbinsX():
                    raise ValueError("Number of bins don't match!")
                for i, val in enumerate(other):
                    self.TH1.SetBinContent(i + 1, self.TH1.GetBinContent(i + 1) - val)
                    self.TH1.SetBinError(i + 1, self.TH1.GetBinError(i + 1) - val)
            else:
                for i in range(self.TH1.GetNbinsX() + 2):
                    self.TH1.SetBinContent(i, self.TH1.GetBinContent(i) - other)

        return self._compute_inplace_op("__iadd__", other)

    @staticmethod
    @overload
    def __gen_axis(bins: list[float] | ArrayLike) -> bh.axis.Axis:
        ...

    @staticmethod
    @overload
    def __gen_axis(bins: tuple[int, float, float], logbins: bool = False) -> bh.axis.Axis:
        ...

    @staticmethod
    def __gen_axis(
        bins: list[float] | ArrayLike | tuple[int, float, float], logbins: bool = False
    ) -> bh.axis.Axis:
        """
        Returns the correct type of boost-histogram axis based on the input bins.

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
        :param logbins: whether logarithmic bins. Silently ignored if using variable bins.
        """
        if len(bins) == 3 and isinstance(bins, tuple):  # type: ignore
            return bh.axis.Regular(*bins, transform=bh.axis.transform.log if logbins else None)
        return bh.axis.Variable(bins)  # type: ignore

    # Variables
    # ===================
    @property
    def n_bins(self) -> int:
        """Get number of bins"""
        return self.axes[0].size

    @property
    def extent(self) -> int:
        """Get axis extent"""
        return self.axes[0].extent

    @property
    def bin_widths(self) -> np.typing.NDArray[1, float]:
        """get bin widths"""
        return self.axes[0].widths

    @property
    def bin_range(self) -> float:
        """get histogram range"""
        return self.bin_edges[-1] - self.bin_edges[0]  # type: ignore

    @property
    def bin_edges(self) -> np.typing.NDArray[1, float]:
        """get bin edges"""
        return self.axes[0].edges

    @property
    def bin_centres(self) -> np.typing.NDArray[1, float]:
        """get bin centres"""
        return self.axes[0].centers

    @property
    def n_entries(self) -> int:
        """Get number of entries"""
        return self.TH1.GetEntries()

    def eff_entries(self, flow: bool = False) -> float:
        """get effective number of entries"""
        return self.bin_sum(flow) * self.bin_sum(flow) / sum(self.sumw2(flow))  # type: ignore

    def sumw2(self, flow: bool = False) -> np.typing.NDArray[1, float]:
        """get squared sum of weights"""
        return self.view(flow).variance  # type: ignore

    def get_error(self, idx: int) -> float:
        """Get ROOT error of bin"""
        return self.TH1.GetBinError(idx + 1)

    def error(self, flow: bool = False) -> np.typing.NDArray[1, float]:
        """get ROOT error"""
        return np.array(ROOT_utils.get_th1_bin_errors(self.TH1, flow))

    def root_sumw2(self, flow: bool = False) -> np.typing.NDArray[1, float]:
        """get squared sum of weights"""
        return np.sqrt(self.sumw2(flow))

    def bin_values(self, flow: bool = False) -> np.typing.NDArray[1, float]:
        """get bin values"""
        return self.values(flow)

    @property
    def areas(self) -> np.typing.NDArray[1, float]:
        """get bin areas"""
        return self.bin_values() * self.bin_widths  # type: ignore

    @property
    def integral(self) -> float:
        """get integral of histogram"""
        return sum(self.areas)  # type: ignore

    def bin_sum(self, flow: bool = False) -> float:
        """Get sum of bin contents"""
        return sum(self.bin_values(flow))  # type: ignore

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

    def divide_binom(self, other: Histogram1D, bayes: bool = True) -> Histogram1D:
        """
        Return self / other with binomial errors.
        Set bayes to true for a bayesian approach to avoid err=0 for equal or zero bins.
        """
        h = self / other

        # errors
        if bayes:
            h.TH1 = ROOT_utils.bayes_divide(self.TH1, other.TH1)
        else:
            h.TH1 = h.TH1.Divide(self.TH1, other.TH1, 1, 1, "B")

        return h

    # Fitting
    # ===================
    def chi_square(self, other: ROOT.TF1 | ROOT.TH1 | Histogram1D) -> tuple[float, float]:
        """Perform chi-squared test. Retun chi2 per degree of freedom, pvalue"""
        h1 = self.TH1
        if isinstance(other, ROOT.TH1):
            return h1.Chi2Test(other, "WWCHI2/NDF"), h1.Chi2Test(other, "WW")
        if isinstance(other, Histogram1D):
            h2 = other.TH1
            return h1.Chi2Test(h2, "WWCHI2/NDF"), h1.Chi2Test(h2, "WW")
        if isinstance(other, ROOT.TF1):
            return h1.Chisquare(other, "WWCHI2/NDF"), h1.Chisquare(other, "WW")
        raise TypeError(f"{type(other)} is an incorrect type for a chi-square test")

    # Conversion
    # ===================
    def non_zero_range(self, tol=1e-15) -> Histogram1D:
        """Return histogram containing non-zero range of self"""
        first_nz_idx = 0
        last_nz_idx = 0
        first_found = False
        for bin_idx, bin_val in enumerate(self.bin_values()):
            in_tol = abs(bin_val) > tol  # type: ignore
            if in_tol and (not first_found):
                first_nz_idx = bin_idx
                first_found = True

            if in_tol:
                last_nz_idx = bin_idx

        if first_nz_idx == last_nz_idx:
            return self
        if first_nz_idx < last_nz_idx:
            return self

        new_hist = self[first_nz_idx:last_nz_idx]

        # define new TH1
        new_hist.TH1 = ROOT.__getattr__(self.TH1.ClassName())(
            self.TH1.GetName(),
            self.TH1.GetTitle(),
            len(new_hist),
            *new_hist.bin_edges(),
        )
        for i in range(first_nz_idx, last_nz_idx):
            new_hist.TH1.SetBinContent(i + 1, new_hist.TH1.GetBinContent(i + 1))
            new_hist.TH1.SetBinError(i + 1, new_hist.TH1.GetBinError(i + 1))

        return new_hist

    def zero_outliers(self, zscore_threshold: float = 6) -> Histogram1D:
        """Sets outliers to zero (for exlusion from a chi-square fit)"""

        new_hist = deepcopy(self)

        zscores_val = (self.bin_values() - np.mean(self.bin_values())) / np.std(self.bin_values())
        zscores_err = (self.error() - np.mean(self.error())) / np.std(self.error())
        idx = np.argwhere(
            (abs(zscores_val) >= zscore_threshold) | (abs(zscores_err) >= zscore_threshold)
        )
        for i in idx:
            new_hist.view().value[i] = 0
            new_hist.view().variance[i] = 0
            new_hist.TH1.SetBinContent(i + 1, 0)
            new_hist.TH1.SetBinError(i + 1, 0)

        return new_hist

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
        out_filename: str | None = None,
        histtype: str = "step",
        show: bool = False,
        flow: bool = False,
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
        :param histtype: histogram type to be passed to mplhep
        :param show: whether to display the plot (plt.show())
        :param flow: whether to plot over/underflow
        :return: matplotlib axis object with plot
        """
        hist = self.copy()

        if scale_by_bin_width:
            self.logger.debug(f"Scaling histogram {self.name} by bin width...")
            hist /= hist.bin_widths  # type: ignore

        # normalise/scale
        if normalise is True:
            hist.normalise()
        elif normalise:
            hist.normalise_to(normalise)

        # set error
        if yerr is True:  # default to whatever error is saved in the TH1
            yerr = hist.error()
        elif isinstance(yerr, str) and (yerr == "sumw"):
            yerr = hist.root_sumw2()
        elif (yerr is not None) and not isinstance(yerr, bool) and not hasattr(yerr, "__len__"):
            raise TypeError(f"yerr should be a bool or iterable of values. Got {yerr}")

        if not ax:
            _, ax = plt.subplots()

        hep.histplot(
            H=hist.bin_values(flow),
            bins=hist.bin_edges,
            ax=ax,
            yerr=yerr,
            w2=hist.sumw2() if w2 else None,
            histtype=histtype,
            flow="show" if flow else None,
            **kwargs,
        )

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
        out_filename: str | None = None,
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
        label: str | None = None,
        fit: bool = False,
        name: str = "",
        out_filename: str | None = None,
        yax_lim: float | tuple[float, float] | None = None,
        display_stats: bool = True,
        fit_empty: bool = False,
        colour: str | None = None,
        exclude_outliers: bool = True,
        **kwargs,
    ) -> Histogram1D:
        """
        Plot (and properly format) ratio between this histogram and another.

        :param other: Other histogram
        :param ax: Axis to plot on. will create new if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2': sqrt(SumW2) errors
                     - 'binom': binomial error
                     - 'carry': carries fractional error from original histogram
                     - shape(N): array of for one-sided errors or list thereof
                     - shape(Nx2): array of for two-sided errors or list thereof
        :param normalise: Whether histograms are normalised before taking ratio
        :param label: Legend label
        :param fit: whether to fit to a 0-degree polynomial and display line, chi-square and p-value
        :param name: histogram name
        :param out_filename: provide filename to print. If not given, nothing is saved
        :param yax_lim: limit y-axis to 1 +/- {yax_lim}
        :param display_stats: whether to display the fit parameters on the plot
        :param fit_empty: Whether to fit on empty bins. If false, fits on non-empty bin range
        :param colour: plot colour. Defaults to black.
        :param exclude_outliers: whether to exclude outliers before fit
        :param kwargs: Args to pass to ax.errorbar()
        :return: axis object with plot
        """
        try:
            np.testing.assert_allclose(np.array(self.bin_edges), np.array(other.bin_edges))
        except AssertionError as e:
            raise ValueError("Bins do not match!") from e
        if not ax:
            _, ax = plt.subplots()

        ax.set_axisbelow(True)  # prevents grid from being drawn above lines
        ax.grid(visible=True, which="both", axis="y")

        # create ratio histogram
        if normalise:
            h_ratio = other.normalised() / self.normalised()
        elif yerr == "binom":
            h_ratio = other.divide_binom(self)
            yerr = True
        else:
            h_ratio = other / self

        # set name
        if name:
            h_ratio.name = name

        # set errors
        if yerr is True:  # default
            err = h_ratio.error()
        elif yerr == "sumw2":
            err = h_ratio.sumw2()
        elif yerr == "carry":
            err = (self.root_sumw2() / self.bin_values()) * h_ratio.bin_values()  # type: ignore
            for i in range(h_ratio.TH1.GetNbinsX()):
                h_ratio.TH1.SetBinError(i + 1, yerr[i])  # type: ignore
        elif isinstance(yerr, str):
            raise ValueError(f"Unknown error type {yerr}")
        else:
            err = np.zeros(self.n_bins)

        if fit:
            if h_ratio.TH1.GetEntries() == 0:
                self.logger.warning("Ratio histogram empty. Skipping fit.")
            else:
                h_ratio.do_fit(
                    ax=ax,
                    colour=colour,
                    display_stats=display_stats,
                    fit_empty=fit_empty,
                    exclude_outliers=exclude_outliers,
                )

        ax.axhline(1.0, linestyle="--", linewidth=1.0, c="k")
        ax.errorbar(
            h_ratio.bin_centres,
            h_ratio.bin_values(),
            xerr=h_ratio.bin_widths / 2,  # type: ignore
            yerr=err,
            linestyle="None",
            label=label,
            marker=".",
            c=colour if colour else "k",
            **kwargs,
        )

        if yax_lim:
            if isinstance(yax_lim, tuple):
                ymin = yax_lim[0]
                ymax = yax_lim[1]
            else:
                ymin = 1 - yax_lim
                ymax = 1 + yax_lim
            ax.set_ylim(ymin, ymax)

            # annotate points outside range with a little arrow
            for i, bin_val in enumerate(h_ratio.bin_values()):
                if (bin_val < ymin) or (bin_val > ymax):
                    is_above = bin_val > ymax

                    plt.annotate(
                        "",
                        xy=(h_ratio.bin_centres[i], ymax if is_above else ymin),  # type: ignore
                        xytext=(  # type: ignore
                            h_ratio.bin_centres[i],
                            ((ymax - ymin) * 0.7 + ymin)
                            if is_above
                            else ((ymax - ymin) * 0.3 + ymin),
                        ),
                        xycoords="data",
                        textcoords="data",
                        arrowprops={
                            "arrowstyle": "-|>",
                            "connectionstyle": "arc3,rad=0",
                            "color": colour if colour else "r",
                        },
                    )

        else:
            # I don't know why the matplotlib automatic yaxis scaling doesn't work, but I've done it for them here
            # ymax = np.max(np.ma.masked_invalid(h_ratio.bin_values() + (err / 2)))  # type: ignore
            # ymin = np.min(np.ma.masked_invalid(h_ratio.bin_values() - (err / 2)))  # type: ignore
            ymax = np.max(np.ma.masked_invalid(h_ratio.bin_values()))  # type: ignore
            ymin = np.min(np.ma.masked_invalid(h_ratio.bin_values()))  # type: ignore
            vspace = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - vspace, ymax + vspace)

        if out_filename:
            plt.savefig(out_filename, bbox_inches="tight")
            self.logger.info(f"image saved in {out_filename}")

        return h_ratio

    def do_fit(
        self,
        ax: plt.Axes | None = None,
        colour: str | tuple = "k",
        display_stats: bool = False,
        fit_empty: bool = False,
        exclude_outliers: bool = False,
    ) -> float | None:
        """Perform chi-square fit on histogram and optionally display on axis"""

        if not fit_empty:
            fit_hist = self.non_zero_range()
        else:
            fit_hist = deepcopy(self)

        if exclude_outliers:
            fit_hist = fit_hist.zero_outliers()

        with redirect_stdout() as fit_output:
            opt = "FSNMultithread"
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                opt += "V"  # verbose
            else:
                opt += "Q"  # quiet mode
            fit_results = fit_hist.TH1.Fit("pol0", "VFSNMultithread")

        if not fit_output.getvalue():
            self.logger.warning("Fit result is empty. Skipping..")
            return None

        self.logger.debug(
            f"ROOT fit output:\n"
            f"==========================================================================\n"
            f"{fit_output.getvalue()}"
            f"=========================================================================="
        )
        c = fit_results.Parameters()[0]
        fit_err = fit_results.Errors()[0]

        if ax is None:
            return c

        # display fit line
        fit_col = "r" if (colour is None) or (colour == "black") or (colour == "k") else colour
        ax.fill_between(
            [self.bin_edges[0], self.bin_edges[-1]],  # type: ignore
            [c - fit_err],
            [c + fit_err],
            color=fit_col,
            alpha=0.3,
        )
        ax.axhline(c, color=fit_col, linewidth=1.0)

        if display_stats:
            textstr = "\n".join(
                (
                    # r"$\chi^2=%.3f$" % fit_results.Chi2(),
                    # r"$\mathrm{NDF}=%.3f$" % fit_results.Ndf(),
                    r"$\epsilon=%.2f\pm%.3f$"
                    % (c, fit_err),
                )
            )
            # dumb workaround to avoid the stats boxes from overlapping eachother
            loc = "upper left"
            for artist in ax.get_children():
                if isinstance(artist, AnchoredText) and (
                    r"$\chi^2=" in artist.get_children()[0].get_text()
                ):
                    loc = "lower left"
            stats_box = AnchoredText(textstr, loc=loc, frameon=False, prop={"fontsize": "x-small"})
            ax.add_artist(stats_box)

        return c
