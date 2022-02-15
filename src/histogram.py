from __future__ import annotations

import logging
from typing import Union, List, Tuple, Any, overload

import ROOT
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike

from src.logger import get_logger
from utils.context import redirect_stdout

# settings
ROOT.TH1.AddDirectory(False)  # stops TH1s from being saved and prevents overwrite warnings
ROOT.TH1.SetDefaultSumw2()  # Sets weighted binning in all ROOT histograms by default
ROOT.gROOT.SetBatch()  # Prevents TCanvas popups
plt.style.use(hep.style.ATLAS)  # set atlas-style plots
np.seterr(invalid='ignore')  # ignore division by zero errors


# TODO: 2D hist
class Histogram1D(bh.Histogram, family=None):
    """
    Wrapper around boost-histogram histogram for 1D
    """
    @overload
    def __init__(self, bins: List[float], logger: logging.Logger = None, name: str = '', title: str = '') -> None:
        ...

    @overload
    def __init__(self,
                 bins: Tuple[int, float, float],
                 logbins: bool = False,
                 logger: logging.Logger = None,
                 name: str = '',
                 title: str = ''
                 ) -> None:
        ...

    @overload
    def __init__(self,
                 var: ArrayLike = None,
                 bins: Union[List[float], Tuple[int, float, float], bh.axis.Axis] = (10, 0, 10),
                 weight: Union[ArrayLike, int] = None,
                 logbins: bool = False,
                 logger: logging.Logger = None,
                 name: str = '',
                 title: str = '',
                 **kwargs
                 ) -> None:
        ...

    def __init__(self,
                 var: ArrayLike = None,
                 bins: Union[List[float], Tuple[int, float, float], bh.axis.Axis] = (10, 0, 10),
                 weight: Union[ArrayLike, float] = None,
                 logbins: bool = False,
                 logger: logging.Logger = None,
                 name: str = '',
                 title: str = '',
                 **kwargs
                 ):
        """
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param var: iterable of values to fill histogram 
        :param weight: iterable of weights corresponding to each variable value or value to weight by
        :param logbins: whether logarithmic binnings
        :param kwargs: keyword arguments to pass to boost_histogram.Histogram()
        """
        # to avoid issues with copying
        if isinstance(var, bh._core.hist.any_weight):
            super().__init__(var)

        else:
            if logger is None:
                logger = get_logger()
            self.logger = logger

            self.logger.debug(f"Initialising histogram {name}...")

            # TH1
            self.TH1 = ROOT.TH1F(name, title, *self.__get_TH1_bins(bins))

            self.name = name

            # get axis
            axis = (
                bins if isinstance(bins, bh.axis.Axis)
                else self.__gen_axis(bins, logbins)
            )
            super().__init__(
                axis,
                storage=bh.storage.Weight(),
                **kwargs
            )
            if var is not None:
                self.Fill(var, weight=weight)

    def Fill(self, var: ArrayLike, weight: ArrayLike = None) -> Histogram1D:
        self.logger.debug(f"Filling histogram {self.name} with {len(var)} events..")
        super().fill(var, weight=weight, threads=0)

        if weight is None:
            weight = np.ones(len(var))
        for v, w in zip(var, weight):
            self.TH1.Fill(v, w)
        return self

    def __truediv__(self, other: Union[bh.Histogram, "np.typing.NDArray[Any]", float]) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        result = self.copy()
        return result.__itruediv__(other)

    def __itruediv__(self, other: Union[bh.Histogram, "np.typing.NDArray[Any]", float]) -> Histogram1D:
        """boost-histogram doesn't allow dividing weighted histograms so implement that here"""
        if isinstance(other, Histogram1D):
            # Scale variances based on ROOT method. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l02929
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            clsq = c1 * c1
            variance = (self.view(flow=True).variance * clsq + other.view(flow=True).variance * c0 * c0) / (clsq * clsq)

            # set division by zero to zero
            self.view(flow=True).value = np.divide(self.view(flow=True).value, other.view(flow=True).value,
                                                   out=np.zeros_like(self.view(flow=True).value),
                                                   where=other.view(flow=True).value != 0)
            self.view(flow=True).variance = variance
            self.TH1.Divide(other.TH1)
            return self
        else:
            # scale TH1 properly
            if hasattr(other, '__iter__'):
                for i, val in enumerate(other):
                    self.TH1.SetBinContent(i, self.TH1.GetBinContent(i) / val)
                    self.TH1.SetBinError(i, self.TH1.GetBinError(i) / val)
            else:
                self.TH1.Scale(1 / other)
            # let boost-histogram handle return
            return self._compute_inplace_op("__itruediv__", other)

    def __mul__(self, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        """boost-histogram doesn't allow multiplying weighted histograms so implement that here"""
        result = self.copy()
        return result.__imul__(other)

    def __imul__(self, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        """boost-histogram doesn't allow multiplying weighted histograms so implement that here"""
        if isinstance(other, Histogram1D):
            # Scale variances based on ROOT method. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l06116
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            variance = self.view(flow=True).variance * c1 * c1 + other.view(flow=True).variance * c0 * c0

            self.view(flow=True).value.__imul__(other.view(flow=True).value)
            self.view(flow=True).variance = variance
            self.TH1.Multiply(other.TH1)
            return self
        else:
            # scale TH1 properly
            if hasattr(other, '__iter__'):
                for i, val in enumerate(other):
                    self.TH1.SetBinValue(i, self.TH1.GetBinContent(i) * val)
                    self.TH1.SetBinError(i, self.TH1.GetBinError(i) * val)
            else:
                self.TH1.Scale(other)
            return self._compute_inplace_op("__imul__", other)

    @staticmethod
    @overload
    def __gen_axis(bins: List[float]) -> bh.axis.Axis:
        ...

    @staticmethod
    @overload
    def __gen_axis(bins: Tuple[int, float, float], logbins: bool = False) -> bh.axis.Axis:
        ...

    @staticmethod
    def __gen_axis(bins: Union[List[float], Tuple[int, float, float]], logbins: bool = False) -> bh.axis.Axis:
        """
        Returns the correct type of boost-histogram axis based on the input bins.

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
        :param logbins: whether logarithmic bins. Silently ignored if using variable bins.
        """
        if isinstance(bins, tuple):
            if len(bins) != 3:
                raise ValueError("Tuple of bins should be formatted like (n_bins, start, stop).")
            return bh.axis.Regular(*bins, transform=bh.axis.transform.log if logbins else None)

        elif isinstance(bins, list):
            return bh.axis.Variable(bins)

        else:
            raise TypeError(f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. "
                            f"Got {bins} of type {type(bins)}")

    @staticmethod
    def __get_TH1_bins(bins: Union[List[float], Tuple[int, float, float], bh.axis.Axis]
                       ) -> Union[Tuple[int, list], Tuple[int, float, float]]:
        """Format bins for TH1 constructor"""
        if isinstance(bins, list):
            return len(bins), bins
        elif isinstance(bins, tuple):
            if len(bins) != 3:
                raise ValueError("Tuple of bins should be formatted like (n_bins, start, stop).")
            return bins
        elif isinstance(bins, bh.axis.Axis):
            return bins.size, bins.edges

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
    def bin_edges(self,) -> np.array:
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

    @property
    def eff_entries(self) -> float:
        """get effective number of entries"""
        return self.bin_sum() * self.bin_sum() / sum(self.sumw2())

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
    # properties beginning with 'R' are calculated using ROOT
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
        return self.normalised_to(1.)

    def normalised_to(self, factor: float) -> Histogram1D:
        """Return histogram normalised to factor"""
        return self.scaled(factor / self.bin_sum())

    def scaled(self, scale_factor: float) -> Histogram1D:
        """Return rescaled histogram"""
        return self.copy() * scale_factor

    def normalise(self) -> None:
        """Normalise histogram to unity inplace"""
        self.normalise_to(1.)

    def normalise_to(self, factor: float) -> None:
        """Normalise histogram to factor inplace"""
        self.scale(factor / self.bin_sum())

    def scale(self, scale_factor: float) -> None:
        """Return rescaled histogram"""
        self.__imul__(scale_factor)

    # Fitting
    # ===================
    def chi_square(self, other: Union[ROOT.TF1, ROOT.TH1, Histogram1D]) -> Tuple[float, float]:
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
    def to_TH1(self, name: str = 'name', title: str = 'title') -> ROOT.TH1F:
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
            yerr: Union[ArrayLike, bool] = True,
            w2: bool = False,
            normalise: Union[float, bool] = False,
            scale_by_bin_width: bool = False,
            stats_box: bool = False,
            **kwargs
    ) -> None:
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
        :return: None
        """
        # normalise to value or unity
        if not normalise:
            self.logger.debug(f"Plotting histogram {self.name}...")
            hist = self.copy()
        elif normalise is True:
            self.logger.debug(f"Plotting histogram {self.name} normalised to unity...")
            hist = self.normalised()
        else:
            self.logger.debug(f"Plotting normalised histogram {self.name} normalised to {normalise}...")
            hist = self.normalised_to(normalise)

        # set error
        if yerr is True:
            yerr = hist.root_sumw2()
        elif not hasattr(yerr, '__len__'):
            raise TypeError(f"yerr should be a bool or iterable of values. Got {yerr}")

        bin_vals = hist.bin_values()
        if scale_by_bin_width:
            bin_vals /= hist.bin_widths
            if hasattr(yerr, '__len__'):
                yerr /= hist.bin_widths

        hep.histplot(hist, ax=ax, yerr=yerr, w2=hist.sumw2() if w2 else None, **kwargs)

        if stats_box:
            # dumb workaround to avoid the stats boxes from overlapping eachother
            xy = (.75, .61)
            for artist in ax.get_children():
                if isinstance(artist, plt.Text):
                    if r'$\mu=' in artist.get_text():
                        xy = (.75, .38)

            textstr = '\n'.join((
                r"$\mathbf{" + self.name + "}$",
                r'$\mu=%.2f\pm%.2f$' % (self.mean, self.mean_error),
                r'$\sigma=%.2f\pm%.2f$' % (self.std, self.std_error),
                r'$\mathrm{Entries}: %.0f$' % self.n_entries))
            ax.text(x=xy[0], y=xy[1], s=textstr, transform=ax.transAxes, fontsize='small')

    def plot_ratio(
            self,
            other: Histogram1D,
            ax: plt.Axes = None,
            yerr: Union[ArrayLike, bool] = True,
            normalise: bool = False,
            label: str = None,
            fit: bool = False,
            **kwargs
    ) -> None:
        """
        Plot (and properly format) ratio between this histogram and another.

        :param other: Other histogram
        :param ax: Axis to plot on. will create new if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param normalise: Whether histograms are normalised before taking ratio
        :param label: Legend label
        :param fit: whether to fit to a 0-degree polynomial and display line, chi-square and p-value
        :param kwargs: Args to pass to ax.errorbar()
        :return: None
        """
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Bins do not match!")
        if not ax:
            _, ax = plt.subplots()

        if normalise:
            h_ratio = self.normalised() / other.normalised()
        else:
            h_ratio = self / other

        if yerr is True:
            yerr = h_ratio.root_sumw2()

        if fit:
            self.logger.info("Performing fit on ratio..")
            with redirect_stdout() as fit_output:
                fit_results = h_ratio.TH1.Fit('pol0', 'VFSN0')
            self.logger.debug(f"ROOT fit output: \n"
                              f"==========================================================================\n"
                              f"{fit_output.getvalue()}"
                              f"==========================================================================\n")
            c = fit_results.Parameters()[0]
            err = fit_results.Errors()[0]
            ax.fill_between([self.bin_edges[0], self.bin_edges[-1]], [c - err], [c + err], color='r', alpha=0.3)
            ax.axhline(c, color='r', linewidth=1.)
            textstr = '\n'.join((
                r'$\chi^2=%.2f$' % fit_results.Chi2(),
                r'$\mathrm{NDF}=%.2f$' % fit_results.Ndf(),
                r'$c=%.2f\pm%.2f$' % (c, err))
            )
            stats_box = AnchoredText(textstr, loc='upper left', frameon=False, prop=dict(fontsize="small"))
            ax.add_artist(stats_box)

        ax.axhline(1., linestyle='--', linewidth=1., c='k')
        ax.errorbar(h_ratio.bin_centres, h_ratio.bin_values(), xerr=h_ratio.bin_widths / 2, yerr=yerr,
                    linestyle='None', label=label, **kwargs)
        ax.grid(visible=True, which='both', axis='y')

        # POSSIBLY NOT NEEDED BUT IT TOOK ME A WHOLE DAY TO FIGURE THIS OUT SO I'M KEEPING IT
        # relimit y axes
        # ymax = ymin = 1
        # for i, line in enumerate(ax.lines):
        #     data = line.get_ydata()
        #     if len(data) < 1: continue  # why is there a line with no data appearing??
        #     ymax = np.max(np.ma.masked_invalid(np.append(data, ymax)))
        #     ymin = np.min(np.ma.masked_invalid(np.append(data, ymin)))
        # vspace = (ymax - ymin) * 0.1
        # ax.set_ylim(bottom=ymin - vspace, top=ymax + vspace)
