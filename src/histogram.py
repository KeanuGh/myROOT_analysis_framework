from __future__ import annotations

from typing import Union, List, Tuple, Any, overload

import ROOT
import boost_histogram as bh
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

plt.style.use(hep.style.ATLAS)  # set atlas-style plots
np.seterr(invalid='ignore')  # ignore division by zero errors


# TODO: 2D hist
class Histogram1D(bh.Histogram, family=None):
    """
    Wrapper around boost-histogram histogram for 1D
    """

    @overload
    def __init__(self, bins: List[float]) -> None:
        ...

    @overload
    def __init__(self, bins: Tuple[int, float, float], logbins: ...) -> None:
        ...

    @overload
    def __init__(self,
                 var: ArrayLike = ...,
                 bins: Union[List[float], Tuple[int, float, float], bh.axis.Axis] = (10, 0, 10),
                 logbins: bool = ...,
                 weight: Union[ArrayLike, int] = ...,
                 **kwargs
                 ) -> None:
        ...

    def __init__(self,
                 var: ArrayLike = None,
                 bins: Union[List[float], Tuple[int, float, float], bh.axis.Axis] = (10, 0, 10),
                 weight: Union[ArrayLike, float] = None,
                 logbins: bool = False,
                 **kwargs
                 ):
        """
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param var: iterable of values to fill histogram 
        :param weight: iterable of weights corresponding to each variable value or value to weight by
        :param logbins: whether logarithmic binnings
        :param kwargs: keyword arguments that would be passed to boost_histogram.Histogram()
        """
        # to avoid issues with copying
        if isinstance(var, bh._core.hist.any_weight):
            super().__init__(var)

        else:
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
                self.fill(var, weight=weight, threads=0)

    def __truediv__(self: Histogram1D, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        result = self.copy(deep=False)
        return result.__itruediv__(other)

    def __itruediv__(self: Histogram1D, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        if isinstance(other, Histogram1D):
            # Scale variances. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l02929
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            clsq = c1 * c1
            variance = (self.view(flow=True).variance * clsq + other.view(flow=True).variance * c0 * c0) / (clsq * clsq)

            self.view(flow=True).variance = variance
            self.view(flow=True).value /= other.view(flow=True).value
            return self
        else:
            return self._compute_inplace_op("__itruediv__", other)

    def __mul__(self: Histogram1D, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        result = self.copy(deep=False)
        return result.__imul__(other)

    def __rmul__(self: Histogram1D, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        return self * other

    def __imul__(self: Histogram1D, other: Union["bh.Histogram", "np.typing.NDArray[Any]", float]) -> Histogram1D:
        if isinstance(other, Histogram1D):
            # Scale variances. See https://root.cern.ch/doc/master/TH1_8cxx_source.html#l06116
            c0 = self.view(flow=True).value
            c1 = other.view(flow=True).value
            variance = self.view(flow=True).variance * c1 * c1 + other.view(flow=True).variance * c0 * c0

            self.view(flow=True).variance = variance
            self.view(flow=True).value *= other.view(flow=True).value
            return self
        else:
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
    def __gen_axis(bins: Union[List[float], Tuple[int, float, float]],
                   logbins: bool = False
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
                raise ValueError("Transforms cannot be passed to variable bin types")
            return bh.axis.Variable(bins)

        else:
            raise TypeError(f"Bins must be formatted as either tuple (n_bins, start, stop) or a list of bin edges. "
                            f"Got {bins} of type {type(bins)}")

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
    def bin_edges(self,) -> np.array:
        """get bin edges"""
        return self.axes[0].edges

    @property
    def bin_centres(self) -> np.array:
        """get bin centres"""
        return self.axes[0].centers

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

    @property
    def bin_sum(self) -> float:
        """Get sum of bin contents"""
        return sum(self.bin_values())

    # Scaling
    # ===================
    def normalised(self) -> Histogram1D:
        """Return histogram normalised to area"""
        return self.scaled(1 / self.bin_sum)

    def scaled(self, scale_factor: float) -> Histogram1D:
        """Return rescaled histogram"""
        return self.copy() * scale_factor

    # Fitting
    # ===================
    def chi_square_fit(self, other: Union[ROOT.TF1, ROOT.TH1, Histogram1D]) -> Tuple[float, float]:
        """Perform chi-squared test. Retun chi2 per degree of freedom, pvalue"""
        h1 = self.to_TH1()
        if isinstance(other, ROOT.TH1):
            return h1.Chi2Test(other, "CHI2/NDF"), h1.Chi2Test(other, "WW")
        elif isinstance(other, Histogram1D):
            h2 = other.to_TH1()
            return h1.Chi2Test(h2, "CHI2/NDF"), h1.Chi2Test(h2, "WW")
        elif isinstance(other, ROOT.TF1):
            return h1.Chisquare(other, "CHI2/NDF"), h1.Chisquare(other, "WW")
        else:
            raise TypeError(f"{type(other)} is an incorrect type for a chi-square test")

    # Conversion
    # ===================
    def to_TH1(self, name: str = 'name', title: str = 'title') -> ROOT.TH1F:
        """Convert Histogram to ROOT TH1F"""
        try:
            h_root = ROOT.TH1F(name, title, self.n_bins, self.bin_edges)
        except TypeError as e:
            raise e

        # fill TH1
        for idx, bin_cont in np.ndenumerate(self.view(flow=True)):
            h_root.SetBinContent(*idx, bin_cont[0])  # bin value
            h_root.SetBinError(*idx, np.sqrt(bin_cont[1]))  # root sum of weights

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
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: None
        """
        # normalise to value or unity
        if not normalise:
            hist = self.copy()
        elif normalise is True:
            hist = self.normalised()
        elif isinstance(normalise, (float, int)):
            hist = self.scaled(normalise)
        else:
            raise TypeError("'normalise' must be a float, int or boolean")

        if yerr is True:
            yerr = hist.root_sumw2()
        elif not hasattr(yerr, '__len__'):
            raise TypeError(f"yerr should be a bool or iterable of values. Got {yerr}")

        bin_vals = hist.bin_values
        if scale_by_bin_width:
            bin_vals /= hist.bin_widths
            if hasattr(yerr, '__len__'):
                yerr /= hist.bin_widths

        hep.histplot(bin_vals, bins=hist.bin_edges, ax=ax, yerr=yerr, w2=hist.sumw2() if w2 else None, **kwargs)

    def plot_ratio(
            self,
            other: Histogram1D,
            ax: plt.Axes = None,
            yerr: Union[ArrayLike, bool] = True,
            normalise: bool = False,
            label: str = None,
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
        :param kwargs: Args to pass to ax.errorbar()
        :return: None
        """
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Bins do not match!")
        if not ax:
            _, ax = plt.subplots()

        h_ratio = self / other
        if normalise:
            h_ratio = h_ratio.normalised()

        if yerr is True:
            yerr = h_ratio.root_sumw2()

        ax.errorbar(h_ratio.bin_centres, h_ratio.bin_values(), xerr=h_ratio.bin_widths / 2, yerr=yerr,
                    linestyle='None', label=label, **kwargs)
        ax.axhline(1., linestyle='--', linewidth=1., c='k')

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
