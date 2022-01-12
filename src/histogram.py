from __future__ import annotations

from typing import Union, List, Tuple, overload

import boost_histogram as bh
import mplhep as hep
import numpy as np
from boost_histogram._core.hist import any_weight
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

plt.style.use(hep.style.ATLAS)


# TODO: 2D hist
class Histogram1D(bh.Histogram):
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
        if isinstance(var, any_weight):
            # to avoid issues with copying
            super().__init__(var)

        else:
            # get axis
            axis = (
                self._get_axis(bins, logbins)
                if not isinstance(bins, bh.axis.Axis) else bins
            )

            super().__init__(
                axis,
                storage=bh.storage.Weight(),
                **kwargs
            )
            if var is not None:
                self.fill(var, weight=weight, threads=0)

    @staticmethod
    @overload
    def _get_axis(bins: List[float]) -> bh.axis.Axis:
        ...

    @staticmethod
    @overload
    def _get_axis(bins: Tuple[int, float, float], logbins: bool = False) -> bh.axis.Axis:
        ...

    @staticmethod
    def _get_axis(bins: Union[List[float], Tuple[int, float, float]],
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

    @property
    def sumw2(self) -> np.array:
        """get squared sum of weights"""
        return self.view().variance

    @property
    def root_sumw2(self) -> np.array:
        """get squared sum of weights"""
        return np.sqrt(self.sumw2)

    @property
    def bin_widths(self) -> np.array:
        """get bin widths"""
        return self.axes[0].widths

    @property
    def bin_edges(self) -> np.array:
        """get bin edges"""
        return self.axes[0].edges

    @property
    def bin_centres(self) -> np.array:
        """get bin centres"""
        return self.axes[0].centers

    @property
    def bin_values(self) -> np.array:
        """get bin values"""
        return self.values()

    @property
    def integral(self) -> float:
        """get integral of histogram"""
        return sum(self.bin_values)

    def normalised(self) -> Histogram1D:
        """Return histogram normalised to area"""
        return self.rescaled(self.integral)

    def rescaled(self, scale_factor: float) -> Histogram1D:
        """Return rescaled histogram"""
        return self.copy() / scale_factor

    def plot(self,
             ax: plt.Axes = None,
             yerr: Union[ArrayLike, str] = 'rsumw2',
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
            hist = self.rescaled(normalise)
        else:
            raise TypeError("'normalise' must be a float, int or boolean")

        if w2:
            sumw2 = hist.sumw2
        else: sumw2 = None

        if yerr is None:
            pass
        elif isinstance(yerr, str):
            if yerr == 'rsumw2':
                yerr = hist.root_sumw2
            elif yerr == 'sqrtN':
                yerr = True
            else:
                raise ValueError(f"Valid yerrs: 'rsumw2', 'sqrtN'. Got {yerr}")
        elif not hasattr(yerr, '__len__'):
            raise TypeError(f"Valid yerr string values: 'rsumw2', 'sqrtN' or iterable of values. Got {yerr}")

        bin_vals = hist.bin_values
        if scale_by_bin_width:
            bin_vals /= hist.bin_widths
            if hasattr(yerr, '__len__'):
                yerr /= hist.bin_widths

        hep.histplot(bin_vals, bins=hist.bin_edges, ax=ax, yerr=yerr, w2=sumw2, **kwargs)

    def plot_ratio(
            self,
            other: Histogram1D,
            ax: plt.Axes = None,
            yerr: Union[ArrayLike, str] = 'rsumw2',
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

        if normalise:
            h = self.normalised()
            o = other.normalised()
        else:
            h = self.copy()
            o = other.copy()
        vals = h.bin_values / o.bin_values

        if yerr is None:
            pass
        elif isinstance(yerr, str):
            # propagate errors
            if yerr == 'rsumw2':
                yerr = vals * np.sqrt((h.root_sumw2 / h.bin_values) ** 2
                                      + (o.root_sumw2 / o.bin_values) ** 2)
            elif yerr == 'sqrtN':
                yerr = vals * np.sqrt((1 / h.bin_values) + (1 / o.bin_values))
            else:
                raise ValueError(f"Valid yerr string values: 'rsumw2', 'sqrtN'. Got {yerr}")
        elif not hasattr(yerr, '__len__'):
            raise TypeError(f"Valid yerrs: 'rsumw2', 'sqrtN' or iterable of values. Got {yerr}")

        ax.errorbar(h.bin_centres, vals, xerr=h.bin_widths / 2, yerr=yerr,
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
