from typing import Union, List, Tuple, overload

import boost_histogram as bh
import mplhep as hep
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

plt.style.use(hep.style.ATLAS)


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
                 bins: Union[List[float], Tuple[int, float, float]],
                 logbins: bool = ...,
                 weight: Union[ArrayLike, int] = ...,
                 var: ArrayLike = ...,
                 **kwargs
                 ) -> None:
        ...

    def __init__(self,
                 bins: Union[List[float], Tuple[int, float, float]],
                 var: ArrayLike = None,
                 weight: Union[ArrayLike, int] = None,
                 logbins: bool = False,
                 **kwargs
                 ):
        """
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param var:
        :param logbins: whether logarithmic binnings
        :param kwargs: keyword arguments that would be passed to numpy.histogram()
        """
        super().__init__(
            self._get_axis(bins, logbins),
            storage=bh.storage.Weight(),
            **kwargs
        )
        if var is not None:
            self.fill(var, weight=weight, threads=0)
            return
        else:
            return

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
                            f"Got {bins}")

    @property
    def sumw2(self) -> np.array:
        """get squared sum of weights"""
        return self.view().variance

    @property
    def root_sumw2(self) -> np.array:
        """get squared sum of weights"""
        return np.sqrt(self.view().variance)

    @property
    def bin_integrals(self) -> np.array:
        """get areas of each bin"""
        return np.prod(self.axes.widths)

    @property
    def integral(self):
        """get integral or area of histogram"""
        return sum(self.bin_integrals)

    def plot(self,
             ax: plt.Axes,
             yerr: Union[ArrayLike, bool] = None,
             w2: bool = False,
             normalise: Union[float, bool] = False,
             scale_by_bin_width: bool = False,
             **kwargs
             ) -> None:
        """
        Plot histogram on axis ax

        :param ax: matplotlib Axes object to plot on
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - True, sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param w2: Whether to do a poissonian interval error calculation based on weights
        :param normalise: value to normalise to, or 'True', for a normalisation of 1
        :param scale_by_bin_width: whether to scale histogram by bin widths
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: None
        """
        hist = self.copy()

        if scale_by_bin_width:
            hist /= hist.axes[0].widths

        if w2:
            sumw2 = hist.sumw2 if w2 else None
            if normalise:
                sumw2 /= hist.areas
        else: sumw2 = None

        if isinstance(normalise, float) or isinstance(normalise, int):
            bin_vals = normalise * hist.values()
        else: bin_vals = hist.values()

        hep.histplot(bin_vals, bins=hist.axes[0].edges, ax=ax, yerr=yerr, w2=sumw2, density=bool(normalise), **kwargs)
