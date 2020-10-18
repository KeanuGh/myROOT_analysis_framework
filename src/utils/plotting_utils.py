import boost_histogram as bh
import numpy as np
from typing import Tuple, Optional
from utils.axis_labels import labels_xs
from warnings import warn


def scale_to_crosssection(hist: bh.Histogram, luminosity) -> None:
    """Scales histogram to cross-section. Currently undefined for multidimensional histograms"""
    if len(hist.axes) != 1:
        raise Exception("Currently undefined behaviour for multi-dimentional histograms")
    hist /= luminosity
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
             f"Axis labels blank.", UserWarning)
        xlabel = None
        ylabel = None
    return xlabel, ylabel
