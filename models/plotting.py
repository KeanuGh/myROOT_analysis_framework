from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import ROOT

from src.histogram import Histogram1D


class PlotOpts(TypedDict):
    """Per-histogram options for plotting."""

    vals: list[str | Histogram1D | ROOT.TH1]
    hists: list[Histogram1D]
    datasets: list[str | None]
    systematics: list[str | None]
    selections: list[str]
    uncert: list[np.typing.NDArray[np.float64]]
    labels: list[str]
    colours: list[str | tuple[float, ...]]
    linestyles: list[str]
    histstyles: list[str]


PlotKwargs = dict[str, Any]


@dataclass(slots=True)
class ProfileOpts:
    """
    Options for building ROOT profile from RDataFrame columns.

    :param x: x-axis column name.
    :param y: y-axis column name.
    :param weight: name of column to apply as weight.
    :param option: option paramter to pass to `TProfile1DModel()`
        (see https://root.cern.ch/doc/master/classTProfile.html#a1ff9340284c73ce8762ab6e7dc0e6725)
    """

    x: str
    y: str
    weight: str = ""
    option: str = ""


@dataclass(slots=True)
class Hist2dOpts:
    """
    Options for building ROOT 2D histogram from RDataFrame columns.

    :param x: x-axis column name.
    :param y: y-axis column name.
    :param weight: name of column to apply as weight.
    """

    x: str
    y: str
    weight: str = ""
