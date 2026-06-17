from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import ROOT


class HistResultPtr(Protocol):
    """Lazy ROOT histogram result returned by RDataFrame actions."""

    def GetValue(self) -> ROOT.TH1: ...


FloatArray = np.typing.NDArray[np.float64]


@dataclass(slots=True)
class AnalysisPath:
    """Container for paths needed by an analysis."""

    output_dir: Path

    plot_dir: Path = field(init=False, default_factory=Path)
    latex_dir: Path = field(init=False, default_factory=Path)
    root_dir: Path = field(init=False, default_factory=Path)
    log_dir: Path = field(init=False, default_factory=Path)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.plot_dir = Path(self.output_dir) / "plots"
        self.root_dir = Path(self.output_dir) / "root"
        self.latex_dir = Path(self.output_dir) / "LaTeX"
        self.log_dir = Path(self.output_dir) / "logs"

    def __setattr__(self, key, value):
        value = Path(value)
        value.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, key, value)


@dataclass(slots=True)
class BinByBinUnfoldingResult:
    """Container for explicit bin-by-bin unfolding outputs."""

    unfolded: ROOT.TH1
    correction: ROOT.TH1
