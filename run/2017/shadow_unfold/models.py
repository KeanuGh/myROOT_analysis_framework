from dataclasses import dataclass

import ROOT


@dataclass(frozen=True)
class FakeControlRegion:
    """Reco-region used to derive fake factors before applying them in the SR."""

    selection_tag: str
    output_tag: str
    cuts: tuple
    shared_across_configs: bool


@dataclass(frozen=True)
class ShadowConfig:
    """One phase-space definition used in the closure test."""

    label: str
    unfolded_var: str | None
    mtw_min: float
    taupt_min: float
    met_min: float


@dataclass(frozen=True)
class ResponseComponents:
    """Response pieces needed by RooUnfold and closure diagnostics."""

    response: ROOT.RooUnfoldResponse
    reco: ROOT.TH1
    truth: ROOT.TH1
    matrix: ROOT.TH2
