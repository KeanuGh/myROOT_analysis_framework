import numpy as np
import ROOT

from shadow_unfold.models import ResponseComponents
from src.analysis import Analysis
from utils.ROOT_utils import get_th1_bin_edges


def crop_to_nominal_binning(source: ROOT.TH1, target: ROOT.TH1, name: str) -> ROOT.TH1:
    """Clone source into target binning, dropping one leading shadow bin if present."""
    source_edges = get_th1_bin_edges(source)
    target_edges = get_th1_bin_edges(target)

    if len(source_edges) == len(target_edges) and all(
        abs(float(source_edge) - float(target_edge)) < 1e-6
        for source_edge, target_edge in zip(source_edges, target_edges, strict=True)
    ):
        clone = source.Clone(name)
        clone.SetDirectory(0)
        return clone

    has_one_leading_shadow_bin = len(source_edges) == len(target_edges) + 1 and all(
        abs(float(source_edge) - float(target_edge)) < 1e-6
        for source_edge, target_edge in zip(source_edges[1:], target_edges, strict=True)
    )
    if not has_one_leading_shadow_bin:
        raise ValueError(
            f"Cannot crop histogram '{source.GetName()}' to '{target.GetName()}': "
            "source is neither identical to target nor target plus one leading shadow bin."
        )

    cropped = ROOT.TH1D(name, source.GetTitle(), len(target_edges) - 1, target_edges)
    cropped.SetDirectory(0)
    for bin_i in range(1, target.GetNbinsX() + 1):
        cropped.SetBinContent(bin_i, source.GetBinContent(bin_i + 1))
        cropped.SetBinError(bin_i, source.GetBinError(bin_i + 1))
    return cropped


def covariance_from_hist(h: ROOT.TH1, name: str) -> ROOT.TH2D:
    """Build a diagonal covariance matrix from a histogram's bin errors."""
    cov = ROOT.TH2D(name, name, h.GetNbinsX(), 0, h.GetNbinsX(), h.GetNbinsX(), 0, h.GetNbinsX())
    cov.SetDirectory(0)
    for bin_i in range(1, h.GetNbinsX() + 1):
        cov.SetBinContent(bin_i, bin_i, h.GetBinError(bin_i) ** 2)
    return cov


def unfold_histogram(
    analysis: Analysis,
    hist: ROOT.TH1,
    response: ResponseComponents,
    iter_count: int,
) -> tuple[ROOT.TH1, ROOT.TH2]:
    """Unfold one histogram and return the unfolded histogram plus covariance."""
    if iter_count == 0 and response.reco.GetNbinsX() == response.truth.GetNbinsX():
        result = analysis.unfold_bin_by_bin(hist, response.reco, response.truth)
        return result.unfolded, covariance_from_hist(result.unfolded, f"{hist.GetName()}_cov")

    if iter_count == 0:
        unfolded = ROOT.RooUnfoldBinByBin(response.response, hist)
    else:
        unfolded = ROOT.RooUnfoldBayes(response.response, hist, iter_count)
    return unfolded.Hunfold(), ROOT.TH2D(unfolded.Eunfold())


def scale_and_crop_unfolded(
    hist: ROOT.TH1,
    nominal_truth: ROOT.TH1,
    name: str,
    lumi: float,
) -> ROOT.TH1:
    """Convert unfolded event yields to cross-section and crop any leading shadow bin."""
    scaled = hist.Clone(f"{name}_scaled")
    scaled.SetDirectory(0)
    scaled.Scale(1 / lumi)
    return crop_to_nominal_binning(scaled, nominal_truth, name)


def closure_metrics(unfolded_signal: ROOT.TH1, truth: ROOT.TH1) -> tuple[float, float, float]:
    """Return mean deviation, max deviation, and integral ratio against truth."""
    deviations = []
    for bin_i in range(1, truth.GetNbinsX() + 1):
        truth_val = truth.GetBinContent(bin_i)
        if truth_val == 0:
            continue
        deviations.append(abs(unfolded_signal.GetBinContent(bin_i) / truth_val - 1))
    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    max_dev = float(np.max(deviations)) if deviations else 0.0
    integral_ratio = (
        unfolded_signal.Integral() / truth.Integral() if truth.Integral() != 0 else float("nan")
    )
    return mean_dev, max_dev, float(integral_ratio)
