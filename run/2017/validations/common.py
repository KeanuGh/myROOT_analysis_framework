from __future__ import annotations

import sys
from pathlib import Path

import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from samples import NOMINAL_NAME  # noqa: E402

YEAR = 2017
WP = "medium"
FAKES_SOURCE = "TauPt"
VALIDATION_OUTPUT = REPO_ROOT / "outputs" / "validate_shadow_fakes"
SIDEBAND_OUTPUT = VALIDATION_OUTPUT / "sideband_transfer"
SIDEBAND_HIST_CACHE = SIDEBAND_OUTPUT / "root" / "validate_shadow_fakes_sideband_transfer.root"
MEASURED_HIST_CACHE = (
    REPO_ROOT
    / "outputs"
    / "analysis_shadow_unfold"
    / "measured"
    / "root"
    / "analysis_shadow_unfold_measured.root"
)
MC_SAMPLES = ("wtaunu_had", "wtaunu_lep", "wlnu", "zll", "top", "diboson")


def hist_integral(hist: ROOT.TH1) -> float:
    return float(hist.Integral())


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else float("nan")


def fake_factor_bin_health(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
) -> tuple[int, int]:
    denominator_total = sum(
        abs(denominator.GetBinContent(bin_idx))
        for bin_idx in range(1, denominator.GetNbinsX() + 1)
    )
    tiny_threshold = max(10.0, 0.01 * denominator_total)
    negative_numerator_bins = 0
    tiny_denominator_bins = 0
    for bin_idx in range(1, denominator.GetNbinsX() + 1):
        if numerator.GetBinContent(bin_idx) < 0:
            negative_numerator_bins += 1
        denominator_value = denominator.GetBinContent(bin_idx)
        if denominator_value <= 0 or abs(denominator_value) < tiny_threshold:
            tiny_denominator_bins += 1
    return negative_numerator_bins, tiny_denominator_bins


def get_root_hist(root_file: Path, hist_path: str) -> ROOT.TH1:
    with ROOT.TFile(str(root_file), "READ") as file:
        hist = file.Get(hist_path)
        if not hist:
            raise KeyError(f"Missing histogram '{hist_path}' in {root_file}")
        cloned = hist.Clone()
        cloned.SetDirectory(0)
        return cloned


def dataset_hist_path(selection: str, variable: str) -> str:
    return f"{NOMINAL_NAME}/{selection}/{variable}"


def sideband_dataset_file(dataset: str) -> Path:
    return SIDEBAND_OUTPUT / "root" / f"{dataset}.root"


def sideband_transfer_hist(hist_name: str) -> ROOT.TH1:
    return get_root_hist(SIDEBAND_HIST_CACHE, hist_name)


def sideband_dataset_hist(dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(sideband_dataset_file(dataset), dataset_hist_path(selection, variable))


def sum_hists(hists: list[ROOT.TH1]) -> ROOT.TH1:
    if not hists:
        raise ValueError("Cannot sum an empty histogram list")
    total = hists[0].Clone()
    total.SetDirectory(0)
    for hist in hists[1:]:
        total.Add(hist)
    return total


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
