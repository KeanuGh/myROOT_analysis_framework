from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import MC_SAMPLES, VALIDATION_OUTPUT, hist_integral, ratio, write_markdown  # noqa: E402
from samples import NOMINAL_NAME  # noqa: E402

VARIABLE = "MTW"
FAKES_SOURCE = "TauPt"
SUMMARY_PATH = VALIDATION_OUTPUT / "coarse_3prong" / "coarse_3prong_fake_summary.md"


@dataclass(frozen=True)
class CachedFakeTest:
    label: str
    cache_file: Path
    dataset_root_dir: Path
    prefix: str
    validate_pass_selection: str


TESTS = (
    CachedFakeTest(
        label="independent MET split",
        cache_file=VALIDATION_OUTPUT
        / "sideband_transfer"
        / "root"
        / "validate_shadow_fakes_sideband_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "sideband_transfer" / "root",
        prefix="MTW_shadow_bin_300_met_cr_split_medium_3prong_TauPt_src",
        validate_pass_selection="MTW_shadow_bin_300_met_cr_split_medium_3prong_validate_passID",
    ),
    CachedFakeTest(
        label="upper-MET internal slice",
        cache_file=VALIDATION_OUTPUT
        / "met_cut_comparison"
        / "root"
        / "validate_met_binned_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "met_cut_comparison" / "root",
        prefix="MTW_shadow_bin_300_mtw_overlap_MET_120_170_medium_3prong_TauPt_src",
        validate_pass_selection="MTW_shadow_bin_300_mtw_overlap_MET_120_170_medium_3prong_validate_passID",
    ),
    CachedFakeTest(
        label="derive MET < 170",
        cache_file=VALIDATION_OUTPUT
        / "met_cut_comparison"
        / "root"
        / "validate_met_binned_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "met_cut_comparison" / "root",
        prefix="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt170_medium_3prong_TauPt_src",
        validate_pass_selection="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt170_medium_3prong_validate_passID",
    ),
    CachedFakeTest(
        label="derive MET < 120",
        cache_file=VALIDATION_OUTPUT
        / "met_cut_comparison"
        / "root"
        / "validate_met_binned_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "met_cut_comparison" / "root",
        prefix="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt120_medium_3prong_TauPt_src",
        validate_pass_selection="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt120_medium_3prong_validate_passID",
    ),
)

BIN_SCHEMES = (
    ("nominal 8 bins", (170.0, 200.0, 250.0, 300.0, 350.0, 425.0, 500.0, 600.0, 1000.0)),
    ("merge >=300", (170.0, 200.0, 250.0, 300.0, 1000.0)),
    ("merge >=250", (170.0, 200.0, 250.0, 1000.0)),
    ("two bins", (170.0, 250.0, 1000.0)),
    ("inclusive", (170.0, 1000.0)),
)


def get_hist(root_file: Path, hist_path: str) -> ROOT.TH1:
    with ROOT.TFile(str(root_file), "READ") as file:
        hist = file.Get(hist_path)
        if not hist:
            raise KeyError(f"Missing histogram '{hist_path}' in {root_file}")
        cloned = hist.Clone()
        cloned.SetDirectory(0)
        return cloned


def dataset_hist(root_dir: Path, dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_hist(root_dir / f"{dataset}.root", f"{NOMINAL_NAME}/{selection}/{variable}")


def validation_target(root_dir: Path, selection: str) -> ROOT.TH1:
    target = dataset_hist(root_dir, "data", selection, VARIABLE)
    for dataset in MC_SAMPLES:
        target.Add(dataset_hist(root_dir, dataset, f"trueTau_{selection}", VARIABLE), -1.0)
    target.SetName(f"{selection}_{VARIABLE}_data_minus_nonfake")
    target.SetDirectory(0)
    return target


def bin_edges(hist: ROOT.TH1) -> list[float]:
    axis = hist.GetXaxis()
    return [axis.GetBinLowEdge(bin_idx) for bin_idx in range(1, hist.GetNbinsX() + 1)] + [
        axis.GetBinUpEdge(hist.GetNbinsX())
    ]


def grouped_bin_ranges(hist: ROOT.TH1, scheme_edges: tuple[float, ...]) -> list[range]:
    axis = hist.GetXaxis()
    ranges: list[range] = []
    for low, high in zip(scheme_edges[:-1], scheme_edges[1:], strict=True):
        first = axis.FindFixBin(low + 1e-6)
        last = axis.FindFixBin(high - 1e-6)
        ranges.append(range(first, last + 1))
    return ranges


def sum_bins(hist: ROOT.TH1, bins: range) -> float:
    return sum(float(hist.GetBinContent(bin_idx)) for bin_idx in bins)


def grouped_prediction(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
    fail_input: ROOT.TH1,
    scheme_edges: tuple[float, ...],
) -> tuple[float, int, int, float]:
    prediction = 0.0
    negative_groups = 0
    nonpositive_denominator_groups = 0
    max_fake_factor = 0.0
    for bins in grouped_bin_ranges(numerator, scheme_edges):
        num = sum_bins(numerator, bins)
        den = sum_bins(denominator, bins)
        fail = sum_bins(fail_input, bins)
        if num < 0:
            negative_groups += 1
        if den <= 0:
            nonpositive_denominator_groups += 1
            continue
        fake_factor = num / den
        max_fake_factor = max(max_fake_factor, abs(fake_factor))
        prediction += fake_factor * fail
    return prediction, negative_groups, nonpositive_denominator_groups, max_fake_factor


if __name__ == "__main__":
    lines = [
        "# Coarse 3-prong fake-factor validation",
        "",
        "This cache-only validation recomputes 3-prong fake predictions after merging the "
        "`TauPt` fake-factor source bins. It does not run ROOT event loops and it does not "
        "change the nominal fake model.",
        "",
        "The test answers whether the observed 3-prong pathology is mainly caused by the "
        "fine `TauPt` binning used to form the fake factor.",
        "",
        "| Test | Source binning | CR numerator | CR denominator | Validation fail input | "
        "Predicted fakes | Validation target | Prediction / target | "
        r"Negative numerator groups | Non-positive denominator groups | Max \|FF\| |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for test in TESTS:
        numerator = get_hist(test.cache_file, f"{test.prefix}_{FAKES_SOURCE}_FF_numerator")
        denominator = get_hist(test.cache_file, f"{test.prefix}_{FAKES_SOURCE}_FF_denominator")
        fail_input = get_hist(test.cache_file, f"{test.prefix}_{FAKES_SOURCE}_FF_fakes_data_est")
        target = validation_target(test.dataset_root_dir, test.validate_pass_selection)

        source_edges = bin_edges(numerator)
        target_integral = hist_integral(target)

        for scheme_name, scheme_edges in BIN_SCHEMES:
            if scheme_edges[0] < source_edges[0] or scheme_edges[-1] > source_edges[-1]:
                raise ValueError(f"{scheme_name} edges are outside cached histogram binning")
            prediction, negative_groups, nonpositive_groups, max_fake_factor = grouped_prediction(
                numerator,
                denominator,
                fail_input,
                scheme_edges,
            )
            lines.append(
                f"| {test.label} | {scheme_name} | {hist_integral(numerator):.3f} | "
                f"{hist_integral(denominator):.3f} | {hist_integral(fail_input):.3f} | "
                f"{prediction:.3f} | {target_integral:.3f} | "
                f"{ratio(prediction, target_integral):.3f} | {negative_groups} | "
                f"{nonpositive_groups} | {max_fake_factor:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "If coarser binning moves the prediction/target ratio materially closer to one "
            "and removes negative numerator groups, then 3-prong binning is a plausible "
            "analysis issue. If the ratios barely move, the problem is not primarily the "
            "fine `TauPt` source binning.",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
