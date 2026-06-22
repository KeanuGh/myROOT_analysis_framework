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
SUMMARY_PATH = VALIDATION_OUTPUT / "inclusive_prong" / "inclusive_prong_fake_summary.md"


@dataclass(frozen=True)
class CachedProngTest:
    label: str
    cache_file: Path
    dataset_root_dir: Path
    prefix_template: str
    validate_pass_template: str


TESTS = (
    CachedProngTest(
        label="independent MET split",
        cache_file=VALIDATION_OUTPUT
        / "sideband_transfer"
        / "root"
        / "validate_shadow_fakes_sideband_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "sideband_transfer" / "root",
        prefix_template="MTW_shadow_bin_300_met_cr_split_medium_{prong}prong_TauPt_src",
        validate_pass_template="MTW_shadow_bin_300_met_cr_split_medium_{prong}prong_validate_passID",
    ),
    CachedProngTest(
        label="MTW split",
        cache_file=VALIDATION_OUTPUT
        / "sideband_transfer"
        / "root"
        / "validate_shadow_fakes_sideband_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "sideband_transfer" / "root",
        prefix_template="MTW_shadow_bin_300_mtw_cr_split_medium_{prong}prong_TauPt_src",
        validate_pass_template="MTW_shadow_bin_300_mtw_cr_split_medium_{prong}prong_validate_passID",
    ),
    CachedProngTest(
        label="derive MET < 170",
        cache_file=VALIDATION_OUTPUT
        / "met_cut_comparison"
        / "root"
        / "validate_met_binned_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "met_cut_comparison" / "root",
        prefix_template="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt170_medium_{prong}prong_TauPt_src",
        validate_pass_template="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt170_medium_{prong}prong_validate_passID",
    ),
    CachedProngTest(
        label="derive MET < 120",
        cache_file=VALIDATION_OUTPUT
        / "met_cut_comparison"
        / "root"
        / "validate_met_binned_transfer.root",
        dataset_root_dir=VALIDATION_OUTPUT / "met_cut_comparison" / "root",
        prefix_template="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt120_medium_{prong}prong_TauPt_src",
        validate_pass_template="MTW_shadow_bin_300_met_cut_choice_derive_MET_lt120_medium_{prong}prong_validate_passID",
    ),
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


def add_hist(total: ROOT.TH1 | None, hist: ROOT.TH1) -> ROOT.TH1:
    if total is None:
        clone = hist.Clone()
        clone.SetDirectory(0)
        return clone
    total.Add(hist)
    return total


def summed_hists(hists: list[ROOT.TH1]) -> ROOT.TH1:
    total: ROOT.TH1 | None = None
    for hist in hists:
        total = add_hist(total, hist)
    if total is None:
        raise ValueError("Cannot sum an empty histogram list")
    return total


def validation_target(root_dir: Path, selections: tuple[str, ...]) -> ROOT.TH1:
    target: ROOT.TH1 | None = None
    for selection in selections:
        prong_target = dataset_hist(root_dir, "data", selection, VARIABLE)
        for dataset in MC_SAMPLES:
            prong_target.Add(dataset_hist(root_dir, dataset, f"trueTau_{selection}", VARIABLE), -1.0)
        target = add_hist(target, prong_target)
    if target is None:
        raise ValueError("Cannot build validation target without selections")
    target.SetName(f"{VARIABLE}_inclusive_prong_validation_target")
    target.SetDirectory(0)
    return target


def source_prediction(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
    fail_input: ROOT.TH1,
) -> tuple[float, int, int, float]:
    prediction = 0.0
    negative_numerator_bins = 0
    nonpositive_denominator_bins = 0
    max_fake_factor = 0.0
    for bin_idx in range(1, numerator.GetNbinsX() + 1):
        num = float(numerator.GetBinContent(bin_idx))
        den = float(denominator.GetBinContent(bin_idx))
        fail = float(fail_input.GetBinContent(bin_idx))
        if num < 0:
            negative_numerator_bins += 1
        if den <= 0:
            nonpositive_denominator_bins += 1
            continue
        fake_factor = num / den
        max_fake_factor = max(max_fake_factor, abs(fake_factor))
        prediction += fake_factor * fail
    return prediction, negative_numerator_bins, nonpositive_denominator_bins, max_fake_factor


def prong_prefix(test: CachedProngTest, prong: int) -> str:
    return test.prefix_template.format(prong=prong)


if __name__ == "__main__":
    lines = [
        "# Inclusive-prong fake-factor validation",
        "",
        "This cache-only validation tests whether combining 1-prong and 3-prong candidates "
        "into one fake factor transfers better than the prong-split method. It reads saved "
        "fake-factor internals and validation histograms; it does not run ROOT event loops.",
        "",
        "| Test | Model | CR numerator | CR denominator | Validation fail input | "
        "Predicted fakes | Validation target | Prediction / target | "
        r"Negative numerator bins | Non-positive denominator bins | Max \|FF\| |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for test in TESTS:
        prong_predictions: list[float] = []
        numerator_hists: list[ROOT.TH1] = []
        denominator_hists: list[ROOT.TH1] = []
        fail_input_hists: list[ROOT.TH1] = []
        validate_pass_selections = tuple(
            test.validate_pass_template.format(prong=prong) for prong in (1, 3)
        )
        target = validation_target(test.dataset_root_dir, validate_pass_selections)
        target_integral = hist_integral(target)

        for prong in (1, 3):
            prefix = prong_prefix(test, prong)
            numerator = get_hist(test.cache_file, f"{prefix}_{FAKES_SOURCE}_FF_numerator")
            denominator = get_hist(test.cache_file, f"{prefix}_{FAKES_SOURCE}_FF_denominator")
            fail_input = get_hist(test.cache_file, f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est")
            prediction, _neg_bins, _bad_den_bins, _max_ff = source_prediction(
                numerator,
                denominator,
                fail_input,
            )
            prong_predictions.append(prediction)
            numerator_hists.append(numerator)
            denominator_hists.append(denominator)
            fail_input_hists.append(fail_input)

        split_prediction = sum(prong_predictions)
        split_numerator = sum(hist_integral(hist) for hist in numerator_hists)
        split_denominator = sum(hist_integral(hist) for hist in denominator_hists)
        split_fail_input = sum(hist_integral(hist) for hist in fail_input_hists)
        lines.append(
            f"| {test.label} | prong split | {split_numerator:.3f} | "
            f"{split_denominator:.3f} | {split_fail_input:.3f} | "
            f"{split_prediction:.3f} | {target_integral:.3f} | "
            f"{ratio(split_prediction, target_integral):.3f} | - | - | - |"
        )

        inclusive_numerator = summed_hists(numerator_hists)
        inclusive_denominator = summed_hists(denominator_hists)
        inclusive_fail_input = summed_hists(fail_input_hists)
        inclusive_prediction, negative_bins, nonpositive_denominator_bins, max_fake_factor = (
            source_prediction(
                inclusive_numerator,
                inclusive_denominator,
                inclusive_fail_input,
            )
        )
        lines.append(
            f"| {test.label} | inclusive 1+3 | {hist_integral(inclusive_numerator):.3f} | "
            f"{hist_integral(inclusive_denominator):.3f} | "
            f"{hist_integral(inclusive_fail_input):.3f} | {inclusive_prediction:.3f} | "
            f"{target_integral:.3f} | {ratio(inclusive_prediction, target_integral):.3f} | "
            f"{negative_bins} | {nonpositive_denominator_bins} | {max_fake_factor:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "If the inclusive 1+3 ratio is closer to unity than the prong-split ratio across "
            "independent sideband tests, then an inclusive fake factor is worth treating as "
            "a candidate model. If it only improves one pathological row while worsening "
            "validated rows, it is likely a cancellation rather than a better physical model.",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
