from __future__ import annotations

from pathlib import Path

import ROOT
from common import (
    MC_SAMPLES,
    MEASURED_HIST_CACHE,
    VALIDATION_OUTPUT,
    dataset_hist_path,
    get_root_hist,
    hist_integral,
    ratio,
    write_markdown,
)

VARIABLE = "MTW"
FAKES_SOURCE = "TauPt"
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_250")
PRONGS = (1, 3)
SOURCE_ROOT_DIR = (
    VALIDATION_OUTPUT.parents[1] / "outputs" / "analysis_shadow_unfold" / "measured" / "root"
)
OUTPUT_DIR = VALIDATION_OUTPUT / "failid_fake_application_breakdown"
SUMMARY_PATH = OUTPUT_DIR / "failid_fake_application_breakdown_summary.md"
RUN_EVENT_LOOPS_IF_CACHE_MISSING = False


def dataset_file(dataset: str) -> Path:
    return SOURCE_ROOT_DIR / f"{dataset}.root"


def dataset_hist(dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(dataset_file(dataset), dataset_hist_path(selection, variable))


def analysis_hist(hist_name: str) -> ROOT.TH1:
    return get_root_hist(MEASURED_HIST_CACHE, hist_name)


if __name__ == "__main__":
    if not SOURCE_ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Missing measured ROOT directory: {SOURCE_ROOT_DIR}")
    if not MEASURED_HIST_CACHE.is_file():
        raise FileNotFoundError(f"Missing analysis histogram cache: {MEASURED_HIST_CACHE}")

    lines = [
        "# Fail-ID fake-application breakdown",
        "",
        "This cache-only validation diagnoses how small `TauPt` fake factors become a "
        "large final fake prediction after being applied to the SR fail-ID input.",
        "",
        f"- source ROOT directory: `{SOURCE_ROOT_DIR.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- analysis fake cache: `{MEASURED_HIST_CACHE.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- event loops enabled for missing truth-category histograms: `{RUN_EVENT_LOOPS_IF_CACHE_MISSING}`",
        "",
        "## Source-bin contribution summary",
        "",
        "| Configuration | Prong | TauPt bin [GeV] | SR fail-ID input | Fake factor | "
        "Predicted fakes in source bin | Fraction of prong prediction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        for prong in PRONGS:
            prefix = f"{config}_medium_{prong}prong_lowMET"
            ff_hist = analysis_hist(f"{prefix}_{FAKES_SOURCE}_FF")
            fail_input = analysis_hist(f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est")
            prong_prediction = sum(
                fail_input.GetBinContent(bin_idx) * ff_hist.GetBinContent(bin_idx)
                for bin_idx in range(1, ff_hist.GetNbinsX() + 1)
            )
            for bin_idx in range(1, ff_hist.GetNbinsX() + 1):
                low_edge = ff_hist.GetXaxis().GetBinLowEdge(bin_idx)
                high_edge = ff_hist.GetXaxis().GetBinUpEdge(bin_idx)
                ff_value = ff_hist.GetBinContent(bin_idx)
                fail_value = fail_input.GetBinContent(bin_idx)
                predicted = ff_value * fail_value
                lines.append(
                    f"| {config} | {prong} | {low_edge:.0f}-{high_edge:.0f} | "
                    f"{fail_value:.3f} | {ff_value:.5f} | {predicted:.3f} | "
                    f"{ratio(predicted, prong_prediction):.3f} |"
                )

    lines.extend(
        [
            "",
            "## SR fail-ID MC breakdown by sample",
            "",
            "This table uses the saved SR fail-ID `TauPt` histograms. `Fake-like MC` is "
            "`all selected MC - trueTau/nonfake MC` using the current framework "
            "`trueTau_...` selection.",
            "",
            "| Configuration | Prong | Sample | All SR fail-ID MC | TrueTau/nonfake MC | "
            "Fake-like MC | Fake-like fraction |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for config in CONFIGS:
        for prong in PRONGS:
            selection = f"{config}_medium_{prong}prong_SR_failID"
            for sample in MC_SAMPLES:
                all_hist = dataset_hist(sample, selection, FAKES_SOURCE)
                nonfake_hist = dataset_hist(sample, f"trueTau_{selection}", FAKES_SOURCE)
                all_yield = hist_integral(all_hist)
                nonfake_yield = hist_integral(nonfake_hist)
                fake_like_yield = all_yield - nonfake_yield
                lines.append(
                    f"| {config} | {prong} | `{sample}` | {all_yield:.3f} | "
                    f"{nonfake_yield:.3f} | {fake_like_yield:.3f} | "
                    f"{ratio(fake_like_yield, all_yield):.3f} |"
                )

    lines.extend(
        [
            "",
            "## Truth-category availability",
            "",
            "The current saved `analysis_shadow_unfold` cache contains the framework "
            "`trueTau_...` nonfake split, but not separate hadronic-tau, leptonic-tau, "
            "electron, muon, photon, and unmatched category histograms. This script "
            "therefore stays cache-only and reports the available sample-level "
            "fake-like/nonfake split. A category-level extension should be added only "
            "if `RUN_EVENT_LOOPS_IF_CACHE_MISSING` is deliberately enabled in a later "
            "version.",
            "",
            "## Interpretation guide",
            "",
            "Large fake yields with small fake factors imply that the SR fail-ID input is "
            "large. The source-bin table identifies whether the predicted fakes are "
            "dominated by a small number of `TauPt` bins. The sample table identifies "
            "whether the fail-ID application region is dominated by MC components that "
            "should already be replaced by the data-driven fake estimate.",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
