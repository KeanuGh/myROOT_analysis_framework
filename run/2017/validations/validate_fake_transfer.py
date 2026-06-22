from __future__ import annotations

import ROOT
from common import (
    FAKES_SOURCE,
    MC_SAMPLES,
    SIDEBAND_HIST_CACHE,
    VALIDATION_OUTPUT,
    WP,
    fake_factor_bin_health,
    hist_integral,
    ratio,
    sideband_dataset_hist,
    sideband_transfer_hist,
    sum_hists,
    write_markdown,
)

CONFIG_LABEL = "MTW_shadow_bin_300"
TRANSFER_TEST = "met_cr_split"
PRONGS = (1, 3)
VARIABLE = "MTW"
SUMMARY_PATH = VALIDATION_OUTPUT / "fake_transfer_smoke_summary.md"


def validation_target(selection: str) -> ROOT.TH1:
    data = sideband_dataset_hist("data", selection, VARIABLE)
    nonfake = sum_hists(
        [
            sideband_dataset_hist(mc_sample, f"trueTau_{selection}", VARIABLE)
            for mc_sample in MC_SAMPLES
        ]
    )
    target = data.Clone()
    target.SetDirectory(0)
    target.Add(nonfake, -1.0)
    return target


if __name__ == "__main__":
    if not SIDEBAND_HIST_CACHE.is_file():
        raise FileNotFoundError(
            "Missing sideband transfer cache. Run the sideband transfer validation once "
            f"before using this cache-only smoke test: {SIDEBAND_HIST_CACHE}"
        )

    rows: list[tuple[str, int, float, float, int, int, float, float, float, float]] = []
    feasibility_rows: list[tuple[str, int, str, str, str, str]] = []

    for prong in PRONGS:
        prefix = f"{CONFIG_LABEL}_{TRANSFER_TEST}_{WP}_{prong}prong"
        estimate_name = f"{prefix}_{FAKES_SOURCE}_src"
        validate_pass = f"{prefix}_validate_passID"

        numerator = sideband_transfer_hist(f"{estimate_name}_{FAKES_SOURCE}_FF_numerator")
        denominator = sideband_transfer_hist(f"{estimate_name}_{FAKES_SOURCE}_FF_denominator")
        validation_fail_input = sideband_transfer_hist(
            f"{estimate_name}_{FAKES_SOURCE}_FF_fakes_data_est"
        )
        prediction = sideband_transfer_hist(f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src")
        target = validation_target(validate_pass)
        negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
            numerator, denominator
        )

        rows.append(
            (
                CONFIG_LABEL,
                prong,
                hist_integral(numerator),
                hist_integral(denominator),
                negative_numerator_bins,
                tiny_denominator_bins,
                hist_integral(validation_fail_input),
                hist_integral(prediction),
                hist_integral(target),
                ratio(hist_integral(prediction), hist_integral(target)),
            )
        )
        feasibility_rows.append(
            (
                CONFIG_LABEL,
                prong,
                "TauPt x MET_met",
                "MET < 120",
                "120 <= MET < 170",
                "not directly derivable from this independent split",
            )
        )

    lines = [
        "# Fake-transfer smoke validation",
        "",
        "This is the fast cache-only fake-transfer smoke test.",
        "It reads the existing sideband validation ROOT outputs and does not run unfolding.",
        "",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- transfer test: `{TRANSFER_TEST}`",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- sideband cache: `{SIDEBAND_HIST_CACHE.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        "",
        "## 1D TauPt transfer result",
        "",
        "| Configuration | Prong | CR numerator | CR denominator | Negative numerator bins | "
        "Tiny/non-positive denominator bins | Validation fail-ID input | Predicted fakes | "
        "Validation target | Prediction / target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (
        config_label,
        prong,
        numerator,
        denominator,
        negative_numerator_bins,
        tiny_denominator_bins,
        validation_fail_input,
        prediction,
        target,
        prediction_over_target,
    ) in rows:
        lines.append(
            f"| {config_label} | {prong} | {numerator:.3f} | {denominator:.3f} | "
            f"{negative_numerator_bins} | {tiny_denominator_bins} | "
            f"{validation_fail_input:.3f} | {prediction:.3f} | {target:.3f} | "
            f"{prediction_over_target:.3f} |"
        )

    lines.extend(
        [
            "",
            "## TauPt x MET_met feasibility",
            "",
            "A literal two-dimensional `(TauPt, MET_met)` fake factor cannot be independently "
            "derived in `MET < 120` and then applied in `120 <= MET < 170` using the same "
            "MET bins, because the derivation and validation regions do not overlap in MET. "
            "This smoke test therefore checks whether MET transfer is the problematic axis; "
            "it does not claim to validate a nominal 2D fake factor.",
            "",
            "| Configuration | Prong | Candidate source model | Derivation MET region | "
            "Validation MET region | Status |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for config_label, prong, source_model, derivation_met, validation_met, status in feasibility_rows:
        lines.append(
            f"| {config_label} | {prong} | {source_model} | {derivation_met} | "
            f"{validation_met} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "If the 1D `TauPt` transfer ratio is far from unity here, the next real test should "
            "be a MET-dependence model with overlapping MET bins in the derivation region, or a "
            "MET-threshold systematic following the ATLAS tau+MET fake-estimation logic.",
        ]
    )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
