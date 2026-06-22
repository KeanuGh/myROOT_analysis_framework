from __future__ import annotations

from common import (
    FAKES_SOURCE,
    VALIDATION_OUTPUT,
    WP,
    fake_factor_bin_health,
    hist_integral,
    ratio,
    sideband_transfer_hist,
    write_markdown,
)

CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_200", "MTW_shadow_bin_300")
TRANSFER_TESTS = ("met_cr_split", "sr_met_proxy", "mtw_cr_split")
PRONGS = (1, 3)
SUMMARY_PATH = VALIDATION_OUTPUT / "fake_factor_health_summary.md"


if __name__ == "__main__":
    lines = [
        "# Fake-factor health summary",
        "",
        "This cheap diagnostic reports source-variable fake-factor health from cached "
        "sideband-transfer histograms.",
        "",
        "| Configuration | Transfer test | Prong | CR numerator | CR denominator | "
        "Negative numerator bins | Tiny/non-positive denominator bins | Validation fail input | "
        "Average fake factor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        for transfer_test in TRANSFER_TESTS:
            for prong in PRONGS:
                prefix = f"{config}_{transfer_test}_{WP}_{prong}prong_{FAKES_SOURCE}_src"
                # This script is intentionally read-only; regenerate upstream caches if missing.
                try:
                    numerator = sideband_transfer_hist(f"{prefix}_{FAKES_SOURCE}_FF_numerator")
                    denominator = sideband_transfer_hist(f"{prefix}_{FAKES_SOURCE}_FF_denominator")
                    fail_input = sideband_transfer_hist(
                        f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est"
                    )
                except KeyError:
                    continue

                neg_num, tiny_den = fake_factor_bin_health(numerator, denominator)
                lines.append(
                    f"| {config} | {transfer_test} | {prong} | "
                    f"{hist_integral(numerator):.3f} | {hist_integral(denominator):.3f} | "
                    f"{neg_num} | {tiny_den} | {hist_integral(fail_input):.3f} | "
                    f"{ratio(hist_integral(numerator), hist_integral(denominator)):.4f} |"
                )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
