from __future__ import annotations

from dataclasses import dataclass

import ROOT
from common import (  # noqa: E402
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    dataset_hist_path,
    get_root_hist,
    hist_integral,
    ratio,
    sum_hists,
    write_markdown,
)

CONFIG_LABEL = "MTW_shadow_bin_300"
VARIABLE = "MTW"
FAKES_SOURCE = "TauPt"
PRONGS = (1, 3)
SOURCE_ROOT_DIR = VALIDATION_OUTPUT / "low_met_fake_region" / "root"
FAKE_CACHE = SOURCE_ROOT_DIR / "validate_low_met_fake_region.root"
OUTPUT_DIR = VALIDATION_OUTPUT / "prong_balance_scale"
SUMMARY_PATH = OUTPUT_DIR / "prong_balance_scale_summary.md"


@dataclass(frozen=True)
class Region:
    key: str
    label: str


@dataclass(frozen=True)
class FakeMethod:
    key: str
    label: str


REGIONS = (
    Region("nominal_mtw_control_metlt170", "MTW >= 350, MET < 170"),
    Region("signal_like_metgt170", "MTW >= 350, MET >= 170"),
)
FAKE_METHODS = (
    FakeMethod("current_mtw_shadow_metlt170", "current MTW-shadow CR"),
    FakeMethod("low_met_fake_enriched", "low-MET fake-enriched CR"),
)


def dataset_file(dataset: str):
    return SOURCE_ROOT_DIR / f"{dataset}.root"


def selection_name(region: Region, prong: int) -> str:
    return f"{CONFIG_LABEL}_{region.key}_medium_{prong}prong_validate_passID"


def read_dataset_hist(dataset: str, selection: str) -> ROOT.TH1:
    return get_root_hist(dataset_file(dataset), dataset_hist_path(selection, VARIABLE))


def fake_hist_name(method: FakeMethod, region: Region, prong: int) -> str:
    return (
        f"{CONFIG_LABEL}_{method.key}_medium_{prong}prong_{region.key}_"
        f"{FAKES_SOURCE}_src_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"
    )


if __name__ == "__main__":
    if not SOURCE_ROOT_DIR.is_dir() or not FAKE_CACHE.is_file():
        raise FileNotFoundError(
            "Missing low-MET fake-region cache. Run "
            "`pixi run python run/2017/validations/validate_low_met_fake_region.py` first."
        )

    rows: dict[tuple[str, str, int], dict[str, float]] = {}

    for region in REGIONS:
        for prong in PRONGS:
            selection = selection_name(region, prong)
            data = hist_integral(read_dataset_hist("data", selection))
            wtaunu_had = hist_integral(read_dataset_hist("wtaunu_had", f"trueTau_{selection}"))
            other_nonfake = hist_integral(
                sum_hists(
                    [
                        read_dataset_hist(sample, f"trueTau_{selection}")
                        for sample in MC_SAMPLES
                        if sample != "wtaunu_had"
                    ]
                )
            )

            for method in FAKE_METHODS:
                fake_prediction = hist_integral(get_root_hist(FAKE_CACHE, fake_hist_name(method, region, prong)))
                implied_wtaunu_had = data - fake_prediction - other_nonfake
                rows[(region.key, method.key, prong)] = {
                    "data": data,
                    "fake_prediction": fake_prediction,
                    "other_nonfake": other_nonfake,
                    "wtaunu_had": wtaunu_had,
                    "implied_wtaunu_had": implied_wtaunu_had,
                    "scale_factor": ratio(implied_wtaunu_had, wtaunu_had),
                }

    lines = [
        "# Implied wtaunu_had prong-balance scale factors",
        "",
        "This cache-only validation asks what prong-dependent `wtaunu_had` scale factor "
        "would be implied after subtracting a fake estimate and all other nonfake MC.",
        "",
        "The calculation is:",
        "",
        "```text",
        "target wtaunu_had = data - fake prediction - other nonfake MC",
        "SF_prong = target wtaunu_had / wtaunu_had MC",
        "```",
        "",
        f"- source ROOT directory: `{SOURCE_ROOT_DIR.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- fake cache: `{FAKE_CACHE.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- target variable: `{VARIABLE}`",
        f"- configuration: `{CONFIG_LABEL}`",
        "",
        "## Implied scale factors",
        "",
        "| Region | Fake model | Prong | Data | Fake prediction | Other nonfake MC | "
        "wtaunu_had MC | Data - fakes - other nonfake | Implied wtaunu_had SF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        for method in FAKE_METHODS:
            for prong in PRONGS:
                row = rows[(region.key, method.key, prong)]
                lines.append(
                    f"| {region.label} | {method.label} | {prong} | "
                    f"{row['data']:.3f} | {row['fake_prediction']:.3f} | "
                    f"{row['other_nonfake']:.3f} | {row['wtaunu_had']:.3f} | "
                    f"{row['implied_wtaunu_had']:.3f} | {row['scale_factor']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Prong-scale comparison",
            "",
            "| Region | Fake model | SF 1-prong | SF 3-prong | SF 3-prong / SF 1-prong |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        for method in FAKE_METHODS:
            sf_1p = rows[(region.key, method.key, 1)]["scale_factor"]
            sf_3p = rows[(region.key, method.key, 3)]["scale_factor"]
            lines.append(
                f"| {region.label} | {method.label} | {sf_1p:.3f} | {sf_3p:.3f} | "
                f"{ratio(sf_3p, sf_1p):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A stable prong-dependent nonfake correction would require the implied 3-prong "
            "scale factor to be consistently below the 1-prong scale factor across independent "
            "regions and fake models. If the implied scale factors change strongly between the "
            "control-MET and high-MET regions, then a single central correction is not defensible.",
        ]
    )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
