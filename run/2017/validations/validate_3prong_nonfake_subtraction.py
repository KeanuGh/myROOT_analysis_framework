from __future__ import annotations

from ctypes import c_double
from dataclasses import dataclass

import ROOT
from common import (  # noqa: E402
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    dataset_hist_path,
    get_root_hist,
    hist_integral,
    ratio,
    write_markdown,
)

CONFIG_LABEL = "MTW_shadow_bin_300"
VARIABLE = "MTW"
PRONGS = (1, 3)
SOURCE_ROOT_DIR = VALIDATION_OUTPUT / "low_met_fake_region" / "root"
OUTPUT_DIR = VALIDATION_OUTPUT / "nonfake_subtraction"
SUMMARY_PATH = OUTPUT_DIR / "nonfake_subtraction_summary.md"


@dataclass(frozen=True)
class Region:
    key: str
    label: str


REGIONS = (
    Region("nominal_mtw_control_metlt170", "MTW >= 350, MET < 170"),
    Region("signal_like_metgt170", "MTW >= 350, MET >= 170"),
)


def dataset_file(dataset: str):
    return SOURCE_ROOT_DIR / f"{dataset}.root"


def selection_name(region: Region, prong: int) -> str:
    return f"{CONFIG_LABEL}_{region.key}_medium_{prong}prong_validate_passID"


def read_hist(dataset: str, selection: str) -> ROOT.TH1:
    return get_root_hist(dataset_file(dataset), dataset_hist_path(selection, VARIABLE))


def hist_entries(hist: ROOT.TH1) -> float:
    return float(hist.GetEntries())


def hist_error(hist: ROOT.TH1) -> float:
    error = c_double(0.0)
    hist.IntegralAndError(1, hist.GetNbinsX(), error)
    return float(error.value)


def add_hist(first: ROOT.TH1, second: ROOT.TH1) -> ROOT.TH1:
    total = first.Clone()
    total.SetDirectory(0)
    total.Add(second)
    return total


if __name__ == "__main__":
    if not SOURCE_ROOT_DIR.is_dir():
        raise FileNotFoundError(
            "Missing low-MET fake-region cache. Run "
            "`pixi run python run/2017/validations/validate_low_met_fake_region.py` first."
        )

    rows: dict[tuple[str, int], dict[str, float]] = {}
    component_yields: dict[tuple[str, int, str], float] = {}
    component_entries: dict[tuple[str, int, str], float] = {}
    component_errors: dict[tuple[str, int, str], float] = {}
    data_hists: dict[tuple[str, int], ROOT.TH1] = {}
    nonfake_hists: dict[tuple[str, int], ROOT.TH1] = {}

    for region in REGIONS:
        for prong in PRONGS:
            selection = selection_name(region, prong)
            data_hist = read_hist("data", selection)
            data_hists[(region.key, prong)] = data_hist

            nonfake_hist: ROOT.TH1 | None = None
            for sample in MC_SAMPLES:
                hist = read_hist(sample, f"trueTau_{selection}")
                component_yields[(region.key, prong, sample)] = hist_integral(hist)
                component_entries[(region.key, prong, sample)] = hist_entries(hist)
                component_errors[(region.key, prong, sample)] = hist_error(hist)
                nonfake_hist = hist if nonfake_hist is None else add_hist(nonfake_hist, hist)

            if nonfake_hist is None:
                raise ValueError("No nonfake histograms were loaded.")
            nonfake_hists[(region.key, prong)] = nonfake_hist

            data_yield = hist_integral(data_hist)
            nonfake_yield = hist_integral(nonfake_hist)
            rows[(region.key, prong)] = {
                "data": data_yield,
                "data_entries": hist_entries(data_hist),
                "nonfake": nonfake_yield,
                "nonfake_error": hist_error(nonfake_hist),
                "residual": data_yield - nonfake_yield,
                "nonfake_over_data": ratio(nonfake_yield, data_yield),
            }

    lines = [
        "# 3-prong nonfake-subtraction validation",
        "",
        "This cache-only validation diagnoses why the high-MET 3-prong pass-ID target "
        "`data - nonfake MC` is negative. It reuses the low-MET fake-region validation "
        "histograms and does not run any ROOT dataframe event loops.",
        "",
        f"- source ROOT directory: `{SOURCE_ROOT_DIR.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- target variable: `{VARIABLE}`",
        f"- configuration: `{CONFIG_LABEL}`",
        "",
        "## Data minus nonfake summary",
        "",
        "| Region | Prong | Data yield | Data entries | Nonfake MC | Nonfake stat. err. | "
        "Data - nonfake | Nonfake / data |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        for prong in PRONGS:
            row = rows[(region.key, prong)]
            lines.append(
                f"| {region.label} | {prong} | {row['data']:.3f} | "
                f"{row['data_entries']:.0f} | {row['nonfake']:.3f} | "
                f"{row['nonfake_error']:.3f} | {row['residual']:.3f} | "
                f"{row['nonfake_over_data']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Prong ratios",
            "",
            "| Region | Quantity | 1-prong | 3-prong | 3-prong / 1-prong |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        for key, label in (
            ("data", "data"),
            ("nonfake", "total nonfake MC"),
            ("residual", "data - nonfake"),
        ):
            one_prong = rows[(region.key, 1)][key]
            three_prong = rows[(region.key, 3)][key]
            lines.append(
                f"| {region.label} | {label} | {one_prong:.3f} | {three_prong:.3f} | "
                f"{ratio(three_prong, one_prong):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Nonfake MC component breakdown",
            "",
            "| Region | Prong | Component | Yield | Stat. err. | Entries | Fraction of nonfake | Fraction of data |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        for prong in PRONGS:
            data_yield = rows[(region.key, prong)]["data"]
            nonfake_yield = rows[(region.key, prong)]["nonfake"]
            for sample in MC_SAMPLES:
                value = component_yields[(region.key, prong, sample)]
                lines.append(
                    f"| {region.label} | {prong} | `{sample}` | {value:.3f} | "
                    f"{component_errors[(region.key, prong, sample)]:.3f} | "
                    f"{component_entries[(region.key, prong, sample)]:.0f} | "
                    f"{ratio(value, nonfake_yield):.3f} | {ratio(value, data_yield):.3f} |"
                )

    problem_region = REGIONS[1]
    problem_prong = 3
    data_hist = data_hists[(problem_region.key, problem_prong)]
    nonfake_hist = nonfake_hists[(problem_region.key, problem_prong)]
    lines.extend(
        [
            "",
            "## Problematic high-MET 3-prong bins",
            "",
            "This table shows the bin-by-bin pass-ID residual for the region where the "
            "integrated target is negative.",
            "",
            "| MTW bin [GeV] | Data | Nonfake MC | Data - nonfake | Nonfake / data |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bin_idx in range(1, data_hist.GetNbinsX() + 1):
        data_value = data_hist.GetBinContent(bin_idx)
        nonfake_value = nonfake_hist.GetBinContent(bin_idx)
        residual = data_value - nonfake_value
        if data_value == 0 and nonfake_value == 0:
            continue
        low_edge = data_hist.GetXaxis().GetBinLowEdge(bin_idx)
        high_edge = data_hist.GetXaxis().GetBinUpEdge(bin_idx)
        lines.append(
            f"| {low_edge:.0f}-{high_edge:.0f} | {data_value:.3f} | "
            f"{nonfake_value:.3f} | {residual:.3f} | {ratio(nonfake_value, data_value):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The negative high-MET 3-prong target is not produced by the fake estimate. "
            "It is already present in the pass-ID validation target because the summed "
            "truth-matched nonfake MC is larger than the data yield. This makes the "
            "high-MET 3-prong proxy unsuitable as a standalone fake-factor closure test.",
            "",
            "The component table identifies which nonfake samples dominate the subtraction. "
            "If `wtaunu_had` remains the dominant term, the next physics question is the "
            "reconstructed 3-prong modelling of the signal-like nonfake component rather "
            "than another fake-factor binning tweak.",
        ]
    )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
