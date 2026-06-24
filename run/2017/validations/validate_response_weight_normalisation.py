from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binnings import BINNINGS  # noqa: E402
from common import VALIDATION_OUTPUT, WP, hist_integral, ratio, write_markdown  # noqa: E402
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, signal_sample  # noqa: E402
from shadow_unfold.models import ShadowConfig  # noqa: E402
from shadow_unfold.selections import build_fiducial_truth_cuts, build_reco_sr_cuts  # noqa: E402

from src.cutting import Cut  # noqa: E402
from utils.variable_names import variable_data  # noqa: E402

OUTPUT_DIR = VALIDATION_OUTPUT / "response_weight_normalisation"
SUMMARY_PATH = OUTPUT_DIR / "response_weight_normalisation_summary.md"
CSV_PATH = OUTPUT_DIR / "response_weight_normalisation.csv"

CURRENT_RESPONSE_ROOT = (
    REPO_ROOT / "outputs" / "analysis_shadow_unfold" / "response" / "root" / "wtaunu_had.root"
)

# Keep false by default. Turning this on performs a targeted ROOT pass into
# outputs/validate_shadow_fakes/response_weight_normalisation/fresh_standard_response/.
RUN_FRESH_STANDARD_RESPONSE_BUILD = True
FRESH_RESPONSE_OUTPUT = OUTPUT_DIR / "fresh_standard_response"

CONFIGS = (
    ShadowConfig("no_shadow_bin", None, mtw_min=350.0, taupt_min=170.0, met_min=150.0),
    ShadowConfig("MTW_shadow_bin_250", "MTW", mtw_min=250.0, taupt_min=170.0, met_min=150.0),
)
VARS = ("MTW",)
HISTOGRAMS = ("MTW", "MTW_TruthMTW")
COMPATIBLE_MIN_RATIO = 0.2
COMPATIBLE_MAX_RATIO = 5.0

PASS_MEDIUM = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
    r"(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
SKIP_SYS = {
    r"^TAUS_TRUEHADTAU_SME_TES_.*",
    r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
    r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
}


def root_hist(root_file: Path, path: str) -> ROOT.TH1 | None:
    with ROOT.TFile(str(root_file), "READ") as file:
        hist = file.Get(path)
        if not hist:
            return None
        cloned = hist.Clone()
        cloned.SetDirectory(0)
        return cloned


def top_directories(root_file: Path) -> list[str]:
    if not root_file.is_file():
        return []
    with ROOT.TFile(str(root_file), "READ") as file:
        return [
            key.GetName() for key in file.GetListOfKeys() if key.GetClassName() == "TDirectoryFile"
        ]


def response_selections() -> dict[str, list[Cut]]:
    selections: dict[str, list[Cut]] = {}
    for config in CONFIGS:
        reco_cuts = build_reco_sr_cuts(config)
        truth_cuts = build_fiducial_truth_cuts(config)
        selections[f"{config.label}_{WP}_reco_tau"] = reco_cuts + [PASS_MEDIUM]
        selections[f"{config.label}_{WP}_truth_reco_tau"] = truth_cuts + reco_cuts + [PASS_MEDIUM]
    return selections


def response_binnings() -> dict[str, dict[str, np.ndarray]]:
    binnings = {"": dict(BINNINGS)}
    mtw_250_bins = np.array(
        [250.0, *[edge for edge in BINNINGS["MTW"] if edge > 250.0]],
        dtype=float,
    )
    for selection in response_selections():
        if selection.startswith("MTW_shadow_bin_250"):
            binnings[rf"^{re.escape(selection)}$"] = {
                **BINNINGS,
                "MTW": mtw_250_bins,
                "TruthMTW": mtw_250_bins,
            }
    return binnings


def build_fresh_standard_response_cache() -> Path:
    from src.analysis import Analysis
    from utils.plotting_tools import Hist2dOpts

    output_file = FRESH_RESPONSE_OUTPUT / "root" / "wtaunu_had.root"
    if output_file.is_file():
        output_file.unlink()

    label_regex = "|".join(re.escape(config.label) for config in CONFIGS)
    truth_histogram_vars = {variable_data[var]["truth"] for var in VARS}
    selections = response_selections()
    Analysis(
        {"wtaunu_had": signal_sample(selections=selections)},
        year=2017,
        rerun=True,
        regen_histograms=True,
        do_systematics=True,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_response_weight_normalisation_fresh_response",
        output_dir=FRESH_RESPONSE_OUTPUT,
        log_level=10,
        log_out="both",
        extract_vars={
            "MTW",
            "TauPt",
            "MET_met",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TruthMTW",
            "VisTruthTauPt",
            "TruthNeutrinoPt",
            "VisTruthTauEta",
            "TruthTau_nChargedTracks",
            "TruthTau_isHadronic",
            "eventNumber",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=set(VARS) | truth_histogram_vars,
        hists_2d={
            f"{var}_{variable_data[var]['truth']}": Hist2dOpts(
                var,
                variable_data[var]["truth"],
                "reco_weight",
            )
            for var in VARS
        },
        do_unweighted=True,
        systematics_for_selection={rf"^({label_regex})_{WP}_(reco_tau|truth_reco_tau)$"},
        skip_sys=SKIP_SYS,
        binnings=response_binnings(),
    )
    return output_file


def summarise_response_root(root_file: Path, source: str) -> list[dict[str, object]]:
    sys_dirs = [
        directory
        for directory in top_directories(root_file)
        if directory.startswith("TAUS_TRUEHADTAU_EFF_")
    ]
    rows: list[dict[str, object]] = []
    for selection in response_selections():
        for hist_name in HISTOGRAMS:
            nominal = root_hist(root_file, f"{NOMINAL_NAME}/{selection}/{hist_name}")
            if nominal is None:
                rows.append(
                    {
                        "source": source,
                        "selection": selection,
                        "histogram": hist_name,
                        "systematic": "missing_nominal",
                        "nominal_integral": float("nan"),
                    }
                )
                continue

            nominal_integral = hist_integral(nominal)
            unweighted = root_hist(
                root_file,
                f"{NOMINAL_NAME}/{selection}/{hist_name}_unweighted",
            )
            unweighted_integral = hist_integral(unweighted) if unweighted else float("nan")
            for sys_name in sys_dirs:
                varied = root_hist(root_file, f"{sys_name}/{selection}/{hist_name}")
                if varied is None:
                    continue
                varied_integral = hist_integral(varied)
                varied_over_nominal = ratio(varied_integral, nominal_integral)
                varied_over_unweighted = ratio(varied_integral, unweighted_integral)
                compatible = (
                    COMPATIBLE_MIN_RATIO <= abs(varied_over_nominal) <= COMPATIBLE_MAX_RATIO
                )
                rows.append(
                    {
                        "source": source,
                        "selection": selection,
                        "histogram": hist_name,
                        "systematic": sys_name,
                        "nominal_integral": nominal_integral,
                        "unweighted_integral": unweighted_integral,
                        "varied_integral": varied_integral,
                        "varied_over_nominal": varied_over_nominal,
                        "varied_over_unweighted": varied_over_unweighted,
                        "compatible": compatible,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "selection",
        "histogram",
        "systematic",
        "nominal_integral",
        "unweighted_integral",
        "varied_integral",
        "varied_over_nominal",
        "varied_over_unweighted",
        "compatible",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_lines(rows: list[dict[str, object]]) -> list[str]:
    incompatible = [
        row for row in rows if row.get("compatible") is False and row.get("histogram") == "MTW"
    ]
    worst = sorted(
        incompatible,
        key=lambda row: abs(float(row["varied_over_nominal"])),
        reverse=True,
    )[:12]

    lines = [
        "# Response systematic weight normalisation diagnostic",
        "",
        "This diagnostic checks whether tau-efficiency response variations are "
        "normalised like the nominal response histogram.",
        "",
        f"- Current response cache: `{CURRENT_RESPONSE_ROOT}`",
        f"- Full CSV: `{CSV_PATH}`",
        f"- Fresh standard-response build enabled: `{RUN_FRESH_STANDARD_RESPONSE_BUILD}`",
        "",
        "Interpretation:",
        "",
        "- If the current cache is bad but the fresh standard build is good, the "
        "problem is cache/build-path mixing.",
        "- If both are bad, the efficiency systematic weight formula is wrong for "
        "the matched response selection.",
        "- If the varied integral is close to the unweighted integral, the "
        "systematic branch is behaving like an order-one per-event factor rather "
        "than a luminosity-normalised event weight.",
        "",
        "## Largest incompatible MTW variations",
        "",
        "| source | selection | systematic | nominal | unweighted | varied | varied/nominal | varied/unweighted |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not worst:
        lines.append("| - | - | - | - | - | - | - | - |")
    for row in worst:
        lines.append(
            "| {source} | {selection} | {systematic} | {nominal:.3g} | "
            "{unweighted:.3g} | {varied:.3g} | {ratio_nom:.3g} | "
            "{ratio_unw:.3g} |".format(
                source=row["source"],
                selection=row["selection"],
                systematic=row["systematic"],
                nominal=float(row["nominal_integral"]),
                unweighted=float(row["unweighted_integral"]),
                varied=float(row["varied_integral"]),
                ratio_nom=float(row["varied_over_nominal"]),
                ratio_unw=float(row["varied_over_unweighted"]),
            )
        )
    return lines


def main() -> None:
    rows = summarise_response_root(CURRENT_RESPONSE_ROOT, "current_analysis_response")
    if RUN_FRESH_STANDARD_RESPONSE_BUILD:
        fresh_root = build_fresh_standard_response_cache()
        rows.extend(summarise_response_root(fresh_root, "fresh_standard_response"))
    write_csv(CSV_PATH, rows)
    write_markdown(SUMMARY_PATH, markdown_lines(rows))
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
