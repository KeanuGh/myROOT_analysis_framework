from __future__ import annotations

import csv
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

from binnings import BINNINGS  # noqa: E402
from common import (  # noqa: E402
    FAKES_SOURCE,
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    WP,
    dataset_hist_path,
    get_root_hist,
    hist_integral,
    ratio,
    sum_hists,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, mc_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import sum_th1s  # noqa: E402

YEAR = 2017
CONFIGS = {
    "no_shadow_bin": 350,
    "MTW_shadow_bin_250": 250,
}
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
OUTPUT_DIR = VALIDATION_OUTPUT / "failid_truth_categories"
SUMMARY_PATH = OUTPUT_DIR / "failid_truth_categories_summary.md"
CATEGORY_TOTAL_CSV = OUTPUT_DIR / "failid_truth_category_totals.csv"
CATEGORY_BIN_CSV = OUTPUT_DIR / "failid_truth_category_source_bins.csv"
MEASURED_ROOT_DIR = REPO_ROOT / "outputs" / "analysis_shadow_unfold" / "measured" / "root"


@dataclass(frozen=True)
class TruthCategory:
    key: str
    label: str
    cut: Cut


PASS_RECO_PRESELECTION = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) "
    r"&& passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + "
    r"MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
FAIL_MEDIUM = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
    r"(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
PASS_ETA = Cut(
    r"$|\eta^{\tau_\mathrm{had-vis}} < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
TRUTH_CATEGORIES = (
    TruthCategory(
        "hadronic_tau",
        "Hadronic tau",
        Cut("matched hadronic tau", "MatchedTruthParticle_isHadronicTau == true"),
    ),
    TruthCategory(
        "leptonic_tau",
        "Leptonic tau",
        Cut("matched leptonic tau", "MatchedTruthParticle_isLeptonicTau == true"),
    ),
    TruthCategory(
        "electron",
        "Electron",
        Cut("matched electron", "MatchedTruthParticle_isElectron == true"),
    ),
    TruthCategory(
        "muon",
        "Muon",
        Cut("matched muon", "MatchedTruthParticle_isMuon == true"),
    ),
    TruthCategory(
        "photon",
        "Photon",
        Cut("matched photon", "MatchedTruthParticle_isPhoton == true"),
    ),
    TruthCategory(
        "jet_or_unmatched",
        "Jet-like or unmatched",
        Cut(
            "jet-like or unmatched",
            "MatchedTruthParticle_isTau == false && "
            "MatchedTruthParticle_isElectron == false && "
            "MatchedTruthParticle_isMuon == false && "
            "MatchedTruthParticle_isPhoton == false",
        ),
    ),
)


def dataset_root_files_exist() -> bool:
    root_dir = OUTPUT_DIR / "root"
    return all((root_dir / f"{sample}.root").is_file() for sample in MC_SAMPLES)


def selection_name(config: str, prong: int, suffix: str = "all") -> str:
    return f"{config}_{WP}_{prong}prong_SR_failID_{suffix}"


def hist_bin_rows(hist: ROOT.TH1) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for bin_idx in range(1, hist.GetNbinsX() + 1):
        low = hist.GetXaxis().GetBinLowEdge(bin_idx)
        high = hist.GetXaxis().GetBinUpEdge(bin_idx)
        rows.append((f"{low:.0f}-{high:.0f}", float(hist.GetBinContent(bin_idx))))
    return rows


def existing_data_integral(config: str, prong: int) -> float:
    data_file = MEASURED_ROOT_DIR / "data.root"
    if not data_file.is_file():
        return float("nan")
    selection = f"{config}_{WP}_{prong}prong_SR_failID"
    try:
        return hist_integral(get_root_hist(data_file, dataset_hist_path(selection, FAKES_SOURCE)))
    except KeyError:
        return float("nan")


def analysis_hist(analysis: Analysis, sample: str, selection: str) -> ROOT.TH1:
    return analysis.get_hist(
        FAKES_SOURCE,
        dataset=sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cache_ready = dataset_root_files_exist()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_ready
    )
    if not cache_ready and not run_event_loops:
        raise FileNotFoundError(
            "Missing fail-ID truth-category validation cache and event loops are disabled: "
            f"{OUTPUT_DIR / 'root'}"
        )

    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        Cut(r"$E_T^{\mathrm{miss}} > 170$", "MET_met > 170"),
        FAIL_MEDIUM,
    ]
    mc_selections: dict[str, list[Cut]] = {}
    for config, mtw_min in CONFIGS.items():
        for prong in PRONGS:
            cuts = [
                *base_cuts,
                Cut(r"$m_T^W$ threshold", f"MTW > {mtw_min:g}"),
                Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}"),
            ]
            mc_selections[selection_name(config, prong)] = cuts
            for category in TRUTH_CATEGORIES:
                mc_selections[selection_name(config, prong, category.key)] = [
                    *cuts,
                    category.cut,
                ]

    analysis = Analysis(
        mc_samples(mc_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_failid_truth_categories",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            FAKES_SOURCE,
            "MTW",
            "MET_met",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
            "MatchedTruthParticle_isTau",
            "MatchedTruthParticle_isHadronicTau",
            "MatchedTruthParticle_isLeptonicTau",
            "MatchedTruthParticle_isElectron",
            "MatchedTruthParticle_isMuon",
            "MatchedTruthParticle_isPhoton",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={FAKES_SOURCE},
        binnings={"": BINNINGS},
    )

    total_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    lines = [
        "# Fail-ID truth-category audit",
        "",
        "This validation decomposes the SR fail-ID `TauPt` application population by "
        "matched truth category. It is designed to explain what object type fills the "
        "large low-`TauPt` fail-ID input used by the fake-factor estimate.",
        "",
        f"- output directory: `{OUTPUT_DIR.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        f"- cached dataset ROOT files: `{(OUTPUT_DIR / 'root').relative_to(REPO_ROOT)}`",
        f"- source-bin CSV: `{CATEGORY_BIN_CSV.relative_to(REPO_ROOT)}`",
        f"- category-total CSV: `{CATEGORY_TOTAL_CSV.relative_to(REPO_ROOT)}`",
        "",
        "## Category totals summed over MC samples",
        "",
        "| Configuration | Prong | Data fail-ID | All MC fail-ID | Category | "
        "Category yield | Fraction of all MC |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]

    for config in CONFIGS:
        for prong in PRONGS:
            all_selection = selection_name(config, prong)
            all_mc_hist = sum_th1s(
                *[analysis_hist(analysis, sample, all_selection) for sample in MC_SAMPLES]
            )
            all_mc_yield = hist_integral(all_mc_hist)
            data_yield = existing_data_integral(config, prong)

            category_hists: dict[str, ROOT.TH1] = {}
            for category in TRUTH_CATEGORIES:
                category_selection = selection_name(config, prong, category.key)
                category_hist = sum_th1s(
                    *[
                        analysis_hist(analysis, sample, category_selection)
                        for sample in MC_SAMPLES
                    ]
                )
                category_hists[category.key] = category_hist
                category_yield = hist_integral(category_hist)
                total_rows.append(
                    {
                        "configuration": config,
                        "prong": prong,
                        "sample": "all_mc",
                        "category": category.key,
                        "yield": f"{category_yield:.6g}",
                        "fraction_of_all_mc": f"{ratio(category_yield, all_mc_yield):.6g}",
                    }
                )
                lines.append(
                    f"| {config} | {prong} | {data_yield:.3f} | {all_mc_yield:.3f} | "
                    f"{category.label} | {category_yield:.3f} | "
                    f"{ratio(category_yield, all_mc_yield):.3f} |"
                )

                for bin_label, value in hist_bin_rows(category_hist):
                    bin_rows.append(
                        {
                            "configuration": config,
                            "prong": prong,
                            "sample": "all_mc",
                            "category": category.key,
                            "TauPt_bin_GeV": bin_label,
                            "yield": f"{value:.6g}",
                        }
                    )

                for sample in MC_SAMPLES:
                    sample_hist = analysis_hist(analysis, sample, category_selection)
                    sample_yield = hist_integral(sample_hist)
                    total_rows.append(
                        {
                            "configuration": config,
                            "prong": prong,
                            "sample": sample,
                            "category": category.key,
                            "yield": f"{sample_yield:.6g}",
                            "fraction_of_all_mc": f"{ratio(sample_yield, all_mc_yield):.6g}",
                        }
                    )
                    for bin_label, value in hist_bin_rows(sample_hist):
                        bin_rows.append(
                            {
                                "configuration": config,
                                "prong": prong,
                                "sample": sample,
                                "category": category.key,
                                "TauPt_bin_GeV": bin_label,
                                "yield": f"{value:.6g}",
                            }
                        )

            category_sum = sum_hists(list(category_hists.values()))
            category_sum_yield = hist_integral(category_sum)
            lines.append(
                f"| {config} | {prong} | {data_yield:.3f} | {all_mc_yield:.3f} | "
                f"category sum check | {category_sum_yield:.3f} | "
                f"{ratio(category_sum_yield, all_mc_yield):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "Use the category-total table to identify the object type dominating the SR "
            "fail-ID application population. Use the source-bin CSV to inspect the "
            "`TauPt = 170-200` and `200-250` GeV bins, which dominate the current fake "
            "prediction in the fail-ID application breakdown.",
            "",
            "The `jet_or_unmatched` category is defined explicitly as no tau, electron, "
            "muon, or photon match. It does not use `MatchedTruthParticle_isJet`, because "
            "that helper ignores the photon flag in this framework.",
        ]
    )

    write_csv(CATEGORY_TOTAL_CSV, total_rows)
    write_csv(CATEGORY_BIN_CSV, bin_rows)
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
