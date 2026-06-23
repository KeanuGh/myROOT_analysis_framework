from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    WP,
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import sum_th1s  # noqa: E402

YEAR = 2017
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_WIDTH_COMPARISON = True
OUTPUT_DIR = VALIDATION_OUTPUT / "tau_width_composition"
SUMMARY_PATH = OUTPUT_DIR / "tau_width_composition_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_tau_width_composition.root"

WIDTH_VARIABLES = (
    "TauTrackWidthPt1000PV",
    "TauTrackWidthPt500PV",
    "TauTrackWidthPt1000TV",
    "TauTrackWidthPt500TV",
)
WIDTH_BINS = np.linspace(0.0, 0.35, 36, dtype="double")


@dataclass(frozen=True)
class Region:
    key: str
    label: str
    cuts: tuple[Cut, ...]


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
    r"$|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
PASS_TRUETAU = Cut(
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || "
    "MatchedTruthParticle_isMuon == true || "
    "MatchedTruthParticle_isElectron == true",
)


def fake_like_hist(analysis: Analysis, variable: str, selection: str) -> ROOT.TH1:
    data = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    fake_like = data - nonfake
    fake_like.SetName(f"{selection}_{variable}_data_minus_nonfake")
    fake_like.SetDirectory(0)
    return fake_like


def clone_normalised(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    integral = hist_integral(clone)
    if integral > 0:
        clone.Scale(1.0 / integral)
    return clone


def mean_width(hist: ROOT.TH1) -> float:
    total = hist_integral(hist)
    if total == 0:
        return float("nan")
    weighted_sum = sum(
        hist.GetBinContent(bin_idx) * hist.GetBinCenter(bin_idx)
        for bin_idx in range(1, hist.GetNbinsX() + 1)
    )
    return weighted_sum / total


def l1_shape_distance(first: ROOT.TH1, second: ROOT.TH1) -> float:
    first_norm = clone_normalised(first, f"{first.GetName()}_shape")
    second_norm = clone_normalised(second, f"{second.GetName()}_shape")
    return 0.5 * sum(
        abs(first_norm.GetBinContent(bin_idx) - second_norm.GetBinContent(bin_idx))
        for bin_idx in range(1, first_norm.GetNbinsX() + 1)
    )


def negative_bins(hist: ROOT.TH1) -> int:
    return sum(
        1
        for bin_idx in range(1, hist.GetNbinsX() + 1)
        if hist.GetBinContent(bin_idx) < 0
    )


if __name__ == "__main__":
    regions = (
        Region(
            "low_met_derivation_failID",
            "low-MET fake-factor denominator, MET < 100",
            (Cut("MET < 100", "MET_met < 100"),),
        ),
        Region(
            "signal_failID",
            "nominal high-MET anti-ID application, MTW >= 350 and MET >= 170",
            (
                Cut(r"$m_T^W >= 350$", "MTW >= 350"),
                Cut("MET >= 170", "MET_met >= 170"),
            ),
        ),
        Region(
            "high_met_imbalanced_failID",
            "ATLAS-like high-MET anti-ID validation, MTW >= 350, MET >= 170, TauPt/MET < 0.7",
            (
                Cut(r"$m_T^W >= 350$", "MTW >= 350"),
                Cut("MET >= 170", "MET_met >= 170"),
                Cut(r"$p_T^\tau / E_T^\mathrm{miss} < 0.7$", "TauPt / MET_met < 0.7"),
            ),
        ),
    )

    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        FAIL_MEDIUM,
    ]
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        for region in regions:
            selection = f"{region.key}_{WP}_{prong}prong"
            data_selections[selection] = base_cuts + list(region.cuts) + [prong_cut]
            mc_selections[selection] = data_selections[selection]
            mc_selections[f"trueTau_{selection}"] = data_selections[selection] + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_exists
    )

    binnings = {"": {variable: WIDTH_BINS for variable in WIDTH_VARIABLES}}

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_tau_width_composition",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            *WIDTH_VARIABLES,
            "MTW",
            "MET_met",
            "TauPt",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=set(WIDTH_VARIABLES),
        binnings=binnings,
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing tau-width composition cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    fake_like: dict[tuple[str, int, str], ROOT.TH1] = {}
    for variable in WIDTH_VARIABLES:
        for prong in PRONGS:
            for region in regions:
                selection = f"{region.key}_{WP}_{prong}prong"
                fake_like[(variable, prong, region.key)] = fake_like_hist(
                    analysis, variable, selection
                )

    lines = [
        "# Tau-width fake-composition validation",
        "",
        "This validation tests whether tau track-width variables can act as a proxy for "
        "fake-source composition differences between the low-MET fake-factor denominator "
        "and the high-MET anti-ID application regions.",
        "",
        "The closest ATLAS tau+MET analysis uses tau jet seed-width reweighting to assess "
        "quark/gluon composition differences. The available branches here are track-width "
        "proxies rather than an explicitly named jet seed width.",
        "",
        f"- width variables: `{', '.join(WIDTH_VARIABLES)}`",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "## Shape-distance summary",
        "",
        "| Variable | Prong | Comparison | Low-MET fake-like yield | Target fake-like yield | "
        "Low-MET mean | Target mean | Relative mean shift | L1 shape distance | "
        "Low-MET negative bins | Target negative bins |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    comparison_pairs = (
        ("signal_failID", "nominal high-MET anti-ID"),
        ("high_met_imbalanced_failID", "ATLAS-like high-MET imbalanced anti-ID"),
    )
    low_key = "low_met_derivation_failID"

    for variable in WIDTH_VARIABLES:
        for prong in PRONGS:
            low_hist = fake_like[(variable, prong, low_key)]
            low_mean = mean_width(low_hist)
            for target_key, target_label in comparison_pairs:
                target_hist = fake_like[(variable, prong, target_key)]
                target_mean = mean_width(target_hist)
                relative_mean_shift = ratio(target_mean - low_mean, low_mean)
                lines.append(
                    f"| `{variable}` | {prong} | {target_label} | "
                    f"{hist_integral(low_hist):.3f} | {hist_integral(target_hist):.3f} | "
                    f"{low_mean:.5f} | {target_mean:.5f} | "
                    f"{relative_mean_shift:.3f} | "
                    f"{l1_shape_distance(low_hist, target_hist):.3f} | "
                    f"{negative_bins(low_hist)} | {negative_bins(target_hist)} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "A large relative mean shift or L1 shape distance means the fake-like anti-ID "
            "population has a different tau-width shape in the fake-factor determination "
            "region and the application region. That would support treating fake-source "
            "composition as a transfer uncertainty, following the ATLAS tau+MET logic.",
            "",
            "The `data - nonfake MC` subtraction can create negative bins in sparse high-MET "
            "regions. Rows with many negative bins should be treated as qualitative evidence "
            "rather than precise reweighting inputs.",
        ]
    )

    if PLOT_WIDTH_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "tau_width_composition"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])
        for variable in WIDTH_VARIABLES:
            for prong in PRONGS:
                hists = [
                    clone_normalised(
                        fake_like[(variable, prong, low_key)],
                        f"{variable}_{prong}prong_low_met_norm",
                    ),
                    clone_normalised(
                        fake_like[(variable, prong, "signal_failID")],
                        f"{variable}_{prong}prong_signal_norm",
                    ),
                    clone_normalised(
                        fake_like[(variable, prong, "high_met_imbalanced_failID")],
                        f"{variable}_{prong}prong_imbalanced_norm",
                    ),
                ]
                filename = f"{variable}_{prong}prong_tau_width_composition.png"
                analysis.plot(
                    hists,
                    label=[
                        "low-MET FF denominator",
                        "high-MET anti-ID",
                        "high-MET imbalanced anti-ID",
                    ],
                    colour=["tab:blue", "tab:orange", "tab:green"],
                    histstyle=["step", "step", "step"],
                    linestyle=["-", "--", "-."],
                    xlabel=variable,
                    ylabel="Normalised events",
                    title=f"{variable} | {prong}-prong fake-like anti-ID",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    ratio_plot=True,
                    ratio_label="Target / low-MET",
                    ratio_axlim=(0.0, 2.0),
                    label_params={"llabel": "", "loc": 1},
                    legend_params={"fontsize": 9, "loc": "upper right"},
                    filename=filename,
                )
                lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if run_event_loops:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
