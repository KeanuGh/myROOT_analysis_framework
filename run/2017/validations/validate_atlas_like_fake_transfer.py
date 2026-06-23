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

from binnings import BINNINGS  # noqa: E402
from common import (  # noqa: E402
    FAKES_SOURCE,
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    WP,
    fake_factor_bin_health,
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import sum_th1s  # noqa: E402
from utils.variable_names import variable_data  # noqa: E402

YEAR = 2017
VARIABLE = "MTW"
CONFIG_LABEL = "atlas_like_low_met"
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_TRANSFER_COMPARISON = True
OUTPUT_DIR = VALIDATION_OUTPUT / "atlas_like_fake_transfer"
SUMMARY_PATH = OUTPUT_DIR / "atlas_like_fake_transfer_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_atlas_like_fake_transfer.root"


@dataclass(frozen=True)
class MetWindow:
    key: str
    label: str
    cut: Cut


@dataclass(frozen=True)
class ValidationTarget:
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
PASS_MEDIUM = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
    r"(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
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


def validation_target(analysis: Analysis, selection: str) -> ROOT.TH1:
    data = analysis.get_hist(
        VARIABLE,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                VARIABLE,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    target = data - nonfake
    target.SetName(f"{selection}_{VARIABLE}_data_minus_nonfake")
    target.SetDirectory(0)
    return target


if __name__ == "__main__":
    low_met_windows = (
        MetWindow("MET_0_100", "0 <= MET < 100", Cut("MET < 100", "MET_met < 100")),
        MetWindow(
            "MET_30_100",
            "30 <= MET < 100",
            Cut("30 <= MET < 100", "(MET_met >= 30) && (MET_met < 100)"),
        ),
        MetWindow(
            "MET_50_100",
            "50 <= MET < 100",
            Cut("50 <= MET < 100", "(MET_met >= 50) && (MET_met < 100)"),
        ),
        MetWindow(
            "MET_70_100",
            "70 <= MET < 100",
            Cut("70 <= MET < 100", "(MET_met >= 70) && (MET_met < 100)"),
        ),
        MetWindow("MET_0_150", "0 <= MET < 150", Cut("MET < 150", "MET_met < 150")),
    )

    validation_targets = (
        ValidationTarget(
            "atlas_like_high_met_imbalanced",
            "MTW >= 240, MET >= 170, TauPt/MET < 0.7",
            (
                Cut(r"$m_T^W >= 240$", "MTW >= 240"),
                Cut("MET >= 170", "MET_met >= 170"),
                Cut(r"$p_T^\tau / E_T^\mathrm{miss} < 0.7$", "TauPt / MET_met < 0.7"),
            ),
        ),
        ValidationTarget(
            "signal_like_high_met_imbalanced",
            "MTW >= 350, MET >= 170, TauPt/MET < 0.7",
            (
                Cut(r"$m_T^W >= 350$", "MTW >= 350"),
                Cut("MET >= 170", "MET_met >= 170"),
                Cut(r"$p_T^\tau / E_T^\mathrm{miss} < 0.7$", "TauPt / MET_met < 0.7"),
            ),
        ),
    )

    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]

    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        for met_window in low_met_windows:
            derive_prefix = f"{CONFIG_LABEL}_{met_window.key}_{WP}_{prong}prong"
            data_selections[f"{derive_prefix}_derive_passID"] = (
                base_cuts + [met_window.cut, PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{derive_prefix}_derive_failID"] = (
                base_cuts + [met_window.cut, FAIL_MEDIUM, prong_cut]
            )

        for target in validation_targets:
            target_prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            data_selections[f"{target_prefix}_validate_passID"] = (
                base_cuts + list(target.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{target_prefix}_validate_failID"] = (
                base_cuts + list(target.cuts) + [FAIL_MEDIUM, prong_cut]
            )

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_exists
    )

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_atlas_like_fake_transfer",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            VARIABLE,
            FAKES_SOURCE,
            "MET_met",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={VARIABLE, FAKES_SOURCE},
        binnings={"": BINNINGS},
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing ATLAS-like fake-transfer cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    lines = [
        "# ATLAS-like fake-transfer validation",
        "",
        "This validation tests the fake-transfer ingredients suggested by the ATLAS "
        "high-mass tau+MET analysis without changing the nominal unfolding script.",
        "",
        "It derives prong-split `TauPt` fake factors in low-MET fake-enriched regions "
        "and applies them to high-MET imbalanced validation regions.",
        "",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- working point: `{WP}`",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "## Transfer summary",
        "",
        "| MET window | Validation target | Prong | CR numerator | CR denominator | "
        "Negative numerator bins | Tiny/non-positive denominator bins | "
        "Validation fail-ID input | Predicted fakes | Validation target | "
        "Prediction / target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    comparison_hists: dict[tuple[str, int], list[tuple[str, ROOT.TH1]]] = {}

    for target in validation_targets:
        for prong in PRONGS:
            target_prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            target_hist = validation_target(analysis, validate_pass)
            comparison_hists[(target.key, prong)] = [("Data - nonfake MC", target_hist)]

            for met_window in low_met_windows:
                derive_prefix = f"{CONFIG_LABEL}_{met_window.key}_{WP}_{prong}prong"
                derive_pass = f"{derive_prefix}_derive_passID"
                derive_fail = f"{derive_prefix}_derive_failID"
                estimate_name = f"{derive_prefix}_{target.key}_{FAKES_SOURCE}_src"

                if not loaded_hists:
                    analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        (VARIABLE,),
                        derive_pass,
                        derive_fail,
                        validate_pass,
                        validate_fail,
                        f"trueTau_{derive_pass}",
                        f"trueTau_{derive_fail}",
                        f"trueTau_{validate_pass}",
                        f"trueTau_{validate_fail}",
                        name=estimate_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

                numerator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_numerator"]
                denominator = analysis.histograms[
                    f"{estimate_name}_{FAKES_SOURCE}_FF_denominator"
                ]
                validation_fail_input = analysis.histograms[
                    f"{estimate_name}_{FAKES_SOURCE}_FF_fakes_data_est"
                ]
                prediction = analysis.histograms[
                    f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                    numerator, denominator
                )
                predicted_integral = hist_integral(prediction)
                target_integral = hist_integral(target_hist)

                lines.append(
                    f"| {met_window.label} | {target.label} | {prong} | "
                    f"{hist_integral(numerator):.3f} | "
                    f"{hist_integral(denominator):.3f} | "
                    f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                    f"{hist_integral(validation_fail_input):.3f} | "
                    f"{predicted_integral:.3f} | {target_integral:.3f} | "
                    f"{ratio(predicted_integral, target_integral):.3f} |"
                )
                comparison_hists[(target.key, prong)].append((met_window.label, prediction))

    lines.extend(
        [
            "",
            "## MET-window envelope guide",
            "",
            "The ATLAS tau+MET note treats variations of the low-MET fake-factor "
            "determination window as a fake-transfer systematic. In this validation, "
            "the spread across the MET-window rows is the quantity to inspect. A stable "
            "candidate model should keep `Prediction / target` near unity without creating "
            "negative numerator bins or tiny denominator bins.",
            "",
            "The high-MET imbalanced target with `MTW >= 240` is closest to the ATLAS-style "
            "validation region. The `MTW >= 350` target is more signal-like for this analysis "
            "and is expected to have less statistical power.",
            "",
            "## Source-composition note",
            "",
            "The closest ATLAS tau+MET analysis uses tau jet seed-width reweighting to "
            "assess quark/gluon composition differences between the fake-factor "
            "determination region and the anti-ID application region. This script does not "
            "perform that check because the framework does not currently expose an obvious "
            "tau seed-width variable. If such a branch exists in the ntuples, it should be "
            "added as a separate read-only branch audit before changing the nominal fake model.",
        ]
    )

    if PLOT_TRANSFER_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "atlas_like_fake_transfer"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])

        for (target_key, prong), labelled_hists in comparison_hists.items():
            filename = f"{CONFIG_LABEL}_{target_key}_{prong}prong_atlas_like_fake_transfer.png"
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
                histstyle=["step"] * len(labelled_hists),
                linestyle=["-", "-", "--", "-.", ":", (0, (3, 1, 1, 1))],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{target_key} | {prong}-prong",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                logx=True,
                ratio_plot=True,
                ratio_label="Prediction / target",
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
