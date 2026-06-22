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
CONFIG_LABEL = "MTW_shadow_bin_300"
MTW_SHADOW_MIN = 300
MTW_NOMINAL_MIN = 350
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_TRANSFER_COMPARISON = True
OUTPUT_DIR = VALIDATION_OUTPUT / "low_met_fake_region"
SUMMARY_PATH = OUTPUT_DIR / "low_met_fake_region_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_low_met_fake_region.root"


@dataclass(frozen=True)
class FakeMethod:
    key: str
    label: str
    cuts: tuple[Cut, ...]


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
    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    derivation_methods = (
        FakeMethod(
            "current_mtw_shadow_metlt170",
            "current MTW-shadow CR, MET < 170",
            (
                Cut(
                    r"shadow $m_T^W$ sideband",
                    f"(MTW >= {MTW_SHADOW_MIN:g}) && (MTW < {MTW_NOMINAL_MIN:g})",
                ),
                Cut("MET < 170", "MET_met < 170"),
            ),
        ),
        FakeMethod(
            "low_met_fake_enriched",
            "low-MET fake-enriched CR, MET < 100",
            (Cut("MET < 100", "MET_met < 100"),),
        ),
    )
    validation_targets = (
        ValidationTarget(
            "nominal_mtw_control_metlt170",
            "MTW >= 350, MET < 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET < 170", "MET_met < 170"),
            ),
        ),
        ValidationTarget(
            "signal_like_metgt170",
            "MTW >= 350, MET >= 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET >= 170", "MET_met >= 170"),
            ),
        ),
    )

    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        for method in derivation_methods:
            prefix = f"{CONFIG_LABEL}_{method.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_derive_passID"] = (
                base_cuts + list(method.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_derive_failID"] = (
                base_cuts + list(method.cuts) + [FAIL_MEDIUM, prong_cut]
            )

        for target in validation_targets:
            prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + list(target.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + list(target.cuts) + [FAIL_MEDIUM, prong_cut]
            )

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (not LOAD_SAVED_HISTS or not cache_exists)

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_low_met_fake_region",
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
            "Missing low-MET fake-region cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    lines = [
        "# Low-MET fake-enriched fake-factor validation",
        "",
        "This validation tests an ATLAS-like low-MET fake-enriched fake-factor derivation "
        "against the current MTW-shadow control-region derivation. It is a validation-only "
        "study, not a nominal unfolding change.",
        "",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "## Transfer summary",
        "",
        "| Method | Target | Prong | CR numerator | CR denominator | "
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

            for method in derivation_methods:
                method_prefix = f"{CONFIG_LABEL}_{method.key}_{WP}_{prong}prong"
                derive_pass = f"{method_prefix}_derive_passID"
                derive_fail = f"{method_prefix}_derive_failID"
                estimate_name = f"{method_prefix}_{target.key}_{FAKES_SOURCE}_src"

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
                denominator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_denominator"]
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
                    f"| {method.label} | {target.label} | {prong} | "
                    f"{hist_integral(numerator):.3f} | {hist_integral(denominator):.3f} | "
                    f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                    f"{hist_integral(validation_fail_input):.3f} | {predicted_integral:.3f} | "
                    f"{target_integral:.3f} | {ratio(predicted_integral, target_integral):.3f} |"
                )
                comparison_hists[(target.key, prong)].append((method.label, prediction))

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "The low-MET fake-enriched method is interesting only if it improves the "
            "`Prediction / target` ratio without creating unhealthy source bins. If it "
            "does better in the `MTW >= 350, MET < 170` target but not in the signal-like "
            "`MET >= 170` proxy, the difference should be treated as a transfer uncertainty "
            "rather than a nominal model change.",
            "",
            "The current MTW-shadow control-region row is included so the low-MET method can "
            "be judged against the current fake-factor construction in the same script and "
            "with the same target definitions.",
        ]
    )

    if PLOT_TRANSFER_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "low_met_fake_region"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])
        for (target_key, prong), labelled_hists in comparison_hists.items():
            filename = f"{CONFIG_LABEL}_{target_key}_{prong}prong_low_met_fake_region.png"
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange"],
                histstyle=["step", "step", "step"],
                linestyle=["-", "-", "--"],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{CONFIG_LABEL} | {target_key} | {prong}-prong",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                logx=True,
                ratio_plot=True,
                ratio_label="Prediction / target",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=filename,
            )
            lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if run_event_loops:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
