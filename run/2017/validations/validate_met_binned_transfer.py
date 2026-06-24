from __future__ import annotations

import sys
from pathlib import Path

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
MET_SLICES = (
    ("MET_lt120", "MET < 120", "MET_met < 120"),
    ("MET_120_170", "120 <= MET < 170", "(MET_met >= 120) && (MET_met < 170)"),
)
MET_CUT_COMPARISON_DERIVATIONS = (
    ("derive_MET_lt170", "derive MET < 170", "MET_met < 170"),
    ("derive_MET_lt120", "derive MET < 120", "MET_met < 120"),
)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_MET_CUT_COMPARISON = True
OUTPUT_DIR = VALIDATION_OUTPUT / "met_cut_comparison"
SUMMARY_PATH = OUTPUT_DIR / "met_cut_comparison_summary.md"


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


def validation_target(analysis: Analysis, selection: str):
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
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    selection_binnings = {"": BINNINGS}

    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    derivation_mtw = Cut(
        r"shadow $m_T^W$ sideband",
        f"(MTW >= {MTW_SHADOW_MIN:g}) && (MTW < {MTW_NOMINAL_MIN:g})",
    )
    validation_mtw = Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}")

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        for met_key, _met_label, met_cut_expr in MET_SLICES:
            met_cut = Cut(_met_label, met_cut_expr)
            prefix = f"{CONFIG_LABEL}_mtw_overlap_{met_key}_{WP}_{prong}prong"

            data_selections[f"{prefix}_derive_passID"] = (
                base_cuts + [derivation_mtw, met_cut, PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_derive_failID"] = (
                base_cuts + [derivation_mtw, met_cut, FAIL_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + [validation_mtw, met_cut, PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + [validation_mtw, met_cut, FAIL_MEDIUM, prong_cut]
            )

            for suffix in (
                "derive_passID",
                "derive_failID",
                "validate_passID",
                "validate_failID",
            ):
                selection = f"{prefix}_{suffix}"
                mc_selections[selection] = data_selections[selection]
                mc_selections[f"trueTau_{selection}"] = data_selections[selection] + [PASS_TRUETAU]

        for derivation_key, derivation_label, derivation_met_expr in MET_CUT_COMPARISON_DERIVATIONS:
            prefix = f"{CONFIG_LABEL}_met_cut_choice_{derivation_key}_{WP}_{prong}prong"
            derivation_met = Cut(derivation_label, derivation_met_expr)
            validation_met = Cut("validate MET < 170", "MET_met < 170")

            data_selections[f"{prefix}_derive_passID"] = (
                base_cuts + [derivation_mtw, derivation_met, PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_derive_failID"] = (
                base_cuts + [derivation_mtw, derivation_met, FAIL_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + [validation_mtw, validation_met, PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + [validation_mtw, validation_met, FAIL_MEDIUM, prong_cut]
            )

            for suffix in (
                "derive_passID",
                "derive_failID",
                "validate_passID",
                "validate_failID",
            ):
                selection = f"{prefix}_{suffix}"
                mc_selections[selection] = data_selections[selection]
                mc_selections[f"trueTau_{selection}"] = data_selections[selection] + [PASS_TRUETAU]

    cache_file = OUTPUT_DIR / "root" / "validate_met_binned_transfer.root"
    cache_exists = cache_file.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (not LOAD_SAVED_HISTS or not cache_exists)

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_met_binned_transfer",
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
        binnings=selection_binnings,
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(cache_file)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing MET-binned transfer cache and event loops are disabled: "
            f"{cache_file}"
        )

    lines = [
        "# MET-binned fake-transfer validation",
        "",
        "This validation keeps the fake-factor source as `TauPt`, but derives and applies it "
        "separately in coarse MET slices. The derivation and validation regions overlap in "
        "MET and are split instead by MTW, so this is a genuine test of whether adding MET "
        "dependence improves transfer.",
        "",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- derivation MTW: `{MTW_SHADOW_MIN} <= MTW < {MTW_NOMINAL_MIN}`",
        f"- validation MTW: `MTW >= {MTW_NOMINAL_MIN}`",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- cache file: `{cache_file.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "| MET slice | Prong | CR numerator | CR denominator | Negative numerator bins | "
        "Tiny/non-positive denominator bins | Validation fail-ID input | Predicted fakes | "
        "Validation target | Prediction / target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for met_key, met_label, _met_cut_expr in MET_SLICES:
        for prong in PRONGS:
            prefix = f"{CONFIG_LABEL}_mtw_overlap_{met_key}_{WP}_{prong}prong"
            estimate_name = f"{prefix}_{FAKES_SOURCE}_src"
            derive_pass = f"{prefix}_derive_passID"
            derive_fail = f"{prefix}_derive_failID"
            validate_pass = f"{prefix}_validate_passID"
            validate_fail = f"{prefix}_validate_failID"

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
            target = validation_target(analysis, validate_pass)
            negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                numerator, denominator
            )

            predicted_integral = hist_integral(prediction)
            target_integral = hist_integral(target)
            lines.append(
                f"| {met_label} | {prong} | {hist_integral(numerator):.3f} | "
                f"{hist_integral(denominator):.3f} | {negative_numerator_bins} | "
                f"{tiny_denominator_bins} | {hist_integral(validation_fail_input):.3f} | "
                f"{predicted_integral:.3f} | {target_integral:.3f} | "
                f"{ratio(predicted_integral, target_integral):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If these ratios are closer to unity than the independent `met_cr_split` test, "
            "then the fake model likely needs explicit MET dependence. If they remain poor, "
            "the problem is more likely the nonfake subtraction, the fail-ID composition, or "
            "the prong-specific control-region definition rather than the missing MET axis alone.",
            "",
            "## MET cut comparison",
            "",
            "This section compares two fake-factor derivation choices against the same "
            "independent validation target: `MTW >= 350` and `MET < 170`. This directly tests "
            "whether reducing the fake-factor control region from `MET < 170` to `MET < 120` "
            "improves the fake prediction.",
            "",
            "| Prong | Derivation region | CR numerator | CR denominator | "
            "Negative numerator bins | Tiny/non-positive denominator bins | "
            "Validation fail-ID input | Predicted fakes | Validation target | "
            "Prediction / target | |ratio - 1| |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    comparison_hists: dict[int, list[tuple[str, object]]] = {}
    for prong in PRONGS:
        target_selection = (
            f"{CONFIG_LABEL}_met_cut_choice_derive_MET_lt170_{WP}_{prong}prong_validate_passID"
        )
        target = validation_target(analysis, target_selection)
        comparison_hists[prong] = [("Data - nonfake MC", target)]

        for derivation_key, derivation_label, _derivation_met_expr in MET_CUT_COMPARISON_DERIVATIONS:
            prefix = f"{CONFIG_LABEL}_met_cut_choice_{derivation_key}_{WP}_{prong}prong"
            estimate_name = f"{prefix}_{FAKES_SOURCE}_src"
            derive_pass = f"{prefix}_derive_passID"
            derive_fail = f"{prefix}_derive_failID"
            validate_pass = f"{prefix}_validate_passID"
            validate_fail = f"{prefix}_validate_failID"

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
            target_integral = hist_integral(target)
            prediction_over_target = ratio(predicted_integral, target_integral)
            lines.append(
                f"| {prong} | {derivation_label} | {hist_integral(numerator):.3f} | "
                f"{hist_integral(denominator):.3f} | {negative_numerator_bins} | "
                f"{tiny_denominator_bins} | {hist_integral(validation_fail_input):.3f} | "
                f"{predicted_integral:.3f} | {target_integral:.3f} | "
                f"{prediction_over_target:.3f} | {abs(prediction_over_target - 1):.3f} |"
            )
            comparison_hists[prong].append((derivation_label, prediction))

    if PLOT_MET_CUT_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "met_cut_comparison"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        for prong, labelled_hists in comparison_hists.items():
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange"],
                histstyle=["step", "step", "step"],
                linestyle=["-", "-", "--"],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{CONFIG_LABEL} | {prong}-prong | fake CR MET cut comparison",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                logx=True,
                ratio_plot=True,
                ratio_label="Prediction / target",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=f"{CONFIG_LABEL}_{prong}prong_MET_cut_fake_transfer_comparison.png",
            )

    if run_event_loops:
        analysis.save_hists(filename=cache_file.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
