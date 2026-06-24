from __future__ import annotations

import sys
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
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils import ROOT_utils  # noqa: E402
from utils.ROOT_utils import sum_th1s  # noqa: E402
from utils.variable_names import variable_data  # noqa: E402

YEAR = 2017
VARIABLE = "MTW"
WIDTH_VARIABLES = ("TauTrackWidthPt1000PV", "TauTrackWidthPt500PV")
CONFIG_LABEL = "tau_width_reweighting"
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_REWEIGHTING_COMPARISON = True
OUTPUT_DIR = VALIDATION_OUTPUT / "tau_width_reweighting"
SUMMARY_PATH = OUTPUT_DIR / "tau_width_reweighting_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_tau_width_reweighting_multiwidth.root"


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


def positive_shape(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    shape = hist.Clone(name)
    shape.SetDirectory(0)
    for bin_idx in range(1, shape.GetNbinsX() + 1):
        if shape.GetBinContent(bin_idx) < 0:
            shape.SetBinContent(bin_idx, 0.0)
            shape.SetBinError(bin_idx, 0.0)
    integral = hist_integral(shape)
    if integral > 0:
        shape.Scale(1.0 / integral)
    return shape


def width_ratio_hist(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
    name: str,
) -> ROOT.TH1:
    ratio_hist = numerator.Clone(name)
    ratio_hist.SetDirectory(0)
    for bin_idx in range(1, ratio_hist.GetNbinsX() + 1):
        numerator_value = numerator.GetBinContent(bin_idx)
        denominator_value = denominator.GetBinContent(bin_idx)
        if denominator_value <= 0:
            ratio_hist.SetBinContent(bin_idx, 0.0)
        else:
            ratio_hist.SetBinContent(bin_idx, numerator_value / denominator_value)
        ratio_hist.SetBinError(bin_idx, 0.0)
    return ratio_hist


def fill_reweighted_prediction(
    analysis: Analysis,
    *,
    width_variable: str,
    selection_fail: str,
    selection_pass: str,
    true_selection_fail: str,
    ff_hist: ROOT.TH1,
    width_ratio: ROOT.TH1,
    output_name: str,
) -> ROOT.TH1:
    ROOT.gInterpreter.Declare(
        f"TH1F* FF_hist_{output_name} = reinterpret_cast<TH1F*>({ROOT.addressof(ff_hist)});"
    )
    ROOT.gInterpreter.Declare(
        f"TH1F* width_ratio_{output_name} = reinterpret_cast<TH1F*>({ROOT.addressof(width_ratio)});"
    )

    weight_col = f"FF_width_weight_{output_name}"
    weight_expr = (
        f"reco_weight"
        f" * FF_hist_{output_name}->GetBinContent(FF_hist_{output_name}->FindBin({FAKES_SOURCE}))"
        f" * width_ratio_{output_name}->GetBinContent(width_ratio_{output_name}->FindBin({width_variable}))"
    )

    h_bins = analysis[analysis.data_sample].get_binnings(VARIABLE, selection_pass)
    h_data = ROOT.TH1F(
        f"{output_name}_data",
        output_name,
        *ROOT_utils.get_TH1_bin_args(**h_bins),
    )
    data_df = analysis[analysis.data_sample].filters[NOMINAL_NAME][selection_fail].df.Define(
        weight_col,
        weight_expr,
    )
    data_ptr = data_df.Fill(h_data, [VARIABLE, weight_col])

    mc_ptrs = []
    for mc_sample in MC_SAMPLES:
        h_mc = ROOT.TH1F(
            f"{output_name}_{mc_sample}",
            output_name,
            *ROOT_utils.get_TH1_bin_args(**h_bins),
        )
        mc_df = analysis[mc_sample].filters[NOMINAL_NAME][true_selection_fail].df.Define(
            weight_col,
            weight_expr,
        )
        mc_ptrs.append(mc_df.Fill(h_mc, [VARIABLE, weight_col]))

    prediction = data_ptr.GetValue() - sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs])
    prediction.SetName(output_name)
    prediction.SetTitle(output_name)
    prediction.SetDirectory(0)
    for bin_idx in range(1, prediction.GetNbinsX() + 1):
        prediction.SetBinError(bin_idx, abs(prediction.GetBinContent(bin_idx)) * 0.1)
    analysis.histograms[output_name] = prediction
    return prediction


if __name__ == "__main__":
    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    low_met_cut = Cut("MET < 100", "MET_met < 100")
    target_cuts = [
        Cut(r"$m_T^W >= 350$", "MTW >= 350"),
        Cut("MET >= 170", "MET_met >= 170"),
    ]

    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        prefix = f"{CONFIG_LABEL}_{WP}_{prong}prong"

        data_selections[f"{prefix}_derive_passID"] = (
            base_cuts + [low_met_cut, PASS_MEDIUM, prong_cut]
        )
        data_selections[f"{prefix}_derive_failID"] = (
            base_cuts + [low_met_cut, FAIL_MEDIUM, prong_cut]
        )
        data_selections[f"{prefix}_target_passID"] = (
            base_cuts + target_cuts + [PASS_MEDIUM, prong_cut]
        )
        data_selections[f"{prefix}_target_failID"] = (
            base_cuts + target_cuts + [FAIL_MEDIUM, prong_cut]
        )

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_exists
    )

    width_bins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25, 0.35]
    binnings = {"": {**BINNINGS, **{variable: width_bins for variable in WIDTH_VARIABLES}}}

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_tau_width_reweighting",
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
            *WIDTH_VARIABLES,
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={VARIABLE, FAKES_SOURCE, *WIDTH_VARIABLES},
        binnings=binnings,
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing tau-width reweighting cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    lines = [
        "# Tau-width reweighting diagnostic",
        "",
        "This validation tests whether a tau-width composition reweighting changes the "
        "low-MET fake-factor prediction in the expected direction. It is not a nominal "
        "analysis correction.",
        "",
        "- width variables: "
        + ", ".join(f"`{width_variable}`" for width_variable in WIDTH_VARIABLES),
        f"- fake-factor source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        "- width weights are uncapped",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "Two directions are tested:",
        "",
        "- `application_to_lowmet`: reweights the high-MET anti-ID application region to "
        "look more like the low-MET fake-factor denominator. This follows the ATLAS-style "
        "idea of using width reweighting as a transfer systematic.",
        "- `lowmet_to_application`: the opposite stress test, upweighting the high-MET-like "
        "width tail. This is diagnostic only.",
        "",
        "| Width variable | Prong | Prediction | Integral | Target integral | Prediction / target | "
        "Relative to nominal fake prediction |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]

    comparison_hists: dict[tuple[str, int], list[tuple[str, ROOT.TH1]]] = {}

    for prong in PRONGS:
        prefix = f"{CONFIG_LABEL}_{WP}_{prong}prong"
        derive_pass = f"{prefix}_derive_passID"
        derive_fail = f"{prefix}_derive_failID"
        target_pass = f"{prefix}_target_passID"
        target_fail = f"{prefix}_target_failID"
        estimate_name = f"{prefix}_{FAKES_SOURCE}_src"

        if not loaded_hists:
            analysis.do_fakes_estimate(
                FAKES_SOURCE,
                (VARIABLE,),
                derive_pass,
                derive_fail,
                target_pass,
                target_fail,
                f"trueTau_{derive_pass}",
                f"trueTau_{derive_fail}",
                f"trueTau_{target_fail}",
                name=estimate_name,
                systematic=NOMINAL_NAME,
                save_intermediates=True,
            )

        target = fake_like_hist(analysis, VARIABLE, target_pass)
        nominal = analysis.histograms[f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"]
        ff_hist = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF"]

        for width_variable in WIDTH_VARIABLES:
            low_width = fake_like_hist(analysis, width_variable, derive_fail)
            target_width = fake_like_hist(analysis, width_variable, target_fail)
            low_shape = positive_shape(low_width, f"{prefix}_{width_variable}_low_shape")
            target_shape = positive_shape(
                target_width,
                f"{prefix}_{width_variable}_target_shape",
            )

            application_to_lowmet = width_ratio_hist(
                low_shape,
                target_shape,
                f"{prefix}_{width_variable}_application_to_lowmet",
            )
            lowmet_to_application = width_ratio_hist(
                target_shape,
                low_shape,
                f"{prefix}_{width_variable}_lowmet_to_application",
            )
            analysis.histograms[application_to_lowmet.GetName()] = application_to_lowmet
            analysis.histograms[lowmet_to_application.GetName()] = lowmet_to_application

            if not loaded_hists:
                app_to_low_prediction = fill_reweighted_prediction(
                    analysis,
                    width_variable=width_variable,
                    selection_fail=target_fail,
                    selection_pass=target_pass,
                    true_selection_fail=f"trueTau_{target_fail}",
                    ff_hist=ff_hist,
                    width_ratio=application_to_lowmet,
                    output_name=(
                        f"{estimate_name}_{width_variable}_{VARIABLE}_"
                        "application_to_lowmet_width_rw"
                    ),
                )
                low_to_app_prediction = fill_reweighted_prediction(
                    analysis,
                    width_variable=width_variable,
                    selection_fail=target_fail,
                    selection_pass=target_pass,
                    true_selection_fail=f"trueTau_{target_fail}",
                    ff_hist=ff_hist,
                    width_ratio=lowmet_to_application,
                    output_name=(
                        f"{estimate_name}_{width_variable}_{VARIABLE}_"
                        "lowmet_to_application_width_rw"
                    ),
                )
            else:
                app_to_low_prediction = analysis.histograms[
                    f"{estimate_name}_{width_variable}_{VARIABLE}_"
                    "application_to_lowmet_width_rw"
                ]
                low_to_app_prediction = analysis.histograms[
                    f"{estimate_name}_{width_variable}_{VARIABLE}_"
                    "lowmet_to_application_width_rw"
                ]

            target_integral = hist_integral(target)
            nominal_integral = hist_integral(nominal)
            predictions = (
                ("nominal low-MET fake factor", nominal),
                ("application_to_lowmet width reweight", app_to_low_prediction),
                ("lowmet_to_application width reweight", low_to_app_prediction),
            )
            comparison_hists[(width_variable, prong)] = [
                ("Data - nonfake MC", target),
                *predictions,
            ]

            for label, prediction in predictions:
                prediction_integral = hist_integral(prediction)
                lines.append(
                    f"| `{width_variable}` | {prong} | {label} | "
                    f"{prediction_integral:.3f} | {target_integral:.3f} | "
                    f"{ratio(prediction_integral, target_integral):.3f} | "
                    f"{ratio(prediction_integral, nominal_integral):.3f} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "`application_to_lowmet` asks how much the fake prediction changes if the "
            "high-MET anti-ID events are reweighted to have the same width shape as the "
            "low-MET fake-factor denominator. A large change would support a composition "
            "systematic. `lowmet_to_application` is the opposite stress test and should not "
            "be used as a nominal correction by itself.",
        ]
    )

    if PLOT_REWEIGHTING_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "tau_width_reweighting"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])
        for (width_variable, prong), labelled_hists in comparison_hists.items():
            filename = f"{width_variable}_{prong}prong_tau_width_reweighting.png"
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange", "tab:green"],
                histstyle=["step", "step", "step", "step"],
                linestyle=["-", "-", "--", "-."],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{width_variable} width reweighting | {prong}-prong",
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

    if run_event_loops and not loaded_hists:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
