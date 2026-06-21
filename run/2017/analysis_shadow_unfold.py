import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT
from src.analysis import Analysis
from src.cutting import Cut
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from utils import ROOT_utils
from utils.helper_functions import smart_join
from utils.plotting_tools import Hist2dOpts, PlotKwargs
from utils.ROOT_utils import get_th1_bin_edges, sum_th1s
from utils.variable_names import variable_data

from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples, signal_sample

YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
WP = "medium"
VARS = ("MTW", "TauPt")
ITERATIONS = (
    0,
    1,
    2,
    # 4,
    # 8,
)
FAKES_SOURCE = "TauPt"
DO_FULL_SYSTEMATICS = False
LOAD_SAVED_HISTS = False

SKIP_SYS = {
    r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
    r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
}

TRUTHS = {
    "MTW": "TruthMTW",
    "TauPt": "VisTruthTauPt",
}
SYMBOLS = {
    "MTW": r"m^W_\mathrm{T}",
    "TauPt": r"p_\mathrm{T}^{\tau_\mathrm{had-vis}}",
}

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
PASS_TRUTH = Cut(r"Pass Truth", r"(passTruth == 1)")
TRUTH_HAD_TAU = Cut(
    r"Truth Hadronic Tau",
    r"TruthTau_isHadronic && ((TruthTau_nChargedTracks == 1) || "
    r"(TruthTau_nChargedTracks == 3))",
)


@dataclass(frozen=True)
class ShadowConfig:
    """One phase-space definition used in the closure test."""

    label: str
    unfolded_var: str | None
    mtw_min: float
    taupt_min: float
    met_min: float


@dataclass(frozen=True)
class ResponseComponents:
    """Response pieces needed by RooUnfold and closure diagnostics."""

    response: ROOT.RooUnfoldResponse
    reco: ROOT.TH1
    truth: ROOT.TH1
    matrix: ROOT.TH2


CONFIGS = (
    ShadowConfig("no_shadow_bin", None, mtw_min=350, taupt_min=170, met_min=170),
    ShadowConfig("MTW_shadow_bin_200", "MTW", mtw_min=200, taupt_min=170, met_min=170),
    ShadowConfig("MTW_shadow_bin_250", "MTW", mtw_min=250, taupt_min=170, met_min=170),
    ShadowConfig("MTW_shadow_bin_300", "MTW", mtw_min=300, taupt_min=170, met_min=170),
    ShadowConfig("TauPt_shadow_bin_200", "TauPt", mtw_min=350, taupt_min=100, met_min=170),
    ShadowConfig("TauPt_shadow_bin_250", "TauPt", mtw_min=350, taupt_min=125, met_min=170),
    ShadowConfig("TauPt_shadow_bin_300", "TauPt", mtw_min=350, taupt_min=150, met_min=170),
)


def crop_to_nominal_binning(source: ROOT.TH1, target: ROOT.TH1, name: str) -> ROOT.TH1:
    """Clone source into target binning, dropping one leading shadow bin if present."""
    source_edges = get_th1_bin_edges(source)
    target_edges = get_th1_bin_edges(target)

    if len(source_edges) == len(target_edges) and all(
        abs(float(source_edge) - float(target_edge)) < 1e-6
        for source_edge, target_edge in zip(source_edges, target_edges, strict=True)
    ):
        clone = source.Clone(name)
        clone.SetDirectory(0)
        return clone

    has_one_leading_shadow_bin = len(source_edges) == len(target_edges) + 1 and all(
        abs(float(source_edge) - float(target_edge)) < 1e-6
        for source_edge, target_edge in zip(source_edges[1:], target_edges, strict=True)
    )
    if not has_one_leading_shadow_bin:
        raise ValueError(
            f"Cannot crop histogram '{source.GetName()}' to '{target.GetName()}': "
            "source is neither identical to target nor target plus one leading shadow bin."
        )

    cropped = ROOT.TH1D(name, source.GetTitle(), len(target_edges) - 1, target_edges)
    cropped.SetDirectory(0)
    for bin_i in range(1, target.GetNbinsX() + 1):
        cropped.SetBinContent(bin_i, source.GetBinContent(bin_i + 1))
        cropped.SetBinError(bin_i, source.GetBinError(bin_i + 1))
    return cropped


def covariance_from_hist(h: ROOT.TH1, name: str) -> ROOT.TH2D:
    """Build a diagonal covariance matrix from a histogram's bin errors."""
    cov = ROOT.TH2D(name, name, h.GetNbinsX(), 0, h.GetNbinsX(), h.GetNbinsX(), 0, h.GetNbinsX())
    cov.SetDirectory(0)
    for bin_i in range(1, h.GetNbinsX() + 1):
        cov.SetBinContent(bin_i, bin_i, h.GetBinError(bin_i) ** 2)
    return cov


def unfold_histogram(
    analysis: Analysis,
    hist: ROOT.TH1,
    response: ResponseComponents,
    iter_count: int,
) -> tuple[ROOT.TH1, ROOT.TH2]:
    """Unfold one histogram and return the unfolded histogram plus covariance."""
    if iter_count == 0 and response.reco.GetNbinsX() == response.truth.GetNbinsX():
        result = analysis.unfold_bin_by_bin(hist, response.reco, response.truth)
        return result.unfolded, covariance_from_hist(result.unfolded, f"{hist.GetName()}_cov")

    if iter_count == 0:
        unfolded = ROOT.RooUnfoldBinByBin(response.response, hist)
    else:
        unfolded = ROOT.RooUnfoldBayes(response.response, hist, iter_count)
    return unfolded.Hunfold(), ROOT.TH2D(unfolded.Eunfold())


def scale_and_crop_unfolded(hist: ROOT.TH1, nominal_truth: ROOT.TH1, name: str) -> ROOT.TH1:
    """Convert unfolded event yields to cross-section and crop any leading shadow bin."""
    scaled = hist.Clone(f"{name}_scaled")
    scaled.SetDirectory(0)
    scaled.Scale(1 / LUMI)
    return crop_to_nominal_binning(scaled, nominal_truth, name)


def closure_metrics(unfolded_signal: ROOT.TH1, truth: ROOT.TH1) -> tuple[float, float, float]:
    """Return mean deviation, max deviation, and integral ratio against truth."""
    deviations = []
    for bin_i in range(1, truth.GetNbinsX() + 1):
        truth_val = truth.GetBinContent(bin_i)
        if truth_val == 0:
            continue
        deviations.append(abs(unfolded_signal.GetBinContent(bin_i) / truth_val - 1))
    mean_dev = float(np.mean(deviations)) if deviations else 0.0
    max_dev = float(np.max(deviations)) if deviations else 0.0
    integral_ratio = (
        unfolded_signal.Integral() / truth.Integral() if truth.Integral() != 0 else float("nan")
    )
    return mean_dev, max_dev, float(integral_ratio)


if __name__ == "__main__":
    output_root = Path(__file__).absolute().parent.parent.parent / "outputs" / Path(__file__).stem
    plotter = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label=Path(__file__).stem,
        output_dir=output_root,
        log_level=10,
        log_out="both",
    )
    plotter.logger.info("Starting analysis_shadow_unfold.py")
    plotter.logger.info("DO_FULL_SYSTEMATICS = %s", DO_FULL_SYSTEMATICS)
    if DO_FULL_SYSTEMATICS:
        plotter.logger.info(
            "Full systematics mode enabled; missing shadow variations will fail loudly."
        )

    # Build all threshold-specific measured-input selections in one pass.
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    response_selections: dict[str, list[Cut]] = {}
    selection_binnings: dict[str, dict[str, np.ndarray]] = {"": BINNINGS}

    for config in CONFIGS:
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        cr_pass = f"{config.label}_{WP}_CR_passID"
        cr_fail = f"{config.label}_{WP}_CR_failID"
        true_sr_pass = f"trueTau_{sr_pass}"
        true_sr_fail = f"trueTau_{sr_fail}"
        true_cr_pass = f"trueTau_{cr_pass}"
        true_cr_fail = f"trueTau_{cr_fail}"
        truth_selection = f"{config.label}_truth_tau"
        reco_selection = f"{config.label}_{WP}_reco_tau"
        truth_reco_selection = f"{config.label}_{WP}_truth_reco_tau"

        reco_sr_cuts = [
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
            PASS_ETA,
            Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
            Cut(r"$E_T^{\mathrm{miss}}$ threshold", f"MET_met > {config.met_min:g}"),
        ]
        reco_cr_cuts = [
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
            PASS_ETA,
            Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
            Cut(r"$E_T^{\mathrm{miss}}$ control region", f"MET_met < {config.met_min:g}"),
        ]
        truth_cuts = [
            PASS_TRUTH,
            Cut(
                r"Pass truth fiducial region",
                f"(VisTruthTauPt > {config.taupt_min:g}) && "
                f"(TruthMTW > {config.mtw_min:g}) && "
                f"(TruthNeutrinoPt > {config.met_min:g})"
                r"&& ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))"
                r"&& (((abs(VisTruthTauEta) < 1.37) || (1.52 < abs(VisTruthTauEta))) "
                r"&& (abs(VisTruthTauEta) < 2.47))",
            ),
            TRUTH_HAD_TAU,
        ]

        data_selections[sr_pass] = reco_sr_cuts + [PASS_MEDIUM]
        data_selections[sr_fail] = reco_sr_cuts + [FAIL_MEDIUM]
        data_selections[cr_pass] = reco_cr_cuts + [PASS_MEDIUM]
        data_selections[cr_fail] = reco_cr_cuts + [FAIL_MEDIUM]

        for selection in (sr_pass, sr_fail, cr_pass, cr_fail):
            mc_selections[selection] = data_selections[selection]
        mc_selections[true_sr_pass] = data_selections[sr_pass] + [PASS_TRUETAU]
        mc_selections[true_sr_fail] = data_selections[sr_fail] + [PASS_TRUETAU]
        mc_selections[true_cr_pass] = data_selections[cr_pass] + [PASS_TRUETAU]
        mc_selections[true_cr_fail] = data_selections[cr_fail] + [PASS_TRUETAU]

        response_selections[truth_selection] = truth_cuts
        response_selections[reco_selection] = reco_sr_cuts + [PASS_MEDIUM]
        response_selections[truth_reco_selection] = truth_cuts + reco_sr_cuts + [PASS_MEDIUM]

        config_binnings = dict(BINNINGS)
        if config.unfolded_var == "MTW":
            mtw_bins = np.array(
                [
                    config.mtw_min,
                    350,
                    375,
                    400,
                    430,
                    465,
                    500,
                    550,
                    600,
                    700,
                    850,
                    1000,
                    2000,
                ],
                dtype="double",
            )
            config_binnings["MTW"] = mtw_bins
            config_binnings["TruthMTW"] = mtw_bins
        elif config.unfolded_var == "TauPt":
            taupt_bins = np.array(
                [config.taupt_min, 170, 200, 250, 300, 350, 425, 500, 600, 1000],
                dtype="double",
            )
            config_binnings["TauPt"] = taupt_bins
            config_binnings["VisTruthTauPt"] = taupt_bins

        for selection in (
            sr_pass,
            sr_fail,
            cr_pass,
            cr_fail,
            true_sr_pass,
            true_sr_fail,
            true_cr_pass,
            true_cr_fail,
            truth_selection,
            reco_selection,
            truth_reco_selection,
        ):
            selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings

    label_regex = "|".join(re.escape(config.label) for config in CONFIGS)

    measured_analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=DO_FULL_SYSTEMATICS,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="analysis_shadow_unfold_measured",
        output_dir=output_root / "measured",
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
            "TauCharge",
        },
        import_missing_columns_as_nan=True,
        histogram_vars=set(VARS),
        systematics_for_selection={rf"^({label_regex})_{WP}_SR_passID$"}
        if DO_FULL_SYSTEMATICS
        else set(),
        skip_sys=SKIP_SYS,
        binnings=selection_binnings,
    )
    measured_analysis.print_metadata_table(datasets=measured_analysis.mc_samples)

    response_analysis = Analysis(
        {"wtaunu_had": signal_sample(selections=response_selections)},
        year=YEAR,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=DO_FULL_SYSTEMATICS,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="analysis_shadow_unfold_response",
        output_dir=output_root / "response",
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
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=set(VARS) | set(TRUTHS.values()),
        hists_2d={
            "MTW_TruthMTW": Hist2dOpts("MTW", "TruthMTW", "reco_weight"),
            "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt", "reco_weight"),
        },
        do_unweighted=True,
        systematics_for_selection={rf"^({label_regex})_{WP}_(reco_tau|truth_reco_tau)$"}
        if DO_FULL_SYSTEMATICS
        else set(),
        skip_sys=SKIP_SYS,
        binnings=selection_binnings,
    )

    nominal_truth_hists = {
        var: response_analysis.get_hist(
            TRUTHS[var],
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection="no_shadow_bin_truth_tau",
        )
        for var in VARS
    }

    closure_rows: list[tuple[str, str, int, float, float, float]] = []

    for config in CONFIGS:
        vars_for_config = VARS if config.unfolded_var is None else (config.unfolded_var,)
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        cr_pass = f"{config.label}_{WP}_CR_passID"
        cr_fail = f"{config.label}_{WP}_CR_failID"
        true_sr_pass = f"trueTau_{sr_pass}"
        true_sr_fail = f"trueTau_{sr_fail}"
        true_cr_pass = f"trueTau_{cr_pass}"
        true_cr_fail = f"trueTau_{cr_fail}"
        truth_selection = f"{config.label}_truth_tau"
        reco_selection = f"{config.label}_{WP}_reco_tau"
        truth_reco_selection = f"{config.label}_{WP}_truth_reco_tau"
        fakes_name = f"{config.label}_{WP}"

        plotter.logger.info("Running variable-specific shadow-bin closure for %s", config.label)
        measured_analysis.do_fakes_estimate(
            FAKES_SOURCE,
            vars_for_config,
            cr_pass,
            cr_fail,
            sr_pass,
            sr_fail,
            true_cr_pass,
            true_cr_fail,
            true_sr_pass,
            true_sr_fail,
            name=fakes_name,
            systematic=NOMINAL_NAME,
            save_intermediates=True,
        )
        if not DO_FULL_SYSTEMATICS:
            plotter.logger.info(
                "DO_FULL_SYSTEMATICS is False: producing central-value closure only for %s.",
                config.label,
            )

        for var in vars_for_config:
            data = measured_analysis.get_hist(
                var,
                dataset=measured_analysis.data_sample,
                systematic=NOMINAL_NAME,
                selection=sr_pass,
            )
            signal = measured_analysis.get_hist(
                var,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=sr_pass,
            )
            backgrounds = [
                measured_analysis.get_hist(
                    var,
                    dataset=background,
                    systematic=NOMINAL_NAME,
                    selection=sr_pass,
                )
                for background in measured_analysis.mc_samples
                if background != "wtaunu_had"
            ]
            fakes = measured_analysis.histograms[
                f"{fakes_name}_{var}_fakes_bkg_{FAKES_SOURCE}_src"
            ]
            background = sum_th1s(*(backgrounds + [fakes]))
            data_sig = data - background
            data_sig.SetName(f"{config.label}_{var}_data_minus_background")
            signal = signal.Clone(f"{config.label}_{var}_signal")
            signal.SetDirectory(0)

            reco = response_analysis.get_hist(
                var,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=reco_selection,
            )
            truth_response = response_analysis.get_hist(
                TRUTHS[var],
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=truth_selection,
            )
            matrix = response_analysis.get_hist(
                f"{var}_{TRUTHS[var]}",
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=truth_reco_selection,
            )
            response = ResponseComponents(
                response=ROOT.RooUnfoldResponse(reco, truth_response, matrix),
                reco=reco,
                truth=truth_response,
                matrix=matrix,
            )
            nominal_truth = nominal_truth_hists[var]
            truth = Histogram1D(th1=nominal_truth) / LUMI

            if DO_FULL_SYSTEMATICS:
                sys_names = sorted(
                    {
                        sys.removesuffix("__1up").removesuffix("__1down")
                        for sys in (
                            response_analysis["wtaunu_had"].eff_sys_set
                            | response_analysis["wtaunu_had"].tes_sys_set
                        )
                    }
                )
                if not sys_names:
                    raise RuntimeError(
                        "DO_FULL_SYSTEMATICS is enabled but no response systematics exist."
                    )

                nominal_unfolded, _ = unfold_histogram(plotter, signal, response, 0)
                nominal_unfolded = scale_and_crop_unfolded(
                    nominal_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_nominal_response_sys_reference",
                )
                response_uncertainties = []
                response_uncertainty_labels = []
                for sys_name in sys_names:
                    up = f"{sys_name}__1up"
                    down = f"{sys_name}__1down"
                    try:
                        reco_up = response_analysis.get_hist(
                            var,
                            dataset="wtaunu_had",
                            systematic=up,
                            selection=reco_selection,
                        )
                        reco_down = response_analysis.get_hist(
                            var,
                            dataset="wtaunu_had",
                            systematic=down,
                            selection=reco_selection,
                        )
                        matrix_up = response_analysis.get_hist(
                            f"{var}_{TRUTHS[var]}",
                            dataset="wtaunu_had",
                            systematic=up,
                            selection=truth_reco_selection,
                        )
                        matrix_down = response_analysis.get_hist(
                            f"{var}_{TRUTHS[var]}",
                            dataset="wtaunu_had",
                            systematic=down,
                            selection=truth_reco_selection,
                        )
                    except KeyError as exc:
                        raise KeyError(
                            f"Missing full shadow-region response systematic for "
                            f"{config.label} {var}: {sys_name}"
                        ) from exc

                    response_up = ResponseComponents(
                        response=ROOT.RooUnfoldResponse(reco_up, truth_response, matrix_up),
                        reco=reco_up,
                        truth=truth_response,
                        matrix=matrix_up,
                    )
                    response_down = ResponseComponents(
                        response=ROOT.RooUnfoldResponse(reco_down, truth_response, matrix_down),
                        reco=reco_down,
                        truth=truth_response,
                        matrix=matrix_down,
                    )
                    unfolded_up, _ = unfold_histogram(plotter, signal, response_up, 0)
                    unfolded_down, _ = unfold_histogram(plotter, signal, response_down, 0)
                    unfolded_up = scale_and_crop_unfolded(
                        unfolded_up,
                        nominal_truth,
                        f"{config.label}_{var}_{sys_name}_up_response_sys",
                    )
                    unfolded_down = scale_and_crop_unfolded(
                        unfolded_down,
                        nominal_truth,
                        f"{config.label}_{var}_{sys_name}_down_response_sys",
                    )
                    uncertainty = ROOT_utils.th1_max_abs_deviation(
                        unfolded_up,
                        unfolded_down,
                        nominal_unfolded,
                    )
                    response_uncertainties.append(
                        ROOT_utils.th1_relative_uncertainty(
                            uncertainty,
                            nominal_unfolded,
                            name=f"{config.label}_{var}_{sys_name}_response_uncertainty",
                        )
                    )
                    response_uncertainty_labels.append(sys_name)

                plotter.paths.plot_dir = (
                    plotter.paths.output_dir / "plots" / config.label / var / "systematics"
                )
                plotter.plot(
                    response_uncertainties,
                    label=response_uncertainty_labels,
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    ylabel="Response uncertainty / %",
                    title=smart_join(config.label, var, "medium ID", sep=" | "),
                    do_stat=False,
                    do_syst=False,
                    logx=True,
                    filename=f"{config.label}_{var}_response_systematics.png",
                )

            plotter.paths.plot_dir = plotter.paths.output_dir / "plots" / config.label / var
            plotter.plot_2d(
                response.matrix,
                ylabel=f"Truth {TRUTHS[var]}",
                xlabel=f"Reco {var}",
                title=smart_join(config.label, var, "response matrix", sep=" | "),
                labels=False,
                label_params={"llabel": ""},
                filename=f"{config.label}_{var}_response_matrix.png",
            )

            unfolded_by_iteration = {}
            for iter_count in ITERATIONS:
                data_unfolded, data_cov = unfold_histogram(plotter, data_sig, response, iter_count)
                signal_unfolded, _ = unfold_histogram(plotter, signal, response, iter_count)
                data_unfolded = scale_and_crop_unfolded(
                    data_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_{iter_count}iter_data_unfolded",
                )
                signal_unfolded = scale_and_crop_unfolded(
                    signal_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_{iter_count}iter_signal_unfolded",
                )
                unfolded_by_iteration[iter_count] = signal_unfolded
                mean_dev, max_dev, integral_ratio = closure_metrics(signal_unfolded, truth.TH1)
                closure_rows.append(
                    (config.label, var, iter_count, mean_dev, max_dev, integral_ratio)
                )

                plotter.paths.plot_dir = plotter.paths.output_dir / "plots" / config.label / var
                plot_args: PlotKwargs = {
                    "label": ["Truth MC", "Unfolded Signal MC", "Unfolded Data"],
                    "colour": ["r", "b", "k"],
                    "histstyle": ["step", "step", "errorbar"],
                    "xlabel": variable_data[var]["name"] + " [GeV]",
                    "kind": "overlay",
                    "do_stat": True,
                    "do_syst": False,
                    "title": smart_join(
                        "Variable-specific shadow-bin unfolding",
                        config.label,
                        "Medium Tau ID",
                        r"$\sqrt{s} = 13$TeV",
                        sep=" | ",
                    ),
                    "scale_by_bin_width": True,
                    "ylabel": (
                        r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                        + SYMBOLS[var]
                        + r"}$ [fb / GeV]"
                    ),
                    "logx": True,
                    "ratio_plot": True,
                    "ratio_label": "Data / MC",
                    "ratio_axlim": (0.5, 1.5),
                    "label_params": {"llabel": "", "loc": 1},
                }
                plotter.plot(
                    [truth, signal_unfolded, data_unfolded],
                    **plot_args,
                    filename=f"{config.label}_{var}_{iter_count}iter_unfolded.png",
                )
                plotter.plot(
                    [truth, signal_unfolded, data_unfolded],
                    **plot_args,
                    logy=True,
                    filename=f"{config.label}_{var}_{iter_count}iter_unfolded_logy.png",
                )
                plotter.plot_2d(
                    data_cov,
                    ylabel=f"Bin({variable_data[var]['name']})",
                    xlabel=f"Bin({variable_data[var]['name']})",
                    title=smart_join(config.label, var, "data covariance", sep=" | "),
                    labels=True,
                    label_params={"llabel": ""},
                    filename=f"{config.label}_{var}_{iter_count}iter_covariance.png",
                )

            plotter.paths.plot_dir = (
                plotter.paths.output_dir / "plots" / config.label / var / "compare"
            )
            plotter.plot(
                [truth] + [unfolded_by_iteration[i] for i in ITERATIONS],
                label=["Truth"]
                + [
                    "Bin-by-bin unfolding"
                    if iter_count == 0
                    else f"Bayesian unfolding, {iter_count} iterations"
                    for iter_count in ITERATIONS
                ],
                xlabel=variable_data[var]["name"] + " [GeV]",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                title=smart_join(config.label, var, "Medium Tau ID", sep=" | "),
                scale_by_bin_width=True,
                ylabel=(
                    r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                    + SYMBOLS[var]
                    + r"}$ [fb / GeV]"
                ),
                logx=True,
                label_params={"llabel": "", "loc": 1},
                filename=f"{config.label}_{var}_iteration_compare.png",
            )

    summary_lines = [
        "# Variable-specific shadow-bin unfolding closure summary",
        "",
        f"DO_FULL_SYSTEMATICS: `{DO_FULL_SYSTEMATICS}`",
        "",
        "| Configuration | Variable | Iterations | Mean deviation | Max deviation | Integral ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config_label, var, iter_count, mean_dev, max_dev, integral_ratio in closure_rows:
        summary_lines.append(
            f"| {config_label} | {var} | {iter_count} | {mean_dev:.3f} | "
            f"{max_dev:.3f} | {integral_ratio:.3f} |"
        )
    summary_path = output_root / "closure_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    plotter.logger.info("Saved closure summary to %s", summary_path)
    plotter.logger.info("DONE.")
