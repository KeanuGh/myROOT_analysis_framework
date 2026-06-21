import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples, signal_sample

from src.analysis import Analysis
from src.cutting import Cut
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from utils import ROOT_utils
from utils.helper_functions import smart_join
from utils.plotting_tools import Hist2dOpts, PlotKwargs
from utils.ROOT_utils import get_th1_bin_edges, sum_th1s
from utils.variable_names import variable_data

YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
WP = "medium"
VARS = ("MTW", "TauPt")
ITERATIONS = (0, 1, 2, 4, 8)
FAKES_SOURCE = "TauPt"
DO_FULL_SYSTEMATICS = False
LOAD_SAVED_HISTS = False

# Keep all existing shadow thresholds. `None` is the no-shadow closure control.
SHADOW_THRESHOLDS = (None, 200, 250, 300)

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
    """One reconstructed/truth phase-space definition for the unfolding test."""

    label: str
    mtw_min: float
    taupt_min: float
    met_min: float
    unfolded_var: str | None = None


@dataclass(frozen=True)
class ResponseComponents:
    """Response pieces needed by RooUnfold and by closure diagnostics."""

    response: ROOT.RooUnfoldResponse
    reco: ROOT.TH1
    truth: ROOT.TH1
    matrix: ROOT.TH2


def shadow_config(threshold: int | None, unfolded_var: str | None = None) -> ShadowConfig:
    """Return a no-shadow config or a variable-specific shadow-bin config."""
    if threshold is None:
        return ShadowConfig("no_shadow_bin", mtw_min=350, taupt_min=170, met_min=170)

    if unfolded_var == "MTW":
        return ShadowConfig(
            f"{unfolded_var}_shadow_bin_{threshold}",
            mtw_min=threshold,
            taupt_min=170,
            met_min=170,
            unfolded_var=unfolded_var,
        )

    if unfolded_var == "TauPt":
        return ShadowConfig(
            f"{unfolded_var}_shadow_bin_{threshold}",
            mtw_min=350,
            taupt_min=threshold / 2,
            met_min=170,
            unfolded_var=unfolded_var,
        )

    raise ValueError(f"Unknown shadow-bin unfolded variable: {unfolded_var}")


def analysis_configs() -> tuple[ShadowConfig, ...]:
    """Build the no-shadow control plus variable-specific shadow configurations."""
    return ShadowConfig(
        "no_shadow_bin",
        mtw_min=350,
        taupt_min=170,
        met_min=170,
    ), *(
        shadow_config(threshold, unfolded_var)
        for unfolded_var in VARS
        for threshold in SHADOW_THRESHOLDS
        if threshold is not None
    )


def variables_for_config(config: ShadowConfig) -> tuple[str, ...]:
    """Return the variables that should be unfolded for one configuration."""
    return VARS if config.unfolded_var is None else (config.unfolded_var,)


def measured_selection_name(config: ShadowConfig, region: str) -> str:
    """Selection name for one measured-input SR/CR and ID region."""
    return f"{config.label}_{WP}_{region}"


def true_tau_selection_name(config: ShadowConfig, region: str) -> str:
    """Truth-matched selection name used for prompt-tau subtraction in fake factors."""
    return f"trueTau_{measured_selection_name(config, region)}"


def truth_selection_name(config: ShadowConfig) -> str:
    """Truth-only response selection name for one threshold configuration."""
    return f"{config.label}_truth_tau"


def reco_selection_name(config: ShadowConfig) -> str:
    """Reco-only response selection name for one threshold configuration."""
    return f"{config.label}_{WP}_reco_tau"


def truth_reco_selection_name(config: ShadowConfig) -> str:
    """Matched truth-and-reco response selection name for one threshold configuration."""
    return f"{config.label}_{WP}_truth_reco_tau"


def variable_binnings(config: ShadowConfig) -> dict[str, np.ndarray]:
    """Build the binning dictionary for one threshold configuration."""
    if config.label == "no_shadow_bin":
        return BINNINGS

    binnings = dict(BINNINGS)
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
        binnings["MTW"] = mtw_bins
        binnings["TruthMTW"] = mtw_bins
    elif config.unfolded_var == "TauPt":
        taupt_bins = np.array(
            [config.taupt_min, 170, 200, 250, 300, 350, 425, 500, 600, 1000],
            dtype="double",
        )
        binnings["TauPt"] = taupt_bins
        binnings["VisTruthTauPt"] = taupt_bins
    else:
        raise ValueError(f"Unknown shadow-bin unfolded variable: {config.unfolded_var}")

    return binnings


def combined_binnings(configs: tuple[ShadowConfig, ...]) -> dict[str, dict[str, np.ndarray]]:
    """Map each threshold-specific selection to its matching bin edges."""
    binnings = {"": BINNINGS}
    for config in configs:
        config_binnings = variable_binnings(config)
        selection_names = [
            measured_selection_name(config, "SR_passID"),
            measured_selection_name(config, "SR_failID"),
            measured_selection_name(config, "CR_passID"),
            measured_selection_name(config, "CR_failID"),
            true_tau_selection_name(config, "SR_passID"),
            true_tau_selection_name(config, "SR_failID"),
            true_tau_selection_name(config, "CR_passID"),
            true_tau_selection_name(config, "CR_failID"),
            truth_selection_name(config),
            reco_selection_name(config),
            truth_reco_selection_name(config),
        ]
        for selection in selection_names:
            binnings[rf"^{re.escape(selection)}$"] = config_binnings
    return binnings


def config_label_regex(configs: tuple[ShadowConfig, ...]) -> str:
    """Regex alternation for all configured threshold labels."""
    return "|".join(re.escape(config.label) for config in configs)


def pass_reco_preselection() -> Cut:
    """Reco preselection shared by the fakes and response inputs."""
    return Cut(
        r"Pass preselection",
        r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) "
        r"&& passMetTrigger && (badJet == 0)"
        r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + "
        r"MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
        r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
    )


def reco_phase_space_cuts(config: ShadowConfig) -> list[Cut]:
    """Reco-side selection defining one shadow-region measured input."""
    return [
        pass_reco_preselection(),
        Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
        PASS_ETA,
        Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
        Cut(r"$E_T^{\mathrm{miss}}$ threshold", f"MET_met > {config.met_min:g}"),
    ]


def truth_phase_space_cuts(config: ShadowConfig) -> list[Cut]:
    """Truth-side fiducial selection matching the response binning."""
    return [
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


def measured_selections(config: ShadowConfig) -> tuple[dict[str, list[Cut]], dict[str, list[Cut]]]:
    """Selections for data/MC measured inputs and the fake-factor estimate."""
    sr_base = reco_phase_space_cuts(config)
    # Keep the CR disjoint from the shadow SR by placing it below the SR MET threshold.
    cr_base = [
        pass_reco_preselection(),
        Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
        PASS_ETA,
        Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
        Cut(r"$E_T^{\mathrm{miss}}$ control region", f"MET_met < {config.met_min:g}"),
    ]
    selections = {
        measured_selection_name(config, "SR_passID"): sr_base + [PASS_MEDIUM],
        measured_selection_name(config, "SR_failID"): sr_base + [FAIL_MEDIUM],
        measured_selection_name(config, "CR_passID"): cr_base + [PASS_MEDIUM],
        measured_selection_name(config, "CR_failID"): cr_base + [FAIL_MEDIUM],
    }
    mc_selections = dict(selections)
    for region in ("SR_passID", "SR_failID", "CR_passID", "CR_failID"):
        selection_name = measured_selection_name(config, region)
        mc_selections[true_tau_selection_name(config, region)] = (
            selections[selection_name] + [PASS_TRUETAU]
        )
    return selections, mc_selections


def response_selections(config: ShadowConfig) -> dict[str, list[Cut]]:
    """Signal-only selections used to build truth, reco, and migration histograms."""
    truth_cuts = truth_phase_space_cuts(config)
    reco_cuts = reco_phase_space_cuts(config)
    return {
        truth_selection_name(config): truth_cuts,
        reco_selection_name(config): reco_cuts + [PASS_MEDIUM],
        truth_reco_selection_name(config): truth_cuts + reco_cuts + [PASS_MEDIUM],
    }


def same_bin_edges(left: ROOT.TH1, right: ROOT.TH1) -> bool:
    """Check whether two ROOT histograms have the same explicit bin edges."""
    left_edges = get_th1_bin_edges(left)
    right_edges = get_th1_bin_edges(right)
    if len(left_edges) != len(right_edges):
        return False
    return all(
        abs(float(left_edge) - float(right_edge)) < 1e-6
        for left_edge, right_edge in zip(left_edges, right_edges, strict=True)
    )


def has_leading_shadow_bin(source: ROOT.TH1, target: ROOT.TH1) -> bool:
    """Check whether target equals source with one extra leading shadow bin."""
    source_edges = get_th1_bin_edges(source)
    target_edges = get_th1_bin_edges(target)
    if len(target_edges) != len(source_edges) + 1:
        return False
    return all(
        abs(float(source_edge) - float(target_edge)) < 1e-6
        for source_edge, target_edge in zip(source_edges, target_edges[1:], strict=True)
    )


def crop_shadow_bin_from_histogram(source: ROOT.TH1, target: ROOT.TH1, name: str) -> ROOT.TH1:
    """Clone source into target binning by dropping a leading shadow bin when needed."""
    if same_bin_edges(source, target):
        clone = source.Clone(name)
        clone.SetDirectory(0)
        return clone

    if not has_leading_shadow_bin(target, source):
        raise ValueError(
            f"Cannot crop histogram '{source.GetName()}' to '{target.GetName()}': "
            "the source binning is not identical and does not contain one leading shadow bin."
        )

    target_edges = get_th1_bin_edges(target)
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


def unfold_label(iter_count: int) -> str:
    """User-facing label for one unfolding configuration."""
    if iter_count == 0:
        return "Bin-by-bin unfolding"
    return f"Bayesian unfolding, {iter_count} iterations"


def all_measured_selections(
    configs: tuple[ShadowConfig, ...],
) -> tuple[dict[str, list[Cut]], dict[str, list[Cut]]]:
    """Combine the measured-input selections for all thresholds into one run."""
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    for config in configs:
        config_data_selections, config_mc_selections = measured_selections(config)
        data_selections.update(config_data_selections)
        mc_selections.update(config_mc_selections)
    return data_selections, mc_selections


def all_response_selections(configs: tuple[ShadowConfig, ...]) -> dict[str, list[Cut]]:
    """Combine the response selections for all thresholds into one run."""
    selections: dict[str, list[Cut]] = {}
    for config in configs:
        selections.update(response_selections(config))
    return selections


def build_measured_analysis(configs: tuple[ShadowConfig, ...], output_root: Path) -> Analysis:
    """Regenerate data, background, and fake-estimate input histograms once."""
    data_selections, mc_selections = all_measured_selections(configs)
    datasets = analysis_samples(mc_selections, data_selections=data_selections, snapshot=False)
    label_regex = config_label_regex(configs)
    return Analysis(
        datasets,
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
        binnings=combined_binnings(configs),
    )


def build_response_analysis(configs: tuple[ShadowConfig, ...], output_root: Path) -> Analysis:
    """Regenerate signal-only response inputs for all thresholds once."""
    selections = all_response_selections(configs)
    label_regex = config_label_regex(configs)
    hists_2d = {
        "MTW_TruthMTW": Hist2dOpts("MTW", "TruthMTW", "reco_weight"),
        "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt", "reco_weight"),
    }
    return Analysis(
        {"wtaunu_had": signal_sample(selections=selections)},
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
        hists_2d=hists_2d,
        do_unweighted=True,
        systematics_for_selection={rf"^({label_regex})_{WP}_(reco_tau|truth_reco_tau)$"}
        if DO_FULL_SYSTEMATICS
        else set(),
        skip_sys=SKIP_SYS,
        binnings=combined_binnings(configs),
    )


def response_components(
    analysis: Analysis,
    config: ShadowConfig,
    var: str,
    systematic: str = NOMINAL_NAME,
) -> ResponseComponents:
    """Load the response components from a signal-only response analysis."""
    reco = analysis.get_hist(
        var,
        dataset="wtaunu_had",
        systematic=systematic,
        selection=reco_selection_name(config),
    )
    truth = analysis.get_hist(
        TRUTHS[var],
        dataset="wtaunu_had",
        systematic=NOMINAL_NAME if systematic != NOMINAL_NAME else systematic,
        selection=truth_selection_name(config),
    )
    matrix = analysis.get_hist(
        f"{var}_{TRUTHS[var]}",
        dataset="wtaunu_had",
        systematic=systematic,
        selection=truth_reco_selection_name(config),
    )
    response = ROOT.RooUnfoldResponse(reco, truth, matrix)
    return ResponseComponents(response=response, reco=reco, truth=truth, matrix=matrix)


def nominal_truth_references(
    response_analysis: Analysis,
    no_shadow_config: ShadowConfig,
) -> dict[str, ROOT.TH1]:
    """Extract nominal fiducial truth histograms used for cropping and closure."""
    return {
        var: response_analysis.get_hist(
            TRUTHS[var],
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection=truth_selection_name(no_shadow_config),
        )
        for var in VARS
    }


def calculate_fake_estimates(analysis: Analysis, config: ShadowConfig) -> None:
    """Calculate the TauPt-sourced fake estimate once for one threshold."""
    if not analysis.data_sample:
        raise ValueError("Measured analysis has no data sample.")

    name = f"{config.label}_medium"
    analysis.do_fakes_estimate(
        FAKES_SOURCE,
        variables_for_config(config),
        measured_selection_name(config, "CR_passID"),
        measured_selection_name(config, "CR_failID"),
        measured_selection_name(config, "SR_passID"),
        measured_selection_name(config, "SR_failID"),
        true_tau_selection_name(config, "CR_passID"),
        true_tau_selection_name(config, "CR_failID"),
        true_tau_selection_name(config, "SR_passID"),
        true_tau_selection_name(config, "SR_failID"),
        name=name,
        systematic=NOMINAL_NAME,
        save_intermediates=True,
    )


def measured_inputs(
    analysis: Analysis, config: ShadowConfig, var: str
) -> tuple[ROOT.TH1, ROOT.TH1]:
    """Return background-subtracted data and signal MC in one shadow-region selection."""
    if not analysis.data_sample:
        raise ValueError("Measured analysis has no data sample.")

    name = f"{config.label}_medium"
    signal_selection = measured_selection_name(config, "SR_passID")
    data = analysis.get_hist(
        var,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=signal_selection,
    )
    signal = analysis.get_hist(
        var,
        dataset="wtaunu_had",
        systematic=NOMINAL_NAME,
        selection=signal_selection,
    )
    backgrounds = [
        analysis.get_hist(
            var,
            dataset=background,
            systematic=NOMINAL_NAME,
            selection=signal_selection,
        )
        for background in analysis.mc_samples
        if background != "wtaunu_had"
    ]
    fakes = analysis.histograms[f"{name}_{var}_fakes_bkg_{FAKES_SOURCE}_src"]
    background = sum_th1s(*(backgrounds + [fakes]))
    data_sig = data - background
    data_sig.SetName(f"{config.label}_{var}_data_minus_background")
    signal = signal.Clone(f"{config.label}_{var}_signal")
    signal.SetDirectory(0)
    return data_sig, signal


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
    return crop_shadow_bin_from_histogram(scaled, nominal_truth, name)


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


def plot_response_systematics(
    plotter: Analysis,
    response_analysis: Analysis,
    config: ShadowConfig,
    var: str,
    nominal_truth: ROOT.TH1,
) -> None:
    """Plot response-only systematic uncertainty diagnostics when requested."""
    if not DO_FULL_SYSTEMATICS:
        return

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
        raise RuntimeError("DO_FULL_SYSTEMATICS is enabled but no response systematics exist.")

    hists = []
    labels = []
    nominal_response = response_components(response_analysis, config, var)
    nominal_signal = response_analysis.get_hist(
        var,
        dataset="wtaunu_had",
        systematic=NOMINAL_NAME,
        selection=reco_selection_name(config),
    )
    nominal_unfolded, _ = unfold_histogram(plotter, nominal_signal, nominal_response, 0)
    nominal_unfolded = scale_and_crop_unfolded(
        nominal_unfolded,
        nominal_truth,
        f"{config.label}_{var}_nominal_response_sys_reference",
    )

    for sys_name in sys_names:
        up = f"{sys_name}__1up"
        down = f"{sys_name}__1down"
        try:
            response_up = response_components(response_analysis, config, var, up)
            response_down = response_components(response_analysis, config, var, down)
        except KeyError as exc:
            raise KeyError(
                f"Missing full shadow-region response systematic for {config.label} {var}: "
                f"{sys_name}"
            ) from exc

        unfolded_up, _ = unfold_histogram(plotter, nominal_signal, response_up, 0)
        unfolded_down, _ = unfold_histogram(plotter, nominal_signal, response_down, 0)
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
        relative_uncertainty = ROOT_utils.th1_relative_uncertainty(
            uncertainty,
            nominal_unfolded,
            name=f"{config.label}_{var}_{sys_name}_response_uncertainty",
        )
        hists.append(relative_uncertainty)
        labels.append(sys_name)

    plotter.paths.plot_dir = (
        plotter.paths.output_dir / "plots" / config.label / var / "systematics"
    )
    plotter.plot(
        hists,
        label=labels,
        xlabel=variable_data[var]["name"] + " [GeV]",
        ylabel="Response uncertainty / %",
        title=smart_join(config.label, var, "medium ID", sep=" | "),
        do_stat=False,
        do_syst=False,
        logx=True,
        filename=f"{config.label}_{var}_response_systematics.png",
    )


def run_config(
    plotter: Analysis,
    config: ShadowConfig,
    measured: Analysis,
    response_analysis: Analysis,
    nominal_truth_hists: dict[str, ROOT.TH1],
) -> list[tuple[str, str, int, float, float, float]]:
    """Run fake estimates, unfolding, and plots for one configuration."""
    plotter.logger.info("Running full shadow-bin closure for %s", config.label)
    calculate_fake_estimates(measured, config)
    if not DO_FULL_SYSTEMATICS:
        plotter.logger.info(
            "DO_FULL_SYSTEMATICS is False: producing central-value closure only for %s.",
            config.label,
        )

    results: list[tuple[str, str, int, float, float, float]] = []

    for var in variables_for_config(config):
        data_sig, signal = measured_inputs(measured, config, var)
        response = response_components(response_analysis, config, var)
        nominal_truth = nominal_truth_hists[var]
        truth = Histogram1D(th1=nominal_truth) / LUMI
        plot_response_systematics(plotter, response_analysis, config, var, nominal_truth)

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

        by_iteration = {}
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
            by_iteration[iter_count] = signal_unfolded
            mean_dev, max_dev, integral_ratio = closure_metrics(signal_unfolded, truth.TH1)
            results.append((config.label, var, iter_count, mean_dev, max_dev, integral_ratio))

            plotter.paths.plot_dir = plotter.paths.output_dir / "plots" / config.label / var
            default_args: PlotKwargs = {
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
                **default_args,
                filename=f"{config.label}_{var}_{iter_count}iter_unfolded.png",
            )
            plotter.plot(
                [truth, signal_unfolded, data_unfolded],
                **default_args,
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
            [truth] + [by_iteration[i] for i in ITERATIONS],
            label=["Truth"] + [unfold_label(i) for i in ITERATIONS],
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

    return results


def write_closure_summary(
    output_root: Path,
    rows: list[tuple[str, str, int, float, float, float]],
) -> None:
    """Write a small markdown summary of the signal-MC closure metrics."""
    lines = [
        "# Variable-specific shadow-bin unfolding closure summary",
        "",
        f"DO_FULL_SYSTEMATICS: `{DO_FULL_SYSTEMATICS}`",
        "",
        "| Configuration | Variable | Iterations | Mean deviation | Max deviation | Integral ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config, var, iter_count, mean_dev, max_dev, integral_ratio in rows:
        lines.append(
            f"| {config} | {var} | {iter_count} | {mean_dev:.3f} | "
            f"{max_dev:.3f} | {integral_ratio:.3f} |"
        )
    summary_path = output_root / "closure_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")


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

    configs = analysis_configs()
    no_shadow_config = shadow_config(None)
    measured_analysis = build_measured_analysis(configs, output_root)
    measured_analysis.print_metadata_table(datasets=measured_analysis.mc_samples)
    response_analysis = build_response_analysis(configs, output_root)
    nominal_truth_hists = nominal_truth_references(response_analysis, no_shadow_config)
    closure_rows = []
    for config in configs:
        closure_rows.extend(
            run_config(
                plotter,
                config,
                measured_analysis,
                response_analysis,
                nominal_truth_hists,
            )
        )

    write_closure_summary(output_root, closure_rows)
    plotter.logger.info("Saved closure summary to %s", output_root / "closure_summary.md")
    plotter.logger.info("DONE.")
