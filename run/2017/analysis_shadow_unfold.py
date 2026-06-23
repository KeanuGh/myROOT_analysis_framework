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
VARS = (
    "MTW",
    # "TauPt",
)
SHADOW_BINS = {
    "MTW": (
        200,
        250,
        300,
    ),
    "TauPt": (
        200,
        250,
        300,
    ),
}
ITERATIONS = (
    0,
    1,
    2,
    # 4,
    # 8,
)
FAKES_SOURCE = "TauPt"
LOAD_SAVED_HISTS = False  # Reuse saved ROOT histograms instead of rebuilding them.
RUN_FAKE_DIAGNOSTICS = True  # Master switch for fake-estimate validation outputs.
BUILD_INCLUSIVE_FAKES = False  # Build inclusive fakes; nominal only if prong-split is disabled.
USE_PRONG_SPLIT_FAKES = True  # Build and use the thesis prong-split fake estimate as nominal.
RUN_FAKE_WIDTH_SYSTEMATIC = True  # Propagate the 1-prong tau-width fake-source uncertainty.
RUN_WTAUNU_PRONG_SYSTEMATIC = True  # Test a prong-composition modelling shift for wtaunu_had.
DO_SPLIT_SAMPLE_CLOSURE = False  # Build even/odd MC closure samples for response validation.
DO_FULL_SYSTEMATICS = False  # Enable full systematic response variations; slow final-mode run.
FAKE_SCALE_SCAN = (0.0, 0.5, 1.0)  # Fake normalisation scales used in the diagnostic scan.
FAKE_DIAGNOSTIC_ITERATION = 1  # Bayesian iteration count used for fake-scale scan plots.
FAKE_WIDTH_SYSTEMATIC_VARS = ("MTW",)  # Keep this focused on the measured observable.
FAKE_WIDTH_VARIABLES = ("TauTrackWidthPt1000PV", "TauTrackWidthPt500PV")
FAKE_WIDTH_MAX_WEIGHT = 5.0
FAKE_WIDTH_BINS = np.array(
    [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25, 0.35],
    dtype="double",
)
WTAUNU_PRONG_SYSTEMATIC_VARS = ("MTW",)  # Keep the first test on the measured observable.
WTAUNU_PRONG_VARIATIONS = {
    "reduce_3prong_20pct": {1: 1.0, 3: 0.8},
}

if not BUILD_INCLUSIVE_FAKES and not USE_PRONG_SPLIT_FAKES:
    raise ValueError("Enable BUILD_INCLUSIVE_FAKES, USE_PRONG_SPLIT_FAKES, or both.")

SKIP_SYS = {
    r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
    r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
}

# CUTS & SELECTIONS
# ========================================================================

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
RESPONSE_SPLIT = Cut(r"Response split", r"(eventNumber % 2) == 0")
PSEUDO_DATA_SPLIT = Cut(r"Pseudo-data split", r"(eventNumber % 2) == 1")


# FAKE CONTROL REGIONS
# ========================================================================
@dataclass(frozen=True)
class FakeControlRegion:
    """Reco-region used to derive fake factors before applying them in the SR."""

    selection_tag: str
    output_tag: str
    cuts: tuple[Cut, ...]


# Original thesis-style fake control region. Uncomment this block and comment
# out the low-MET block below to compare against the current baseline.
# FAKE_CONTROL_REGION = FakeControlRegion(
#     selection_tag="CR",
#     output_tag="",
#     cuts=(
#         PASS_RECO_PRESELECTION,
#         Cut(r"$p_T^\tau$ threshold", "TauPt > {taupt_min:g}"),
#         PASS_ETA,
#         Cut(r"$m_T^W$ threshold", "MTW > {mtw_min:g}"),
#         Cut(r"$E_T^{\mathrm{miss}}$ control region", "MET_met < {met_min:g}"),
#     ),
# )
FAKE_CONTROL_REGION = FakeControlRegion(
    selection_tag="lowMET_CR",
    output_tag="_lowMET",
    cuts=(
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        Cut(r"Low-$E_T^{\mathrm{miss}}$ fake-enriched region", "MET_met < 100"),
    ),
)


# MODELS & CONFIGURATION
# ========================================================================
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
    *(
        ShadowConfig(
            f"MTW_shadow_bin_{threshold}",
            "MTW",
            mtw_min=threshold,
            taupt_min=170,
            met_min=170,
        )
        for threshold in SHADOW_BINS["MTW"]
    ),
    *(
        ShadowConfig(
            f"TauPt_shadow_bin_{threshold}",
            "TauPt",
            mtw_min=350,
            taupt_min=threshold / 2,
            met_min=170,
        )
        for threshold in SHADOW_BINS["TauPt"]
    ),
)


# HISTOGRAM HELPERS
# ========================================================================
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


def fake_like_histogram(
    analysis: Analysis,
    variable: str,
    data_selection: str,
    mc_selection: str,
    name: str,
) -> ROOT.TH1:
    """Return data minus nonfake MC for one variable and selection pair."""
    data_hist = analysis.get_hist(
        variable=variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=data_selection,
        allow_generation=True,
    )
    nonfake_hist = sum_th1s(
        *[
            analysis.get_hist(
                variable=variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=mc_selection,
                allow_generation=True,
            )
            for mc_sample in analysis.mc_samples
        ]
    )
    fake_like = data_hist - nonfake_hist
    fake_like.SetName(name)
    fake_like.SetDirectory(0)
    return fake_like


def positive_unit_shape(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    """Return a positive, unit-normalised shape for composition reweighting."""
    shape = hist.Clone(name)
    shape.SetDirectory(0)
    for bin_idx in range(1, shape.GetNbinsX() + 1):
        if shape.GetBinContent(bin_idx) < 0:
            shape.SetBinContent(bin_idx, 0.0)
            shape.SetBinError(bin_idx, 0.0)
    integral = shape.Integral()
    if integral > 0:
        shape.Scale(1.0 / integral)
    return shape


def make_width_ratio(numerator: ROOT.TH1, denominator: ROOT.TH1, name: str) -> ROOT.TH1:
    """Build a capped width-shape ratio histogram."""
    ratio_hist = numerator.Clone(name)
    ratio_hist.SetDirectory(0)
    for bin_idx in range(1, ratio_hist.GetNbinsX() + 1):
        den = denominator.GetBinContent(bin_idx)
        if den <= 0:
            ratio_hist.SetBinContent(bin_idx, 0.0)
        else:
            ratio_hist.SetBinContent(
                bin_idx,
                min(FAKE_WIDTH_MAX_WEIGHT, numerator.GetBinContent(bin_idx) / den),
            )
        ratio_hist.SetBinError(bin_idx, 0.0)
    return ratio_hist


def fill_width_shifted_fake_prediction(
    analysis: Analysis,
    *,
    target_var: str,
    width_var: str,
    sr_pass_selection: str,
    sr_fail_selection: str,
    true_sr_fail_selection: str,
    ff_hist: ROOT.TH1,
    width_ratio: ROOT.TH1,
    output_name: str,
) -> ROOT.TH1:
    """Fill a fake prediction with the nominal FF and an extra width-composition weight."""
    ROOT.gInterpreter.Declare(
        f"TH1* FF_hist_{output_name} = reinterpret_cast<TH1*>({ROOT.addressof(ff_hist)});"
    )
    ROOT.gInterpreter.Declare(
        f"TH1* width_ratio_{output_name} = "
        f"reinterpret_cast<TH1*>({ROOT.addressof(width_ratio)});"
    )
    weight_col = f"FF_width_weight_{output_name}"
    weight_expr = (
        f"reco_weight"
        f" * FF_hist_{output_name}->GetBinContent("
        f"FF_hist_{output_name}->FindBin({FAKES_SOURCE}))"
        f" * width_ratio_{output_name}->GetBinContent("
        f"width_ratio_{output_name}->FindBin({width_var}))"
    )
    h_bins = analysis[analysis.data_sample].get_binnings(target_var, sr_pass_selection)

    data_hist = ROOT.TH1F(
        f"{output_name}_data",
        output_name,
        *ROOT_utils.get_TH1_bin_args(**h_bins),
    )
    data_ptr = (
        analysis[analysis.data_sample]
        .filters[NOMINAL_NAME][sr_fail_selection]
        .df.Define(weight_col, weight_expr)
        .Fill(data_hist, [target_var, weight_col])
    )

    mc_ptrs = []
    for mc_sample in analysis.mc_samples:
        mc_hist = ROOT.TH1F(
            f"{output_name}_{mc_sample}",
            output_name,
            *ROOT_utils.get_TH1_bin_args(**h_bins),
        )
        mc_ptrs.append(
            analysis[mc_sample]
            .filters[NOMINAL_NAME][true_sr_fail_selection]
            .df.Define(weight_col, weight_expr)
            .Fill(mc_hist, [target_var, weight_col])
        )

    shifted_prediction = data_ptr.GetValue() - sum_th1s(*[ptr.GetValue() for ptr in mc_ptrs])
    shifted_prediction.SetName(output_name)
    shifted_prediction.SetTitle(output_name)
    shifted_prediction.SetDirectory(0)
    for bin_idx in range(1, shifted_prediction.GetNbinsX() + 1):
        shifted_prediction.SetBinError(
            bin_idx,
            abs(shifted_prediction.GetBinContent(bin_idx)) * 0.1,
        )
    analysis.histograms[output_name] = shifted_prediction
    return shifted_prediction


if __name__ == "__main__":
    # SETUP
    # ========================================================================
    output_label = Path(__file__).stem
    output_root = Path(__file__).absolute().parent.parent.parent / "outputs" / output_label
    plotter = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label=output_label,
        output_dir=output_root,
        log_level=10,
        log_out="both",
    )
    plotter.logger.info("Starting analysis_shadow_unfold.py")
    plotter.logger.info("RUN_FAKE_DIAGNOSTICS = %s", RUN_FAKE_DIAGNOSTICS)
    plotter.logger.info(
        "FAKE_CONTROL_REGION selection tag = %s", FAKE_CONTROL_REGION.selection_tag
    )
    plotter.logger.info("BUILD_INCLUSIVE_FAKES = %s", BUILD_INCLUSIVE_FAKES)
    plotter.logger.info("USE_PRONG_SPLIT_FAKES = %s", USE_PRONG_SPLIT_FAKES)
    plotter.logger.info("RUN_FAKE_WIDTH_SYSTEMATIC = %s", RUN_FAKE_WIDTH_SYSTEMATIC)
    plotter.logger.info("RUN_WTAUNU_PRONG_SYSTEMATIC = %s", RUN_WTAUNU_PRONG_SYSTEMATIC)
    plotter.logger.info("DO_SPLIT_SAMPLE_CLOSURE = %s", DO_SPLIT_SAMPLE_CLOSURE)
    plotter.logger.info("DO_FULL_SYSTEMATICS = %s", DO_FULL_SYSTEMATICS)
    if DO_FULL_SYSTEMATICS:
        plotter.logger.info(
            "Full systematics mode enabled; missing shadow variations will fail loudly."
        )

    # SELECTION BUILDING
    # ========================================================================
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    response_selections: dict[str, list[Cut]] = {}
    split_response_selections: dict[str, list[Cut]] = {}
    split_pseudo_data_selections: dict[str, list[Cut]] = {}
    selection_binnings: dict[str, dict[str, np.ndarray]] = {
        "": {**BINNINGS, **{width_var: FAKE_WIDTH_BINS for width_var in FAKE_WIDTH_VARIABLES}}
    }

    for config in CONFIGS:
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        fake_cr_pass = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_passID"
        fake_cr_fail = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_failID"
        true_sr_pass = f"trueTau_{sr_pass}"
        true_sr_fail = f"trueTau_{sr_fail}"
        true_fake_cr_pass = f"trueTau_{fake_cr_pass}"
        true_fake_cr_fail = f"trueTau_{fake_cr_fail}"
        prong_names = {
            prong: {
                "sr_pass": f"{config.label}_{WP}_{prong}prong_SR_passID",
                "sr_fail": f"{config.label}_{WP}_{prong}prong_SR_failID",
                "fake_cr_pass": (
                    f"{config.label}_{WP}_{prong}prong_{FAKE_CONTROL_REGION.selection_tag}_passID"
                ),
                "fake_cr_fail": (
                    f"{config.label}_{WP}_{prong}prong_{FAKE_CONTROL_REGION.selection_tag}_failID"
                ),
            }
            for prong in (1, 3)
        }
        truth_selection = f"{config.label}_truth_tau"
        reco_selection = f"{config.label}_{WP}_reco_tau"
        truth_reco_selection = f"{config.label}_{WP}_truth_reco_tau"
        response_prong_names = {
            prong: {
                "reco": f"{config.label}_{WP}_{prong}prong_reco_tau",
                "truth_reco": f"{config.label}_{WP}_{prong}prong_truth_reco_tau",
            }
            for prong in (1, 3)
        }

        reco_sr_cuts = [
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
            PASS_ETA,
            Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
            Cut(r"$E_T^{\mathrm{miss}}$ threshold", f"MET_met > {config.met_min:g}"),
        ]
        fake_cr_cuts = [
            Cut(
                cut.name,
                cut.cutstr.format(
                    taupt_min=config.taupt_min,
                    mtw_min=config.mtw_min,
                    met_min=config.met_min,
                ),
            )
            for cut in FAKE_CONTROL_REGION.cuts
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
        data_selections[fake_cr_pass] = fake_cr_cuts + [PASS_MEDIUM]
        data_selections[fake_cr_fail] = fake_cr_cuts + [FAIL_MEDIUM]
        plotter.logger.info("Fake control region cuts for %s:", config.label)
        for cut in fake_cr_cuts:
            plotter.logger.info("  %s: %s", cut.name, cut.cutstr)

        for prong, names in prong_names.items():
            pass_prong = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
            data_selections[names["sr_pass"]] = data_selections[sr_pass] + [pass_prong]
            data_selections[names["sr_fail"]] = data_selections[sr_fail] + [pass_prong]
            data_selections[names["fake_cr_pass"]] = data_selections[fake_cr_pass] + [pass_prong]
            data_selections[names["fake_cr_fail"]] = data_selections[fake_cr_fail] + [pass_prong]

        measured_selections = [sr_pass, sr_fail, fake_cr_pass, fake_cr_fail]
        for selection in measured_selections:
            mc_selections[selection] = data_selections[selection]
        for names in prong_names.values():
            prong_selections = [
                names["sr_pass"],
                names["sr_fail"],
                names["fake_cr_pass"],
                names["fake_cr_fail"],
            ]
            for selection in prong_selections:
                mc_selections[selection] = data_selections[selection]

        mc_selections[true_sr_pass] = data_selections[sr_pass] + [PASS_TRUETAU]
        mc_selections[true_sr_fail] = data_selections[sr_fail] + [PASS_TRUETAU]
        mc_selections[true_fake_cr_pass] = data_selections[fake_cr_pass] + [PASS_TRUETAU]
        mc_selections[true_fake_cr_fail] = data_selections[fake_cr_fail] + [PASS_TRUETAU]
        for names in prong_names.values():
            mc_selections[f"trueTau_{names['sr_pass']}"] = data_selections[
                names["sr_pass"]
            ] + [PASS_TRUETAU]
            mc_selections[f"trueTau_{names['sr_fail']}"] = data_selections[
                names["sr_fail"]
            ] + [PASS_TRUETAU]
            mc_selections[f"trueTau_{names['fake_cr_pass']}"] = data_selections[
                names["fake_cr_pass"]
            ] + [PASS_TRUETAU]
            mc_selections[f"trueTau_{names['fake_cr_fail']}"] = data_selections[
                names["fake_cr_fail"]
            ] + [PASS_TRUETAU]

        response_selections[truth_selection] = truth_cuts
        response_selections[reco_selection] = reco_sr_cuts + [PASS_MEDIUM]
        response_selections[truth_reco_selection] = truth_cuts + reco_sr_cuts + [PASS_MEDIUM]
        if RUN_WTAUNU_PRONG_SYSTEMATIC:
            for prong, names in response_prong_names.items():
                pass_prong = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
                response_selections[names["reco"]] = reco_sr_cuts + [PASS_MEDIUM, pass_prong]
                response_selections[names["truth_reco"]] = [
                    *truth_cuts,
                    *reco_sr_cuts,
                    PASS_MEDIUM,
                    pass_prong,
                ]
        split_response_selections[truth_selection] = [RESPONSE_SPLIT] + truth_cuts
        split_response_selections[reco_selection] = [
            RESPONSE_SPLIT,
            *reco_sr_cuts,
            PASS_MEDIUM,
        ]
        split_response_selections[truth_reco_selection] = [
            RESPONSE_SPLIT,
            *truth_cuts,
            *reco_sr_cuts,
            PASS_MEDIUM,
        ]
        split_pseudo_data_selections[truth_selection] = [PSEUDO_DATA_SPLIT] + truth_cuts
        split_pseudo_data_selections[reco_selection] = [
            PSEUDO_DATA_SPLIT,
            *reco_sr_cuts,
            PASS_MEDIUM,
        ]
        split_pseudo_data_selections[truth_reco_selection] = [
            PSEUDO_DATA_SPLIT,
            *truth_cuts,
            *reco_sr_cuts,
            PASS_MEDIUM,
        ]

        config_binnings = {
            **BINNINGS,
            **{width_var: FAKE_WIDTH_BINS for width_var in FAKE_WIDTH_VARIABLES},
        }
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
            fake_cr_pass,
            fake_cr_fail,
            true_sr_pass,
            true_sr_fail,
            true_fake_cr_pass,
            true_fake_cr_fail,
            truth_selection,
            reco_selection,
            truth_reco_selection,
        ):
            selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
        if RUN_WTAUNU_PRONG_SYSTEMATIC:
            for names in response_prong_names.values():
                selection_binnings[rf"^{re.escape(names['reco'])}$"] = config_binnings
                selection_binnings[rf"^{re.escape(names['truth_reco'])}$"] = config_binnings
        for names in prong_names.values():
            prong_selections = [
                names["sr_pass"],
                names["sr_fail"],
                names["fake_cr_pass"],
                names["fake_cr_fail"],
            ]
            for selection in prong_selections:
                selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
                selection_binnings[rf"^{re.escape(f'trueTau_{selection}')}$"] = config_binnings

    label_regex = "|".join(re.escape(config.label) for config in CONFIGS)
    truth_histogram_vars = {variable_data[var]["truth"] for var in VARS}

    # DATAFRAME & HISTOGRAM PRODUCTION
    # ========================================================================
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
            *FAKE_WIDTH_VARIABLES,
        },
        import_missing_columns_as_nan=True,
        histogram_vars={*VARS, *FAKE_WIDTH_VARIABLES},
        systematics_for_selection={rf"^({label_regex})_{WP}_SR_passID$"}
        if DO_FULL_SYSTEMATICS
        else set(),
        skip_sys=SKIP_SYS,
        binnings=selection_binnings,
    )
    load_measured_analysis_hists = LOAD_SAVED_HISTS and measured_analysis.load_hists_if_available()
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
            "eventNumber",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=set(VARS) | truth_histogram_vars,
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

    if DO_SPLIT_SAMPLE_CLOSURE:
        split_response_analysis = Analysis(
            {"wtaunu_had": signal_sample(selections=split_response_selections)},
            year=YEAR,
            rerun=not LOAD_SAVED_HISTS,
            regen_histograms=not LOAD_SAVED_HISTS,
            do_systematics=False,
            metadata_cache=DSID_METADATA_CACHE,
            ttree=NOMINAL_NAME,
            analysis_label="analysis_shadow_unfold_split_response",
            output_dir=output_root / "split_response",
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
                "MTW_TruthMTW": Hist2dOpts("MTW", "TruthMTW", "reco_weight"),
                "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt", "reco_weight"),
            },
            do_unweighted=True,
            binnings=selection_binnings,
        )
        split_pseudo_data_analysis = Analysis(
            {"wtaunu_had": signal_sample(selections=split_pseudo_data_selections)},
            year=YEAR,
            rerun=not LOAD_SAVED_HISTS,
            regen_histograms=not LOAD_SAVED_HISTS,
            do_systematics=False,
            metadata_cache=DSID_METADATA_CACHE,
            ttree=NOMINAL_NAME,
            analysis_label="analysis_shadow_unfold_split_pseudo_data",
            output_dir=output_root / "split_pseudo_data",
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
                "MTW_TruthMTW": Hist2dOpts("MTW", "TruthMTW", "reco_weight"),
                "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt", "reco_weight"),
            },
            do_unweighted=True,
            binnings=selection_binnings,
        )
    else:
        split_response_analysis = None
        split_pseudo_data_analysis = None

    nominal_truth_hists = {
        var: response_analysis.get_hist(
            variable_data[var]["truth"],
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection="no_shadow_bin_truth_tau",
        )
        for var in VARS
    }
    split_nominal_truth_hists = (
        {
            var: split_pseudo_data_analysis.get_hist(
                variable_data[var]["truth"],
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection="no_shadow_bin_truth_tau",
            )
            for var in VARS
        }
        if split_pseudo_data_analysis is not None
        else {}
    )

    closure_rows: list[tuple[str, str, int, float, float, float]] = []
    split_closure_rows: list[tuple[str, str, int, float, float, float]] = []
    fake_budget_rows: list[
        tuple[str, str, float, float, float, float, float, float, float, float]
    ] = []
    fake_scale_rows: list[tuple[str, str, float, float, float, float]] = []
    fake_mc_closure_rows: list[tuple[str, float, float, float, float, float]] = []
    fake_prong_rows: list[tuple[str, str, float, float, float, float]] = []
    fake_width_rows: list[
        tuple[str, str, str, float, float, float, float, float, float]
    ] = []
    wtaunu_prong_rows: list[
        tuple[str, str, str, float, float, float, float, float, float, float]
    ] = []

    # FAKE ESTIMATION, UNFOLDING & DIAGNOSTICS
    # ========================================================================
    for config in CONFIGS:
        # CURRENT SHADOW CONFIGURATION
        # --------------------------------------------------------------------
        vars_for_config = VARS if config.unfolded_var is None else (config.unfolded_var,)
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        fake_cr_pass = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_passID"
        fake_cr_fail = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_failID"
        true_sr_pass = f"trueTau_{sr_pass}"
        true_sr_fail = f"trueTau_{sr_fail}"
        true_fake_cr_pass = f"trueTau_{fake_cr_pass}"
        true_fake_cr_fail = f"trueTau_{fake_cr_fail}"
        truth_selection = f"{config.label}_truth_tau"
        reco_selection = f"{config.label}_{WP}_reco_tau"
        truth_reco_selection = f"{config.label}_{WP}_truth_reco_tau"
        response_prong_names = {
            prong: {
                "reco": f"{config.label}_{WP}_{prong}prong_reco_tau",
                "truth_reco": f"{config.label}_{WP}_{prong}prong_truth_reco_tau",
            }
            for prong in (1, 3)
        }
        fakes_name = f"{config.label}_{WP}{FAKE_CONTROL_REGION.output_tag}"

        plotter.logger.info("Running variable-specific shadow-bin closure for %s", config.label)
        plotter.logger.info(
            "Using fake control region '%s' for %s.",
            FAKE_CONTROL_REGION.selection_tag,
            config.label,
        )

        # DATA-DRIVEN FAKE ESTIMATES
        # --------------------------------------------------------------------
        if not load_measured_analysis_hists:
            if BUILD_INCLUSIVE_FAKES:
                # Inclusive fakes are nominal only when prong-split fakes are disabled.
                measured_analysis.do_fakes_estimate(
                    FAKES_SOURCE,
                    vars_for_config,
                    fake_cr_pass,
                    fake_cr_fail,
                    sr_pass,
                    sr_fail,
                    true_fake_cr_pass,
                    true_fake_cr_fail,
                    true_sr_pass,
                    true_sr_fail,
                    name=fakes_name,
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )

            if USE_PRONG_SPLIT_FAKES:
                # The nominal fake estimate follows the thesis method: split by tau prong
                # and then sum the 1-prong and 3-prong predictions.
                for prong in (1, 3):
                    prong_fakes_name = (
                        f"{config.label}_{WP}_{prong}prong{FAKE_CONTROL_REGION.output_tag}"
                    )
                    prong_fake_cr_pass = (
                        f"{config.label}_{WP}_{prong}prong_"
                        f"{FAKE_CONTROL_REGION.selection_tag}_passID"
                    )
                    prong_fake_cr_fail = (
                        f"{config.label}_{WP}_{prong}prong_"
                        f"{FAKE_CONTROL_REGION.selection_tag}_failID"
                    )

                    measured_analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        vars_for_config,
                        prong_fake_cr_pass,
                        prong_fake_cr_fail,
                        f"{config.label}_{WP}_{prong}prong_SR_passID",
                        f"{config.label}_{WP}_{prong}prong_SR_failID",
                        f"trueTau_{prong_fake_cr_pass}",
                        f"trueTau_{prong_fake_cr_fail}",
                        f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                        f"trueTau_{config.label}_{WP}_{prong}prong_SR_failID",
                        name=prong_fakes_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

        # MC FAKE-CLOSURE DIAGNOSTIC
        # --------------------------------------------------------------------
        if RUN_FAKE_DIAGNOSTICS:
            mc_fake_hists = {}
            for selection_name in (sr_pass, sr_fail, fake_cr_pass, fake_cr_fail):
                all_mc = measured_analysis.sum_hists(
                    [
                        measured_analysis.get_hist(
                            variable=FAKES_SOURCE,
                            dataset=mc_sample,
                            systematic=NOMINAL_NAME,
                            selection=selection_name,
                            allow_generation=True,
                        )
                        for mc_sample in measured_analysis.mc_samples
                    ]
                )
                true_mc = measured_analysis.sum_hists(
                    [
                        measured_analysis.get_hist(
                            variable=FAKES_SOURCE,
                            dataset=mc_sample,
                            systematic=NOMINAL_NAME,
                            selection=f"trueTau_{selection_name}",
                            allow_generation=True,
                        )
                        for mc_sample in measured_analysis.mc_samples
                    ]
                )
                fake_mc = all_mc - true_mc
                fake_mc.SetName(f"{config.label}_{selection_name}_{FAKES_SOURCE}_mc_fake")
                fake_mc.SetDirectory(0)
                mc_fake_hists[selection_name] = fake_mc

            mc_fake_factor = mc_fake_hists[fake_cr_pass] / mc_fake_hists[fake_cr_fail]
            mc_fake_factor.SetName(f"{config.label}_{FAKES_SOURCE}_mc_fake_factor")
            predicted_mc_fake = mc_fake_hists[sr_fail] * mc_fake_factor
            predicted_mc_fake.SetName(f"{config.label}_{FAKES_SOURCE}_predicted_mc_fake")
            actual_mc_fake = mc_fake_hists[sr_pass]
            mean_dev, max_dev, integral_ratio = closure_metrics(predicted_mc_fake, actual_mc_fake)
            fake_mc_closure_rows.append(
                (
                    config.label,
                    actual_mc_fake.Integral(),
                    predicted_mc_fake.Integral(),
                    mean_dev,
                    max_dev,
                    integral_ratio,
                )
            )

            plotter.paths.plot_dir = (
                plotter.paths.output_dir / "plots" / config.label / "fake_diagnostics"
            )
            plotter.plot(
                [actual_mc_fake, predicted_mc_fake],
                label=["Actual MC fake in SR pass-ID", "Predicted MC fake from fake factor"],
                colour=["k", "b"],
                histstyle=["step", "step"],
                xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                title=smart_join(config.label, FAKES_SOURCE, "MC fake closure", sep=" | "),
                scale_by_bin_width=False,
                ylabel="Events",
                logx=True,
                ratio_plot=True,
                ratio_label="Predicted / actual",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                filename=f"{config.label}_{FAKES_SOURCE}_mc_fake_closure.png",
            )

        # CENTRAL-VALUE MODE NOTICE
        # --------------------------------------------------------------------
        if not DO_FULL_SYSTEMATICS:
            plotter.logger.info(
                "DO_FULL_SYSTEMATICS is False: producing central-value closure only for %s.",
                config.label,
            )

        for var in vars_for_config:
            # NOMINAL UNFOLDING INPUTS
            # ----------------------------------------------------------------
            truth_var = variable_data[var]["truth"]
            response_matrix_name = f"{var}_{truth_var}"
            data = measured_analysis.get_hist(
                var,
                dataset=measured_analysis.data_sample,
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
            if USE_PRONG_SPLIT_FAKES:
                prong_fakes_prefix = (
                    f"{config.label}_{WP}_{{prong}}prong{FAKE_CONTROL_REGION.output_tag}"
                )
                prong_fakes = [
                    measured_analysis.histograms[
                        prong_fakes_prefix.format(prong=prong)
                        + f"_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                    ]
                    for prong in (1, 3)
                ]
                fakes = sum_th1s(*prong_fakes)
                fakes.SetName(f"{fakes_name}_{var}_prong_split_fakes_bkg_{FAKES_SOURCE}_src")
                fakes.SetDirectory(0)
                plotter.logger.info(
                    "Using prong-split fake estimate for %s %s from %s.",
                    config.label,
                    var,
                    FAKE_CONTROL_REGION.selection_tag,
                )
            else:
                fakes = measured_analysis.histograms[
                    f"{fakes_name}_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                plotter.logger.info(
                    "Using inclusive fake estimate for %s %s from %s.",
                    config.label,
                    var,
                    FAKE_CONTROL_REGION.selection_tag,
                )
            all_reco_signal = response_analysis.get_hist(
                var,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=reco_selection,
            )
            fiducial_reco_signal = response_analysis.get_hist(
                var,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=truth_reco_selection,
            )
            nonfiducial_signal = all_reco_signal - fiducial_reco_signal
            nonfiducial_signal.SetName(f"{config.label}_{var}_nonfiducial_signal")
            nonfiducial_signal.SetDirectory(0)

            nonfiducial_fraction = (
                nonfiducial_signal.Integral() / all_reco_signal.Integral()
                if all_reco_signal.Integral() != 0
                else 0.0
            )
            plotter.logger.info(
                "Applying nonfiducial signal correction for %s %s: %.1f%% of reco signal.",
                config.label,
                var,
                100 * nonfiducial_fraction,
            )

            background = sum_th1s(*(backgrounds + [fakes]))
            data_sig = data - background - nonfiducial_signal
            data_sig.SetName(f"{config.label}_{var}_data_minus_background_nonfiducial")
            signal = fiducial_reco_signal.Clone(f"{config.label}_{var}_fiducial_signal")
            signal.SetDirectory(0)

            prompt_background = sum_th1s(*backgrounds)
            data_sig_no_fake = data - prompt_background - nonfiducial_signal
            data_sig_no_fake.SetName(f"{config.label}_{var}_data_minus_prompt_nonfiducial")
            fake_budget_rows.append(
                (
                    config.label,
                    var,
                    data.Integral(),
                    prompt_background.Integral(),
                    fakes.Integral(),
                    nonfiducial_signal.Integral(),
                    data_sig.Integral(),
                    data_sig_no_fake.Integral(),
                    fiducial_reco_signal.Integral(),
                    fiducial_reco_signal.Integral() / data_sig.Integral()
                    if data_sig.Integral() != 0
                    else float("nan"),
                )
            )

            truth_response = response_analysis.get_hist(
                truth_var,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=truth_selection,
            )
            matrix = response_analysis.get_hist(
                response_matrix_name,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=truth_reco_selection,
            )
            response = ResponseComponents(
                response=ROOT.RooUnfoldResponse(fiducial_reco_signal, truth_response, matrix),
                reco=fiducial_reco_signal,
                truth=truth_response,
                matrix=matrix,
            )
            nominal_truth = nominal_truth_hists[var]
            truth = Histogram1D(th1=nominal_truth) / LUMI

            # FAKE-SUBTRACTION DIAGNOSTICS
            # ----------------------------------------------------------------
            if RUN_FAKE_DIAGNOSTICS:
                scaled_fake_unfolded = []
                for fake_scale in FAKE_SCALE_SCAN:
                    scaled_fakes = fakes.Clone(
                        f"{config.label}_{var}_fake_scale_{fake_scale:g}_fakes"
                    )
                    scaled_fakes.SetDirectory(0)
                    scaled_fakes.Scale(fake_scale)
                    scaled_data_sig = data - prompt_background - scaled_fakes - nonfiducial_signal
                    scaled_data_sig.SetName(
                        f"{config.label}_{var}_fake_scale_{fake_scale:g}_data_signal"
                    )
                    fake_scale_rows.append(
                        (
                            config.label,
                            var,
                            fake_scale,
                            scaled_data_sig.Integral(),
                            fiducial_reco_signal.Integral(),
                            fiducial_reco_signal.Integral() / scaled_data_sig.Integral()
                            if scaled_data_sig.Integral() != 0
                            else float("nan"),
                        )
                    )
                    scaled_unfolded, _ = unfold_histogram(
                        plotter,
                        scaled_data_sig,
                        response,
                        FAKE_DIAGNOSTIC_ITERATION,
                    )
                    scaled_unfolded = scale_and_crop_unfolded(
                        scaled_unfolded,
                        nominal_truth,
                        f"{config.label}_{var}_fake_scale_{fake_scale:g}_unfolded",
                    )
                    scaled_fake_unfolded.append(scaled_unfolded)

                plotter.paths.plot_dir = (
                    plotter.paths.output_dir / "plots" / config.label / var / "fake_diagnostics"
                )
                plotter.plot(
                    [truth, *scaled_fake_unfolded],
                    label=["Truth MC"]
                    + [f"Unfolded data, fake scale {scale:g}" for scale in FAKE_SCALE_SCAN],
                    colour=["r", "k", "tab:orange", "b"],
                    histstyle=["step", "step", "step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=True,
                    do_syst=False,
                    title=smart_join(config.label, var, "fake scale scan", sep=" | "),
                    scale_by_bin_width=True,
                    ylabel=(
                        r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                        + variable_data[var]["symbol"]
                        + r"}$ [fb / GeV]"
                    ),
                    logx=True,
                    ratio_plot=True,
                    ratio_label="Data / MC",
                    ratio_axlim=(0.5, 1.5),
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_fake_scale_scan.png",
                )

            # WTAUNU PRONG-COMPOSITION MODELLING SYSTEMATIC
            # ----------------------------------------------------------------
            if RUN_WTAUNU_PRONG_SYSTEMATIC and var in WTAUNU_PRONG_SYSTEMATIC_VARS:
                nominal_prong_unfolded, _ = unfold_histogram(
                    plotter,
                    data_sig,
                    response,
                    FAKE_DIAGNOSTIC_ITERATION,
                )
                nominal_prong_unfolded = scale_and_crop_unfolded(
                    nominal_prong_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_wtaunu_prong_nominal_unfolded",
                )

                prong_all_reco = {
                    prong: response_analysis.get_hist(
                        var,
                        dataset="wtaunu_had",
                        systematic=NOMINAL_NAME,
                        selection=names["reco"],
                    )
                    for prong, names in response_prong_names.items()
                }
                prong_fiducial_reco = {
                    prong: response_analysis.get_hist(
                        var,
                        dataset="wtaunu_had",
                        systematic=NOMINAL_NAME,
                        selection=names["truth_reco"],
                    )
                    for prong, names in response_prong_names.items()
                }
                prong_response_matrix = {
                    prong: response_analysis.get_hist(
                        response_matrix_name,
                        dataset="wtaunu_had",
                        systematic=NOMINAL_NAME,
                        selection=names["truth_reco"],
                    )
                    for prong, names in response_prong_names.items()
                }

                for variation_name, prong_scales in WTAUNU_PRONG_VARIATIONS.items():
                    varied_all_reco_signal = None
                    varied_fiducial_reco_signal = None
                    varied_matrix = None
                    for prong in (1, 3):
                        scale = prong_scales[prong]

                        scaled_all = prong_all_reco[prong].Clone(
                            f"{config.label}_{var}_{variation_name}_{prong}prong_all_reco"
                        )
                        scaled_all.SetDirectory(0)
                        scaled_all.Scale(scale)
                        if varied_all_reco_signal is None:
                            varied_all_reco_signal = scaled_all
                        else:
                            varied_all_reco_signal.Add(scaled_all)

                        scaled_fiducial = prong_fiducial_reco[prong].Clone(
                            f"{config.label}_{var}_{variation_name}_{prong}prong_fiducial_reco"
                        )
                        scaled_fiducial.SetDirectory(0)
                        scaled_fiducial.Scale(scale)
                        if varied_fiducial_reco_signal is None:
                            varied_fiducial_reco_signal = scaled_fiducial
                        else:
                            varied_fiducial_reco_signal.Add(scaled_fiducial)

                        scaled_matrix = prong_response_matrix[prong].Clone(
                            f"{config.label}_{var}_{variation_name}_{prong}prong_response"
                        )
                        scaled_matrix.SetDirectory(0)
                        scaled_matrix.Scale(scale)
                        if varied_matrix is None:
                            varied_matrix = scaled_matrix
                        else:
                            varied_matrix.Add(scaled_matrix)

                    varied_nonfiducial_signal = (
                        varied_all_reco_signal - varied_fiducial_reco_signal
                    )
                    varied_nonfiducial_signal.SetName(
                        f"{config.label}_{var}_{variation_name}_nonfiducial_signal"
                    )
                    varied_nonfiducial_signal.SetDirectory(0)
                    varied_data_sig = data - prompt_background - fakes - varied_nonfiducial_signal
                    varied_data_sig.SetName(
                        f"{config.label}_{var}_{variation_name}_data_signal"
                    )
                    varied_data_sig.SetDirectory(0)
                    varied_response = ResponseComponents(
                        response=ROOT.RooUnfoldResponse(
                            varied_fiducial_reco_signal,
                            truth_response,
                            varied_matrix,
                        ),
                        reco=varied_fiducial_reco_signal,
                        truth=truth_response,
                        matrix=varied_matrix,
                    )
                    varied_unfolded, _ = unfold_histogram(
                        plotter,
                        varied_data_sig,
                        varied_response,
                        FAKE_DIAGNOSTIC_ITERATION,
                    )
                    varied_unfolded = scale_and_crop_unfolded(
                        varied_unfolded,
                        nominal_truth,
                        f"{config.label}_{var}_{variation_name}_wtaunu_prong_unfolded",
                    )

                    wtaunu_prong_rows.append(
                        (
                            config.label,
                            var,
                            variation_name,
                            prong_scales[1],
                            prong_scales[3],
                            nonfiducial_signal.Integral(),
                            varied_nonfiducial_signal.Integral(),
                            data_sig.Integral(),
                            varied_data_sig.Integral(),
                            fiducial_reco_signal.Integral()
                            / data_sig.Integral()
                            if data_sig.Integral() != 0
                            else float("nan"),
                            varied_fiducial_reco_signal.Integral()
                            / varied_data_sig.Integral()
                            if varied_data_sig.Integral() != 0
                            else float("nan"),
                        )
                    )

                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir
                        / "plots"
                        / config.label
                        / var
                        / "wtaunu_prong_systematic"
                    )
                    plotter.plot(
                        [truth, nominal_prong_unfolded, varied_unfolded],
                        label=["Truth MC", "Nominal unfolded data", variation_name],
                        colour=["r", "k", "tab:purple"],
                        histstyle=["step", "errorbar", "errorbar"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            variation_name,
                            "wtaunu prong model",
                            sep=" | ",
                        ),
                        scale_by_bin_width=True,
                        ylabel=(
                            r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                            + variable_data[var]["symbol"]
                            + r"}$ [fb / GeV]"
                        ),
                        logx=True,
                        ratio_plot=True,
                        ratio_label="Data / MC",
                        ratio_axlim=(0.5, 1.5),
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{var}_{variation_name}_unfolded.png",
                    )

            # 1-PRONG WIDTH-COMPOSITION SYSTEMATIC PROPAGATION
            # ----------------------------------------------------------------
            if RUN_FAKE_WIDTH_SYSTEMATIC and USE_PRONG_SPLIT_FAKES and (
                var in FAKE_WIDTH_SYSTEMATIC_VARS
            ):
                one_prong_fakes, three_prong_fakes = prong_fakes
                one_prong_name = f"{config.label}_{WP}_1prong{FAKE_CONTROL_REGION.output_tag}"
                one_prong_cr_fail = (
                    f"{config.label}_{WP}_1prong_"
                    f"{FAKE_CONTROL_REGION.selection_tag}_failID"
                )
                one_prong_sr_pass = f"{config.label}_{WP}_1prong_SR_passID"
                one_prong_sr_fail = f"{config.label}_{WP}_1prong_SR_failID"
                true_one_prong_cr_fail = f"trueTau_{one_prong_cr_fail}"
                true_one_prong_sr_fail = f"trueTau_{one_prong_sr_fail}"
                ff_hist = measured_analysis.histograms[f"{one_prong_name}_{FAKES_SOURCE}_FF"]

                nominal_width_unfolded, _ = unfold_histogram(
                    plotter,
                    data_sig,
                    response,
                    FAKE_DIAGNOSTIC_ITERATION,
                )
                nominal_width_unfolded = scale_and_crop_unfolded(
                    nominal_width_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_width_sys_nominal_unfolded",
                )

                for width_var in FAKE_WIDTH_VARIABLES:
                    shifted_name = (
                        f"{one_prong_name}_{width_var}_{var}_"
                        "application_to_lowmet_width_sys_fakes"
                    )
                    if load_measured_analysis_hists:
                        shifted_one_prong_fakes = measured_analysis.histograms.get(shifted_name)
                        if shifted_one_prong_fakes is None:
                            plotter.logger.warning(
                                "Skipping width systematic %s for %s %s: missing saved "
                                "histogram '%s'. Rerun with LOAD_SAVED_HISTS = False.",
                                width_var,
                                config.label,
                                var,
                                shifted_name,
                            )
                            continue
                    else:
                        low_width = fake_like_histogram(
                            measured_analysis,
                            width_var,
                            one_prong_cr_fail,
                            true_one_prong_cr_fail,
                            f"{one_prong_name}_{width_var}_lowmet_fake_like",
                        )
                        target_width = fake_like_histogram(
                            measured_analysis,
                            width_var,
                            one_prong_sr_fail,
                            true_one_prong_sr_fail,
                            f"{one_prong_name}_{width_var}_sr_fake_like",
                        )
                        low_shape = positive_unit_shape(
                            low_width,
                            f"{one_prong_name}_{width_var}_lowmet_width_shape",
                        )
                        target_shape = positive_unit_shape(
                            target_width,
                            f"{one_prong_name}_{width_var}_sr_width_shape",
                        )
                        width_ratio = make_width_ratio(
                            low_shape,
                            target_shape,
                            f"{one_prong_name}_{width_var}_application_to_lowmet_ratio",
                        )
                        measured_analysis.histograms[low_shape.GetName()] = low_shape
                        measured_analysis.histograms[target_shape.GetName()] = target_shape
                        measured_analysis.histograms[width_ratio.GetName()] = width_ratio
                        shifted_one_prong_fakes = fill_width_shifted_fake_prediction(
                            measured_analysis,
                            target_var=var,
                            width_var=width_var,
                            sr_pass_selection=one_prong_sr_pass,
                            sr_fail_selection=one_prong_sr_fail,
                            true_sr_fail_selection=true_one_prong_sr_fail,
                            ff_hist=ff_hist,
                            width_ratio=width_ratio,
                            output_name=shifted_name,
                        )

                    shifted_fakes = sum_th1s(shifted_one_prong_fakes, three_prong_fakes)
                    shifted_fakes.SetName(f"{fakes_name}_{width_var}_{var}_width_sys_fakes")
                    shifted_fakes.SetDirectory(0)
                    shifted_data_sig = (
                        data - prompt_background - shifted_fakes - nonfiducial_signal
                    )
                    shifted_data_sig.SetName(
                        f"{config.label}_{width_var}_{var}_width_sys_data_signal"
                    )
                    shifted_data_sig.SetDirectory(0)
                    shifted_unfolded, _ = unfold_histogram(
                        plotter,
                        shifted_data_sig,
                        response,
                        FAKE_DIAGNOSTIC_ITERATION,
                    )
                    shifted_unfolded = scale_and_crop_unfolded(
                        shifted_unfolded,
                        nominal_truth,
                        f"{config.label}_{width_var}_{var}_width_sys_unfolded",
                    )

                    fake_width_rows.append(
                        (
                            config.label,
                            var,
                            width_var,
                            fakes.Integral(),
                            shifted_fakes.Integral(),
                            data_sig.Integral(),
                            shifted_data_sig.Integral(),
                            fiducial_reco_signal.Integral() / data_sig.Integral()
                            if data_sig.Integral() != 0
                            else float("nan"),
                            fiducial_reco_signal.Integral() / shifted_data_sig.Integral()
                            if shifted_data_sig.Integral() != 0
                            else float("nan"),
                        )
                    )

                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir
                        / "plots"
                        / config.label
                        / var
                        / "fake_width_systematic"
                    )
                    plotter.plot(
                        [fakes, shifted_fakes],
                        label=["Nominal prong-split fakes", f"{width_var} shifted fakes"],
                        colour=["k", "tab:orange"],
                        histstyle=["step", "step"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            width_var,
                            "fake-source composition shift",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Events",
                        logx=True,
                        ratio_plot=True,
                        ratio_label="Shifted / nominal",
                        ratio_axlim=(0.5, 1.5),
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{var}_{width_var}_fake_width_shift.png",
                    )
                    plotter.plot(
                        [truth, nominal_width_unfolded, shifted_unfolded],
                        label=["Truth MC", "Nominal unfolded data", f"{width_var} shifted data"],
                        colour=["r", "k", "tab:orange"],
                        histstyle=["step", "errorbar", "errorbar"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            width_var,
                            "unfolded fake-source shift",
                            sep=" | ",
                        ),
                        scale_by_bin_width=True,
                        ylabel=(
                            r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                            + variable_data[var]["symbol"]
                            + r"}$ [fb / GeV]"
                        ),
                        logx=True,
                        ratio_plot=True,
                        ratio_label="Data / MC",
                        ratio_axlim=(0.5, 1.5),
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{var}_{width_var}_unfolded_width_shift.png",
                    )

            elif RUN_FAKE_WIDTH_SYSTEMATIC and not USE_PRONG_SPLIT_FAKES:
                plotter.logger.warning(
                    "Skipping fake-width systematic for %s %s because prong-split fakes "
                    "are disabled.",
                    config.label,
                    var,
                )

            if RUN_FAKE_DIAGNOSTICS and BUILD_INCLUSIVE_FAKES and USE_PRONG_SPLIT_FAKES:
                inclusive_fakes = measured_analysis.histograms[
                    f"{fakes_name}_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                prong_sum_fakes = fakes
                fake_prong_rows.append(
                    (
                        config.label,
                        var,
                        inclusive_fakes.Integral(),
                        prong_fakes[0].Integral(),
                        prong_fakes[1].Integral(),
                        prong_sum_fakes.Integral(),
                    )
                )
                plotter.plot(
                    [inclusive_fakes, prong_sum_fakes],
                    label=["Inclusive fake estimate", "1-prong + 3-prong fake estimate"],
                    colour=["k", "b"],
                    histstyle=["step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=True,
                    do_syst=False,
                    title=smart_join(config.label, var, "fake prong split", sep=" | "),
                    scale_by_bin_width=False,
                    ylabel="Events",
                    logx=True,
                    ratio_plot=True,
                    ratio_label="Prong sum / inclusive",
                    ratio_axlim=(0.5, 1.5),
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_inclusive_vs_prong_split_fakes.png",
                )

            # RESPONSE SYSTEMATIC DIAGNOSTICS
            # ----------------------------------------------------------------
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

                response_reference_iter = 0
                nominal_unfolded, _ = unfold_histogram(
                    plotter,
                    signal,
                    response,
                    response_reference_iter,
                )
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
                            selection=truth_reco_selection,
                        )
                        reco_down = response_analysis.get_hist(
                            var,
                            dataset="wtaunu_had",
                            systematic=down,
                            selection=truth_reco_selection,
                        )
                        matrix_up = response_analysis.get_hist(
                            response_matrix_name,
                            dataset="wtaunu_had",
                            systematic=up,
                            selection=truth_reco_selection,
                        )
                        matrix_down = response_analysis.get_hist(
                            response_matrix_name,
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
                    unfolded_up, _ = unfold_histogram(
                        plotter,
                        signal,
                        response_up,
                        response_reference_iter,
                    )
                    unfolded_down, _ = unfold_histogram(
                        plotter,
                        signal,
                        response_down,
                        response_reference_iter,
                    )
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

            # RESPONSE MATRIX PLOT
            # ----------------------------------------------------------------
            plotter.paths.plot_dir = plotter.paths.output_dir / "plots" / config.label / var
            plotter.plot_2d(
                response.matrix,
                ylabel=f"Truth {truth_var}",
                xlabel=f"Reco {var}",
                title=smart_join(config.label, var, "response matrix", sep=" | "),
                labels=False,
                label_params={"llabel": ""},
                filename=f"{config.label}_{var}_response_matrix.png",
            )

            # MAIN UNFOLDING RESULT
            # ----------------------------------------------------------------
            unfolded_by_iteration = {}
            iteration_counts = ITERATIONS
            for iter_count in iteration_counts:
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
                        + variable_data[var]["symbol"]
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

            # UNFOLDING ITERATION COMPARISON
            # ----------------------------------------------------------------
            plotter.paths.plot_dir = (
                plotter.paths.output_dir / "plots" / config.label / var / "compare"
            )
            plotter.plot(
                [truth] + [unfolded_by_iteration[i] for i in iteration_counts],
                label=["Truth"]
                + [
                    "Bin-by-bin unfolding"
                    if iter_count == 0
                    else f"Bayesian unfolding, {iter_count} iterations"
                    for iter_count in iteration_counts
                ],
                xlabel=variable_data[var]["name"] + " [GeV]",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                title=smart_join(config.label, var, "Medium Tau ID", sep=" | "),
                scale_by_bin_width=True,
                ylabel=(
                    r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                    + variable_data[var]["symbol"]
                    + r"}$ [fb / GeV]"
                ),
                logx=True,
                label_params={"llabel": "", "loc": 1},
                filename=f"{config.label}_{var}_iteration_compare.png",
            )

            # SPLIT-SAMPLE MC CLOSURE
            # ----------------------------------------------------------------
            if split_response_analysis is not None and split_pseudo_data_analysis is not None:
                split_reco = split_response_analysis.get_hist(
                    var,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=truth_reco_selection,
                )
                split_truth_response = split_response_analysis.get_hist(
                    truth_var,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=truth_selection,
                )
                split_matrix = split_response_analysis.get_hist(
                    response_matrix_name,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=truth_reco_selection,
                )
                split_response = ResponseComponents(
                    response=ROOT.RooUnfoldResponse(split_reco, split_truth_response, split_matrix),
                    reco=split_reco,
                    truth=split_truth_response,
                    matrix=split_matrix,
                )

                pseudo_all_reco_signal = split_pseudo_data_analysis.get_hist(
                    var,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=reco_selection,
                )
                pseudo_fiducial_reco_signal = split_pseudo_data_analysis.get_hist(
                    var,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=truth_reco_selection,
                )
                pseudo_nonfiducial_signal = (
                    pseudo_all_reco_signal - pseudo_fiducial_reco_signal
                )
                pseudo_nonfiducial_signal.SetName(
                    f"{config.label}_{var}_split_nonfiducial_signal"
                )
                pseudo_nonfiducial_signal.SetDirectory(0)
                pseudo_data_signal = pseudo_all_reco_signal - pseudo_nonfiducial_signal
                pseudo_data_signal.SetName(f"{config.label}_{var}_split_pseudo_data_signal")
                pseudo_data_signal.SetDirectory(0)

                split_nominal_truth = split_nominal_truth_hists[var]
                split_truth = Histogram1D(th1=split_nominal_truth) / LUMI
                split_unfolded_by_iteration = {}
                for iter_count in iteration_counts:
                    split_signal_unfolded, _ = unfold_histogram(
                        plotter,
                        pseudo_data_signal,
                        split_response,
                        iter_count,
                    )
                    split_signal_unfolded = scale_and_crop_unfolded(
                        split_signal_unfolded,
                        split_nominal_truth,
                        f"{config.label}_{var}_{iter_count}iter_split_signal_unfolded",
                    )
                    split_unfolded_by_iteration[iter_count] = split_signal_unfolded
                    mean_dev, max_dev, integral_ratio = closure_metrics(
                        split_signal_unfolded,
                        split_truth.TH1,
                    )
                    split_closure_rows.append(
                        (config.label, var, iter_count, mean_dev, max_dev, integral_ratio)
                    )

                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir / "plots" / config.label / var / "split_closure"
                    )
                    split_plot_args: PlotKwargs = {
                        "label": ["Held-out Truth MC", "Unfolded Held-out Signal MC"],
                        "colour": ["r", "b"],
                        "histstyle": ["step", "step"],
                        "xlabel": variable_data[var]["name"] + " [GeV]",
                        "kind": "overlay",
                        "do_stat": True,
                        "do_syst": False,
                        "title": smart_join(
                            "Split-sample MC closure",
                            config.label,
                            "Medium Tau ID",
                            r"$\sqrt{s} = 13$TeV",
                            sep=" | ",
                        ),
                        "scale_by_bin_width": True,
                        "ylabel": (
                            r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                            + variable_data[var]["symbol"]
                            + r"}$ [fb / GeV]"
                        ),
                        "logx": True,
                        "ratio_plot": True,
                        "ratio_label": "Unfolded / Truth",
                        "ratio_axlim": (0.5, 1.5),
                        "label_params": {"llabel": "", "loc": 1},
                    }
                    plotter.plot(
                        [split_truth, split_signal_unfolded],
                        **split_plot_args,
                        filename=f"{config.label}_{var}_{iter_count}iter_split_closure.png",
                    )
                    plotter.plot(
                        [split_truth, split_signal_unfolded],
                        **split_plot_args,
                        logy=True,
                        filename=f"{config.label}_{var}_{iter_count}iter_split_closure_logy.png",
                    )

                plotter.paths.plot_dir = (
                    plotter.paths.output_dir
                    / "plots"
                    / config.label
                    / var
                    / "split_closure"
                    / "compare"
                )
                plotter.plot(
                    [split_truth] + [split_unfolded_by_iteration[i] for i in iteration_counts],
                    label=["Held-out Truth"]
                    + [
                        "Bin-by-bin unfolding"
                        if iter_count == 0
                        else f"Bayesian unfolding, {iter_count} iterations"
                        for iter_count in iteration_counts
                    ],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=True,
                    do_syst=False,
                    title=smart_join(config.label, var, "split-sample closure", sep=" | "),
                    scale_by_bin_width=True,
                    ylabel=(
                        r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                        + variable_data[var]["symbol"]
                        + r"}$ [fb / GeV]"
                    ),
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_split_closure_iteration_compare.png",
                )

    # SUMMARY OUTPUT
    # ========================================================================
    summary_lines = [
        "# Variable-specific shadow-bin unfolding closure summary",
        "",
        f"DO_FULL_SYSTEMATICS: `{DO_FULL_SYSTEMATICS}`",
        f"DO_SPLIT_SAMPLE_CLOSURE: `{DO_SPLIT_SAMPLE_CLOSURE}`",
        f"FAKE_CONTROL_REGION: `{FAKE_CONTROL_REGION.selection_tag}`",
        "",
        "These rows are same-sample MC self-closure checks. They verify bookkeeping, but they are "
        "not independent validation because the signal input and response use the same MC sample.",
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

    if split_closure_rows:
        split_summary_lines = [
            "# Split-sample shadow-bin unfolding closure summary",
            "",
            "The response is built from even-numbered events and the pseudo-data/truth target "
            "from odd-numbered events. This is the independent MC closure test; exact agreement "
            "is not guaranteed by construction.",
            "",
            "| Configuration | Variable | Iterations | Mean deviation | Max deviation | Integral ratio |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for config_label, var, iter_count, mean_dev, max_dev, integral_ratio in split_closure_rows:
            split_summary_lines.append(
                f"| {config_label} | {var} | {iter_count} | {mean_dev:.3f} | "
                f"{max_dev:.3f} | {integral_ratio:.3f} |"
            )
        split_summary_path = output_root / "split_closure_summary.md"
        split_summary_path.write_text("\n".join(split_summary_lines) + "\n")
        plotter.logger.info("Saved split-sample closure summary to %s", split_summary_path)

    if RUN_FAKE_DIAGNOSTICS:
        fake_summary_lines = [
            "# Fake-estimate diagnostics",
            "",
            "These diagnostics test whether the data-driven fake estimate is driving the "
            "normalisation of the unfolded data input. They do not change the nominal unfolded "
            "result; they provide cross-checks of the fake subtraction.",
            "",
            f"Fake control region: `{FAKE_CONTROL_REGION.selection_tag}`.",
            f"Fake-factor source variable: `{FAKES_SOURCE}`.",
            "",
            "## Pre-unfolding budget",
            "",
            "`data_sig` is the nominal unfolded input before unfolding:",
            "`data - prompt backgrounds - fake estimate - nonfiducial signal`.",
            "",
            "| Configuration | Variable | Data | Prompt bkg | Fakes | Nonfid signal | "
            "Data sig | Data sig, no fakes | Fid reco signal | Fid reco / data sig |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fake_budget_rows:
            (
                config_label,
                var,
                data_integral,
                prompt_bkg,
                fakes_integral,
                nonfid,
                data_sig,
                data_sig_no_fake,
                fid_reco,
                fid_over_data_sig,
            ) = row
            fake_summary_lines.append(
                f"| {config_label} | {var} | {data_integral:.3f} | {prompt_bkg:.3f} | "
                f"{fakes_integral:.3f} | {nonfid:.3f} | {data_sig:.3f} | "
                f"{data_sig_no_fake:.3f} | {fid_reco:.3f} | {fid_over_data_sig:.3f} |"
            )

        fake_summary_lines.extend(
            [
                "",
                "## Fake-scale scan",
                "",
                f"The unfolded diagnostic plots use `{FAKE_DIAGNOSTIC_ITERATION}` Bayesian "
                "iteration(s). The table below shows the pre-unfolding signal integral for each "
                "fake scale.",
                "",
                "| Configuration | Variable | Fake scale | Data sig | Fid reco signal | "
                "Fid reco / data sig |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for (
            config_label,
            var,
            fake_scale,
            data_sig,
            fid_reco,
            fid_over_data_sig,
        ) in fake_scale_rows:
            fake_summary_lines.append(
                f"| {config_label} | {var} | {fake_scale:.2f} | {data_sig:.3f} | "
                f"{fid_reco:.3f} | {fid_over_data_sig:.3f} |"
            )

        fake_summary_lines.extend(
            [
                "",
                "## 1-prong tau-width fake-source systematic propagation",
                "",
                "This propagates the candidate 1-prong fake-source composition uncertainty "
                "into the unfolded-input budget. The nominal fake estimate is unchanged. "
                "Only the 1-prong fake component is shifted; the 3-prong fake component is "
                "left nominal because the high-MET 3-prong target is not physically usable.",
                "",
                f"Shifted templates are unfolded with `{FAKE_DIAGNOSTIC_ITERATION}` Bayesian "
                "iteration(s), matching the fake-scale diagnostic.",
                "",
                "| Configuration | Variable | Width proxy | Nominal fakes | Shifted fakes | "
                "Nominal data sig | Shifted data sig | Fid reco / nominal data sig | "
                "Fid reco / shifted data sig |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if fake_width_rows:
            for (
                config_label,
                var,
                width_var,
                nominal_fakes,
                shifted_fakes,
                nominal_data_sig,
                shifted_data_sig,
                fid_over_nominal,
                fid_over_shifted,
            ) in fake_width_rows:
                fake_summary_lines.append(
                    f"| {config_label} | {var} | `{width_var}` | {nominal_fakes:.3f} | "
                    f"{shifted_fakes:.3f} | {nominal_data_sig:.3f} | "
                    f"{shifted_data_sig:.3f} | {fid_over_nominal:.3f} | "
                    f"{fid_over_shifted:.3f} |"
                )
        else:
            fake_summary_lines.append(
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )

        fake_summary_lines.extend(
            [
                "",
                "## wtaunu_had prong-composition modelling systematic",
                "",
                "This optional diagnostic tests the impact of a prong-dependent modelling "
                "variation on the reconstructed `wtaunu_had` signal component. The nominal "
                "result is unchanged. The varied response matrix, reconstructed signal "
                "projection, and nonfiducial correction are built from separate 1-prong and "
                "3-prong response selections using the scale factors shown below.",
                "",
                f"Shifted templates are unfolded with `{FAKE_DIAGNOSTIC_ITERATION}` Bayesian "
                "iteration(s), matching the other fake/subtraction diagnostics.",
                "",
                "| Configuration | Variable | Variation | 1-prong scale | 3-prong scale | "
                "Nominal nonfid | Varied nonfid | Nominal data sig | Varied data sig | "
                "Nominal fid reco / data sig | Varied fid reco / data sig |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if wtaunu_prong_rows:
            for (
                config_label,
                var,
                variation_name,
                one_prong_scale,
                three_prong_scale,
                nominal_nonfid,
                varied_nonfid,
                nominal_data_sig,
                varied_data_sig,
                nominal_fid_over_data,
                varied_fid_over_data,
            ) in wtaunu_prong_rows:
                fake_summary_lines.append(
                    f"| {config_label} | {var} | `{variation_name}` | "
                    f"{one_prong_scale:.3f} | {three_prong_scale:.3f} | "
                    f"{nominal_nonfid:.3f} | {varied_nonfid:.3f} | "
                    f"{nominal_data_sig:.3f} | {varied_data_sig:.3f} | "
                    f"{nominal_fid_over_data:.3f} | {varied_fid_over_data:.3f} |"
                )
        else:
            fake_summary_lines.append(
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )

        fake_summary_lines.extend(
            [
                "",
                "## MC fake closure",
                "",
                "This checks the fake-factor transfer in MC only, using the known non-true-tau "
                "component as the target. The fake factor is built in `TauPt`, matching the "
                "nominal fake source variable. The derivation region is the same one selected "
                "for the data-driven fake estimate in this run.",
                "",
                "| Configuration | Actual MC fake | Predicted MC fake | Mean deviation | "
                "Max deviation | Integral ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for (
            config_label,
            actual_fake,
            predicted_fake,
            mean_dev,
            max_dev,
            integral_ratio,
        ) in fake_mc_closure_rows:
            fake_summary_lines.append(
                f"| {config_label} | {actual_fake:.3f} | {predicted_fake:.3f} | "
                f"{mean_dev:.3f} | {max_dev:.3f} | {integral_ratio:.3f} |"
            )

        fake_summary_lines.extend(
            [
                "",
                "## Inclusive versus prong-split fakes",
                "",
                "The nominal fake estimate is prong-split. This compares the optional inclusive "
                "fake estimate against the sum of separate 1-prong and 3-prong fake estimates.",
                "",
                "| Configuration | Variable | Inclusive fakes | 1-prong fakes | 3-prong fakes | "
                "Prong-sum fakes |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for config_label, var, inclusive, one_prong, three_prong, prong_sum in fake_prong_rows:
            fake_summary_lines.append(
                f"| {config_label} | {var} | {inclusive:.3f} | {one_prong:.3f} | "
                f"{three_prong:.3f} | {prong_sum:.3f} |"
            )

        fake_summary_path = output_root / "fake_diagnostics_summary.md"
        fake_summary_path.write_text("\n".join(fake_summary_lines) + "\n")
        plotter.logger.info("Saved fake diagnostics summary to %s", fake_summary_path)

    if not load_measured_analysis_hists:
        measured_analysis.save_hists()

    plotter.logger.info("DONE.")
