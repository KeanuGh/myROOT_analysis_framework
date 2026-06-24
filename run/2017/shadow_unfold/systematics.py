import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, signal_sample

from shadow_unfold.fakes import (
    fake_like_histogram,
    fill_width_reweighted_fake_prediction_from_factor,
    positive_unit_shape,
    shape_ratio_histogram,
)
from shadow_unfold.models import FakeControlRegion, ResponseComponents
from shadow_unfold.selections import build_reco_sr_cuts, fiducial_truth_hard_cut
from src.analysis import Analysis
from src.cutting import Cut
from src.fakes import fill_fake_predictions_from_factor
from src.unfolding import scale_and_crop_unfolded, unfold_histogram
from utils import ROOT_utils
from utils.plotting_tools import Hist2dOpts
from utils.ROOT_utils import sum_th1s

JET_FAKE_FF_STAT = "JET_FAKE_FF_STAT"
JET_FAKE_MET_WINDOW = "JET_FAKE_MET_WINDOW"
JET_FAKE_TAU_WIDTH_COMPOSITION = "JET_FAKE_TAU_WIDTH_COMPOSITION"
TES_SYS_PATTERN = r"^TAUS_TRUEHADTAU_SME_TES_.*__(1up|1down)$"
NON_TES_OR_NOMINAL_SYS_PATTERN = (
    rf"^(?!{NOMINAL_NAME}$)(?!TAUS_TRUEHADTAU_SME_TES_.*__(1up|1down)$).*"
)


def _response_binnings(config) -> dict[str, np.ndarray]:
    """Return binnings matching the main shadow-unfold response configuration."""
    config_binnings = dict(BINNINGS)
    if config.unfolded_var == "MTW":
        mtw_bins = np.array(
            [
                config.mtw_min,
                *[
                    bin_edge
                    for bin_edge in BINNINGS["MTW"]
                    if bin_edge > config.mtw_min
                ],
            ],
            dtype=float,
        )
        config_binnings["MTW"] = mtw_bins
        config_binnings["TruthMTW"] = mtw_bins
    return config_binnings


def build_tes_response_systematics(
    *,
    configs,
    vars_to_build,
    pass_medium: Cut,
    skip_sys: set[str],
    output_root: Path,
    wp: str,
    year: int,
    log_out: str = "both",
) -> None:
    """Build TES response histograms into the normal response ROOT cache.

    TES shifted trees cannot evaluate the full truth+reco response selection
    directly because the generic builder strips truth-tagged variables from
    non-nominal trees. This uses nominal hard-cut event masks for the fiducial
    truth phase space, then applies only reco cuts in the shifted trees.
    """
    if set(vars_to_build) != {"MTW"}:
        raise RuntimeError(
            "This TES response helper is currently scoped to MTW only. "
            f"Active VARS are {vars_to_build!r}."
        )

    for config in configs:
        selection = f"{config.label}_{wp}_truth_reco_tau"
        reco_cuts = build_reco_sr_cuts(config) + [pass_medium]

        sample = deepcopy(signal_sample(selections={selection: reco_cuts}))
        base_hard_cuts = sample.get("hard_cut", {})
        if not isinstance(base_hard_cuts, dict):
            raise TypeError("Expected signal sample hard_cut to be a subsample dictionary.")
        fiducial_truth_cut = fiducial_truth_hard_cut(config)
        sample["hard_cut"] = {
            subsample: f"({base_cut}) && ({fiducial_truth_cut})"
            for subsample, base_cut in base_hard_cuts.items()
        }

        binnings = {
            "": _response_binnings(config),
            rf"^{re.escape(selection)}$": _response_binnings(config),
        }

        Analysis(
            {"wtaunu_had": sample},
            year=year,
            rerun=True,
            regen_histograms=True,
            do_systematics=True,
            metadata_cache=DSID_METADATA_CACHE,
            ttree=NOMINAL_NAME,
            analysis_label=f"build_shadow_response_tes_{config.label}",
            output_dir=output_root / "response",
            log_level=10,
            log_out=log_out,
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
                "VisTruthTauEta",
                "TruthTau_nChargedTracks",
                "TruthTau_isHadronic",
                "eventNumber",
            },
            import_missing_columns_as_nan=True,
            snapshot=False,
            histogram_vars={"MTW"},
            hists_2d={
                "MTW_TruthMTW": Hist2dOpts(
                    "MTW",
                    "TruthMTW",
                    "reco_weight",
                )
            },
            do_unweighted=True,
            systematics_for_selection={rf"^{re.escape(selection)}$"},
            # Only build TES systematics here. The main script already handles
            # nominal and efficiency-weight response objects. Keep the nominal
            # tree unskipped so Dataset.init_sys() can identify the reference
            # dataframe before it registers the TES shifted trees.
            skip_sys=skip_sys | {r"^TAUS_TRUEHADTAU_EFF_.*", NON_TES_OR_NOMINAL_SYS_PATTERN},
            binnings=binnings,
        )


@dataclass
class UnfoldedSystematicVariation:
    iter_count: int
    nominal_unfolded: ROOT.TH1
    shifted_unfolded: ROOT.TH1
    uncertainty: ROOT.TH1
    relative_uncertainty: ROOT.TH1
    shifted_unfolded_down: ROOT.TH1 | None = None


@dataclass
class TauWidthFakeSystematic:
    name: str
    width_variable: str
    shifted_fakes: ROOT.TH1
    shifted_data_sig: ROOT.TH1
    width_ratio: ROOT.TH1
    variations: list[UnfoldedSystematicVariation]


@dataclass
class FakeSourceSystematic:
    name: str
    shifted_fakes_up: ROOT.TH1
    shifted_fakes_down: ROOT.TH1
    shifted_data_sig_up: ROOT.TH1
    shifted_data_sig_down: ROOT.TH1
    variations: list[UnfoldedSystematicVariation]


def build_unfolded_envelope(
    nominal_unfolded: ROOT.TH1,
    shifted_unfolded: ROOT.TH1,
    *,
    name: str,
) -> tuple[ROOT.TH1, ROOT.TH1]:
    """Return absolute and relative uncertainty from one shifted unfolded result."""
    uncertainty = ROOT_utils.th1_max_abs_deviation(
        shifted_unfolded,
        shifted_unfolded,
        nominal_unfolded,
        name=name,
    )
    relative = ROOT_utils.th1_relative_uncertainty(
        uncertainty,
        nominal_unfolded,
        name=f"{name}_relative",
    )
    return uncertainty, relative


def quadrature_sum_histograms(hists: list[ROOT.TH1], *, name: str) -> ROOT.TH1:
    """Return the bin-wise quadrature sum of histograms with matching binning."""
    if not hists:
        raise ValueError("Cannot build a quadrature sum without input histograms.")
    total = hists[0].Clone(name)
    total.SetDirectory(0)
    for bin_idx in range(1, total.GetNbinsX() + 1):
        total.SetBinContent(
            bin_idx,
            sum(hist.GetBinContent(bin_idx) ** 2 for hist in hists) ** 0.5,
        )
        total.SetBinError(bin_idx, 0.0)
    return total


def histogram_has_finite_content(hist: ROOT.TH1 | ROOT.TH2) -> bool:
    """Return whether a histogram has nonzero finite bin content."""
    total = 0.0
    if hist.InheritsFrom("TH2"):
        for x_bin in range(1, hist.GetNbinsX() + 1):
            for y_bin in range(1, hist.GetNbinsY() + 1):
                value = hist.GetBinContent(x_bin, y_bin)
                if not np.isfinite(value):
                    return False
                total += abs(value)
    else:
        for bin_idx in range(1, hist.GetNbinsX() + 1):
            value = hist.GetBinContent(bin_idx)
            if not np.isfinite(value):
                return False
            total += abs(value)
    return total > 0.0


def _data_sig_from_fakes(
    *,
    data: ROOT.TH1,
    nominal_background: ROOT.TH1,
    fakes: ROOT.TH1,
    nonfiducial_signal: ROOT.TH1,
    name: str,
) -> ROOT.TH1:
    background = sum_th1s(nominal_background, fakes)
    data_sig = data - background - nonfiducial_signal
    data_sig.SetName(name)
    data_sig.SetDirectory(0)
    return data_sig


def _unfold_shift_pair(
    *,
    plotter: Analysis,
    nominal_data_sig: ROOT.TH1,
    shifted_data_sig_up: ROOT.TH1,
    shifted_data_sig_down: ROOT.TH1,
    response: ResponseComponents,
    nominal_truth: ROOT.TH1,
    iterations: tuple[int, ...],
    lumi: float,
    name_prefix: str,
) -> list[UnfoldedSystematicVariation]:
    variations = []
    for iter_count in iterations:
        nominal_unfolded, _ = unfold_histogram(
            plotter,
            nominal_data_sig,
            response,
            iter_count,
        )
        shifted_up_unfolded, _ = unfold_histogram(
            plotter,
            shifted_data_sig_up,
            response,
            iter_count,
        )
        shifted_down_unfolded, _ = unfold_histogram(
            plotter,
            shifted_data_sig_down,
            response,
            iter_count,
        )
        nominal_unfolded = scale_and_crop_unfolded(
            nominal_unfolded,
            nominal_truth,
            f"{name_prefix}_{iter_count}iter_nominal_reference",
            lumi,
        )
        shifted_up_unfolded = scale_and_crop_unfolded(
            shifted_up_unfolded,
            nominal_truth,
            f"{name_prefix}_{iter_count}iter_up",
            lumi,
        )
        shifted_down_unfolded = scale_and_crop_unfolded(
            shifted_down_unfolded,
            nominal_truth,
            f"{name_prefix}_{iter_count}iter_down",
            lumi,
        )
        uncertainty = ROOT_utils.th1_max_abs_deviation(
            shifted_up_unfolded,
            shifted_down_unfolded,
            nominal_unfolded,
            name=f"{name_prefix}_{iter_count}iter_uncertainty",
        )
        relative_uncertainty = ROOT_utils.th1_relative_uncertainty(
            uncertainty,
            nominal_unfolded,
            name=f"{name_prefix}_{iter_count}iter_relative_uncertainty",
        )
        variations.append(
            UnfoldedSystematicVariation(
                iter_count=iter_count,
                nominal_unfolded=nominal_unfolded,
                shifted_unfolded=shifted_up_unfolded,
                uncertainty=uncertainty,
                relative_uncertainty=relative_uncertainty,
                shifted_unfolded_down=shifted_down_unfolded,
            )
        )
    return variations


def _shift_fake_factor(ff_hist: ROOT.TH1, *, name: str, direction: int) -> ROOT.TH1:
    shifted = ff_hist.Clone(name)
    shifted.SetDirectory(0)
    for bin_idx in range(1, shifted.GetNbinsX() + 1):
        shifted.SetBinContent(
            bin_idx,
            ff_hist.GetBinContent(bin_idx) + direction * ff_hist.GetBinError(bin_idx),
        )
        shifted.SetBinError(bin_idx, 0.0)
    return shifted


def build_fake_factor_stat_systematic(
    *,
    measured_analysis: Analysis,
    plotter: Analysis,
    config_label: str,
    wp: str,
    fake_control_region: FakeControlRegion,
    fakes_source: str,
    target_var: str,
    data: ROOT.TH1,
    nominal_background: ROOT.TH1,
    nonfiducial_signal: ROOT.TH1,
    nominal_data_sig: ROOT.TH1,
    response: ResponseComponents,
    nominal_truth: ROOT.TH1,
    iterations: tuple[int, ...],
    lumi: float,
) -> FakeSourceSystematic:
    """Propagate fake-factor bin statistical uncertainties to unfolded data."""
    shifted_by_direction = {}
    for direction_name, direction in (("up", 1), ("down", -1)):
        shifted_prong_fakes = []
        for prong in (1, 3):
            prong_prefix = f"{config_label}_{wp}_{prong}prong{fake_control_region.output_tag}"
            prong_ff = measured_analysis.histograms[f"{prong_prefix}_{fakes_source}_FF"]
            shifted_ff = _shift_fake_factor(
                prong_ff,
                name=(
                    f"{prong_prefix}_{target_var}_{JET_FAKE_FF_STAT}_"
                    f"{direction_name}_{fakes_source}_FF"
                ),
                direction=direction,
            )
            measured_analysis.histograms[shifted_ff.GetName()] = shifted_ff
            prong_sr_pass = f"{config_label}_{wp}_{prong}prong_SR_passID"
            prong_sr_fail = f"{config_label}_{wp}_{prong}prong_SR_failID"
            true_prong_sr_fail = f"trueTau_{prong_sr_fail}"
            output_prefix = (
                f"{prong_prefix}_{target_var}_{JET_FAKE_FF_STAT}_{direction_name}"
            )
            fill_fake_predictions_from_factor(
                measured_analysis,
                target_vars=(target_var,),
                sr_pass_selection=prong_sr_pass,
                sr_fail_selection=prong_sr_fail,
                true_sr_fail_selection=true_prong_sr_fail,
                ff_hist=shifted_ff,
                output_prefix=output_prefix,
                fakes_source=fakes_source,
                systematic="T_s1thv_NOMINAL",
            )
            shifted_prong_fakes.append(
                measured_analysis.histograms[
                    f"{output_prefix}_{target_var}_fakes_bkg_{fakes_source}_src"
                ]
            )
        shifted_fakes = sum_th1s(*shifted_prong_fakes)
        shifted_fakes.SetName(
            f"{config_label}_{target_var}_{JET_FAKE_FF_STAT}_{direction_name}_fakes"
        )
        shifted_fakes.SetDirectory(0)
        measured_analysis.histograms[shifted_fakes.GetName()] = shifted_fakes
        shifted_by_direction[direction_name] = shifted_fakes

    shifted_data_sig_up = _data_sig_from_fakes(
        data=data,
        nominal_background=nominal_background,
        fakes=shifted_by_direction["up"],
        nonfiducial_signal=nonfiducial_signal,
        name=f"{config_label}_{target_var}_{JET_FAKE_FF_STAT}_up_data_sig",
    )
    shifted_data_sig_down = _data_sig_from_fakes(
        data=data,
        nominal_background=nominal_background,
        fakes=shifted_by_direction["down"],
        nonfiducial_signal=nonfiducial_signal,
        name=f"{config_label}_{target_var}_{JET_FAKE_FF_STAT}_down_data_sig",
    )
    variations = _unfold_shift_pair(
        plotter=plotter,
        nominal_data_sig=nominal_data_sig,
        shifted_data_sig_up=shifted_data_sig_up,
        shifted_data_sig_down=shifted_data_sig_down,
        response=response,
        nominal_truth=nominal_truth,
        iterations=iterations,
        lumi=lumi,
        name_prefix=f"{config_label}_{target_var}_{JET_FAKE_FF_STAT}",
    )
    for hist in (shifted_data_sig_up, shifted_data_sig_down):
        measured_analysis.histograms[hist.GetName()] = hist
    for variation in variations:
        for hist in (
            variation.nominal_unfolded,
            variation.shifted_unfolded,
            variation.uncertainty,
            variation.relative_uncertainty,
        ):
            measured_analysis.histograms[hist.GetName()] = hist
    return FakeSourceSystematic(
        name=JET_FAKE_FF_STAT,
        shifted_fakes_up=shifted_by_direction["up"],
        shifted_fakes_down=shifted_by_direction["down"],
        shifted_data_sig_up=shifted_data_sig_up,
        shifted_data_sig_down=shifted_data_sig_down,
        variations=variations,
    )


def build_met_window_fake_systematic(
    *,
    measured_analysis: Analysis,
    plotter: Analysis,
    config_label: str,
    target_var: str,
    data: ROOT.TH1,
    nominal_background: ROOT.TH1,
    nonfiducial_signal: ROOT.TH1,
    nominal_data_sig: ROOT.TH1,
    nominal_fakes: ROOT.TH1,
    shifted_fakes: list[ROOT.TH1],
    response: ResponseComponents,
    nominal_truth: ROOT.TH1,
    iterations: tuple[int, ...],
    lumi: float,
) -> FakeSourceSystematic | None:
    """Build an envelope from alternate low-MET fake-factor regions."""
    if not shifted_fakes:
        return None

    shifted_fakes_up = nominal_fakes.Clone(f"{config_label}_{target_var}_{JET_FAKE_MET_WINDOW}_up_fakes")
    shifted_fakes_down = nominal_fakes.Clone(
        f"{config_label}_{target_var}_{JET_FAKE_MET_WINDOW}_down_fakes"
    )
    shifted_fakes_up.SetDirectory(0)
    shifted_fakes_down.SetDirectory(0)
    for bin_idx in range(1, nominal_fakes.GetNbinsX() + 1):
        nominal_value = nominal_fakes.GetBinContent(bin_idx)
        bin_values = [hist.GetBinContent(bin_idx) for hist in shifted_fakes]
        shifted_fakes_up.SetBinContent(bin_idx, max(bin_values + [nominal_value]))
        shifted_fakes_down.SetBinContent(bin_idx, min(bin_values + [nominal_value]))
        shifted_fakes_up.SetBinError(bin_idx, 0.0)
        shifted_fakes_down.SetBinError(bin_idx, 0.0)

    shifted_data_sig_up = _data_sig_from_fakes(
        data=data,
        nominal_background=nominal_background,
        fakes=shifted_fakes_up,
        nonfiducial_signal=nonfiducial_signal,
        name=f"{config_label}_{target_var}_{JET_FAKE_MET_WINDOW}_up_data_sig",
    )
    shifted_data_sig_down = _data_sig_from_fakes(
        data=data,
        nominal_background=nominal_background,
        fakes=shifted_fakes_down,
        nonfiducial_signal=nonfiducial_signal,
        name=f"{config_label}_{target_var}_{JET_FAKE_MET_WINDOW}_down_data_sig",
    )
    variations = _unfold_shift_pair(
        plotter=plotter,
        nominal_data_sig=nominal_data_sig,
        shifted_data_sig_up=shifted_data_sig_up,
        shifted_data_sig_down=shifted_data_sig_down,
        response=response,
        nominal_truth=nominal_truth,
        iterations=iterations,
        lumi=lumi,
        name_prefix=f"{config_label}_{target_var}_{JET_FAKE_MET_WINDOW}",
    )
    for hist in (
        shifted_fakes_up,
        shifted_fakes_down,
        shifted_data_sig_up,
        shifted_data_sig_down,
    ):
        measured_analysis.histograms[hist.GetName()] = hist
    for variation in variations:
        for hist in (
            variation.nominal_unfolded,
            variation.shifted_unfolded,
            variation.uncertainty,
            variation.relative_uncertainty,
        ):
            measured_analysis.histograms[hist.GetName()] = hist
    return FakeSourceSystematic(
        name=JET_FAKE_MET_WINDOW,
        shifted_fakes_up=shifted_fakes_up,
        shifted_fakes_down=shifted_fakes_down,
        shifted_data_sig_up=shifted_data_sig_up,
        shifted_data_sig_down=shifted_data_sig_down,
        variations=variations,
    )


def build_tau_width_fake_systematic(
    *,
    measured_analysis: Analysis,
    plotter: Analysis,
    config_label: str,
    wp: str,
    fake_control_region: FakeControlRegion,
    fakes_source: str,
    target_var: str,
    data: ROOT.TH1,
    nominal_background: ROOT.TH1,
    nonfiducial_signal: ROOT.TH1,
    nominal_data_sig: ROOT.TH1,
    nominal_fakes: ROOT.TH1,
    three_prong_fakes: ROOT.TH1,
    response: ResponseComponents,
    nominal_truth: ROOT.TH1,
    iterations: tuple[int, ...],
    width_variable: str,
    lumi: float,
) -> TauWidthFakeSystematic:
    """Build the 1-prong tau-width fake-source composition systematic."""
    one_prong_prefix = f"{config_label}_{wp}_1prong{fake_control_region.output_tag}"
    one_prong_ff = measured_analysis.histograms[f"{one_prong_prefix}_{fakes_source}_FF"]
    one_prong_fake_cr_fail = (
        f"{config_label}_{wp}_1prong_{fake_control_region.selection_tag}_failID"
    )
    one_prong_sr_pass = f"{config_label}_{wp}_1prong_SR_passID"
    one_prong_sr_fail = f"{config_label}_{wp}_1prong_SR_failID"
    true_one_prong_sr_fail = f"trueTau_{one_prong_sr_fail}"

    low_width = fake_like_histogram(
        measured_analysis,
        width_variable,
        one_prong_fake_cr_fail,
        name=f"{config_label}_{width_variable}_1prong_lowmet_fake_like",
    )
    target_width = fake_like_histogram(
        measured_analysis,
        width_variable,
        one_prong_sr_fail,
        name=f"{config_label}_{width_variable}_1prong_sr_fake_like",
    )
    low_shape = positive_unit_shape(
        low_width,
        f"{config_label}_{width_variable}_1prong_lowmet_shape",
    )
    target_shape = positive_unit_shape(
        target_width,
        f"{config_label}_{width_variable}_1prong_sr_shape",
    )
    width_ratio = shape_ratio_histogram(
        low_shape,
        target_shape,
        f"{config_label}_{width_variable}_1prong_application_to_lowmet",
    )
    measured_analysis.histograms[width_ratio.GetName()] = width_ratio

    shifted_one_prong_fakes = fill_width_reweighted_fake_prediction_from_factor(
        measured_analysis,
        target_var=target_var,
        sr_pass_selection=one_prong_sr_pass,
        sr_fail_selection=one_prong_sr_fail,
        true_sr_fail_selection=true_one_prong_sr_fail,
        ff_hist=one_prong_ff,
        width_ratio_hist=width_ratio,
        width_variable=width_variable,
        output_name=(
            f"{one_prong_prefix}_{width_variable}_{target_var}_"
            "application_to_lowmet_width_rw"
        ),
        fakes_source=fakes_source,
    )
    shifted_fakes = sum_th1s(shifted_one_prong_fakes, three_prong_fakes)
    shifted_fakes.SetName(
        f"{config_label}_{width_variable}_{target_var}_{JET_FAKE_TAU_WIDTH_COMPOSITION}_fakes"
    )
    shifted_fakes.SetDirectory(0)

    shifted_background = sum_th1s(nominal_background, shifted_fakes)
    shifted_data_sig = data - shifted_background - nonfiducial_signal
    shifted_data_sig.SetName(
        f"{config_label}_{width_variable}_{target_var}_"
        f"{JET_FAKE_TAU_WIDTH_COMPOSITION}_data_sig"
    )
    shifted_data_sig.SetDirectory(0)

    variations = []
    for iter_count in iterations:
        nominal_unfolded, _ = unfold_histogram(
            plotter,
            nominal_data_sig,
            response,
            iter_count,
        )
        shifted_unfolded, _ = unfold_histogram(
            plotter,
            shifted_data_sig,
            response,
            iter_count,
        )
        nominal_unfolded = scale_and_crop_unfolded(
            nominal_unfolded,
            nominal_truth,
            (
                f"{config_label}_{width_variable}_{target_var}_{iter_count}iter_"
                f"{JET_FAKE_TAU_WIDTH_COMPOSITION}_nominal_reference"
            ),
            lumi,
        )
        shifted_unfolded = scale_and_crop_unfolded(
            shifted_unfolded,
            nominal_truth,
            (
                f"{config_label}_{width_variable}_{target_var}_{iter_count}iter_"
                f"{JET_FAKE_TAU_WIDTH_COMPOSITION}_shifted"
            ),
            lumi,
        )
        uncertainty, relative_uncertainty = build_unfolded_envelope(
            nominal_unfolded,
            shifted_unfolded,
            name=(
                f"{config_label}_{width_variable}_{target_var}_{iter_count}iter_"
                f"{JET_FAKE_TAU_WIDTH_COMPOSITION}_uncertainty"
            ),
        )
        for hist in (shifted_fakes, shifted_data_sig, uncertainty, relative_uncertainty):
            measured_analysis.histograms[hist.GetName()] = hist
        variations.append(
            UnfoldedSystematicVariation(
                iter_count=iter_count,
                nominal_unfolded=nominal_unfolded,
                shifted_unfolded=shifted_unfolded,
                uncertainty=uncertainty,
                relative_uncertainty=relative_uncertainty,
            )
        )

    # Keep these nominal inputs alive in the same histogram registry as the shifted outputs.
    measured_analysis.histograms[f"{config_label}_{target_var}_nominal_fakes_reference"] = (
        nominal_fakes
    )
    return TauWidthFakeSystematic(
        name=JET_FAKE_TAU_WIDTH_COMPOSITION,
        width_variable=width_variable,
        shifted_fakes=shifted_fakes,
        shifted_data_sig=shifted_data_sig,
        width_ratio=width_ratio,
        variations=variations,
    )
