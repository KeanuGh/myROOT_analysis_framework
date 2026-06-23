import re
from pathlib import Path

import numpy as np
import ROOT
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples, signal_sample
from shadow_unfold.fakes import (
    fake_like_histogram,
    fill_width_reweighted_fake_prediction_from_factor,
    positive_unit_shape,
    shape_ratio_histogram,
)
from shadow_unfold.models import FakeControlRegion, ResponseComponents, ShadowConfig

from src.analysis import Analysis
from src.cutting import Cut
from src.datasetbuilder import LUMI_YEAR
from src.fakes import fill_fake_predictions_from_factor
from src.histogram import Histogram1D
from src.unfolding import closure_metrics, scale_and_crop_unfolded, unfold_histogram
from utils import ROOT_utils
from utils.helper_functions import smart_join
from utils.plotting_tools import Hist2dOpts, PlotKwargs
from utils.ROOT_utils import sum_th1s
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
        # 200,
        250,
        # 300,
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
DO_FULL_SYSTEMATICS = False  # Enable full systematic response variations; slow final-mode run.
RUN_FAKE_WIDTH_SYSTEMATIC = False  # Propagate the validated 1-prong tau-width fake systematic.
FAKE_WIDTH_VARIABLE = "TauTrackWidthPt1000PV"

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


# FAKE CONTROL REGIONS
# ========================================================================
# Original thesis-style fake control region. Uncomment this block and comment
# out the low-MET block below to compare against the current baseline.
# FAKE_CONTROL_REGION = FakeControlRegion(
#     selection_tag="CR",
#     output_tag="",
#     shared_across_configs=False,
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
    shared_across_configs=True,
    cuts=(
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        Cut(r"Low-$E_T^{\mathrm{miss}}$ fake-enriched region", "MET_met < 100"),
    ),
)


# MODELS & CONFIGURATION
# ========================================================================
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
)


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
    plotter.logger.info(
        "FAKE_CONTROL_REGION selection tag = %s", FAKE_CONTROL_REGION.selection_tag
    )
    plotter.logger.info("DO_FULL_SYSTEMATICS = %s", DO_FULL_SYSTEMATICS)
    plotter.logger.info("RUN_FAKE_WIDTH_SYSTEMATIC = %s", RUN_FAKE_WIDTH_SYSTEMATIC)
    if DO_FULL_SYSTEMATICS:
        plotter.logger.info(
            "Full systematics mode enabled; missing shadow variations will fail loudly."
        )

    # SELECTION BUILDING
    # ========================================================================
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    response_selections: dict[str, list[Cut]] = {}
    width_vars = {FAKE_WIDTH_VARIABLE} if RUN_FAKE_WIDTH_SYSTEMATIC else set()
    selection_binnings: dict[str, dict[str, np.ndarray]] = {
        "": {
            **BINNINGS,
            **{
                width_var: np.array(
                    [
                        0.0,
                        0.005,
                        0.01,
                        0.015,
                        0.02,
                        0.025,
                        0.03,
                        0.04,
                        0.05,
                        0.07,
                        0.1,
                        0.15,
                        0.25,
                        0.35,
                    ],
                    dtype="double",
                )
                for width_var in width_vars
            },
        }
        if RUN_FAKE_WIDTH_SYSTEMATIC
        else BINNINGS
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
            mc_selections[f"trueTau_{names['sr_pass']}"] = data_selections[names["sr_pass"]] + [
                PASS_TRUETAU
            ]
            mc_selections[f"trueTau_{names['sr_fail']}"] = data_selections[names["sr_fail"]] + [
                PASS_TRUETAU
            ]
            mc_selections[f"trueTau_{names['fake_cr_pass']}"] = data_selections[
                names["fake_cr_pass"]
            ] + [PASS_TRUETAU]
            mc_selections[f"trueTau_{names['fake_cr_fail']}"] = data_selections[
                names["fake_cr_fail"]
            ] + [PASS_TRUETAU]

        response_selections[truth_selection] = truth_cuts
        response_selections[reco_selection] = reco_sr_cuts + [PASS_MEDIUM]
        response_selections[truth_reco_selection] = truth_cuts + reco_sr_cuts + [PASS_MEDIUM]

        config_binnings = dict(selection_binnings[""])
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
            *width_vars,
        },
        import_missing_columns_as_nan=True,
        histogram_vars=set(VARS) | width_vars,
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
            f"{var}_{variable_data[var]['truth']}": Hist2dOpts(
                var,
                variable_data[var]["truth"],
                "reco_weight",
            )
            for var in VARS
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
            variable_data[var]["truth"],
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection="no_shadow_bin_truth_tau",
        )
        for var in VARS
    }

    closure_rows: list[tuple[str, str, int, float, float, float]] = []
    fake_budget_rows: list[
        tuple[str, str, float, float, float, float, float, float, float, float]
    ] = []
    fake_width_rows: list[
        tuple[str, str, str, int, float, float, float, float, float, float]
    ] = []
    fake_factor_cache: dict[tuple[str, str, tuple[float, ...]], ROOT.TH1] = {}

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
            ff_bins = measured_analysis[measured_analysis.data_sample].get_binnings(
                FAKES_SOURCE,
                fake_cr_pass,
            )["bins"]
            ff_bin_key = tuple(float(bin_edge) for bin_edge in ff_bins)

            # The nominal fake estimate follows the thesis method: split by tau prong
            # and then sum the 1-prong and 3-prong predictions.
            for prong in (1, 3):
                prong_fakes_name = (
                    f"{config.label}_{WP}_{prong}prong{FAKE_CONTROL_REGION.output_tag}"
                )
                prong_fake_cr_pass = (
                    f"{config.label}_{WP}_{prong}prong_{FAKE_CONTROL_REGION.selection_tag}_passID"
                )
                prong_fake_cr_fail = (
                    f"{config.label}_{WP}_{prong}prong_{FAKE_CONTROL_REGION.selection_tag}_failID"
                )
                prong_sr_pass = f"{config.label}_{WP}_{prong}prong_SR_passID"
                prong_sr_fail = f"{config.label}_{WP}_{prong}prong_SR_failID"
                true_prong_sr_fail = f"trueTau_{prong_sr_fail}"
                cache_key = (
                    FAKE_CONTROL_REGION.selection_tag,
                    f"{prong}prong",
                    ff_bin_key,
                )

                if FAKE_CONTROL_REGION.shared_across_configs and cache_key in fake_factor_cache:
                    cached_ff = fake_factor_cache[cache_key]
                    current_ff = cached_ff.Clone(f"{prong_fakes_name}_{FAKES_SOURCE}_FF")
                    current_ff.SetDirectory(0)
                    measured_analysis.histograms[current_ff.GetName()] = current_ff
                    plotter.logger.info(
                        "Reusing cached %s-prong fake factor for %s from '%s'.",
                        prong,
                        config.label,
                        FAKE_CONTROL_REGION.selection_tag,
                    )
                    fill_fake_predictions_from_factor(
                        measured_analysis,
                        target_vars=vars_for_config,
                        sr_pass_selection=prong_sr_pass,
                        sr_fail_selection=prong_sr_fail,
                        true_sr_fail_selection=true_prong_sr_fail,
                        ff_hist=current_ff,
                        output_prefix=prong_fakes_name,
                        fakes_source=FAKES_SOURCE,
                        systematic=NOMINAL_NAME,
                    )
                else:
                    measured_analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        vars_for_config,
                        prong_fake_cr_pass,
                        prong_fake_cr_fail,
                        prong_sr_pass,
                        prong_sr_fail,
                        f"trueTau_{prong_fake_cr_pass}",
                        f"trueTau_{prong_fake_cr_fail}",
                        f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                        true_prong_sr_fail,
                        name=prong_fakes_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )
                    if FAKE_CONTROL_REGION.shared_across_configs:
                        cached_ff = measured_analysis.histograms[
                            f"{prong_fakes_name}_{FAKES_SOURCE}_FF"
                        ].Clone(f"cached_{FAKE_CONTROL_REGION.selection_tag}_{prong}prong_FF")
                        cached_ff.SetDirectory(0)
                        fake_factor_cache[cache_key] = cached_ff

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
            prong_fakes_prefix = (
                f"{config.label}_{WP}_{{prong}}prong{FAKE_CONTROL_REGION.output_tag}"
            )
            prong_fakes = [
                measured_analysis.histograms[
                    prong_fakes_prefix.format(prong=prong) + f"_{var}_fakes_bkg_{FAKES_SOURCE}_src"
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

            # 1-PRONG FAKE-SOURCE COMPOSITION SYSTEMATIC
            # ----------------------------------------------------------------
            if RUN_FAKE_WIDTH_SYSTEMATIC:
                one_prong_prefix = f"{config.label}_{WP}_1prong{FAKE_CONTROL_REGION.output_tag}"
                one_prong_ff = measured_analysis.histograms[
                    f"{one_prong_prefix}_{FAKES_SOURCE}_FF"
                ]
                one_prong_fake_cr_fail = (
                    f"{config.label}_{WP}_1prong_"
                    f"{FAKE_CONTROL_REGION.selection_tag}_failID"
                )
                one_prong_sr_pass = f"{config.label}_{WP}_1prong_SR_passID"
                one_prong_sr_fail = f"{config.label}_{WP}_1prong_SR_failID"
                true_one_prong_sr_fail = f"trueTau_{one_prong_sr_fail}"
                three_prong_fakes = prong_fakes[1]

                width_var = FAKE_WIDTH_VARIABLE
                low_width = fake_like_histogram(
                    measured_analysis,
                    width_var,
                    one_prong_fake_cr_fail,
                    name=f"{config.label}_{width_var}_1prong_lowmet_fake_like",
                )
                target_width = fake_like_histogram(
                    measured_analysis,
                    width_var,
                    one_prong_sr_fail,
                    name=f"{config.label}_{width_var}_1prong_sr_fake_like",
                )
                low_shape = positive_unit_shape(
                    low_width,
                    f"{config.label}_{width_var}_1prong_lowmet_shape",
                )
                target_shape = positive_unit_shape(
                    target_width,
                    f"{config.label}_{width_var}_1prong_sr_shape",
                )
                width_ratio = shape_ratio_histogram(
                    low_shape,
                    target_shape,
                    f"{config.label}_{width_var}_1prong_application_to_lowmet",
                )
                measured_analysis.histograms[width_ratio.GetName()] = width_ratio

                shifted_one_prong_fakes = fill_width_reweighted_fake_prediction_from_factor(
                    measured_analysis,
                    target_var=var,
                    sr_pass_selection=one_prong_sr_pass,
                    sr_fail_selection=one_prong_sr_fail,
                    true_sr_fail_selection=true_one_prong_sr_fail,
                    ff_hist=one_prong_ff,
                    width_ratio_hist=width_ratio,
                    width_variable=width_var,
                    output_name=(
                        f"{one_prong_prefix}_{width_var}_{var}_"
                        "application_to_lowmet_width_rw"
                    ),
                    fakes_source=FAKES_SOURCE,
                )
                shifted_fakes = sum_th1s(shifted_one_prong_fakes, three_prong_fakes)
                shifted_fakes.SetName(f"{config.label}_{width_var}_{var}_width_shifted_fakes")
                shifted_fakes.SetDirectory(0)

                shifted_background = sum_th1s(*(backgrounds + [shifted_fakes]))
                shifted_data_sig = data - shifted_background - nonfiducial_signal
                shifted_data_sig.SetName(f"{config.label}_{width_var}_{var}_width_shifted_data_sig")
                shifted_data_sig.SetDirectory(0)

                for iter_count in ITERATIONS:
                    nominal_unfolded, _ = unfold_histogram(
                        plotter,
                        data_sig,
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
                        f"{config.label}_{width_var}_{var}_{iter_count}iter_nominal_width_reference",
                        LUMI,
                    )
                    shifted_unfolded = scale_and_crop_unfolded(
                        shifted_unfolded,
                        nominal_truth,
                        f"{config.label}_{width_var}_{var}_{iter_count}iter_width_shifted_unfolded",
                        LUMI,
                    )

                    fake_width_rows.append(
                        (
                            config.label,
                            var,
                            width_var,
                            iter_count,
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
                        [truth, nominal_unfolded, shifted_unfolded],
                        label=[
                            "Truth MC",
                            "Nominal unfolded data",
                            f"{width_var} shifted unfolded data",
                        ],
                        colour=["r", "k", "tab:orange"],
                        histstyle=["step", "errorbar", "errorbar"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            f"{width_var} fake-source systematic",
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
                        ratio_label="Shift / nominal",
                        ratio_axlim=(0.5, 1.5),
                        label_params={"llabel": "", "loc": 1},
                        filename=(
                            f"{config.label}_{var}_{width_var}_"
                            f"{iter_count}iter_fake_width_shift.png"
                        ),
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
                    LUMI,
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
                        LUMI,
                    )
                    unfolded_down = scale_and_crop_unfolded(
                        unfolded_down,
                        nominal_truth,
                        f"{config.label}_{var}_{sys_name}_down_response_sys",
                        LUMI,
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
                    LUMI,
                )
                signal_unfolded = scale_and_crop_unfolded(
                    signal_unfolded,
                    nominal_truth,
                    f"{config.label}_{var}_{iter_count}iter_signal_unfolded",
                    LUMI,
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

    # SUMMARY OUTPUT
    # ========================================================================
    summary_lines = [
        "# Variable-specific shadow-bin unfolding closure summary",
        "",
        f"DO_FULL_SYSTEMATICS: `{DO_FULL_SYSTEMATICS}`",
        f"FAKE_CONTROL_REGION: `{FAKE_CONTROL_REGION.selection_tag}`",
        f"FAKE_FACTOR_SOURCE: `{FAKES_SOURCE}`",
        "FAKE_MODEL: `prong-split`",
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
    summary_lines.extend(
        [
            "",
            "## Pre-unfolding budget",
            "",
            "`data_sig` is the nominal unfolded input before unfolding:",
            "`data - prompt backgrounds - prong-split fake estimate - nonfiducial signal`.",
            "",
            "| Configuration | Variable | Data | Prompt bkg | Fakes | Nonfid signal | "
            "Data sig | Data sig, no fakes | Fid reco signal | Fid reco / data sig |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
        summary_lines.append(
            f"| {config_label} | {var} | {data_integral:.3f} | {prompt_bkg:.3f} | "
            f"{fakes_integral:.3f} | {nonfid:.3f} | {data_sig:.3f} | "
            f"{data_sig_no_fake:.3f} | {fid_reco:.3f} | {fid_over_data_sig:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "## 1-prong tau-width fake-source systematic",
            "",
            f"Enabled: `{RUN_FAKE_WIDTH_SYSTEMATIC}`",
            "",
            "When enabled, this applies the validated `application_to_lowmet` tau-width "
            "shape reweighting to the 1-prong fake component only. The 3-prong fake "
            "component is left nominal because the validation target there is not "
            "physically usable after nonfake subtraction.",
            "",
            "| Configuration | Variable | Width proxy | Iterations | Nominal fakes | Shifted fakes | "
            "Nominal data sig | Shifted data sig | Fid reco / nominal data sig | "
            "Fid reco / shifted data sig |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if fake_width_rows:
        for (
            config_label,
            var,
            width_var,
            iter_count,
            nominal_fakes,
            shifted_fakes,
            nominal_data_sig,
            shifted_data_sig,
            nominal_fid_over_data,
            shifted_fid_over_data,
        ) in fake_width_rows:
            summary_lines.append(
                f"| {config_label} | {var} | `{width_var}` | {iter_count} | "
                f"{nominal_fakes:.3f} | {shifted_fakes:.3f} | {nominal_data_sig:.3f} | "
                f"{shifted_data_sig:.3f} | {nominal_fid_over_data:.3f} | "
                f"{shifted_fid_over_data:.3f} |"
            )
    else:
        summary_lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    summary_path = output_root / "closure_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    plotter.logger.info("Saved closure summary to %s", summary_path)

    if not load_measured_analysis_hists:
        measured_analysis.save_hists()

    plotter.logger.info("DONE.")
