import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples

from src.analysis import Analysis
from src.cutting import Cut
from utils.helper_functions import smart_join
from utils.ROOT_utils import sum_th1s
from utils.variable_names import variable_data

YEAR = 2017
WP = "medium"
VARS = ("MTW",)
SHADOW_BINS = (200, 300)
FAKES_SOURCE = "TauPt"
FAKES_SOURCE_CROSSCHECK = "MTW"
COMPONENT_COLOURS = {
    "data": "k",
    "wtaunu_had": "tab:blue",
    "wtaunu_lep": "tab:cyan",
    "wlnu": "tab:green",
    "zll": "tab:orange",
    "top": "tab:red",
    "diboson": "tab:purple",
}
LOAD_SAVED_HISTS = True  # Reuse saved ROOT histograms, generating only missing pieces.
REUSE_SHADOW_UNFOLD_MEASURED_OUTPUT = True  # Read matching dataset/hist caches from main script.
RUN_EVENT_LEVEL_PRONG_DIAGNOSTICS = (
    False  # One wtaunu_had pass for weighted/unweighted prong checks.
)
PLOT_CONTROL_REGION_COMPOSITION = False  # Plot data and nonfake MC in fake-factor regions.
PLOT_PASS_ID_VALIDATION = False  # Plot pass-ID data-minus-nonfake against fake prediction.
PLOT_FAKE_FACTORS = False  # Plot the 1-prong and 3-prong fake factors.
PLOT_NONFAKE_COMPONENTS = False  # Plot per-sample nonfake MC in the pass-ID signal region.
RUN_FAKE_SOURCE_CROSSCHECK = False  # Expensive MTW-source diagnostic; enable only when needed.
RUN_FULL_FAKE_TRANSFER_VALIDATION = True  # Run cache-only and sideband fake-transfer checks.
RUN_SIDE_BAND_TRANSFER_EVENT_LOOPS = True  # Build new sideband hists when not already cached.
PLOT_FAKE_TRANSFER_VALIDATION = False  # Optional transfer-validation plots.
TRANSFER_VALIDATION_CONFIGS = ("MTW_shadow_bin_300",)  # Keep smoke tests to one shadow setup.
TRANSFER_VALIDATION_TESTS = ("met_cr_split",)  # Directly test the observed MET-transfer issue.
TRANSFER_VALIDATION_PRONGS = (1, 3)  # Check both prongs without broadening the phase space.
RUN_TAUPT_MET_2D_FEASIBILITY_CHECK = True  # Document whether a true 2D FF is derivable here.
PRONG_WEIGHT_LABELS = ("raw", "mcWeight", "weight", "truth_weight", "reco_weight")
DSID_DIAGNOSTIC_STAGE_LABELS = (
    "reco preselection, truth matched",
    "medium pass-ID, truth matched",
)
WEIGHT_DECOMPOSITION_LABELS = (
    "weight",
    "weight / TauRecoSF",
    "weight / selectionSF",
    "weight / TriggerSF",
    "weight / prwWeight",
    "weight / jvtSF",
    "weight / fjvtSF",
)
WEIGHT_DECOMPOSITION_BRANCHES = {
    "TauRecoSF",
    "selectionSF",
    "TriggerSF",
    "prwWeight",
    "jvtSF",
    "fjvtSF",
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


# MODELS & CONFIGURATION
# ========================================================================


@dataclass(frozen=True)
class ShadowConfig:
    label: str
    unfolded_var: str | None
    mtw_min: float
    taupt_min: float
    met_min: float


@dataclass(frozen=True)
class TransferTest:
    name: str
    label: str
    derivation_extra_cuts: tuple[Cut, ...]
    validation_extra_cuts: tuple[Cut, ...]
    is_independent: bool


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
        for threshold in SHADOW_BINS
    ),
)


def hist_integral(hist: ROOT.TH1) -> float:
    return float(hist.Integral())


def fake_factor_bin_health(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
) -> tuple[int, int]:
    denominator_total = sum(
        abs(denominator.GetBinContent(bin_idx))
        for bin_idx in range(1, denominator.GetNbinsX() + 1)
    )
    tiny_threshold = max(10.0, 0.01 * denominator_total)
    negative_numerator_bins = 0
    tiny_denominator_bins = 0
    for bin_idx in range(1, denominator.GetNbinsX() + 1):
        if numerator.GetBinContent(bin_idx) < 0:
            negative_numerator_bins += 1
        denominator_value = denominator.GetBinContent(bin_idx)
        if denominator_value <= 0 or abs(denominator_value) < tiny_threshold:
            tiny_denominator_bins += 1
    return negative_numerator_bins, tiny_denominator_bins


def source_prediction_with_modified_fake_factor(
    numerator: ROOT.TH1,
    denominator: ROOT.TH1,
    sr_fail_fake_like: ROOT.TH1,
    treatment: str,
) -> float:
    first_bin_numerator = 0.0
    first_bin_denominator = 0.0
    if treatment == "merge 170-250 GeV":
        for bin_idx in range(1, denominator.GetNbinsX() + 1):
            bin_low = denominator.GetBinLowEdge(bin_idx)
            bin_high = bin_low + denominator.GetBinWidth(bin_idx)
            if bin_low >= 170 and bin_high <= 250:
                first_bin_numerator += numerator.GetBinContent(bin_idx)
                first_bin_denominator += denominator.GetBinContent(bin_idx)
        merged_fake_factor = (
            first_bin_numerator / first_bin_denominator if first_bin_denominator != 0 else 0.0
        )

    predicted = 0.0
    for bin_idx in range(1, denominator.GetNbinsX() + 1):
        den = denominator.GetBinContent(bin_idx)
        fake_factor = numerator.GetBinContent(bin_idx) / den if den != 0 else 0.0
        bin_low = denominator.GetBinLowEdge(bin_idx)
        bin_high = bin_low + denominator.GetBinWidth(bin_idx)
        if treatment == "floor negative bins":
            fake_factor = max(fake_factor, 0.0)
        elif treatment == "merge 170-250 GeV" and bin_low >= 170 and bin_high <= 250:
            fake_factor = merged_fake_factor
        predicted += sr_fail_fake_like.GetBinContent(bin_idx) * fake_factor
    return predicted


def use_transfer_validation_config(config: ShadowConfig) -> bool:
    return config.label in TRANSFER_VALIDATION_CONFIGS


def use_transfer_validation_test(transfer_test: TransferTest) -> bool:
    return transfer_test.name in TRANSFER_VALIDATION_TESTS


if __name__ == "__main__":
    # SETUP
    # ========================================================================
    output_root = Path(__file__).absolute().parent.parent.parent / "outputs" / Path(__file__).stem
    validator = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label=Path(__file__).stem,
        output_dir=output_root,
        log_level=10,
        log_out="both",
    )
    validator.logger.info("Starting validate_shadow_fakes.py")
    validator.logger.info("LOAD_SAVED_HISTS = %s", LOAD_SAVED_HISTS)
    validator.logger.info(
        "REUSE_SHADOW_UNFOLD_MEASURED_OUTPUT = %s", REUSE_SHADOW_UNFOLD_MEASURED_OUTPUT
    )
    validator.logger.info(
        "RUN_EVENT_LEVEL_PRONG_DIAGNOSTICS = %s", RUN_EVENT_LEVEL_PRONG_DIAGNOSTICS
    )
    validator.logger.info("PLOT_CONTROL_REGION_COMPOSITION = %s", PLOT_CONTROL_REGION_COMPOSITION)
    validator.logger.info("PLOT_PASS_ID_VALIDATION = %s", PLOT_PASS_ID_VALIDATION)
    validator.logger.info("PLOT_FAKE_FACTORS = %s", PLOT_FAKE_FACTORS)
    validator.logger.info("PLOT_NONFAKE_COMPONENTS = %s", PLOT_NONFAKE_COMPONENTS)
    validator.logger.info("RUN_FAKE_SOURCE_CROSSCHECK = %s", RUN_FAKE_SOURCE_CROSSCHECK)
    validator.logger.info(
        "RUN_FULL_FAKE_TRANSFER_VALIDATION = %s", RUN_FULL_FAKE_TRANSFER_VALIDATION
    )
    validator.logger.info(
        "RUN_SIDE_BAND_TRANSFER_EVENT_LOOPS = %s", RUN_SIDE_BAND_TRANSFER_EVENT_LOOPS
    )
    validator.logger.info("PLOT_FAKE_TRANSFER_VALIDATION = %s", PLOT_FAKE_TRANSFER_VALIDATION)
    validator.logger.info("TRANSFER_VALIDATION_CONFIGS = %s", TRANSFER_VALIDATION_CONFIGS)
    validator.logger.info("TRANSFER_VALIDATION_TESTS = %s", TRANSFER_VALIDATION_TESTS)
    validator.logger.info("TRANSFER_VALIDATION_PRONGS = %s", TRANSFER_VALIDATION_PRONGS)
    validator.logger.info(
        "RUN_TAUPT_MET_2D_FEASIBILITY_CHECK = %s", RUN_TAUPT_MET_2D_FEASIBILITY_CHECK
    )

    # SELECTION BUILDING
    # ========================================================================
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    selection_binnings: dict[str, dict[str, np.ndarray]] = {"": BINNINGS}
    transfer_data_selections: dict[str, list[Cut]] = {}
    transfer_mc_selections: dict[str, list[Cut]] = {}
    transfer_selection_binnings: dict[str, dict[str, np.ndarray]] = {"": BINNINGS}
    transfer_tests_by_config: dict[str, tuple[TransferTest, ...]] = {}

    for config in CONFIGS:
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        cr_pass = f"{config.label}_{WP}_CR_passID"
        cr_fail = f"{config.label}_{WP}_CR_failID"
        prong_names = {
            prong: {
                "sr_pass": f"{config.label}_{WP}_{prong}prong_SR_passID",
                "sr_fail": f"{config.label}_{WP}_{prong}prong_SR_failID",
                "cr_pass": f"{config.label}_{WP}_{prong}prong_CR_passID",
                "cr_fail": f"{config.label}_{WP}_{prong}prong_CR_failID",
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
        reco_cr_cuts = [
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
            PASS_ETA,
            Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
            Cut(r"$E_T^{\mathrm{miss}}$ control region", f"MET_met < {config.met_min:g}"),
        ]
        transfer_tests = [
            TransferTest(
                "met_cr_split",
                "derive MET < 120, validate 120 <= MET < 170",
                (
                    Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
                    Cut(r"$E_T^{\mathrm{miss}} < 120$", "MET_met < 120"),
                ),
                (
                    Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
                    Cut(
                        r"$120 <= E_T^{\mathrm{miss}} < 170$",
                        "(MET_met >= 120) && (MET_met < 170)",
                    ),
                ),
                True,
            ),
            TransferTest(
                "sr_met_proxy",
                "derive MET < 170, validate 170 <= MET < 250",
                (
                    Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
                    Cut(r"$E_T^{\mathrm{miss}} < 170$", "MET_met < 170"),
                ),
                (
                    Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
                    Cut(
                        r"$170 <= E_T^{\mathrm{miss}} < 250$",
                        "(MET_met >= 170) && (MET_met < 250)",
                    ),
                ),
                False,
            ),
        ]
        if config.unfolded_var == "MTW":
            transfer_tests.append(
                TransferTest(
                    "mtw_cr_split",
                    "derive shadow MTW CR, validate nominal MTW sideband",
                    (
                        Cut(
                            r"shadow $m_T^W$ sideband",
                            f"(MTW >= {config.mtw_min:g}) && (MTW < 350)",
                        ),
                        Cut(r"$E_T^{\mathrm{miss}} < 170$", "MET_met < 170"),
                    ),
                    (
                        Cut(r"$m_T^W >= 350$", "MTW >= 350"),
                        Cut(r"$E_T^{\mathrm{miss}} < 170$", "MET_met < 170"),
                    ),
                    True,
                )
            )
        transfer_tests_by_config[config.label] = tuple(transfer_tests)
        truth_cuts = [
            PASS_TRUTH,
            TRUTH_HAD_TAU,
            Cut(
                r"Truth fiducial region",
                f"(VisTruthTauPt > {config.taupt_min:g}) && "
                f"(TruthMTW > {config.mtw_min:g}) && "
                f"(TruthNeutrinoPt > {config.met_min:g})"
                r"&& (((abs(VisTruthTauEta) < 1.37) || (1.52 < abs(VisTruthTauEta))) "
                r"&& (abs(VisTruthTauEta) < 2.47))",
            ),
        ]

        data_selections[sr_pass] = reco_sr_cuts + [PASS_MEDIUM]
        data_selections[sr_fail] = reco_sr_cuts + [FAIL_MEDIUM]
        data_selections[cr_pass] = reco_cr_cuts + [PASS_MEDIUM]
        data_selections[cr_fail] = reco_cr_cuts + [FAIL_MEDIUM]

        for prong, names in prong_names.items():
            pass_prong = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
            data_selections[names["sr_pass"]] = data_selections[sr_pass] + [pass_prong]
            data_selections[names["sr_fail"]] = data_selections[sr_fail] + [pass_prong]
            data_selections[names["cr_pass"]] = data_selections[cr_pass] + [pass_prong]
            data_selections[names["cr_fail"]] = data_selections[cr_fail] + [pass_prong]

        for selection in (sr_pass, sr_fail, cr_pass, cr_fail):
            mc_selections[selection] = data_selections[selection]
            mc_selections[f"trueTau_{selection}"] = data_selections[selection] + [PASS_TRUETAU]

        for names in prong_names.values():
            for selection in names.values():
                mc_selections[selection] = data_selections[selection]
                mc_selections[f"trueTau_{selection}"] = data_selections[selection] + [PASS_TRUETAU]

        for prong in (1, 3):
            mc_selections[f"{config.label}_wtaunu_had_truth_{prong}prong"] = truth_cuts + [
                Cut(f"truth {prong}-prong", f"TruthTau_nChargedTracks == {prong}")
            ]
            mc_selections[f"{config.label}_wtaunu_had_reco_preselection_{prong}prong"] = (
                reco_sr_cuts
                + [
                    PASS_TRUETAU,
                    Cut(f"reco {prong}-prong", f"TauNCoreTracks == {prong}"),
                ]
            )

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
        elif config.unfolded_var == "TauPt":
            taupt_bins = np.array(
                [config.taupt_min, 170, 200, 250, 300, 350, 425, 500, 600, 1000],
                dtype="double",
            )
            config_binnings["TauPt"] = taupt_bins

        for selection in (
            sr_pass,
            sr_fail,
            cr_pass,
            cr_fail,
            f"trueTau_{sr_pass}",
            f"trueTau_{sr_fail}",
            f"trueTau_{cr_pass}",
            f"trueTau_{cr_fail}",
        ):
            selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
        for names in prong_names.values():
            for selection in names.values():
                selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
                selection_binnings[rf"^{re.escape(f'trueTau_{selection}')}$"] = config_binnings
        for prong in (1, 3):
            for selection in (
                f"{config.label}_wtaunu_had_truth_{prong}prong",
                f"{config.label}_wtaunu_had_reco_preselection_{prong}prong",
            ):
                selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings

        if RUN_FULL_FAKE_TRANSFER_VALIDATION and use_transfer_validation_config(config):
            transfer_base_cuts = [
                PASS_RECO_PRESELECTION,
                Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
                PASS_ETA,
            ]
            for transfer_test in transfer_tests_by_config[config.label]:
                if not use_transfer_validation_test(transfer_test):
                    continue
                for prong in TRANSFER_VALIDATION_PRONGS:
                    pass_prong = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
                    selection_prefix = f"{config.label}_{transfer_test.name}_{WP}_{prong}prong"
                    transfer_data_selections[f"{selection_prefix}_derive_passID"] = (
                        transfer_base_cuts
                        + list(transfer_test.derivation_extra_cuts)
                        + [PASS_MEDIUM, pass_prong]
                    )
                    transfer_data_selections[f"{selection_prefix}_derive_failID"] = (
                        transfer_base_cuts
                        + list(transfer_test.derivation_extra_cuts)
                        + [FAIL_MEDIUM, pass_prong]
                    )
                    transfer_data_selections[f"{selection_prefix}_validate_passID"] = (
                        transfer_base_cuts
                        + list(transfer_test.validation_extra_cuts)
                        + [PASS_MEDIUM, pass_prong]
                    )
                    transfer_data_selections[f"{selection_prefix}_validate_failID"] = (
                        transfer_base_cuts
                        + list(transfer_test.validation_extra_cuts)
                        + [FAIL_MEDIUM, pass_prong]
                    )
                    for suffix in (
                        "derive_passID",
                        "derive_failID",
                        "validate_passID",
                        "validate_failID",
                    ):
                        selection = f"{selection_prefix}_{suffix}"
                        transfer_mc_selections[selection] = transfer_data_selections[selection]
                        transfer_mc_selections[f"trueTau_{selection}"] = transfer_data_selections[
                            selection
                        ] + [PASS_TRUETAU]
                        transfer_selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
                        transfer_selection_binnings[rf"^{re.escape(f'trueTau_{selection}')}$"] = (
                            config_binnings
                        )

    # DATAFRAME & HISTOGRAM PRODUCTION
    # ========================================================================
    shadow_unfold_measured_output = (
        Path(__file__).absolute().parent.parent.parent
        / "outputs"
        / "analysis_shadow_unfold"
        / "measured"
    )
    can_reuse_shadow_unfold = (
        LOAD_SAVED_HISTS
        and REUSE_SHADOW_UNFOLD_MEASURED_OUTPUT
        and (shadow_unfold_measured_output / "root" / "data.root").is_file()
    )
    analysis_output_dir = shadow_unfold_measured_output if can_reuse_shadow_unfold else output_root
    if can_reuse_shadow_unfold:
        validator.logger.info("Reusing measured dataset cache from %s", analysis_output_dir)
    else:
        validator.logger.info("Using validation output cache at %s", analysis_output_dir)

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label=Path(__file__).stem,
        output_dir=analysis_output_dir,
        log_level=10,
        log_out="console" if can_reuse_shadow_unfold else "both",
        extract_vars={
            "MTW",
            "TauPt",
            "MET_met",
            "TauEta",
            "mcWeight",
            "weight",
            *WEIGHT_DECOMPOSITION_BRANCHES,
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
            "VisTruthTauPt",
            "VisTruthTauEta",
            "TruthMTW",
            "TruthNeutrinoPt",
            "TruthTau_nChargedTracks",
            "TruthTau_isHadronic",
            "passTruth",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=set(VARS) | {FAKES_SOURCE},
        do_unweighted=True,
        binnings=selection_binnings,
    )
    if can_reuse_shadow_unfold:
        analysis.paths.output_dir = output_root
        analysis.paths.plot_dir = output_root / "plots"
        analysis.paths.root_dir = output_root / "root"
        analysis.paths.latex_dir = output_root / "LaTeX"

    if LOAD_SAVED_HISTS and can_reuse_shadow_unfold:
        loaded_analysis_hists = analysis.load_hists_if_available(
            shadow_unfold_measured_output / "root" / "analysis_shadow_unfold_measured.root"
        )
        if not loaded_analysis_hists:
            loaded_analysis_hists = analysis.load_hists_if_available()
    else:
        loaded_analysis_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available()
    analysis.print_metadata_table(datasets=analysis.mc_samples)

    composition_rows: list[tuple[str, str, str, float, float, float, float]] = []
    nonfake_component_rows: list[tuple[str, str, str, float, float, float]] = []
    taupt_contribution_rows: list[tuple[str, str, float, float, float, float, float, float]] = []
    low_taupt_summary_rows: list[tuple[str, str, float, float, float]] = []
    prong_balance_rows: list[tuple[str, str, float, float, float]] = []
    wtaunu_had_stage_rows: list[tuple[str, str, float, float, float]] = []
    truth_cut_prong_rows: list[tuple[str, str, str, str, float, float, float, float, float]] = []
    reco_prong_weight_rows: list[tuple[str, str, float, float, float, float, float, float, float]] = []
    reco_weight_component_rows: list[tuple[str, str, str, float, float, float, float]] = []
    dsid_prong_weight_rows: list[tuple[str, int, str, str, str, float, float, float, float]] = []
    weight_decomposition_rows: list[tuple[str, str, str, float, float, float, float]] = []
    pass_validation_rows: list[tuple[str, str, float, float, float]] = []
    fake_source_crosscheck_rows: list[tuple[str, str, float, float, float]] = []
    fake_factor_health_rows: list[
        tuple[str, str, str, str, str, float, float, int, int, float, float, float, float]
    ] = []
    negative_bin_rows: list[
        tuple[str, str, str, str, float, float, int, int, float, float, float, float]
    ] = []
    transfer_validation_rows: list[
        tuple[str, str, str, str, str, float, float, int, int, float, float, float, float]
    ] = []
    taupt_met_feasibility_rows: list[tuple[str, str, str, str, str, str]] = []

    # FAKE ESTIMATION
    # ========================================================================
    for config in CONFIGS:
        vars_for_config = VARS if config.unfolded_var is None else (config.unfolded_var,)

        if not loaded_analysis_hists:
            for prong in (1, 3):
                analysis.do_fakes_estimate(
                    FAKES_SOURCE,
                    vars_for_config,
                    f"{config.label}_{WP}_{prong}prong_CR_passID",
                    f"{config.label}_{WP}_{prong}prong_CR_failID",
                    f"{config.label}_{WP}_{prong}prong_SR_passID",
                    f"{config.label}_{WP}_{prong}prong_SR_failID",
                    f"trueTau_{config.label}_{WP}_{prong}prong_CR_passID",
                    f"trueTau_{config.label}_{WP}_{prong}prong_CR_failID",
                    f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                    f"trueTau_{config.label}_{WP}_{prong}prong_SR_failID",
                    name=f"{config.label}_{WP}_{prong}prong",
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )
        if RUN_FAKE_SOURCE_CROSSCHECK and FAKES_SOURCE_CROSSCHECK != FAKES_SOURCE:
            target_vars = tuple(var for var in vars_for_config if var == FAKES_SOURCE_CROSSCHECK)
            if target_vars:
                for prong in (1, 3):
                    crosscheck_name = (
                        f"{config.label}_{WP}_{prong}prong_{FAKES_SOURCE_CROSSCHECK}_src"
                    )
                    analysis.do_fakes_estimate(
                        FAKES_SOURCE_CROSSCHECK,
                        target_vars,
                        f"{config.label}_{WP}_{prong}prong_CR_passID",
                        f"{config.label}_{WP}_{prong}prong_CR_failID",
                        f"{config.label}_{WP}_{prong}prong_SR_passID",
                        f"{config.label}_{WP}_{prong}prong_SR_failID",
                        f"trueTau_{config.label}_{WP}_{prong}prong_CR_passID",
                        f"trueTau_{config.label}_{WP}_{prong}prong_CR_failID",
                        f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                        f"trueTau_{config.label}_{WP}_{prong}prong_SR_failID",
                        name=crosscheck_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=False,
                    )

    # CACHE-ONLY FAKE-FACTOR HEALTH
    # ========================================================================
    if RUN_FULL_FAKE_TRANSFER_VALIDATION:
        validator.logger.info("Running cache-only fake-factor diagnostics...")
        for config in CONFIGS:
            for prong in (1, 3):
                prefix = f"{config.label}_{WP}_{prong}prong"
                numerator = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_numerator"]
                denominator = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_denominator"]
                sr_fail_fake_like = analysis.histograms[
                    f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est"
                ]
                predicted = analysis.histograms[f"{prefix}_MTW_fakes_bkg_{FAKES_SOURCE}_src"]
                negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                    numerator, denominator
                )
                target_data = analysis.get_hist(
                    "MTW",
                    dataset=analysis.data_sample,
                    systematic=NOMINAL_NAME,
                    selection=f"{config.label}_{WP}_{prong}prong_SR_passID",
                    allow_generation=True,
                )
                target_nonfake = sum_th1s(
                    *[
                        analysis.get_hist(
                            "MTW",
                            dataset=mc_sample,
                            systematic=NOMINAL_NAME,
                            selection=f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                            allow_generation=True,
                        )
                        for mc_sample in analysis.mc_samples
                    ]
                )
                target = target_data - target_nonfake
                fake_factor_health_rows.append(
                    (
                        config.label,
                        f"{prong}-prong",
                        FAKES_SOURCE,
                        "nominal CR",
                        "SR pass-ID MTW",
                        hist_integral(numerator),
                        hist_integral(denominator),
                        negative_numerator_bins,
                        tiny_denominator_bins,
                        hist_integral(sr_fail_fake_like),
                        hist_integral(predicted),
                        hist_integral(target),
                        hist_integral(predicted) / hist_integral(target)
                        if hist_integral(target) != 0
                        else float("nan"),
                    )
                )

        config = CONFIGS[0]
        prong = 3
        prefix = f"{config.label}_{WP}_{prong}prong"
        numerator = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_numerator"]
        denominator = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_denominator"]
        sr_fail_fake_like = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est"]
        negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
            numerator, denominator
        )
        target_data = analysis.get_hist(
            FAKES_SOURCE,
            dataset=analysis.data_sample,
            systematic=NOMINAL_NAME,
            selection=f"{config.label}_{WP}_{prong}prong_SR_passID",
            allow_generation=True,
        )
        target_nonfake = sum_th1s(
            *[
                analysis.get_hist(
                    FAKES_SOURCE,
                    dataset=mc_sample,
                    systematic=NOMINAL_NAME,
                    selection=f"trueTau_{config.label}_{WP}_{prong}prong_SR_passID",
                    allow_generation=True,
                )
                for mc_sample in analysis.mc_samples
            ]
        )
        target = target_data - target_nonfake
        for treatment in ("nominal signed", "floor negative bins", "merge 170-250 GeV"):
            predicted = source_prediction_with_modified_fake_factor(
                numerator,
                denominator,
                sr_fail_fake_like,
                treatment,
            )
            negative_bin_rows.append(
                (
                    config.label,
                    f"{prong}-prong",
                    FAKES_SOURCE,
                    treatment,
                    hist_integral(numerator),
                    hist_integral(denominator),
                    negative_numerator_bins,
                    tiny_denominator_bins,
                    hist_integral(sr_fail_fake_like),
                    predicted,
                    hist_integral(target),
                    predicted / hist_integral(target)
                    if hist_integral(target) != 0
                    else float("nan"),
                )
            )

    # SIDE-BAND FAKE-TRANSFER VALIDATION
    # ========================================================================
    if RUN_FULL_FAKE_TRANSFER_VALIDATION:
        sideband_output = output_root / "sideband_transfer"
        sideband_hist_cache = (
            sideband_output / "root" / "validate_shadow_fakes_sideband_transfer.root"
        )
        sideband_dataset_cache_available = (sideband_output / "root").is_dir() and any(
            path.name != sideband_hist_cache.name
            for path in (sideband_output / "root").glob("*.root")
        )
        sideband_hist_cache_available = sideband_hist_cache.is_file()
        run_sideband_event_loops = RUN_SIDE_BAND_TRANSFER_EVENT_LOOPS and (
            not LOAD_SAVED_HISTS or not sideband_hist_cache_available
        )

        if sideband_hist_cache_available and LOAD_SAVED_HISTS:
            validator.logger.info(
                "Using cached sideband transfer histograms from %s", sideband_output
            )
        elif sideband_dataset_cache_available and LOAD_SAVED_HISTS:
            validator.logger.info(
                "Sideband dataset cache exists, but transfer histograms are missing; "
                "running one batched histogram pass."
            )
        elif run_sideband_event_loops:
            validator.logger.info("Running new ROOT sideband transfer selections...")
        else:
            validator.logger.info(
                "Skipping sideband event loops; using cached sideband histograms only."
            )

        sideband_analysis = None
        loaded_sideband_hists = False
        if sideband_dataset_cache_available or run_sideband_event_loops:
            sideband_analysis = Analysis(
                analysis_samples(
                    transfer_mc_selections,
                    data_selections=transfer_data_selections,
                    snapshot=False,
                ),
                year=YEAR,
                rerun=run_sideband_event_loops,
                regen_histograms=run_sideband_event_loops,
                do_systematics=False,
                metadata_cache=DSID_METADATA_CACHE,
                ttree=NOMINAL_NAME,
                analysis_label="validate_shadow_fakes_sideband_transfer",
                output_dir=sideband_output,
                log_level=10,
                log_out="both" if run_sideband_event_loops else "console",
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
                snapshot=False,
                histogram_vars={"MTW", FAKES_SOURCE},
                binnings=transfer_selection_binnings,
            )
            loaded_sideband_hists = (
                LOAD_SAVED_HISTS and sideband_analysis.load_hists_if_available()
            )

        if sideband_analysis and (loaded_sideband_hists or run_sideband_event_loops):
            for config in CONFIGS:
                if not use_transfer_validation_config(config):
                    continue
                for transfer_test in transfer_tests_by_config[config.label]:
                    if not use_transfer_validation_test(transfer_test):
                        continue
                    for prong in TRANSFER_VALIDATION_PRONGS:
                        prefix = f"{config.label}_{transfer_test.name}_{WP}_{prong}prong"
                        estimate_name = f"{prefix}_{FAKES_SOURCE}_src"
                        derive_pass = f"{prefix}_derive_passID"
                        derive_fail = f"{prefix}_derive_failID"
                        validate_pass = f"{prefix}_validate_passID"
                        validate_fail = f"{prefix}_validate_failID"

                        if not loaded_sideband_hists:
                            sideband_analysis.do_fakes_estimate(
                                FAKES_SOURCE,
                                ("MTW",),
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

                        numerator = sideband_analysis.histograms[
                            f"{estimate_name}_{FAKES_SOURCE}_FF_numerator"
                        ]
                        denominator = sideband_analysis.histograms[
                            f"{estimate_name}_{FAKES_SOURCE}_FF_denominator"
                        ]
                        validation_fail_input = sideband_analysis.histograms[
                            f"{estimate_name}_{FAKES_SOURCE}_FF_fakes_data_est"
                        ]
                        prediction = sideband_analysis.histograms[
                            f"{estimate_name}_MTW_fakes_bkg_{FAKES_SOURCE}_src"
                        ]
                        negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                            numerator, denominator
                        )

                        target_data = sideband_analysis.get_hist(
                            "MTW",
                            dataset=sideband_analysis.data_sample,
                            systematic=NOMINAL_NAME,
                            selection=validate_pass,
                            allow_generation=True,
                        )
                        target_nonfake = sum_th1s(
                            *[
                                sideband_analysis.get_hist(
                                    "MTW",
                                    dataset=mc_sample,
                                    systematic=NOMINAL_NAME,
                                    selection=f"trueTau_{validate_pass}",
                                    allow_generation=True,
                                )
                                for mc_sample in sideband_analysis.mc_samples
                            ]
                        )
                        target = target_data - target_nonfake
                        transfer_validation_rows.append(
                            (
                                config.label,
                                f"{prong}-prong",
                                FAKES_SOURCE,
                                transfer_test.name,
                                transfer_test.label,
                                hist_integral(numerator),
                                hist_integral(denominator),
                                negative_numerator_bins,
                                tiny_denominator_bins,
                                hist_integral(validation_fail_input),
                                hist_integral(prediction),
                                hist_integral(target),
                                hist_integral(prediction) / hist_integral(target)
                                if hist_integral(target) != 0
                                else float("nan"),
                            )
                        )

                        if (
                            RUN_TAUPT_MET_2D_FEASIBILITY_CHECK
                            and transfer_test.name == "met_cr_split"
                        ):
                            taupt_met_feasibility_rows.append(
                                (
                                    config.label,
                                    f"{prong}-prong",
                                    "TauPt x MET_met",
                                    "MET < 120",
                                    "120 <= MET < 170",
                                    "not directly derivable from this independent split",
                                )
                            )

                        if PLOT_FAKE_TRANSFER_VALIDATION:
                            sideband_analysis.paths.plot_dir = (
                                output_root
                                / "plots"
                                / "fake_transfer_validation"
                                / config.label
                                / transfer_test.name
                            )
                            sideband_analysis.plot(
                                [target, prediction],
                                label=[
                                    "Data - nonfake MC in validation pass-ID",
                                    "Fake prediction",
                                ],
                                colour=["k", "tab:blue"],
                                histstyle=["step", "step"],
                                xlabel=variable_data["MTW"]["name"] + " [GeV]",
                                kind="overlay",
                                do_stat=True,
                                do_syst=False,
                                title=smart_join(
                                    config.label,
                                    f"{prong}-prong",
                                    transfer_test.name,
                                    sep=" | ",
                                ),
                                scale_by_bin_width=False,
                                ylabel="Events",
                                logx=True,
                                ratio_plot=True,
                                ratio_label="Prediction / target",
                                ratio_axlim=(0.0, 3.0),
                                label_params={"llabel": "", "loc": 1},
                                filename=(
                                    f"{config.label}_{transfer_test.name}_{prong}prong"
                                    "_fake_transfer_validation.png"
                                ),
                            )

            if run_sideband_event_loops:
                sideband_analysis.save_hists()

    # CONTROL-REGION COMPOSITION
    # ========================================================================
    for config in CONFIGS:
        for prong_label, selection_prefix in (
            ("1-prong", f"{config.label}_{WP}_1prong"),
            ("3-prong", f"{config.label}_{WP}_3prong"),
        ):
            for region in ("CR_passID", "CR_failID", "SR_passID"):
                selection = f"{selection_prefix}_{region}"
                data_hist = analysis.get_hist(
                    FAKES_SOURCE,
                    dataset=analysis.data_sample,
                    systematic=NOMINAL_NAME,
                    selection=selection,
                    allow_generation=True,
                )
                nonfake_mc_hists = [
                    analysis.get_hist(
                        FAKES_SOURCE,
                        dataset=mc_sample,
                        systematic=NOMINAL_NAME,
                        selection=f"trueTau_{selection}",
                        allow_generation=True,
                    )
                    for mc_sample in analysis.mc_samples
                ]
                nonfake_mc = sum_th1s(*nonfake_mc_hists)
                data_minus_nonfake = data_hist - nonfake_mc
                data_minus_nonfake.SetName(f"{selection}_{FAKES_SOURCE}_data_minus_nonfake")
                data_minus_nonfake.SetDirectory(0)

                composition_rows.append(
                    (
                        config.label,
                        prong_label,
                        region,
                        data_hist.Integral(),
                        nonfake_mc.Integral(),
                        data_minus_nonfake.Integral(),
                        nonfake_mc.Integral() / data_hist.Integral()
                        if data_hist.Integral() != 0
                        else float("nan"),
                    )
                )

                if PLOT_CONTROL_REGION_COMPOSITION and region in {"CR_passID", "CR_failID"}:
                    analysis.paths.plot_dir = (
                        output_root / "plots" / config.label / prong_label / "composition"
                    )
                    analysis.plot(
                        [data_hist, nonfake_mc, data_minus_nonfake],
                        label=["Data", "Nonfake MC", "Data - nonfake MC"],
                        colour=["k", "tab:blue", "tab:orange"],
                        histstyle=["step", "step", "step"],
                        xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(config.label, prong_label, region, sep=" | "),
                        scale_by_bin_width=False,
                        ylabel="Events",
                        logx=True,
                        ratio_plot=True,
                        ratio_label="Ratio to data",
                        ratio_axlim=(0.0, 1.5),
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{prong_label}_{region}_{FAKES_SOURCE}_composition.png",
                    )

    # PASS-ID NONFAKE MC COMPONENTS
    # ========================================================================
    for config in CONFIGS:
        for prong_label, selection_prefix in (
            ("1-prong", f"{config.label}_{WP}_1prong"),
            ("3-prong", f"{config.label}_{WP}_3prong"),
        ):
            selection = f"{selection_prefix}_SR_passID"
            data_hist = analysis.get_hist(
                FAKES_SOURCE,
                dataset=analysis.data_sample,
                systematic=NOMINAL_NAME,
                selection=selection,
                allow_generation=True,
            )
            component_hists = []
            for mc_sample in analysis.mc_samples:
                hist = analysis.get_hist(
                    FAKES_SOURCE,
                    dataset=mc_sample,
                    systematic=NOMINAL_NAME,
                    selection=f"trueTau_{selection}",
                    allow_generation=True,
                )
                component_hists.append(hist)
                nonfake_component_rows.append(
                    (
                        config.label,
                        prong_label,
                        mc_sample,
                        data_hist.Integral(),
                        hist.Integral(),
                        hist.Integral() / data_hist.Integral()
                        if data_hist.Integral() != 0
                        else float("nan"),
                    )
                )

            if PLOT_NONFAKE_COMPONENTS:
                analysis.paths.plot_dir = (
                    output_root / "plots" / config.label / prong_label / "nonfake_components"
                )
                analysis.plot(
                    [data_hist, *component_hists],
                    label=["Data", *analysis.mc_samples],
                    colour=[
                        COMPONENT_COLOURS["data"],
                        *[
                            COMPONENT_COLOURS.get(mc_sample, f"C{idx}")
                            for idx, mc_sample in enumerate(analysis.mc_samples)
                        ],
                    ],
                    histstyle=["step"] * (len(component_hists) + 1),
                    xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=True,
                    do_syst=False,
                    title=smart_join(config.label, prong_label, "SR pass-ID nonfake MC", sep=" | "),
                    scale_by_bin_width=False,
                    ylabel="Events",
                    logx=True,
                    ratio_plot=True,
                    ratio_label="Component / data",
                    ratio_axlim=(0.0, 1.5),
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{prong_label}_SR_passID_nonfake_components.png",
                )

    # TAUPT CONTRIBUTIONS TO THE FAKE PREDICTION
    # ========================================================================
    for config in CONFIGS:
        for prong in (1, 3):
            prefix = f"{config.label}_{WP}_{prong}prong"
            fake_factor = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF"]
            fail_fake_like = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_FF_fakes_data_est"]
            predicted_fake = analysis.histograms[f"{prefix}_{FAKES_SOURCE}_fakes_bkg_{FAKES_SOURCE}"]
            total_prediction = predicted_fake.Integral()
            low_taupt_prediction = 0.0

            for bin_idx in range(1, predicted_fake.GetNbinsX() + 1):
                predicted = predicted_fake.GetBinContent(bin_idx)
                bin_low = predicted_fake.GetBinLowEdge(bin_idx)
                bin_high = bin_low + predicted_fake.GetBinWidth(bin_idx)
                if bin_low >= 170 and bin_high <= 250:
                    low_taupt_prediction += predicted
                taupt_contribution_rows.append(
                    (
                        config.label,
                        f"{prong}-prong",
                        bin_low,
                        bin_high,
                        fail_fake_like.GetBinContent(bin_idx),
                        fake_factor.GetBinContent(bin_idx),
                        predicted,
                        predicted / total_prediction if total_prediction != 0 else float("nan"),
                    )
                )
            low_taupt_summary_rows.append(
                (
                    config.label,
                    f"{prong}-prong",
                    low_taupt_prediction,
                    total_prediction,
                    low_taupt_prediction / total_prediction if total_prediction != 0 else float("nan"),
                )
            )

    # PASS-ID PRONG BALANCE
    # ========================================================================
    for config in CONFIGS:
        prong_values: dict[str, dict[int, float]] = {
            "data": {},
            "wtaunu_had nonfake MC": {},
            "total nonfake MC": {},
            "data - nonfake MC target": {},
            "fake prediction": {},
        }
        for prong in (1, 3):
            selection = f"{config.label}_{WP}_{prong}prong_SR_passID"
            data_hist = analysis.get_hist(
                FAKES_SOURCE,
                dataset=analysis.data_sample,
                systematic=NOMINAL_NAME,
                selection=selection,
                allow_generation=True,
            )
            wtaunu_had_hist = analysis.get_hist(
                FAKES_SOURCE,
                dataset="wtaunu_had",
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            total_nonfake_hist = sum_th1s(
                *[
                    analysis.get_hist(
                        FAKES_SOURCE,
                        dataset=mc_sample,
                        systematic=NOMINAL_NAME,
                        selection=f"trueTau_{selection}",
                        allow_generation=True,
                    )
                    for mc_sample in analysis.mc_samples
                ]
            )
            fake_prediction = analysis.histograms[
                f"{config.label}_{WP}_{prong}prong_{FAKES_SOURCE}_fakes_bkg_{FAKES_SOURCE}"
            ]

            prong_values["data"][prong] = data_hist.Integral()
            prong_values["wtaunu_had nonfake MC"][prong] = wtaunu_had_hist.Integral()
            prong_values["total nonfake MC"][prong] = total_nonfake_hist.Integral()
            prong_values["data - nonfake MC target"][prong] = (
                data_hist.Integral() - total_nonfake_hist.Integral()
            )
            prong_values["fake prediction"][prong] = fake_prediction.Integral()

        for component, values in prong_values.items():
            one_prong = values[1]
            three_prong = values[3]
            prong_balance_rows.append(
                (
                    config.label,
                    component,
                    one_prong,
                    three_prong,
                    three_prong / one_prong if one_prong != 0 else float("nan"),
                )
            )

    # WTAUNU_HAD EVENT-LEVEL PRONG DIAGNOSTICS
    # ========================================================================
    if RUN_EVENT_LEVEL_PRONG_DIAGNOSTICS:
        wtaunu_dsids = sorted(analysis.metadata.dataset_dsids["wtaunu_had"])
        dsid_index_cases = "\n".join(
            f"                if (mcChannel == {dsid}) {{ return {idx}; }}"
            for idx, dsid in enumerate(wtaunu_dsids)
        )
        mtw_thresholds = ", ".join(f"{config.mtw_min:g}" for config in CONFIGS)
        taupt_thresholds = ", ".join(f"{config.taupt_min:g}" for config in CONFIGS)
        met_thresholds = ", ".join(f"{config.met_min:g}" for config in CONFIGS)
        ROOT.gInterpreter.Declare(
            f"""
        #include <atomic>
        #include <cmath>

        namespace shadowFakeValidation {{
            std::atomic<double> truthProngWeighted[3][3][5][2];
            std::atomic<unsigned long long> truthProngUnweighted[3][3][5][2];
            std::atomic<double> recoProngWeighted[3][4][2];
            std::atomic<unsigned long long> recoProngUnweighted[3][4][2];
            std::atomic<double> recoProngWeightComponents[3][4][5][2];
            std::atomic<double> dsidRecoProngWeightComponents[3][{len(wtaunu_dsids)}][2][5][2];
            std::atomic<double> recoWeightDecomposition[3][{len(WEIGHT_DECOMPOSITION_LABELS)}][2];
            std::atomic<double> dsidRecoWeightDecomposition
                [3][{len(wtaunu_dsids)}][{len(WEIGHT_DECOMPOSITION_LABELS)}][2];
            double mtwThresholds[3] = {{{mtw_thresholds}}};
            double tauptThresholds[3] = {{{taupt_thresholds}}};
            double metThresholds[3] = {{{met_thresholds}}};

            int dsidIndex(int mcChannel) {{
{dsid_index_cases}
                return -1;
            }}

            void addAtomic(std::atomic<double>& target, double value) {{
                double current = target.load(std::memory_order_relaxed);
                while (!target.compare_exchange_weak(
                    current, current + value, std::memory_order_relaxed
                )) {{}}
            }}

            template <typename Weight, typename Component>
            double removeWeightComponent(Weight weight, Component component) {{
                double componentValue = static_cast<double>(component);
                if (!std::isfinite(componentValue) || componentValue == 0.0) {{
                    return 0.0;
                }}
                return static_cast<double>(weight) / componentValue;
            }}

            void resetTruthProngSculpting() {{
                for (int config = 0; config < 3; ++config) {{
                    for (int sample = 0; sample < 3; ++sample) {{
                        for (int stage = 0; stage < 5; ++stage) {{
                            for (int prong = 0; prong < 2; ++prong) {{
                                truthProngWeighted[config][sample][stage][prong].store(0.0);
                                truthProngUnweighted[config][sample][stage][prong].store(0);
                            }}
                        }}
                    }}
                    for (int stage = 0; stage < 4; ++stage) {{
                        for (int prong = 0; prong < 2; ++prong) {{
                            recoProngWeighted[config][stage][prong].store(0.0);
                            recoProngUnweighted[config][stage][prong].store(0);
                            for (int weight = 0; weight < 5; ++weight) {{
                                recoProngWeightComponents[config][stage][weight][prong].store(0.0);
                            }}
                        }}
                    }}
                    for (int dsid = 0; dsid < {len(wtaunu_dsids)}; ++dsid) {{
                        for (int stage = 0; stage < 2; ++stage) {{
                            for (int weight = 0; weight < 5; ++weight) {{
                                for (int prong = 0; prong < 2; ++prong) {{
                                    dsidRecoProngWeightComponents[config][dsid][stage][weight][prong]
                                        .store(0.0);
                                }}
                            }}
                        }}
                    }}
                    for (int weight = 0; weight < {len(WEIGHT_DECOMPOSITION_LABELS)}; ++weight) {{
                        for (int prong = 0; prong < 2; ++prong) {{
                            recoWeightDecomposition[config][weight][prong].store(0.0);
                        }}
                    }}
                    for (int dsid = 0; dsid < {len(wtaunu_dsids)}; ++dsid) {{
                        for (int weight = 0; weight < {len(WEIGHT_DECOMPOSITION_LABELS)}; ++weight) {{
                            for (int prong = 0; prong < 2; ++prong) {{
                                dsidRecoWeightDecomposition[config][dsid][weight][prong]
                                    .store(0.0);
                            }}
                        }}
                    }}
                }}
            }}

            template <typename McChannel, typename SampleID, typename PassTruth, typename TruthHad,
                      typename TruthNTracks, typename VisPt, typename TruthMtw,
                      typename NuPt, typename TruthEta, typename TruthWeight,
                      typename PassReco, typename TauBaseline, typename TauCharge,
                      typename PassMetTrigger, typename BadJet, typename MatchedTau,
                      typename MatchedHadTau, typename MatchedMuon, typename MatchedElectron,
                      typename MatchedPhoton, typename RecoNTracks, typename TauPt,
                      typename TauEta, typename RecoMtw, typename Met,
                      typename EleScore, typename JetScore, typename McWeight,
                      typename EventWeight, typename RecoWeight, typename TauRecoSf,
                      typename SelectionSf, typename TriggerSf, typename PrwWeight,
                      typename JvtSf, typename FjvtSf>
            int recordProngDiagnostics(
                McChannel mcChannel,
                SampleID sampleID,
                PassTruth passTruth,
                TruthHad truthHad,
                TruthNTracks truthNTracks,
                VisPt visPt,
                TruthMtw truthMtw,
                NuPt nuPt,
                TruthEta truthEta,
                TruthWeight truthWeight,
                PassReco passReco,
                TauBaseline tauBaseline,
                TauCharge tauCharge,
                PassMetTrigger passMetTrigger,
                BadJet badJet,
                MatchedTau matchedTau,
                MatchedHadTau matchedHadTau,
                MatchedMuon matchedMuon,
                MatchedElectron matchedElectron,
                MatchedPhoton matchedPhoton,
                RecoNTracks recoNTracks,
                TauPt tauPt,
                TauEta tauEta,
                RecoMtw recoMtw,
                Met met,
                EleScore eleScore,
                JetScore jetScore,
                McWeight mcWeight,
                EventWeight eventWeight,
                RecoWeight recoWeight,
                TauRecoSf tauRecoSf,
                SelectionSf selectionSf,
                TriggerSf triggerSf,
                PrwWeight prwWeight,
                JvtSf jvtSf,
                FjvtSf fjvtSf
            ) {{
                int dsid = dsidIndex(static_cast<int>(mcChannel));
                int subsampleIndex = -1;
                if (sampleID == sampleToId_wtaunu_had["full"]) {{
                    subsampleIndex = 1;
                }} else if (sampleID == sampleToId_wtaunu_had["lm_cut"]) {{
                    subsampleIndex = 2;
                }}

                int sampleIndices[2] = {{0, subsampleIndex}};
                int nSamples = (subsampleIndex >= 0) ? 2 : 1;

                if (passTruth == 1 && truthHad && (truthNTracks == 1 || truthNTracks == 3)) {{
                    int truthProngIndex = (truthNTracks == 1) ? 0 : 1;
                    for (int config = 0; config < 3; ++config) {{
                        bool passStage[5];
                        passStage[0] = true;
                        passStage[1] = passStage[0] && (visPt > tauptThresholds[config]);
                        passStage[2] = passStage[1] && (truthMtw > mtwThresholds[config]);
                        passStage[3] = passStage[2] && (nuPt > metThresholds[config]);
                        passStage[4] = passStage[3] &&
                            (((std::abs(truthEta) < 1.37) || (1.52 < std::abs(truthEta))) &&
                             (std::abs(truthEta) < 2.47));

                        for (int sampleSlot = 0; sampleSlot < nSamples; ++sampleSlot) {{
                            int sample = sampleIndices[sampleSlot];
                            for (int stage = 0; stage < 5; ++stage) {{
                                if (!passStage[stage]) {{
                                    continue;
                                }}
                                addAtomic(
                                    truthProngWeighted[config][sample][stage][truthProngIndex],
                                    static_cast<double>(truthWeight)
                                );
                                truthProngUnweighted[config][sample][stage][truthProngIndex]
                                    .fetch_add(1, std::memory_order_relaxed);
                            }}
                        }}
                    }}
                }}

                if (recoNTracks == 1 || recoNTracks == 3) {{
                    int recoProngIndex = (recoNTracks == 1) ? 0 : 1;
                    bool truthMatched = matchedHadTau || matchedMuon || matchedElectron;
                    bool passRecoBase =
                        (passReco == 1) &&
                        (tauBaseline == 1) &&
                        (std::abs(tauCharge) == 1) &&
                        passMetTrigger &&
                        (badJet == 0) &&
                        ((matchedTau + matchedElectron + matchedMuon + matchedPhoton) <= 1);
                    bool passMedium =
                        (eleScore > 0.1) &&
                        (((jetScore > 0.25) && (recoNTracks == 1)) ||
                         ((jetScore > 0.4) && (recoNTracks == 3)));

                    for (int config = 0; config < 3; ++config) {{
                        bool passSignalRegion =
                            passRecoBase &&
                            (tauPt > tauptThresholds[config]) &&
                            (((std::abs(tauEta) < 1.37) || (1.52 < std::abs(tauEta))) &&
                             (std::abs(tauEta) < 2.47)) &&
                            (recoMtw > mtwThresholds[config]) &&
                            (met > metThresholds[config]);

                        bool recoStage[4];
                        recoStage[0] = passSignalRegion;
                        recoStage[1] = passSignalRegion && truthMatched;
                        recoStage[2] = passSignalRegion && passMedium;
                        recoStage[3] = passSignalRegion && passMedium && truthMatched;
                        double weights[5] = {{
                            1.0,
                            static_cast<double>(mcWeight),
                            static_cast<double>(eventWeight),
                            static_cast<double>(truthWeight),
                            static_cast<double>(recoWeight),
                        }};
                        double decomposedWeights[{len(WEIGHT_DECOMPOSITION_LABELS)}] = {{
                            static_cast<double>(eventWeight),
                            removeWeightComponent(eventWeight, tauRecoSf),
                            removeWeightComponent(eventWeight, selectionSf),
                            removeWeightComponent(eventWeight, triggerSf),
                            removeWeightComponent(eventWeight, prwWeight),
                            removeWeightComponent(eventWeight, jvtSf),
                            removeWeightComponent(eventWeight, fjvtSf),
                        }};

                        for (int stage = 0; stage < 4; ++stage) {{
                            if (!recoStage[stage]) {{
                                continue;
                            }}
                            addAtomic(
                                recoProngWeighted[config][stage][recoProngIndex],
                                static_cast<double>(recoWeight)
                            );
                            recoProngUnweighted[config][stage][recoProngIndex]
                                .fetch_add(1, std::memory_order_relaxed);
                            for (int weightIndex = 0; weightIndex < 5; ++weightIndex) {{
                                addAtomic(
                                    recoProngWeightComponents[config][stage][weightIndex][recoProngIndex],
                                    weights[weightIndex]
                                );
                            }}
                        }}

                        if (dsid >= 0) {{
                            bool dsidStage[2] = {{
                                recoStage[1],
                                recoStage[3],
                            }};
                            for (int stage = 0; stage < 2; ++stage) {{
                                if (!dsidStage[stage]) {{
                                    continue;
                                }}
                                for (int weightIndex = 0; weightIndex < 5; ++weightIndex) {{
                                    addAtomic(
                                        dsidRecoProngWeightComponents
                                            [config][dsid][stage][weightIndex][recoProngIndex],
                                        weights[weightIndex]
                                    );
                                }}
                            }}
                        }}

                        if (recoStage[3]) {{
                            for (int weightIndex = 0; weightIndex < {len(WEIGHT_DECOMPOSITION_LABELS)}; ++weightIndex) {{
                                addAtomic(
                                    recoWeightDecomposition[config][weightIndex][recoProngIndex],
                                    decomposedWeights[weightIndex]
                                );
                                if (dsid >= 0) {{
                                    addAtomic(
                                        dsidRecoWeightDecomposition
                                            [config][dsid][weightIndex][recoProngIndex],
                                        decomposedWeights[weightIndex]
                                    );
                                }}
                            }}
                        }}
                    }}
                }}
                return 0;
            }}

            double getTruthProngWeighted(int config, int sample, int stage, int prong) {{
                return truthProngWeighted[config][sample][stage][prong].load();
            }}

            unsigned long long getTruthProngUnweighted(
                int config, int sample, int stage, int prong
            ) {{
                return truthProngUnweighted[config][sample][stage][prong].load();
            }}

            double getRecoProngWeighted(int config, int stage, int prong) {{
                return recoProngWeighted[config][stage][prong].load();
            }}

            unsigned long long getRecoProngUnweighted(int config, int stage, int prong) {{
                return recoProngUnweighted[config][stage][prong].load();
            }}

            double getRecoProngWeightComponent(
                int config, int stage, int weightIndex, int prong
            ) {{
                return recoProngWeightComponents[config][stage][weightIndex][prong].load();
            }}

            double getDsidRecoProngWeightComponent(
                int config, int dsid, int stage, int weightIndex, int prong
            ) {{
                return dsidRecoProngWeightComponents[config][dsid][stage][weightIndex][prong]
                    .load();
            }}

            double getRecoWeightDecomposition(int config, int weightIndex, int prong) {{
                return recoWeightDecomposition[config][weightIndex][prong].load();
            }}

            double getDsidRecoWeightDecomposition(
                int config, int dsid, int weightIndex, int prong
            ) {{
                return dsidRecoWeightDecomposition[config][dsid][weightIndex][prong].load();
            }}
        }}
        """
        )
        ROOT.shadowFakeValidation.resetTruthProngSculpting()
        (
            analysis["wtaunu_had"]
            .rdataframes[NOMINAL_NAME]
            .Define(
                "_prong_diagnostics",
                "shadowFakeValidation::recordProngDiagnostics("
                "mcChannel, SampleID, passTruth, TruthTau_isHadronic, TruthTau_nChargedTracks, "
                "VisTruthTauPt, TruthMTW, TruthNeutrinoPt, VisTruthTauEta, truth_weight, "
                "passReco, TauBaselineWP, TauCharge, passMetTrigger, badJet, "
                "MatchedTruthParticle_isTau, MatchedTruthParticle_isHadronicTau, "
                "MatchedTruthParticle_isMuon, MatchedTruthParticle_isElectron, "
                "MatchedTruthParticle_isPhoton, TauNCoreTracks, TauPt, TauEta, MTW, "
                "MET_met, TauBDTEleScore, TauRNNJetScore, mcWeight, weight, reco_weight, "
                "TauRecoSF, selectionSF, TriggerSF, prwWeight, jvtSF, fjvtSF)",
            )
            .Sum("_prong_diagnostics")
            .GetValue()
        )

        subsample_labels = ("all", "full", "lm_cut")
        truth_stage_labels = (
            "passTruth + hadronic prong",
            "+ VisTruthTauPt",
            "+ TruthMTW",
            "+ TruthNeutrinoPt",
            "+ truth eta",
        )
        for config_idx, config in enumerate(CONFIGS):
            for subsample_idx, subsample_label in enumerate(subsample_labels):
                for stage_idx, stage_label in enumerate(truth_stage_labels):
                    one_prong_weighted = ROOT.shadowFakeValidation.getTruthProngWeighted(
                        config_idx, subsample_idx, stage_idx, 0
                    )
                    three_prong_weighted = ROOT.shadowFakeValidation.getTruthProngWeighted(
                        config_idx, subsample_idx, stage_idx, 1
                    )
                    one_prong_unweighted = float(
                        ROOT.shadowFakeValidation.getTruthProngUnweighted(
                            config_idx, subsample_idx, stage_idx, 0
                        )
                    )
                    three_prong_unweighted = float(
                        ROOT.shadowFakeValidation.getTruthProngUnweighted(
                            config_idx, subsample_idx, stage_idx, 1
                        )
                    )
                    truth_cut_prong_rows.append(
                        (
                            config.label,
                            subsample_label,
                            stage_label,
                            "weighted",
                            one_prong_weighted,
                            three_prong_weighted,
                            three_prong_weighted / one_prong_weighted
                            if one_prong_weighted != 0
                            else float("nan"),
                            one_prong_unweighted,
                            three_prong_unweighted,
                        )
                    )

            for stage_label, stage_idx in (
                ("truth fiducial", 4),
                ("reco preselection", 1),
                ("medium pass-ID", 3),
            ):
                if stage_label == "truth fiducial":
                    one_prong = ROOT.shadowFakeValidation.getTruthProngWeighted(config_idx, 0, stage_idx, 0)
                    three_prong = ROOT.shadowFakeValidation.getTruthProngWeighted(config_idx, 0, stage_idx, 1)
                else:
                    one_prong = ROOT.shadowFakeValidation.getRecoProngWeighted(config_idx, stage_idx, 0)
                    three_prong = ROOT.shadowFakeValidation.getRecoProngWeighted(config_idx, stage_idx, 1)
                wtaunu_had_stage_rows.append(
                    (
                        config.label,
                        stage_label,
                        one_prong,
                        three_prong,
                        three_prong / one_prong if one_prong != 0 else float("nan"),
                    )
                )

            for stage_label, stage_idx in (
                ("reco preselection, all reco candidates", 0),
                ("reco preselection, truth matched", 1),
                ("medium pass-ID, all reco candidates", 2),
                ("medium pass-ID, truth matched", 3),
            ):
                one_prong_weighted = ROOT.shadowFakeValidation.getRecoProngWeighted(
                    config_idx, stage_idx, 0
                )
                three_prong_weighted = ROOT.shadowFakeValidation.getRecoProngWeighted(
                    config_idx, stage_idx, 1
                )
                one_prong_unweighted = float(
                    ROOT.shadowFakeValidation.getRecoProngUnweighted(config_idx, stage_idx, 0)
                )
                three_prong_unweighted = float(
                    ROOT.shadowFakeValidation.getRecoProngUnweighted(config_idx, stage_idx, 1)
                )
                weighted_ratio = (
                    three_prong_weighted / one_prong_weighted
                    if one_prong_weighted != 0
                    else float("nan")
                )
                unweighted_ratio = (
                    three_prong_unweighted / one_prong_unweighted
                    if one_prong_unweighted != 0
                    else float("nan")
                )
                reco_prong_weight_rows.append(
                    (
                        config.label,
                        stage_label,
                        one_prong_weighted,
                        three_prong_weighted,
                        weighted_ratio,
                        one_prong_unweighted,
                        three_prong_unweighted,
                        unweighted_ratio,
                        weighted_ratio / unweighted_ratio if unweighted_ratio != 0 else float("nan"),
                    )
                )
                raw_ratio = unweighted_ratio
                for weight_idx, weight_label in enumerate(PRONG_WEIGHT_LABELS):
                    one_prong = ROOT.shadowFakeValidation.getRecoProngWeightComponent(
                        config_idx, stage_idx, weight_idx, 0
                    )
                    three_prong = ROOT.shadowFakeValidation.getRecoProngWeightComponent(
                        config_idx, stage_idx, weight_idx, 1
                    )
                    ratio = three_prong / one_prong if one_prong != 0 else float("nan")
                    reco_weight_component_rows.append(
                        (
                            config.label,
                            stage_label,
                            weight_label,
                            one_prong,
                            three_prong,
                            ratio,
                            ratio / raw_ratio if raw_ratio != 0 else float("nan"),
                        )
                    )

            for dsid_idx, dsid in enumerate(wtaunu_dsids):
                phys_short = analysis.metadata[dsid].phys_short
                for stage_idx, stage_label in enumerate(DSID_DIAGNOSTIC_STAGE_LABELS):
                    raw_one_prong = ROOT.shadowFakeValidation.getDsidRecoProngWeightComponent(
                        config_idx, dsid_idx, stage_idx, 0, 0
                    )
                    raw_three_prong = ROOT.shadowFakeValidation.getDsidRecoProngWeightComponent(
                        config_idx, dsid_idx, stage_idx, 0, 1
                    )
                    raw_ratio = (
                        raw_three_prong / raw_one_prong
                        if raw_one_prong != 0
                        else float("nan")
                    )
                    for weight_idx, weight_label in enumerate(PRONG_WEIGHT_LABELS):
                        one_prong = ROOT.shadowFakeValidation.getDsidRecoProngWeightComponent(
                            config_idx, dsid_idx, stage_idx, weight_idx, 0
                        )
                        three_prong = ROOT.shadowFakeValidation.getDsidRecoProngWeightComponent(
                            config_idx, dsid_idx, stage_idx, weight_idx, 1
                        )
                        ratio = three_prong / one_prong if one_prong != 0 else float("nan")
                        dsid_prong_weight_rows.append(
                            (
                                config.label,
                                dsid,
                                phys_short,
                                stage_label,
                                weight_label,
                                one_prong,
                                three_prong,
                                ratio,
                                ratio / raw_ratio if raw_ratio != 0 else float("nan"),
                            )
                        )

            dsid_700451_idx = wtaunu_dsids.index(700451) if 700451 in wtaunu_dsids else None
            for scope_label, dsid_idx in (("all DSIDs", None), ("DSID 700451", dsid_700451_idx)):
                if dsid_idx is None and scope_label != "all DSIDs":
                    continue
                for weight_idx, weight_label in enumerate(WEIGHT_DECOMPOSITION_LABELS):
                    if dsid_idx is None:
                        one_prong = ROOT.shadowFakeValidation.getRecoWeightDecomposition(
                            config_idx, weight_idx, 0
                        )
                        three_prong = ROOT.shadowFakeValidation.getRecoWeightDecomposition(
                            config_idx, weight_idx, 1
                        )
                    else:
                        one_prong = ROOT.shadowFakeValidation.getDsidRecoWeightDecomposition(
                            config_idx, dsid_idx, weight_idx, 0
                        )
                        three_prong = ROOT.shadowFakeValidation.getDsidRecoWeightDecomposition(
                            config_idx, dsid_idx, weight_idx, 1
                        )
                    total = one_prong + three_prong
                    weight_decomposition_rows.append(
                        (
                            config.label,
                            scope_label,
                            weight_label,
                            one_prong,
                            three_prong,
                            three_prong / one_prong if one_prong != 0 else float("nan"),
                            three_prong / total if total != 0 else float("nan"),
                        )
                    )
    else:
        for config in CONFIGS:
            wtaunu_had_stage_rows.append((config.label, "event-level diagnostics skipped", 0, 0, float("nan")))

    # PASS-ID VALIDATION & FAKE-FACTOR SHAPES
    # ========================================================================
    for config in CONFIGS:
        vars_for_config = VARS if config.unfolded_var is None else (config.unfolded_var,)

        if PLOT_FAKE_FACTORS:
            fake_factors = [
                analysis.histograms[f"{config.label}_{WP}_{prong}prong_{FAKES_SOURCE}_FF"]
                for prong in (1, 3)
            ]
            analysis.paths.plot_dir = output_root / "plots" / config.label / "fake_factors"
            analysis.plot(
                fake_factors,
                label=["1-prong fake factor", "3-prong fake factor"],
                colour=["tab:blue", "tab:orange"],
                histstyle=["step", "step"],
                xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                title=smart_join(config.label, "fake factors", sep=" | "),
                scale_by_bin_width=False,
                ylabel="Fake factor",
                logx=True,
                ratio_plot=True,
                ratio_label="3-prong / 1-prong",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                filename=f"{config.label}_{FAKES_SOURCE}_prong_fake_factors.png",
            )

        for var in vars_for_config:
            data_hist = analysis.get_hist(
                var,
                dataset=analysis.data_sample,
                systematic=NOMINAL_NAME,
                selection=f"{config.label}_{WP}_SR_passID",
                allow_generation=True,
            )
            nonfake_mc = sum_th1s(
                *[
                    analysis.get_hist(
                        var,
                        dataset=mc_sample,
                        systematic=NOMINAL_NAME,
                        selection=f"trueTau_{config.label}_{WP}_SR_passID",
                        allow_generation=True,
                    )
                    for mc_sample in analysis.mc_samples
                ]
            )
            data_minus_nonfake = data_hist - nonfake_mc
            data_minus_nonfake.SetName(f"{config.label}_{var}_sr_pass_data_minus_nonfake")
            data_minus_nonfake.SetDirectory(0)

            prong_fakes = [
                analysis.histograms[
                    f"{config.label}_{WP}_{prong}prong_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                for prong in (1, 3)
            ]
            prong_sum_fakes = sum_th1s(*prong_fakes)
            prong_sum_fakes.SetName(f"{config.label}_{var}_prong_split_fakes")
            prong_sum_fakes.SetDirectory(0)

            pass_validation_rows.append(
                (
                    config.label,
                    var,
                    data_minus_nonfake.Integral(),
                    prong_sum_fakes.Integral(),
                    prong_sum_fakes.Integral() / data_minus_nonfake.Integral()
                    if data_minus_nonfake.Integral() != 0
                    else float("nan"),
                )
            )

            if RUN_FAKE_SOURCE_CROSSCHECK and var == FAKES_SOURCE_CROSSCHECK:
                crosscheck_fakes = [
                    analysis.histograms[
                        f"{config.label}_{WP}_{prong}prong_{FAKES_SOURCE_CROSSCHECK}_src_"
                        f"{var}_fakes_bkg_{FAKES_SOURCE_CROSSCHECK}_src"
                    ]
                    for prong in (1, 3)
                ]
                crosscheck_sum_fakes = sum_th1s(*crosscheck_fakes)
                crosscheck_sum_fakes.SetName(f"{config.label}_{var}_mtw_sourced_fakes")
                crosscheck_sum_fakes.SetDirectory(0)
                fake_source_crosscheck_rows.append(
                    (
                        config.label,
                        var,
                        prong_sum_fakes.Integral(),
                        crosscheck_sum_fakes.Integral(),
                        crosscheck_sum_fakes.Integral() / prong_sum_fakes.Integral()
                        if prong_sum_fakes.Integral() != 0
                        else float("nan"),
                    )
                )

            if PLOT_PASS_ID_VALIDATION:
                analysis.paths.plot_dir = output_root / "plots" / config.label / var / "pass_id"
                analysis.plot(
                    [data_minus_nonfake, prong_sum_fakes],
                    label=["Data - nonfake MC in pass-ID", "Prong-split fake prediction"],
                    colour=["k", "tab:blue"],
                    histstyle=["step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=True,
                    do_syst=False,
                    title=smart_join(config.label, var, "pass-ID fake validation", sep=" | "),
                    scale_by_bin_width=False,
                    ylabel="Events",
                    logx=True,
                    ratio_plot=True,
                    ratio_label="Prediction / target",
                    ratio_axlim=(0.0, 2.0),
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_pass_id_fake_validation.png",
                )

    # SUMMARY OUTPUT
    # ========================================================================
    summary_lines = [
        "# Shadow fake validation",
        "",
        "This script validates the fake estimate used by the shadow-bin unfolding workflow.",
        "It does not run unfolding.",
        "",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- tau ID: `{WP}`",
        "- nominal fake method: separate 1-prong and 3-prong fake estimates, then sum",
        f"- fake-source cross-check: `{FAKES_SOURCE_CROSSCHECK}`",
        f"- reused `analysis_shadow_unfold` measured output: `{can_reuse_shadow_unfold}`",
        "",
        "## Control-region composition",
        "",
        "`Fake-like yield` is `data - nonfake MC`, where `nonfake MC` uses the same truth-matched "
        "selections passed to `Analysis.do_fakes_estimate`.",
        "",
        "| Configuration | Prong | Region | Data | Nonfake MC | Fake-like yield | Nonfake / data |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (
        config_label,
        prong_label,
        region,
        data,
        nonfake,
        fake_like,
        nonfake_over_data,
    ) in composition_rows:
        summary_lines.append(
            f"| {config_label} | {prong_label} | {region} | {data:.3f} | "
            f"{nonfake:.3f} | {fake_like:.3f} | {nonfake_over_data:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Pass-ID nonfake MC components",
            "",
            "This breaks down the nonfake MC subtraction used in the pass-ID signal selection. "
            "Large entries here make the fake-like validation target smaller before the fake "
            "prediction is even applied.",
            "",
            "| Configuration | Prong | Dataset | Data | Nonfake MC component | Component / data |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for config_label, prong_label, dataset, data, component, component_over_data in (
        nonfake_component_rows
    ):
        summary_lines.append(
            f"| {config_label} | {prong_label} | {dataset} | {data:.3f} | "
            f"{component:.3f} | {component_over_data:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## TauPt-bin contributions to the fake prediction",
            "",
            "The fake factor is measured in `TauPt`, so this table shows which source bins "
            "actually create the predicted fake yield.",
            "",
            "| Configuration | Prong | TauPt low | TauPt high | SR fail data - nonfake MC | Fake factor | Predicted fakes | Fraction of prong total |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        prong_label,
        bin_low,
        bin_high,
        fail_fake_like,
        fake_factor,
        predicted,
        fraction,
    ) in taupt_contribution_rows:
        summary_lines.append(
            f"| {config_label} | {prong_label} | {bin_low:.3g} | {bin_high:.3g} | "
            f"{fail_fake_like:.3f} | {fake_factor:.4f} | {predicted:.3f} | {fraction:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "### Low-TauPt dominance",
            "",
            "This condenses the table above into the fake yield from `170 <= TauPt < 250` GeV, "
            "where most of the 1-prong fake prediction is produced.",
            "",
            "| Configuration | Prong | Predicted fakes, 170-250 GeV | Total predicted fakes | Fraction from 170-250 GeV |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for config_label, prong_label, low_prediction, total_prediction, fraction in (
        low_taupt_summary_rows
    ):
        summary_lines.append(
            f"| {config_label} | {prong_label} | {low_prediction:.3f} | "
            f"{total_prediction:.3f} | {fraction:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Pass-ID prong balance",
            "",
            "This checks whether the 3-prong/1-prong balance is consistent between data, "
            "`wtaunu_had`, total nonfake MC, the fake-like target, and the fake prediction. "
            "A large mismatch here points to a prong-modelling or nonfake-subtraction issue "
            "rather than only a fake-factor-transfer issue.",
            "",
            "| Configuration | Component | 1-prong | 3-prong | 3-prong / 1-prong |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for config_label, component, one_prong, three_prong, ratio in prong_balance_rows:
        summary_lines.append(
            f"| {config_label} | {component} | {one_prong:.3f} | {three_prong:.3f} | "
            f"{ratio:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "### Pass-ID 3-prong fractions before fake subtraction",
            "",
            "This is the same pass-ID signal-region balance as a within-component 3-prong "
            "fraction. It is easier to compare with data than the 3p/1p ratio.",
            "",
            "| Configuration | Component | Total yield | 3-prong fraction |",
            "|---|---|---:|---:|",
        ]
    )
    for config_label, component, one_prong, three_prong, _ratio in prong_balance_rows:
        total = one_prong + three_prong
        summary_lines.append(
            f"| {config_label} | {component} | {total:.3f} | "
            f"{three_prong / total if total != 0 else float('nan'):.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## wtaunu_had prong balance by stage",
            "",
            "This checks where the `wtaunu_had` 3-prong excess enters: at truth fiducial "
            "level, after reco preselection, or only after the medium pass-ID requirement.",
            "",
            "| Configuration | Stage | 1-prong | 3-prong | 3-prong / 1-prong |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for config_label, stage, one_prong, three_prong, ratio in wtaunu_had_stage_rows:
        summary_lines.append(
            f"| {config_label} | {stage} | {one_prong:.3f} | {three_prong:.3f} | "
            f"{ratio:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## wtaunu_had truth-cut prong sculpting",
            "",
            "This checks whether the large truth-fiducial 3-prong/1-prong ratio is already "
            "present before fiducial cuts, or whether a specific truth cut sculpts the prong "
            "composition. Weighted yields use `truth_weight`; unweighted yields are raw "
            "selected event counts.",
            "",
            "| Configuration | Subsample | Truth stage | 1-prong weighted | 3-prong weighted | "
            "Weighted 3/1 | 1-prong unweighted | 3-prong unweighted | Unweighted 3/1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        subsample_label,
        stage_label,
        _yield_type,
        one_prong_weighted,
        three_prong_weighted,
        weighted_ratio,
        one_prong_unweighted,
        three_prong_unweighted,
    ) in truth_cut_prong_rows:
        unweighted_ratio = (
            three_prong_unweighted / one_prong_unweighted
            if one_prong_unweighted != 0
            else float("nan")
        )
        summary_lines.append(
            f"| {config_label} | {subsample_label} | {stage_label} | "
            f"{one_prong_weighted:.3f} | {three_prong_weighted:.3f} | "
            f"{weighted_ratio:.3f} | {one_prong_unweighted:.0f} | "
            f"{three_prong_unweighted:.0f} | {unweighted_ratio:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## wtaunu_had reco weight and truth-match prong checks",
            "",
            "This checks whether the reconstructed `wtaunu_had` 3-prong excess is amplified "
            "by event weights or by the truth-matching requirement used in the nonfake MC "
            "subtraction. Weighted yields use `reco_weight`; unweighted yields are raw "
            "selected event counts.",
            "",
            "| Configuration | Stage | 1-prong weighted | 3-prong weighted | Weighted 3/1 | "
            "1-prong unweighted | 3-prong unweighted | Unweighted 3/1 | Weighted/unweighted ratio shift |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        stage_label,
        one_prong_weighted,
        three_prong_weighted,
        weighted_ratio,
        one_prong_unweighted,
        three_prong_unweighted,
        unweighted_ratio,
        ratio_shift,
    ) in reco_prong_weight_rows:
        summary_lines.append(
            f"| {config_label} | {stage_label} | {one_prong_weighted:.3f} | "
            f"{three_prong_weighted:.3f} | {weighted_ratio:.3f} | "
            f"{one_prong_unweighted:.0f} | {three_prong_unweighted:.0f} | "
            f"{unweighted_ratio:.3f} | {ratio_shift:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "### wtaunu_had reco prong ratios by weight definition",
            "",
            "This splits the same reconstructed selections by weight definition. `raw` is the "
            "unweighted event count, `mcWeight` and `weight` are ntuple branches, and "
            "`truth_weight`/`reco_weight` are the framework's luminosity- and DSID-scaled "
            "weights.",
            "",
            "| Configuration | Stage | Weight definition | 1-prong | 3-prong | 3-prong / 1-prong | Ratio / raw ratio |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        stage_label,
        weight_label,
        one_prong,
        three_prong,
        ratio,
        ratio_over_raw,
    ) in reco_weight_component_rows:
        summary_lines.append(
            f"| {config_label} | {stage_label} | {weight_label} | {one_prong:.3f} | "
            f"{three_prong:.3f} | {ratio:.3f} | {ratio_over_raw:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## wtaunu_had DSID-level reco prong ratios",
            "",
            "This breaks the same reconstructed `wtaunu_had` prong check down by DSID. "
            "The most important row for the current problem is `medium pass-ID, truth matched` "
            "with `reco_weight`, because that is the dominant nonfake signal subtraction used "
            "in the fake validation target.",
            "",
            "### Top DSID contributors after medium pass-ID",
            "",
            "| Configuration | DSID | Physics short | Weight definition | 1-prong | 3-prong | 3-prong / 1-prong | Fraction of weighted 3-prong yield |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for config in CONFIGS:
        top_rows = [
            row
            for row in dsid_prong_weight_rows
            if row[0] == config.label
            and row[3] == "medium pass-ID, truth matched"
            and row[4] == "reco_weight"
        ]
        total_three_prong = sum(row[6] for row in top_rows)
        for (
            config_label,
            dsid,
            phys_short,
            _stage_label,
            weight_label,
            one_prong,
            three_prong,
            ratio,
            _ratio_over_raw,
        ) in sorted(top_rows, key=lambda row: row[6], reverse=True):
            summary_lines.append(
                f"| {config_label} | {dsid} | `{phys_short}` | {weight_label} | "
                f"{one_prong:.3f} | {three_prong:.3f} | {ratio:.3f} | "
                f"{three_prong / total_three_prong if total_three_prong != 0 else float('nan'):.3f} |"
            )

    summary_lines.extend(
        [
            "",
            "### DSID selected-yield fractions and internal pronginess",
            "",
            "This normalises the DSID rows in two different ways. `Raw selected yield frac` "
            "and `Reco selected yield frac` say how much of the selected `wtaunu_had` sample "
            "comes from that DSID. The internal 3-prong fractions say whether that DSID is "
            "itself more or less 3-prong-heavy.",
            "",
            "| Configuration | DSID | Physics short | Raw selected yield frac | Raw 3-prong fraction | Reco selected yield frac | Reco 3-prong fraction | Reco/raw 3-prong fraction shift |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for config in CONFIGS:
        raw_rows = {
            row[1]: row
            for row in dsid_prong_weight_rows
            if row[0] == config.label
            and row[3] == "medium pass-ID, truth matched"
            and row[4] == "raw"
        }
        reco_rows = {
            row[1]: row
            for row in dsid_prong_weight_rows
            if row[0] == config.label
            and row[3] == "medium pass-ID, truth matched"
            and row[4] == "reco_weight"
        }
        total_raw = sum(row[5] + row[6] for row in raw_rows.values())
        total_reco = sum(row[5] + row[6] for row in reco_rows.values())
        for dsid, reco_row in sorted(
            reco_rows.items(), key=lambda item: item[1][5] + item[1][6], reverse=True
        ):
            raw_row = raw_rows[dsid]
            raw_total = raw_row[5] + raw_row[6]
            reco_total = reco_row[5] + reco_row[6]
            raw_three_fraction = raw_row[6] / raw_total if raw_total != 0 else float("nan")
            reco_three_fraction = reco_row[6] / reco_total if reco_total != 0 else float("nan")
            summary_lines.append(
                f"| {config.label} | {dsid} | `{reco_row[2]}` | "
                f"{raw_total / total_raw if total_raw != 0 else float('nan'):.3f} | "
                f"{raw_three_fraction:.3f} | "
                f"{reco_total / total_reco if total_reco != 0 else float('nan'):.3f} | "
                f"{reco_three_fraction:.3f} | "
                f"{reco_three_fraction / raw_three_fraction if raw_three_fraction != 0 else float('nan'):.3f} |"
            )

    summary_lines.extend(
        [
            "",
            "### Full DSID weight comparison",
            "",
            "| Configuration | DSID | Physics short | Stage | Weight definition | 1-prong | 3-prong | 3-prong / 1-prong | Ratio / raw ratio |",
            "|---|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        dsid,
        phys_short,
        stage_label,
        weight_label,
        one_prong,
        three_prong,
        ratio,
        ratio_over_raw,
    ) in dsid_prong_weight_rows:
        summary_lines.append(
            f"| {config_label} | {dsid} | `{phys_short}` | {stage_label} | "
            f"{weight_label} | {one_prong:.3f} | {three_prong:.3f} | "
            f"{ratio:.3f} | {ratio_over_raw:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## wtaunu_had weight-component removal diagnostic",
            "",
            "This tests which exposed branch in the DTA `weight` product most affects the "
            "medium pass-ID, truth-matched prong balance. Each row recomputes the same selected "
            "yield with the named component divided out of `weight`; it is a diagnostic only, "
            "not an analysis prescription.",
            "",
            "| Configuration | Scope | Weight expression | 1-prong | 3-prong | 3p/1p | 3-prong fraction |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (
        config_label,
        scope_label,
        weight_label,
        one_prong,
        three_prong,
        ratio,
        three_fraction,
    ) in weight_decomposition_rows:
        summary_lines.append(
            f"| {config_label} | {scope_label} | `{weight_label}` | "
            f"{one_prong:.3f} | {three_prong:.3f} | {ratio:.3f} | {three_fraction:.3f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Pass-ID fake validation",
            "",
            "This compares the prong-split fake prediction against `data - nonfake MC` in the "
            "pass-ID signal selection. It is a direct closure-style check of the subtraction "
            "target, not an independent validation region.",
            "",
            "| Configuration | Variable | Data - nonfake MC | Prong-split fakes | Prediction / target |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for config_label, var, target, prediction, ratio in pass_validation_rows:
        summary_lines.append(
            f"| {config_label} | {var} | {target:.3f} | {prediction:.3f} | {ratio:.3f} |"
        )

    if fake_source_crosscheck_rows:
        summary_lines.extend(
            [
                "",
                "## Fake-source variable cross-check",
                "",
                f"The nominal estimate is sourced in `{FAKES_SOURCE}`. This cross-check rebuilds "
                f"the same pass-ID fake prediction using `{FAKES_SOURCE_CROSSCHECK}` as the fake-factor "
                "source variable. It tests whether the large fake subtraction is mainly caused by "
                "parameterising the fake factor in the wrong variable.",
                "",
                "| Configuration | Target variable | TauPt-sourced fakes | MTW-sourced fakes | MTW / TauPt |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for config_label, var, taupt_sourced, mtw_sourced, ratio in fake_source_crosscheck_rows:
            summary_lines.append(
                f"| {config_label} | {var} | {taupt_sourced:.3f} | {mtw_sourced:.3f} | "
                f"{ratio:.3f} |"
            )

    if RUN_FULL_FAKE_TRANSFER_VALIDATION:
        summary_lines.extend(
            [
                "",
                "## Full fake-transfer validation",
                "",
                "This section collects the fast fake-transfer checks. Cache-only rows reuse the "
                "`analysis_shadow_unfold` measured ROOT output. Sideband rows are produced from "
                "the isolated `outputs/validate_shadow_fakes/sideband_transfer/` cache when "
                "available, or from the minimal new sideband selections when explicitly enabled.",
                "",
                "The current transfer-validation defaults are intentionally narrow, so this can "
                "act as a smoke test for the real analysis rather than a full production sweep.",
                "",
                f"- transfer-validation configs: `{TRANSFER_VALIDATION_CONFIGS}`",
                f"- transfer-validation tests: `{TRANSFER_VALIDATION_TESTS}`",
                f"- transfer-validation prongs: `{TRANSFER_VALIDATION_PRONGS}`",
                "",
                "### Cache-only fake-factor health",
                "",
                "| Configuration | Prong | Source variable | Derivation region | Validation region | "
                "CR numerator | CR denominator | Negative numerator bins | Tiny/non-positive denominator bins | "
                "Validation fail-ID input | Predicted fakes | Validation target | Prediction / target |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (
            config_label,
            prong_label,
            source_var,
            derivation_region,
            validation_region,
            cr_numerator,
            cr_denominator,
            negative_numerator_bins,
            tiny_denominator_bins,
            validation_fail_input,
            predicted,
            target,
            ratio,
        ) in fake_factor_health_rows:
            summary_lines.append(
                f"| {config_label} | {prong_label} | {source_var} | {derivation_region} | "
                f"{validation_region} | {cr_numerator:.3f} | {cr_denominator:.3f} | "
                f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                f"{validation_fail_input:.3f} | {predicted:.3f} | {target:.3f} | "
                f"{ratio:.3f} |"
            )

        summary_lines.extend(
            [
                "",
                "### Negative-bin treatment",
                "",
                "This is a source-variable diagnostic for the no-shadow 3-prong fake factor. "
                "It can be done from cached `TauPt` histograms. It does not exactly reproduce "
                "the target `MTW` shape under altered fake-factor bin treatments, because that "
                "would require event-level or 2D `TauPt` versus `MTW` information.",
                "",
                "| Configuration | Prong | Source variable | Treatment | CR numerator | CR denominator | "
                "Negative numerator bins | Tiny/non-positive denominator bins | SR fail input | "
                "Predicted fakes | Pass-ID target | Prediction / target |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (
            config_label,
            prong_label,
            source_var,
            treatment,
            cr_numerator,
            cr_denominator,
            negative_numerator_bins,
            tiny_denominator_bins,
            validation_fail_input,
            predicted,
            target,
            ratio,
        ) in negative_bin_rows:
            summary_lines.append(
                f"| {config_label} | {prong_label} | {source_var} | {treatment} | "
                f"{cr_numerator:.3f} | {cr_denominator:.3f} | "
                f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                f"{validation_fail_input:.3f} | {predicted:.3f} | {target:.3f} | "
                f"{ratio:.3f} |"
            )

        summary_lines.extend(
            [
                "",
                "### Independent sideband transfer",
                "",
                "| Configuration | Prong | Source variable | Transfer test | Region definition | "
                "CR numerator | CR denominator | Negative numerator bins | Tiny/non-positive denominator bins | "
                "Validation fail-ID input | Predicted fakes | Validation target | Prediction / target |",
                "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if transfer_validation_rows:
            for (
                config_label,
                prong_label,
                source_var,
                transfer_test,
                region_label,
                cr_numerator,
                cr_denominator,
                negative_numerator_bins,
                tiny_denominator_bins,
                validation_fail_input,
                predicted,
                target,
                ratio,
            ) in transfer_validation_rows:
                summary_lines.append(
                    f"| {config_label} | {prong_label} | {source_var} | {transfer_test} | "
                    f"{region_label} | {cr_numerator:.3f} | {cr_denominator:.3f} | "
                    f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                    f"{validation_fail_input:.3f} | {predicted:.3f} | {target:.3f} | "
                    f"{ratio:.3f} |"
                )
        else:
            summary_lines.append(
                "| sideband transfer | - | - | skipped | no cached sideband histograms and "
                "event loops disabled | nan | nan | 0 | 0 | nan | nan | nan | nan |"
            )

        if taupt_met_feasibility_rows:
            summary_lines.extend(
                [
                    "",
                    "### TauPt x MET_met fake-factor feasibility",
                    "",
                    "A literal two-dimensional `(TauPt, MET_met)` fake factor cannot be "
                    "independently derived in `MET < 120` and then applied in "
                    "`120 <= MET < 170` using the same MET bins, because the derivation and "
                    "validation MET bins do not overlap. The current smoke test therefore uses "
                    "the selected MET-transfer row to decide whether MET dependence is worth "
                    "modelling, rather than claiming a validated 2D fake factor.",
                    "",
                    "| Configuration | Prong | Candidate source model | Derivation MET region | "
                    "Validation MET region | Status |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for (
                config_label,
                prong_label,
                source_model,
                derivation_met,
                validation_met,
                status,
            ) in taupt_met_feasibility_rows:
                summary_lines.append(
                    f"| {config_label} | {prong_label} | {source_model} | "
                    f"{derivation_met} | {validation_met} | {status} |"
                )

        large_transfer_mismatches = [
            row for row in transfer_validation_rows if np.isfinite(row[-1]) and row[-1] > 1.5
        ]
        negative_source_shift = {
            treatment: predicted
            for (
                _config_label,
                _prong_label,
                _source_var,
                treatment,
                _cr_numerator,
                _cr_denominator,
                _negative_numerator_bins,
                _tiny_denominator_bins,
                _validation_fail_input,
                predicted,
                _target,
                _ratio,
            ) in negative_bin_rows
        }
        nominal_negative_prediction = negative_source_shift.get("nominal signed", float("nan"))
        floored_negative_prediction = negative_source_shift.get("floor negative bins", float("nan"))
        merged_negative_prediction = negative_source_shift.get("merge 170-250 GeV", float("nan"))
        summary_lines.extend(
            [
                "",
                "### Recommendation",
                "",
            ]
        )
        if transfer_validation_rows and large_transfer_mismatches:
            summary_lines.append(
                "The sideband transfer rows contain prediction/target ratios above `1.5`, "
                "so the `TauPt` fake factor should not yet be treated as validated in the "
                "shadow-bin phase space. The next defensible analysis development would be "
                "a sideband-validated MET-dependence model. A literal 2D `(TauPt, MET_met)` "
                "fake factor needs a derivation region with overlapping MET bins, or it should "
                "be treated as a systematic/threshold-variation study rather than a nominal "
                "correction."
            )
        elif transfer_validation_rows:
            summary_lines.append(
                "The sideband transfer rows do not show a large integral overprediction by "
                "this simple threshold test. If shapes are also acceptable, the current "
                "`TauPt` fake factor may be adequate for the tested sidebands."
            )
        else:
            summary_lines.append(
                "No sideband transfer rows were produced in this run. The cache-only checks "
                "remain useful, but they are not enough to validate fake-factor transfer."
            )
        summary_lines.append(
            "`MTW` as a fake-factor source remains diagnostic only, because `MTW` is the "
            "measured observable and using it directly could sculpt the unfolded spectrum."
        )
        if np.isfinite(nominal_negative_prediction):
            summary_lines.append(
                "For the no-shadow 3-prong source-variable diagnostic, the nominal signed "
                f"prediction is `{nominal_negative_prediction:.3f}`, the negative-bin-floored "
                f"prediction is `{floored_negative_prediction:.3f}`, and the merged low-`TauPt` "
                f"prediction is `{merged_negative_prediction:.3f}`. This should be used to "
                "decide whether bin merging is worth testing in a later nominal-analysis variant, "
                "not as an automatic correction."
            )

    summary_path = output_root / "shadow_fake_validation_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    validator.logger.info("Saved fake validation summary to %s", summary_path)

    if not loaded_analysis_hists:
        analysis.save_hists()

    validator.logger.info("DONE.")
