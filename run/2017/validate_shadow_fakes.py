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
RUN_EVENT_LEVEL_PRONG_DIAGNOSTICS = True  # One wtaunu_had pass for weighted/unweighted prong checks.
PLOT_CONTROL_REGION_COMPOSITION = False  # Plot data and nonfake MC in fake-factor regions.
PLOT_PASS_ID_VALIDATION = False  # Plot pass-ID data-minus-nonfake against fake prediction.
PLOT_FAKE_FACTORS = False  # Plot the 1-prong and 3-prong fake factors.
PLOT_NONFAKE_COMPONENTS = False  # Plot per-sample nonfake MC in the pass-ID signal region.


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

    # SELECTION BUILDING
    # ========================================================================
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    selection_binnings: dict[str, dict[str, np.ndarray]] = {"": BINNINGS}

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
    pass_validation_rows: list[tuple[str, str, float, float, float]] = []

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
            double mtwThresholds[3] = {{{mtw_thresholds}}};
            double tauptThresholds[3] = {{{taupt_thresholds}}};
            double metThresholds[3] = {{{met_thresholds}}};

            void addAtomic(std::atomic<double>& target, double value) {{
                double current = target.load(std::memory_order_relaxed);
                while (!target.compare_exchange_weak(
                    current, current + value, std::memory_order_relaxed
                )) {{}}
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
                        }}
                    }}
                }}
            }}

            template <typename SampleID, typename PassTruth, typename TruthHad,
                      typename TruthNTracks, typename VisPt, typename TruthMtw,
                      typename NuPt, typename TruthEta, typename TruthWeight,
                      typename PassReco, typename TauBaseline, typename TauCharge,
                      typename PassMetTrigger, typename BadJet, typename MatchedTau,
                      typename MatchedHadTau, typename MatchedMuon, typename MatchedElectron,
                      typename MatchedPhoton, typename RecoNTracks, typename TauPt,
                      typename TauEta, typename RecoMtw, typename Met,
                      typename EleScore, typename JetScore, typename RecoWeight>
            int recordProngDiagnostics(
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
                RecoWeight recoWeight
            ) {{
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
                "SampleID, passTruth, TruthTau_isHadronic, TruthTau_nChargedTracks, "
                "VisTruthTauPt, TruthMTW, TruthNeutrinoPt, VisTruthTauEta, truth_weight, "
                "passReco, TauBaselineWP, TauCharge, passMetTrigger, badJet, "
                "MatchedTruthParticle_isTau, MatchedTruthParticle_isHadronicTau, "
                "MatchedTruthParticle_isMuon, MatchedTruthParticle_isElectron, "
                "MatchedTruthParticle_isPhoton, TauNCoreTracks, TauPt, TauEta, MTW, "
                "MET_met, TauBDTEleScore, TauRNNJetScore, reco_weight)",
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

    summary_path = output_root / "shadow_fake_validation_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    validator.logger.info("Saved fake validation summary to %s", summary_path)

    if not loaded_analysis_hists:
        analysis.save_hists()

    validator.logger.info("DONE.")
