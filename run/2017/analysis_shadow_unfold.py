"""Produce the reduced 2017 shadow-bin unfolding workflow for the MTW study.

This script deliberately consolidates the pieces that used to be spread across
the simple stack, fake-estimate, efficiency/acceptance, and unfolding scripts.
The order below is the intended reading order:

1. define the physics configuration and fake-control regions;
2. construct all reconstructed, MC-contamination, and response selections;
3. build or load the measured and response histogram caches;
4. build prong-split data-driven jet-fake estimates;
5. construct the background-subtracted reconstructed input;
6. propagate fake-source and optional response systematics;
7. unfold, plot, and write a compact markdown summary.

Validation-only studies should live under ``run/2017/validations``. This file is
the production-style runner for the current candidate analysis prescription.
"""

import re
from pathlib import Path

import numpy as np
import ROOT
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples, signal_sample
from shadow_unfold.models import FakeControlRegion, ResponseComponents, ShadowConfig
from shadow_unfold.selections import (
    PASS_ETA,
    PASS_RECO_PRESELECTION,
    build_fiducial_truth_cuts,
    build_reco_sr_cuts,
)
from shadow_unfold.systematics import (
    JET_FAKE_FF_STAT,
    JET_FAKE_MET_WINDOW,
    JET_FAKE_TAU_WIDTH_COMPOSITION,
    build_fake_factor_stat_systematic,
    build_met_window_fake_systematic,
    build_tau_width_fake_systematic,
    build_tes_response_systematics,
    histogram_has_finite_content,
    quadrature_sum_histograms,
)

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

# Core measurement scope. Keep this intentionally narrow: the thesis-level
# measurement target is MTW, and the shadow-bin study currently compares only
# the no-shadow case and one MTW shadow threshold.
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
    4,
    # 8,
)
FAKES_SOURCE = "TauPt"


def response_cache_mismatches_selections(
    cache_file: Path,
    selections: dict[str, list[Cut]],
) -> list[str]:
    """Return response selections whose cached cutflow no longer matches the script."""
    if not cache_file.is_file():
        return [f"{cache_file} is missing"]

    mismatches: list[str] = []
    with ROOT.TFile(str(cache_file), "READ") as root_file:
        for selection, cuts in selections.items():
            cutflow = root_file.Get(f"{NOMINAL_NAME}/{selection}/cutflow")
            if not cutflow:
                mismatches.append(f"{selection}: missing nominal cutflow")
                continue
            if selection.endswith("_truth_reco_tau"):
                # The TES helper stores the fiducial truth requirement as a
                # hard cut for this matched response selection, so its cutflow
                # can have fewer displayed filter bins while representing the
                # same selected phase space. The separate truth_tau and reco_tau
                # selections catch stale truth/reco definitions.
                continue
            if cutflow.GetNbinsX() != len(cuts):
                mismatches.append(
                    f"{selection}: cached cutflow has {cutflow.GetNbinsX()} bins, "
                    f"current selection has {len(cuts)} cuts"
                )
    return mismatches


# Runtime switches. The fake-source switches are intentionally independent:
# they do not change the nominal fake estimate, they only build uncertainty
# envelopes around it.
LOAD_SAVED_HISTS = True  # Reuse saved ROOT histograms instead of rebuilding them.
USE_MC_CONTAMINATION_SUBTRACTION = True  # Replace jet-fake-like MC with data-driven fakes.
DO_FULL_SYSTEMATICS = True  # Enable full systematic response variations; slow final-mode run.
RUN_FAKE_FF_STAT_SYSTEMATIC = True  # Propagate fake-factor bin statistical uncertainty.
RUN_FAKE_MET_WINDOW_SYSTEMATIC = True  # Envelope alternate low-MET fake-factor regions.
RUN_FAKE_WIDTH_SYSTEMATIC = True  # Propagate the validated 1-prong tau-width fake systematic.
FAKE_WIDTH_VARIABLE = "TauTrackWidthPt1000PV"

SKIP_SYS = {
    r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
    r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
}

# CUTS & SELECTIONS
# ========================================================================
# Naming convention:
# - passID/failID are reconstructed tau-ID regions;
# - trueTau_* selections subtract MC contamination in the fake-factor method;
# - truth_tau/truth_reco_tau are the fiducial response selections.

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
PASS_TRUETAU = Cut(
    r"MC contamination",
    "MatchedTruthParticle_isHadronicTau == true || "
    "MatchedTruthParticle_isMuon == true || "
    "MatchedTruthParticle_isElectron == true",
    # This selection defines the MC contamination subtracted in the fake-factor
    # method. The "trueTau_" histogram prefix is a saved-histogram name, not
    # the fiducial signal truth definition.
    # To test photon-matched candidates as MC contamination, uncomment the line below
    # and rebuild measured histograms. Cached trueTau histograms use this active
    # definition and must not be mixed with a different contamination definition.
    # " || MatchedTruthParticle_isPhoton == true",
)


# FAKE CONTROL REGIONS
# ========================================================================
# The active fake control region defines where the fake factor is measured.
# Its pass/fail ID regions are then applied to SR anti-ID events. Keep exactly
# one FAKE_CONTROL_REGION active so the nominal fake prescription is obvious.
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

# These regions are not alternate central models. They are only used to build
# the JET_FAKE_MET_WINDOW envelope around the active low-MET fake estimate.
FAKE_MET_WINDOW_SYSTEMATIC_REGIONS = (
    FakeControlRegion(
        selection_tag="lowMET30to100_CR",
        output_tag="_lowMET30to100",
        shared_across_configs=True,
        cuts=(
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
            PASS_ETA,
            Cut(
                r"$30 <= E_T^{\mathrm{miss}} < 100$ fake-region variation",
                "(MET_met >= 30) && (MET_met < 100)",
            ),
        ),
    ),
    FakeControlRegion(
        selection_tag="lowMET50to100_CR",
        output_tag="_lowMET50to100",
        shared_across_configs=True,
        cuts=(
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
            PASS_ETA,
            Cut(
                r"$50 <= E_T^{\mathrm{miss}} < 100$ fake-region variation",
                "(MET_met >= 50) && (MET_met < 100)",
            ),
        ),
    ),
    FakeControlRegion(
        selection_tag="lowMET70to100_CR",
        output_tag="_lowMET70to100",
        shared_across_configs=True,
        cuts=(
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
            PASS_ETA,
            Cut(
                r"$70 <= E_T^{\mathrm{miss}} < 100$ fake-region variation",
                "(MET_met >= 70) && (MET_met < 100)",
            ),
        ),
    ),
    FakeControlRegion(
        selection_tag="lowMET0to150_CR",
        output_tag="_lowMET0to150",
        shared_across_configs=True,
        cuts=(
            PASS_RECO_PRESELECTION,
            Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
            PASS_ETA,
            Cut(r"$E_T^{\mathrm{miss}} < 150$ fake-region variation", "MET_met < 150"),
        ),
    ),
)


# MODELS & CONFIGURATION
# ========================================================================
# A ShadowConfig defines a complete measured/reconstructed/truth phase space.
# ``no_shadow_bin`` uses the nominal SR cuts; ``MTW_shadow_bin_250`` lowers the
# reconstructed and truth MTW threshold so migrations through the nominal lower
# edge can be modelled explicitly.
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
    plotter.logger.info("RUN_FAKE_FF_STAT_SYSTEMATIC = %s", RUN_FAKE_FF_STAT_SYSTEMATIC)
    plotter.logger.info("RUN_FAKE_MET_WINDOW_SYSTEMATIC = %s", RUN_FAKE_MET_WINDOW_SYSTEMATIC)
    plotter.logger.info("RUN_FAKE_WIDTH_SYSTEMATIC = %s", RUN_FAKE_WIDTH_SYSTEMATIC)
    if DO_FULL_SYSTEMATICS:
        plotter.logger.info(
            "Full systematics mode enabled; missing shadow variations will fail loudly."
        )

    # SELECTION BUILDING
    # ========================================================================
    # Build selection dictionaries for the two Analysis objects below:
    #
    # measured_analysis:
    #   data and all MC backgrounds in reconstructed SR/fake-factor regions.
    # response_analysis:
    #   signal-only truth, reco, and truth+reco selections used to construct
    #   RooUnfoldResponse objects.
    #
    # This is verbose because every config needs inclusive selections, prong
    # splits, MC-contamination versions, and optional MET-window systematic CRs.
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    response_selections: dict[str, list[Cut]] = {}

    # The tau-width systematic needs an extra histogram variable and a dedicated
    # binning. If the switch is off, the normal binnings are left untouched.
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
        # Selection-name stems. These strings are intentionally repeated later
        # when reading histograms; keeping the format local makes cache names
        # predictable and easier to grep.
        sr_pass = f"{config.label}_{WP}_SR_passID"
        sr_fail = f"{config.label}_{WP}_SR_failID"
        fake_cr_pass = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_passID"
        fake_cr_fail = f"{config.label}_{WP}_{FAKE_CONTROL_REGION.selection_tag}_failID"
        true_sr_pass = f"trueTau_{sr_pass}"
        true_sr_fail = f"trueTau_{sr_fail}"
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

        # Reconstructed SR cuts define the data spectrum to be unfolded and the
        # reco side of the response. Shadow configs lower one threshold here.
        reco_sr_cuts = build_reco_sr_cuts(config)

        # Fake regions are formatted per config so the thesis-style CR can use
        # config thresholds, while the active low-MET region keeps fixed cuts.
        fake_regions_for_config = [FAKE_CONTROL_REGION]
        if RUN_FAKE_MET_WINDOW_SYSTEMATIC:
            fake_regions_for_config.extend(FAKE_MET_WINDOW_SYSTEMATIC_REGIONS)
        fake_region_cuts = {
            fake_region.selection_tag: [
                Cut(
                    cut.name,
                    cut.cutstr.format(
                        taupt_min=config.taupt_min,
                        mtw_min=config.mtw_min,
                        met_min=config.met_min,
                    ),
                )
                for cut in fake_region.cuts
            ]
            for fake_region in fake_regions_for_config
        }
        fake_cr_cuts = fake_region_cuts[FAKE_CONTROL_REGION.selection_tag]

        # Fiducial truth phase space. This must match the reported measurement
        # target, not merely the reconstructed selection.
        truth_cuts = build_fiducial_truth_cuts(config)

        data_selections[sr_pass] = reco_sr_cuts + [PASS_MEDIUM]
        data_selections[sr_fail] = reco_sr_cuts + [FAIL_MEDIUM]
        plotter.logger.info("Fake control region cuts for %s:", config.label)
        for cut in fake_cr_cuts:
            plotter.logger.info("  %s: %s", cut.name, cut.cutstr)

        # Create inclusive pass/fail fake-control selections for every active
        # fake-region definition, then create prong-split names used by the
        # fake-factor estimate.
        fake_region_names: dict[str, dict[int, dict[str, str]]] = {}
        for fake_region in fake_regions_for_config:
            region_pass = f"{config.label}_{WP}_{fake_region.selection_tag}_passID"
            region_fail = f"{config.label}_{WP}_{fake_region.selection_tag}_failID"
            region_cuts = fake_region_cuts[fake_region.selection_tag]
            data_selections[region_pass] = region_cuts + [PASS_MEDIUM]
            data_selections[region_fail] = region_cuts + [FAIL_MEDIUM]
            fake_region_names[fake_region.selection_tag] = {}
            for prong in (1, 3):
                fake_region_names[fake_region.selection_tag][prong] = {
                    "fake_cr_pass": (
                        f"{config.label}_{WP}_{prong}prong_{fake_region.selection_tag}_passID"
                    ),
                    "fake_cr_fail": (
                        f"{config.label}_{WP}_{prong}prong_{fake_region.selection_tag}_failID"
                    ),
                }

        # Prong-split fake factors are the nominal method. The SR pass/fail
        # selections and every fake CR variant need 1-prong and 3-prong versions.
        for prong, names in prong_names.items():
            pass_prong = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
            data_selections[names["sr_pass"]] = data_selections[sr_pass] + [pass_prong]
            data_selections[names["sr_fail"]] = data_selections[sr_fail] + [pass_prong]
            for fake_region in fake_regions_for_config:
                region_pass = f"{config.label}_{WP}_{fake_region.selection_tag}_passID"
                region_fail = f"{config.label}_{WP}_{fake_region.selection_tag}_failID"
                region_prong_names = fake_region_names[fake_region.selection_tag][prong]
                data_selections[region_prong_names["fake_cr_pass"]] = data_selections[
                    region_pass
                ] + [pass_prong]
                data_selections[region_prong_names["fake_cr_fail"]] = data_selections[
                    region_fail
                ] + [pass_prong]

        # Data selections are mirrored into MC selections so all backgrounds can
        # be stacked in the same reconstructed regions.
        measured_selections = [sr_pass, sr_fail]
        for fake_region in fake_regions_for_config:
            measured_selections.extend(
                [
                    f"{config.label}_{WP}_{fake_region.selection_tag}_passID",
                    f"{config.label}_{WP}_{fake_region.selection_tag}_failID",
                ]
            )
        for selection in measured_selections:
            mc_selections[selection] = data_selections[selection]
        for names in prong_names.values():
            prong_selections = [
                names["sr_pass"],
                names["sr_fail"],
            ]
            for selection in prong_selections:
                mc_selections[selection] = data_selections[selection]
        for region_prong_names in fake_region_names.values():
            for names in region_prong_names.values():
                for selection in (names["fake_cr_pass"], names["fake_cr_fail"]):
                    mc_selections[selection] = data_selections[selection]

        # MC-contamination selections are the simulated real-object component
        # subtracted from the fake-factor numerator, denominator, and SR anti-ID
        # application region.
        mc_selections[true_sr_pass] = data_selections[sr_pass] + [PASS_TRUETAU]
        mc_selections[true_sr_fail] = data_selections[sr_fail] + [PASS_TRUETAU]
        for fake_region in fake_regions_for_config:
            region_pass = f"{config.label}_{WP}_{fake_region.selection_tag}_passID"
            region_fail = f"{config.label}_{WP}_{fake_region.selection_tag}_failID"
            mc_selections[f"trueTau_{region_pass}"] = data_selections[region_pass] + [PASS_TRUETAU]
            mc_selections[f"trueTau_{region_fail}"] = data_selections[region_fail] + [PASS_TRUETAU]
        for names in prong_names.values():
            mc_selections[f"trueTau_{names['sr_pass']}"] = data_selections[names["sr_pass"]] + [
                PASS_TRUETAU
            ]
            mc_selections[f"trueTau_{names['sr_fail']}"] = data_selections[names["sr_fail"]] + [
                PASS_TRUETAU
            ]
        for region_prong_names in fake_region_names.values():
            for names in region_prong_names.values():
                mc_selections[f"trueTau_{names['fake_cr_pass']}"] = data_selections[
                    names["fake_cr_pass"]
                ] + [PASS_TRUETAU]
                mc_selections[f"trueTau_{names['fake_cr_fail']}"] = data_selections[
                    names["fake_cr_fail"]
                ] + [PASS_TRUETAU]

        # Response selections are signal-only: truth, reco, and matched
        # truth+reco. The last one fills both the reco projection and migration
        # matrix used by RooUnfold.
        response_selections[truth_selection] = truth_cuts
        response_selections[reco_selection] = reco_sr_cuts + [PASS_MEDIUM]
        response_selections[truth_reco_selection] = truth_cuts + reco_sr_cuts + [PASS_MEDIUM]

        # Shadow-bin configs require config-local MTW bin edges. TruthMTW must
        # use the same edges as reco MTW so the response matrix is well-defined.
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

        # Register config-specific binnings for every generated selection name.
        # Regex keys are used because the Analysis object resolves binnings by
        # selection pattern.
        for selection in (
            sr_pass,
            sr_fail,
            fake_cr_pass,
            fake_cr_fail,
            true_sr_pass,
            true_sr_fail,
            truth_selection,
            reco_selection,
            truth_reco_selection,
        ):
            selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
        for fake_region in fake_regions_for_config:
            for selection in (
                f"{config.label}_{WP}_{fake_region.selection_tag}_passID",
                f"{config.label}_{WP}_{fake_region.selection_tag}_failID",
                f"trueTau_{config.label}_{WP}_{fake_region.selection_tag}_passID",
                f"trueTau_{config.label}_{WP}_{fake_region.selection_tag}_failID",
            ):
                selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
        for names in prong_names.values():
            prong_selections = [
                names["sr_pass"],
                names["sr_fail"],
            ]
            for selection in prong_selections:
                selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
                selection_binnings[rf"^{re.escape(f'trueTau_{selection}')}$"] = config_binnings
        for region_prong_names in fake_region_names.values():
            for names in region_prong_names.values():
                for selection in (names["fake_cr_pass"], names["fake_cr_fail"]):
                    selection_binnings[rf"^{re.escape(selection)}$"] = config_binnings
                    selection_binnings[rf"^{re.escape(f'trueTau_{selection}')}$"] = config_binnings

    label_regex = "|".join(re.escape(config.label) for config in CONFIGS)
    truth_histogram_vars = {variable_data[var]["truth"] for var in VARS}

    # DATAFRAME & HISTOGRAM PRODUCTION
    # ========================================================================
    # measured_analysis owns data, all MC backgrounds, MC-contamination
    # selections, fake-factor internals, fake predictions, and measured-input
    # systematic variations.
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

    # response_analysis owns only the signal sample and all histograms required
    # to build the response matrix. It is intentionally separate so the response
    # cache is not polluted with background/fake-control selections.
    #
    # The response cache must match the exact current selection definitions.
    # Selection names are deliberately stable, so a stale ROOT cache can contain
    # a same-named cutflow from an older definition. Rebuild the response cache
    # once when that happens; do not import stale response histograms.
    response_cache_file = output_root / "response/root/wtaunu_had.root"
    response_cache_mismatches = (
        response_cache_mismatches_selections(response_cache_file, response_selections)
        if LOAD_SAVED_HISTS
        else []
    )
    for mismatch in response_cache_mismatches:
        plotter.logger.warning("Response cache mismatch: %s", mismatch)
    rebuild_response_hists = (not LOAD_SAVED_HISTS) or bool(response_cache_mismatches)

    response_analysis_kwargs = {
        "data_dict": {"wtaunu_had": signal_sample(selections=response_selections)},
        "year": YEAR,
        "do_systematics": DO_FULL_SYSTEMATICS,
        "metadata_cache": DSID_METADATA_CACHE,
        "ttree": NOMINAL_NAME,
        "output_dir": output_root / "response",
        "log_level": 10,
        "log_out": "both",
        "extract_vars": {
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
        "import_missing_columns_as_nan": True,
        "snapshot": False,
        "histogram_vars": set(VARS) | truth_histogram_vars,
        "hists_2d": {
            f"{var}_{variable_data[var]['truth']}": Hist2dOpts(
                var,
                variable_data[var]["truth"],
                "reco_weight",
            )
            for var in VARS
        },
        "do_unweighted": True,
        "systematics_for_selection": {rf"^({label_regex})_{WP}_(reco_tau|truth_reco_tau)$"}
        if DO_FULL_SYSTEMATICS
        else set(),
        "skip_sys": SKIP_SYS,
        "binnings": selection_binnings,
    }

    if rebuild_response_hists:
        plotter.logger.info("Rebuilding response histogram cache before import.")
        Analysis(
            analysis_label="analysis_shadow_unfold_response_build",
            rerun=True,
            regen_histograms=True,
            **response_analysis_kwargs,
        )

    # TES shifted trees need nominal truth-fiducial event masks before their
    # full response objects can be written. Build those response histograms into
    # the normal response cache automatically in full-systematics mode. The
    # final response_analysis import below then sees both the standard response
    # histograms and the helper-built TES response histograms.
    if DO_FULL_SYSTEMATICS:
        plotter.logger.info("Ensuring TES response-systematic histograms are available...")
        build_tes_response_systematics(
            configs=CONFIGS,
            vars_to_build=VARS,
            pass_medium=PASS_MEDIUM,
            skip_sys=SKIP_SYS,
            output_root=output_root,
            wp=WP,
            year=YEAR,
            log_out="both",
        )

    response_analysis = Analysis(
        analysis_label="analysis_shadow_unfold_response",
        rerun=False,
        regen_histograms=False,
        **response_analysis_kwargs,
    )

    # The no-shadow nominal truth histogram is the reference truth spectrum for
    # scaled/cropped unfolded results, including shadow-bin closure comparisons.
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
        tuple[
            str,
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    fake_width_rows: list[
        tuple[
            str,
            str,
            str,
            str,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    fake_source_systematic_rows: list[
        tuple[
            str,
            str,
            str,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    fake_factor_cache: dict[tuple[str, str, tuple[float, ...]], ROOT.TH1] = {}
    generated_missing_fake_hists = False

    # FAKE ESTIMATION, UNFOLDING & DIAGNOSTICS
    # ========================================================================
    # The outer loop runs one complete analysis prescription per phase-space
    # config. The inner variable loop then constructs a response and unfolds the
    # configured observable in that phase space.
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
        truth_selection = f"{config.label}_truth_tau"
        reco_selection = f"{config.label}_{WP}_reco_tau"
        truth_reco_selection = f"{config.label}_{WP}_truth_reco_tau"
        fakes_name = f"{config.label}_{WP}{FAKE_CONTROL_REGION.output_tag}"

        plotter.logger.info("Running variable-specific shadow-bin closure for %s", config.label)
        plotter.logger.info(
            "Using fake control region '%s' for %s.",
            FAKE_CONTROL_REGION.selection_tag,
            config.label,
        )

        # DATA-DRIVEN FAKE ESTIMATES
        # --------------------------------------------------------------------
        # Build fake factors before entering the variable loop so one source FF
        # can be applied to all target variables for the config. With the active
        # low-MET region, fake factors are shared across configs and only the SR
        # application changes.
        fake_estimate_regions = [FAKE_CONTROL_REGION]
        if RUN_FAKE_MET_WINDOW_SYSTEMATIC:
            fake_estimate_regions.extend(FAKE_MET_WINDOW_SYSTEMATIC_REGIONS)

        for fake_region in fake_estimate_regions:
            region_fake_cr_pass = f"{config.label}_{WP}_{fake_region.selection_tag}_passID"
            ff_bins = measured_analysis[measured_analysis.data_sample].get_binnings(
                FAKES_SOURCE,
                region_fake_cr_pass,
            )["bins"]
            ff_bin_key = tuple(float(bin_edge) for bin_edge in ff_bins)

            # The nominal fake estimate follows the thesis method: split by tau prong
            # and then sum the 1-prong and 3-prong predictions. Systematic variants use
            # the same application region but derive the fake factor in alternate CRs.
            for prong in (1, 3):
                prong_fakes_name = f"{config.label}_{WP}_{prong}prong{fake_region.output_tag}"
                expected_fakes = [
                    f"{prong_fakes_name}_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                    for var in vars_for_config
                ]
                expected_fakes.append(f"{prong_fakes_name}_{FAKES_SOURCE}_FF")
                if all(name in measured_analysis.histograms for name in expected_fakes):
                    continue

                # Missing fake products are generated even in LOAD_SAVED_HISTS
                # mode. This lets old caches be reused while still adding new
                # systematic-only fake-control regions.
                generated_missing_fake_hists = True
                prong_fake_cr_pass = (
                    f"{config.label}_{WP}_{prong}prong_{fake_region.selection_tag}_passID"
                )
                prong_fake_cr_fail = (
                    f"{config.label}_{WP}_{prong}prong_{fake_region.selection_tag}_failID"
                )
                prong_sr_pass = f"{config.label}_{WP}_{prong}prong_SR_passID"
                prong_sr_fail = f"{config.label}_{WP}_{prong}prong_SR_failID"
                true_prong_sr_fail = f"trueTau_{prong_sr_fail}"
                cache_key = (
                    fake_region.selection_tag,
                    f"{prong}prong",
                    ff_bin_key,
                )

                if fake_region.shared_across_configs and cache_key in fake_factor_cache:
                    # Same FF derivation region and source binning: reuse the
                    # cached FF and only apply it to the config-specific SR
                    # anti-ID events.
                    cached_ff = fake_factor_cache[cache_key]
                    current_ff = cached_ff.Clone(f"{prong_fakes_name}_{FAKES_SOURCE}_FF")
                    current_ff.SetDirectory(0)
                    measured_analysis.histograms[current_ff.GetName()] = current_ff
                    plotter.logger.info(
                        "Reusing cached %s-prong fake factor for %s from '%s'.",
                        prong,
                        config.label,
                        fake_region.selection_tag,
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
                    # Build the FF from data minus MC-contamination in pass/fail
                    # CR selections, then apply it to SR anti-ID events. The
                    # Analysis API delegates to the batched implementation.
                    measured_analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        vars_for_config,
                        prong_fake_cr_pass,
                        prong_fake_cr_fail,
                        prong_sr_pass,
                        prong_sr_fail,
                        f"trueTau_{prong_fake_cr_pass}",
                        f"trueTau_{prong_fake_cr_fail}",
                        true_prong_sr_fail,
                        name=prong_fakes_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )
                    if fake_region.shared_across_configs:
                        cached_ff = measured_analysis.histograms[
                            f"{prong_fakes_name}_{FAKES_SOURCE}_FF"
                        ].Clone(f"cached_{fake_region.selection_tag}_{prong}prong_FF")
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
            # The nominal measured input is:
            #   data - MC-contamination backgrounds - data-driven jet fakes
            #        - reconstructed nonfiducial signal.
            #
            # ``all_background`` is retained only as an all-MC background
            # subtraction diagnostic.
            truth_var = variable_data[var]["truth"]
            response_matrix_name = f"{var}_{truth_var}"
            data = measured_analysis.get_hist(
                var,
                dataset=measured_analysis.data_sample,
                systematic=NOMINAL_NAME,
                selection=sr_pass,
            )
            all_backgrounds = [
                measured_analysis.get_hist(
                    var,
                    dataset=background,
                    systematic=NOMINAL_NAME,
                    selection=sr_pass,
                )
                for background in measured_analysis.mc_samples
                if background != "wtaunu_had"
            ]
            mc_contamination_backgrounds = [
                measured_analysis.get_hist(
                    var,
                    dataset=background,
                    systematic=NOMINAL_NAME,
                    selection=true_sr_pass,
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

            # Nonfiducial signal is reconstructed signal that passes reco SR
            # but lies outside the fiducial truth definition. It is subtracted
            # before unfolding so the measured input targets the same fiducial
            # phase space as the truth spectrum.
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

            all_background = sum_th1s(*all_backgrounds)
            all_background.SetName(f"{config.label}_{var}_all_mc_background")
            all_background.SetDirectory(0)
            mc_contamination_background = sum_th1s(*mc_contamination_backgrounds)
            mc_contamination_background.SetName(
                f"{config.label}_{var}_mc_contamination_background"
            )
            mc_contamination_background.SetDirectory(0)
            jet_fake_like_background = all_background - mc_contamination_background
            jet_fake_like_background.SetName(f"{config.label}_{var}_jet_fake_like_mc_background")
            jet_fake_like_background.SetDirectory(0)

            # Literature-aligned bookkeeping: the data-driven fake estimate
            # replaces the jet-fake-like MC component. Setting
            # USE_MC_CONTAMINATION_SUBTRACTION=False switches to all-MC
            # background subtraction for comparison/debugging.
            nominal_background = (
                mc_contamination_background if USE_MC_CONTAMINATION_SUBTRACTION else all_background
            )
            background = sum_th1s(nominal_background, fakes)
            data_sig = data - background - nonfiducial_signal
            data_sig.SetName(f"{config.label}_{var}_data_minus_background_nonfiducial")
            signal = fiducial_reco_signal.Clone(f"{config.label}_{var}_fiducial_signal")
            signal.SetDirectory(0)

            all_background_with_fakes = sum_th1s(all_background, fakes)
            data_sig_all_bkg_with_fakes = data - all_background_with_fakes - nonfiducial_signal
            data_sig_all_bkg_with_fakes.SetName(
                f"{config.label}_{var}_data_minus_all_background_fakes_nonfiducial"
            )
            data_sig_no_fake = data - mc_contamination_background - nonfiducial_signal
            data_sig_no_fake.SetName(
                f"{config.label}_{var}_data_minus_mc_contamination_nonfiducial"
            )
            fake_budget_rows.append(
                (
                    config.label,
                    var,
                    data.Integral(),
                    all_background.Integral(),
                    mc_contamination_background.Integral(),
                    jet_fake_like_background.Integral(),
                    fakes.Integral(),
                    nonfiducial_signal.Integral(),
                    data_sig.Integral(),
                    data_sig_all_bkg_with_fakes.Integral(),
                    data_sig_no_fake.Integral(),
                    fiducial_reco_signal.Integral(),
                    fiducial_reco_signal.Integral() / data_sig.Integral()
                    if data_sig.Integral() != 0
                    else float("nan"),
                )
            )

            # Build the nominal response from fiducial reco signal, fiducial
            # truth signal, and the migration matrix between them.
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
            fake_source_relative_uncertainties: dict[int, list[ROOT.TH1]] = {
                iter_count: [] for iter_count in ITERATIONS
            }

            # FAKE-FACTOR STATISTICAL SYSTEMATIC
            # ----------------------------------------------------------------
            # Vary FF bins by their own uncertainties, reapply to SR anti-ID
            # data, unfold the shifted inputs, and store the max deviation from
            # nominal as a per-bin unfolded uncertainty.
            if RUN_FAKE_FF_STAT_SYSTEMATIC:
                ff_stat_systematic = build_fake_factor_stat_systematic(
                    measured_analysis=measured_analysis,
                    plotter=plotter,
                    config_label=config.label,
                    wp=WP,
                    fake_control_region=FAKE_CONTROL_REGION,
                    fakes_source=FAKES_SOURCE,
                    target_var=var,
                    data=data,
                    nominal_background=nominal_background,
                    nonfiducial_signal=nonfiducial_signal,
                    nominal_data_sig=data_sig,
                    response=response,
                    nominal_truth=nominal_truth,
                    iterations=ITERATIONS,
                    lumi=LUMI,
                )
                for prong in (1, 3):
                    prong_prefix = (
                        f"{config.label}_{WP}_{prong}prong{FAKE_CONTROL_REGION.output_tag}"
                    )
                    nominal_ff = measured_analysis.histograms[f"{prong_prefix}_{FAKES_SOURCE}_FF"]
                    ff_up = measured_analysis.histograms[
                        f"{prong_prefix}_{var}_{JET_FAKE_FF_STAT}_up_{FAKES_SOURCE}_FF"
                    ]
                    ff_down = measured_analysis.histograms[
                        f"{prong_prefix}_{var}_{JET_FAKE_FF_STAT}_down_{FAKES_SOURCE}_FF"
                    ]
                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir
                        / "plots"
                        / config.label
                        / var
                        / "fake_source_systematics"
                    )
                    plotter.plot(
                        [nominal_ff, ff_up, ff_down],
                        label=["Nominal FF", "FF stat up", "FF stat down"],
                        colour=["k", "tab:blue", "tab:cyan"],
                        histstyle=["step", "step", "step"],
                        xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            f"{prong}-prong fake-factor stat shifts",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Fake factor",
                        logx=True,
                        label_params={"llabel": "", "loc": 1},
                        filename=(
                            f"{config.label}_{var}_{prong}prong_fake_factor_stat_source.png"
                        ),
                    )
                plotter.plot(
                    [
                        fakes,
                        ff_stat_systematic.shifted_fakes_up,
                        ff_stat_systematic.shifted_fakes_down,
                    ],
                    label=["Nominal fakes", "FF stat up", "FF stat down"],
                    colour=["k", "tab:blue", "tab:cyan"],
                    histstyle=["step", "step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        "fake-factor statistical fake-yield shift",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Predicted fakes",
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_fake_factor_stat_fake_yield.png",
                )
                plotter.plot(
                    [
                        data_sig,
                        ff_stat_systematic.shifted_data_sig_up,
                        ff_stat_systematic.shifted_data_sig_down,
                    ],
                    label=["Nominal data sig", "FF stat up", "FF stat down"],
                    colour=["k", "tab:blue", "tab:cyan"],
                    histstyle=["step", "step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        "fake-factor statistical data-signal shift",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Background-subtracted data",
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=f"{config.label}_{var}_fake_factor_stat_data_sig.png",
                )
                for variation in ff_stat_systematic.variations:
                    fake_source_relative_uncertainties[variation.iter_count].append(
                        variation.relative_uncertainty
                    )
                    nominal_unfolded_integral = variation.nominal_unfolded.Integral()
                    uncertainty_integral = variation.uncertainty.Integral()
                    fake_source_systematic_rows.append(
                        (
                            config.label,
                            var,
                            ff_stat_systematic.name,
                            variation.iter_count,
                            fakes.Integral(),
                            ff_stat_systematic.shifted_fakes_up.Integral(),
                            ff_stat_systematic.shifted_fakes_down.Integral(),
                            data_sig.Integral(),
                            ff_stat_systematic.shifted_data_sig_up.Integral(),
                            ff_stat_systematic.shifted_data_sig_down.Integral(),
                            nominal_unfolded_integral,
                            variation.shifted_unfolded.Integral(),
                            uncertainty_integral,
                            uncertainty_integral / abs(nominal_unfolded_integral)
                            if nominal_unfolded_integral
                            else float("nan"),
                        )
                    )
                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir
                        / "plots"
                        / config.label
                        / var
                        / "fake_source_systematics"
                    )
                    plotter.plot(
                        [variation.relative_uncertainty],
                        label=[JET_FAKE_FF_STAT],
                        colour=["tab:blue"],
                        histstyle=["step"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            "fake-factor statistical uncertainty",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Relative uncertainty / %",
                        logx=True,
                        label_params={"llabel": "", "loc": 1},
                        filename=(
                            f"{config.label}_{var}_{variation.iter_count}iter_"
                            "fake_factor_stat_uncertainty.png"
                        ),
                    )
                    plotter.plot(
                        [
                            truth,
                            variation.nominal_unfolded,
                            variation.shifted_unfolded,
                            variation.shifted_unfolded_down,
                        ],
                        label=[
                            "Truth MC",
                            "Nominal unfolded data",
                            "FF stat up unfolded data",
                            "FF stat down unfolded data",
                        ],
                        colour=["r", "k", "tab:blue", "tab:cyan"],
                        histstyle=["step", "errorbar", "errorbar", "errorbar"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=True,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            "fake-factor statistical unfolded shift",
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
                            f"{config.label}_{var}_{variation.iter_count}iter_"
                            "fake_factor_stat_unfolded_shift.png"
                        ),
                    )

            # MET-WINDOW FAKE-SOURCE SYSTEMATIC
            # ----------------------------------------------------------------
            # Derive the FF in ATLAS-like alternate low-MET windows, take the
            # envelope of the resulting fake predictions, and propagate that
            # envelope through unfolding.
            if RUN_FAKE_MET_WINDOW_SYSTEMATIC:
                met_window_fakes = []
                for fake_region in FAKE_MET_WINDOW_SYSTEMATIC_REGIONS:
                    region_prong_fakes = []
                    for prong in (1, 3):
                        prefix = f"{config.label}_{WP}_{prong}prong{fake_region.output_tag}"
                        hist_name = f"{prefix}_{var}_fakes_bkg_{FAKES_SOURCE}_src"
                        if hist_name in measured_analysis.histograms:
                            region_prong_fakes.append(measured_analysis.histograms[hist_name])
                    if len(region_prong_fakes) == 2:
                        region_fakes = sum_th1s(*region_prong_fakes)
                        region_fakes.SetName(f"{config.label}_{var}{fake_region.output_tag}_fakes")
                        region_fakes.SetDirectory(0)
                        met_window_fakes.append(region_fakes)
                    else:
                        plotter.logger.warning(
                            "Skipping MET-window fake systematic for %s %s %s; "
                            "missing one or more prong fake predictions.",
                            config.label,
                            var,
                            fake_region.selection_tag,
                        )
                met_window_systematic = build_met_window_fake_systematic(
                    measured_analysis=measured_analysis,
                    plotter=plotter,
                    config_label=config.label,
                    target_var=var,
                    data=data,
                    nominal_background=nominal_background,
                    nonfiducial_signal=nonfiducial_signal,
                    nominal_data_sig=data_sig,
                    nominal_fakes=fakes,
                    shifted_fakes=met_window_fakes,
                    response=response,
                    nominal_truth=nominal_truth,
                    iterations=ITERATIONS,
                    lumi=LUMI,
                )
                if met_window_systematic is not None:
                    plotter.paths.plot_dir = (
                        plotter.paths.output_dir
                        / "plots"
                        / config.label
                        / var
                        / "fake_source_systematics"
                    )
                    for prong in (1, 3):
                        nominal_prefix = (
                            f"{config.label}_{WP}_{prong}prong{FAKE_CONTROL_REGION.output_tag}"
                        )
                        ff_hists = [
                            measured_analysis.histograms[f"{nominal_prefix}_{FAKES_SOURCE}_FF"]
                        ]
                        ff_labels = ["0 <= MET < 100"]
                        ff_colours = ["k"]
                        for fake_region, colour in zip(
                            FAKE_MET_WINDOW_SYSTEMATIC_REGIONS,
                            ["tab:green", "tab:olive", "tab:purple", "tab:brown"],
                            strict=True,
                        ):
                            prefix = f"{config.label}_{WP}_{prong}prong{fake_region.output_tag}"
                            hist_name = f"{prefix}_{FAKES_SOURCE}_FF"
                            if hist_name in measured_analysis.histograms:
                                ff_hists.append(measured_analysis.histograms[hist_name])
                                ff_labels.append(fake_region.selection_tag)
                                ff_colours.append(colour)
                        plotter.plot(
                            ff_hists,
                            label=ff_labels,
                            colour=ff_colours,
                            histstyle=["step"] * len(ff_hists),
                            xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                            kind="overlay",
                            do_stat=False,
                            do_syst=False,
                            title=smart_join(
                                config.label,
                                var,
                                f"{prong}-prong MET-window fake factors",
                                sep=" | ",
                            ),
                            scale_by_bin_width=False,
                            ylabel="Fake factor",
                            logx=True,
                            label_params={"llabel": "", "loc": 1},
                            filename=(
                                f"{config.label}_{var}_{prong}prong_met_window_fake_factors.png"
                            ),
                        )
                    plotter.plot(
                        [
                            fakes,
                            met_window_systematic.shifted_fakes_up,
                            met_window_systematic.shifted_fakes_down,
                        ],
                        label=[
                            "Nominal fakes",
                            "MET-window envelope up",
                            "MET-window envelope down",
                        ],
                        colour=["k", "tab:green", "tab:olive"],
                        histstyle=["step", "step", "step"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            "MET-window fake-yield shift",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Predicted fakes",
                        logx=True,
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{var}_met_window_fake_yield.png",
                    )
                    plotter.plot(
                        [
                            data_sig,
                            met_window_systematic.shifted_data_sig_up,
                            met_window_systematic.shifted_data_sig_down,
                        ],
                        label=[
                            "Nominal data sig",
                            "MET-window envelope up",
                            "MET-window envelope down",
                        ],
                        colour=["k", "tab:green", "tab:olive"],
                        histstyle=["step", "step", "step"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            "MET-window data-signal shift",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Background-subtracted data",
                        logx=True,
                        label_params={"llabel": "", "loc": 1},
                        filename=f"{config.label}_{var}_met_window_data_sig.png",
                    )
                    for variation in met_window_systematic.variations:
                        fake_source_relative_uncertainties[variation.iter_count].append(
                            variation.relative_uncertainty
                        )
                        nominal_unfolded_integral = variation.nominal_unfolded.Integral()
                        uncertainty_integral = variation.uncertainty.Integral()
                        fake_source_systematic_rows.append(
                            (
                                config.label,
                                var,
                                met_window_systematic.name,
                                variation.iter_count,
                                fakes.Integral(),
                                met_window_systematic.shifted_fakes_up.Integral(),
                                met_window_systematic.shifted_fakes_down.Integral(),
                                data_sig.Integral(),
                                met_window_systematic.shifted_data_sig_up.Integral(),
                                met_window_systematic.shifted_data_sig_down.Integral(),
                                nominal_unfolded_integral,
                                variation.shifted_unfolded.Integral(),
                                uncertainty_integral,
                                uncertainty_integral / abs(nominal_unfolded_integral)
                                if nominal_unfolded_integral
                                else float("nan"),
                            )
                        )
                        plotter.paths.plot_dir = (
                            plotter.paths.output_dir
                            / "plots"
                            / config.label
                            / var
                            / "fake_source_systematics"
                        )
                        plotter.plot(
                            [variation.relative_uncertainty],
                            label=[JET_FAKE_MET_WINDOW],
                            colour=["tab:green"],
                            histstyle=["step"],
                            xlabel=variable_data[var]["name"] + " [GeV]",
                            kind="overlay",
                            do_stat=False,
                            do_syst=False,
                            title=smart_join(
                                config.label,
                                var,
                                "MET-window fake-source uncertainty",
                                sep=" | ",
                            ),
                            scale_by_bin_width=False,
                            ylabel="Relative uncertainty / %",
                            logx=True,
                            label_params={"llabel": "", "loc": 1},
                            filename=(
                                f"{config.label}_{var}_{variation.iter_count}iter_"
                                "met_window_uncertainty.png"
                            ),
                        )
                        plotter.plot(
                            [
                                truth,
                                variation.nominal_unfolded,
                                variation.shifted_unfolded,
                                variation.shifted_unfolded_down,
                            ],
                            label=[
                                "Truth MC",
                                "Nominal unfolded data",
                                "MET-window up unfolded data",
                                "MET-window down unfolded data",
                            ],
                            colour=["r", "k", "tab:green", "tab:olive"],
                            histstyle=["step", "errorbar", "errorbar", "errorbar"],
                            xlabel=variable_data[var]["name"] + " [GeV]",
                            kind="overlay",
                            do_stat=True,
                            do_syst=False,
                            title=smart_join(
                                config.label,
                                var,
                                "MET-window unfolded shift",
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
                                f"{config.label}_{var}_{variation.iter_count}iter_"
                                "met_window_unfolded_shift.png"
                            ),
                        )

            # 1-PRONG FAKE-SOURCE COMPOSITION SYSTEMATIC
            # ----------------------------------------------------------------
            # The tau-width validation showed a meaningful 1-prong composition
            # difference between the low-MET derivation region and the SR
            # anti-ID application region. Treat that as an uncertainty envelope,
            # not a central correction.
            if RUN_FAKE_WIDTH_SYSTEMATIC:
                tau_width_systematic = build_tau_width_fake_systematic(
                    measured_analysis=measured_analysis,
                    plotter=plotter,
                    config_label=config.label,
                    wp=WP,
                    fake_control_region=FAKE_CONTROL_REGION,
                    fakes_source=FAKES_SOURCE,
                    target_var=var,
                    data=data,
                    nominal_background=nominal_background,
                    nonfiducial_signal=nonfiducial_signal,
                    nominal_data_sig=data_sig,
                    nominal_fakes=fakes,
                    three_prong_fakes=prong_fakes[1],
                    response=response,
                    nominal_truth=nominal_truth,
                    iterations=ITERATIONS,
                    width_variable=FAKE_WIDTH_VARIABLE,
                    lumi=LUMI,
                )
                plotter.paths.plot_dir = (
                    plotter.paths.output_dir
                    / "plots"
                    / config.label
                    / var
                    / "fake_width_systematic"
                )
                plotter.plot(
                    [tau_width_systematic.width_ratio],
                    label=[f"{tau_width_systematic.width_variable} application/low-MET"],
                    colour=["tab:orange"],
                    histstyle=["step"],
                    xlabel=tau_width_systematic.width_variable,
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        "tau-width transfer weight",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Shape weight",
                    logx=False,
                    label_params={"llabel": "", "loc": 1},
                    filename=(
                        f"{config.label}_{var}_{tau_width_systematic.width_variable}_"
                        "transfer_weight.png"
                    ),
                )
                plotter.plot(
                    [fakes, tau_width_systematic.shifted_fakes],
                    label=["Nominal fakes", "Tau-width shifted fakes"],
                    colour=["k", "tab:orange"],
                    histstyle=["step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        "tau-width fake-yield shift",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Predicted fakes",
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=(
                        f"{config.label}_{var}_{tau_width_systematic.width_variable}_"
                        "fake_yield.png"
                    ),
                )
                plotter.plot(
                    [data_sig, tau_width_systematic.shifted_data_sig],
                    label=["Nominal data sig", "Tau-width shifted data sig"],
                    colour=["k", "tab:orange"],
                    histstyle=["step", "step"],
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        "tau-width data-signal shift",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Background-subtracted data",
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=(
                        f"{config.label}_{var}_{tau_width_systematic.width_variable}_data_sig.png"
                    ),
                )
                for variation in tau_width_systematic.variations:
                    nominal_unfolded = variation.nominal_unfolded
                    shifted_unfolded = variation.shifted_unfolded
                    uncertainty = variation.uncertainty
                    relative_uncertainty = variation.relative_uncertainty
                    fake_source_relative_uncertainties[variation.iter_count].append(
                        relative_uncertainty
                    )
                    nominal_unfolded_integral = nominal_unfolded.Integral()
                    shifted_unfolded_integral = shifted_unfolded.Integral()
                    unfolded_shift = abs(shifted_unfolded_integral - nominal_unfolded_integral)
                    fake_width_rows.append(
                        (
                            config.label,
                            var,
                            tau_width_systematic.name,
                            tau_width_systematic.width_variable,
                            variation.iter_count,
                            fakes.Integral(),
                            tau_width_systematic.shifted_fakes.Integral(),
                            data_sig.Integral(),
                            tau_width_systematic.shifted_data_sig.Integral(),
                            fiducial_reco_signal.Integral() / data_sig.Integral()
                            if data_sig.Integral() != 0
                            else float("nan"),
                            fiducial_reco_signal.Integral()
                            / tau_width_systematic.shifted_data_sig.Integral()
                            if tau_width_systematic.shifted_data_sig.Integral() != 0
                            else float("nan"),
                            nominal_unfolded_integral,
                            shifted_unfolded_integral,
                            unfolded_shift,
                            unfolded_shift / abs(nominal_unfolded_integral)
                            if nominal_unfolded_integral
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
                            f"{tau_width_systematic.width_variable} shifted unfolded data",
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
                            f"{tau_width_systematic.width_variable} fake-source systematic",
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
                            f"{config.label}_{var}_{tau_width_systematic.width_variable}_"
                            f"{variation.iter_count}iter_fake_width_shift.png"
                        ),
                    )
                    plotter.plot(
                        [relative_uncertainty],
                        label=[JET_FAKE_TAU_WIDTH_COMPOSITION],
                        colour=["tab:orange"],
                        histstyle=["step"],
                        xlabel=variable_data[var]["name"] + " [GeV]",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        title=smart_join(
                            config.label,
                            var,
                            "tau-width fake-source uncertainty",
                            sep=" | ",
                        ),
                        scale_by_bin_width=False,
                        ylabel="Relative uncertainty / %",
                        logx=True,
                        label_params={"llabel": "", "loc": 1},
                        filename=(
                            f"{config.label}_{var}_{tau_width_systematic.width_variable}_"
                            f"{variation.iter_count}iter_fake_width_uncertainty.png"
                        ),
                    )

            # COMBINED FAKE-SOURCE UNCERTAINTY
            # ----------------------------------------------------------------
            # Combine the fake-source relative uncertainties in quadrature, as
            # in the thesis systematic prescription. This is only the fake-source
            # group, not the final total experimental uncertainty.
            for iter_count, uncertainty_hists in fake_source_relative_uncertainties.items():
                if not uncertainty_hists:
                    continue
                combined_fake_source_uncertainty = quadrature_sum_histograms(
                    uncertainty_hists,
                    name=(
                        f"{config.label}_{var}_{iter_count}iter_"
                        "combined_fake_source_relative_uncertainty"
                    ),
                )
                measured_analysis.histograms[combined_fake_source_uncertainty.GetName()] = (
                    combined_fake_source_uncertainty
                )
                plotter.paths.plot_dir = (
                    plotter.paths.output_dir
                    / "plots"
                    / config.label
                    / var
                    / "fake_source_systematics"
                )
                plotter.plot(
                    [*uncertainty_hists, combined_fake_source_uncertainty],
                    label=[
                        *[
                            hist.GetName()
                            .replace(f"{config.label}_{var}_", "")
                            .replace(f"{iter_count}iter_", "")
                            .replace("_relative_uncertainty", "")
                            .replace("_uncertainty_relative", "")
                            for hist in uncertainty_hists
                        ],
                        "Combined fake-source",
                    ],
                    colour=[
                        *["tab:blue", "tab:green", "tab:orange"][: len(uncertainty_hists)],
                        "k",
                    ],
                    histstyle=["step"] * (len(uncertainty_hists) + 1),
                    xlabel=variable_data[var]["name"] + " [GeV]",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    title=smart_join(
                        config.label,
                        var,
                        f"{iter_count}-iteration fake-source uncertainties",
                        sep=" | ",
                    ),
                    scale_by_bin_width=False,
                    ylabel="Relative uncertainty / %",
                    logx=True,
                    label_params={"llabel": "", "loc": 1},
                    filename=(
                        f"{config.label}_{var}_{iter_count}iter_"
                        "combined_fake_source_uncertainty.png"
                    ),
                )

            # RESPONSE SYSTEMATIC DIAGNOSTICS
            # ----------------------------------------------------------------
            # Slow final-mode diagnostic: use varied signal response objects for
            # TES/efficiency-style systematics where available. This remains
            # separate from the fake-source systematic envelopes above.
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
                skipped_response_systematics = []
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
                        if not all(
                            histogram_has_finite_content(hist)
                            for hist in (reco_up, reco_down, matrix_up, matrix_down)
                        ):
                            raise ValueError(
                                "one or more varied response histograms are empty or non-finite"
                            )
                    except (KeyError, ValueError) as exc:
                        skipped_response_systematics.append(sys_name)
                        plotter.logger.warning(
                            "Skipping response systematic '%s' for %s %s because "
                            "the full up/down response objects are unavailable: %s",
                            sys_name,
                            config.label,
                            var,
                            exc,
                        )
                        continue

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

                if skipped_response_systematics:
                    plotter.logger.warning(
                        "Skipped %d response systematics for %s %s. First skipped: %s",
                        len(skipped_response_systematics),
                        config.label,
                        var,
                        ", ".join(skipped_response_systematics[:5]),
                    )
                if not response_uncertainties:
                    raise RuntimeError(
                        f"DO_FULL_SYSTEMATICS is enabled, but no complete response "
                        f"systematic variations were available for {config.label} {var}."
                    )

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
            # Always save the response matrix for each config. It is the first
            # thing to inspect if closure looks odd.
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
            # Produce nominal unfolded data and same-sample signal-MC closure for
            # every configured RooUnfold iteration. Iteration 0 is bin-by-bin.
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
            # Put all unfolded signal-MC closure iterations on one plot. This is
            # a quick regularisation sanity check.
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
    # The markdown summary is the machine-readable bridge between this runner,
    # the validation scripts, and the thesis update report. Keep the table order
    # matched to the row lists filled above:
    #
    #   closure_rows                -> same-sample unfolding closure
    #   fake_budget_rows            -> pre-unfolding event budget
    #   fake_source_systematic_rows -> FF-stat and MET-window envelopes
    #   fake_width_rows             -> tau-width composition envelope
    summary_lines = [
        "# Variable-specific shadow-bin unfolding closure summary",
        "",
        f"DO_FULL_SYSTEMATICS: `{DO_FULL_SYSTEMATICS}`",
        f"RUN_FAKE_FF_STAT_SYSTEMATIC: `{RUN_FAKE_FF_STAT_SYSTEMATIC}`",
        f"RUN_FAKE_MET_WINDOW_SYSTEMATIC: `{RUN_FAKE_MET_WINDOW_SYSTEMATIC}`",
        f"RUN_FAKE_WIDTH_SYSTEMATIC: `{RUN_FAKE_WIDTH_SYSTEMATIC}`",
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

    # Pre-unfolding budget: this is the most useful table for diagnosing
    # whether the unfolded data/MC normalisation is already determined before
    # RooUnfold ever sees the input.
    summary_lines.extend(
        [
            "",
            "## Pre-unfolding budget",
            "",
            "`data_sig` is the nominal unfolded input before unfolding:",
            (
                "`data - MC-contamination backgrounds - prong-split fake estimate - "
                "nonfiducial signal`."
                if USE_MC_CONTAMINATION_SUBTRACTION
                else (
                    "`data - all MC backgrounds - prong-split fake estimate - nonfiducial signal`."
                )
            ),
            "",
            (
                "`Data sig, all bkg + fakes diagnostic` records the all-MC "
                "background subtraction cross-check for comparison only."
            ),
            "",
            "| Configuration | Variable | Data | All MC bkg | MC-contam bkg | "
            "Jet-fake-like MC bkg | Fakes | Nonfid signal | "
            "Data sig, MC-contam bkg + fakes | Data sig, all bkg + fakes diagnostic | "
            "Data sig, no fakes | Fid reco signal | Fid reco / data sig |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fake_budget_rows:
        (
            config_label,
            var,
            data_integral,
            all_bkg,
            mc_contamination_bkg,
            jet_fake_like_bkg,
            fakes_integral,
            nonfid,
            data_sig,
            data_sig_all_bkg_with_fakes,
            data_sig_no_fake,
            fid_reco,
            fid_over_data_sig,
        ) = row
        summary_lines.append(
            f"| {config_label} | {var} | {data_integral:.3f} | {all_bkg:.3f} | "
            f"{mc_contamination_bkg:.3f} | {jet_fake_like_bkg:.3f} | {fakes_integral:.3f} | "
            f"{nonfid:.3f} | {data_sig:.3f} | {data_sig_all_bkg_with_fakes:.3f} | "
            f"{data_sig_no_fake:.3f} | {fid_reco:.3f} | {fid_over_data_sig:.3f} |"
        )

    # Fake-source systematic table: these are propagated through unfolding and
    # reported as shifted fake yield, shifted background-subtracted input, and
    # shifted unfolded spectrum. The plotted per-bin shapes live under
    # ``plots/<config>/<var>/fake_source_systematics``.
    summary_lines.extend(
        [
            "",
            "## Fake-source systematic envelopes",
            "",
            "`JET_FAKE_FF_STAT` shifts the fake-factor bins by their statistical "
            "uncertainties. `JET_FAKE_MET_WINDOW` envelopes the low-MET fake-factor "
            "regions `[30,100]`, `[50,100]`, `[70,100]`, and `[0,150]` GeV against "
            "the nominal `[0,100]` GeV region.",
            "",
            "| Configuration | Variable | Systematic | Iterations | Nominal fakes | "
            "Fakes up | Fakes down | Nominal data sig | Data sig up | Data sig down | "
            "Nominal unfolded | Shifted-up unfolded | Unfolded uncertainty | "
            "Relative integral shift |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if fake_source_systematic_rows:
        for (
            config_label,
            var,
            systematic_name,
            iter_count,
            nominal_fakes,
            fakes_up,
            fakes_down,
            nominal_data_sig,
            data_sig_up,
            data_sig_down,
            nominal_unfolded,
            shifted_up_unfolded,
            unfolded_uncertainty,
            relative_integral_shift,
        ) in fake_source_systematic_rows:
            summary_lines.append(
                f"| {config_label} | {var} | `{systematic_name}` | {iter_count} | "
                f"{nominal_fakes:.3f} | {fakes_up:.3f} | {fakes_down:.3f} | "
                f"{nominal_data_sig:.3f} | {data_sig_up:.3f} | {data_sig_down:.3f} | "
                f"{nominal_unfolded:.6g} | {shifted_up_unfolded:.6g} | "
                f"{unfolded_uncertainty:.6g} | {relative_integral_shift:.3f} |"
            )
    else:
        summary_lines.append(
            "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )

    # Tau-width is separated from the generic fake-source table because it is
    # explicitly a 1-prong composition envelope, not a change to the central fake
    # estimate. Keeping it separate avoids accidentally interpreting it as a
    # new nominal prescription.
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
            "physically usable after MC-contamination subtraction.",
            "",
            "| Configuration | Variable | Systematic | Width proxy | Iterations | "
            "Nominal fakes | Shifted fakes | Nominal data sig | Shifted data sig | "
            "Fid reco / nominal data sig | Fid reco / shifted data sig | "
            "Nominal unfolded | Shifted unfolded | Unfolded abs shift | "
            "Unfolded rel shift |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if fake_width_rows:
        for (
            config_label,
            var,
            systematic_name,
            width_var,
            iter_count,
            nominal_fakes,
            shifted_fakes,
            nominal_data_sig,
            shifted_data_sig,
            nominal_fid_over_data,
            shifted_fid_over_data,
            nominal_unfolded,
            shifted_unfolded,
            unfolded_abs_shift,
            unfolded_rel_shift,
        ) in fake_width_rows:
            summary_lines.append(
                f"| {config_label} | {var} | `{systematic_name}` | `{width_var}` | "
                f"{iter_count} | "
                f"{nominal_fakes:.3f} | {shifted_fakes:.3f} | {nominal_data_sig:.3f} | "
                f"{shifted_data_sig:.3f} | {nominal_fid_over_data:.3f} | "
                f"{shifted_fid_over_data:.3f} | {nominal_unfolded:.6g} | "
                f"{shifted_unfolded:.6g} | {unfolded_abs_shift:.6g} | "
                f"{unfolded_rel_shift:.3f} |"
            )
    else:
        summary_lines.append(
            "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
            "n/a | n/a | n/a | n/a | n/a |"
        )
    summary_path = output_root / "closure_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    plotter.logger.info("Saved closure summary to %s", summary_path)

    if (not load_measured_analysis_hists) or generated_missing_fake_hists:
        measured_analysis.save_hists()

    plotter.logger.info("DONE.")
