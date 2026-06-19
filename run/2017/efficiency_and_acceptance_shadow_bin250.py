from pathlib import Path

import numpy as np
from binnings import BINNINGS
from samples import DSID_METADATA_CACHE, signal_sample

from src.analysis import Analysis
from src.cutting import Cut
from utils.helper_functions import smart_join
from utils.plotting_tools import Hist2dOpts, PlotKwargs
from utils.ROOT_utils import bayes_divide, normalise_migration_hist
from utils.variable_names import variable_data

datasets = {"wtaunu_had": signal_sample()}

# CUTS & SELECTIONS
# ========================================================================
pass_reco = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
pass_truth = Cut(
    r"Pass Truth",
    r"(passTruth == 1)",
)
truth_tau = Cut(
    r"Truth Hadronic Tau",
    r"TruthTau_isHadronic && ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))",
)
truth_tau_1prong = Cut(
    r"1-prong truth",
    r"TruthTau_nChargedTracks == 1",
)
truth_tau_3prong = Cut(
    r"3-prong truth",
    r"TruthTau_nChargedTracks == 3",
)
truth_tau_plus = Cut(
    r"1-prong truth",
    r"TruthTauCharge == 1",
)
truth_tau_minus = Cut(
    r"3-prong truth",
    r"TruthTauCharge == -1",
)
reco_tau_1prong = Cut(
    r"1-prong Reconstructed Hadronic Tau",
    "(TauNCoreTracks == 1) && (TruthTau_nChargedTracks == 1)",
)
reco_tau_3prong = Cut(
    r"3-prong Reconstructed Hadronic Tau",
    "(TauNCoreTracks == 3) && (TruthTau_nChargedTracks == 3)",
)
reco_tau_plus = Cut(
    r"Positive Tau",
    "(TauCharge == 1) && (TruthTauCharge == 1)",
)
reco_tau_minus = Cut(
    r"Negative Tau",
    "(TauCharge == -1) && (TruthTauCharge == -1)",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_SR_reco = Cut(
    r"Pass SR Reco",
    r"(TauPt > 125) && (MET_met > 125) && (MTW > 250)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))"
    r"&& (((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
pass_SR_truth = Cut(
    r"Pass SR Truth",
    r"(VisTruthTauPt > 125) && (TruthMTW > 250) && (TruthNeutrinoPt > 125)"
    r"&& ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))"
    r"&& (((abs(VisTruthTauEta) < 1.37) || (1.52 < abs(VisTruthTauEta))) && (abs(VisTruthTauEta) < 2.47))",
)
pass_SR = pass_SR_truth
truth_cuts = [
    pass_truth,
    pass_SR,
    truth_tau,
]
reco_cuts = [
    pass_reco,
    pass_SR_reco,
]
selections: dict[str, list[Cut]] = {
    # PASS TRUTH
    # ================================================
    "truth_tau": truth_cuts,
    "1prong_truth_tau": truth_cuts + [truth_tau_1prong],
    "3prong_truth_tau": truth_cuts + [truth_tau_3prong],
    "tauplus_truth_tau": truth_cuts + [truth_tau_plus],
    "tauminus_truth_tau": truth_cuts + [truth_tau_minus],
    # PASS RECO
    # ================================================
    "loose_reco_tau": reco_cuts + [pass_loose],
    "loose_1prong_reco_tau": reco_cuts + [pass_loose, reco_tau_1prong],
    "loose_3prong_reco_tau": reco_cuts + [pass_loose, reco_tau_3prong],
    "loose_tauplus_reco_tau": reco_cuts + [pass_loose, reco_tau_plus],
    "loose_tauminus_reco_tau": reco_cuts + [pass_loose, reco_tau_minus],
    "medium_reco_tau": reco_cuts + [pass_medium],
    "medium_1prong_reco_tau": reco_cuts + [pass_medium, reco_tau_1prong],
    "medium_3prong_reco_tau": reco_cuts + [pass_medium, reco_tau_3prong],
    "medium_tauplus_reco_tau": reco_cuts + [pass_medium, reco_tau_plus],
    "medium_tauminus_reco_tau": reco_cuts + [pass_medium, reco_tau_minus],
    "tight_reco_tau": reco_cuts + [pass_tight],
    "tight_1prong_reco_tau": reco_cuts + [pass_tight, reco_tau_1prong],
    "tight_3prong_reco_tau": reco_cuts + [pass_tight, reco_tau_3prong],
    "tight_tauplus_reco_tau": reco_cuts + [pass_tight, reco_tau_plus],
    "tight_tauminus_reco_tau": reco_cuts + [pass_tight, reco_tau_minus],
    # PASS TRUTH AND RECO
    # ================================================
    # fmt: off
    "loose_truth_reco_tau": truth_cuts + reco_cuts + [pass_loose],
    "loose_1prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_loose, truth_tau_1prong,
                                                             reco_tau_1prong],
    "loose_3prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_loose, truth_tau_3prong,
                                                             reco_tau_3prong],
    "loose_tauplus_truth_reco_tau": truth_cuts + reco_cuts + [pass_loose, truth_tau_plus,
                                                              reco_tau_plus],
    "loose_tauminus_truth_reco_tau": truth_cuts + reco_cuts + [pass_loose, truth_tau_minus,
                                                               reco_tau_minus],
    "medium_truth_reco_tau": truth_cuts + reco_cuts + [pass_medium],
    "medium_1prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_medium, truth_tau_1prong,
                                                              reco_tau_1prong],
    "medium_3prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_medium, truth_tau_3prong,
                                                              reco_tau_3prong],
    "medium_tauplus_truth_reco_tau": truth_cuts + reco_cuts + [pass_medium, truth_tau_plus,
                                                               reco_tau_plus],
    "medium_tauminus_truth_reco_tau": truth_cuts + reco_cuts + [pass_medium, truth_tau_minus,
                                                                reco_tau_minus],
    "tight_truth_reco_tau": truth_cuts + reco_cuts + [pass_tight],
    "tight_1prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_tight, truth_tau_1prong,
                                                             reco_tau_1prong],
    "tight_3prong_truth_reco_tau": truth_cuts + reco_cuts + [pass_tight, truth_tau_3prong,
                                                             reco_tau_3prong],
    "tight_tauplus_truth_reco_tau": truth_cuts + reco_cuts + [pass_tight, truth_tau_plus,
                                                              reco_tau_plus],
    "tight_tauminus_truth_reco_tau": truth_cuts + reco_cuts + [pass_tight, truth_tau_minus,
                                                               reco_tau_minus],
    # fmt: on
}

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "TauPt",
    "VisTruthTauPt",
    "MTW",
    "TruthMTW",
    "MET_met",
    "TruthNeutrinoPt",
]
measurement_vars_unitless = [
    "TauEta",
    "VisTruthTauEta",
    "TauPhi",
    "VisTruthTauPhi",
    "TruthNeutrinoPhi",
    "MET_phi",
    "AbsDeltaPhi_tau_met",
    "TruthAbsDeltaPhi_tau_met",
    "TauPt_div_MET",
    "TruthTauPt_div_MET",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
reco_measurement_vars = [
    v
    for v in measurement_vars
    if (variable_data[v]["tag"] == "reco") and (v not in ("TauPt_res_frac", "TauPt_res"))
]
truths = {
    "MTW": "TruthMTW",
    "TauPt": "VisTruthTauPt",
    "TauEta": "VisTruthTauEta",
    "TauPhi": "VisTruthTauPhi",
    "MET_met": "TruthNeutrinoPt",
    "MET_phi": "TruthNeutrinoPhi",
    "AbsDeltaPhi_tau_met": "TruthAbsDeltaPhi_tau_met",
    "TauPt_div_MET": "TruthTauPt_div_MET",
}
# Keep all variables used by efficiency/acceptance plots and migration/response matrices.
histogram_vars = set(reco_measurement_vars) | set(truths.values())

# define 2d histograms
hists_2d = {
    "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt", "reco_weight"),
    "TauEta_VisTruthTauEta": Hist2dOpts("TauEta", "VisTruthTauEta", "reco_weight"),
    "TauPhi_VisTruthTauPhi": Hist2dOpts("TauPhi", "VisTruthTauPhi", "reco_weight"),
    "MTW_TruthMTW": Hist2dOpts("MTW", "TruthMTW", "reco_weight"),
    "MET_met_TruthNeutrinoPt": Hist2dOpts("MET_met", "TruthNeutrinoPt", "reco_weight"),
    "MET_phi_TruthNeutrinoPhi": Hist2dOpts("MET_phi", "TruthNeutrinoPhi", "reco_weight"),
    "AbsDeltaPhi_tau_met_TruthAbsDeltaPhi_tau_met": Hist2dOpts(
        "AbsDeltaPhi_tau_met", "TruthAbsDeltaPhi_tau_met", "reco_weight"
    ),
    "TauPt_div_MET_TruthTauPt_div_MET": Hist2dOpts(
        "TauPt_div_MET", "TruthTauPt_div_MET", "reco_weight"
    ),
}
NOMINAL_NAME = "T_s1thv_NOMINAL"
LOAD_SAVED_HISTS = False
mtw_bins = np.array(
    [250, 350, 375, 400, 430, 465, 500, 550, 600, 700, 850, 1000, 2000], dtype="double"
)
taupt_bins = np.array([125, 170, 200, 250, 300, 350, 425, 500, 600, 1000], dtype="double")


def run_analysis() -> Analysis:
    """Run analysis"""

    return Analysis(
        datasets,
        year=2017,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=False,
        # regen_metadata=True,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        selections=selections,
        analysis_label=Path(__file__).stem,
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars=histogram_vars,
        hists_2d=hists_2d,
        do_unweighted=True,
        binnings={
            "": BINNINGS
                | {
                    "MTW": mtw_bins,
                    "TruthMTW": mtw_bins,
                    "TauPt": taupt_bins,
                    "MET_met": taupt_bins,
                    "TruthNeutrinoPt": taupt_bins,
                    "TruthTauPt": taupt_bins,
                }
        },
    )


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()
    load_analysis_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available()

    analysis.full_cutflow_printout(datasets=["wtaunu_had"])
    base_plotting_dir = analysis.paths.plot_dir

    # print histograms
    if not LOAD_SAVED_HISTS:
        for dataset in analysis:
            dataset.histogram_printout(to_file="txt", to_dir=analysis.paths.latex_dir)

    working_points = ("loose", "medium", "tight")
    tau_sections = ("", "1prong_", "3prong_", "tauplus_", "tauminus_")

    # CALCULATE EFFICIENCY AND ACCEPTANCE
    # ========================================================================
    for wp in working_points:
        for sec in tau_sections:
            for var in reco_measurement_vars:
                if not load_analysis_hists:
                    analysis.histograms[f"{wp}_{sec}{var}_efficiency"] = bayes_divide(
                        analysis.get_hist(
                            var + "_unweighted",
                            "wtaunu_had",
                            NOMINAL_NAME,
                            f"{wp}_{sec}truth_reco_tau",
                        ),
                        analysis.get_hist(
                            var + "_unweighted",
                            "wtaunu_had",
                            NOMINAL_NAME,
                            f"{sec}truth_tau",
                        ),
                    )
                    analysis.histograms[f"{wp}_{sec}{var}_acceptance"] = bayes_divide(
                        analysis.get_hist(
                            var + "_unweighted",
                            "wtaunu_had",
                            NOMINAL_NAME,
                            f"{wp}_{sec}truth_reco_tau",
                        ),
                        analysis.get_hist(
                            var + "_unweighted",
                            "wtaunu_had",
                            NOMINAL_NAME,
                            f"{wp}_{sec}reco_tau",
                        ),
                    )

                # Plots alone
                # =======================================================================
                default_args: PlotKwargs = {
                    "dataset": "wtaunu_had",
                    "systematic": NOMINAL_NAME,
                    "selection": f"{wp}_{sec}reco_tau",
                    "do_stat": True,
                    "do_syst": False,
                    "label": None,
                    "label_params": {"llabel": "Simulation", "loc": 1},
                    "title": smart_join(
                        "2017",
                        "1-prong Taus"
                        if (sec == "1prong_")
                        else ("3-prong Taus" if (sec == "3prong_") else ""),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                }

                analysis.paths.plot_dir = base_plotting_dir / wp / sec
                if var in measurement_vars_mass:
                    default_args.update(
                        {"logx": True, "xlabel": variable_data[var]["name"] + " [GeV]"}
                    )
                elif var in measurement_vars_unitless:
                    default_args.update({"logx": False, "xlabel": variable_data[var]["name"]})

                # mental health
                analysis.plot(var, **default_args, filename=f"{wp}_{sec}{var}.png")

                default_args.update({"y_axlim": (0, 1.3), "hline_at": 1})

                analysis.plot(
                    val=analysis.histograms[f"{wp}_{sec}{var}_efficiency"],
                    ylabel=r"$\epsilon_\mathrm{selection}$",
                    colour="r",
                    **default_args,
                    filename=f"{wp}_{sec}{var}_efficiency.png",
                )
                analysis.plot(
                    val=analysis.histograms[f"{wp}_{sec}{var}_acceptance"],
                    ylabel=r"$f_\mathrm{in}$",
                    colour="r",
                    **default_args,
                    filename=f"{wp}_{sec}{var}_acceptance.png",
                )
                truth_label = variable_data[truths[var]]["name"] + (
                    " [GeV]" if var in measurement_vars_mass else ""
                )
                reco_label = variable_data[var]["name"] + (
                    " [GeV]" if var in measurement_vars_mass else ""
                )
                migration_hist = normalise_migration_hist(
                    analysis.get_hist(
                        f"{var}_{truths[var]}",
                        dataset="wtaunu_had",
                        systematic=NOMINAL_NAME,
                        selection=f"{wp}_{sec}truth_reco_tau",
                    )
                )
                analysis.plot_2d(
                    migration_hist,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=f"{wp}_{sec}truth_reco_tau",
                    ylabel=truth_label,
                    xlabel=reco_label,
                    title=f"{reco_label.removesuffix(' [GeV]')} Migration Matrix [%] | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    labels=True,
                    logx=True if var in measurement_vars_mass else False,
                    logy=True if var in measurement_vars_mass else False,
                    label_params={"llabel": "Simulation"},
                    filename=f"{wp}_{sec}{var}_migration.png",
                )

                # resonance
                response = analysis.get_hist(
                    f"{var}_{truths[var]}",
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=f"{wp}_{sec}truth_reco_tau",
                ).Clone()
                response.Scale(1 / response.GetEffectiveEntries())
                analysis.plot_2d(
                    response,
                    dataset="wtaunu_had",
                    systematic=NOMINAL_NAME,
                    selection=f"{wp}_{sec}truth_reco_tau",
                    ylabel=truth_label,
                    xlabel=reco_label,
                    title=f"{reco_label.removesuffix(' [GeV]')} Resonance | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    labels=True,
                    logx=True if var in measurement_vars_mass else False,
                    logy=True if var in measurement_vars_mass else False,
                    label_params={"llabel": "Simulation"},
                    filename=f"{wp}_{sec}{var}_response.png",
                )

    # START OF PRONG LOOP
    # ========================================================================
    for sec in tau_sections:
        for var in reco_measurement_vars:
            default_args.update(
                {
                    "label": ["loose", "medium", "tight"],
                    "title": smart_join(
                        "1-prong Taus"
                        if (sec == "1prong_")
                        else ("3-prong Taus" if (sec == "3prong_") else ""),
                        "2017",
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                }
            )

            analysis.paths.plot_dir = base_plotting_dir / sec
            if var in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[var]["name"] + " [GeV]"})
            elif var in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[var]["name"]})

            analysis.plot(
                val=[
                    analysis.histograms[f"loose_{sec}{var}_efficiency"],
                    analysis.histograms[f"medium_{sec}{var}_efficiency"],
                    analysis.histograms[f"tight_{sec}{var}_efficiency"],
                ],
                ylabel=r"$\epsilon_\mathrm{selection}$",
                **default_args,
                filename=f"{sec}wp_compare_{var}_efficiency.png",
            )
            analysis.plot(
                val=[
                    analysis.histograms[f"loose_{sec}{var}_acceptance"],
                    analysis.histograms[f"medium_{sec}{var}_acceptance"],
                    analysis.histograms[f"tight_{sec}{var}_acceptance"],
                ],
                ylabel=r"$f_\mathrm{in}$",
                **default_args,
                filename=f"{sec}wp_compare_{var}_acceptance.png",
            )

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in working_points:
        for var in reco_measurement_vars:
            default_args.update(
                {
                    "label": ["1-prong", "3-prong"],
                    "title": smart_join(
                        f"{wp} WP",
                        "2017",
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                }
            )

            analysis.paths.plot_dir = base_plotting_dir / wp
            if var in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[var]["name"] + " [GeV]"})
            elif var in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[var]["name"]})

            analysis.plot(
                val=[
                    analysis.histograms[f"{wp}_1prong_{var}_efficiency"],
                    analysis.histograms[f"{wp}_3prong_{var}_efficiency"],
                ],
                ylabel=r"$\epsilon_\mathrm{selection}$",
                **default_args,
                filename=f"{wp}_prong_compare_{var}_efficiency.png",
            )
            analysis.plot(
                val=[
                    analysis.histograms[f"{wp}_1prong_{var}_acceptance"],
                    analysis.histograms[f"{wp}_3prong_{var}_acceptance"],
                ],
                ylabel=r"$f_\mathrm{in}$",
                **default_args,
                filename=f"{wp}_prong_compare_{var}_acceptance.png",
            )

    if not load_analysis_hists:
        analysis.save_hists()
    analysis.logger.info("DONE.")
