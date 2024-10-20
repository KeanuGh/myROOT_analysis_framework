from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from src.dataset import ProfileOpts
from src.histogram import Histogram1D
from utils.plotting_tools import get_axis_labels
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-08-28/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
DO_SYS = False

datasets: Dict[str, Dict] = {
    # SIGNAL
    # ====================================================================
    "wtaunu": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$W\rightarrow\tau\nu$",
        "is_signal": True,
    },
    # BACKGROUNDS
    # ====================================================================
    # W -> light lepton
    "wlnu": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Wmunu_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Wmunu_mW_120*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_mW_120*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$W\rightarrow (e/\mu)\nu$",
    },
    # Z -> ll
    "zll": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zee_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Ztautau_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zee_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Znunu_pTV2*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
    },
    "top": {
        "data_path": [
            DTA_PATH / "*PP8_singletop*/*.root",
            DTA_PATH / "*PP8_tchan*/*.root",
            DTA_PATH / "*PP8_Wt_DR_dilepton*/*.root",
            DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
        ],
        "label": "Top",
    },
    # DIBOSON
    "diboson": {
        "data_path": [
            DTA_PATH / "*Sh_2212_llll*/*.root",
            DTA_PATH / "*Sh_2212_lllv*/*.root",
            DTA_PATH / "*Sh_2212_llvv*/*.root",
            DTA_PATH / "*Sh_2212_lvvv*/*.root",
            DTA_PATH / "*Sh_2212_vvvv*/*.root",
            DTA_PATH / "*Sh_2211_ZqqZll*/*.root",
            DTA_PATH / "*Sh_2211_ZbbZll*/*.root",
            DTA_PATH / "*Sh_2211_WqqZll*/*.root",
            DTA_PATH / "*Sh_2211_WlvWqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZbb*/*.root",
        ],
        "label": "Diboson",
    },
    # DATA
    # ====================================================================
    "data": {
        # "data_path": DTA_PATH / "*data17*/*.root",
        "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
        "label": "data",
        "is_data": True,
    },
}

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)",
)
pass_taupt170 = Cut(
    r"$p_T^\tau > 170$",
    r"TauPt > 170",
)
pass_mtw150 = Cut(
    r"$m_T^W > 150$",
    r"MTW > 150",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
fail_loose = Cut(
    r"\mathrm{Fail Loose ID}",
    "(TauBDTEleScore > 0.05) && "
    "!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium = Cut(
    r"\mathrm{Fail Medium ID}",
    "(TauBDTEleScore > 0.1) && "
    "!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight = Cut(
    r"\mathrm{Fail Tight ID}",
    "(TauBDTEleScore > 0.15) && "
    "!((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_met150 = Cut(
    r"$E_T^{\mathrm{miss}} > 150$",
    r"MET_met > 150",
)
pass_100met = Cut(
    r"$E_T^{\mathrm{miss}} < 100$",
    r"MET_met < 100",
)
pass_1prong = Cut(
    "1-prong",
    "TauNCoreTracks == 1",
)
pass_3prong = Cut(
    "3-prong",
    "TauNCoreTracks == 3",
)
pass_truetau = Cut(
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
)

# selections
selections_loose: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_loose,
        pass_met150,
    ],
    "loose_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_loose,
        pass_met150,
    ],
    "loose_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_loose,
        pass_100met,
    ],
    "loose_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_loose,
        pass_100met,
    ],
}
selections_medium: dict[str, list[Cut]] = {
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_medium,
        pass_met150,
    ],
    "medium_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_medium,
        pass_met150,
    ],
    "medium_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_medium,
        pass_100met,
    ],
    "medium_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_medium,
        pass_100met,
    ],
}
selections_tight: dict[str, list[Cut]] = {
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_tight,
        pass_met150,
    ],
    "tight_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_tight,
        pass_met150,
    ],
    "tight_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_tight,
        pass_100met,
    ],
    "tight_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        fail_tight,
        pass_100met,
    ],
}
selections = selections_loose | selections_medium | selections_tight

# define selection for MC samples
selections_list = list(selections.keys())
selections_cuts = list(selections.values())
for selection, cut_list in zip(selections_list, selections_cuts):
    selections[f"trueTau_{selection}"] = cut_list + [pass_truetau]
    # define selections for 1- or 3- tau prongs
    for cutstr, cut_name in [
        ("TauNCoreTracks == 1", "1prong"),
        ("TauNCoreTracks == 3", "3prong"),
    ]:
        selections[f"{cut_name}_{selection}"] = cut_list + [Cut(cut_name, cutstr)]
        selections[f"trueTau_{cut_name}_{selection}"] = cut_list + [
            pass_truetau,
            Cut(cut_name, cutstr),
        ]

# VARIABLES
# ========================================================================
wanted_variables = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "DeltaPhi_tau_met",
    "TauPt_div_MET",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TruthTauPt",
    "TruthTauEta",
    "TruthTauPhi",
    "TauNCoreTracks",
    # "TruthTau_nChargedTracks",
    # "TruthTau_nNeutralTracks",
    "TauPt_res",
    "TauPt_diff",
    "MatchedTruthParticlePt",
    "MatchedTruthParticle_isTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isPhoton",
    "MatchedTruthParticle_isJet",
    "nJets",
}
measurement_vars_mass = [
    "TauPt",
    "MTW",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "nJets",
    "TauNCoreTracks",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "DeltaPhi_tau_met",
    "TauPt_div_MET",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
profile_vars = [
    "TauPt_res",
    "TauPt_diff",
    "MatchedTruthParticlePt",
    "MatchedTruthParticle_isTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isPhoton",
    "MatchedTruthParticle_isJet",
]
# define which profiles to calculate
profiles: dict[str, ProfileOpts] = dict()
for measurement_var in measurement_vars:
    for prof_var in profile_vars:
        profiles[f"{measurement_var}_{prof_var}"] = ProfileOpts(
            x=measurement_var,
            y=prof_var,
            weight="" if "MatchedTruthParticle" in prof_var else "reco_weight",
        )
NOMINAL_NAME = "T_s1thv_NOMINAL"


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=2017,
        rerun=False,
        # regen_histograms=True,
        do_systematics=DO_SYS,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        cuts=selections,
        analysis_label="analysis_full",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        profiles=profiles,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 21),
                "TauPt": np.geomspace(170, 1000, 21),
                "TauEta": np.linspace(-2.5, 2.5, 21),
                "EleEta": np.linspace(-2.5, 2.5, 21),
                "MuonEta": np.linspace(-2.5, 2.5, 21),
                "MET_met": np.geomspace(150, 1000, 21),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 21),
                "TauPt_div_MET": np.linspace(0, 3, 61),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TruthTauPt": np.geomspace(1, 1000, 21),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TauPt_res": np.linspace(-1, 1, 51),
                "TauPt_diff": np.linspace(-300, 300, 51),
                "badJet": (2, 0, 2),
            },
            ".*_CR_.*ID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
        },
    )


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()
    base_plotting_dir = analysis.paths.plot_dir
    all_samples = [analysis.data_sample] + analysis.mc_samples
    mc_samples = analysis.mc_samples
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    # for dataset in analysis:
    #     ROOT.RDF.SaveGraph(
    #         dataset.filters["T_s1thv_NOMINAL"]["loose_SR_passID"].df,
    #         f"{analysis.paths.output_dir}/{dataset.name}_SR_graph.dot",
    #     )

    # print histograms
    for dataset in analysis:
        dataset.histogram_printout(to_file="txt", to_dir=analysis.paths.latex_dir)

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in ("loose", "medium", "tight"):
        wp_dir = base_plotting_dir / wp

        # PLOT TRUTH (for mental health) (no selection)
        # ========================================================================
        analysis.paths.plot_dir = wp_dir / "truth"
        default_args = {
            "dataset": mc_samples,
            "title": f"Truth Taus | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
            "do_stat": True,
            "do_sys": False,
            "selection": "",
            "ratio_plot": False,
            "stats_box": False,
        }
        analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
        analysis.plot(val="TruthTauPt", **default_args, logx=True)
        analysis.plot(val="TruthTauEta", **default_args)
        analysis.plot(val="TruthTauPhi", **default_args)

        default_args["selection"] = f"trueTau_{wp}_SR_passID"
        analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
        analysis.plot(val="TruthTauPt", **default_args, logx=True)
        analysis.plot(val="TruthTauEta", **default_args)
        analysis.plot(val="TruthTauPhi", **default_args)

        # tau pt resolution
        analysis.plot(
            val="TauPt_res",
            dataset="wtaunu",
            xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            filename="wtaunu_taupt_resolution.png",
        )
        analysis.plot(
            val="TauPt_diff",
            dataset="wtaunu",
            xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            filename="wtaunu_taupt_truthrecodiff.png",
        )
        analysis.plot(
            val="MTW_TauPt_res",
            dataset="wtaunu",
            selection=f"{wp}_SR_passID",
            ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
            xlabel=r"$m_W^T$ [GeV]",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            y_axlim=(-10, 10),
            filename="wtaunu_mtw_taupt_profile.png",
        )
        analysis.plot(
            val="TauPt_res",
            selection=[
                f"trueTau_{wp}_SR_passID",
                f"trueTau_{wp}_SR_failID",
                f"trueTau_{wp}_CR_passID",
                f"trueTau_{wp}_CR_failID",
            ],
            dataset="wtaunu",
            label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
            xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            ratio_plot=False,
            filename="wtaunu_taupt_resolution_selections.png",
        )
        analysis.plot(
            val="TauPt_diff",
            selection=[
                f"trueTau_{wp}_SR_passID",
                f"trueTau_{wp}_SR_failID",
                f"trueTau_{wp}_CR_passID",
                f"trueTau_{wp}_CR_failID",
            ],
            dataset="wtaunu",
            label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
            xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            ratio_plot=False,
            filename="wtaunu_taupt_truthrecodiff_selections.png",
        )
        analysis.plot(
            val="MTW_TauPt_res",
            selection=[
                f"trueTau_{wp}_SR_passID",
                f"trueTau_{wp}_SR_failID",
                f"trueTau_{wp}_CR_passID",
                f"trueTau_{wp}_CR_failID",
            ],
            dataset="wtaunu",
            label=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
            ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
            xlabel=r"$m_W^T$ [GeV]",
            title=(
                r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
            ),
            y_axlim=(-10, 10),
            ratio_plot=False,
            filename="wtaunu_mtw_taupt_profile_selections.png",
        )

        # FAKES ESTIMATE
        # ========================================================================
        for prong_str in ["1prong", "3prong", ""]:
            nprong = prong_str + "_" if prong_str else ""

            for fakes_source in ["TauPt", "MTW"]:
                analysis.do_fakes_estimate(
                    fakes_source,
                    measurement_vars,
                    f"{nprong}{wp}_CR_passID",
                    f"{nprong}{wp}_CR_failID",
                    f"{nprong}{wp}_SR_passID",
                    f"{nprong}{wp}_SR_failID",
                    f"trueTau_{nprong}{wp}_CR_passID",
                    f"trueTau_{nprong}{wp}_CR_failID",
                    f"trueTau_{nprong}{wp}_SR_passID",
                    f"trueTau_{nprong}{wp}_SR_failID",
                    name=f"{nprong}{wp}",
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )

                # Intermediates
                # ----------------------------------------------------------------------------
                CR_passID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{nprong}{wp}_CR_passID",
                )
                CR_failID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{nprong}{wp}_CR_failID",
                )
                SR_failID_data = analysis.get_hist(
                    fakes_source,
                    "data",
                    systematic=NOMINAL_NAME,
                    selection=f"{nprong}{wp}_SR_failID",
                )
                CR_passID_mc = analysis.get_hist(
                    f"{nprong}{wp}_all_mc_{fakes_source}_trueTau_{nprong}{wp}_CR_passID"
                )
                CR_failID_mc = analysis.get_hist(
                    f"{nprong}{wp}_all_mc_{fakes_source}_trueTau_{nprong}{wp}_CR_failID"
                )
                SR_failID_mc = analysis.get_hist(
                    f"{nprong}{wp}_all_mc_{fakes_source}_trueTau_{nprong}{wp}_SR_failID"
                )

                analysis.paths.plot_dir = wp_dir / "fakes_intermediates"
                analysis.plot(
                    [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
                    label=["CR_passID_data", "CR_failID_data", "CR_passID_mc", "CR_failID_mc"],
                    do_stat=False,
                    logy=True,
                    xlabel=fakes_source,
                    ratio_plot=False,
                    filename=f"{nprong}{wp}_FF_histograms_{fakes_source}.png",
                )
                analysis.plot(
                    [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
                    label=["CR_failID_data - CR_failID_mc", "CR_passID_data - CR_passID_mc"],
                    do_stat=False,
                    logy=True,
                    xlabel=fakes_source,
                    ratio_plot=True,
                    filename=f"{nprong}{wp}_FF_histograms_diff_{fakes_source}.png",
                    ratio_label="Fake Factor",
                )
                analysis.plot(
                    [SR_failID_data, SR_failID_mc],
                    label=["SR_failID_data", "SR_failID_mc"],
                    do_stat=False,
                    logy=True,
                    xlabel=fakes_source,
                    ratio_plot=False,
                    filename=f"{nprong}{wp}_FF_calculation_{fakes_source}.png",
                )
                analysis.plot(
                    SR_failID_data - SR_failID_mc,
                    label=["SR_failID_data - SR_failID_mc"],
                    do_stat=False,
                    logy=True,
                    xlabel=fakes_source,
                    ratio_plot=False,
                    filename=f"{nprong}{wp}_FF_calculation_delta_SR_fail_{fakes_source}.png",
                )

                # Fake factors
                # ----------------------------------------------------------------------------
                analysis.paths.plot_dir = wp_dir / "fake_factors"
                analysis.plot(
                    val=f"{nprong}{wp}_{fakes_source}_FF",
                    xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                    do_stat=False,
                    logx=False,
                    logy=False,
                    ylabel="Fake factor",
                    filename=f"{nprong}{wp}_{fakes_source}_FF.png",
                )

                # Stacks with Fakes background
                # ----------------------------------------------------------------------------
                analysis.paths.plot_dir = wp_dir / "fakes_stacks"
                # log axes
                default_args = {
                    "dataset": all_samples + [None],
                    "systematic": NOMINAL_NAME,
                    "selection": (
                        [f"{nprong}{wp}_SR_passID"]
                        + [f"trueTau_{nprong}{wp}_SR_passID"] * len(mc_samples)
                        + [None]
                    ),
                    "label": [analysis[ds].label for ds in all_samples] + ["Multijet"],
                    "colour": [analysis[ds].colour for ds in all_samples] + [analysis.fakes_colour],
                    "title": f"{fakes_source} fakes binning | data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    "do_stat": False,
                    "do_sys": DO_SYS,
                    "suffix": "fake_scaled_log",
                    "ratio_plot": True,
                    "ratio_axlim": (0.8, 1.2),
                    "kind": "stack",
                }

                def FF_vars(s: str) -> list[str]:
                    """List of variable names for each sample"""
                    return [s] * (len(all_samples)) + [
                        f"{nprong}{wp}_{s}_fakes_bkg_{fakes_source}_src"
                    ]

                # mass variables
                for v in measurement_vars_mass:
                    analysis.plot(
                        val=FF_vars(v),
                        logx=True,
                        logy=True,
                        **default_args,
                        xlabel=variable_data[v]["name"] + " [GeV]",
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_log.png",
                    )
                    analysis.plot(
                        val=FF_vars(v),
                        logx=False,
                        logy=False,
                        **default_args,
                        xlabel=variable_data[v]["name"] + " [GeV]",
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_liny.png",
                    )
                # massless
                for v in measurement_vars_unitless:
                    analysis.plot(
                        val=FF_vars(v),
                        **default_args,
                        logy=True,
                        logx=False,
                        xlabel=variable_data[v]["name"],
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_log.png",
                    )
                    analysis.plot(
                        val=FF_vars(v),
                        **default_args,
                        logy=False,
                        logx=False,
                        xlabel=variable_data[v]["name"],
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_liny.png",
                    )

        # compare fake factors
        analysis.paths.plot_dir = wp_dir / "fakes_comparisons"
        for fakes_source in ["MTW", "TauPt"]:
            analysis.plot(
                val=[f"1prong_{wp}_{fakes_source}_FF", f"3prong_{wp}_{fakes_source}_FF"],
                label=["1-prong", "3-prong"],
                xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                do_stat=False,
                selection="",
                logx=False,
                logy=False,
                ylabel="Fake factor",
                filename=f"{fakes_source}_{wp}_FF_compare.png",
            )

        # Direct data scaling comparison
        # ----------------------------------------------------------------------------
        # log axes
        default_args = {
            "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
            "label": ["Data SR", "MC + TauPt Fakes", "MC + MTW Fakes"],
            "colour": ["k", "b", "r"],
            "do_stat": True,
            "do_sys": DO_SYS,
            "suffix": "fake_scaled_log",
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
        }

        def FF_full_bkg(variable: str, t: str) -> Histogram1D:
            """Sum of all backgrounds + signal + FF"""
            return Histogram1D(
                th1=analysis.sum_hists(
                    [
                        analysis.get_hist(
                            variable=variable,
                            dataset=ds_,
                            systematic=NOMINAL_NAME,
                            selection=f"trueTau_{wp}_SR_passID",
                        )
                        for ds_ in mc_samples
                    ]
                )
                + analysis.get_hist(f"{wp}_{variable}_fakes_bkg_{t}_src")
            )

        # mass variables
        for v in measurement_vars_mass:
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    FF_full_bkg(v, "TauPt"),
                    FF_full_bkg(v, "MTW"),
                ],
                logy=True,
                logx=True,
                **default_args,
                xlabel=variable_data[v]["name"] + " [GeV]",
                filename=f"{wp}_FF_compare_{v}_log.png",
            )
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    FF_full_bkg(v, "TauPt"),
                    FF_full_bkg(v, "MTW"),
                ],
                logy=False,
                logx=False,
                **default_args,
                xlabel=variable_data[v]["name"] + " [GeV]",
                filename=f"{wp}_FF_compare_{v}_liny.png",
            )
        # massless
        for v in measurement_vars_unitless:
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    FF_full_bkg(v, "TauPt"),
                    FF_full_bkg(v, "MTW"),
                ],
                logy=True,
                logx=False,
                **default_args,
                xlabel=variable_data[v]["name"],
                filename=f"{wp}_FF_compare_{v}_log.png",
            )
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    FF_full_bkg(v, "TauPt"),
                    FF_full_bkg(v, "MTW"),
                ],
                logx=False,
                logy=False,
                **default_args,
                xlabel=variable_data[v]["name"],
                filename=f"{wp}_FF_compare_{v}_liny.png",
            )

        if DO_SYS:
            # SYSTEMATIC UNCERTAINTIES
            # ===========================================================================
            sys_list = analysis["wtaunu"].sys_list  # list of systematic variations
            default_args = {
                "label": sys_list,
                "do_stat": False,
                "do_sys": False,
                "ratio_plot": False,
                "legend_params": {"ncols": 1, "fontsize": 8},
            }

            # for each sample
            for selection in selections:
                for mc_sample in mc_samples:
                    analysis.paths.plot_dir = wp_dir / "systematics" / mc_sample / selection
                    default_args["title"] = (
                        f"{mc_sample} | "
                        f"{selection} | "
                        f"mc16d | "
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
                    )

                    # mass variables
                    for s in ("pct", "tot"):
                        if s == "pct":
                            ylabel = "Percentage uncertainty / %"
                        else:
                            ylabel = "Absolute uncertainty"

                        for v in measurement_vars_mass:
                            analysis.plot(
                                val=[
                                    analysis[mc_sample].get_hist(
                                        variable=f"{v}_{sys_name}_{s}_uncert",
                                        systematic=NOMINAL_NAME,
                                        selection=selection,
                                    )
                                    for sys_name in analysis["wtaunu"].sys_list
                                ],
                                logy=False,
                                logx=False,
                                ylabel=ylabel,
                                **default_args,
                                xlabel=variable_data[v]["name"] + " [GeV]",
                                filename=f"{v}_sys_{s}_uncert_liny.png",
                            )

                        # massless
                        for v in measurement_vars_unitless:
                            analysis.plot(
                                val=[
                                    analysis[mc_sample].get_hist(
                                        variable=f"{v}_{sys_name}_{s}_uncert",
                                        systematic=NOMINAL_NAME,
                                        selection=f"{wp}_SR_passID",
                                    )
                                    for sys_name in analysis["wtaunu"].sys_list
                                ],
                                logx=False,
                                logy=False,
                                ylabel=ylabel,
                                **default_args,
                                xlabel=variable_data[v]["name"],
                                filename=f"{v}_sys_{s}_uncert_liny.png",
                            )

        # NO FAKES
        # ===========================================================================
        analysis.paths.plot_dir = wp_dir / "no_fakes"
        default_args = {
            "dataset": all_samples,
            "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
            "do_stat": True,
            "do_sys": False,
            "selection": f"{wp}_SR_passID",
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
        }

        # mass-like variables
        for var in measurement_vars_mass:
            analysis.plot(
                val=var,
                **default_args,
                logx=True,
                logy=True,
                filename=f"{wp}_{var}_stack_no_fakes_log.png",
            )
            analysis.plot(
                val=var,
                **default_args,
                logy=False,
                logx=False,
                filename=f"{wp}_{var}_stack_no_fakes_liny.png",
            )

        # unitless variables
        for var in measurement_vars_unitless:
            analysis.plot(
                val=var,
                **default_args,
                logx=False,
                logy=True,
                filename=f"{wp}_{var}_stack_no_fakes_log.png",
            )
            analysis.plot(
                val=var,
                **default_args,
                logy=False,
                logx=False,
                filename=f"{wp}_{var}_stack_no_fakes_liny.png",
            )

        # Fakes distribution across kinematic variable for signal MC
        # -----------------------------------------------------------------------
        analysis.paths.plot_dir = wp_dir / "fakes_distributions"
        for var in measurement_vars:
            xlabel = get_axis_labels(var)[0]
            sel = f"{wp}_SR_passID"
            mc = "wtaunu"

            # for all MC
            wtaunu_el_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isElectron", dataset=d, selection=sel
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isElectron_{sel}_PROFILE",
            )
            wtaunu_mu_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isMuon", dataset=d, selection=sel
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isMuon_{sel}_PROFILE",
            )
            wtaunu_ph_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isPhoton", dataset=d, selection=sel
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isPhoton_{sel}_PROFILE",
            )
            wtaunu_jet_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(f"{var}_MatchedTruthParticle_isJet", dataset=d, selection=sel)
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isJet_{sel}_PROFILE",
            )
            wtaunu_true_taus = analysis.sum_hists(
                [
                    analysis.get_hist(f"{var}_MatchedTruthParticle_isTau", dataset=d, selection=sel)
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isTau_{sel}_PROFILE",
            )
            analysis.plot(
                [
                    wtaunu_jet_fakes,
                    wtaunu_ph_fakes,
                    wtaunu_mu_fakes,
                    wtaunu_el_fakes,
                    wtaunu_true_taus,
                ],
                label=[
                    "Jet-matched fake taus",
                    "Photon-matched fake taus",
                    "Muon-matched fake taus",
                    "electron-matched fake taus",
                    "True taus",
                ],
                sort=False,
                do_stat=False,
                colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
                title=f"Fake fractions for {var} in {sel} for all MC in SR",
                y_axlim=(0, 1),
                kind="stack",
                xlabel=xlabel,
                ylabel="Fraction of fake matched taus in signal MC",
                filename=f"{wp}_all_mc_{var}_{sel}_fake_fractions.png",
            )

            # for all MC
            el_fakes = analysis.get_hist(
                f"{var}_MatchedTruthParticle_isElectron", dataset=mc, selection=sel
            )
            mu_fakes = analysis.get_hist(
                f"{var}_MatchedTruthParticle_isMuon", dataset=mc, selection=sel
            )
            ph_fakes = analysis.get_hist(
                f"{var}_MatchedTruthParticle_isPhoton", dataset=mc, selection=sel
            )
            jet_fakes = analysis.get_hist(
                f"{var}_MatchedTruthParticle_isJet", dataset=mc, selection=sel
            )
            true_taus = analysis.get_hist(
                f"{var}_MatchedTruthParticle_isTau", dataset=mc, selection=sel
            )

            sel_hist = analysis.get_hist(var, "wtaunu", selection=sel, TH1=True)
            nbins = sel_hist.GetNbinsX()

            analysis.plot(
                [jet_fakes, ph_fakes, mu_fakes, el_fakes, true_taus],
                label=[
                    "Jet Fakes",
                    "Photon Fakes",
                    "Muon Fakes",
                    "Electron Fakes",
                    "True taus",
                ],
                sort=False,
                do_stat=False,
                colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
                title=f"Fake fractions for {var} in {mc} for SR",
                y_axlim=(0, 1),
                kind="stack",
                xlabel=xlabel,
                ylabel="Fraction of fake matched taus in signal MC",
                filename=f"{wp}_{mc}_{var}_{sel}_fake_fractions.png",
            )

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
