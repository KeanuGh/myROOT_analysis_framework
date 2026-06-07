from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from src.analysis import Analysis
from src.cutting import Cut
from src.histogram import Histogram1D
from utils.helper_functions import smart_join, get_base_sys_name
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
YEAR = 2017
DO_SYS = True

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
pass_taupt170 = Cut(
    r"$p_T^\tau > 170$",
    r"TauPt > 170",
)
pass_mtw350 = Cut(
    r"$m_T^W > 350$",
    r"MTW > 350",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
fail_loose_jet = Cut(
    r"\mathrm{Fail Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
fail_loose_lep = Cut(
    r"\mathrm{Fail Loose ID}",
    r"!(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium_jet = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium_lep = Cut(
    r"\mathrm{Fail Medium ID}",
    r"!(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight_jet = Cut(
    r"\mathrm{Fail Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight_lep = Cut(
    r"\mathrm{Fail Tight ID}",
    r"!(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
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
pass_truetau_jet = Cut(
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
)
pass_truetau_lep = Cut(
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isJet == true || MatchedTruthParticle_isPhoton == true",
)
fail_truetau_jet = Cut(
    r"Fake Tau",
    "!(MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true)",
)
fail_truetau_lep = Cut(
    r"Fake Tau",
    "!(MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isJet == true || MatchedTruthParticle_isPhoton == true)",
)

# selections
selections_loose: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_loose,
        pass_met150,
    ],
    "loose_SR_failID_jet": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose_jet,
        pass_met150,
    ],
    "loose_SR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose_lep,
        pass_met150,
    ],
    "loose_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_loose,
        pass_100met,
    ],
    "loose_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose_jet,
        pass_100met,
    ],
    "loose_CR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose_lep,
        pass_100met,
    ],
}
selections_medium: dict[str, list[Cut]] = {
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_medium,
        pass_met150,
    ],
    "medium_SR_failID_jet": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium_jet,
        pass_met150,
    ],
    "medium_SR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium_lep,
        pass_met150,
    ],
    "medium_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_medium,
        pass_100met,
    ],
    "medium_CR_failID_jet": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium_jet,
        pass_100met,
    ],
    "medium_CR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium_lep,
        pass_100met,
    ],
}
selections_tight: dict[str, list[Cut]] = {
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_tight,
        pass_met150,
    ],
    "tight_SR_failID_jet": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight_jet,
        pass_met150,
    ],
    "tight_SR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight_lep,
        pass_met150,
    ],
    "tight_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_tight,
        pass_100met,
    ],
    "tight_CR_failID_jet": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight_jet,
        pass_100met,
    ],
    "tight_CR_failID_lep": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight_lep,
        pass_100met,
    ],
}
selections = selections_loose | selections_medium | selections_tight

# define selection for MC samples
selections_list = list(selections.keys())
selections_cuts = list(selections.values())
for selection, cut_list in zip(selections_list, selections_cuts):
    if "_lep" not in selection:
        selections[f"trueTau_jet_{selection}"] = cut_list + [pass_truetau_jet]
    if "_jet" not in selection:
        selections[f"trueTau_lep_{selection}"] = cut_list + [pass_truetau_lep]
    # define selections for 1- or 3- tau prongs
    for cutstr, cut_name in [
        ("TauNCoreTracks == 1", "1prong"),
        ("TauNCoreTracks == 3", "3prong"),
    ]:
        selections[f"{cut_name}_{selection}"] = cut_list + [Cut(cut_name, cutstr)]
        if "_lep" not in selection:
            selections[f"trueTau_jet_{cut_name}_{selection}"] = cut_list + [
                pass_truetau_jet,
                Cut(cut_name, cutstr),
            ]
        if "_jet" not in selection:
            selections[f"trueTau_lep_{cut_name}_{selection}"] = cut_list + [
                pass_truetau_lep,
                Cut(cut_name, cutstr),
            ]
# for data
selections_notruth = {n: s for n, s in selections.items() if not n.startswith("trueTau_")}
#
# # to compare fakes
# selections |= {
#     "fakeTau_jet_loose_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
#     "fakeTau_jet_medium_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
#     "fakeTau_jet_tight_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
#     "fakeTau_lep_loose_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
#     "fakeTau_lep_medium_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
#     "fakeTau_lep_tight_SR_passID": [
#         pass_presel,
#         pass_taupt170,
#         pass_mtw350,
#         pass_loose,
#         pass_met150,
#         fail_truetau_jet,
#     ],
# }

# VARIABLES
# ========================================================================
wanted_variables = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "TauRNNJetScore",
    "TauBDTEleScore",
    # "DeltaPhi_tau_met",
    "TauNCoreTracks",
}
measurement_vars_mass = [
    "TauPt",
    "MTW",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "TauBDTEleScore",
    "TauRNNJetScore",
    # "DeltaPhi_tau_met",
    "TauNCoreTracks",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"

datasets: Dict[str, Dict] = {
    # DATA
    # ====================================================================
    "data": {
        # "data_path": DTA_PATH / "*data17*/*.root",
        "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
        "label": "data",
        "is_data": True,
        "selections": selections_notruth,
        "snapshot": {"selections": selections_notruth, "systematics": NOMINAL_NAME},
        # "rerun": True,
        # "regen_histograms": True,
    },
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
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
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
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
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
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
    },
    "top": {
        "data_path": [
            DTA_PATH / "*PP8_singletop*/*.root",
            DTA_PATH / "*PP8_tchan*/*.root",
            DTA_PATH / "*PP8_Wt_DR_dilepton*/*.root",
            DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
        ],
        "label": "Top",
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
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
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
    },
}


def run_analysis() -> Analysis:
    """Run analysis"""
    n_edges = 16
    return Analysis(
        datasets,
        year=YEAR,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=DO_SYS,
        # regen_metadata=True,
        # output_dir="/eos/home-k/kghorban/framework_outputs/analysis_main",
        ttree=NOMINAL_NAME,
        analysis_label="analysis_main_lepfakes",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        skip_sys={
            r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
            r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
        },
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, n_edges),
                "TauPt": np.geomspace(170, 1000, n_edges),
                "TauEta": np.linspace(-2.5, 2.5, n_edges),
                "EleEta": np.linspace(-2.5, 2.5, n_edges),
                "MuonEta": np.linspace(-2.5, 2.5, n_edges),
                "MET_met": np.geomspace(150, 1000, n_edges),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, n_edges),
                "TauPt_div_MET": np.linspace(0, 3, 21),
                "TauRNNJetScore": np.linspace(0, 1, 36),
                "TauBDTEleScore": np.linspace(0, 1, 36),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TruthTauPt": np.geomspace(1, 1000, 21),
            },
            ".*_CR_.*ID": {
                "MET_met": np.geomspace(1, 100, n_edges),
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
    for mc in mc_samples:
        analysis[mc].calculate_systematic_uncertainties()
    jetfakes_colour = next(analysis.c_iter)
    leptonfakes_colour = next(analysis.c_iter)
    # for dataset in analysis:
    #     ROOT.RDF.SaveGraph(
    #         dataset.filters["T_s1thv_NOMINAL"]["loose_SR_passID"].df,
    #         f"{analysis.paths.output_dir}/{dataset.name}_SR_graph.dot",
    #     )

    # # print histograms
    # for dataset in analysis:
    #     dataset.histogram_printout(to_file="txt", to_dir=analysis.paths.latex_dir)
    wps = (
        "loose",
        "medium",
        "tight",
    )
    prongs = (
        "1prong",
        "3prong",
        "",
    )
    fakes_sources = (
        "MTW",
        "TauPt",
    )
    source_colours = [
        "darkviolet",
        "mediumblue",
    ]

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in wps:
        wp_dir = base_plotting_dir / wp

        # FAKES ESTIMATE
        # ========================================================================
        for prong_str in prongs:
            nprong = prong_str + "_" if prong_str else ""

            for fakes_source in fakes_sources:
                for fakes_type in ("jet", "lep"):
                    # jet fakes
                    analysis.do_fakes_estimate(
                        fakes_source,
                        measurement_vars,
                        f"{nprong}{wp}_CR_passID",
                        f"{nprong}{wp}_CR_failID_{fakes_type}",
                        f"{nprong}{wp}_SR_passID",
                        f"{nprong}{wp}_SR_failID_{fakes_type}",
                        f"trueTau_{fakes_type}_{nprong}{wp}_CR_passID",
                        f"trueTau_{fakes_type}_{nprong}{wp}_CR_failID",
                        f"trueTau_{fakes_type}_{nprong}{wp}_SR_passID",
                        f"trueTau_{fakes_type}_{nprong}{wp}_SR_failID",
                        name=f"{nprong}{wp}_{fakes_type}fake",
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
                        f"{nprong}{wp}_{fakes_type}fake_all_mc_{fakes_source}_trueTau_{fakes_type}_{nprong}{wp}_CR_passID"
                    )
                    CR_failID_mc = analysis.get_hist(
                        f"{nprong}{wp}_{fakes_type}fake_all_mc_{fakes_source}_trueTau_{fakes_type}_{nprong}{wp}_CR_failID"
                    )
                    SR_failID_mc = analysis.get_hist(
                        f"{nprong}{wp}_{fakes_type}fake_all_mc_{fakes_source}_trueTau_{fakes_type}_{nprong}{wp}_SR_failID"
                    )
                    default_args = {
                        "do_stat": False,
                        "logx": True,
                        "logy": True,
                        "xlabel": variable_data[fakes_source]["name"] + " [GeV]",
                        "ylabel": "Weighted events",
                        "title": smart_join(
                            f"{variable_data[fakes_source]['name']} binning",
                            f"{fakes_type.title()} fakes",
                            wp.title(),
                            str(YEAR),
                            f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                            sep=" | ",
                        ),
                    }

                    analysis.paths.plot_dir = wp_dir / "fakes_intermediates" / fakes_type
                    analysis.plot(
                        [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
                        label=[
                            r"$N^{\mathrm{CR}}_{\mathrm{passID,data}}$",
                            r"$N^{\mathrm{CR}}_{\mathrm{failID,data}}$",
                            r"$N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                            r"$N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                        ],
                        **default_args,
                        filename=f"{nprong}{wp}_{fakes_type}fake_FF_histograms_{fakes_source}.png",
                    )
                    analysis.plot(
                        [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
                        label=[
                            r"$N^{\mathrm{CR}}_{\mathrm{failID,data}} - N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                            r"$N^{\mathrm{CR}}_{\mathrm{passID,data}} - N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                        ],
                        **default_args,
                        ratio_plot=True,
                        filename=f"{nprong}{wp}_{fakes_type}fake_FF_histograms_diff_{fakes_source}.png",
                        ratio_label="FF",
                    )
                    analysis.plot(
                        [SR_failID_data, SR_failID_mc],
                        label=["SR_failID_data", "SR_failID_mc"],
                        **default_args,
                        filename=f"{nprong}{wp}_{fakes_type}fake_FF_calculation_{fakes_source}.png",
                    )
                    analysis.plot(
                        SR_failID_data - SR_failID_mc,
                        label="SR_failID_data - SR_failID_mc",
                        **default_args,
                        filename=f"{nprong}{wp}_{fakes_type}fake_FF_calculation_delta_SR_fail_{fakes_source}.png",
                    )

                    # Fake factors
                    # ----------------------------------------------------------------------------
                    analysis.paths.plot_dir = wp_dir / "fake_factors"
                    analysis.plot(
                        val=f"{nprong}{wp}_{fakes_type}fake_{fakes_source}_FF",
                        xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                        do_stat=False,
                        logx=False,
                        logy=False,
                        ylabel="Fake factor",
                        filename=f"{nprong}{wp}_{fakes_type}fake_{fakes_source}_FF.png",
                    )

                # Stacks with Fakes background
                # ----------------------------------------------------------------------------
                analysis.paths.plot_dir = wp_dir / "fakes_stacks"
                # log axes
                default_args = {
                    "dataset": all_samples + [None, None],
                    "systematic": NOMINAL_NAME,
                    "selection": (
                            [f"{nprong}{wp}_SR_passID"]
                            + [f"{nprong}{wp}_SR_passID"] * len(mc_samples)
                            + [None, None]
                    ),
                    "label": [analysis[ds].label for ds in all_samples]
                             + ["Jet Fakes"]
                             + ["Lepton Fakes"],
                    "colour": [analysis[ds].colour for ds in all_samples]
                              + [jetfakes_colour, leptonfakes_colour],
                    "title": smart_join(
                        f"{wp.title()} ID SR",
                        str(2017),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                    "ylabel": "Events",
                    "do_stat": True,
                    "do_syst": True,
                    "ratio_plot": True,
                    "ratio_axlim": (0.5, 1.5),
                    "kind": "stack",
                }


                def FF_vars(s: str) -> list[str]:
                    """List of variable names for each sample"""
                    return [s] * (len(all_samples)) + [
                        f"{nprong}{wp}_jetfake_{s}_fakes_bkg_{fakes_source}_src",
                        f"{nprong}{wp}_lepfake_{s}_fakes_bkg_{fakes_source}_src",
                    ]


                # mass variables
                for v in measurement_vars:
                    if v in measurement_vars_mass:
                        default_args.update(
                            {"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"}
                        )
                    elif v in measurement_vars_unitless:
                        default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})
                    analysis.plot(
                        val=FF_vars(v),
                        **default_args,
                        logy=True,
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_log.png",
                    )
                    analysis.plot(
                        val=FF_vars(v),
                        **default_args,
                        logy=False,
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_liny.png",
                    )

        # Fake factors
        # ----------------------------------------------------------------------------
        for fakes_type in ("jet", "lep"):
            analysis.paths.plot_dir = wp_dir / "fakes_comparisons" / fakes_type
            for fakes_source in fakes_sources:
                analysis.plot(
                    val=[
                        f"1prong_{wp}_{fakes_type}_{fakes_source}_FF",
                        f"3prong_{wp}_{fakes_type}_{fakes_source}_FF",
                        f"{wp}_{fakes_type}_{fakes_source}_FF",
                    ],
                    label=[
                        "1-prong",
                        "3-prong",
                        "Inclusive",
                    ],
                    title=smart_join(
                        f"{wp.title()} ID",
                        str(2017),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                    xlabel=r"$p_T^\tau$ [GeV]" if fakes_source == "TauPt" else r"$m_T^W$ [GeV]",
                    do_stat=True,
                    logx=True,
                    logy=False,
                    ylabel="Fake factor",
                    filename=f"{wp}_{fakes_type}_{fakes_source}_FF_prong_compare.png",
                )

            for v in measurement_vars:
                # compare fake factors to "true tau" MC selections
                # TODO: prongs?
                analysis.plot(
                    val=v,
                    dataset=mc_samples,
                    selection=f"fakeTau_{fakes_type}_{wp}_SR_passID",
                    plot_as_data=[
                        analysis.histograms[f"{wp}_{fakes_type}_{v}_fakes_bkg_{fakes_source}_src"]
                        for fakes_source in fakes_sources
                    ],
                    title=smart_join(
                        f"{wp.title()} ID Signal Region",
                        str(2017),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                    data_label=[
                        rf"$m_\mathrm{{T}}^W$ Fake {fakes_type.title()} Estimate",
                        rf"$p_\mathrm{{T}}^\tau$ Fake {fakes_type.title()} Estimate",
                    ],
                    data_colour=source_colours,
                    do_stat=True,
                    do_syst=False,
                    logy=False,
                    ratio_plot=True,
                    ratio_label="Fake Est. / MC",
                    ratio_axlim=(0.5, 1.5),
                    logx=True if v in measurement_vars_mass else False,
                    kind="stack",
                    filename=f"{wp}_{v}_fakes_stack_multijet_est_true.png",
                )

        # Direct data scaling comparison
        # ----------------------------------------------------------------------------
        # log axes
        default_args = {
            "title": f"{wp.title()} ID | {YEAR} | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
            "label": ["Data SR", "MC + MTW Fakes", "MC + TauPt Fakes", "MC w/ no Fakes"],
            "colour": ["k"] + source_colours + ["teal"],
            "do_stat": True,
            "do_syst": DO_SYS,
            "suffix": "fake_scaled_log",
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
        }


        def full_bkg(variable: str, t: str | None = None) -> Histogram1D:
            """Sum of all backgrounds + signal + FF"""
            h = Histogram1D(
                th1=analysis.sum_hists(
                    [
                        analysis.get_hist(
                            variable=variable,
                            dataset=ds_,
                            systematic=NOMINAL_NAME,
                            selection=f"{wp}_SR_passID",
                        )
                        for ds_ in mc_samples
                    ]
                )
            )
            if t:
                h += Histogram1D(
                    th1=analysis.get_hist(f"{wp}_jet_{variable}_fakes_bkg_{t}_src")
                ) + Histogram1D(th1=analysis.get_hist(f"{wp}_lep_{variable}_fakes_bkg_{t}_src"))
            return h


        for v in measurement_vars:
            if v in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    full_bkg(v, "MTW"),
                    full_bkg(v, "TauPt"),
                    full_bkg(v, None),
                ],
                logy=True,
                **default_args,
                filename=f"{wp}_FF_compare_{v}_log.png",
            )
            analysis.plot(
                val=[
                    analysis.get_hist(variable=v, dataset="data", selection=f"{wp}_SR_passID"),
                    full_bkg(v, "MTW"),
                    full_bkg(v, "TauPt"),
                    full_bkg(v, None),
                ],
                logy=False,
                **default_args,
                filename=f"{wp}_FF_compare_{v}_liny.png",
            )

        if DO_SYS:
            # SYSTEMATIC UNCERTAINTIES
            # ===========================================================================

            def strip_sys_prefix(s: str) -> str:
                """remove prefix from sys names"""
                return (
                    s.removeprefix("TAUS_TRUEHADTAU_").removeprefix("TES_SME").removeprefix("EFF_")
                )


            # list of systematic variations
            sys_list_eff = sorted(set(get_base_sys_name(s) for s in analysis["wtaunu"].eff_sys_set))
            sys_list_tes = sorted(set(get_base_sys_name(s) for s in analysis["wtaunu"].tes_sys_set))
            cmap = plt.get_cmap("jet")
            colours_eff = [tuple(c) for c in cmap(np.linspace(0, 1.0, len(sys_list_eff)))]
            colours_tes = [tuple(c) for c in cmap(np.linspace(0, 1.0, len(sys_list_tes)))]

            # for each sample
            for mc_sample in mc_samples:
                # mass variables
                for nprong in ("", "1prong_", "3prong_"):
                    selection = f"{nprong}{wp}_SR_passID"

                    for v in measurement_vars:
                        analysis.paths.plot_dir = wp_dir / "systematics" / mc_sample / nprong
                        default_args = {
                            "do_stat": False,
                            "do_syst": False,
                            "ratio_plot": False,
                            "ratio_err": None,
                            "logy": False,
                            "legend_params": {"ncols": 1, "fontsize": 8},
                            "ylabel": "Percentage uncertainty / %",
                            "title": (
                                f"{datasets[mc_sample]['label']} | "
                                f"Signal Region | "
                                f"{YEAR} | "
                                f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
                            ),
                        }
                        if v in measurement_vars_mass:
                            default_args.update(
                                {"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"}
                            )
                        elif v in measurement_vars_unitless:
                            default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})

                        analysis.plot(
                            val=[
                                analysis[mc_sample].get_hist(
                                    variable=f"{v}_{sys_name}_pct_uncert",
                                    systematic=NOMINAL_NAME,
                                    selection=selection,
                                )
                                for sys_name in sys_list_eff
                            ],
                            label=[strip_sys_prefix(sys) for sys in sys_list_eff],
                            colour=colours_eff,
                            y_axlim=(0, 20),
                            **default_args,
                            filename=f"{v}_sys_eff_pct_uncert_liny.png",
                        )
                        analysis.plot(
                            val=[
                                analysis[mc_sample].get_hist(
                                    variable=f"{v}_{sys_name}_pct_uncert",
                                    systematic=NOMINAL_NAME,
                                    selection=selection,
                                )
                                for sys_name in sys_list_tes
                            ],
                            label=[strip_sys_prefix(sys) for sys in sys_list_tes],
                            y_axlim=(0, 50),
                            colour=colours_tes,
                            **default_args,
                            filename=f"{v}_sys_tes_pct_uncert_liny.png",
                        )

                        # detector systematics
                        default_args.update(
                            {
                                "selection": selection,
                                "dataset": mc_sample,
                                "ratio_plot": True,
                                "ratio_label": "sys/nominal",
                                "ratio_axlim": (0.8, 1.2),
                                "logx": True if v in measurement_vars_mass else False,
                                "ylabel": "Weighted Entries",
                            }
                        )
                        analysis.plot(
                            val=v,
                            systematic=[
                                NOMINAL_NAME,
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1up",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1down",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt__1up",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt__1down",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt__1up",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt__1down",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt__1up",
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt__1down",
                            ],
                            label=[
                                "Nominal",
                                "Endcap_LowPt__1up",
                                "Endcap_LowPt__1down",
                                "Endcap_HighPt__1up",
                                "Endcap_HighPt__1down",
                                "Barrel_LowPt__1up",
                                "Barrel_LowPt__1down",
                                "Barrel_HighPt__1up",
                                "Barrel_HighPt__1down",
                            ],
                            colour=[
                                "k",
                                (0.0, 0.0, 0.5, 1.0),
                                (0.0, 0.0, 0.5, 1.0),
                                (0.0, 0.8333333333333334, 1.0, 1.0),
                                (0.0, 0.8333333333333334, 1.0, 1.0),
                                (1.0, 0.9012345679012348, 0.0, 1.0),
                                (1.0, 0.9012345679012348, 0.0, 1.0),
                                (0.5, 0.0, 0.0, 1.0),
                                (0.5, 0.0, 0.0, 1.0),
                            ],
                            linestyle=[
                                "solid",
                                "solid",
                                "dotted",
                                "solid",
                                "dotted",
                                "solid",
                                "dotted",
                                "solid",
                                "dotted",
                            ],
                            **default_args,
                            filename=f"{v}_sys_DETECTOR.png",
                        )
                        # TRIGGER systematics
                        analysis.plot(
                            val=v,
                            systematic=[
                                NOMINAL_NAME,
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718__1up",
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718__1down",
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718__1up",
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718__1down",
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718__1up",
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718__1down",
                            ],
                            label=[
                                "Nominal",
                                "TRIGGER_SYST161718__1up",
                                "TRIGGER_SYST161718__1down",
                                "TRIGGER_STATMC161718_1up",
                                "TRIGGER_STATMC161718_1down",
                                "TRIGGER_STATDATA161718__1up",
                                "TRIGGER_STATDATA161718__1down",
                            ],
                            colour=[
                                "k",
                                (0.0, 0.0, 0.5, 1.0),
                                (0.0, 0.0, 0.5, 1.0),
                                (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                                (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                                (0.5, 0.0, 0.0, 1.0),
                                (0.5, 0.0, 0.0, 1.0),
                            ],
                            linestyle=[
                                "solid",
                                "solid",
                                "dotted",
                                "solid",
                                "dotted",
                                "solid",
                                "dotted",
                            ],
                            **default_args,
                            filename=f"{v}_sys_TRIGGER.png",
                        )

        # NO FAKES
        # ===========================================================================
        default_args = {
            "dataset": all_samples,
            "do_stat": True,
            "do_syst": False,
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
        }

        # see try different selections
        for selection in [
            f"{wp}_SR_passID",
            f"{wp}_SR_failID",
            f"{wp}_CR_passID",
            f"{wp}_CR_failID",
            f"1prong_{wp}_SR_passID",
            f"1prong_{wp}_SR_failID",
            f"1prong_{wp}_CR_passID",
            f"1prong_{wp}_CR_failID",
            f"3prong_{wp}_SR_passID",
            f"3prong_{wp}_SR_failID",
            f"3prong_{wp}_CR_passID",
            f"3prong_{wp}_CR_failID",
        ]:
            default_args["title"] = smart_join(
                f"Data {YEAR}",
                "3-prong Taus"
                if "3prong" in selection
                else ("1-prong Taus" if "1prong" in selection else ""),
                f"{wp.title()} Tau ID",
                f"{analysis.global_lumi / 1000: .3g}fb$ ^ {{-1}}$",
                sep=" | ",
            )
            analysis.paths.plot_dir = wp_dir / "no_fakes" / selection
            default_args["selection"] = selection

            for var in measurement_vars:
                if var in measurement_vars_mass:
                    default_args.update(
                        {"logx": True, "xlabel": variable_data[var]["name"] + " [GeV]"}
                    )
                elif var in measurement_vars_unitless:
                    default_args.update({"logx": False, "xlabel": variable_data[var]["name"]})

                analysis.plot(
                    val=var,
                    **default_args,
                    logy=True,
                    filename=f"{wp}_{var}_stack_no_fakes_log.png",
                )
                analysis.plot(
                    val=var,
                    **default_args,
                    logy=False,
                    filename=f"{wp}_{var}_stack_no_fakes_liny.png",
                )

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
