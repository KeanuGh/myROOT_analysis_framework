from pathlib import Path
from typing import Dict

import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from src.histogram import Histogram1D
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
DO_SYS = False

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger"
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
# for data
selections_notruth = {n: s for n, s in selections.items() if not n.startswith("trueTau_")}

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
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"

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
        "selections": selections,
    },
    # DATA
    # ====================================================================
    "data": {
        # "data_path": DTA_PATH / "*data17*/*.root",
        "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
        "label": "data",
        "is_data": True,
        "selections": selections_notruth,
    },
}


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=DO_SYS,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        analysis_label="analysis_main",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 11),
                "TauPt": np.geomspace(170, 1000, 11),
                "TauEta": np.linspace(-2.5, 2.5, 11),
                "EleEta": np.linspace(-2.5, 2.5, 11),
                "MuonEta": np.linspace(-2.5, 2.5, 11),
                "MET_met": np.geomspace(150, 1000, 11),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 11),
                "TauPt_div_MET": np.linspace(0, 3, 21),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TruthTauPt": np.geomspace(1, 1000, 21),
            },
            ".*_CR_.*ID": {
                "MET_met": np.geomspace(1, 100, 11),
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
                    label="SR_failID_data - SR_failID_mc",
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
                    "do_syst": DO_SYS,
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
            "do_syst": DO_SYS,
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
            # list of systematic variations
            sys_list = sorted(list(analysis["wtaunu"].eff_sys_set | analysis["wtaunu"].tes_sys_set))
            default_args = {
                "label": sys_list,
                "do_stat": False,
                "do_syst": False,
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
                                    for sys_name in sys_list
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
                                    for sys_name in sys_list
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
            "do_syst": False,
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

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
