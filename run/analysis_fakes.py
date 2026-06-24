from pathlib import Path

import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from utils.helper_functions import smart_join
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
YEAR = 2017

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
fail_loose = Cut(
    r"\mathrm{Fail Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight = Cut(
    r"\mathrm{Fail Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_met170 = Cut(
    r"$E_T^{\mathrm{miss}} > 170$",
    r"MET_met > 170",
)
pass_150met = Cut(
    r"$E_T^{\mathrm{miss}} < 150$",
    r"MET_met < 150",
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
fail_truetau = Cut(
    r"Fake Tau",
    "!(MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true)",
)

# selections
selections_loose: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_loose,
        pass_met170,
    ],
    "loose_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose,
        pass_met170,
    ],
    "loose_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_loose,
        pass_150met,
    ],
    "loose_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_loose,
        pass_150met,
    ],
}
selections_medium: dict[str, list[Cut]] = {
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_medium,
        pass_met170,
    ],
    "medium_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium,
        pass_met170,
    ],
    "medium_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_medium,
        pass_150met,
    ],
    "medium_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_medium,
        pass_150met,
    ],
}
selections_tight: dict[str, list[Cut]] = {
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_tight,
        pass_met170,
    ],
    "tight_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight,
        pass_met170,
    ],
    "tight_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_tight,
        pass_150met,
    ],
    "tight_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        fail_tight,
        pass_150met,
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

datasets: dict[str, dict] = {
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
        do_systematics=False,
        # regen_metadata=True,
        # output_dir="/eos/home-k/kghorban/framework_outputs/analysis_main",
        ttree=NOMINAL_NAME,
        analysis_label="analysis_main",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        systematics_for_selection={".*SR_passID"},
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
    fakes_colour = next(analysis.c_iter)

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
                analysis.do_fakes_estimate(
                    fakes_source,
                    measurement_vars,
                    f"{nprong}{wp}_CR_passID",
                    f"{nprong}{wp}_CR_failID",
                    f"{nprong}{wp}_SR_passID",
                    f"{nprong}{wp}_SR_failID",
                    f"trueTau_{nprong}{wp}_CR_passID",
                    f"trueTau_{nprong}{wp}_CR_failID",
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
                default_args = {
                    "do_stat": False,
                    "logx": True,
                    "logy": True,
                    "xlabel": variable_data[fakes_source]["name"] + " [GeV]",
                    "ylabel": "Weighted events",
                    "title": smart_join(
                        f"{variable_data[fakes_source]['name']} fakes binning",
                        wp.title(),
                        str(YEAR),
                        f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        sep=" | ",
                    ),
                }

                analysis.paths.plot_dir = wp_dir / "fakes_intermediates"
                analysis.plot(
                    [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
                    label=[
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,data}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,data}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                    ],
                    **default_args,
                    filename=f"{nprong}{wp}_FF_histograms_{fakes_source}.png",
                )
                analysis.plot(
                    [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
                    label=[
                        r"$N^{\mathrm{CR}}_{\mathrm{failID,data}} - N^{\mathrm{CR}}_{\mathrm{failID,MC}}$",
                        r"$N^{\mathrm{CR}}_{\mathrm{passID,data}} - N^{\mathrm{CR}}_{\mathrm{passID,MC}}$",
                    ],
                    **default_args,
                    ratio_plot=True,
                    filename=f"{nprong}{wp}_FF_histograms_diff_{fakes_source}.png",
                    ratio_label="FF",
                )
                analysis.plot(
                    [SR_failID_data, SR_failID_mc],
                    label=["SR_failID_data", "SR_failID_mc"],
                    **default_args,
                    filename=f"{nprong}{wp}_FF_calculation_{fakes_source}.png",
                )
                analysis.plot(
                    SR_failID_data - SR_failID_mc,
                    label="SR_failID_data - SR_failID_mc",
                    **default_args,
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
                            + [f"{nprong}{wp}_SR_passID"] * len(mc_samples)
                            + [None]
                    ),
                    "label": [analysis[ds].label for ds in all_samples] + ["Fake Jet Estimate"],
                    "colour": [analysis[ds].colour for ds in all_samples] + [fakes_colour],
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


                # mass variables
                for v in measurement_vars:
                    if v in measurement_vars_mass:
                        default_args.update(
                            {"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"}
                        )
                    elif v in measurement_vars_unitless:
                        default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})
                    ff_vals = [v] * len(all_samples) + [
                        f"{nprong}{wp}_{v}_fakes_bkg_{fakes_source}_src"
                    ]
                    analysis.plot(
                        val=ff_vals,
                        **default_args,
                        logy=True,
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_log.png",
                    )
                    analysis.plot(
                        val=ff_vals,
                        **default_args,
                        logy=False,
                        filename=f"{nprong}{wp}_{v}_fakes_stack_{fakes_source}_liny.png",
                    )

        # Fake factors
        # ----------------------------------------------------------------------------
        analysis.paths.plot_dir = wp_dir / "fakes_comparisons"
        for fakes_source in fakes_sources:
            analysis.plot(
                val=[
                    f"1prong_{wp}_{fakes_source}_FF",
                    f"3prong_{wp}_{fakes_source}_FF",
                    f"{wp}_{fakes_source}_FF",
                ],
                label=[
                    "1-prong",
                    "3-prong",
                    "1 + 3 prong",
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
                filename=f"{wp}_{fakes_source}_FF_prong_compare.png",
            )

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
