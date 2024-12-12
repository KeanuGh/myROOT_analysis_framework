from pathlib import Path
from typing import Dict

import ROOT
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
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
fail_truetau = Cut(
    r"Fake Tau",
    "!(MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true)",
)

# selections
selections: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_loose,
        pass_met170,
    ],
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_medium,
        pass_met170,
    ],
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw350,
        pass_tight,
        pass_met170,
    ],
}

# VARIABLES
# ========================================================================
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
    "TauNCoreTracks",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"

datasets: Dict[str, Dict] = {
    # DATA
    # ====================================================================
    "data": {
        "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
        "label": "data",
        "is_data": True,
        "selections": selections,
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
}
mtw_bins = np.array([350, 375, 400, 430, 465, 500, 550, 600, 700, 850, 1000, 2000], dtype="double")
taupt_bins = np.array([170, 200, 250, 300, 350, 425, 500, 600, 750, 900, 1000], dtype="double")


def run_analysis() -> Analysis:
    """Run analysis"""
    n_edges = 16
    return Analysis(
        datasets,
        year=YEAR,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=True,
        # regen_metadata=True,
        # output_dir="/eos/home-k/kghorban/framework_outputs/analysis_main",
        ttree=NOMINAL_NAME,
        analysis_label=Path(__file__).stem,
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        skip_sys={
            r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
            r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
        },
        binnings={
            "": {
                "MTW": mtw_bins,
                "MET_met": np.geomspace(170, 1000, n_edges),
                "TauPt": taupt_bins,
                "TauEta": np.linspace(-2.5, 2.5, n_edges),
                "TauPhi": np.linspace(-np.pi, np.pi, n_edges),
                "TauRNNJetScore": np.linspace(0, 1, 36),
                "TauBDTEleScore": np.linspace(0, 1, 36),
                "TauNCoreTracks": np.linspace(0, 4, 5),
            },
        },
    )


def strip_sys_prefix(s: str) -> str:
    """remove prefix from sys names"""
    return s.removeprefix("TAUS_TRUEHADTAU_").removeprefix("TES_SME").removeprefix("EFF_")


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()
    base_plotting_dir = analysis.paths.plot_dir
    all_samples = [analysis.data_sample] + analysis.mc_samples
    mc_samples = analysis.mc_samples
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    fakes_colour = next(analysis.c_iter)

    for wp in ("loose", "medium", "tight"):
        wp_dir = base_plotting_dir / wp

        # UNFOLD
        # ===============================================================================
        analysis.paths.plot_dir = wp_dir / "unfolded"
        for var in measurement_vars:
            truths = {
                "TauPt": "VisTruthTauPt",
                "MTW": "TruthMTW",
                "MET_met": "TruthNeutrinoPt",
                "TauEta": "TruthTauEta",
                "TauPhi": "TruthTauPhi",
            }
            if var not in truths:
                continue
            analysis.logger.info(f"Performing unfolding for {var} in {wp}")

            # get signal + background
            response = analysis.get_response_histogram(
                varname_reco=var,
                varname_truth=truths[var],
                dataset="wtaunu",
                selection=f"{wp}_reco_tau",
                systematic=NOMINAL_NAME,
            )

            def unfold_bayes(h: ROOT.TH1, i: int) -> ROOT.TH1:
                return ROOT.RooUnfoldBayes(response, h, i).Hreco()

            # get unfolded mc
            mc_samples_unfolded = {
                ds: unfold_bayes(
                    analysis[ds].get_hist(
                        variable=var,
                        systematic=NOMINAL_NAME,
                        selection=f"{wp}_SR_passID",
                    ),
                    2,
                )
                for ds in analysis.mc_samples
            }
            # unfolded data
            unfolded_data = unfold_bayes(
                analysis["data"].get_hist(
                    variable=var,
                    systematic=NOMINAL_NAME,
                    selection=f"{wp}_SR_passID",
                ),
                2,
            )

            # get fakes
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"analysis_fakes_{YEAR}/root/analysis_fakes_{YEAR}.root"
                )
            ) as file:
                fakes_hist = file.Get(f"{wp}_{var}_fakes_bkg_TauPt_src")
                fakes_hist.SetDirectory(0)
            # unfold fakes
            unfolded_fakes = unfold_bayes(fakes_hist, 2)

            analysis.plot(
                val=list(mc_samples_unfolded.values()) + [unfolded_fakes, unfolded_data],
                dataset=mc_samples + [None, "data"],
                selection=f"{wp}_SR_passID",
                systematic=NOMINAL_NAME,
                label=[analysis[ds].label for ds in mc_samples] + ["Fake Jets", "Data"],
                xlabel=(
                    variable_data[var]["name"] + (" [GeV]" if var in measurement_vars_mass else "")
                ),
                colour=[analysis[ds].colour for ds in mc_samples] + [fakes_colour, "k"],
                kind="stack",
                do_stat=False,
                do_syst=False,
                title=smart_join(
                    f"Unfolded data {YEAR}",
                    f"{wp.title()} Tau ID",
                    f"{analysis.global_lumi / 1000: .3g}fb$^{{-1}}$",
                    sep=" | ",
                ),
                scale_by_bin_width=True if var in measurement_vars_mass else False,
                ylabel="Events / GeV" if var in measurement_vars_mass else "Events",
                ratio_plot=True,
                ratio_label="Data / MC",
                ratio_axlim=(0.8, 1.2),
                label_params={"llabel": "Preliminary"},
                filename=f"{wp}_{var}_unfolded.png",
            )
            analysis.plot(
                val=list(mc_samples_unfolded.values()) + [unfolded_fakes, unfolded_data],
                dataset=mc_samples + [None, "data"],
                selection=f"{wp}_SR_passID",
                systematic=NOMINAL_NAME,
                label=[analysis[ds].label for ds in mc_samples] + ["Fake Jets", "Data"],
                xlabel=(
                    variable_data[var]["name"] + (" [GeV]" if var in measurement_vars_mass else "")
                ),
                colour=[analysis[ds].colour for ds in mc_samples] + [fakes_colour, "k"],
                kind="stack",
                do_stat=False,
                do_syst=False,
                title=smart_join(
                    f"Unfolded data {YEAR}",
                    f"{wp.title()} Tau ID",
                    f"{analysis.global_lumi / 1000: .3g}fb$^{{-1}}$",
                    sep=" | ",
                ),
                scale_by_bin_width=True if var in measurement_vars_mass else False,
                ylabel="Events / GeV" if var in measurement_vars_mass else "Events",
                ratio_plot=True,
                logy=True,
                ratio_label="Data / MC",
                ratio_axlim=(0.8, 1.2),
                label_params={"llabel": "Preliminary"},
                filename=f"{wp}_{var}_unfolded_logy.png",
            )

        # WITH FAKES
        # ===========================================================================
        fakes_colour = next(analysis.c_iter)
        default_args = {
            "dataset": mc_samples + [None, "data"],
            "selection": ([f"{wp}_SR_passID"] * len(mc_samples) + [None] + [f"{wp}_SR_passID"]),
            "label": [analysis[ds].label for ds in mc_samples] + ["Fake Jets", "Data"],
            "colour": [analysis[ds].colour for ds in mc_samples] + [fakes_colour, "k"],
            "systematic": NOMINAL_NAME,
            "do_stat": True,
            "do_syst": True,
            "ratio_plot": True,
            "ratio_axlim": (0.5, 1.5),
            "kind": "stack",
        }

        selection = f"{wp}_SR_passID"
        default_args["title"] = smart_join(
            f"Data {YEAR}",
            f"{wp.title()} Tau ID",
            f"{analysis.global_lumi / 1000: .3g}fb$^{{-1}}$",
            sep=" | ",
        )
        analysis.paths.plot_dir = wp_dir / "fakes"
        default_args["selection"] = selection

        for var in measurement_vars:
            if "Truth" in var:
                continue

            if var in measurement_vars_mass:
                default_args.update(
                    {
                        "logx": False,
                        "xlabel": variable_data[var]["name"] + " [GeV]",
                        "scale_by_bin_width": True,
                        "ylabel": "Events/GeV",
                    }
                )
            elif var in measurement_vars_unitless:
                default_args.update(
                    {
                        "logx": False,
                        "xlabel": variable_data[var]["name"],
                        "scale_by_bin_width": False,
                        "ylabel": "Events",
                    }
                )

            # get fakes
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"analysis_fakes_{YEAR}/root/analysis_fakes_{YEAR}.root"
                )
            ) as file:
                fakes_hist = file.Get(f"{wp}_{var}_fakes_bkg_TauPt_src")
                fakes_hist.SetDirectory(0)

            def FF_vars(s: str) -> list[str]:
                """List of variable names for each sample"""
                return [s] * (len(mc_samples)) + [fakes_hist, s]

            analysis.plot(
                val=FF_vars(var),
                **default_args,
                logy=True,
                filename=f"{wp}_{var}_stack_fakes_log.png",
            )
            analysis.plot(
                val=FF_vars(var),
                **default_args,
                logy=False,
                filename=f"{wp}_{var}_stack_fakes_liny.png",
            )

        # NO FAKES
        # ===========================================================================
        default_args = {
            "dataset": all_samples,
            "do_stat": True,
            "do_syst": True,
            "ratio_plot": True,
            "ratio_axlim": (0.5, 1.5),
            "kind": "stack",
        }

        # see try different selections
        selection = f"{wp}_SR_passID"
        default_args["title"] = smart_join(
            f"Data {YEAR}",
            f"{wp.title()} Tau ID",
            f"{analysis.global_lumi / 1000: .3g}fb$^{{-1}}$",
            sep=" | ",
        )
        analysis.paths.plot_dir = wp_dir / "no_fakes"
        default_args["selection"] = selection

        for var in measurement_vars:
            if "Truth" in var:
                continue

            if var in measurement_vars_mass:
                default_args.update(
                    {
                        "logx": False,
                        "xlabel": variable_data[var]["name"] + " [GeV]",
                        "scale_by_bin_width": True,
                        "ylabel": "Events/GeV",
                    }
                )
            elif var in measurement_vars_unitless:
                default_args.update(
                    {
                        "logx": False,
                        "xlabel": variable_data[var]["name"],
                        "scale_by_bin_width": False,
                        "ylabel": "Events",
                    }
                )

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

        # SYSTEMATIC UNCERTAINTIES
        # ===========================================================================
        # list of systematic variations
        sys_list_eff = sorted(set(get_base_sys_name(s) for s in analysis["wtaunu"].eff_sys_set))
        sys_list_tes = sorted(set(get_base_sys_name(s) for s in analysis["wtaunu"].tes_sys_set))
        cmap = plt.get_cmap("jet")
        colours_eff = [tuple(c) for c in cmap(np.linspace(0, 1.0, len(sys_list_eff)))]
        colours_tes = [tuple(c) for c in cmap(np.linspace(0, 1.0, len(sys_list_tes)))]

        # for each sample
        for mc_sample in mc_samples:
            # mass variables
            selection = f"{wp}_SR_passID"

            for v in measurement_vars:
                analysis.paths.plot_dir = wp_dir / "systematics" / mc_sample
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
                        "ratio_axlim": (0.5, 1.5),
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

    analysis.save_hists()
    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
