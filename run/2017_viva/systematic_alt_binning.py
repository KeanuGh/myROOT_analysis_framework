from pathlib import Path
from typing import Dict

import ROOT
import numpy as np
from matplotlib import pyplot as plt

from binnings import BINNINGS
from src.analysis import Analysis
from src.cutting import Cut
from utils.ROOT_utils import get_th1_bin_edges
from utils.helper_functions import get_base_sys_name
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
YEAR = 2017
DO_SYS = True

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger && (badJet == 0)"
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
pass_eta = Cut(
    r"$|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
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
        pass_eta,
        pass_mtw350,
        pass_loose,
        pass_met170,
    ],
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_medium,
        pass_met170,
    ],
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_tight,
        pass_met170,
    ],
}
# define selection for prongs
selections_list = list(selections.keys())
selections_cuts = list(selections.values())
for selection, cut_list in zip(selections_list, selections_cuts):
    # define selections for 1- or 3- tau prongs
    for cutstr, cut_name in [
        ("TauNCoreTracks == 1", "1prong"),
        ("TauNCoreTracks == 3", "3prong"),
        ("TauCharge == 1", "tauplus"),
        ("TauCharge == -1", "tauminus"),
    ]:
        selections[f"{cut_name}_{selection}"] = cut_list + [Cut(cut_name, cutstr)]

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "MTW",
    "TauPt",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "MET_phi",
    "AbsDeltaPhi_tau_met",
    "TauPt_div_MET",
]
measurement_vars = measurement_vars_mass + measurement_vars_unitless
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
    "wtaunu_had": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_H_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {
            "lm_cut": "TruthBosonM < 120",
            "full": "TruthTau_isHadronic",
        },
        "label": r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}$",
        "is_signal": True,
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
    },
    # BACKGROUNDS
    # ====================================================================
    # W -> tau -> lepton
    "wtaunu_lep": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_L_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {
            "lm_cut": "TruthBosonM < 120",
            "full": "!TruthTau_isHadronic",
        },
        "label": r"$W\rightarrow\tau\nu\rightarrow l\nu$",
        "is_signal": True,
        "snapshot": {"selections": list(selections.keys()), "systematics": NOMINAL_NAME},
        "selections": selections,
    },
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


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=YEAR,
        rerun=True,
        regen_histograms=True,
        do_systematics=True,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        analysis_label=Path(__file__).stem + "_viva",
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        skip_sys={
            r".*TAUS_TRUEHADTAU_EFF_RNNID_.*",
            r".*TAUS_TRUEHADTAU_EFF_JETID_.*",
        },
        binnings={
            "": BINNINGS
            | {
                "MTW": np.array([350, 600, 1000, 2000]),
                "TauPt": np.array([170, 250, 500, 1000]),
                "MET_met": np.array([170, 250, 500, 1000]),
                "TauEta": np.array([-2.5, -1.5, 1.5, 2.5]),
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
    for mc in mc_samples:
        analysis[mc].calculate_systematic_uncertainties()
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    fakes_colour = next(analysis.c_iter)
    truths = {
        "TauPt": "VisTruthTauPt",
        "MTW": "TruthMTW",
        "MET_met": "TruthNeutrinoPt",
        "TauEta": "VisTruthTauEta",
        "TauPhi": "VisTruthTauPhi",
    }
    sec_labels = {
        "": "",
        "1prong_": "1-prong taus",
        "3prong_": "3-prong taus",
        "tauplus_": r"$\tau^+$",
        "tauminus_": r"$\tau^-$",
    }

    for wp in ("loose", "medium", "tight"):
        for sec in ("", "1prong_", "3prong_", "tauplus_", "tauminus_"):
            wp_dir = base_plotting_dir / wp / sec

            for var in measurement_vars:
                if var not in truths:
                    continue

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
                selection = f"{sec}{wp}_SR_passID"

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

                    # save these for later
                    uncert_name = f"{v}_{sec}{wp}_{mc_sample}_uncert"
                    h = analysis[mc_sample].get_hist(
                        variable=v,
                        systematic=NOMINAL_NAME,
                        selection=selection,
                    )
                    tot_uncert = analysis.get_systematic_uncertainty(v, mc_sample, selection)[0]
                    h_uncert = ROOT.TH1F(
                        uncert_name, uncert_name, h.GetNbinsX(), get_th1_bin_edges(h)
                    )
                    for i in range(h.GetNbinsX()):
                        h_uncert.SetBinContent(i + 1, tot_uncert[i])
                    analysis.histograms[uncert_name] = h_uncert

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
                            "do_stat": False,
                            "do_syst": False,
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
