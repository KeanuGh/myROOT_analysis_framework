from pathlib import Path
from typing import Dict
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from src.cutfile import Cut
from src.analysis import Analysis

# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2023-10-25/")
DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
DATA_OUT_DIR = Path("/eos/home-k/kghorban/framework_outputs/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # DATA
        # ====================================================================
        "data": {
            "data_path": DTA_PATH / "*data17*/*.root",
            "label": "data",
        },

        # SIGNAL
        # ====================================================================
        "wtaunu_lm": {
            "data_path": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"$W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_hm": {
            "data_path": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"$W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },

        # BACKGROUNDS
        # ====================================================================

        # W -> light lepton
        "wlv_lm": {
            "data_path": [
                DTA_PATH / "*Sh_2211_Wmunu_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_maxHTpTV2*/*.root",
            ],
            "hard_cut": "TruthBosonM < 120",
            "label": r"$W\rightarrow (e/\mu)\nu$",
            "merge_into": "wlnu",
        },
        "wlv_hm": {
            "data_path": [
                DTA_PATH / "*Sh_2211_Wmunu_mW_120*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_mW_120*/*.root",
            ],
            "hard_cut": "TruthBosonM >= 120",
            "label": r"$W\rightarrow (e/\mu)\nu$",
            "merge_into": "wlnu",
        },

        # Z -> TauTau
        "ztautau_lm": {
            "data_path": DTA_PATH / "*Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
            "merge_into": "zll",
        },
        "ztautau_hm": {
            "data_path": DTA_PATH / "*Sh_2211_Ztautau_mZ_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
            "merge_into": "zll",
        },

        # Z -> Light Lepton
        "zll_lm": {
            "data_path": [
                DTA_PATH / "*Sh_2211_Zee_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_maxHTpTV2*/*.root",
            ],
            "hard_cut": "TruthBosonM < 120",
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
            "merge_into": "zll",
        },
        "zll_hm": {
            "data_path": [
                DTA_PATH / "*Sh_2211_Zmumu_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zee_mZ_120*/*.root",
            ],
            "hard_cut": "TruthBosonM >= 120",
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
            "merge_into": "zll",
        },

        # Z -> Neutrinos
        "znunu": {
            "data_path": DTA_PATH / "*Sh_2211_Znunu_pTV2*/*.root",
            "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
            "merge_into": "zll",
        },

        # TTBAR/TOP
        "ttbar": {
            "data_path": DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
            "label": r"$t\bar{t}$",
        },
        "other_top": {
            "data_path": [
                DTA_PATH / "*PP8_singletop*/*.root",
                DTA_PATH / "*PP8_tchan*/*.root",
                DTA_PATH / "*PP8_Wt_DR_dilepton*/*.root",
            ],
            "label": "Other Top",
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
    }

    cuts: dict[str, list[Cut]] = {
        "SR_passID" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Pass Loose ID}",
                r"TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 150",
                r"MET_met > 150",
            ),

        ],
        "SR_failID" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 150",
                r"MET_met > 150",
            ),
            Cut(
                r"\mathrm{Fail Loose ID}",
                r"!TauLooseWP",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
        ],
        "CR_passID" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Pass Loose ID}",
                r"TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} < 100",
                r"MET_met < 100",
            ),
        ],
        "CR_failID": [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Fail Loose ID}",
                r"!TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} < 100",
                r"MET_met < 100",
            ),
        ],

        # for MC
        "SR_passID_trueTau" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Pass Loose ID}",
                r"TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 150",
                r"MET_met > 150",
            ),
            Cut(
                r"True Tau",
                r"!isnan(TruthTauPt)",
            ),
        ],
        "SR_failID_trueTau" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Fail Loose ID}",
                r"!TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 150",
                r"MET_met > 150",
            ),
            Cut(
                r"True Tau",
                r"!isnan(TruthTauPt)",
            ),
        ],
        "CR_passID_trueTau" : [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Pass Loose ID}",
                r"TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} < 100",
                r"MET_met < 100",
            ),
            Cut(
                r"True Tau",
                r"!isnan(TruthTauPt)",
            ),
        ],
        "CR_failID_trueTau": [
            Cut(
                r"Pass preselection",
                r"passReco",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
            Cut(
                r"\mathrm{Fail Loose ID}",
                r"!TauLooseWP",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} < 100",
                r"MET_met < 100",
            ),
            Cut(
                r"True Tau",
                r"!isnan(TruthTauPt)",
            ),
        ]
    }

    wanted_branches = {
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
    }
    mc_samples = [
        "wtaunu",
        "wlnu",
        "zll",
        "ttbar",
        "other_top",
        "diboson",
    ]
    all_samples = ["data"] + mc_samples
    analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2017",
        # regen_histograms=True,
        ttree="T_s1thv_NOMINAL",
        cuts=cuts,
        analysis_label="fakes_estimate",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
        extract_vars=wanted_branches,
        binnings={
            "": {
            "MTW": np.geomspace(150, 1000, 20),
            "TauPt": np.geomspace(170, 1000, 20),
            "TauEta": np.linspace(-2.47, 2.47, 20),
            "EleEta": np.linspace(-2.47, 2.47, 20),
            "MuonEta": np.linspace(-2.5, 2.5, 20),
            "MET_met": np.geomspace(150, 1000, 20),
            "DeltaPhi_tau_met": np.linspace(0, 3.5, 20),
            "TauPt_div_MET": np.linspace(0, 3.5, 20),
            "TauRNNJetScore": np.linspace(0, 1, 20),
            "TauBDTEleScore": np.linspace(0, 1, 20),
            },
            "CR_failID": {
                "MET_met": np.geomspace(1, 100, 20),
            },
            "CR_passID": {
                "MET_met": np.geomspace(1, 100, 20),
            },
            "CR_failID_trueTau": {
                "MET_met": np.geomspace(1, 100, 20),
            },
            "CR_passID_trueTau": {
                "MET_met": np.geomspace(1, 100, 20),
            },
        },
    )
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis["wtaunu"].is_signal = True

    # set colours for samples
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in mc_samples:
        c = next(c_iter)
        analysis[ds].colour = c

    # FAKES ESTIMATE
    # ========================================================================
    def sum_mc_for_region(region):
            return reduce(
                lambda x, y: x + y, 
                [analysis.get_hist(var, mc_ds, region) for mc_ds in mc_samples]
            )

    FF_vars = [
        "TauEta",
        "TauPhi",
        "TauPt",
        "MTW",
    ]
    for var in FF_vars:
        analysis.logger.info("Calculating fake factors for %s...", var)

        # calculate FF histograms
        CR_passID_data = analysis.get_hist(var, "data", "CR_passID")
        CR_failID_data = analysis.get_hist(var, "data", "CR_failID")
        SR_passID_data = analysis.get_hist(var, "data", "SR_passID")
        SR_failID_data = analysis.get_hist(var, "data", "SR_failID")

        CR_passID_mc = sum_mc_for_region("CR_passID_trueTau")
        CR_failID_mc = sum_mc_for_region("CR_passID_trueTau")
        SR_passID_mc = sum_mc_for_region("CR_passID_trueTau")
        SR_failID_mc = sum_mc_for_region("CR_passID_trueTau")

        FF_hist = (CR_passID_data - CR_passID_mc) / (CR_failID_data - CR_failID_mc)
        analysis.histograms[f"{var}_FF"] = FF_hist.TH1

        analysis.histograms[f"{var}_FF_scaled"] = ((SR_failID_data - SR_failID_mc) * FF_hist).TH1


    # HISTORGRAMS
    # ========================================================================
    # argument dicts
        
    # No fakes scaling
    # ---------------------------------------------------------------------------- 
    default_args = {
        "datasets": mc_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi/1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "cut": True,
    }

    # mass-like variables
    for var in [
        "MET_met",
        "TauPt",
        "MTW",
    ]:
        analysis.stack_plot(var=var, **default_args, logx=True, data=True)
        analysis.stack_plot(var=var, **default_args, logy=False, data=True, suffix="liny")

    # unitless variables
    for var in [
        "TauEta",
        "TauPhi",
        "TauPt_div_MET",
        "DeltaPhi_tau_met",
        "TauRNNJetScore",
        "TauBDTEleScore",
    ]:
        analysis.stack_plot(var=var, **default_args, data=True)
        analysis.stack_plot(var=var, **default_args, logy=False, data=True, suffix="liny")

    # Fake factors
    # ----------------------------------------------------------------------------
    default_args = {
        "yerr": False,
        "cut": False,
        "logy": False
    }
    analysis.plot_hist(var="TauPt_FF", logx=True, xlabel=r"$p_T^\tau$ Fake Factors", **default_args)
    analysis.plot_hist(var="MTW_FF", logx=True, xlabel=r"$M_T^W$ Fake Factors", **default_args)
    analysis.plot_hist(var="TauEta_FF", logx=False, xlabel=r"$\eta^\tau$ Fake Factors", **default_args)
    analysis.plot_hist(var="TauPhi_FF", logx=False, xlabel=r"$\phi^\tau$ Fake Factors", **default_args)

    # Fake scaled stacks
    # ----------------------------------------------------------------------------
    # log axes
    default_args = {
        "datasets": all_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi/1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "logy": True,
        "cut": "SR_passID",
        "suffix": "fake_scaled_log"
    }
    def FF_vars(s: str) -> list[str]:
        """List of variable names for each sample"""
        return [f"{s}_FF_scaled"] + [s] * len(mc_samples)

    analysis.stack_plot(var=FF_vars("TauPt"), logx=True, **default_args, xlabel=r"$p_T^\tau$ [GeV]")
    analysis.stack_plot(var=FF_vars("MTW"), logx=True, **default_args, xlabel=r"$M_T^W$ [GeV]", )
    analysis.stack_plot(var=FF_vars("TauEta"), **default_args, xlabel=r"$\eta^\tau$")
    analysis.stack_plot(var=FF_vars("TauPhi"), **default_args, xlabel=r"$\phi^\tau$")

    # linear axes
    default_args["logy"] = False
    default_args["logx"] = False
    default_args["suffix"] = "fake_scaled_linear"
    analysis.stack_plot(var=FF_vars("TauPt"), **default_args, xlabel=r"$p_T^\tau$ [GeV]")
    analysis.stack_plot(var=FF_vars("MTW"), **default_args, xlabel=r"$M_T^W$ [GeV]")
    analysis.stack_plot(var=FF_vars("TauEta"), **default_args, xlabel=r"$\eta^\tau$")
    analysis.stack_plot(var=FF_vars("TauPhi"), **default_args, xlabel=r"$\phi^\tau$")

    # Direct data scaling comparison
    # ----------------------------------------------------------------------------
    # log axes
    default_args = {
        "title": f"data17 | mc16d | {analysis.global_lumi/1000:.3g}" + r"fb$^{-1}$",
        "labels": ["SR Fake Scaling", "SR No Scaling"],
        "yerr": True,
        "logy": True,
        "cut": "SR_passID",
        "suffix": "fake_scaled_log"
    }
    analysis.plot_hist(
        var=["TauPt_FF_scaled", "data_TauPt_SR_passID_cut"], 
        logx=True,
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
    )
    analysis.plot_hist(
        var=["MTW_FF_scaled", "data_MTW_SR_passID_cut"],
        logx=True,
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
    )
    analysis.plot_hist(
        var=["TauEta_FF_scaled", "data_TauEta_SR_passID_cut"], 
        **default_args,
        xlabel=r"$\eta^\tau$",
    )
    analysis.plot_hist(
        var=["TauPhi_FF_scaled", "data_TauPhi_SR_passID_cut"],
        **default_args,
        xlabel=r"$\phi_T^\tau$",
    )

    # linear axes
    default_args["logy"] = False
    default_args["logx"] = False
    default_args["suffix"] = "fake_scaled_linear"
    analysis.plot_hist(
        var=["TauPt_FF_scaled", "data_TauPt_SR_passID_cut"], 
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
    )
    analysis.plot_hist(
        var=["MTW_FF_scaled", "data_MTW_SR_passID_cut"],
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
    )
    analysis.plot_hist(
        var=["TauEta_FF_scaled", "data_TauEta_SR_passID_cut"], 
        **default_args,
        xlabel=r"$\eta^\tau$",
    )
    analysis.plot_hist(
        var=["TauPhi_FF_scaled", "data_TauPhi_SR_passID_cut"],
        **default_args,
        xlabel=r"$\phi_T^\tau$",
    )

    analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
