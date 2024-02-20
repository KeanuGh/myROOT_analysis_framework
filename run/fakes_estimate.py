from functools import reduce
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutfile import Cut

DTA_PATH = Path("/data/DTA_outputs/2024-02-05/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # DATA
        # ====================================================================
        "data": {
            "data_path": DTA_PATH / "*data17*/*.root",
            "label": "data",
            "is_data": True,
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

    # CUTS & SELECTIONS
    # ========================================================================
    pass_presel = Cut(
        r"Pass preselection",
        r"(passReco == 1) & (TauBaselineWP == 1)",
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
        # r"TauLooseWP & (TauRNNJetScore > 0.25)",
        r"(TauBDTEleScore > 0.05) && "
        r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
    )
    fail_loose = Cut(
        r"\mathrm{Fail Loose ID}",
        # r"(TauLooseWP == 0) & (0.01 < TauRNNJetScore < 0.15)",
        r"(TauBDTEleScore > 0.05) && "
        "!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
    )
    pass_met150 = Cut(
        r"$E_T^{\mathrm{miss}} > 150$",
        r"MET_met > 150",
    )
    pass_100met = Cut(
        r"$E_T^{\mathrm{miss}} < 100$",
        r"MET_met < 100",
    )
    pass_truetau = Cut(
        r"True Tau",
        r"TruthTau_isHadronic == 1",
        # r"!isnan(TruthTauPt) "
        # r"& (TruthTauPt > 170) "
        # r"& ((isnan(TruthTauEta) || (abs(TruthTauEta) < 1.37 || 1.52 < abs(TruthTauEta) < 2.47)))",
    )

    # selections
    selections: dict[str, list[Cut]] = {
        "SR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
        ],
        "SR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
        ],
        "CR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
        ],
        "CR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
        ],
        # for MC
        "SR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
            pass_truetau,
        ],
        "SR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
            pass_truetau,
        ],
        "CR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
            pass_truetau,
        ],
        "CR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
            pass_truetau,
        ],
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
        "TruthTauPt",
        "TruthTauEta",
        "TruthTauPhi",
        "TauNCoreTracks",
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

    # RUN
    # ========================================================================
    analysis = Analysis(
        datasets,
        year=2017,
        # regen_histograms=True,
        # regen_metadata=True,
        ttree="T_s1thv_NOMINAL",
        cuts=selections,
        analysis_label="fakes_estimate",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=wanted_branches,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 21),
                "TauPt": np.geomspace(170, 1000, 21),
                "TauEta": np.linspace(-2.47, 2.47, 21),
                "EleEta": np.linspace(-2.47, 2.47, 21),
                "MuonEta": np.linspace(-2.5, 2.5, 21),
                "MET_met": np.geomspace(150, 1000, 21),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 21),
                "TauPt_div_MET": np.linspace(0, 3, 61),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TruthTauPt": np.geomspace(1, 1000, 21),
                "TauNCoreTracks": np.linspace(0, 4, 5),
            },
            "CR_failID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_failID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
        },
    )
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    analysis["wtaunu"].is_signal = True

    # set colours for samples
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in mc_samples:
        c = next(c_iter)
        analysis[ds].colour = c
    analysis["data"].colour = "k"

    # FAKES ESTIMATE
    # ========================================================================
    def sum_mc_for_region(var_, region):
        """Return sum of MC histograms for variable in region"""
        return reduce(
            lambda x, y: x + y, [analysis.get_hist(var_, mc_ds, region) for mc_ds in mc_samples]
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

        CR_passID_mc = sum_mc_for_region(var, "CR_passID_trueTau")
        CR_failID_mc = sum_mc_for_region(var, "CR_failID_trueTau")
        SR_passID_mc = sum_mc_for_region(var, "SR_passID_trueTau")
        SR_failID_mc = sum_mc_for_region(var, "SR_failID_trueTau")
        analysis.histograms[f"CR_passID_mc_{var}"] = CR_passID_mc.TH1
        analysis.histograms[f"CR_failID_mc_{var}"] = CR_failID_mc.TH1
        analysis.histograms[f"SR_passID_mc_{var}"] = SR_passID_mc.TH1
        analysis.histograms[f"SR_failID_mc_{var}"] = SR_failID_mc.TH1

        FF_hist = (CR_passID_data - CR_passID_mc) / (CR_failID_data - CR_failID_mc)
        analysis.histograms[f"{var}_FF"] = FF_hist.TH1
        analysis.histograms[f"{var}_FF_scaled"] = ((SR_failID_data - SR_failID_mc) * FF_hist).TH1

        # plot these histograms, for mental health
        analysis.plot_hist(
            [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
            labels=["CR_passID_data", "CR_failID_data", "CR_passID_mc", "CR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_histograms_{var}.png",
        )
        analysis.plot_hist(
            [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
            labels=["CR_failID_data - CR_failID_mc", "CR_passID_data - CR_passID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=True,
            filename=f"FF_histograms_diff_{var}.png",
            ratio_label="Fake Factor",
        )
        analysis.plot_hist(
            [SR_failID_data, SR_failID_mc],
            labels=["SR_failID_data", "SR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_calculation_{var}.png",
        )
        analysis.plot_hist(
            SR_failID_data - SR_failID_mc,
            labels=["SR_failID_data - SR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_calculation_delta_SR_fail_{var}.png",
        )

    # HISTORGRAMS
    # ========================================================================
    # truth taus for mental health
    default_args = {
        "datasets": all_samples,
        "title": f"TRUTH | data17(?) | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "cut": False,
        "ratio_plot": False,
        "stats_box": False,
    }
    analysis.plot_hist(var="TruthTauPt", **default_args, logx=True)
    analysis.plot_hist(var="TruthTauEta", **default_args)
    analysis.plot_hist(var="TruthTauPhi", **default_args)

    default_args["cut"] = "SR_failID_trueTau"
    analysis.plot_hist(var="TruthTauPt", **default_args, logx=True)
    analysis.plot_hist(var="TruthTauEta", **default_args)
    analysis.plot_hist(var="TruthTauPhi", **default_args)

    # No fakes scaling
    # ----------------------------------------------------------------------------
    default_args = {
        "datasets": all_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "cut": True,
        "ratio_plot": True,
        # "ratio_axlim": (0, 2),
    }

    # mass-like variables
    for var in [
        "MET_met",
        "TauPt",
        "MTW",
    ]:
        analysis.stack_plot(var=var, **default_args, logx=True)
        analysis.stack_plot(var=var, **default_args, logy=False, suffix="liny")

    # unitless variables
    for var in [
        "TauEta",
        "TauPhi",
        # "TauPt_div_MET",
        # "DeltaPhi_tau_met",
        "TauRNNJetScore",
        "TauBDTEleScore",
        "TauNCoreTracks",
    ]:
        analysis.stack_plot(var=var, **default_args)
        analysis.stack_plot(var=var, **default_args, logy=False, suffix="liny")

    # Fake factors
    # ----------------------------------------------------------------------------
    default_args = {"yerr": False, "cut": False, "logy": False, "ylabel": "Fake factor"}
    analysis.plot_hist(
        var="TauPt_FF",
        logx=True,
        xlabel=r"$p_T^\tau$ [GeV]",
        **default_args,
    )
    analysis.plot_hist(
        var="MTW_FF",
        logx=True,
        xlabel=r"$M_T^W$ [GeV]",
        **default_args,
    )
    analysis.plot_hist(
        var="TauEta_FF",
        logx=False,
        xlabel=r"$\eta^\tau$",
        **default_args,
    )
    analysis.plot_hist(
        var="TauPhi_FF",
        logx=False,
        xlabel=r"$\phi^\tau$",
        **default_args,
    )

    # Fake scaled stacks
    # ----------------------------------------------------------------------------
    # log axes
    default_args = {
        "datasets": all_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "logy": True,
        "suffix": "fake_scaled_log",
        "ratio_plot": True,
    }

    def FF_vars(s: str) -> list[str]:
        """List of variable names for each sample"""
        return [f"{s}_FF_scaled"] + [f"{s}_SR_passID_cut"] * len(mc_samples)

    analysis.stack_plot(
        var=FF_vars("TauPt"),
        logx=True,
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="TauPt_FF_scaled.png",
    )
    analysis.stack_plot(
        var=FF_vars("MTW"),
        logx=True,
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="MTW_FF_scaled.png",
    )
    analysis.stack_plot(
        var=FF_vars("TauEta"),
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="TauEta_FF_scaled.png",
    )
    analysis.stack_plot(
        var=FF_vars("TauPhi"),
        **default_args,
        xlabel=r"$\phi^\tau$",
        filename="TauPhi_FF_scaled.png",
    )

    # linear axes
    default_args["logy"] = False
    default_args["logx"] = False
    default_args["suffix"] = "fake_scaled_linear"
    analysis.stack_plot(
        var=FF_vars("TauPt"),
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="TauPt_FF_scaled_liny.png",
    )
    analysis.stack_plot(
        var=FF_vars("MTW"),
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="MTW_FF_scaled_liny.png",
    )
    analysis.stack_plot(
        var=FF_vars("TauEta"),
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="TauEta_FF_scaled_liny.png",
    )
    analysis.stack_plot(
        var=FF_vars("TauPhi"),
        **default_args,
        xlabel=r"$\phi^\tau$",
        filename="TauPhi_FF_scaled_liny.png",
    )

    # Direct data scaling comparison
    # ----------------------------------------------------------------------------
    # log axes
    default_args = {
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "labels": ["SR Fake Scaling", "SR No Scaling"],
        "yerr": True,
        "logy": True,
        "cut": "SR_passID",
        "suffix": "fake_scaled_log",
    }
    analysis.plot_hist(
        var=["TauPt_FF_scaled", "data_TauPt_SR_passID_cut"],
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="FF_compare_TauPt.png",
    )
    analysis.plot_hist(
        var=["MTW_FF_scaled", "data_MTW_SR_passID_cut"],
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="FF_compare_MTW.png",
    )
    analysis.plot_hist(
        var=["TauEta_FF_scaled", "data_TauEta_SR_passID_cut"],
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="FF_compare_TauEta.png",
    )
    analysis.plot_hist(
        var=["TauPhi_FF_scaled", "data_TauPhi_SR_passID_cut"],
        **default_args,
        xlabel=r"$\phi_T^\tau$",
        filename="FF_compare_Tauphi.png",
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
