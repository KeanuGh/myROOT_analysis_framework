from functools import reduce
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutfile import Cut
from src.dataset import ProfileOpts
from src.histogram import Histogram1D

DTA_PATH = Path("/data/DTA_outputs/2024-02-22/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # DATA
        # ====================================================================
        "data": {
            # "data_path": DTA_PATH / "*data17*/*.root",
            "data_path": Path("/data/DTA_outputs/2024-03-05/*data17*/*.root"),
            "label": "data",
            "is_data": True,
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
        },
        # BACKGROUNDS
        # ====================================================================
        # W -> light lepton
        "wlv": {
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
    }

    # CUTS & SELECTIONS
    # ========================================================================
    pass_presel = Cut(
        r"Pass preselection",
        r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1)"
        r"&& (MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton <= 1)",
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
    pass_met150 = Cut(
        r"$E_T^{\mathrm{miss}} > 150$",
        r"MET_met > 150",
    )
    pass_100met = Cut(
        r"$E_T^{\mathrm{miss}} < 100$",
        r"MET_met < 100",
    )
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    pass_truetau = Cut(
        r"True Tau",
        "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
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
    measurement_vars = [
        "TauEta",
        "TauPhi",
        "TauPt",
        "MTW",
        "nJets",
    ]
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
            "noID": {
                "MTW": np.geomspace(1, 1000, 51),
                "TauPt": np.geomspace(1, 1000, 51),
                "MET_met": np.geomspace(1, 1000, 51),
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
    all_samples = analysis.mc_samples + [analysis.data_sample]
    mc_samples = analysis.mc_samples
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)

    # set colours for samples
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in mc_samples:
        c = next(c_iter)
        analysis[ds].colour = c
    analysis["data"].colour = "k"

    # FAKES ESTIMATE
    # ========================================================================
    for fakes_source in ["TauPt", "MTW"]:
        analysis.do_fakes_estimate(fakes_source, measurement_vars)

        # Fake factors
        # ----------------------------------------------------------------------------
        default_args = {
            "yerr": False,
            "cut": False,
            "logy": False,
            "ylabel": "Fake factor",
        }
        analysis.plot(
            var="TauPt_FF",
            logx=True,
            xlabel=r"$p_T^\tau$ [GeV]",
            **default_args,
            filename="TauPt_FF.png",
        )

        # Stacks with Fakes background
        # ----------------------------------------------------------------------------
        # log axes
        default_args = {
            "datasets": all_samples + [None],
            "labels": [analysis[ds].label for ds in all_samples] + ["Multijet"],
            "colours": [analysis[ds].colour for ds in all_samples] + [next(c_iter)],
            "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
            "yerr": True,
            "logy": True,
            "suffix": "fake_scaled_log",
            "ratio_plot": True,
            "kind": "stack",
        }

        def FF_vars(s: str) -> list[str]:
            """List of variable names for each sample"""
            return [f"{ds_}_{s}_SR_passID_cut" for ds_ in all_samples] + [
                f"{s}_fakes_bkg_{fakes_source}_src"
            ]

        analysis.plot(
            var=FF_vars("TauPt"),
            logx=True,
            **default_args,
            xlabel=r"$p_T^\tau$ [GeV]",
            filename=f"TauPt_fakes_stack_{fakes_source}.png",
        )
        analysis.plot(
            var=FF_vars("MTW"),
            logx=True,
            **default_args,
            xlabel=r"$M_T^W$ [GeV]",
            filename=f"MTW_fakes_stack_{fakes_source}.png",
        )
        analysis.plot(
            var=FF_vars("TauEta"),
            **default_args,
            xlabel=r"$\eta^\tau$",
            filename=f"TauEta_fakes_stack_{fakes_source}.png",
        )
        analysis.plot(
            var=FF_vars("TauPhi"),
            **default_args,
            xlabel=r"$\phi^\tau$",
            filename=f"TauPhi_fakes_stack_{fakes_source}d.png",
        )

        # linear axes
        default_args["logy"] = False
        default_args["logx"] = False
        default_args["suffix"] = "fake_scaled_linear"
        analysis.plot(
            var=FF_vars("TauPt"),
            **default_args,
            xlabel=r"$p_T^\tau$ [GeV]",
            filename=f"TauPt_fakes_stack_{fakes_source}_liny.png",
        )
        analysis.plot(
            var=FF_vars("MTW"),
            **default_args,
            xlabel=r"$M_T^W$ [GeV]",
            filename=f"MTW_fakes_stack_{fakes_source}_liny.png",
        )
        analysis.plot(
            var=FF_vars("TauEta"),
            **default_args,
            xlabel=r"$\eta^\tau$",
            filename=f"TauEta_fakes_stack_{fakes_source}_liny.png",
        )
        analysis.plot(
            var=FF_vars("TauPhi"),
            **default_args,
            xlabel=r"$\phi^\tau$",
            filename=f"TauPhi_fakes_stack_{fakes_source}_liny.png",
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
            "ratio_plot": True,
        }

        def FF_full_bkg(s: str) -> Histogram1D:
            """Sum of all backgrounds + signal + FF"""
            return reduce(
                (lambda x, y: x + y), [analysis.get_hist(s, ds_, "SR_passID") for ds_ in mc_samples]
            ) + analysis.get_hist(f"{s}_fakes_bkg_{fakes_source}_src")

        analysis.plot(
            var=[FF_full_bkg("TauPt"), "data_TauPt_SR_passID_cut"],
            **default_args,
            xlabel=r"$p_T^\tau$ [GeV]",
            filename=f"FF_compare_TauPt_{fakes_source}_src.png",
        )
        analysis.plot(
            var=[FF_full_bkg("MTW"), "data_MTW_SR_passID_cut"],
            **default_args,
            xlabel=r"$M_T^W$ [GeV]",
            filename=f"FF_compare_MTW_{fakes_source}_src.png",
        )
        analysis.plot(
            var=[FF_full_bkg("TauEta"), "data_TauEta_SR_passID_cut"],
            **default_args,
            xlabel=r"$\eta^\tau$",
            filename=f"FF_compare_TauEta_{fakes_source}_src.png",
        )
        analysis.plot(
            var=[FF_full_bkg("TauPhi"), "data_TauPhi_SR_passID_cut"],
            **default_args,
            xlabel=r"$\phi_T^\tau$",
            filename=f"FF_compare_Tauphi_{fakes_source}_src.png",
        )

        # linear axes
        default_args["logy"] = False
        default_args["logx"] = False
        default_args["suffix"] = "fake_scaled_linear"
        analysis.plot(
            var=[FF_full_bkg("TauPt"), "data_TauPt_SR_passID_cut"],
            **default_args,
            xlabel=r"$p_T^\tau$ [GeV]",
            filename=f"FF_compare_TauPt_{fakes_source}_src_liny.png",
        )
        analysis.plot(
            var=[FF_full_bkg("MTW"), "data_MTW_SR_passID_cut"],
            **default_args,
            xlabel=r"$M_T^W$ [GeV]",
            filename=f"FF_compare_MTW_{fakes_source}_src_liny.png",
        )
        analysis.plot(
            var=[FF_full_bkg("TauEta"), "data_TauEta_SR_passID_cut"],
            **default_args,
            xlabel=r"$\eta^\tau$",
            filename=f"FF_compare_TauEta_{fakes_source}_src_liny.png",
        )
        analysis.plot(
            var=[FF_full_bkg("TauPhi"), "data_TauPhi_SR_passID_cut"],
            **default_args,
            xlabel=r"$\phi_T^\tau$",
            filename=f"FF_compare_TauPhi_{fakes_source}_src_liny.png",
        )

    # No Fakes
    # ----------------------------------------------------------------------------
    default_args = {
        "datasets": all_samples,
        "title": f"data17 | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "cut": [
            "SR_passID",
            "SR_failID",
            "CR_passID",
            "CR_failID",
        ],
        "ratio_plot": True,
        # "ratio_axlim": (0, 2),
        "kind": "stack",
    }

    # mass-like variables
    for var in [
        "MET_met",
        "TauPt",
        "MTW",
    ]:
        analysis.plot(var=var, **default_args, logx=True)
        analysis.plot(var=var, **default_args, logy=False, suffix="liny")

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
        analysis.plot(var=var, **default_args)
        analysis.plot(var=var, **default_args, logy=False, suffix="liny")

    # truth hists for mental health
    default_args = {
        "datasets": mc_samples,
        "title": f"Truth Taus | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "yerr": True,
        "cut": False,
        "ratio_plot": False,
        "stats_box": False,
    }
    analysis.plot(var="MatchedTruthParticlePt", **default_args, logx=True)
    analysis.plot(var="TruthTauPt", **default_args, logx=True)
    analysis.plot(var="TruthTauEta", **default_args)
    analysis.plot(var="TruthTauPhi", **default_args)

    default_args["cut"] = "SR_passID_trueTau"
    analysis.plot(var="MatchedTruthParticlePt", **default_args, logx=True)
    analysis.plot(var="TruthTauPt", **default_args, logx=True)
    analysis.plot(var="TruthTauEta", **default_args)
    analysis.plot(var="TruthTauPhi", **default_args)

    # tau pt resolution
    analysis.plot(
        var="TauPt_res",
        datasets="wtaunu",
        xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        filename="wtaunu_taupt_resolution.png",
        logy=False,
    )
    analysis.plot(
        var="TauPt_diff",
        datasets="wtaunu",
        xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        filename="wtaunu_taupt_truthrecodiff.png",
        logy=False,
    )
    analysis.plot(
        var="MTW_TauPt_res_SR_passID_cut_PROFILE",
        datasets="wtaunu",
        ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        xlabel=r"$m_W^T$ [GeV]",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        y_axlim=(-1, 1),
        filename="wtaunu_mtw_taupt_profile.png",
        logy=False,
    )
    analysis.plot(
        var=[
            "TauPt_res_SR_passID_trueTau_cut",
            "TauPt_res_SR_failID_trueTau_cut",
            "TauPt_res_CR_passID_trueTau_cut",
            "TauPt_res_CR_failID_trueTau_cut",
        ],
        datasets="wtaunu",
        labels=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        ratio_plot=False,
        filename="wtaunu_taupt_resolution_selections.png",
        logy=False,
    )
    analysis.plot(
        var=[
            "TauPt_diff_SR_passID_trueTau_cut",
            "TauPt_diff_SR_failID_trueTau_cut",
            "TauPt_diff_CR_passID_trueTau_cut",
            "TauPt_diff_CR_failID_trueTau_cut",
        ],
        datasets="wtaunu",
        labels=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        ratio_plot=False,
        filename="wtaunu_taupt_truthrecodiff_selections.png",
        logy=False,
    )
    analysis.plot(
        var=[
            "MTW_TauPt_res_SR_passID_trueTau_cut_PROFILE",
            "MTW_TauPt_res_SR_failID_trueTau_cut_PROFILE",
            "MTW_TauPt_res_CR_passID_trueTau_cut_PROFILE",
            "MTW_TauPt_res_CR_failID_trueTau_cut_PROFILE",
        ],
        datasets="wtaunu",
        labels=["SR_passID", "SR_failID", "CR_passID", "CR_failID"],
        ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
        xlabel=r"$m_W^T$ [GeV]",
        title=r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
        + f"{analysis.global_lumi / 1000:.3g}"
        + r"fb$^{-1}$",
        y_axlim=(-1, 1),
        ratio_plot=False,
        filename="wtaunu_mtw_taupt_profile_selections.png",
        logy=False,
    )

    # analysis.histogram_printout()
    analysis.save_histograms()
    analysis.logger.info("DONE.")
