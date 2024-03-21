from functools import reduce
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.analysis import Analysis
from src.cutfile import Cut
from src.dataset import ProfileOptions
from src.histogram import Histogram1D, TH1_bin_edges
from utils.plotting_tools import get_axis_labels

DTA_PATH = Path("/data/DTA_outputs/2024-02-22/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
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
        # "ttbar": {
        #     "data_path": DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
        #     "label": r"$t\bar{t}$",
        # },
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
            # "import_missing_columns_as_nan": True,
        },
    }

    # CUTS & SELECTIONS
    # ========================================================================
    pass_presel = Cut(
        r"Pass preselection",
        r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1)"
        r"&& (LeadingJetPt > 10)"
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
    mc_samples = [
        "wtaunu",
        "wlnu",
        "zll",
        "top",
        "diboson",
    ]
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
    profiles: dict[str, ProfileOptions] = dict()
    for meas_var in measurement_vars:
        for prof_var in profile_vars:
            profiles[f"{meas_var}_{prof_var}"] = ProfileOptions(
                x=meas_var,
                y=prof_var,
                weight="" if "MatchedTruthParticle" in prof_var else "reco_weight",
            )
    all_samples = ["data"] + mc_samples

    # RUN
    # ========================================================================
    analysis = Analysis(
        datasets,
        year=2017,
        regen_histograms=True,
        # regen_metadata=True,
        ttree="T_s1thv_NOMINAL",
        cuts=selections,
        analysis_label="fakes_estimate",
        dataset_type="dta",
        # log_level=10,
        log_out="console",
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
    analysis.full_cutflow_printout(datasets=all_samples)
    analysis.print_metadata_table(datasets=mc_samples)
    analysis["wtaunu"].is_signal = True

    # set colours for samples
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in mc_samples:
        c = next(c_iter)
        analysis[ds].colour = c
    analysis["data"].colour = "k"

    # analysis.logger.info("Histograms: ")
    # analysis.logger.info(analysis.histograms.keys())

    # FAKES ESTIMATE
    # ========================================================================
    integrals = {}
    for var in measurement_vars:
        xlabel = get_axis_labels(var)[0]

        # Fakes distribution across kinematic variable for signal MC
        # -----------------------------------------------------------------------
        for mc in mc_samples + ["all_mc"]:
            for selection in [
                "SR_passID",
                "SR_failID",
                "CR_passID",
                "CR_failID",
            ]:
                if mc == "all_mc":
                    ele_fakes = analysis.sum_hists(
                        [
                            f"{mc}_{var}_MatchedTruthParticle_isElectron_{selection}_cut_PROFILE"
                            for mc in mc_samples
                        ],
                        f"{mc}_{var}_MatchedTruthParticle_isElectron_{selection}_cut_PROFILE",
                    )
                    muon_fakes = analysis.sum_hists(
                        [
                            f"{mc}_{var}_MatchedTruthParticle_isMuon_{selection}_cut_PROFILE"
                            for mc in mc_samples
                        ],
                        f"{mc}_{var}_MatchedTruthParticle_isMuon_{selection}_cut_PROFILE",
                    )
                    photon_fakes = analysis.sum_hists(
                        [
                            f"{mc}_{var}_MatchedTruthParticle_isPhoton_{selection}_cut_PROFILE"
                            for mc in mc_samples
                        ],
                        f"{mc}_{var}_MatchedTruthParticle_isPhoton_{selection}_cut_PROFILE",
                    )
                    jet_fakes = analysis.sum_hists(
                        [
                            f"{mc}_{var}_MatchedTruthParticle_isJet_{selection}_cut_PROFILE"
                            for mc in mc_samples
                        ],
                        f"{mc}_{var}_MatchedTruthParticle_isJet_{selection}_cut_PROFILE",
                    )
                    true_taus = analysis.sum_hists(
                        [
                            f"{mc}_{var}_MatchedTruthParticle_isTau_{selection}_cut_PROFILE"
                            for mc in mc_samples
                        ],
                        f"{mc}_{var}_MatchedTruthParticle_isTau_{selection}_cut_PROFILE",
                    )
                else:
                    ele_fakes = analysis.histograms[
                        f"{mc}_{var}_MatchedTruthParticle_isElectron_{selection}_cut_PROFILE"
                    ]
                    muon_fakes = analysis.histograms[
                        f"{mc}_{var}_MatchedTruthParticle_isMuon_{selection}_cut_PROFILE"
                    ]
                    photon_fakes = analysis.histograms[
                        f"{mc}_{var}_MatchedTruthParticle_isPhoton_{selection}_cut_PROFILE"
                    ]
                    jet_fakes = analysis.histograms[
                        f"{mc}_{var}_MatchedTruthParticle_isJet_{selection}_cut_PROFILE"
                    ]
                    true_taus = analysis.histograms[
                        f"{mc}_{var}_MatchedTruthParticle_isTau_{selection}_cut_PROFILE"
                    ]
                sel_hist = analysis.get_hist(var, "wtaunu", selection, TH1=True)
                nbins = sel_hist.GetNbinsX()
                bin_edges = TH1_bin_edges(sel_hist)

                for logy in (True, False):
                    analysis.plot(
                        [jet_fakes, photon_fakes, muon_fakes, ele_fakes, true_taus],
                        labels=[
                            "Jet Fakes",
                            "Photon Fakes",
                            "Muon Fakes",
                            "Electron Fakes",
                            "True taus",
                        ],
                        sort=False,
                        logx=False,
                        logy=logy,
                        yerr=False,
                        colours=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
                        title=f"Fake fractions for {var} in {selection}",
                        y_axlim=(0, 1),
                        kind="stack",
                        xlabel=xlabel,
                        ylabel="Fraction of fake matched taus in signal MC",
                        filename=f"{mc}_{var}_{selection}_fake_fractions{'_liny' if not logy else ''}.png",
                    )

        # calculate FF histograms
        # -----------------------------------------------------------------------
        analysis.logger.info("Calculating fake factors for %s...", var)
        CR_passID_data = analysis.get_hist(var, "data", "CR_passID", TH1=True)
        CR_failID_data = analysis.get_hist(var, "data", "CR_failID", TH1=True)
        SR_passID_data = analysis.get_hist(var, "data", "SR_passID", TH1=True)
        SR_failID_data = analysis.get_hist(var, "data", "SR_failID", TH1=True)
        CR_passID_mc = analysis.sum_hists(
            [f"{ds}_{var}_CR_passID_trueTau_cut" for ds in mc_samples]
        )
        CR_failID_mc = analysis.sum_hists(
            [f"{ds}_{var}_CR_failID_trueTau_cut" for ds in mc_samples]
        )
        SR_passID_mc = analysis.sum_hists(
            [f"{ds}_{var}_SR_passID_trueTau_cut" for ds in mc_samples]
        )
        SR_failID_mc = analysis.sum_hists(
            [f"{ds}_{var}_SR_failID_trueTau_cut" for ds in mc_samples]
        )
        analysis.histograms[f"all_mc_{var}_CR_passID_mc_cut"] = CR_passID_mc
        analysis.histograms[f"all_mc_{var}_CR_failID_mc_cut"] = CR_failID_mc
        analysis.histograms[f"all_mc_{var}_SR_passID_mc_cut"] = SR_passID_mc
        analysis.histograms[f"all_mc_{var}_SR_failID_mc_cut"] = SR_failID_mc

        # FF calculation
        FF_hist = (CR_passID_data - CR_passID_mc) / (CR_failID_data - CR_failID_mc)
        SR_data_fakes = (SR_failID_data - SR_failID_mc) * FF_hist

        analysis.histograms[f"{var}_FF"] = FF_hist
        analysis.histograms[f"{var}_fakes_bkg"] = SR_data_fakes

        integrals[var] = {
            "CR_passID_mc": CR_passID_mc.Integral(),
            "CR_failID_mc": CR_failID_mc.Integral(),
            "SR_passID_mc": SR_passID_mc.Integral(),
            "SR_failID_mc": SR_failID_mc.Integral(),
            "CR_passID_data": CR_passID_data.Integral(),
            "CR_failID_data": CR_failID_data.Integral(),
            "SR_passID_data": SR_passID_data.Integral(),
            "SR_failID_data": SR_failID_data.Integral(),
            "FF_numerator": (CR_passID_data - CR_passID_mc).Integral(),
            "FF_denominator": (CR_failID_data - CR_failID_mc).Integral(),
            "FF_hist": FF_hist.Integral(),
            "SR_failID_diff": (SR_failID_data - SR_failID_mc).Integral(),
            "SR_data_fakes": SR_data_fakes.Integral(),
        }

        CR_passID_mc = Histogram1D(th1=CR_passID_mc)
        CR_failID_mc = Histogram1D(th1=CR_failID_mc)
        SR_passID_mc = Histogram1D(th1=SR_passID_mc)
        SR_failID_mc = Histogram1D(th1=SR_failID_mc)
        CR_passID_data = Histogram1D(th1=CR_passID_data)
        CR_failID_data = Histogram1D(th1=CR_failID_data)
        SR_passID_data = Histogram1D(th1=SR_passID_data)
        SR_failID_data = Histogram1D(th1=SR_failID_data)

        # plot intermediate histograms for by-eye verification
        # -----------------------------------------------------------------------
        analysis.plot(
            [CR_passID_data, CR_failID_data, CR_passID_mc, CR_failID_mc],
            labels=["CR_passID_data", "CR_failID_data", "CR_passID_mc", "CR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_histograms_{var}.png",
        )
        analysis.plot(
            [CR_failID_data - CR_failID_mc, CR_passID_data - CR_passID_mc],
            labels=["CR_failID_data - CR_failID_mc", "CR_passID_data - CR_passID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=True,
            filename=f"FF_histograms_diff_{var}.png",
            ratio_label="Fake Factor",
        )
        analysis.plot(
            [SR_failID_data, SR_failID_mc],
            labels=["SR_failID_data", "SR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_calculation_{var}.png",
        )
        analysis.plot(
            SR_failID_data - SR_failID_mc,
            labels=["SR_failID_data - SR_failID_mc"],
            yerr=False,
            xlabel=var,
            ratio_plot=False,
            filename=f"FF_calculation_delta_SR_fail_{var}.png",
        )

    integrals = pd.DataFrame.from_dict(integrals)
    analysis.logger.info("Table of FF histogram integrals:")
    analysis.logger.info(tabulate(integrals, headers="keys", tablefmt="fancy_grid"))

    # HISTORGRAMS
    # ========================================================================
    # truth taus for mental health
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
    )
    analysis.plot(
        var="MTW_FF",
        logx=True,
        xlabel=r"$m_T^W$ [GeV]",
        **default_args,
    )
    analysis.plot(
        var="TauEta_FF",
        logx=False,
        xlabel=r"$\eta^\tau$",
        **default_args,
    )
    analysis.plot(
        var="TauPhi_FF",
        logx=False,
        xlabel=r"$\phi^\tau$",
        **default_args,
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
        return [f"{ds_}_{s}_SR_passID_cut" for ds_ in all_samples] + [f"{s}_fakes_bkg"]

    analysis.plot(
        var=FF_vars("TauPt"),
        logx=True,
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="TauPt_FF_scaled.png",
    )
    analysis.plot(
        var=FF_vars("MTW"),
        logx=True,
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="MTW_FF_scaled.png",
    )
    analysis.plot(
        var=FF_vars("TauEta"),
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="TauEta_FF_scaled.png",
    )
    analysis.plot(
        var=FF_vars("TauPhi"),
        **default_args,
        xlabel=r"$\phi^\tau$",
        filename="TauPhi_FF_scaled.png",
    )

    # linear axes
    default_args["logy"] = False
    default_args["logx"] = False
    default_args["suffix"] = "fake_scaled_linear"
    analysis.plot(
        var=FF_vars("TauPt"),
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="TauPt_FF_scaled_liny.png",
    )
    analysis.plot(
        var=FF_vars("MTW"),
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="MTW_FF_scaled_liny.png",
    )
    analysis.plot(
        var=FF_vars("TauEta"),
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="TauEta_FF_scaled_liny.png",
    )
    analysis.plot(
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

    def FF_full_bkg(s: str) -> Histogram1D:
        """Sum of all backgrounds + signal + FF"""
        return reduce(
            (lambda x, y: x + y), [analysis.get_hist(s, ds_, "SR_passID") for ds_ in mc_samples]
        ) + analysis.get_hist(f"{s}_fakes_bkg")

    analysis.plot(
        var=[FF_full_bkg("TauPt"), "data_TauPt_SR_passID_cut"],
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="FF_compare_TauPt.png",
    )
    analysis.plot(
        var=[FF_full_bkg("MTW"), "data_MTW_SR_passID_cut"],
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="FF_compare_MTW.png",
    )
    analysis.plot(
        var=[FF_full_bkg("TauEta"), "data_TauEta_SR_passID_cut"],
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="FF_compare_TauEta.png",
    )
    analysis.plot(
        var=[FF_full_bkg("TauPhi"), "data_TauPhi_SR_passID_cut"],
        **default_args,
        xlabel=r"$\phi_T^\tau$",
        filename="FF_compare_Tauphi.png",
    )

    # linear axes
    default_args["logy"] = False
    default_args["logx"] = False
    default_args["suffix"] = "fake_scaled_linear"
    analysis.plot(
        var=[FF_full_bkg("TauPt"), "data_TauPt_SR_passID_cut"],
        **default_args,
        xlabel=r"$p_T^\tau$ [GeV]",
        filename="FF_compare_TauPt_liny.png",
    )
    analysis.plot(
        var=[FF_full_bkg("MTW"), "data_MTW_SR_passID_cut"],
        **default_args,
        xlabel=r"$M_T^W$ [GeV]",
        filename="FF_compare_MTW_liny.png",
    )
    analysis.plot(
        var=[FF_full_bkg("TauEta"), "data_TauEta_SR_passID_cut"],
        **default_args,
        xlabel=r"$\eta^\tau$",
        filename="FF_compare_TauEta_liny.png",
    )
    analysis.plot(
        var=[FF_full_bkg("TauPhi"), "data_TauPhi_SR_passID_cut"],
        **default_args,
        xlabel=r"$\phi_T^\tau$",
        filename="FF_compare_TauPhi_liny.png",
    )

    # analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
