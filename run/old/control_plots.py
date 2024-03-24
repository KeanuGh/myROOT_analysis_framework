from typing import Dict

import matplotlib.pyplot as plt  # type: ignore

from histogram import Histogram1D  # type: ignore
from run import cuts
from src.analysis import Analysis
from src.cutflow import Cut

DTA_PATH = "/data/DTA_outputs/2023-10-25/"
# DTA_PATH = "/data/DTA_outputs/2023-06-29/"
DTA_PATH_H = DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root"
DTA_PATH_HM = DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root"
DTA_PATH_INC = DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_*maxHTpTV2*/*.root"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"
cuts_mu = cuts.cuts_reco_mu
cuts_e = cuts.cuts_reco_e
cuts_had = cuts.cuts_reco_had

if __name__ == "__main__":
    # some separate cuts
    def add_deltaRtaumu_cut(cut_list: list[Cut], deltaR: float) -> list[Cut]:
        for i, cut in enumerate(cut_list):
            if "TauPt" in cut.name:
                break
        deltar_cut = Cut(
            r"$\Delta R_{\tau---\mu} < " + f"{deltaR}$", r"DeltaR_tau_mu < " + str(deltaR)
        )
        return cut_list[: i - 1] + [deltar_cut] + cut_list[i - 1 :]

    # some separate cuts
    def add_deltaRtaue_cut(cut_list: list[Cut], deltaR: float) -> list[Cut]:
        for i, cut in enumerate(cut_list):
            if "TauPt" in cut.name:
                break
        deltar_cut = Cut(
            r"$\Delta R_{\tau---\e} < " + f"{deltaR}$", r"DeltaR_tau_e < " + str(deltaR)
        )
        return cut_list[: i - 1] + [deltar_cut] + cut_list[i - 1 :]

    cuts_mu = {
        "R1": add_deltaRtaumu_cut(cuts_mu, 1.0),
        "R05": add_deltaRtaumu_cut(cuts_mu, 0.5),
        "R01": add_deltaRtaumu_cut(cuts_mu, 0.1),
    }
    cuts_e = {
        "R1": add_deltaRtaue_cut(cuts_e, 1.0),
        "R05": add_deltaRtaue_cut(cuts_e, 0.5),
        "R01": add_deltaRtaue_cut(cuts_e, 0.1),
    }
    cuts_had = {
        "R1": add_deltaRtaumu_cut(cuts_had, 1.0),
        "R05": add_deltaRtaumu_cut(cuts_had, 0.5),
        "R01": add_deltaRtaumu_cut(cuts_had, 0.1),
    }

    datasets: Dict[str, Dict] = {
        # inclusive
        # "wtaunu_inc_mu": {
        #     "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
        #     "ttree": "T_s1tmv_NOMINAL",
        #     "cuts": cuts_mu,
        #     # "regen_histograms": False,
        #     "label": r"Inclusive $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        # },
        # "wtaunu_inc_e": {
        #     "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
        #     "ttree": "T_s1tev_NOMINAL",
        #     "cuts": cuts_e,
        #     # "regen_histograms": False,
        #     "label": r"Inclusive $W\rightarrow\tau\nu\rightarrow e\nu$",
        # },
        # "wtaunu_inc_h": {
        #     "data_path": DTA_PATH_H,
        #     "ttree": "T_s1thv_NOMINAL",
        #     "cuts": cuts_had,
        #     # "regen_histograms": True,
        #     "label": r"Inclusive $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
        # },
        # low-mass (inclusive with mass cut)
        "wtaunu_lm_mu": {
            "data_path": DTA_PATH_INC,
            # "ttree": "T_s1tmv_NOMINAL",
            "cuts": cuts_mu,
            "hard_cut": "TruthBosonM < 120",
            # "regen_histograms": False,
            "label": r"$m_W < 120$ $W\rightarrow\tau\nu\rightarrow\mu\nu$",
            "merge_into": "wtaunu_mu",
        },
        "wtaunu_lm_e": {
            "data_path": DTA_PATH_INC,
            # "ttree": "T_s1tev_NOMINAL",
            "cuts": cuts_e,
            "hard_cut": "TruthBosonM < 120",
            # "regen_histograms": False,
            "label": r"$m_W < 120$ $W\rightarrow\tau\nu\rightarrow e\nu$",
            "merge_into": "wtaunu_e",
        },
        "wtaunu_lm_h": {
            "data_path": DTA_PATH_INC,
            # "ttree": "T_s1thv_NOMINAL",
            "cuts": cuts_had,
            "hard_cut": "TruthBosonM < 120",
            # "regen_histograms": True,
            "label": r"$m_W < 120$ $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
            "merge_into": "wtaunu_h",
        },
        # high-mass
        "wtaunu_hm_mu": {
            "data_path": DTA_PATH_HM,
            # "ttree": "T_s1tmv_NOMINAL",
            "cuts": cuts_mu,
            # "regen_histograms": False,
            "label": r"$m_W > 120$ $W\rightarrow\tau\nu\rightarrow\mu\nu$",
            "hard_cut": "TruthBosonM > 120",
            "merge_into": "wtaunu_mu",
        },
        "wtaunu_hm_e": {
            "data_path": DTA_PATH_HM,
            # "ttree": "T_s1tev_NOMINAL",
            "cuts": cuts_e,
            # "regen_histograms": False,
            "label": r"$m_W > 120$ $W\rightarrow\tau\nu\rightarrow e\nu$",
            "hard_cut": "TruthBosonM > 120",
            "merge_into": "wtaunu_e",
        },
        "wtaunu_hm_h": {
            "data_path": DTA_PATH_HM,
            # "ttree": "T_s1thv_NOMINAL",
            "cuts": cuts_had,
            # "regen_histograms": False,
            "label": r"$m_W > 120$ $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
            "hard_cut": "TruthBosonM > 120",
            "merge_into": "wtaunu_h",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2017",
        ttree={"T_s1thv_NOMINAL", "T_s1tmv_NOMINAL", "T_s1tev_NOMINAL"},
        # regen_histograms=True,
        analysis_label="control_plots",
        lepton="tau",
        extract_vars=cuts.import_vars_reco | cuts.import_vars_truth,
        dataset_type="dta",
        log_level=10,
        log_out="both",
        # binnings=override_binnings,
    )
    my_analysis["wtaunu_h"].label = r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$"
    my_analysis["wtaunu_mu"].label = r"$W\rightarrow\tau\nu\rightarrow\mu\nu$"
    my_analysis["wtaunu_e"].label = r"$W\rightarrow\tau\nu\rightarrow e\nu$"
    my_analysis.cutflow_printout(latex=True)

    for datasets in [["wtaunu_h", "wtaunu_mu", "wtaunu_e"]]:
        for cut in (True, False):
            title = f"reco | mc16d | {my_analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$"
            if cut:
                title += " | CUT"
            default_args = {
                "datasets": datasets,
                "ratio_axlim": True,
                "ratio_fit": True,
                "cut": cut,
                "title": title,
            }

            # some manual overlays
            my_analysis.plot_hist(**default_args, var="MTW", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TauPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TauEta")
            my_analysis.plot_hist(**default_args, var="TauPhi")

            # my_analysis.plot_hist(**default_args, var="TruthBosonM", logx=True, logy=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # # Truth
    # # -----------------------------------
    # for dataset in ["wtaunu_h", "wtaunu_mu", "wtaunu_e"]:
    #     default_args = dict(
    #         datasets=dataset, title=my_analysis[dataset].label, stats_box=True, ratio_axlim=1.5
    #     )
    #
    #     my_analysis.plot_hist(
    #         **default_args, var="truth_weight", logx=False, logy=True, bins=(100, -1000, 1000)
    #     )
    #     my_analysis.plot_hist(
    #         **default_args, var="reco_weight", logx=False, logy=True, bins=(100, -1000, 1000)
    #     )
    #
    #     # out and in fiducial region
    #     for cut in (False, True):
    #         default_args["cut"] = cut
    #         # TRUTH
    #         # =================================================================
    #         # PT
    #         my_analysis.plot_hist(**default_args, var="TruthElePt", logx=True, logy=True)
    #         my_analysis.plot_hist(**default_args, var="TruthMuonPt", logx=True, logy=True)
    #         my_analysis.plot_hist(**default_args, var="TruthTauPt", logx=True, logy=True)
    #
    #         # eta
    #         my_analysis.plot_hist(**default_args, var="TruthMuonEta", logy=False)
    #         my_analysis.plot_hist(**default_args, var="TruthEleEta", logy=False)
    #         my_analysis.plot_hist(**default_args, var="TruthTauEta", logy=False)
    #
    #         # phi
    #         my_analysis.plot_hist(**default_args, var="TruthMuonPhi", logy=False)
    #         my_analysis.plot_hist(**default_args, var="TruthElePhi", logy=False)
    #         my_analysis.plot_hist(**default_args, var="TruthTauPhi", logy=False)
    #
    #         # MLL
    #         my_analysis.plot_hist(**default_args, var="TruthBosonM", logx=True, logy=True)
    #
    #         # MET
    #         my_analysis.plot_hist(**default_args, var="TruthNeutrinoPt", logx=True, logy=True)
    #         my_analysis.plot_hist(**default_args, var="TruthNeutrinoEta", logy=False)
    #         my_analysis.plot_hist(**default_args, var="TruthNeutrinoPhi", logy=False)
    #
    #         # tau-vis
    #         my_analysis.plot_hist(**default_args, var="VisTruthTauPt", logx=True, logy=True)
    #         my_analysis.plot_hist(**default_args, var="VisTruthTauEta", logy=False)
    #         my_analysis.plot_hist(**default_args, var="VisTruthTauPhi", logy=False)
    #
    #         # Overlays
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPt", "TruthElePt"],
    #             labels=[r"$p^\tau_T$", r"$p^e_T$"],
    #             logx=True,
    #             logy=True
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauEta", "TruthEleEta"],
    #             labels=[r"$\eta_\tau$", r"$\eta_e$"],
    #             logy=False
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPhi", "TruthElePhi"],
    #             labels=[r"$\phi_\tau$", r"$\phi_e$"],
    #             logy=False
    #         )
    #
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPt", "TruthMuonPt"],
    #             labels=[r"$p^\tau_T$", r"$p^\mu_T$"],
    #             logx=True,
    #             logy=True
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauEta", "TruthMuonEta"],
    #             labels=[r"$\eta_\tau$", r"$\eta_\mu$"],
    #             logy=False
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPhi", "TruthMuonPhi"],
    #             labels=[r"$\phi_\tau$", r"$\phi_\mu$"],
    #             logy=False
    #         )
    #
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPt", "VisTruthTauPt"],
    #             labels=[r"$p^\tau_T$", r"$p^{\tau\mathrm{-vis}}_T$"],
    #             logx=True,
    #             logy=True
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauEta", "VisTruthTauEta"],
    #             labels=[r"$\eta_\tau$", r"$\eta_{\tau\mathrm{-vis}}$"],
    #             logy=False
    #         )
    #         my_analysis.plot_hist(
    #             **default_args,
    #             var=["TruthTauPhi", "VisTruthTauPhi"],
    #             labels=[r"$\phi_\tau$", r"$\phi_{\tau\mathrm{-vis}}$"],
    #             logy=False
    #         )
    #
    # # RECO
    # # -----------------------------------
    # # PT
    # my_analysis.plot_hist(**default_args, var="ElePt", logx=True, logy=True)
    # my_analysis.plot_hist(**default_args, var="MuonPt", logx=True, logy=True)
    # my_analysis.plot_hist(**default_args, var="TauPt", logx=True, logy=True)
    #
    # # eta
    # my_analysis.plot_hist(**default_args, var="MuonEta", logy=False)
    # my_analysis.plot_hist(**default_args, var="EleEta", logy=False)
    # my_analysis.plot_hist(**default_args, var="TauEta", logy=False)
    #
    # # phi
    # my_analysis.plot_hist(**default_args, var="MuonPhi", logy=False)
    # my_analysis.plot_hist(**default_args, var="ElePhi", logy=False)
    # my_analysis.plot_hist(**default_args, var="TauPhi", logy=False)
    #
    # # MTW
    # my_analysis.plot_hist(**default_args, var="MTW", logx=True, logy=True)
    #
    # # MET
    # my_analysis.plot_hist(**default_args, var="MET_met", logx=True, logy=True)
    # my_analysis.plot_hist(**default_args, var="MET_phi", logy=False)
    #
    # # JET
    # my_analysis.plot_hist(**default_args, var="JetPt", logx=True, logy=True)
    # my_analysis.plot_hist(**default_args, var="JetEta", logy=False)
    # my_analysis.plot_hist(**default_args, var="JetPhi", logy=False)

    # # Overlays
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauPt", "ElePt"],
    #     labels=[r"$p^\tau_T$", r"$p^e_T$"],
    #     logx=True,
    #     logy=True
    # )
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauEta", "EleEta"],
    #     labels=[r"$\eta_\tau$", r"$\eta_e$"],
    #     logy=False
    # )
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauPhi", "ElePhi"],
    #     labels=[r"$\phi_\tau$", r"$\phi_e$"],
    #     logy=False
    # )
    #
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauPt", "MuonPt"],
    #     labels=[r"$p^\tau_T$", r"$p^\mu_T$"],
    #     logx=True,
    #     logy=True
    # )
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauEta", "MuonEta"],
    #     labels=[r"$\eta_\tau$", r"$\eta_\mu$"],
    #     logy=False
    # )
    # my_analysis.plot_hist(
    #     **default_args,
    #     var=["TauPhi", "MuonPhi"],
    #     labels=[r"$\phi_\tau$", r"$\phi_\mu$"],
    #     logy=False
    # )

    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")
