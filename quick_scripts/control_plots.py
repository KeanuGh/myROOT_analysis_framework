from typing import Dict

import matplotlib.pyplot as plt  # type: ignore

from histogram import Histogram1D  # type: ignore
from src.analysis import Analysis

DTA_PATH = "/data/DTA_outputs/2023-10-19/"
# DTA_PATH = "/data/DTA_outputs/2023-06-29/"
DTA_PATH_H = DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root"
DTA_PATH_HM = DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # inclusive
        # "wtaunu_inc_mu_dta": {
        #     "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
        #     "TTree_name": "T_s1tmv_NOMINAL",
        #     "cutfile": "../options/DTA_cuts/dta_full_mu.txt",
        #     # "hard_cut": "BosonM < 120",
        #     # "regen_histograms": False,
        #     "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        # },
        # "wtaunu_inc_e_dta": {
        #     "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
        #     "TTree_name": "T_s1tev_NOMINAL",
        #     "cutfile": "../options/DTA_cuts/dta_full_e.txt",
        #     # "hard_cut": "BosonM < 120",
        #     # "regen_histograms": False,
        #     "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        # },
        "wtaunu_inc_h_dta": {
            "data_path": DTA_PATH_H,
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            "hard_cut": "TruthBosonM < 100",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
            "merge_into": "wtaunu_h_dta",
        },
        # # high-mass
        # "wtaunu_hm_e_dta": {
        #     "data_path": DTA_PATH_HM,
        #     "TTree_name": "T_s1tev_NOMINAL",
        #     "cutfile": "../options/DTA_cuts/dta_full_e.txt",
        #     # "regen_histograms": False,
        #     "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        # },
        # # high-mass
        # "wtaunu_hm_m_dta": {
        #     "data_path": DTA_PATH_HM,
        #     "TTree_name": "T_s1tmv_NOMINAL",
        #     "cutfile": "../options/DTA_cuts/dta_full_e.txt",
        #     # "regen_histograms": False,
        #     "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        # },
        # high-mass
        "wtaunu_hm_h_dta": {
            "data_path": DTA_PATH_HM,
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
            "merge_into": "wtaunu_h_dta",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        regen_histograms=True,
        analysis_label="control_plots",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
    )
    my_analysis.cutflow_printout(latex=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # Truth
    # -----------------------------------
    datasets["wtaunu_h_dta"] = {"label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$"}
    for dataset_name, dataset in datasets.items():
        default_args = dict(datasets=dataset_name, title=dataset["label"], stats_box=True)

        # out and in fiducial region
        for cut in (False, True):
            default_args["cut"] = cut

            # PT
            my_analysis.plot_hist(**default_args, var="TruthElePt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TruthMuonPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TruthTauPt", logx=True, logy=True)

            # eta
            my_analysis.plot_hist(**default_args, var="TruthMuonEta", logy=False)
            my_analysis.plot_hist(**default_args, var="TruthEleEta", logy=False)
            my_analysis.plot_hist(**default_args, var="TruthTauEta", logy=False)

            # phi
            my_analysis.plot_hist(**default_args, var="TruthMuonPhi", logy=False)
            my_analysis.plot_hist(**default_args, var="TruthElePhi", logy=False)
            my_analysis.plot_hist(**default_args, var="TruthTauPhi", logy=False)

            # MLL
            my_analysis.plot_hist(**default_args, var="TruthBosonM", logx=True, logy=True)

            # MET
            my_analysis.plot_hist(**default_args, var="TruthNeutrinoPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TruthNeutrinoEta", logy=False)
            my_analysis.plot_hist(**default_args, var="TruthNeutrinoPhi", logy=False)

            # tau-vis
            my_analysis.plot_hist(**default_args, var="VisTruthTauPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="VisTruthTauEta", logy=False)
            my_analysis.plot_hist(**default_args, var="VisTruthTauPhi", logy=False)

            # Overlays
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPt", "TruthElePt"],
                labels=[r"$p^\tau_T$", r"$p^e_T$"],
                logx=True,
                logy=True
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauEta", "TruthEleEta"],
                labels=[r"$\eta_\tau$", r"$\eta_e$"],
                logy=False
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPhi", "TruthElePhi"],
                labels=[r"$\phi_\tau$", r"$\phi_e$"],
                logy=False
            )

            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPt", "TruthMuonPt"],
                labels=[r"$p^\tau_T$", r"$p^\mu_T$"],
                logx=True,
                logy=True
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauEta", "TruthMuonEta"],
                labels=[r"$\eta_\tau$", r"$\eta_\mu$"],
                logy=False
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPhi", "TruthMuonPhi"],
                labels=[r"$\phi_\tau$", r"$\phi_\mu$"],
                logy=False
            )

            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPt", "VisTruthTauPt"],
                labels=[r"$p^\tau_T$", r"$p^{\tau\mathrm{-vis}}_T$"],
                logx=True,
                logy=True
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauEta", "VisTruthTauEta"],
                labels=[r"$\eta_\tau$", r"$\eta_{\tau\mathrm{-vis}}$"],
                logy=False
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TruthTauPhi", "VisTruthTauPhi"],
                labels=[r"$\phi_\tau$", r"$\phi_{\tau\mathrm{-vis}}$"],
                logy=False
            )

            # RECO
            # -----------------------------------
            # PT
            my_analysis.plot_hist(**default_args, var="ElePt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="MuonPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="TauPt", logx=True, logy=True)

            # eta
            my_analysis.plot_hist(**default_args, var="MuonEta", logy=False)
            my_analysis.plot_hist(**default_args, var="EleEta", logy=False)
            my_analysis.plot_hist(**default_args, var="TauEta", logy=False)

            # phi
            my_analysis.plot_hist(**default_args, var="MuonPhi", logy=False)
            my_analysis.plot_hist(**default_args, var="ElePhi", logy=False)
            my_analysis.plot_hist(**default_args, var="TauPhi", logy=False)

            # MTW
            my_analysis.plot_hist(**default_args, var="MTW", logx=True, logy=True)

            # MET
            my_analysis.plot_hist(**default_args, var="MET_met", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="MET_phi", logy=False)

            # JET
            my_analysis.plot_hist(**default_args, var="JetPt", logx=True, logy=True)
            my_analysis.plot_hist(**default_args, var="JetEta", logy=False)
            my_analysis.plot_hist(**default_args, var="JetPhi", logy=False)

            # Overlays
            my_analysis.plot_hist(
                **default_args,
                var=["TauPt", "ElePt"],
                labels=[r"$p^\tau_T$", r"$p^e_T$"],
                logx=True,
                logy=True
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TauEta", "EleEta"],
                labels=[r"$\eta_\tau$", r"$\eta_e$"],
                logy=False
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TauPhi", "ElePhi"],
                labels=[r"$\phi_\tau$", r"$\phi_e$"],
                logy=False
            )

            my_analysis.plot_hist(
                **default_args,
                var=["TauPt", "MuonPt"],
                labels=[r"$p^\tau_T$", r"$p^\mu_T$"],
                logx=True,
                logy=True
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TauEta", "MuonEta"],
                labels=[r"$\eta_\tau$", r"$\eta_\mu$"],
                logy=False
            )
            my_analysis.plot_hist(
                **default_args,
                var=["TauPhi", "MuonPhi"],
                labels=[r"$\phi_\tau$", r"$\phi_\mu$"],
                logy=False
            )

    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")
