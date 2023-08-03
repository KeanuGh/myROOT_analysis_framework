from typing import Dict

import matplotlib.pyplot as plt  # type: ignore

from histogram import Histogram1D  # type: ignore
from src.analysis import Analysis

DTA_PATH = "/data/DTA_outputs/2023-06-29/"
DTA_PATH_H = "/data/DTA_outputs/2023-07-17/user.kghorban.Sh_2211_Wtaunu_H*/*.root"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tmv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_mu.txt",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tev_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_e.txt",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH_H,
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
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
        # hard_cut=truth_cuts,
    )
    my_analysis.cutflow_printout(latex=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # Truth
    # -----------------------------------
    for dataset_name, dataset in datasets.items():
        default_args = dict(
            datasets=dataset_name, title=dataset["label"], stats_box=True, weight="truth_weight"
        )

        # PT
        my_analysis.plot_hist(**default_args, var="TruthElePt", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="TruthMuonPt", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="TruthTauPt", logx=True, logy=True)

        # eta
        my_analysis.plot_hist(**default_args, var="TruthMuonEta")
        my_analysis.plot_hist(**default_args, var="TruthEleEta")
        my_analysis.plot_hist(**default_args, var="TruthTauEta")

        # phi
        my_analysis.plot_hist(**default_args, var="TruthMuonPhi")
        my_analysis.plot_hist(**default_args, var="TruthElePhi")
        my_analysis.plot_hist(**default_args, var="TruthTauPhi")

        # MLL
        my_analysis.plot_hist(**default_args, var="TruthBosonM", logx=True, logy=True)

        # MET
        my_analysis.plot_hist(**default_args, var="TruthMetPhi", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="TruthMetPhi")
        my_analysis.plot_hist(**default_args, var="TruthMetEta")

    # RECO
    # -----------------------------------
    for dataset_name, dataset in datasets.items():
        default_args = dict(
            datasets=dataset_name, title=dataset["label"], stats_box=True, weight="reco_weight"
        )

        # PT
        my_analysis.plot_hist(**default_args, var="ElePt", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="MuonPt", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="TauPt", logx=True, logy=True)

        # eta
        my_analysis.plot_hist(**default_args, var="MuonEta")
        my_analysis.plot_hist(**default_args, var="EleEta")
        my_analysis.plot_hist(**default_args, var="TauEta")

        # phi
        my_analysis.plot_hist(**default_args, var="MuonPhi")
        my_analysis.plot_hist(**default_args, var="ElePhi")
        my_analysis.plot_hist(**default_args, var="TauPhi")

        # MTW
        my_analysis.plot_hist(**default_args, var="MTW", logx=True, logy=True)

        # MET
        my_analysis.plot_hist(**default_args, var="MET_met", logx=True, logy=True)
        my_analysis.plot_hist(**default_args, var="MET_phi")

    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")
