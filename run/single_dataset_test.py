from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.histogram import Histogram1D
from src.analysis import Analysis

DTA_PATH = "/eos/home-k/kghorban/DTA_OUT/2023-10-25/"
DATA_OUT_DIR = "/eos/home-k/kghorban/framework_outputs/"
ROOT_DIR = "/afs/cern.ch/user/k/kghorban/framework/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        "wmunu_hm_BFilter_mu": {
            # "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter*/*.root",
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter.e8351.MC16d.v1.2023-11-10_histograms.root/user.kghorban.35460771._000001.histograms.root",            
            "ttree": "T_s1tmv_NOMINAL",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
            "label": r"BFilter muon tree high-mass $W\rightarrow\mu\nu$",
        },
        "wmunu_hm_BFilter_e": {
            # "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter*/*.root",
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter.e8351.MC16d.v1.2023-11-10_histograms.root/user.kghorban.35460771._000001.histograms.root",            
            "ttree": "T_s1tev_NOMINAL",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
            "label": r"BFilter electron tree high-mass $W\rightarrow\mu\nu$",
        },
        "wmunu_hm_BFilter_had": {
            # "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter*/*.root",
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter.e8351.MC16d.v1.2023-11-10_histograms.root/user.kghorban.35460771._000001.histograms.root",            
            "ttree": "T_s1thv_NOMINAL",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
            "label": r"BFilter hadron tree high-mass $W\rightarrow\mu\nu$",
        },
        # "wmunu_hm_BFilter": {
        #     # "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter*/*.root",
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_BFilter.e8351.MC16d.v1.2023-11-10_histograms.root/user.kghorban.35460771._000002.histograms.root",
        #     "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"BFilter high-mass $W\rightarrow\mu\nu$",
        #     "merge_into": "wmunu_hm_full"
        # },
        # "wmunu_hm_CFilterBVeto": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_CFilterBVeto*/*.root",
        #     "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"CFilterBVeto high-mass $W\rightarrow\mu\nu$",
        #     "merge_into": "wmunu_hm_full"
        # },
        # "wmunu_hm_CVetoBVeto": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS_CVetoBVeto*/*.root",
        #     "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"CVetoBVeto high-mass $W\rightarrow\mu\nu$",
        #     "merge_into": "wmunu_hm_full"
        # },
        # "wmunu_hm_e": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS*/*.root",
        #     "ttree": {"T_s1tev_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"high-mass $W\rightarrow\mu\nu$ electron channel",
        # },
        # "wmunu_hm_mu": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS*/*.root",
        #     "ttree": {"T_s1tmv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"high-mass $W\rightarrow\mu\nu$ muon chanel",
        # },
        # "wmunu_hm_had": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS*/*.root",
        #     "ttree": {"T_s1thv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"high-mass $W\rightarrow\mu\nu$ hadron channel",
        # },

        # "wmunu_hm_full": {
        #     "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120_ECMS*/*.root",
        #     "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
        #     "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
        #     "label": r"Full high-mass $W\rightarrow\mu\nu$",
        # }
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        regen_histograms=True,
        analysis_label="single_dataset_test",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        binnings={
            "truth_weight": (100, -50, 50),
            "reco_weight": (100, -50, 50),
            "mcWeight": (100, -2e7, 2e7),
            "prwWeight": (50, 0, 2),
            "rwCorr": (50, 0, 2),
            "TruthBosonM": np.geomspace(120, 5000, 20),
        },

    )
    # my_analysis["wmunu_hm_full"].label = r"merged high-mass $W\rightarrow\mu\nu$"
    my_analysis.cutflow_printout(latex=True)

    # HISTORGRAMS
    # ========================================================================================
    # -----------------------------------
    # argument dicts
    # decay_ds = ["wmunu_hm_mu", "wmunu_hm_e", "wmunu_hm_had"]
    flavour_ds = ["wmunu_hm_BFilter", "wmunu_hm_CFilterBVeto", "wmunu_hm_CVetoBVeto"]

    ratio_args = {
        "ratio_axlim": 1.5,
        "stats_box": True,
        "ratio_fit": True,
    }
    mass_args = {
        "logbins": True,
        "logx": True,
        # "ratio_axlim": 1.5,
    }
    truth_weighted_args = {
        "weight": "truth_weight",
        "title": "truth - 36.2fb$^{-1}$",
    }
    reco_weighted_args = {
        "weight": "reco_weight",
        "title": "reco - 36.2fb$^{-1}$",
    }

    # TRUTH
    # -----------------------------------
    for datasets in list(datasets.keys()):
                    #  + [flavour_ds, decay_ds]:
        my_analysis.plot_hist(
            datasets,
            "TruthBosonM",
            **mass_args,
            **truth_weighted_args,
            **ratio_args,
        )
        # truth taus
        my_analysis.plot_hist(
            datasets,
            "TruthMuonPt",
            **mass_args,
            **truth_weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            datasets,
            "TruthMuonEta",
            **truth_weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            datasets,
            "TruthMuonPhi",
            logy=False,
            **truth_weighted_args,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "truth_weight",
            logy=True,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "mcWeight",
            logy=True,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "prwWeight",
            logy=True,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "rwCorr",
            logy=True,
            **ratio_args,
        )

        # RECO
        # -----------------------------------
        my_analysis.plot_hist(
            datasets,
            "MuonPt",
            **mass_args,
            **reco_weighted_args,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "MuonEta",
            **reco_weighted_args,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "MuonPhi",
            **reco_weighted_args,
            **ratio_args,
        )

        my_analysis.plot_hist(
            datasets,
            "MET_met",
            **mass_args,
            **reco_weighted_args,
            **ratio_args,
        )

        # my_analysis.plot_hist(
        #     flavour_ds,
        #     "MTW",
        #     **mass_args,
        #     **reco_weighted_args,
        #     **ratio_args,
        # )

    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")
