from typing import Dict, List
from copy import deepcopy
import numpy as np

from src.analysis import Analysis

DTA_PATH = "/eos/home-k/kghorban/DTA_OUT/2023-10-25/"
DATA_OUT_DIR = "/eos/home-k/kghorban/framework_outputs/"
ROOT_DIR = "/afs/cern.ch/user/k/kghorban/framework/"

if __name__ == "__main__":
    datasets_dict: Dict[str, Dict] = {
        # ttbar
        "ttbar": {
            "data_path": DTA_PATH + "user.kghorban.PP8_ttbar_hdamp258p75_dil*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"PP8 ttbar",
        },

        # mu
        "wmunu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },

        # e
        "wenu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wenu_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },

        # tau
        "wtaunu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
    }

    # separate datasets in to different channels
    datasets_dict_full = deepcopy(datasets_dict)
    for name, dataset in datasets_dict.items():
        if "merge_into" in dataset:
             mergable = True
        else:
             mergable = False

        # muon channel
        dataset_mu = deepcopy(dataset)
        dataset_mu["ttree"] = "T_s1tmv_NOMINAL"
        dataset_mu["cutfile"] = ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt"
        dataset_mu["label"] = dataset["label"] + r" (mu chan.)"
        if mergable:
            dataset_mu["merge_into"] = dataset["merge_into"] + "_mu"
        datasets_dict_full[name + "_mu"] = dataset_mu

        # e channel
        dataset_e = deepcopy(dataset)
        dataset_e["ttree"] = "T_s1tev_NOMINAL"
        dataset_e["cutfile"] = ROOT_DIR + "/options/DTA_cuts/dta_full_e.txt"
        dataset_e["label"] = dataset["label"] + r" (e chan.)"
        if mergable:
            dataset_e["merge_into"] = dataset["merge_into"] + "_e"
        datasets_dict_full[name + "_e"] = dataset_e

        # had channel
        dataset_had = deepcopy(dataset)
        dataset_had["ttree"] = "T_s1thv_NOMINAL"
        dataset_had["cutfile"] = ROOT_DIR + "/options/DTA_cuts/dta_full_had.txt"
        dataset_had["label"] = dataset["label"] + r" (had chan.)"
        if mergable:
            dataset_had["merge_into"] = dataset["merge_into"] + "_had"
        datasets_dict_full[name + "_had"] = dataset_had

    analysis = Analysis(
        datasets_dict_full,
        data_dir=DATA_OUT_DIR,
        year="2017",
        regen_histograms=True,
        analysis_label="stack_prelim",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        binnings={
            "MTW": np.geomspace(120, 3000, 20),
            "TauPt": np.geomspace(170, 3000, 20),
            "MET_met": np.geomspace(1, 3000, 20),
        },
    )

    # labels for merged datasets
    analysis["wtaunu"].label = r"SH2211 $W\rightarrow\tau\nu$"
    analysis["wmunu"].label  = r"$W\rightarrow\mu\nu$"
    analysis["wenu"].label   = r"$W\rightarrow e\nu$"
    
    analysis["wtaunu_mu"].label = r"SH2211 $W\rightarrow\tau\nu$ (mu chan.)"
    analysis["wmunu_mu"].label  = r"SH2211 $W\rightarrow\mu\nu$ (mu chan.)"
    analysis["wenu_mu"].label   = r"SH2211 $W\rightarrow e\nu$ (mu chan.)"

    analysis["wtaunu_e"].label = r"SH2211 $W\rightarrow\tau\nu$ (e chan.)"
    analysis["wmunu_e"].label  = r"SH2211 $W\rightarrow\mu\nu$ (e chan.)"
    analysis["wenu_e"].label   = r"SH2211 $W\rightarrow e\nu$ (e chan.)"

    analysis["wtaunu_had"].label = r"SH2211 $W\rightarrow\tau\nu$ (had chan.)"
    analysis["wmunu_had"].label  = r"SH2211 $W\rightarrow\mu\nu$ (had chan.)"
    analysis["wenu_had"].label   = r"SH2211 $W\rightarrow e\nu$ (had chan.)"

    # HISTORGRAMS
    # ==================================================================================================================
    # argument dicts
    datasets = ["ttbar", "wmunu", "wenu", "wtaunu"]
    datasets_mu = [ds + "_mu" for ds in datasets]
    datasets_e = [ds + "_e" for ds in datasets]
    datasets_had = [ds + "_had" for ds in datasets]

    mass_args = {
        "scale_by_bin_width": True,
        "ylabel": "Entries / bin width",
        "logx": True,
    }
    reco_weighted_args = {
        "title": f"reco | mc16d | {analysis.global_lumi/1000:.3g}" + r"fb$^{-1}$"
    }

    full_datasets = [ds for ds in datasets_dict_full if ("_lm" not in ds) and ("_hm" not in ds)]

    # RECO
    # -----------------------------------
    for dataset_list in [
        datasets,
        datasets_mu,
        datasets_e,
        datasets_had,
    ] + full_datasets:
        for cut in (True, False):
            reco_weighted_args["cut"] = cut

            analysis.stack_plot(
                dataset_list,
                "TauPt",
                **mass_args,
                **reco_weighted_args,
            )
            
            analysis.stack_plot(
                dataset_list,
                "TauEta",
                **reco_weighted_args,
            )
            
            analysis.stack_plot(
                dataset_list,
                "TauPhi",
                **reco_weighted_args,
            )
            
            analysis.stack_plot(
                dataset_list,
                "MET_met",
                **mass_args,
                **reco_weighted_args,
            )

            analysis.stack_plot(
                dataset_list,
                "MTW",
                **mass_args,
                **reco_weighted_args,
            )

    analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
