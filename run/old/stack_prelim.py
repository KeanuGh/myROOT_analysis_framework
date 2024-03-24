from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np

from run import cuts
from src.analysis import Analysis

DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2023-10-25/")
DATA_OUT_DIR = Path("/eos/home-k/kghorban/framework_outputs/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")
extract_vars = cuts.import_vars_reco | cuts.import_vars_truth

if __name__ == "__main__":
    datasets_dict_full: Dict[str, Dict] = {
        # mu
        "wmunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        # e
        "wenu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        # tau
        "wtaunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"SH2211 high-mass $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"SH2211 inclusive $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        # ttbar
        "ttbar": {
            "data_path": DTA_PATH / "user.kghorban.PP8_ttbar_hdamp258p75*/*.root",
            "label": r"PP8 ttbar",
        },
    }

    # separate datasets in to different channels (and ONLY use these)
    datasets_dict_sep = dict()
    for name, dataset in datasets_dict_full.items():
        if "merge_into" in dataset:
            mergable = True
        else:
            mergable = False

        # muon channel
        dataset_mu = deepcopy(dataset)
        dataset_mu["ttree"] = "T_s1tmv_NOMINAL"
        # dataset_mu["cutfile"] = CUTFILE_DIR / "dta_full_mu.txt"
        dataset_mu["cuts"] = cuts.cuts_reco_mu
        dataset_mu["label"] = dataset["label"] + r" (mu chan.)"
        if mergable:
            dataset_mu["merge_into"] = dataset["merge_into"] + "_mu"
        datasets_dict_sep[name + "_mu"] = dataset_mu

        # e channel
        dataset_e = deepcopy(dataset)
        dataset_e["ttree"] = "T_s1tev_NOMINAL"
        # dataset_e["cutfile"] = CUTFILE_DIR / "dta_full_e.txt"
        dataset_e["cuts"] = cuts.cuts_reco_e
        dataset_e["label"] = dataset["label"] + r" (e chan.)"
        if mergable:
            dataset_e["merge_into"] = dataset["merge_into"] + "_e"
        datasets_dict_sep[name + "_e"] = dataset_e

        # had channel
        dataset_had = deepcopy(dataset)
        dataset_had["ttree"] = "T_s1thv_NOMINAL"
        # dataset_had["cutfile"] = CUTFILE_DIR / "dta_full_had.txt"
        dataset_had["cuts"] = cuts.cuts_reco_had
        dataset_had["label"] = dataset["label"] + r" (had chan.)"
        if mergable:
            dataset_had["merge_into"] = dataset["merge_into"] + "_had"
        datasets_dict_sep[name + "_had"] = dataset_had

    analysis = Analysis(
        datasets_dict_sep,
        data_dir=DATA_OUT_DIR,
        year="2017",
        regen_histograms=True,
        analysis_label="stack_prelim",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=extract_vars,
        binnings={
            "MTW": np.geomspace(150, 3000, 20),
            "TauPt": np.geomspace(170, 3000, 20),
            "TauEta": np.linspace(-2.47, 2.47, 20),
            "EleEta": np.linspace(-2.47, 2.47, 20),
            "MuonEta": np.linspace(-2.5, 2.5, 20),
            "MET_met": np.geomspace(1, 3000, 20),
        },
    )
    analysis.cutflow_printout(latex=True)

    # # labels for merged datasets
    # analysis["wtaunu"].label = r"SH2211 $W\rightarrow\tau\nu$"
    # analysis["wmunu"].label  = r"$W\rightarrow\mu\nu$"
    # analysis["wenu"].label   = r"$W\rightarrow e\nu$"

    analysis["wtaunu_mu"].label = r"SH2211 $W\rightarrow\tau\nu$ (mu chan.)"
    analysis["wmunu_mu"].label = r"SH2211 $W\rightarrow\mu\nu$ (mu chan.)"
    analysis["wenu_mu"].label = r"SH2211 $W\rightarrow e\nu$ (mu chan.)"

    analysis["wtaunu_e"].label = r"SH2211 $W\rightarrow\tau\nu$ (e chan.)"
    analysis["wmunu_e"].label = r"SH2211 $W\rightarrow\mu\nu$ (e chan.)"
    analysis["wenu_e"].label = r"SH2211 $W\rightarrow e\nu$ (e chan.)"

    analysis["wtaunu_had"].label = r"SH2211 $W\rightarrow\tau\nu$ (had chan.)"
    analysis["wmunu_had"].label = r"SH2211 $W\rightarrow\mu\nu$ (had chan.)"
    analysis["wenu_had"].label = r"SH2211 $W\rightarrow e\nu$ (had chan.)"

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

    # RECO PLOT LOOP
    # -----------------------------------
    for dataset_list in [
        datasets_mu,
        datasets_e,
        datasets_had,
    ]:
        for cut in (True, False):
            default_args = {
                "datasets": dataset_list,
                "title": f"reco | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
                "cut": cut,
                "suffix": "DeltaR05" if (dataset_list != datasets_had and cut) else "",
                # only leptonic channels need a DeltaR cut
            }

            # mass-like variables
            for var in [
                "TauPt",
                "MET_met",
                "MTW",
            ]:
                analysis.stack_plot(var=var, **default_args, **mass_args)
                analysis.plot_hist(var=var, **default_args, **mass_args, ratio_plot=False)

            # unitless variables
            for var in [
                "TauEta",
                "TauPhi",
                "DeltaR_tau_mu",
                "DeltaR_tau_e",
            ]:
                analysis.stack_plot(var=var, **default_args)
                analysis.plot_hist(var=var, **default_args, ratio_plot=False)

    analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
