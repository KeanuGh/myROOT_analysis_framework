from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from run import cuts
from src.analysis import Analysis

DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2023-10-25/")
DATA_OUT_DIR = Path("/eos/home-k/kghorban/framework_outputs/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")
extract_vars = cuts.import_vars_reco

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # data
        "data": {
            "data_path": DTA_PATH / "user.kghorban.data17*/*.root",
            "label": "data17",
        },
        # signal
        "wtaunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        # backgrounds
        "wmunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wenu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "ztautau_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Ztautau_mZ_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "ztautau_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "zmumu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zmumu_mZ_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "zmumu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zmumu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "zee_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zee_mZ_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "zee_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zee_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "ttbar": {
            "data_path": DTA_PATH / "user.kghorban.PP8_ttbar_hdamp258p75*/*.root",
            "label": r"PP8 $t\bar{t}$",
        },
    }

    analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2017",
        # regen_histograms=True,
        ttree="T_s1thv_NOMINAL",
        cuts=cuts.cuts_reco_had,
        analysis_label="stack_full",
        dataset_type="dta",
        log_level=10,
        log_out="console",
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

    datasets_merged = [
        "wtaunu",
        "wmunu",
        "wenu",
        "zll",
        "ttbar",
    ]
    # set colours for merged datasets only
    c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for ds in datasets_merged:
        c = next(c_iter)
        analysis[ds].colour = c
        analysis.logger.info(f"Colour for {ds}: {c}")

    # HISTORGRAMS
    # ========================================================================
    # argument dicts
    mass_args = {
        # "scale_by_bin_width": True,
        # "ylabel": "Entries / bin width",
        "ylabel": "Entries",
        "logx": True,
    }
    default_args = {
        "datasets": datasets_merged,
        "title": f"mc16d | {analysis.global_lumi/1000:.3g}" + r"fb$^{-1}$",
        "cut": True,
    }

    # mass-like variables
    for var in [
        "TauPt",
        "MET_met",
        "MTW",
    ]:
        analysis.stack_plot(
            var=var, **default_args, **mass_args, data=True, filename=var + "_stack.png"
        )
        analysis.plot_hist(
            var=var, **default_args, **mass_args, ratio_plot=False, filename=var + ".png"
        )

    # unitless variables
    for var in [
        "TauEta",
        "TauPhi",
    ]:
        analysis.stack_plot(var=var, **default_args, data=True, filename=var + "_stack.png")
        analysis.plot_hist(var=var, **default_args, ratio_plot=False, filename=var + ".png")

    # analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
