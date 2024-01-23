from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.cutfile import Cut
from src.analysis import Analysis

DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2023-10-25/")
DATA_OUT_DIR = Path("/eos/home-k/kghorban/framework_outputs/")
CUTFILE_DIR = Path("/afs/cern.ch/user/k/kghorban/framework/options/DTA_cuts/reco")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # data
        "data": {
            "data_path": DTA_PATH / "user.kghorban.data17*/*.root",
            "label": "data17",
        },
        # signal
        "wtaunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        # backgrounds
        "wmunu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wenu_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "ztautau_lm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "ztautau_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Ztautau_mZ_120*/*.root",
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
        "zmumu_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zmumu_mZ_120*/*.root",
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
        "zee_hm": {
            "data_path": DTA_PATH / "user.kghorban.Sh_2211_Zee_mZ_120*/*.root",
            "hard_cut": "TruthBosonM >= 120",
            "label": r"Sh2211 $Z\rightarrow ll$",
            "merge_into": "zll",
        },
        "ttbar": {
            "data_path": DTA_PATH / "user.kghorban.PP8_ttbar_hdamp258p75*/*.root",
            "label": r"PP8 $t\bar{t}$",
        },
    }

    cuts: dict[str, list[Cut]] = {
        "medTau": [
            Cut(
                r"\mathrm{pass trigger}",
                r"passTrigger",
            ),
            Cut(
                r"\mathrm{medium tau}",
                r"TauMediumWP",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 85",
                r"MET_met > 85",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
        ],
        "looseTau" : [
            Cut(
                r"\mathrm{pass trigger}",
                r"passTrigger",
            ),
            Cut(
                r"\mathrm{loose tau}",
                r"TauLooseWP",
            ),
            Cut(
                r"$p_T^\tau > 170$",
                r"TauPt > 170",
            ),
            Cut(
                r"$E_T^{\mathrm{miss}} > 85",
                r"MET_met > 85",
            ),
            Cut(
                r"$m_T^W > 150$",
                r"MTW > 150",
            ),
        ]
    }

    wanted_branches = {
        "TauEta",
        "TauPhi",
        "TauPt",
        "MET_met",
        "MET_phi",
        "MTW",
    }

    analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2017",
        # regen_histograms=True,
        ttree="T_s1thv_NOMINAL",
        cuts=cuts,
        analysis_label="stack_full",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=wanted_branches,
        binnings={
            "MTW": np.geomspace(150, 2000, 20),
            "TauPt": np.geomspace(170, 2000, 20),
            "TauEta": np.linspace(-2.47, 2.47, 20),
            "EleEta": np.linspace(-2.47, 2.47, 20),
            "MuonEta": np.linspace(-2.5, 2.5, 20),
            "MET_met": np.geomspace(85, 2000, 20),
        },
    )
    analysis.cutflow_printout(latex=True)
    analysis["wtaunu"].is_signal = True

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
        analysis.stack_plot(var=var, **default_args, logx=True, data=True)
        analysis.stack_plot(var=var, **default_args, logy=False, data=True, suffix="liny")
        analysis.stack_plot(var=var, **default_args, logx=True, data=True, scale_by_bin_width=True, ylabel="Entries / bin width")
        analysis.plot_hist(var=var, **default_args, logx=True, ratio_plot=False)

    # unitless variables
    for var in [
        "TauEta",
        "TauPhi",
    ]:
        analysis.stack_plot(var=var, **default_args, data=True)
        analysis.plot_hist(var=var, **default_args, ratio_plot=False)

    analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
