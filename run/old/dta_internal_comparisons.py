from typing import Dict

import numpy as np

from src.analysis import Analysis

DTA_PATH = "/data/DTA_outputs/2023-10-25/"
# DTA_PATH = "/data/DTA_outputs/2023-06-29/"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets_dict: Dict[str, Dict] = {
        # high-mass - leptons
        "wtaunu_hm_mu_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "TTree_name": "T_s1tmv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_mu.txt",
            # "regen_histograms": False,
            "label": r"high-mass $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wtaunu_hm_e_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "TTree_name": "T_s1tev_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_e.txt",
            # "regen_histograms": False,
            "label": r"high-mass $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_hm_h_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            # "regen_histograms": False,
            "label": r"high-mass $W\rightarrow\tau\nu\rightarrow\mathrm{had}\nu$",
        },
        # high-mass - jet filters
        "wtaunu_hm_BFilter_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120_ECMS_BFilter*/*.root",
            "TTree_name": {"T_s1tev_NOMINAL", "T_s1tmv_NOMINAL", "T_s1thv_NOMINAL"},
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            # "regen_histograms": False,
            "label": r"high-mass BFilter $W\rightarrow\tau\nu$",
        },
        "wtaunu_hm_CFilterBVeto_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120_ECMS_CFilterBVeto*/*.root",
            "TTree_name": {"T_s1tev_NOMINAL", "T_s1tmv_NOMINAL", "T_s1thv_NOMINAL"},
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            # "regen_histograms": False,
            "label": r"high-mass CFilterBVeto $W\rightarrow\tau\nu$",
        },
        "wtaunu_hm_CVetoBVeto_dta": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120_ECMS_CVetoBVeto*/*.root",
            "TTree_name": {"T_s1tev_NOMINAL", "T_s1tmv_NOMINAL", "T_s1thv_NOMINAL"},
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            # "regen_histograms": False,
            "label": r"high-mass CVetoBVeto $W\rightarrow\tau\nu$",
        },
        # inclusive - leptons
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full_mu.txt",
            "TTree_name": "T_s1tmv_NOMINAL",
            # "regen_histograms": False,
            "label": r"inclusive $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full_e.txt",
            "TTree_name": "T_s1tev_NOMINAL",
            # "regen_histograms": False,
            "label": r"inclusive $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"inclusive $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
        },
        # inclusive - jet filters
        "wtaunu_CVetoBVeto_l_dta": {
            "data_path": DTA_PATH + "/*L_maxHTpTV2_CVetoBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive CVetoBVeto $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_CVetoBVeto_dta",
        },
        "wtaunu_CFilterBVeto_l_dta": {
            "data_path": DTA_PATH + "/*L_maxHTpTV2_CFilterBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive CFilterBVeto $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_CFilterBVeto_dta",
        },
        "wtaunu_BFilter_l_dta": {
            "data_path": DTA_PATH + "/*L_maxHTpTV2_BFilter*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive BFilter $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_BFilter_dta",
        },
        "wtaunu_CVetoBVeto_h_dta": {
            "data_path": DTA_PATH + "/*H_maxHTpTV2_CVetoBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive CVetoBVeto $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_CVetoBVeto_dta",
        },
        "wtaunu_CFilterBVeto_h_dta": {
            "data_path": DTA_PATH + "/*H_maxHTpTV2_CFilterBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive CFilterBVeto $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_CFilterBVeto_dta",
        },
        "wtaunu_BFilter_h_dta": {
            "data_path": DTA_PATH + "/*H_maxHTpTV2_BFilter*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"inclusive BFilter $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu_BFilter_dta",
        },
    }

    my_analysis = Analysis(
        datasets_dict,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        regen_histograms=True,
        analysis_label="dta_internal_comparisons",
        hard_cut="TruthBosonM > 120",
        lepton="tau",
        dataset_type="dta",
        log_level=10,
        log_out="console",
        binnings={
            "truth_weight": (100, -1000, 1000),
            "reco_weight": (100, -1000, 1000),
            "mcWeight": (100, -2e7, 2e7),
            "prwWeight": (50, 0, 2),
            "rwCorr": (50, 0, 2),
            "TruthBosonM": np.geomspace(120, 5000, 20),
        },
    )

    BR = len(my_analysis["wtaunu_mu_dta"]) / len(my_analysis["wtaunu_e_dta"])
    my_analysis.logger.info(f"BRANCHING RATIO tau->munu / tau->enu:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_CFilterBVeto_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"CFilterBVeto / CVetoBVeto:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_BFilter_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"BFilter / CVetoBVeto:  {BR:.5f}")

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    # argument dicts
    lepton_ds = ["wtaunu_mu_dta", "wtaunu_e_dta", "wtaunu_h_dta"]
    flavour_ds = ["wtaunu_BFilter_dta", "wtaunu_CFilterBVeto_dta", "wtaunu_CVetoBVeto_dta"]
    lepton_ds_hm = ["wtaunu_hm_mu_dta", "wtaunu_hm_e_dta", "wtaunu_hm_h_dta"]
    flavour_ds_hm = [
        "wtaunu_hm_BFilter_dta",
        "wtaunu_hm_CFilterBVeto_dta",
        "wtaunu_hm_CVetoBVeto_dta",
    ]

    ratio_args = {
        "ratio_axlim": 1.5,
        "stats_box": False,
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
    for datasets in [
        lepton_ds,
        flavour_ds,
        lepton_ds_hm,
        flavour_ds_hm,
        *zip(lepton_ds_hm, lepton_ds),
        *zip(flavour_ds_hm, flavour_ds),
    ]:
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
            "TruthTauPt",
            **mass_args,
            **truth_weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            datasets,
            "TruthTauEta",
            **truth_weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            datasets,
            "TruthTauPhi",
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
        lepton_ds,
        "MuonPt",
        **mass_args,
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MuonEta",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MuonPhi",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        lepton_ds,
        "ElePt",
        **reco_weighted_args,
        **mass_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "EleEta",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "ElePhi",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        lepton_ds,
        "TauPt",
        **mass_args,
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TauEta",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TauPhi",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MET_met",
        **mass_args,
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MET_phi",
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MTW",
        **mass_args,
        **reco_weighted_args,
        **ratio_args,
    )

    my_analysis.histogram_printout()

    my_analysis.logger.info("DONE.")
