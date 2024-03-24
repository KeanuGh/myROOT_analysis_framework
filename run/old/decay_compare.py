from typing import Dict

import numpy as np

import cuts
from src.analysis import Analysis

# DTA_PATH = "/eos/home-k/kghorban/DTA_OUT/2024-02-05/"
DTA_PATH = "/data/DTA_outputs/2024-02-05/"
ROOT_DIR = "/afs/cern.ch/user/k/kghorban/framework/"

if __name__ == "__main__":
    datasets_dict: Dict[str, Dict] = {
        # mu
        "wmunu_hm": {
            "data_path": DTA_PATH + "*Sh_2211_Wmunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_lm": {
            "data_path": DTA_PATH + "/*Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        # e
        "wenu_hm": {
            "data_path": DTA_PATH + "*Sh_2211_Wenu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_lm": {
            "data_path": DTA_PATH + "/*Sh_2211_Wenu_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_e.txt",
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        # tau
        "wtaunu_hm": {
            "data_path": DTA_PATH + "*Sh_2211_Wtaunu_mW_120*/*.root",
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH + "/*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
    }

    analysis = Analysis(
        datasets_dict,
        year="2017",
        # regen_histograms=True,
        analysis_label="decay_compare",
        dataset_type="dta",
        ttree="T_s1thv_NOMINAL",
        # log_level=10,
        log_out="both",
        cuts=cuts.cuts_reco_had,
        extract_vars=cuts.import_vars_reco | cuts.import_vars_truth,
        binnings={
            "": {
                "DeltaR_tau_mu": np.linspace(0, 15, 20),
                "DeltaR_tau_e": np.linspace(0, 15, 20),
            }
        },
    )
    analysis["wtaunu"].label = r"$W\rightarrow\tau\nu$"
    analysis["wmunu"].label = r"$W\rightarrow\mu\nu$"
    analysis["wenu"].label = r"$W\rightarrow e\nu$"

    # setup merged labels

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    # argument dicts
    datasets = ["wtaunu", "wmunu", "wenu"]
    lumi_str = f"truth - {analysis.global_lumi / 1000 :.3g}" + r"fb$^{-1}$"

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

    # TRUTH
    # -----------------------------------
    analysis.plot_hist(
        "TruthBosonM",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )
    # truth leptons
    analysis.plot_hist(
        ["TruthTauPt", "TruthMuonPt", "TruthElePt"],
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $p_T$ [GeV]",
    )
    analysis.plot_hist(
        ["TruthTauEta", "TruthMuonEta", "TruthEleEta"],
        datasets,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decau lepton $\eta$",
    )
    analysis.plot_hist(
        ["TruthTauPhi", "TruthMuonPhi", "TruthElePhi"],
        datasets,
        logy=False,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $\phi$",
    )
    analysis.plot_hist(
        "TruthTauPt",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        "TruthTauEta",
        datasets,
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        "TruthTauPhi",
        datasets,
        **ratio_args,
        title=lumi_str,
    )

    # RECO
    # ----------------------------------
    lumi_str = f"reco - {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$"
    analysis.plot_hist(
        ["TauPt", "MuonPt", "ElePt"],
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $p_T$ [GeV]",
        cut=True,
    )
    analysis.plot_hist(
        ["TauEta", "MuonEta", "EleEta"],
        datasets,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Reco decay lepton $\eta$",
        cut=True,
    )
    analysis.plot_hist(
        ["TauPhi", "MuonPhi", "ElePhi"],
        datasets,
        logy=False,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Reco decay lepton $\phi$",
        cut=True,
    )

    analysis.plot_hist(
        "TauPt",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        "DeltaR_tau_mu",
        datasets,
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        "DeltaR_tau_e",
        datasets,
        **ratio_args,
        title=lumi_str,
    )

    analysis.plot_hist(
        "TauPt",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )
    analysis.plot_hist(
        "MuonPt",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )
    analysis.plot_hist(
        "ElePt",
        datasets,
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )

    for d in datasets:
        analysis.plot_hist(
            "TruthBosonM",
            d,
            **mass_args,
            **ratio_args,
            title=analysis[d].label + " | " + lumi_str,
        )
        # analysis.plot_hist(
        #     datasets=d, var="truth_weight", logx=False, logy=True, bins=(100, -1000, 1000)
        # )
        # analysis.plot_hist(
        #     datasets=d, var="reco_weight", logx=False, logy=True, bins=(100, -1000, 1000)
        # )

        for lepton in ["Tau", "Muon", "Ele"]:
            analysis.plot_hist(
                f"Truth{lepton}Pt",
                d,
                **mass_args,
                **ratio_args,
                title=analysis[d].label + " | " + lumi_str,
            )
            analysis.plot_hist(
                f"Truth{lepton}Eta",
                d,
                **ratio_args,
                title=analysis[d].label + " | " + lumi_str,
            )
            analysis.plot_hist(
                f"Truth{lepton}Phi",
                d,
                logy=False,
                **ratio_args,
                title=analysis[d].label + " | " + lumi_str,
            )

    # # RECO
    # # -----------------------------------
    # analysis.plot_hist(
    #     lepton_ds,
    #     "MuonPt",
    #     **mass_args,
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "MuonEta",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "MuonPhi",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     lepton_ds,
    #     "ElePt",
    #     **reco_weighted_args,
    #     **mass_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "EleEta",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "ElePhi",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     lepton_ds,
    #     "TauPt",
    #     **mass_args,
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "TauEta",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "TauPhi",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "MET_met",
    #     **mass_args,
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.plot_hist(
    #     flavour_ds,
    #     "MET_phi",
    #     **reco_weighted_args,
    #     **ratio_args,
    # )

    # analysis.histogram_printout()
    analysis.save_histograms()

    analysis.logger.info("DONE.")
