from typing import Dict

import numpy as np

from src.analysis import Analysis
import cuts

DTA_PATH = "/eos/home-k/kghorban/DTA_OUT/2023-10-25/"
DATA_OUT_DIR = "/eos/home-k/kghorban/framework_outputs/"
ROOT_DIR = "/afs/cern.ch/user/k/kghorban/framework/"

if __name__ == "__main__":
    datasets_dict: Dict[str, Dict] = {
        # mu
        "wmunu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },

        # e
        "wenu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },
        "wenu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wenu_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_e.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow e\nu$",
            "merge_into": "wenu",
        },

        # tau
        "wtaunu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wtaunu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
    }

    analysis = Analysis(
        datasets_dict,
        data_dir=DATA_OUT_DIR,
        year="2017",
        # regen_histograms=True,
        analysis_label="decay_compare",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
        cuts=cuts.cuts_reco_had,
        extract_vars=cuts.import_vars_reco | cuts.import_vars_truth,
        binnings={
             "DeltaR_tau_mu": np.linspace(0, 15, 20),
             "DeltaR_tau_e": np.linspace(0, 15, 20),
        },
    )
    analysis["wtaunu"].label = r"$W\rightarrow\tau\nu$"
    analysis["wmunu"].label  = r"$W\rightarrow\mu\nu$"
    analysis["wenu"].label   = r"$W\rightarrow e\nu$"
    
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
        datasets,
        "TruthBosonM",
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )
    # truth leptons
    analysis.plot_hist(
        datasets,
        ["TruthTauPt", "TruthMuonPt", "TruthElePt"],
        **mass_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $p_T$ [GeV]"
    )
    analysis.plot_hist(
        datasets,
        ["TruthTauEta", "TruthMuonEta", "TruthEleEta"],
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decau lepton $\eta$",
    )
    analysis.plot_hist(
        datasets,
        ["TruthTauPhi", "TruthMuonPhi", "TruthElePhi"],
        logy=False,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $\phi$"
    )
    analysis.plot_hist(
        datasets,
        "TruthTauPt",
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )

    # RECO
    # ----------------------------------
    lumi_str = f"reco - {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$"
    analysis.plot_hist(
        datasets,
        ["TauPt", "MuonPt", "ElePt"],
        **mass_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Truth decay lepton $p_T$ [GeV]",
        cut=True,
    )
    analysis.plot_hist(
        datasets,
        ["TauEta", "MuonEta", "EleEta"],
        **ratio_args,
        title=lumi_str,
        xlabel=r"Reco decay lepton $\eta$",
        cut=True,
    )
    analysis.plot_hist(
        datasets,
        ["TauPhi", "MuonPhi", "ElePhi"],
        logy=False,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Reco decay lepton $\phi$",
        cut=True,
    )

    analysis.plot_hist(
        datasets,
        "TauPt",
        **mass_args,
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        datasets,
        "DeltaR_tau_mu",
        **ratio_args,
        title=lumi_str,
    )
    analysis.plot_hist(
        datasets,
        "DeltaR_tau_e",
        **ratio_args,
        title=lumi_str,
    )

    analysis.plot_hist(
        datasets,
        "TauPt",
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )
    analysis.plot_hist(
        datasets,
        "MuonPt",
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )
    analysis.plot_hist(
        datasets,
        "ElePt",
        **mass_args,
        **ratio_args,
        title=lumi_str,
        cut=True,
    )

    for d in datasets:
            analysis.plot_hist(
                d,
                "TruthBosonM",
                **mass_args,
                **ratio_args,
                title=analysis[d].label + " | " + lumi_str,
            )
            analysis.plot_hist(
                d, var="truth_weight", logx=False, logy=True, bins=(100, -1000, 1000)
            )
            analysis.plot_hist(
                d, var="reco_weight", logx=False, logy=True, bins=(100, -1000, 1000)
            )

            for lepton in ["Tau", "Muon", "Ele"]:
                analysis.plot_hist(
                    d,
                    f"Truth{lepton}Pt",
                    **mass_args,
                    **ratio_args,
                    title=analysis[d].label + " | " + lumi_str,
                )
                analysis.plot_hist(
                    d,
                    f"Truth{lepton}Eta",
                    **ratio_args,
                    title=analysis[d].label + " | " + lumi_str,
                )
                analysis.plot_hist(
                    d,
                    f"Truth{lepton}Phi",
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
