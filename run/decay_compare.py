from typing import Dict

from src.analysis import Analysis

DTA_PATH = "/eos/home-k/kghorban/DTA_OUT/2023-10-25/"
DATA_OUT_DIR = "/eos/home-k/kghorban/framework_outputs/"
ROOT_DIR = "/afs/cern.ch/user/k/kghorban/framework/"

if __name__ == "__main__":
    datasets_dict: Dict[str, Dict] = {
        # mu
        "wmunu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wmunu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },
        "wmunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wmunu_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_mu.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\mu\nu$",
            "merge_into": "wmunu",
        },

        # e
        "wenu_hm": {
            "data_path": DTA_PATH + "user.kghorban.Sh_2211_Wenu_mW_120*/*.root",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_e.txt",
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
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_had.txt",
            "hard_cut": "TruthBosonM > 120",
            "label": r"high-mass $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
        "wtaunu_lm": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "cutfile": ROOT_DIR + "/options/DTA_cuts/dta_full_had.txt",
            "ttree": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthBosonM < 120",
            "label": r"inclusive $W\rightarrow\tau\nu$",
            "merge_into": "wtaunu",
        },
    }

    analysis = Analysis(
        datasets_dict,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # regen_histograms=True,
        analysis_label="decay_compare",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
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
    lumi_str = r"truth - 36.2fb$^{-1}$"

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
    }
    reco_weighted_args = {
        "weight": "reco_weight",
    }

    # TRUTH
    # -----------------------------------
    analysis.plot_hist(
        datasets,
        "TruthBosonM",
        **mass_args,
        **truth_weighted_args,
        **ratio_args,
        title=lumi_str,
    )
    # truth taus
    analysis.plot_hist(
        datasets,
        ["TruthTauPt", "TruthMuonPt", "TruthElePt"],
        **mass_args,
        **truth_weighted_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Lepton $p_T$ [GeV]"
    )
    analysis.plot_hist(
        datasets,
        ["TruthTauEta", "TruthMuonEta", "TruthEleEta"],
        **truth_weighted_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Lepton $\eta$",
    )
    analysis.plot_hist(
        datasets,
        ["TruthTauPhi", "TruthMuonPhi", "TruthElePhi"],
        logy=False,
        **truth_weighted_args,
        **ratio_args,
        title=lumi_str,
        xlabel=r"Lepton $\phi$"
    )

    for d in datasets:
            analysis.plot_hist(
                d,
                "TruthBosonM",
                **mass_args,
                **truth_weighted_args,
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
                    **truth_weighted_args,
                    **ratio_args,
                    title=analysis[d].label + " | " + lumi_str,
                )
                analysis.plot_hist(
                    d,
                    f"Truth{lepton}Eta",
                    **truth_weighted_args,
                    **ratio_args,
                    title=analysis[d].label + " | " + lumi_str,
                )
                analysis.plot_hist(
                    d,
                    f"Truth{lepton}Phi",
                    logy=False,
                    **truth_weighted_args,
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
