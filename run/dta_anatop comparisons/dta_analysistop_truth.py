import numpy as np

from src.analysis import Analysis
from utils.plotting_tools import default_mass_bins

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-01-31/"
ANALYSISTOP_PATH = "/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets = {
        # dta w->taunu->munu
        "wtaunu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*/*.root",
            "cutfile": "../../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "lepton": "tau",
            "dataset_type": "dta",
            # "force_rebuild": True,
            # 'force_recalc_weights': True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu$",
        },
        # analysistop w->taunu->munu
        "wtaunu_analysistop": {
            "data_path": ANALYSISTOP_PATH + "/w*taunu_*/*.root",
            "cutfile": "../../options/DTA_cuts/analysistop_full.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W\rightarrow\tau\nu$",
        },
        "wtaunu_analysistop_peak": {
            "data_path": ANALYSISTOP_PATH + "/w*taunu/*.root",
            "cutfile": "../../options/DTA_cuts/analysistop_full.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            "hard_cut": "MC_WZ_m < 120",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W\rightarrow\tau\nu$",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # force_rebuild=True,
        analysis_label="dta_analysistop_compare",
        # skip_verify_pkl=True,
        validate_duplicated_events=False,
        # regen_histograms=True,
        force_recalc_cuts=True,
        log_level=10,
        log_out="both",
    )
    # my_analysis.merge_datasets('wtaunu_mu_dta', 'wtaunu_e_dta', verify=True)
    my_analysis.merge_datasets("wtaunu_analysistop", "wtaunu_analysistop_peak")
    my_analysis["wtaunu_analysistop"].gen_histograms()
    my_analysis["wtaunu_analysistop"].dsid_metadata_printout()

    # calculate boson E
    # my_analysis["wtaunu_dta"]["CalcBosonE"] = (
    #     my_analysis["wtaunu_dta"]["TruthTauE"] + my_analysis["wtaunu_dta"]["TruthNeutrinoE"]
    # )

    # HISTORGRAMS
    # ==================================================================================================================
    # look at monte carlo weights
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -5, 5), filename_suffix='_5', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -10000, 10000), filename_suffix='full', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -1e6, 1e6), filename_suffix='_mill', logy=True, yerr=None)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -5e9, 5e9), logy=True, filename_suffix='_bill', yerr=None)

    # argument dicts
    ratio_args = {
        # "ratio_axlim": 1.5,
        "ratio_label": "Powheg/Sherpa",
        "stats_box": False,
        "ratio_fit": True,
    }
    truth_mass_args = {
        "bins": (30, 1, 5000),
        "logbins": True,
        "logx": True,
        "ratio_axlim": 1.5,
    }
    truth_weighted_args = {
        "weight": "truth_weight",
        "prefix": "truth_inclusive",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    # TRUTH
    # -----------------------------------
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthMTW", "mt_born"],
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthBosonM", "MC_WZ_dilep_m_born"],
        xlabel="Truth Boson $M$",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )

    # truth taus
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPt", "MC_WZmu_el_pt_born"],
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauEta", "MC_WZmu_el_eta_born"],
        bins=(30, -5, 5),
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPhi", "MC_WZmu_el_phi_born"],
        bins=(30, -np.pi, np.pi),
        logy=False,
        **truth_weighted_args,
        **ratio_args,
    )

    # vis tau
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["VisTruthTauPt", "MC_WZmu_el_pt_born"],
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["VisTruthTauEta", "MC_WZmu_el_eta_born"],
        bins=(30, -5, 5),
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["VisTruthTauPhi", "MC_WZmu_el_phi_born"],
        bins=(30, -np.pi, np.pi),
        logy=False,
        **truth_weighted_args,
        **ratio_args,
    )

    # CUTS
    # ------------------------------------------------------------------
    cut_mass_args = {
        "bins": default_mass_bins,
        "logx": True,
        "logy": True,
        # "ratio_axlim": 1.5,
    }
    cut_truth_args = {
        "cut": True,
        "title": "truth - 36.2fb$^{-1}$",
    }
    cut_reco_args = {
        "cut": True,
        "title": "reco - 36.2fb$^{-1}$",
    }
    el_weighted_args = {
        "weight": ["ele_weight_reco", "reco_weight"],
    }
    mu_weighted_args = {
        "weight": ["muon_weight_reco", "reco_weight"],
    }
    tau_weighted_args = {
        "weight": ["tau_weight_reco", "reco_weight"],
    }
    # my_analysis.apply_cuts()
    # my_analysis.wtaunu_dta.cutflow_printout()

    # truth
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthMTW", "mt_born"],
        **cut_mass_args,
        **cut_truth_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthBosonM", "MC_WZ_dilep_m_born"],
        xlabel="Truth Boson $M$",
        **cut_mass_args,
        **cut_truth_args,
        **ratio_args,
    )

    # truth taus
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPt", "MC_WZmu_el_pt_born"],
        **cut_mass_args,
        **cut_truth_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauEta", "MC_WZmu_el_eta_born"],
        bins=(30, -5, 5),
        **cut_truth_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPhi", "MC_WZmu_el_phi_born"],
        bins=(30, -np.pi, np.pi),
        logy=False,
        **cut_truth_args,
        **ratio_args,
    )

    # reco
    my_analysis.plot_hist(
        "wtaunu_dta",
        "TauPt",
        **cut_mass_args,
        **cut_reco_args,
        **tau_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        "wtaunu_dta",
        "TauPhi",
        bins=(30, -np.pi, np.pi),
        **cut_reco_args,
        **tau_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        "wtaunu_dta",
        "TauEta",
        bins=(30, -5, 5),
        **cut_reco_args,
        **tau_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["ElePt", "el_pt"],
        xlabel="Truth Boson $M$",
        **cut_mass_args,
        **cut_reco_args,
        **el_weighted_args,
        **ratio_args,
    )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["Ele_delta_z0_sintheta", "e_delta_z0_sintheta"],
    #     bins=(30, 0, 2 * np.pi),
    #     **cut_reco_args,
    #     **el_weighted_args,
    #     **ratio_args,
    # )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["ElePhi", "el_phi"],
        bins=(30, -np.pi, np.pi),
        **cut_reco_args,
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["EleEta", "el_eta"],
        bins=(30, -5, 5),
        **cut_reco_args,
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["MuonPt", "mu_pt"],
        **cut_mass_args,
        **cut_reco_args,
        **mu_weighted_args,
        **ratio_args,
    )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["Muon_delta_z0_sintheta", "mu_delta_z0_sintheta"],
    #     bins=(30, 0, 2 * np.pi),
    #     **cut_reco_args,
    #     **mu_weighted_args,
    #     **ratio_args,
    # )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["MuonPhi", "mu_phi"],
        bins=(30, -np.pi, np.pi),
        **cut_reco_args,
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["MuonEta", "mu_eta"],
        bins=(30, -5, 5),
        **cut_reco_args,
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["MET_met", "met_met"],
        weight="reco_weight",
        **cut_mass_args,
        **cut_reco_args,
        **ratio_args,
    )

    # my_analysis["wtaunu_dta"].dsid_metadata_printout()
    my_analysis["wtaunu_analysistop"].dsid_metadata_printout()
    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")
