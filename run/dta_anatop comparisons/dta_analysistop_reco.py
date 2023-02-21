import numpy as np

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-01-31/"
ANALYSISTOP_PATH = "/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets = {
        # dta w->taunu->munu
        "wtaunu_L_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile_path": "../../options/DTA_cuts/dta_reco.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "lepton": "tau",
            "dataset_type": "dta",
            "hard_cut": "TruthTau_decay_mode == 1",
            "force_rebuild": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        # analysistop
        "wtaunu_analysistop_reco": {
            "data_path": ANALYSISTOP_PATH + "/w*taunu_*/*.root",
            "cutfile_path": "../../options/DTA_cuts/analysistop.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wtaunu_analysistop_reco_peak": {
            "data_path": ANALYSISTOP_PATH + "/w*taunu/*.root",
            "cutfile_path": "../../options/DTA_cuts/analysistop.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            "hard_cut": "MC_WZ_m < 120",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # force_rebuild=True,
        analysis_label="dta_analysistop_reco",
        # skip_verify_pkl=True,
        # validate_duplicated_events=False,
        log_level=10,
        log_out="console",
    )

    my_analysis.merge_datasets("wtaunu_analysistop_reco", "wtaunu_analysistop_reco_peak")

    my_analysis.apply_cuts()

    # HISTORGRAMS
    # ==================================================================================================================
    mass_bins = np.array(
        [
            130,
            140.3921,
            151.6149,
            163.7349,
            176.8237,
            190.9588,
            206.2239,
            222.7093,
            240.5125,
            259.7389,
            280.5022,
            302.9253,
            327.1409,
            353.2922,
            381.5341,
            412.0336,
            444.9712,
            480.5419,
            518.956,
            560.4409,
            605.242,
            653.6246,
            705.8748,
            762.3018,
            823.2396,
            889.0486,
            960.1184,
            1036.869,
            1119.756,
            1209.268,
            1305.936,
            1410.332,
            1523.072,
            1644.825,
            1776.311,
            1918.308,
            2071.656,
            2237.263,
            2416.107,
            2609.249,
            2817.83,
            3043.085,
            3286.347,
            3549.055,
            3832.763,
            4139.151,
            4470.031,
            4827.361,
            5213.257,
        ]
    )

    # argument dicts
    ratio_args = {
        # "ratio_axlim": 1.5,
        "ratio_label": "Powheg/Sherpa",
        "stats_box": False,
        "ratio_fit": True,
    }
    reco_mass_args = {
        # "bins": mass_bins,
        "bins": (30, 1, 5000),
        "logbins": True,
        "logx": True,
        "ratio_axlim": 1.5,
    }
    weighted_args = {
        "weight": "reco_weight",
        "name_prefix": "weighted",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }
    el_weighted_args = {
        "weight": ["ele_reco_weight", "reco_weight"],
        "name_prefix": "weighted",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }
    mu_weighted_args = {
        "weight": ["muon_reco_weight", "reco_weight"],
        "name_prefix": "weighted",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }
    tau_weighted_args = {
        "weight": ["tau_reco_weight", "reco_weight"],
        "name_prefix": "weighted",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    # RECO
    # ------------------------------------------------------
    my_analysis.plot_hist(
        "wtaunu_L_dta",
        "TauPt",
        **reco_mass_args,
        **weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        "wtaunu_L_dta",
        "TauPhi",
        bins=(30, -np.pi, np.pi),
        **weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        "wtaunu_L_dta",
        "TauEta",
        bins=(30, -5, 5),
        **weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["ElePt", "el_pt"],
        xlabel="Truth Boson $M$",
        **reco_mass_args,
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["Ele_delta_z0_sintheta", "el_delta_z0_sintheta"],
        bins=(30, -5, 5),
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["Ele_delta_z0_sintheta", "el_delta_z0_sintheta"],
        bins=(30, 0, 2 * np.pi),
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["ElePhi", "el_phi"],
        bins=(30, -np.pi, np.pi),
        **el_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["EleEta", "el_eta"],
        bins=(30, -5, 5),
        **weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["MuonPt", "mu_pt"],
        **reco_mass_args,
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["Muon_delta_z0_sintheta", "mu_delta_z0_sintheta"],
        bins=(30, 0, 2 * np.pi),
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["MuonPhi", "mu_phi"],
        bins=(30, -np.pi, np.pi),
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["MuonEta", "mu_eta"],
        bins=(30, -5, 5),
        **mu_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_L_dta", "wtaunu_analysistop_reco"],
        ["MET_met", "met_met"],
        **reco_mass_args,
        **weighted_args,
        **ratio_args,
    )

    my_analysis["wtaunu_L_dta"].dsid_metadata_printout()
    my_analysis["wtaunu_analysistop_reco"].dsid_metadata_printout()
    my_analysis.histogram_printout()

    my_analysis.logger.info("DONE.")
