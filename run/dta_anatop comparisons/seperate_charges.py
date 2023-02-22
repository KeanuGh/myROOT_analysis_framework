import numpy as np

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-01-31/"
ANALYSISTOP_PATH = "/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/data/dataset_pkl_outputs/"


def main():
    datasets = {
        # # plus
        # # dta w->taunu->munu
        # "wplustaunu_dta": {
        #     "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*/*.root",
        #     "cutfile": "../../options/DTA_cuts/dta_truth.txt",
        #     "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
        #     "hard_cut": "TruthTauCharge == 1",
        #     "lepton": "tau",
        #     "dataset_type": "dta",
        #     # "force_rebuild": True,
        #     "label": r"Sherpa 2211 $W^+\rightarrow\tau\nu$",
        # },
        # # analysistop w->taunu->munu
        # "wplustaunu_analysistop": {
        #     "data_path": ANALYSISTOP_PATH + "/wplustaunu_*/*.root",
        #     "cutfile": "../../options/DTA_cuts/analysistop.txt",
        #     "lepton": "tau",
        #     "dataset_type": "analysistop",
        #     # "force_rebuild": True,
        #     "label": r"Powheg/Pythia 8 $W^+\rightarrow\tau\nu$",
        # },
        # "wplustaunu_analysistop_peak": {
        #     "data_path": ANALYSISTOP_PATH + "/wplustaunu/*.root",
        #     "cutfile": "../../options/DTA_cuts/analysistop.txt",
        #     "lepton": "tau",
        #     "dataset_type": "analysistop",
        #     "hard_cut": "MC_WZ_m < 120",
        #     # "force_rebuild": True,
        #     "label": r"Powheg/Pythia 8 $W^+\rightarrow\tau\nu$",
        # },
        # minus
        # dta w->taunu->munu
        "wmintaunu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*/*.root",
            "cutfile": "../../options/DTA_cuts/dta_truth.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            "hard_cut": "TruthTauCharge == -1",
            "lepton": "tau",
            "dataset_type": "dta",
            # "force_rebuild": True,
            # 'force_recalc_weights': True,
            "label": r"Sherpa 2211 $W^-\rightarrow\tau\nu$",
        },
        # analysistop w->taunu->munu
        "wmintaunu_analysistop": {
            "data_path": ANALYSISTOP_PATH + "/wmintaunu_*/*.root",
            "cutfile": "../../options/DTA_cuts/analysistop.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W^-\rightarrow\tau\nu$",
        },
        "wmintaunu_analysistop_peak": {
            "data_path": ANALYSISTOP_PATH + "/wmintaunu/*.root",
            "cutfile": "../../options/DTA_cuts/analysistop.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            "hard_cut": "MC_WZ_m < 120",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W^-\rightarrow\tau\nu$",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # force_rebuild=True,
        analysis_label="dta_analysistop_sep_charge",
        skip_verify_pkl=True,
        log_level=10,
        log_out="both",
    )

    # HISTORGRAMS
    # ==================================================================================================================
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
    weighted_args = {
        "weight": "truth_weight",
        "name_prefix": "weighted",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    # TRUTH
    # -----------------------------------
    for charge in [
        # "plus",
        "min",
    ]:
        my_analysis.merge_datasets(
            f"w{charge}taunu_analysistop", f"w{charge}taunu_analysistop_peak"
        )

        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthDilepM", "MC_WZ_dilep_m_born"],
            **truth_mass_args,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthBosonM", "MC_WZ_dilep_m_born"],
            xlabel="Truth Boson $M$",
            **truth_mass_args,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthTauPt", "MC_WZmu_el_pt_born"],
            **truth_mass_args,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthTauEta", "MC_WZmu_el_eta_born"],
            bins=(30, -5, 5),
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthTauPhi", "MC_WZmu_el_phi_born"],
            bins=(30, -np.pi, np.pi),
            logy=False,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthNeutrinoPt", "MC_WZmu_el_pt_born"],
            **truth_mass_args,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthNeutrinoEta", "MC_WZmu_el_eta_born"],
            bins=(30, -5, 5),
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthNeutrinoPhi", "MC_WZmu_el_phi_born"],
            bins=(30, -np.pi, np.pi),
            logy=False,
            **weighted_args,
            **ratio_args,
        )
        my_analysis.plot_hist(
            [f"w{charge}taunu_dta", f"w{charge}taunu_analysistop"],
            ["TruthMTW", "mt_born"],
            **truth_mass_args,
            **weighted_args,
            **ratio_args,
        )

        my_analysis[f"w{charge}taunu_dta"].dsid_metadata_printout()
        my_analysis[f"w{charge}taunu_analysistop"].dsid_metadata_printout()

    my_analysis.histogram_printout()
    my_analysis.logger.info("DONE.")


if __name__ == "__main__":
    main()
