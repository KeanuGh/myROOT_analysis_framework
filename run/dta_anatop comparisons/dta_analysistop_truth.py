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
        # dta w->taunu->munu
        "wtaunu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_*/*.root",
            "cutfile": "../../options/DTA_cuts/dta_truth.txt",
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
            "cutfile": "../../options/DTA_cuts/analysistop.txt",
            "lepton": "tau",
            "dataset_type": "analysistop",
            # "force_rebuild": True,
            "label": r"Powheg/Pythia 8 $W\rightarrow\tau\nu$",
        },
        "wtaunu_analysistop_peak": {
            "data_path": ANALYSISTOP_PATH + "/w*taunu/*.root",
            "cutfile": "../../options/DTA_cuts/analysistop.txt",
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
        force_rebuild=True,
        analysis_label="dta_analysistop_compare",
        skip_verify_pkl=True,
        validate_duplicated_events=False,
        force_recalc_cuts=True,
        log_level=10,
        log_out="console",
    )
    # my_analysis.merge_datasets('wtaunu_mu_dta', 'wtaunu_e_dta', verify=True)
    my_analysis.merge_datasets("wtaunu_analysistop", "wtaunu_analysistop_peak")

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
    truth_mass_args = {
        "bins": mass_bins,
        # "bins": (30, 1, 5000),
        "logbins": True,
        "logx": True,
        "ratio_axlim": 1.5,
    }
    weighted_args = {
        "weight": "truth_weight",
        "name_prefix": "cut",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    my_analysis.apply_cuts(truth=True)

    # TRUTH
    # -----------------------------------
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthMTW", "mt_born"],
        **truth_mass_args,
        **weighted_args,
        **ratio_args
    )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthDilepM", "MC_WZ_dilep_m_born"],
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthBosonM", "MC_WZ_dilep_m_born"],
        xlabel="Truth Boson $M$",
        **truth_mass_args,
        **weighted_args,
        **ratio_args
    )
    # my_analysis.plot_hist(
    #     "wtaunu_dta",
    #     ["CalcBosonE", "TruthBosonE"],
    #     labels=["Calculated Boson $E$", "Truth Boson $E$"],
    #     xlabel="$E$ [GeV]",
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )

    # truth taus
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPt", "MC_WZmu_el_pt_born"],
        **truth_mass_args,
        **weighted_args,
        **ratio_args
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauEta", "MC_WZmu_el_eta_born"],
        bins=(30, -5, 5),
        **weighted_args,
        **ratio_args
    )
    my_analysis.plot_hist(
        ["wtaunu_dta", "wtaunu_analysistop"],
        ["TruthTauPhi", "MC_WZmu_el_phi_born"],
        bins=(30, -np.pi, np.pi),
        logy=False,
        **weighted_args,
        **ratio_args
    )

    # neutrinos
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPt1", "MC_WZmu_el_pt_born"],
    #     labels=["Sherpa Truth Neutrino 1", "Powheg Truth Neutrino"],
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoEta1", "MC_WZmu_el_eta_born"],
    #     labels=["Sherpa Truth Neutrino 1", "Powheg Truth Neutrino"],
    #     bins=(30, -5, 5),
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPhi1", "MC_WZmu_el_phi_born"],
    #     labels=["Sherpa Truth Neutrino 1", "Powheg Truth Neutrino"],
    #     bins=(30, -np.pi, np.pi),
    #     logy=False,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPt2", "MC_WZmu_el_pt_born"],
    #     labels=["Sherpa Truth Neutrino 2", "Powheg Truth Neutrino"],
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoEta2", "MC_WZmu_el_eta_born"],
    #     labels=["Sherpa Truth Neutrino 2", "Powheg Truth Neutrino"],
    #     bins=(30, -5, 5),
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPhi2", "MC_WZmu_el_phi_born"],
    #     labels=["Sherpa Truth Neutrino 2", "Powheg Truth Neutrino"],
    #     bins=(30, -np.pi, np.pi),
    #     logy=False,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPt3", "MC_WZmu_el_pt_born"],
    #     labels=["Sherpa Truth Neutrino 3", "Powheg Truth Neutrino"],
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoEta3", "MC_WZmu_el_eta_born"],
    #     labels=["Sherpa Truth Neutrino 3", "Powheg Truth Neutrino"],
    #     bins=(30, -5, 5),
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["TruthNeutrinoPhi3", "MC_WZmu_el_phi_born"],
    #     labels=["Sherpa Truth Neutrino 3", "Powheg Truth Neutrino"],
    #     bins=(30, -np.pi, np.pi),
    #     logy=False,
    #     **weighted_args,
    #     **ratio_args
    # )
    #
    # # tau against neutrino
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_dta"],
    #     ["TruthNeutrinoPt1", "TruthTauPt"],
    #     labels=["Sherpa Truth Neutrino 1", "Sherpa Truth Tau"],
    #     xlabel="$p_T$ [GeV]",
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # # tau against neutrino
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_dta"],
    #     ["TruthNeutrinoPt2", "TruthTauPt"],
    #     labels=["Sherpa Truth Neutrino 2", "Sherpa Truth Tau"],
    #     xlabel="$p_T$ [GeV]",
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # # tau against neutrino
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_dta"],
    #     ["TruthNeutrinoPt3", "TruthTauPt"],
    #     labels=["Sherpa Truth Neutrino 3", "Sherpa Truth Tau"],
    #     xlabel="$p_T$ [GeV]",
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )

    # vis tau
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["VisTruthTauPt", "MC_WZmu_el_pt_born"],
    #     **truth_mass_args,
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["VisTruthTauEta", "MC_WZmu_el_eta_born"],
    #     bins=(30, -5, 5),
    #     **weighted_args,
    #     **ratio_args
    # )
    # my_analysis.plot_hist(
    #     ["wtaunu_dta", "wtaunu_analysistop"],
    #     ["VisTruthTauPhi", "MC_WZmu_el_phi_born"],
    #     bins=(30, -np.pi, np.pi),
    #     logy=False,
    #     **weighted_args,
    #     **ratio_args
    # )

    my_analysis["wtaunu_dta"].dsid_metadata_printout()
    my_analysis["wtaunu_analysistop"].dsid_metadata_printout()
    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    my_analysis.logger.info("DONE.")


if __name__ == "__main__":
    main()
