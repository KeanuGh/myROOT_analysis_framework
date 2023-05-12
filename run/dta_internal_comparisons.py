from typing import Dict

from numpy import pi

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-05-05/"
ANALYSISTOP_PATH = "/mnt/D/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/mnt/D/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # dta w->taunu->munu
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1tmv_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1tev_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
        },
        "wtaunu_CVetoBVeto_dta": {
            "data_path": DTA_PATH + "/*CVetoBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"CVetoBVeto",
        },
        "wtaunu_CFilterBVeto_dta": {
            "data_path": DTA_PATH + "/*CFilterBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"CFilterBVeto",
        },
        "wtaunu_BFilter_dta": {
            "data_path": DTA_PATH + "/*BFilter*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"BFilter",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # regen_histograms=True,
        analysis_label="dta_internal_comparisons",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
    )
    # my_analysis.merge_datasets('wtaunu_e_dta', 'wtaunu_e_dta_peak')

    BR = len(my_analysis["wtaunu_mu_dta"]) / len(my_analysis["wtaunu_e_dta"])
    my_analysis.logger.info(f"BRANCHING RATIO tau->munu / tau->enu:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_CFilterBVeto_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"CFilterBVeto / CVetoBVeto:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_BFilter_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"BFilter / CVetoBVeto:  {BR:.5f}")

    # my_analysis.apply_cuts(truth=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    # argument dicts
    lepton_ds = ["wtaunu_mu_dta", "wtaunu_e_dta", "wtaunu_h_dta"]
    flavour_ds = ["wtaunu_BFilter_dta", "wtaunu_CFilterBVeto_dta", "wtaunu_CVetoBVeto_dta"]

    ratio_args = {
        # "ratio_axlim": 1.5,
        "stats_box": False,
        "ratio_fit": True,
    }
    truth_mass_args = {
        "bins": (30, 1, 5000),
        "logbins": True,
        "logx": True,
        # "ratio_axlim": 1.5,
    }
    truth_weighted_args = {
        "weight": "truth_weight",
        "prefix": "truth_inclusive",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    # TRUTH
    # -----------------------------------
    # leptons
    my_analysis.plot_hist(
        lepton_ds,
        "TruthMTW",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthBosonM",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    # truth taus
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauPt",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauEta",
        bins=(30, -5, 5),
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauPhi",
        bins=(30, -pi, pi),
        logy=False,
        **truth_weighted_args,
        **ratio_args,
    )

    #
    # # hadron
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthTauPt',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthTauEta',
    #                       bins=(30, -5, 5), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthTauPhi',
    #                       bins=(30, -pi, pi), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthNeutrinoPt',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthNeutrinoEta',
    #                       bins=(30, -5, 5), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthNeutrinoPhi',
    #                       bins=(30, -pi, pi), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False)
    #
    # my_analysis.plot_hist('wtaunu_h_dta', 'TruthMTW',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True)
    #
    # # incoming quark flavour
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthTauPt',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthTauEta',
    #                       bins=(30, -5, 5), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthTauPhi',
    #                       bins=(30, -pi, pi), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthNeutrinoPt',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthNeutrinoEta',
    #                       bins=(30, -5, 5), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthNeutrinoPhi',
    #                       bins=(30, -pi, pi), weight='base_weight',
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta', 'wtaunu_CFilterBVeto_dta', 'wtaunu_BFilter_dta'], 'TruthMTW',
    #                       bins=(30, 1, 5000), weight='base_weight', ratio_axlim=1.5,
    #                       title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    # my_analysis.histogram_printout()

    #
    # # normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauPt',
    #                       bins=(30, 1, 5000), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauEta',
    #                       bins=(30, -5, 5), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauPhi',
    #                       bins=(30, -pi, pi), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoPt',
    #                       bins=(30, 1, 5000), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoEta',
    #                       bins=(30, -5, 5), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoPhi',
    #                       bins=(30, -pi, pi), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthMTW',
    #                       bins=(30, 1, 5000), weight='truth_weight',
    #                       title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)
    #
    # # RECO
    # # -----------------------------------
    # # calculate specific weights
    # my_analysis.logger.info("Calculating weights...")
    # my_analysis['wtaunu_mu_dta']['lep_reco_weight'] =    my_analysis['wtaunu_mu_dta']['reco_weight'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_recoSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_isoSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Muon_ttvaSF']
    # my_analysis['wtaunu_e_dta']['lep_reco_weight'] =     my_analysis['wtaunu_mu_dta']['reco_weight'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Ele_recoSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Ele_idSF'] \
    #                                                    * my_analysis['wtaunu_mu_dta']['Ele_isoSF']
    # my_analysis['wtaunu_mu_dta'].dropna('lep_reco_weight', drop_inf=True)
    # my_analysis['wtaunu_mu_dta'].dropna('lep_reco_weight', drop_inf=True)
    #
    # # un-normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonPt', 'ElePt'],
    #                       bins=(30, 1, 5000), weight='lep_reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonEta', 'EleEta'],
    #                       bins=(30, -5, 5), weight='lep_reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonPhi', 'ElePhi'],
    #                       bins=(30, -pi, pi), weight='lep_reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MET_met',
    #                       bins=(30, 1, 5000), weight='reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MET_phi',
    #                       bins=(30, -pi, pi), weight='reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MTW',
    #                       bins=(30, 1, 5000), weight='reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['Muon_d0sig', 'Ele_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight='lep_reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['Muon_delta_z0_sintheta', 'mu_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight='lep_reco_weight',
    #                       normalise=False, title='reco - 36.2fb$^{-1}$', **ratio_args)
    #
    # # normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonPt', 'ElePt'],
    #                       bins=(30, 1, 5000), weight='lep_reco_weight',
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonEta', 'EleEta'],
    #                       bins=(30, -5, 5), weight='lep_reco_weight',
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['MuonPhi', 'ElePhi'],
    #                       bins=(30, -pi, pi), weight='lep_reco_weight',
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MET_met',
    #                       bins=(30, 1, 5000), weight='reco_weight',
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MET_phi',
    #                       bins=(30, -pi, pi), weight='reco_weight',
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'MTW',
    #                       bins=(30, 1, 5000), weight='reco_weight',
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['Muon_d0sig', 'Ele_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], ['Muon_delta_z0_sintheta', 'Ele_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight='reco_weight',
    #                       normalise=True, title='normalised to 1', **ratio_args)

    my_analysis.logger.info("DONE.")
