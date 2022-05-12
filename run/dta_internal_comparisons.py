from numpy import pi

from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        # dta w->taunu->munu
        'wtaunu_mu_dta': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_tau_to_muons.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'hard_cut': 'Muonic Tau',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'force_rebuild': False,
            'validate_duplicated_events': False,
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
        'wtaunu_e_dta': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_tau_to_electrons.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'hard_cut': 'Electronic Tau',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'force_rebuild': False,
            'validate_duplicated_events': False,
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$',
        },
    }

    my_analysis = Analysis(datasets, data_dir='/mnt/D/data/dataset_pkl_outputs/', year='2015+2016',
                           analysis_label='dta_internal_comparisons', skip_verify_pkl=False,
                           log_level=10, log_out='both', timedatelog=True, separate_loggers=False)
    # my_analysis.merge_datasets('wtaunu_e_dta', 'wtaunu_e_dta_peak')

    BR = len(my_analysis['wtaunu_mu_dta']) / len(my_analysis['wtaunu_e_dta'])
    my_analysis.logger.info("BRANCHING RATIO tau->munu / tau->enu: ", BR)

    my_analysis.apply_cuts(truth=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # ratio plot arguments
    ratio_args = {
        # 'ratio_axlim': 1.5,
        'stats_box': True,
        'ratio_fit': True
    }

    # TRUTH
    # -----------------------------------
    # unnormalised
    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauPt',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthTauPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoPt',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthNeutrinoPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_e_dta'], 'TruthMTW',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)
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
