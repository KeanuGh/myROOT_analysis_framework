from numpy import pi

from src.analysis import Analysis

if __name__ == '__main__':
    datasets = {
        # dta w->taunu->munu
        'wtaunu_mu_dta': {
            'data_path': '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_tau_to_muons.txt',
            'TTree_name': 'T_s1tlv_NOMINAL',
            'year': '2015+2016',
            'hard_cut': 'Muonic Tau',
            'lepton': 'tau',
            'dataset_type': 'dta',
            'label': r'Sherpa $W\rightarrow\tau\nu\rightarrow l$',
        },
        # analysistop w->taunu->munu
        'wtaunu_analysistop': {
            'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu_*/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'label': r'Powheg $W\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wtaunu_analysistop_peak': {
            'data_path': '/mnt/D/data/analysistop_out/mc16a/w*taunu/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'hard_cut': 'M_W',
            'label': r'Powheg $W\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
    }

    my_analysis = Analysis(datasets, data_dir='/mnt/D/data/dataset_pkl_outputs/',
                           analysis_label='dta_analysistop_compare', force_rebuild=False, skip_verify_pkl=False,
                           log_level=10, log_out='both', timedatelog=True, separate_loggers=False)

    # calculate specific weights
    my_analysis.logger.info("Calculating weights...")
    my_analysis['wtaunu_mu_dta']['weight_loose'] = my_analysis['wtaunu_mu_dta']['truth_weight'] * my_analysis['wtaunu_mu_dta']['TauSF_LooseWP']
    my_analysis['wtaunu_mu_dta']['muon_reco_weight'] = my_analysis['wtaunu_mu_dta']['reco_weight'] \
                                                       * my_analysis['wtaunu_mu_dta']['Muon_recoSF'] \
                                                       * my_analysis['wtaunu_mu_dta']['Muon_isoSF'] \
                                                       * my_analysis['wtaunu_mu_dta']['Muon_ttvaSF']

    # my_analysis.apply_cuts(truth=True)

    my_analysis.merge_datasets('wtaunu_analysistop', 'wtaunu_analysistop_peak')

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
    # normalised
    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='normalised to 1', normalise=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='normalised to 1', normalise=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPt', 'MC_WZneutrino_pt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoEta', 'MC_WZneutrino_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='normalised to 1', normalise=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPhi', 'MC_WZneutrino_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='normalised to 1', normalise=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='normalised to 1', normalise=True, logx=True, logbins=True, **ratio_args)

    # unnormalised
    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPt', 'MC_WZneutrino_pt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoEta', 'MC_WZneutrino_eta_born'],
                          bins=(30, -5, 5), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPhi', 'MC_WZneutrino_phi_born'],
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.logger.info("dropping null weights...")
    my_analysis['wtaunu_mu_dta'].df.dropna(subset='muon_reco_weight', inplace=True)

    # # RECO
    # # -----------------------------------
    # # normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPt', 'mu_pt'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonEta', 'mu_eta'],
    #                       bins=(30, -5, 5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPhi', 'mu_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_met', 'met_met'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_phi', 'met_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MTW', 'mu_mt_reco'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_d0sig', 'mu_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_delta_z0_sintheta', 'mu_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=True, title='normalised to 1', **ratio_args)
    #
    # # un-normalised
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPt', 'mu_pt'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonEta', 'mu_eta'],
    #                       bins=(30, -5, 5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MuonPhi', 'mu_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_met', 'met_met'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MET_phi', 'met_phi'],
    #                       bins=(30, -pi, pi), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['MTW', 'mu_mt_reco'],
    #                       bins=(30, 1, 5000), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', logx=True, logbins=True, **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_d0sig', 'mu_d0sig'],
    #                       bins=(30, -3.5, 3.5), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', **ratio_args)
    #
    # my_analysis.plot_hist(['wtaunu_mu_dta', 'wtaunu_analysistop'], ['Muon_delta_z0_sintheta', 'mu_delta_z0_sintheta'],
    #                       bins=(30, -1, 1), weight=['muon_reco_weight', 'reco_weight'],
    #                       normalise=False, title='36.2fb$^{-1}$', **ratio_args)
