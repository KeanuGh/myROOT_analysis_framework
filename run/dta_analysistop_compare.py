from numpy import pi

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = '/data/DTA_outputs/2022-08-24/'
ANALYSISTOP_PATH = '/data/analysistop_out/mc16a/'
DATA_OUT_DIR = '/data/dataset_pkl_outputs/'


if __name__ == '__main__':
    datasets = {
        # dta w->taunu->munu
        'wtaunu_dta': {
            'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            # 'TTree_name': 'T_s1thv_NOMINAL',
            'TTree_name': {'T_s1thv_NOMINAL', 'T_s1tev_NOMINAL', 'T_s1tmv_NOMINAL'},
            # 'hard_cut': 'Muonic Tau',
            'lepton': 'tau',
            'dataset_type': 'dta',
            # 'force_rebuild': True,
            # 'force_recalc_weights': True,
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu$',
        },
        # 'wtaunu_h_dta': {
        #     'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_H*/*.root',
        #     'cutfile_path': '../options/DTA_cuts/dta_init.txt',
        #     'TTree_name': 'T_s1thv_NOMINAL',
        #     # 'hard_cut': 'Muonic Tau',
        #     'lepton': 'tau',
        #     'dataset_type': 'dta',
        #     'force_rebuild': True,
        #     # 'force_recalc_weights': True,
        #     'validate_duplicated_events': False,
        #     'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow h\nu$',
        # },
        # 'wtaunu_e_dta': {
        #     'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
        #     'cutfile_path': '../options/DTA_cuts/dta_init.txt',
        #     'TTree_name': 'T_s1tev_NOMINAL',
        #     # 'hard_cut': 'Muonic Tau',
        #     'lepton': 'tau',
        #     'dataset_type': 'dta',
        #     'force_rebuild': True,
        #     # 'force_recalc_weights': True,
        #     'validate_duplicated_events': False,
        #     'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$',
        # },
        # 'wtaunu_mu_dta': {
        #     'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
        #     'cutfile_path': '../options/DTA_cuts/dta_init.txt',
        #     'TTree_name': 'T_s1tmv_NOMINAL',
        #     # 'hard_cut': 'Muonic Tau',
        #     'lepton': 'tau',
        #     'dataset_type': 'dta',
        #     'force_rebuild': True,
        #     # 'force_recalc_weights': True,
        #     'validate_duplicated_events': False,
        #     'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        # },
        # analysistop w->taunu->munu
        'wtaunu_analysistop': {
            'data_path': ANALYSISTOP_PATH + '/w*taunu_*/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            # 'force_rebuild': False,
            'label': r'Powheg $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
        'wtaunu_analysistop_peak': {
            'data_path': ANALYSISTOP_PATH + '/w*taunu/*.root',
            'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
            'lepton': 'tau',
            'dataset_type': 'analysistop',
            'hard_cut': 'M_W',
            # 'force_rebuild': True,
            'label': r'Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year='2015+2016',
        force_rebuild=True,
        analysis_label='dta_analysistop_compare',
        skip_verify_pkl=False,
        # force_recalc_cuts=True,
        # log_level=10,
        log_out='both',
    )
    # my_analysis.merge_datasets('wtaunu_mu_dta', 'wtaunu_e_dta', verify=True)
    my_analysis.merge_datasets('wtaunu_analysistop', 'wtaunu_analysistop_peak')

    # BR-scaled weight
    my_analysis['wtaunu_dta'].df['br_scaled_weight'] = my_analysis['wtaunu_dta']['truth_weight'] / my_analysis.global_lumi
    # my_analysis['wtaunu_analysistop'].df['br_scaled_weight'] = my_analysis['wtaunu_analysistop']['truth_weight'] * 0.1138 / my_analysis.global_lumi
    my_analysis['wtaunu_analysistop'].df['br_scaled_weight'] = my_analysis['wtaunu_analysistop']['truth_weight'] / my_analysis.global_lumi

    # HISTORGRAMS
    # ==================================================================================================================
    # look at monte carlo weights
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -5, 5), filename_suffix='_5', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_analysistop', 'weight_mc', bins=(100, -10000, 10000), filename_suffix='full', yerr=None, logy=True)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -1e6, 1e6), filename_suffix='_mill', logy=True, yerr=None)
    # my_analysis.plot_hist('wtaunu_mu_dta', 'weight_mc', bins=(50, -5e9, 5e9), logy=True, filename_suffix='_bill', yerr=None)

    # mass_bins = np.array(
    #     [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
    #      302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
    #      705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
    #      1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
    #      3832.763, 4139.151, 4470.031, 4827.361, 5213.257])

    # argument dicts
    ratio_args = {
        # 'ratio_axlim': 1.5,
        'ratio_label': 'Powheg/Sherpa',
        'stats_box': False,
        'ratio_fit': True,
    }
    mass_args = {
        'bins': (30, 1, 5000),
        'logbins': True,
        'logx': True,
        'ratio_axlim': 10,
    }
    unweighted_args = {
        'ylabel': 'Entries',
        'weight': 1,
        'name_prefix': 'unweighted',
        'title': 'truth - unweighted',
        'normalise': False,
    }
    weighted_args = {
        'weight': 'truth_weight',
        'name_prefix': 'weighted',
        'title': 'truth - 36.2fb$^{-1}$',
        'normalise': False,
    }
    bin_scaled_args = {
        'weight': 'truth_weight',
        'name_prefix': 'bin_scaled',
        'title': 'truth - 36.2fb$^{-1}$',
        'normalise': False,
        'scale_by_bin_width': True,
    }
    normed_args = {
        'ylabel': 'Normalised Entries',
        'weight': 'truth_weight',
        'name_prefix': 'normalised',
        'title': 'truth - normalised to unity',
        'normalise': True,
        'scale_by_bin_width': True,
    }
    br_weighted_args = {
        'weight': 'br_scaled_weight',
        'name_prefix': 'br_weighted',
        'title': 'truth - 36.2fb$^{-1}$',
        'normalise': False,
    }
    bin_br_scaled_args = {
        'weight': ['br_scaled_weight', 'truth_weight'],
        'name_prefix': 'bin_br_scaled',
        'title': 'truth - 36.2fb$^{-1}$',
        'normalise': False,
        'scale_by_bin_width': True,
    }

    # TRUTH
    # -----------------------------------
    for arg_dict in (
            # unweighted_args,
            # weighted_args,
            # bin_scaled_args,
            br_weighted_args,
            # bin_br_scaled_args,
            # normed_args
    ):
        # my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthBosonM', 'MC_WZ_dilep_m_born'],
        #                       **mass_args, **arg_dict, **ratio_args)
        my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
                              **mass_args, **arg_dict, **ratio_args)
        my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
                              bins=(30, -5, 5), **arg_dict, **ratio_args)
        my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
                              bins=(30, -pi, pi), **arg_dict, **ratio_args)
        # my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPt', 'MC_WZneutrino_pt_born'],
        #                       **mass_args, **arg_dict, **ratio_args)
        # my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoEta', 'MC_WZneutrino_eta_born'],
        #                       bins=(30, -5, 5), **arg_dict, **ratio_args)
        # my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthNeutrinoPhi', 'MC_WZneutrino_phi_born'],
        #                       bins=(30, -pi, pi), **arg_dict, **ratio_args)
        my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
                              **mass_args, **arg_dict, **ratio_args)

    my_analysis['wtaunu_dta'].dsid_metadata_printout()
    my_analysis['wtaunu_analysistop'].dsid_metadata_printout()
    my_analysis.histogram_printout()

    # # apply muon cut
    # my_analysis.apply_cuts(datasets='wtaunu_dta', labels='Muonic Tau')
    #
    # # replot
    # for arg_dict in (
    #         weighted_args,
    #         bin_scaled_args,
    # ):
    #     arg_dict['name_prefix'] = 'muon_cut_' + arg_dict['name_prefix']
    #     my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPt', 'MC_WZmu_el_pt_born'],
    #                           **mass_args, **arg_dict, **ratio_args)
    #     my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauEta', 'MC_WZmu_el_eta_born'],
    #                           bins=(30, -5, 5), **arg_dict, **ratio_args)
    #     my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthTauPhi', 'MC_WZmu_el_phi_born'],
    #                           bins=(30, -pi, pi), **arg_dict, **ratio_args)
    #     my_analysis.plot_hist(['wtaunu_dta', 'wtaunu_analysistop'], ['TruthMTW', 'mt_born'],
    #                           **mass_args, **arg_dict, **ratio_args)
    #
    # my_analysis['wtaunu_dta'].dsid_metadata_printout()
    # my_analysis['wtaunu_analysistop'].dsid_metadata_printout()

    my_analysis.logger.info("DONE.")
