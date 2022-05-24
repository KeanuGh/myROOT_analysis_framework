from numpy import pi

from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = '/mnt/D/data/DTA_outputs/'
ANALYSISTOP_PATH = '/mnt/D/data/analysistop_out/mc16a/'
DATA_OUT_DIR = '/mnt/D/data/dataset_pkl_outputs/'

if __name__ == '__main__':
    datasets = {
        # dta w->taunu->munu
        'wtaunu_l_dta': {
            'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_L*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
        },
        'wtaunu_CVetoBVeto_dta_l': {
            'data_path': DTA_PATH + '/*_L_*CVetoBVeto*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'force_rebuild': False,
            'validate_duplicated_events': False,
            'label': r'CVetoBVeto',
        },
        'wtaunu_CFilterBVeto_dta_l': {
            'data_path': DTA_PATH + '/*_L_*CFilterBVeto*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'force_rebuild': False,
            'validate_duplicated_events': False,
            'label': r'CFilterBVeto',
        },
        'wtaunu_BFilter_dta_l': {
            'data_path': DTA_PATH + '/*_L_*BFilter*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'force_rebuild': False,
            'validate_duplicated_events': False,
            'label': r'BFilter',
        },
    }

    my_analysis = Analysis(datasets, data_dir=DATA_OUT_DIR, year='2015+2016', TTree_name='T_s1tlv_NOMINAL',
                           analysis_label='dta_incoming_quark_study_L', skip_verify_pkl=False, lepton='tau',
                           dataset_type='dta', log_level=10, log_out='both', timedatelog=True, separate_loggers=False)
    # my_analysis.merge_datasets('wtaunu_e_dta', 'wtaunu_e_dta_peak')

    my_analysis.apply_cuts(truth=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # ratio plot arguments
    ratio_args = {
        # 'ratio_axlim': 1.5,
        'stats_box': True,
        'ratio_fit': True
    }

    my_analysis['wtaunu_l_dta'].plot_dsid('TruthMTW', weight='truth_weight', bins=(30, 1, 5000), logx=True, logy=True, logbins=True)

    # TRUTH
    # -----------------------------------
    # unnormalised
    # incoming quark flavour
    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthTauPt',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthTauEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthTauPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthNeutrinoPt',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthNeutrinoEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthNeutrinoPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_l', 'wtaunu_CFilterBVeto_dta_l', 'wtaunu_BFilter_dta_l'], 'TruthMTW',
                          bins=(30, 1, 5000), weight='truth_weight', ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.logger.info("DONE.")
