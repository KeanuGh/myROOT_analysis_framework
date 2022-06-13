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
        'wtaunu_h_dta': {
            'data_path': DTA_PATH + '/user.kghorban.Sh_2211_Wtaunu_H*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'label': r'Sherpa 2211 $W\rightarrow\tau\nu\rightarrow H$',
        },
        'wtaunu_CVetoBVeto_dta_h': {
            'data_path': DTA_PATH + '/*_H_*CVetoBVeto*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'label': r'CVetoBVeto',
        },
        'wtaunu_CFilterBVeto_dta_h': {
            'data_path': DTA_PATH + '/*_H_*CFilterBVeto*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'label': r'CFilterBVeto',
        },
        'wtaunu_BFilter_dta_h': {
            'data_path': DTA_PATH + '/*_H_*BFilter*/*.root',
            'cutfile_path': '../options/DTA_cuts/dta_init.txt',
            'label': r'BFilter',
        },
    }

    my_analysis = Analysis(datasets, data_dir=DATA_OUT_DIR, year='2015+2016', TTree_name='T_s1thv_NOMINAL',
                           analysis_label='dta_incoming_quark_study_H', skip_verify_pkl=False, lepton='tau',  # force_rebuild=True,
                           dataset_type='dta', log_level=10, log_out='both', force_rebuild=True)
    # my_analysis.merge_datasets('wtaunu_e_dta', 'wtaunu_e_dta_peak')

    my_analysis.apply_cuts(truth=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # ratio plot arguments
    ratio_args = {
        # 'ratio_axlim': 1.5,
        'stats_box': True,
        'ratio_fit': False
    }

    my_analysis['wtaunu_h_dta'].plot_dsid('TruthMTW', weight='truth_weight', bins=(30, 1, 5000), logx=True, logy=True, logbins=True)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthMTW',
                          bins=(30, 1, 5000), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthTauPt',
                          bins=(30, 1, 5000), weight='truth_weight', # ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthTauEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthTauPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthNeutrinoPt',
                          bins=(30, 1, 5000), weight='truth_weight', # ratio_axlim=1.5,
                          title='truth - 36.2fb$^{-1}$', normalise=False, logx=True, logbins=True, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthNeutrinoEta',
                          bins=(30, -5, 5), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.plot_hist(['wtaunu_CVetoBVeto_dta_h', 'wtaunu_CFilterBVeto_dta_h', 'wtaunu_BFilter_dta_h'], 'TruthNeutrinoPhi',
                          bins=(30, -pi, pi), weight='truth_weight',
                          title='truth - 36.2fb$^{-1}$', normalise=False, **ratio_args)

    my_analysis.logger.info("DONE.")
