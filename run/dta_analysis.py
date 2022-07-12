from src.analysis import Analysis

DTA_PATH = '/mnt/D/data/DTA_outputs/2022-06-08/'
DATA_OUT_DIR = '/mnt/D/data/dataset_pkl_outputs/'
datasets = {
    'wtaunu_dta_cvetobveto': {
        'data_path': DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto*_histograms.root/*.root',
        'label': r'C Veto B Veto',
    },
    'wtaunu_dta_cfilterbveto': {
        'data_path': DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto*_histograms.root/*.root',
        'label': r'C Filter B Veto',
    },
    'wtaunu_dta_bfilter': {
        'data_path': DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_BFilter*_histograms.root/*.root',
        'label': r'B Filter',
    },
}

my_analysis = Analysis(
    datasets,
    analysis_label='dta_analysis',
    force_rebuild=False,
    TTree_name='T_s1tlv_NOMINAL',
    dataset_type='dta',
    log_level=10,
    data_dir=DATA_OUT_DIR,
    log_out='both',
    lepton='tau',
    cutfile_path='../options/DTA_cuts/dta_init.txt',
    validate_duplicated_events=False,
    force_recalc_weights=True,
)

for ds in datasets:
    my_analysis.plot_hist(ds, 'TruthTauPt',  bins=(30, 1, 5000), weight='truth_weight', stats_box=True)
    my_analysis.plot_hist(ds, 'TruthTauEta', bins=(30, -5, 5),   weight='truth_weight', stats_box=True)
    my_analysis.plot_hist(ds, 'TruthBosonM', bins=(30, 1, 5000), weight='truth_weight', stats_box=True)
    my_analysis.plot_hist(ds, 'TruthMTW',    bins=(30, 1, 5000), weight='truth_weight', stats_box=True)

my_analysis.logger.info("DONE")
