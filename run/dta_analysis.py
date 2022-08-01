from tabulate import tabulate

from src.analysis import Analysis
from utils.PMG_tool import get_crossSection

DTA_PATH = '/data/DTA_outputs/2022-06-08/'
DATA_OUT_DIR = '/data/dataset_pkl_outputs/'
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
    # force_rebuild=True,
    TTree_name='T_s1tlv_NOMINAL',
    dataset_type='dta',
    log_level=10,
    data_dir=DATA_OUT_DIR,
    log_out='both',
    lepton='tau',
    cutfile_path='../options/DTA_cuts/dta_init.txt',
    validate_duplicated_events=False,
    # force_recalc_weights=True,
)

reg_metadata = []
bin_metadata = []
for ds in datasets:
    dsid = my_analysis[ds].df.index[0][0]
    my_analysis[ds].label += '_' + str(dsid)  # set label to include DSID

    # my_analysis.plot_hist(ds, 'TruthTauPt',  bins=(30, 1, 5000), weight='truth_weight', stats_box=True)
    # my_analysis.plot_hist(ds, 'TruthTauEta', bins=(30, -5, 5),   weight='truth_weight', stats_box=True)
    # my_analysis.plot_hist(ds, 'TruthBosonM', bins=(30, 1, 5000), weight='truth_weight', stats_box=True)
    h = my_analysis.plot_hist(ds, 'TruthMTW',    bins=(30, 1, 5000), weight='truth_weight', stats_box=True)
    reg_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, get_crossSection(dsid)])
    reg_metadata.sort(key=lambda row: row[0])

    h = my_analysis.plot_hist(ds, 'TruthMTW',    bins=(30, 1, 5000), weight='truth_weight', stats_box=True, scale_by_bin_width=True, name_prefix='bin_scaled')
    bin_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, get_crossSection(dsid)])
    bin_metadata.sort(key=lambda row: row[0])

my_analysis.save_histograms()
headers = ['DSID', 'Name', 'Entries', 'Bin sum', 'Integral', 'PMG cross-section']
my_analysis.logger.info("Regular:")
my_analysis.logger.info(tabulate(reg_metadata, headers=headers))
my_analysis.logger.info("Bin-scaled:")
my_analysis.logger.info(tabulate(bin_metadata, headers=headers))


my_analysis.logger.info("DONE")
