import numpy as np
from tabulate import tabulate

from src.analysis import Analysis
from utils import PMG_tool, ROOT_utils

bins = np.array(
    [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
     302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
     705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
     1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
     3832.763, 4139.151, 4470.031, 4827.361, 5213.257])

DTA_PATH = '/data/DTA_outputs/2022-08-24/'
DATA_OUT_DIR = '/data/dataset_pkl_outputs/'
datasets = {
    'wtaunu_l_dta_cvetobveto':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto*_histograms.root/*.root',
        'TTree_name':['T_s1tmv_NOMINAL', 'T_s1tev_NOMINAL'],
        'label':r'C Veto B Veto',
    },
    'wtaunu_l_dta_cfilterbveto':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_CFilterBVeto*_histograms.root/*.root',
        'TTree_name':['T_s1tmv_NOMINAL', 'T_s1tev_NOMINAL'],
        'label':r'C Filter B Veto',
    },
    'wtaunu_l_dta_bfilter':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_BFilter*_histograms.root/*.root',
        'TTree_name':['T_s1tmv_NOMINAL', 'T_s1tev_NOMINAL'],
        'label':r'B Filter',
    },
    'wtaunu_h_dta_cvetobveto':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto*_histograms.root/*.root',
        'TTree_name':'T_s1thv_NOMINAL',
        'label':r'C Veto B Veto',
    },
    'wtaunu_h_dta_cfilterbveto':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_CFilterBVeto*_histograms.root/*.root',
        'TTree_name':'T_s1thv_NOMINAL',
        'label':r'C Filter B Veto',
    },
    'wtaunu_h_dta_bfilter':{
        'data_path':DTA_PATH + 'user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_BFilter*_histograms.root/*.root',
        'TTree_name':'T_s1thv_NOMINAL',
        'label':r'B Filter',
    },
}

my_analysis = Analysis(
    datasets,
    analysis_label='dta_analysis',
    # force_rebuild=True,
    dataset_type='dta',
    # log_level=10,
    data_dir=DATA_OUT_DIR,
    log_out='both',
    lepton='tau',
    cutfile_path='../options/DTA_cuts/dta_init.txt',
    # force_recalc_weights=True,
)

for wgt_ver in (1, 2):
    base_no_filteff = []
    base_metadata = []
    lumi_1_metadata = []
    reg_metadata = []
    bin_metadata = []

    print("Using histogram sumw" if wgt_ver == 1 else "Using manual sumw:")
    for ds in datasets:
        dsid = my_analysis[ds].df.index[0][0]
        my_analysis[ds].label += '_' + str(dsid)  # set label to include DSID
        xs = PMG_tool.get_crossSection(dsid)
        filt_eff = PMG_tool.get_genFiltEff(dsid)
        kFactor = PMG_tool.get_kFactor(dsid)

        if wgt_ver == 1:
            sumw = ROOT_utils.get_dta_sumw(datasets[ds]['data_path'])
        else:
            sumw = my_analysis[ds]['weight_mc'].sum()

        # calculate new weights
        print("calculating weights...")
        my_analysis[ds]['base_no_filteff_wtg'] = my_analysis[ds]['weight_mc'] * xs / sumw
        my_analysis[ds]['base_wtg'] = my_analysis[ds]['weight_mc'] * xs * filt_eff / sumw
        my_analysis[ds]['truth_weight_lumi1'] = my_analysis[ds]['weight_mc'] * xs * filt_eff * kFactor / sumw
        my_analysis[ds]['truth_weight'] = my_analysis[ds]['weight_mc'] * my_analysis[
            ds].lumi * xs * filt_eff * kFactor / sumw

        h = my_analysis.plot_hist(ds, 'TruthMTW', bins=bins, weight='base_no_filteff_wtg', stats_box=True,
                                  name_prefix='base_no_filteff')
        base_no_filteff.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, filt_eff, sumw,
                                my_analysis[ds]['base_no_filteff_wtg'].mean(), xs])

        h = my_analysis.plot_hist(ds, 'TruthMTW', bins=bins, weight='base_wtg', stats_box=True,
                                  name_prefix='base')
        base_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, filt_eff, sumw,
                              my_analysis[ds]['base_wtg'].mean(), xs])

        h = my_analysis.plot_hist(ds, 'TruthMTW', bins=bins, weight='truth_weight_lumi1', stats_box=True,
                                  name_prefix='lumi1')
        lumi_1_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, filt_eff, sumw,
                                my_analysis[ds]['truth_weight_lumi1'].mean(), xs])

        h = my_analysis.plot_hist(ds, 'TruthMTW', bins=bins, weight='truth_weight', stats_box=True, name_prefix='truth')
        reg_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, filt_eff, sumw,
                             my_analysis[ds]['truth_weight'].mean(), xs])

        h = my_analysis.plot_hist(ds, 'TruthMTW', bins=bins, weight='truth_weight', stats_box=True,
                                  scale_by_bin_width=True, name_prefix='bin_scaled')
        bin_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, filt_eff, sumw,
                             my_analysis[ds]['truth_weight'].mean(), xs])

    # sort
    base_no_filteff.sort(key=lambda row:row[0])
    base_metadata.sort(key=lambda row:row[0])
    lumi_1_metadata.sort(key=lambda row:row[0])
    reg_metadata.sort(key=lambda row:row[0])
    bin_metadata.sort(key=lambda row:row[0])

    my_analysis.save_histograms()
    headers = ['DSID', 'Name', 'Entries', 'Bin sum', 'Integral', 'Filt. eff.', 'sum wgt.', 'avg. wgt.',
               'PMG cross-section']
    my_analysis.logger.info("base no filter efficiency:")
    my_analysis.logger.info(tabulate(base_no_filteff, headers=headers))
    my_analysis.logger.info("Base:")
    my_analysis.logger.info(tabulate(base_metadata, headers=headers))
    my_analysis.logger.info("with kfactor and lumi 1:")
    my_analysis.logger.info(tabulate(lumi_1_metadata, headers=headers))
    my_analysis.logger.info("lumi data:")
    my_analysis.logger.info(tabulate(reg_metadata, headers=headers))
    my_analysis.logger.info("Bin-scaled:")
    my_analysis.logger.info(tabulate(bin_metadata, headers=headers))

my_analysis.logger.info("DONE")
