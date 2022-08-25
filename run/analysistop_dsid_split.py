import pathlib

import numpy as np
from tabulate import tabulate

from analysis import Analysis
from utils.PMG_tool import get_crossSection

ANALYSISTOP_PATH = pathlib.Path('/data/analysistop_out/mc16a')
DATA_OUT_DIR = '/data/dataset_pkl_outputs/'
bins = np.array(
    [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
     302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
     705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
     1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
     3832.763, 4139.151, 4470.031, 4827.361, 5213.257])
# BRANCH = "MC_WZmu_el_pt_born"
BRANCH = "MC_WZ_dilep_m_born"

# create datasets for each directory (per dsid) in path
datasets = dict()
dataset_dirs = [d for d in ANALYSISTOP_PATH.iterdir() if d.is_dir()]
for dataset_dir in dataset_dirs:
    datasets[str(dataset_dir.name) + '_analysistop'] = {
        'data_path': dataset_dir / '*.root',
        'label': dataset_dir.name,
        'cutfile_path': '../options/DTA_cuts/analysistop.txt',
    }
    # add MTW cut to peak samples
    if dataset_dir.name in ('wmintaunu', 'wplustaunu'):
        datasets[str(dataset_dir.name) + '_analysistop']['cutfile_path'] = '../options/DTA_cuts/analysistop_peak.txt'
        datasets[str(dataset_dir.name) + '_analysistop']['hard cut'] = 'M_W'
datasets |= {
    # analysistop w->taunu->munu
    'wplustaunu_full_analysistop': {
        'data_path': ANALYSISTOP_PATH / 'wplustaunu_*/*.root',
        'cutfile_path': '../options/DTA_cuts/analysistop.txt',
        'label': r'Powheg $W\rightarrow\tau\nu\rightarrow \mu\nu$',
    },
    'wplustaunu_full_analysistop_peak': {
        'data_path': ANALYSISTOP_PATH / 'wplustaunu/*.root',
        'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
        'hard_cut': 'M_W',
        # 'force_rebuild': True,
        'label': r'Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
    },
    # analysistop w->taunu->munu
    'wmintaunu_full_analysistop': {
        'data_path': ANALYSISTOP_PATH / 'wmintaunu_*/*.root',
        'cutfile_path': '../options/DTA_cuts/analysistop.txt',
        # 'force_rebuild': False,
        'label': r'Powheg $W\rightarrow\tau\nu\rightarrow \mu\nu$',
    },
    'wmintaunu_full_analysistop_peak': {
        'data_path': ANALYSISTOP_PATH / 'wmintaunu/*.root',
        'cutfile_path': '../options/DTA_cuts/analysistop_peak.txt',
        'hard_cut': 'M_W',
        # 'force_rebuild': True,
        'label': r'Powheg/Pythia 8 $W\rightarrow\tau\nu\rightarrow \mu\nu$',
    },
}


my_analysis = Analysis(
    datasets,
    analysis_label='analysistop_dsid_analysis',
    # force_rebuild=True,
    TTree_name='truth',
    dataset_type='analysistop',
    lumi_year='2016+2015',
    # log_level=10,
    data_dir=DATA_OUT_DIR,
    log_out='console',
    lepton='tau',
    validate_duplicated_events=False,
    # force_recalc_weights=True,
)
my_analysis.merge_datasets('wplustaunu_full_analysistop', 'wplustaunu_full_analysistop_peak')
my_analysis.merge_datasets('wmintaunu_full_analysistop', 'wmintaunu_full_analysistop_peak')

base_metadata = []
reg_metadata = []
bin_scaled_metadata = []
min_xs_sum = 0
plus_xs_sum = 0
for ds in my_analysis.datasets:
    dsid = my_analysis[ds].df.index[0][0]
    my_analysis[ds].label += '_' + str(dsid)  # set label to include DSID

    xs = get_crossSection(dsid)
    # get total cross-section
    if ds not in ('wmintaunu', 'wplustaunu'):
        if 'min' in ds: min_xs_sum += xs
        elif 'plus' in ds: plus_xs_sum += xs

    # a new weight
    my_analysis[ds]['base_weight'] = my_analysis[ds]['weight_mc'] * abs(my_analysis[ds]['weight_mc']) / my_analysis[ds]['totalEventsWeighted']

    # base weight histogram
    h = my_analysis.plot_hist(ds, 'MC_WZ_dilep_m_born',  bins=bins, weight='base_weight', logx=True, stats_box=True,
                              name_prefix='base_weighted')
    base_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, xs])
    base_metadata.sort(key=lambda row: row[0])

    # regular histogram
    h = my_analysis.plot_hist(ds, 'MC_WZ_dilep_m_born',  bins=bins, weight='truth_weight', logx=True, stats_box=True)
    reg_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, xs])
    reg_metadata.sort(key=lambda row: row[0])

    # bin-scaled histogram
    h = my_analysis.plot_hist(ds, 'MC_WZ_dilep_m_born', bins=bins, weight='truth_weight', logx=True, stats_box=True,
                              scale_by_bin_width=True, name_prefix='bin_scaled')
    bin_scaled_metadata.append([dsid, h[0].name, h[0].n_entries, h[0].bin_sum(True), h[0].integral, xs])
    bin_scaled_metadata.sort(key=lambda row: row[0])

my_analysis.save_histograms()

# print histogram metadata
headers = ['DSID', 'Name', 'Entries', 'Bin sum', 'Integral', 'PMG cross-section']
total = ['TOTAL', '-', ]
my_analysis.logger.info("Base:")
my_analysis.logger.info(tabulate(base_metadata, headers=headers))
my_analysis.logger.info("Regular:")
my_analysis.logger.info(tabulate(reg_metadata, headers=headers))
my_analysis.logger.info("Bin-width-scaled:")
my_analysis.logger.info(tabulate(bin_scaled_metadata, headers=headers))

# compare with jesal for both plus and minus
for c in ('wmin', 'wplus'):
    ds_name = f'{c}taunu_analysistop'
    ds_name_full = f'{c}taunu_full_analysistop'

    # merge all wmin
    wc_strs = [ds for ds in datasets.keys()
               if (c in ds) and ds not in (ds_name_full, ds_name, ds_name_full + '_peak')]
    my_analysis.merge_datasets(ds_name, *wc_strs)
    my_analysis[ds_name].dsid_metadata_printout()

    print(f'{c} full length: ', len(my_analysis[ds_name_full]))
    print(f'{c} merged length: ', len(my_analysis[ds_name]))
    my_analysis[ds_name].dsid_metadata_printout()
    my_analysis[ds_name_full].dsid_metadata_printout()

    # compare with unseparated
    my_analysis.plot_hist([ds_name_full, ds_name], BRANCH, labels=[f'{c}_full', f'{c}_merged'], bins=bins, logx=True, logy=True, weight='truth_weight', ratio_fit=True, stats_box=True)

    # # import jesal histogram
    # h_jesal_root = ROOT.TFile(f"../{c}taunu_{'wminus' if c == 'wmin' else 'wplus'}_Total.root").Get("h_WZ_dilep_m_born")
    # my_analysis.logger.info(f"jesal integral: {h_jesal_root.Integral()}")
    #
    # h_jesal = Histogram1D(th1=h_jesal_root, logger=my_analysis.logger, name=f'jesal_hisogram_{c}')
    #
    # h = Histogram1D(var=my_analysis[ds_name][BRANCH], bins=bins, logger=my_analysis.logger,
    #                 weight=my_analysis[ds_name]['truth_weight'], name=f'keanu_histogram_{c}')
    # my_analysis.logger.info(f"my_integral: {h.integral}")
    # my_analysis.logger.info(f"my_integral (ROOT): {h.TH1.Integral()}")
    # my_analysis.logger.info(f"my_bin_sum (flow): {h.bin_sum(True)}")
    # my_analysis.logger.info(f"my_bin_sum (no flow): {h.bin_sum(False)}")
    # my_analysis.logger.info(f"cross-section sum: {min_xs_sum if c == 'min' else plus_xs_sum}")
    #
    # # plot
    # # ========================
    # fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.1, wspace=0)
    # ax.set_xticklabels([])
    # ax.set_xlabel('')
    # set_axis_options(ax, BRANCH, bins, 'tau', logx=True, logy=True, diff_xs=False)
    # set_axis_options(axis=ratio_ax, var_name=BRANCH, bins=bins, lepton='tau',
    #                  xlabel=r'Born $m_{ll}$', ylabel='Ratio', title='', logx=True, logy=False, label=False)
    #
    # h_jesal.plot(ax=ax, stats_box=True, label=f'jesal {c}')
    # h.plot(ax=ax, stats_box=True, label=f'me {c}')
    # ax.legend(fontsize=10, loc='upper right')
    # h.plot_ratio(h_jesal, yerr='carry', ax=ratio_ax, fit=True, yax_lim=0.05)
    # fig.savefig(f"{my_analysis.paths['plot_dir']}/{c}_compare", bbox_inches='tight')
    # plt.show()

my_analysis.logger.info("DONE")
