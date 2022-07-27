import pathlib

import ROOT
import matplotlib.pyplot as plt
import numpy as np

from analysis import Analysis
from src.histogram import Histogram1D
from utils.plotting_utils import set_axis_options

ANALYSISTOP_PATH = pathlib.Path('/data/analysistop_out/mc16a')
DATA_OUT_DIR = '/data/dataset_pkl_outputs/'
bins = np.array(
    [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
     302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
     705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
     1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
     3832.763, 4139.151, 4470.031, 4827.361, 5213.257])
BRANCH = "MC_WZ_dilep_m_born"

# create datasets for each directory (per dsid) in path
datasets = dict()
dataset_dirs = [d for d in ANALYSISTOP_PATH.iterdir() if d.is_dir()]
for dataset_dir in dataset_dirs:
    datasets[str(dataset_dir.name) + '_analysistop'] = {
        'data_path': dataset_dir / '*.root',
        'label': dataset_dir.name
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
    log_out='both',
    lepton='tau',
    cutfile_path='../options/DTA_cuts/analysistop.txt',
    validate_duplicated_events=False,
    # force_recalc_weights=True,
)

for ds in datasets:
    my_analysis.plot_hist(ds, 'MC_WZ_dilep_m_born',  bins=bins, weight='truth_weight', logx=True, stats_box=True)
my_analysis.save_histograms()

# merge all wmin
my_analysis['wmintaunu_analysistop'].dsid_metadata_printout()
wmin_strs = [s for s in datasets.keys() if
             ('wmin' in s) and (s != 'wmintaunu_analysistop')]

my_analysis.merge_datasets('wmintaunu_analysistop', *wmin_strs)
my_analysis['wmintaunu_analysistop'].dsid_metadata_printout()

# import jesal histogram
h_jesal_root = ROOT.TFile("../wmintaunu_wminus_Total.root").Get("h_WZ_dilep_m_born")
h_jesal = Histogram1D(th1=h_jesal_root, logger=my_analysis.logger)

h = Histogram1D(var=my_analysis['wmintaunu_analysistop'][BRANCH], bins=bins, logger=my_analysis.logger,
                weight=my_analysis['wmintaunu_analysistop']['truth_weight'])

# plot
# ========================
fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0)
set_axis_options(ax, BRANCH, bins, 'tau', logx=True, logy=True, diff_xs=True)
set_axis_options(axis=ratio_ax, var_name=BRANCH, bins=bins, lepton='tau',
                 xlabel=r'Born $m_{ll}$', ylabel='Ratio', title='', logx=True, logy=False, label=False)
ax.set_xticklabels([])
ax.set_xlabel('')

h_jesal.plot(ax=ax, stats_box=True, label='jesal')
h.plot(ax=ax, stats_box=True, label='me')
ax.legend(fontsize=10, loc='upper right')
h.plot_ratio(h_jesal, ax=ratio_ax, fit=True)
plt.show()

my_analysis.logger.info("DONE")
