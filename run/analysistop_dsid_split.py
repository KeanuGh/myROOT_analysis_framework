import pathlib

import numpy as np

from analysis import Analysis

ANALYSISTOP_PATH = pathlib.Path('/mnt/D/data/analysistop_out/mc16a/')
DATA_OUT_DIR = '/mnt/D/data/dataset_pkl_outputs/'
bins = np.array([130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022, 302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246, 705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072, 1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055, 3832.763, 4139.151, 4470.031, 4827.361, 5213.257])

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
    force_rebuild=False,
    TTree_name='truth',
    dataset_type='analysistop',
    # log_level=10,
    data_dir=DATA_OUT_DIR,
    log_out='both',
    lepton='tau',
    cutfile_path='../options/DTA_cuts/analysistop.txt',
    validate_duplicated_events=False,
    force_recalc_weights=False,
)

for ds in datasets:
    my_analysis.plot_hist(ds, 'MC_WZmu_el_pt_born',  bins=bins, weight='truth_weight', logx=True, stats_box=True)
    my_analysis.plot_hist(ds, 'MC_WZmu_el_eta_born', bins=(30, -5, 5),   weight='truth_weight', stats_box=True)
    my_analysis.plot_hist(ds, 'MC_WZ_dilep_m_born',  bins=bins, weight='truth_weight', logx=True, stats_box=True)
    my_analysis.plot_hist(ds, 'mt_born',             bins=bins, weight='truth_weight', logx=True, stats_box=True)

# merge all wmin
wmin_strs = [s for s in datasets.keys() if ('wmin' in s) and s != 'wmintaunu_analysistop']
print(f'{wmin_strs =}')

my_analysis.merge_datasets('wmintaunu_analysistop', *wmin_strs)

my_analysis.plot_hist('wmintaunu_analysistop', 'MC_WZ_dilep_m_born',
                      bins=bins, weight='truth_weight', logx=True, stats_box=True, filename_suffix='TOTAL')

my_analysis.logger.info("DONE")
