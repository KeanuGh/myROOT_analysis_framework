"""
Contains default global variables for analysis
"""
from math import pi
import os

# OS SETTINGS
# ====================
n_threads = os.cpu_count() // 2  # number of threads used for filling histograms

# MISC SETTINGS
# ====================
lepton = 'lepton'  # name of lepton
lumi = 140.  # global luminosity to rescale to

# DEFAULT BINNINGS
# ====================
phibins = (20, -2 * pi, 2 * pi)
etabins = (20, -10, 10)

# FILEPATHS
# ====================
out_dir = '../../outputs/'  # where outputs go
out_plots_dir = out_dir + 'plots/'  # where plots go
pkl_df_filepath = out_dir + 'data/{}_df.pkl'  # pickle file containing extracted data, format to used TTree
# pkl_hist_filepath = out_dir + "histograms.pkl"  # pickle file to place histograms into
backup_dir = '../../analysis_save_state/'  # where backups go
backup_cutfiles_dir = backup_dir + 'cutfiles/'  # _cutfile backups
latex_table_dir = out_dir + "LaTeX_cutflow_table/"  # where to print latex cutflow table
