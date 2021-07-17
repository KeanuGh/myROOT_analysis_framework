"""
Contains default global variables for analysis and variables to be set at runtime
"""
import os
from math import pi

# OS SETTINGS
# ====================
n_threads = os.cpu_count() // 2  # number of threads used for filling histograms

# MISC SETTINGS
# ====================
force_rebuild = False  # whether to force the rebuilding of dataframes
cut_label = ' CUT'  # label to add to boolean cut columns in dataframe
lumi = 140.  # global luminosity to rescale to

# DEFAULT BINNINGS
# ====================
not_log = [  # variables containing these substrings will never be plotted with log_x
    '_phi_',
    '_eta_',
]
phibins = (20, -pi, pi)
etabins = (20, -10, 10)

special_binning = {
    '_eta_': etabins,
    '_phi_': phibins,
    'w_y': etabins
}

# FILEPATHS
# ====================
out_dir = '../outputs/'  # where outputs go
plot_dir = out_dir + '{}/plots/'  # where plots go
pkl_df_filepath = out_dir + '{}/data/'  # pickle file containing extracted data, format to used dataset
pkl_hist_dir = out_dir + "{}/histograms/"  # pickle file to place histograms into
backup_cutfiles_dir = out_dir + '{}/cutfiles/'  # _cutfile backups
latex_table_dir = out_dir + "{}/LaTeX_cutflow_table/"  # where to print latex cutflow table
