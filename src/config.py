"""
Contains default global variables for analysis and variables to be set at runtime
"""
import os
from math import pi

# OS SETTINGS
# ====================
n_threads = os.cpu_count()  # number of threads used for filling histograms

# MISC SETTINGS
# ====================
cut_label = ' CUT'  # label to add to boolean cut columns in dataframe

# DEFAULT BINNINGS
# ====================
not_log = [  # variables containing these substrings will never be plotted with log_x
    '_phi_',
    '_eta_',
    'w_y',
]
phibins = (20, -pi, pi)  # default binnings for these variables
etabins = (20, -10, 10)  #

special_binning = {
    '_eta_': etabins,
    '_phi_': phibins,
    'w_y': etabins,
}
