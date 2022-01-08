"""
Contains default global variables for analysis and variables to be set at runtime
"""
import os

# OS SETTINGS
# ====================
n_threads = os.cpu_count()  # number of threads used for filling histograms

# MISC SETTINGS
# ====================
cut_label = ' CUT'  # label to add to boolean cut columns in dataframe
