from utils.cutfile_parser import parse_cutfile
import uproot4 as uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import boost_histogram as bh
import mplhep as hep

# SETUP
plt.style.use([hep.style.ATLAS])
filename = '../data/wminmunu_MC.root'
out_path = '../outputs/'
cutfile = '../options/cutfile.txt'

# extract
truth = uproot.open(filename)["truth"]

# parse cutfile
cuts, vars_to_cut, options = parse_cutfile(cutfile)
# strictly necessary variables
vars_to_cut.append('weight_mc')

# into pandas
truth_df = truth.arrays(library='pd', filter_name=vars_to_cut)

# map weight column
truth_df['weight'] = truth_df['weight_mc'].map(lambda w: 1 if w > 0 else -1)

