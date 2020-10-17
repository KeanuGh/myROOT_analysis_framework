import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import pandas as pd

# project imports
from utils.cutfile_parser import parse_cutfile, gen_cutgroups, compare_backup
from utils.pandas_utils import build_analysis_dataframe
from utils.axis_labels import labels_xs
from utils.dataframe_utils import gen_weight_column, rescale_to_GeV
from utils.plotting_tools import scale_to_crosssection

# for file manipulation
from shutil import copyfile
from time import strftime
from datetime import datetime
import os


# ===========================
# ========= SETUP ===========
# ===========================
# set ATLAS style plots
plt.style.use([hep.style.ATLAS,
               {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
               ])

# options
TTree_name = 'truth'  # name of TTree to extract from root file
cut_label = ' CUT'  # label to use for boolean cut columns in dataframe
printouts = True  # printout variables info

# filepaths
input_root_file = '../data/wminmunu_MC.root'
cutfile = '../options/cutfile.txt'
out_dir = '../outputs/'  # where outputs go
out_plots_dir = out_dir + 'plots/'  # where plots go
pkl_df_filepath = out_dir + TTree_name + '_df.pkl'  # pickle file containing extracted data as pandas dataframe
# pkl_hist_filepath = out_dir + "histograms.pkl"  # pickle file to place histograms into
backup_dir = '../analysis_save_state/'  # where backups go
backup_cutfiles_dir = backup_dir + 'cutfiles/'  # cutfile backups
latex_table_dir = out_dir + "LaTeX_cutflow_table/"  # where to print latex cutflow table

# multithreading
n_threads = os.cpu_count() // 2


# ============================
# ======  READ CUTFILE =======
# ============================
# TODO: split these into separate functions/methods (tuple outputs are confusing)
# parse cutfile
cut_dicts, vars_to_cut, options = parse_cutfile(cutfile)

# check if cutfile backups exist
build_dataframe, make_backup = compare_backup(cutfile, backup_cutfiles_dir, pkl_df_filepath)


# ===============================
# ==== EXTRACT & CLEAN DATA =====
# ===============================
if build_dataframe:
    tree_df = build_analysis_dataframe(cut_dicts, vars_to_cut,
                                       input_root_file, TTree_name, pkl_df_filepath)
else:
    tree_df = pd.read_pickle(pkl_df_filepath)

# extract cutgroups
cutgroups = gen_cutgroups(cut_dicts)

# map weights column
tree_df['weight'] = gen_weight_column(tree_df)

# rescale MeV columns to GeV
rescale_to_GeV(tree_df, inplace=True)

# TODO: Check that variables being extracted are contained in the axis labels dictionary, or some weird behaviour
#  will occur.


# ===============================
# ======= APPLYING CUTS =========
# ===============================
print("applying cuts...")
for cut in cut_dicts:
    if not cut['is_symmetric']:
        if cut['moreless'] == '>':
            tree_df[cut['name'] + cut_label] = tree_df[cut['cut_var']] > cut['cut_val']
        elif cut['moreless'] == '<':
            tree_df[cut['name'] + cut_label] = tree_df[cut['cut_var']] < cut['cut_val']
        else:
            raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")

    else:
        # take absolute value instead
        if cut['moreless'] == '>':
            tree_df[cut['name'] + cut_label] = tree_df[cut['cut_var']].abs() > cut['cut_val']
        elif cut['moreless'] == '<':
            tree_df[cut['name'] + cut_label] = tree_df[cut['cut_var']].abs() < cut['cut_val']
        else:
            raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")


# ===============================
# ==== CALCULATING LUMI & XS ====
# =============================
n_events_tot = len(tree_df.index)
cross_section = tree_df["weight_mc"].abs().sum() / n_events_tot
lumi = tree_df['weight'].sum() / cross_section


# ===============================
# =========== PLOTS =============
# ===============================
# copy full dataframe. Have it reduce rows for each cut loop to form cutflow
cutflow_df = tree_df.copy()

n_bins = 30
binrange = (1, 500)
eta_binrange = (-20, 20)

# any of the substrings in this list shouldn't be binned logarithmically (may need to double check this)
not_log = [
    'phi',
    'eta',
]

for var_to_cut in vars_to_cut:
    # fill (threaded)
    # WARNING: check whether inputs contain any of the substrings in because they are binned differently

    # whether or not bins should be logarithmic bins
    is_logbins = not any(map(var_to_cut.__contains__, not_log))

    # INCLUSIVE PLOT
    # setup inclusive histogram
    if is_logbins:
        hist = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log))
    else:
        hist = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange))

    # fill
    print(f"Filling histogram for {var_to_cut}...")
    hist.fill(tree_df[var_to_cut], threads=n_threads)

    # rescale for cross-sections
    scale_to_crosssection(hist, luminosity=lumi)

    # plot
    hep.histplot(hist, label='Inclusive')

    # PLOT CUTS
    for cutgroup in cutgroups.keys():
        print(f"    - filling cutgroup '{cutgroup}'")
        # get column names for boolean columns in dataframe containing the cuts
        cut_rows = [cut_name+cut_label for cut_name in cutgroups[cutgroup]]

        # setup
        if is_logbins:
            cut_hist = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log))
        else:
            cut_hist = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange))

        # fill
        cut_hist.fill(tree_df[tree_df[cut_rows].any(1)][var_to_cut], threads=n_threads)

        # scale
        scale_to_crosssection(cut_hist, luminosity=lumi)

        # plot
        hep.histplot(cut_hist, label=cutgroup)

    # log y axis, unless plotting Bjorken X
    if 'PDFinfo_X' not in var_to_cut:
        plt.semilogy()

    # apply axis labels
    if var_to_cut not in labels_xs:
        raise ValueError(f"Axis labels for {var_to_cut} not found in in label lookup dictionary")
    plt.xlabel(labels_xs[var_to_cut]['xlabel'])
    plt.ylabel(labels_xs[var_to_cut]['ylabel'])
    plt.legend()

    # TODO: ACCEPTANCE PLOT
    # TODO: CUTFLOW HISTOGRAM

    # save figure
    hep.atlas.label(data=False, paper=False, year=datetime.now().year)
    out_png_file = out_plots_dir + f"{var_to_cut}_XS.png"
    plt.savefig(out_png_file)
    print(f"Figure saved to {out_png_file}")

    # clear for next plot
    plt.clf()


# ===============================
# ========= PRINTOUTS ===========
# ===============================
if printouts:
    # print cutflow output
    name_len = max([len(cut['name']) for cut in cut_dicts])
    var_len = max([len(cut['cut_var']) for cut in cut_dicts])
    print(f"\n========== CUTS USED ============")
    for cut in cut_dicts:
        if not cut['is_symmetric']:
            print(f"{cut['name']:<{name_len}}: "
                  f"{cut['cut_var']:>{var_len}} {cut['moreless']} {cut['cut_val']}")
        else:
            print(f"{cut['name']:<{name_len}}: "
                  f"{cut['cut_var']:>{var_len}} {cut['moreless']} |{cut['cut_val']}|")

    # cutflow printout
    print(f"\n=========== CUTFLOW =============")
    print("Cut " + " " * (name_len - 3) +
          "Events " + " " * (len(str(n_events_tot)) - 6) +
          "Ratio Cum. Ratio")
    # first line is inclusive sample
    print("Inclusive " + " " * (name_len - 9) + f"{n_events_tot} -     -")

    # perform cutflow and print
    # TODO: PERFORM CUTFLOW WHEN FILLING CUTFLOW HISTOGRAM, NOT WHEN DOING PRINTOUTS
    prev_n = n_events_tot
    for cut in cut_dicts:
        # reduce df each time for sequential ratio
        cutflow_df = cutflow_df[cutflow_df[cut['name'] + cut_label]]

        # calculations
        n_events_cut = len(cutflow_df.index)
        ratio = n_events_cut / prev_n
        cum_ratio = n_events_cut / n_events_tot

        # print line
        print(f"{cut['name']:<{name_len}} "
              f"{n_events_cut:<{len(str(n_events_tot))}} "
              f"{ratio:.3f} "
              f"{cum_ratio:.3f}")
        prev_n = n_events_cut

    # kinematics printout
    print(f"\n========== KINEMATICS ===========\n"
          f"cross-section: {cross_section:.2f} fb\n"
          f"luminosity   : {lumi:.2f} fb-1\n"
          )

    # TODO: print to LaTeX file (do cutflow histogram first)
    # # To LaTeX (only if the cutflow has actually changed or directory is empty)
    # if make_backup or os.listdir(latex_table_dir) == 0:
    #     latex_filepath = latex_table_dir + "cutflow_" + strftime("%Y-%m-%d_%H-%M-%S") + ".tex"
    #     with open(latex_filepath, "w") as f:
    #         ...
    #     print(f"Saved LaTeX cutflow table in {latex_filepath}")

# if new cutfile, save backup
if make_backup:
    cutfile_backup_filepath = backup_cutfiles_dir + "cutfile_" + strftime("%Y-%m-%d_%H-%M-%S")
    copyfile(cutfile, cutfile_backup_filepath)
    print(f"Backup cutfile saved in {cutfile_backup_filepath}")
