import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import pandas as pd
import numpy as np

# project imports
from utils.cutfile_parser import parse_cutfile, gen_cutgroups, compare_backup
from utils.pandas_utils import build_analysis_dataframe


# ===========================
# ========= SETUP ===========
# ===========================
# set atlas plotting style
plt.style.use([hep.style.ATLAS])
# filepaths
input_root_file = '../data/wminmunu_MC.root'
cutfile = '../options/cutfile.txt'
out_dir = '../outputs/'  # where outputs go
backup_dir = '../analysis_save_state/'  # where backups go
backup_cutfiles_dir = backup_dir + 'cutfiles/'
TTree_name = 'truth'  # name of TTree to extract from
pkl_filepath = out_dir + TTree_name + '_df.plk'  # pickle file containing extracted data as pandas dataframe
# printout variables info
printouts = True

# ============================
# ======  READ CUTFILE =======
# ============================
# parse cutfile
cut_list_dicts, vars_to_cut, options = parse_cutfile(cutfile)

# check if cutfile backups exist
build_dataframe, make_backup = compare_backup(cutfile, backup_cutfiles_dir, pkl_filepath)


# ===========================
# ====== EXTRACT DATA =======
# ===========================
if build_dataframe:
    tree_df = build_analysis_dataframe(cut_list_dicts, vars_to_cut,
                                       input_root_file, TTree_name, pkl_filepath)
else:
    tree_df = pd.read_pickle(pkl_filepath)

# extract cutgroups
cutgroups = gen_cutgroups(cut_list_dicts)


# ===========================
# ====== SET WEIGHTS ========
# ===========================
# map weights column
tree_df['weight'] = tree_df['weight_mc'].map(lambda w: 1 if w > 0 else -1)


# ===========================
# ===== APPLYING CUTS =======
# ===========================
print("applying cuts...")
for cut in cut_list_dicts:
    if not cut['is_symmetric']:
        if cut['moreless'] == '>':
            tree_df[cut['name']+' CUT'] = tree_df[cut['cut_var']] > cut['cut_val']
        elif cut['moreless'] == '<':
            tree_df[cut['name']+' CUT'] = tree_df[cut['cut_var']] < cut['cut_val']
        else:
            raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")

    else:
        # take absolute value instead
        if cut['moreless'] == '>':
            tree_df[cut['name']+' CUT'] = tree_df[cut['cut_var']].abs() > cut['cut_val']
        elif cut['moreless'] == '<':
            tree_df[cut['name']+' CUT'] = tree_df[cut['cut_var']].abs() < cut['cut_val']
        else:
            raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")


# ===========================
# ========= PLOTS ===========
# ===========================
# TODO: output cutflow histogram & plots


# ===========================
# == CALCULATING LUMI & XS ==
# ===========================
weight_mc = tree_df["weight_mc"]
n_events_tot = len(tree_df.index)
cross_section = sum(np.absolute(weight_mc)) / n_events_tot
lumi = tree_df['weight'].sum() / cross_section


# ===========================
# ======= PRINTOUTS =========
# ===========================
if printouts:
    # print cutflow output
    name_len = max([len(cut['name']) for cut in cut_list_dicts])
    var_len = max([len(cut['cut_var']) for cut in cut_list_dicts])
    print(f"\n========== CUTS USED ============")
    for cut in cut_list_dicts:
        if not cut['is_symmetric']:
            print(f"{cut['name']:<{name_len}}: "
                  f"{cut['cut_var']:>{var_len}} {cut['moreless']} {cut['cut_val']}")
        else:
            print(f"{cut['name']:<{name_len}}: "
                  f"{cut['cut_var']:>{var_len}} {cut['moreless']} |{cut['cut_val']}|")

    # cutflow printout
    print(f"\n=========== CUTFLOW =============")
    print("Cut "+" "*(name_len-3) +
          "Events "+" "*(len(str(n_events_tot))-6) +
          "Ratio Cum. Ratio")
    # first line is inclusive sample
    print("Inclusive "+" "*(name_len-9) + f"{n_events_tot} -     -")

    # perform cutflow and print
    # TODO: PERFORM CUTFLOW WHEN FILLING CUTFLOW HISTOGRAM, NOT WHEN DOING PRINTOUTS
    prev_n = n_events_tot
    # copy full dataframe. Have it reduce rows for each cut loop
    cutflow_df = tree_df.copy()
    for cut in cut_list_dicts:
        cutflow_df = cutflow_df[cutflow_df[cut['name']+' CUT']]
        n_events_cut = len(cutflow_df.index)
        ratio = n_events_cut / prev_n
        cum_ratio = n_events_cut / n_events_tot
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

    # TODO: PRINT TO TEX FILE

# if new cutfile, save backup
if make_backup:
    from shutil import copyfile
    from time import strftime
    cutfile_backup_filepath = backup_cutfiles_dir + "cutfile_" + strftime("%Y-%m-%d_%H-%M-%S")
    copyfile(cutfile, cutfile_backup_filepath)
    print(f"Backup cutfile saved in {cutfile_backup_filepath}")
