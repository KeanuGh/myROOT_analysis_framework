import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep
import pandas as pd

# project imports
from utils.cutflow import Cutflow
from utils.cutfile_parser import parse_cutfile, gen_cutgroups, compare_backup
from utils.plotting_utils import scale_to_crosssection, get_root_sumw2_1d, get_axis_labels
from utils.dataframe_utils import (build_analysis_dataframe, apply_cuts,
                                   gen_weight_column, rescale_to_gev,
                                   get_crosssection, get_luminosity)

# for file manipulation
from shutil import copyfile
from time import strftime
from datetime import datetime
import os


def my_analysis():
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
    input_root_file = '../../data/wminmunu_MC.root'
    cutfile = '../../options/cutfile.txt'
    out_dir = '../../outputs/'  # where outputs go
    out_plots_dir = out_dir + 'plots/'  # where plots go
    pkl_df_filepath = out_dir + TTree_name + '_df.pkl'  # pickle file containing extracted data as pandas dataframe
    # pkl_hist_filepath = out_dir + "histograms.pkl"  # pickle file to place histograms into
    backup_dir = '../../analysis_save_state/'  # where backups go
    backup_cutfiles_dir = backup_dir + 'cutfiles/'  # cutfile backups
    latex_table_dir = out_dir + "LaTeX_cutflow_table/"  # where to print latex cutflow table

    # multithreading
    n_threads = os.cpu_count() // 2

    # ============================
    # ======  READ CUTFILE =======
    # ============================
    # parse cutfile
    cut_dicts, vars_to_cut, options = parse_cutfile(cutfile)

    # check if cutfile backups exist
    build_dataframe, make_backup = compare_backup(cutfile, backup_cutfiles_dir, pkl_df_filepath)

    # ===============================
    # ==== EXTRACT & CLEAN DATA =====
    # ===============================
    # TODO: py-TChaining (maybe use pyROOT to actually TChain?) or awkward-arrays
    if build_dataframe:
        tree_df = build_analysis_dataframe(cut_dicts, vars_to_cut, input_root_file,
                                           TTree_name, pkl_filepath=pkl_df_filepath)
    else:
        tree_df = pd.read_pickle(pkl_df_filepath)

    # extract cutgroups
    cutgroups = gen_cutgroups(cut_dicts)

    # map weights column
    n_events_tot = len(tree_df.index)  # this will be useful later
    tree_df['weight'] = gen_weight_column(tree_df)

    # rescale MeV columns to GeV
    rescale_to_gev(tree_df, inplace=True)

    # ===============================
    # ======= APPLYING CUTS =========
    # ===============================
    apply_cuts(tree_df, cut_dicts=cut_dicts, cut_label=cut_label, printout=True)

    # ===============================
    # ==== CALCULATING LUMI & XS ====
    # ===============================
    cross_section = get_crosssection(tree_df, n_events=n_events_tot)
    lumi = get_luminosity(tree_df, xs=cross_section)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    plot_width = 15
    plot_height = 15

    n_bins = 30
    binrange = (1, 500)
    eta_binrange = (-20, 20)

    # any of the substrings in this list shouldn't be binned logarithmically
    # (may need to double check this as it can cause problems if the substrings appear elsewhere)
    not_log = [
        '_phi_',
        '_eta_',
    ]

    for var_to_cut in vars_to_cut:
        print(f"Generating histogram for {var_to_cut}...")
        fig, (fig_ax, accept_ax) = plt.subplots(1, 2)

        # whether or not bins should be logarithmic bins
        is_logbins = not any(map(var_to_cut.__contains__, not_log))

        # get axis labels (xlabel, ylabel)
        xlabel, ylabel = get_axis_labels(var_to_cut)

        # INCLUSIVE PLOT
        # ================
        # setup inclusive histogram
        if is_logbins:
            h_inclusive = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log),
                                       storage=bh.storage.Weight())
        else:
            h_inclusive = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange),
                                       storage=bh.storage.Weight())

        h_inclusive.fill(tree_df[var_to_cut], weight=tree_df['weight'], threads=n_threads)  # fill
        scale_to_crosssection(h_inclusive, luminosity=lumi)  # scale
        yerr = get_root_sumw2_1d(h_inclusive)  # get sum of weights squared

        # plot
        hep.histplot(h_inclusive.view().value, bins=h_inclusive.axes[0].edges,
                     ax=fig_ax, yerr=yerr, label='Inclusive')

        # PLOT CUTS
        # ================
        for cutgroup in cutgroups.keys():
            print(f"    - generating cutgroup '{cutgroup}'")
            # get column names for boolean columns in dataframe containing the cuts
            cut_rows = [cut_name + cut_label for cut_name in cutgroups[cutgroup]]

            # setup
            if is_logbins:
                h_cut = bh.Histogram(bh.axis.Regular(n_bins, *binrange, transform=bh.axis.transform.log),
                                     storage=bh.storage.Weight())
            else:
                h_cut = bh.Histogram(bh.axis.Regular(n_bins, *eta_binrange),
                                     storage=bh.storage.Weight())

            cut_df = tree_df[tree_df[cut_rows].any(1)]
            h_cut.fill(cut_df[var_to_cut], weight=cut_df['weight'], threads=n_threads)  # fill
            scale_to_crosssection(h_cut, luminosity=lumi)  # scale
            cut_yerr = get_root_sumw2_1d(h_cut)

            # plot
            hep.histplot(h_cut.view().value, bins=h_cut.axes[0].edges,
                         ax=fig_ax, yerr=cut_yerr, label=cutgroup)

            # RATIO PLOT
            # ================
            hep.histplot(h_cut.view().value / h_inclusive.view().value,
                         bins=h_cut.axes[0].edges, ax=accept_ax, label=cutgroup)

        # log y axis, unless plotting Bjorken X
        if 'PDFinfo_X' not in var_to_cut:
            fig_ax.semilogy()

        # apply axis options
        if is_logbins:  # set axis edge at 0
            fig_ax.set_xlim(*binrange)
        else:
            fig_ax.set_xlim(*eta_binrange)
        fig_ax.set_xlabel(xlabel)
        fig_ax.set_ylabel(ylabel)
        fig_ax.legend()
        # hep.box_aspect(fig_ax)  # makes just the main figure a square (& too small)

        # ratio plot
        if is_logbins:  # set axis edge at 0
            accept_ax.set_xlim(*binrange)
        else:
            accept_ax.set_xlim(*eta_binrange)
        accept_ax.set_xlabel(xlabel)
        accept_ax.set_ylabel("Acceptance")
        accept_ax.legend()

        # set dimensions manually
        fig.set_figheight(plot_height)
        fig.set_figwidth(plot_width * 2)

        # save figure
        hep.atlas.label(data=False, ax=fig_ax, paper=False, year=datetime.now().year)
        out_png_file = out_plots_dir + f"{var_to_cut}_XS.png"
        fig.savefig(out_png_file)
        print(f"Figure saved to {out_png_file}")
        plt.clf()  # clear for next plot

    # CUTFLOW
    # ================
    cutflow = Cutflow(tree_df, cut_dicts, cut_label)

    # plot histograms
    cutflow.print_histogram(out_plots_dir)
    cutflow.print_histogram(out_plots_dir, ratio=True)
    cutflow.print_histogram(out_plots_dir, cummulative=True)

    # plot latex table if it doesn't exist
    if make_backup or len(os.listdir(latex_table_dir)) == 0:
        cutflow.print_latex_table(latex_table_dir)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    if printouts:
        # cutflow printout
        cutflow.terminal_printout()

        # kinematics printout
        print(f"\n========== KINEMATICS ===========\n"
              f"cross-section: {cross_section:.2f} fb\n"
              f"luminosity   : {lumi:.2f} fb-1\n"
              )

    # if new cutfile, save backup
    if make_backup:
        cutfile_backup_filepath = backup_cutfiles_dir + "cutfile_" + strftime("%Y-%m-%d_%H-%M-%S")
        copyfile(cutfile, cutfile_backup_filepath)
        print(f"Backup cutfile saved in {cutfile_backup_filepath}")


if __name__ == '__main__':
    my_analysis()
