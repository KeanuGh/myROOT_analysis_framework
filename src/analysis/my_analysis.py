import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

# project imports
from utils.cutflow import Cutflow
from utils.cutfile_parser import parse_cutfile, gen_cutgroups, compare_backup
from utils.plotting_utils import plot_overlay_and_acceptance
from utils.dataframe_utils import (build_analysis_dataframe, create_cut_columns,
                                   gen_weight_column, rescale_to_gev,
                                   get_cross_section, get_luminosity)

# for file manipulation
from shutil import copyfile
import time
import os


class Analysis:
    # ===========================
    # ========= SETUP ===========
    # ===========================
    # multithreading
    n_threads = os.cpu_count() // 2

    # set ATLAS style plots
    plt.style.use([hep.style.ATLAS,
                   {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
                   ])

    # options
    TTree_name = 'truth'  # name of TTree to extract from root file
    cut_label = ' CUT'  # label to use for boolean cut columns in dataframe

    # filepaths
    input_root_file = '../../data/wminmunu_MC.root'
    cutfile = '../../options/cutfile.txt'
    out_dir = '../../outputs/'  # where outputs go
    out_plots_dir = out_dir + 'plots/'  # where plots go
    pkl_df_filepath = out_dir + TTree_name + '_df.pkl'  # pickle file containing extracted data
    # pkl_hist_filepath = out_dir + "histograms.pkl"  # pickle file to place histograms into
    backup_dir = '../../analysis_save_state/'  # where backups go
    backup_cutfiles_dir = backup_dir + 'cutfiles/'  # cutfile backups
    latex_table_dir = out_dir + "LaTeX_cutflow_table/"  # where to print latex cutflow table

    def __init__(self):
        # ============================
        # ======  READ CUTFILE =======
        # ============================
        # parse cutfile
        self.cut_dicts, self.vars_to_cut, self.options = parse_cutfile(self.cutfile)

        # check if cutfile backups exist
        self._build_dataframe, self._make_backup = compare_backup(self.cutfile,
                                                                  self.backup_cutfiles_dir,
                                                                  self.pkl_df_filepath)

        # ===============================
        # ==== EXTRACT & CLEAN DATA =====
        # ===============================
        # TODO: py-TChaining (maybe use pyROOT to actually TChain?) or awkward-arrays
        if self._build_dataframe:
            self.tree_df = build_analysis_dataframe(self.cut_dicts,
                                                    self.vars_to_cut,
                                                    self.input_root_file,
                                                    self.TTree_name,
                                                    pkl_filepath=self.pkl_df_filepath
                                                    )
        else:
            self.tree_df = pd.read_pickle(self.pkl_df_filepath)

        # extract cutgroups
        self.cutgroups = gen_cutgroups(self.cut_dicts)

        # map weights column
        n_events_tot = len(self.tree_df.index)  # this will be useful later
        self.tree_df['weight'] = gen_weight_column(self.tree_df)

        # rescale MeV columns to GeV
        rescale_to_gev(self.tree_df, inplace=True)

        # ===============================
        # ======= APPLYING CUTS =========
        # ===============================
        create_cut_columns(self.tree_df,
                           cut_dicts=self.cut_dicts,
                           cut_label=self.cut_label,
                           printout=True)

        # ===============================
        # ==== CALCULATING LUMI & XS ====
        # ===============================
        self.cross_section = get_cross_section(self.tree_df,
                                               n_events=n_events_tot)
        self.lumi = get_luminosity(self.tree_df,
                                   xs=self.cross_section)

        # ===============================
        # ========== CUTFLOW ============
        # ===============================
        self.Cutflow = Cutflow(self.tree_df, self.cut_dicts, self.cut_label)

        # plot latex table if it doesn't exist
        if self._make_backup or len(os.listdir(self.latex_table_dir)) == 0:
            self.Cutflow.print_latex_table(self.latex_table_dir)

        # if new cutfile, save backup
        if self._make_backup:
            cutfile_backup_filepath = self.backup_cutfiles_dir + "cutfile_" + time.strftime("%Y-%m-%d_%H-%M-%S")
            copyfile(self.cutfile, cutfile_backup_filepath)
            print(f"Backup cutfile saved in {cutfile_backup_filepath}")

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_with_cuts(self):
        # any of the substrings in this list shouldn't be binned logarithmically
        # (may need to double check this as it can cause problems if the substrings appear elsewhere)
        not_log = [
            '_phi_',
            '_eta_',
        ]

        for var_to_plot in self.vars_to_cut:
            plot_overlay_and_acceptance(var_to_plot,
                                        df=self.tree_df,
                                        cutgroups=self.cutgroups,
                                        lumi=self.lumi,
                                        dir_path=self.out_plots_dir,
                                        cut_label=self.cut_label,
                                        not_log=not_log,
                                        n_threads=self.n_threads,
                                        )

    def gen_cutflow_hist(self, ratio: bool = False, cummulative: bool = False):
        # plot histograms
        self.Cutflow.print_histogram(filepath=self.out_plots_dir)
        if ratio:
            self.Cutflow.print_histogram(filepath=self.out_plots_dir, ratio=True)
        if cummulative:
            self.Cutflow.print_histogram(filepath=self.out_plots_dir, cummulative=True)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(self):
        self.Cutflow.terminal_printout()

    def kinematics_printouts(self):
        # kinematics printout
        print(f"\n========== KINEMATICS ===========\n"
              f"cross-section: {self.cross_section:.2f} fb\n"
              f"luminosity   : {self.lumi:.2f} fb-1\n"
              )


if __name__ == '__main__':
    my_analysis = Analysis()

    # pipeline
    my_analysis.plot_with_cuts()
    my_analysis.gen_cutflow_hist(ratio=True, cummulative=True)
    my_analysis.cutflow_printout()
    my_analysis.kinematics_printouts()
