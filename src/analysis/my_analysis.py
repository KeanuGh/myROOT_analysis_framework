import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from typing import Optional, List
from os import cpu_count

# project imports
from utils.cutflow import Cutflow
from utils.cutfile_utils import parse_cutfile, gen_cutgroups, compare_cutfile_backup, backup_cutfile
from utils.file_utils import identical_to_backup, delete_file, is_dir_empty, get_last_backup
from utils.plotting_utils import plot_overlay_and_acceptance
from utils.dataframe_utils import (build_analysis_dataframe, create_cut_columns,
                                   gen_weight_column, rescale_to_gev,
                                   get_cross_section, get_luminosity)


class Analysis:
    # ===========================
    # ========= SETUP ===========
    # ===========================
    # multithreading
    n_threads = cpu_count() // 2

    # set ATLAS style plots
    plt.style.use([hep.style.ATLAS,
                   {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
                   ])

    # options
    TTree = 'truth'  # name of TTree to extract from root file
    cut_label = ' CUT'  # label to use for boolean cut columns in dataframe

    # filepaths
    out_dir = '../../outputs/'  # where outputs go
    out_plots_dir = out_dir + 'plots/'  # where plots go
    pkl_df_filepath = out_dir + 'data/' + TTree + '_df.pkl'  # pickle file containing extracted data
    # pkl_hist_filepath = out_dir + "histograms.pkl"  # pickle file to place histograms into
    backup_dir = '../../analysis_save_state/'  # where backups go
    backup_cutfiles_dir = backup_dir + 'cutfiles/'  # _cutfile backups
    latex_table_dir = out_dir + "LaTeX_cutflow_table/"  # where to print latex cutflow table

    def __init__(self, root_path: str, cutfile: str, lepton: str,
                 force_rebuild: bool = False):
        # set
        self._lepton_name = lepton
        self._cutfile = cutfile

        # ============================
        # ======  READ CUTFILE =======
        # ============================
        # parse _cutfile
        self.cut_dicts, self.vars_to_cut, self.options = parse_cutfile(self._cutfile)

        # check if _cutfile backups exist
        self._build_dataframe, self._make_backup = compare_cutfile_backup(self._cutfile,
                                                                          self.backup_cutfiles_dir,
                                                                          self.pkl_df_filepath)

        # ===============================
        # ==== EXTRACT & CLEAN DATA =====
        # ===============================
        if self._build_dataframe or force_rebuild:
            self.tree_df = build_analysis_dataframe(self.cut_dicts,
                                                    self.vars_to_cut,
                                                    root_path,
                                                    self.TTree,
                                                    pkl_filepath=self.pkl_df_filepath)
        else:
            self.tree_df = pd.read_pickle(self.pkl_df_filepath)

        # extract cutgroups
        self.cutgroups = gen_cutgroups(self.cut_dicts)

        # map weights column
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
        self.cross_section = get_cross_section(self.tree_df)
        self.lumi = get_luminosity(self.tree_df, xs=self.cross_section)

        # ===============================
        # ========== CUTFLOW ============
        # ===============================
        self.Cutflow = Cutflow(self.tree_df, self.cut_dicts, self.cut_label)

        # plot latex table if it doesn't exist and is different to the last file
        if self._make_backup or is_dir_empty(self.latex_table_dir):
            last_backup = get_last_backup(self.latex_table_dir)
            latex_file = self.Cutflow.print_latex_table(self.latex_table_dir)
            if identical_to_backup(latex_file, backup_file=last_backup):
                delete_file(latex_file)

        # if new _cutfile, save backup
        if self._make_backup:
            backup_cutfile(self.backup_cutfiles_dir, self._cutfile)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_with_cuts(self, scaling: Optional[str] = None, not_log_add: Optional[List[str]] = None) -> None:
        """
        Plots each variable to cut from _cutfile with each cutgroup applied
        :param scaling: either 'xs':     cross section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param not_log_add: Any extra variables that shouldn't be binned in log(x).
                            Currently defaults only '_eta_' and '_phi_'
        """
        # any of the substrings in this list shouldn't be binned logarithmically
        # (may need to double check this as it can cause problems if the substrings appear elsewhere)
        not_log = [
            '_phi_',
            '_eta_',
        ]
        if not_log_add:
            not_log += not_log_add

        for var_to_plot in self.vars_to_cut:
            plot_overlay_and_acceptance(var_to_plot,
                                        df=self.tree_df,
                                        cutgroups=self.cutgroups,
                                        lumi=self.lumi,
                                        dir_path=self.out_plots_dir,
                                        cut_label=self.cut_label,
                                        not_log=not_log,
                                        n_threads=self.n_threads,
                                        lepton=self._lepton_name,
                                        scaling=scaling,
                                        )

    def gen_cutflow_hist(self, ratio: bool = False, cummulative: bool = False):
        """Generates and saves cutflow histograms"""
        self.Cutflow.print_histogram(filepath=self.out_plots_dir)
        if ratio:
            self.Cutflow.print_histogram(filepath=self.out_plots_dir, ratio=True)
        if cummulative:
            self.Cutflow.print_histogram(filepath=self.out_plots_dir, cummulative=True)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(self) -> None:
        """Prints cutflow table to terminal"""
        self.Cutflow.terminal_printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        print(f"\n========== KINEMATICS ===========\n"
              f"cross-section: {self.cross_section:.2f} fb\n"
              f"luminosity   : {self.lumi:.2f} fb-1\n"
              )


if __name__ == '__main__':
    my_analysis = Analysis(root_path='../../data/mc16d_wmintaunu/*',
                           cutfile='../../options/cutfile.txt',
                           lepton='tau',
                           force_rebuild=False
                           )

    # pipeline
    my_analysis.plot_with_cuts(scaling='xs')
    my_analysis.gen_cutflow_hist(ratio=True, cummulative=True)
    my_analysis.cutflow_printout()
    my_analysis.kinematics_printouts()
