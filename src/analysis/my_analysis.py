import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
from itertools import combinations
from warnings import warn

# project imports
import analysis.config as config
from analysis.cutflow import Cutflow
from utils.plotting_utils import (
    plot_1d_overlay_and_acceptance_cutgroups,
    plot_2d_cutgroups
)
from utils.cutfile_utils import (
    parse_cutfile,
    gen_cutgroups,
    if_build_dataframe,
    if_make_cutfile_backup,
    backup_cutfile
)
from utils.file_utils import (
    identical_to_backup,
    delete_file,
    get_last_backup,
    get_filename, makedir
)
from utils.dataframe_utils import (
    build_analysis_dataframe,
    create_cut_columns,
    gen_weight_column,
    gen_weight_column_slices,
    rescale_to_gev,
    get_cross_section,
    get_luminosity
)


# TODO: Generate logs


class Analysis:
    # ===========================
    # ========= SETUP ===========
    # ===========================

    # options
    cut_label = ' CUT'  # label to use for boolean cut columns in dataframe

    # filepaths
    out_plots_dir = config.out_plots_dir
    # pkl_hist_filepath = config.pkl_hist_filepath
    backup_cutfiles_dir = config.backup_cutfiles_dir
    latex_table_dir = config.latex_table_dir

    def __init__(self, data: Dict[str, Dict[str]],
                 cutfile: str,
                 lepton: Optional[str],
                 force_rebuild: bool = False,
                 grouped_cutflow: bool = True,
                 global_lumi: Optional[float] = None,
                 phibins: Optional[Union[tuple, list]] = None,
                 etabins: Optional[Union[tuple, list]] = None,
                 ):
        """
        TODO __init__ doc string
        - Currently loops over each substep for each dataframe rather than looping over the entire analysis.
        - Currently applies the same cuts to ALL dataframes. Instead I could allow each dataframe to read in its own
          cutfile
        I may want to change these behaviours later.

        :param data: Dictionary of dictionaries containing paths to root files and the tree to extract from each.
        The key to the top-level dictionary is the label assigned to the dataset. This variable will eventually contain
        the entire dataframe and the path to its backup pickle file under 'df' and 'pkl_path' keys.
        :param cutfile:
        :param lepton:
        :param force_rebuild:
        :param grouped_cutflow:
        :param phibins:
        :param etabins:
        """
        self.data = data

        # set analysis options
        if lepton:
            config.lepton = lepton
        if global_lumi:
            config.lumi = global_lumi
        self._cutfile = cutfile
        self._not_log = [
            '_phi_',
            '_eta_',
        ]

        # variables that require special (default) binning
        if not etabins:
            config.etabins = etabins
        if not phibins:
            config.phibins = phibins
        self._special_binning = {
            '_eta_': config.etabins,
            '_phi_': config.phibins,
        }

        # generate dataframe pickle filepaths
        for name in data:
            self.data[name]['pkl_path'] = config.pkl_df_filepath.format(name)

        # ============================
        # ===== PROCESS CUTFILE ======
        # ============================
        # the name of the cutfile sets the name of the analysis for
        self._cutfile_name = get_filename(self._cutfile)

        # parse cutfile
        self.cut_dicts, self.vars_to_cut, self.options = parse_cutfile(self._cutfile)

        # extract cutgroups
        print("Extracting cutgroups...")
        self.cutgroups = gen_cutgroups(self.cut_dicts)

        # check if a backup of the input cutfile should be made
        self._make_backup = if_make_cutfile_backup(self._cutfile, self.backup_cutfiles_dir)

        # if new cutfile, save backup
        if self._make_backup:
            backup_cutfile(self.backup_cutfiles_dir, self._cutfile)

        # check which dataframes need to be rebuilt
        for name in self.data:
            data[name]['rebuild'] = if_build_dataframe(self._cutfile,
                                                       self._make_backup,
                                                       self.backup_cutfiles_dir,
                                                       data[name]['pkl_path']
                                                       )

        # place plots in outputs/plots/<cutfile name>
        self.plot_dir = self.out_plots_dir + self._cutfile_name.rstrip('.txt') + '/'
        makedir(self.plot_dir)

        # ===============================
        # ==== EXTRACT & CLEAN DATA =====
        # ===============================
        for name in self.data:
            if self.data[name]['rebuild'] or force_rebuild:
                print(f"\nBuilding {name} dataframe...")
                self.data[name]['df'] = build_analysis_dataframe(self.cut_dicts,
                                                                 self.vars_to_cut,
                                                                 self.data[name]['path'],
                                                                 self.data[name]['TTree'],
                                                                 self.data[name]['pkl_path']
                                                                 )
            else:
                print(f"Reading data for {name} dataframe from {self.data[name]['pkl_path']}...")
                self.data[name]['df'] = pd.read_pickle(self.data[name]['pkl_path'])

        # map weights column
        for name in self.data:
            if self.data[name]['slices']:
                self.data[name]['df']['weight'] = gen_weight_column_slices(self.data[name]['df'])
            else:
                self.data[name]['df']['weight'] = gen_weight_column(self.data[name]['df'])

        # rescale MeV columns to GeV
        for name in self.data:
            rescale_to_gev(self.data[name]['df'])

        # ===============================
        # ======= APPLYING CUTS =========
        # ===============================
        for name in self.data:
            create_cut_columns(self.data[name]['df'],
                               cut_dicts=self.cut_dicts,
                               cut_label=self.cut_label,
                               printout=True)

        # ===============================
        # ==== CALCULATING LUMI & XS ====
        # ===============================
        for name in self.data:
            self.cross_section = get_cross_section(self.tree_df)
            self.luminosity = get_luminosity(self.tree_df, xs=self.cross_section)

        # ===============================
        # ========== CUTFLOW ============
        # ===============================
        self.cutflow = Cutflow(df=self.tree_df,
                               cut_dicts=self.cut_dicts,
                               cutgroups=self.cutgroups if grouped_cutflow else None,
                               cut_label=self.cut_label,
                               sequential=self.options['sequential'])

    # ===============================
    # =========== PLOTS =============
    # ===============================
    # TODO: simple functions that take variables as input and plot 1d/2d histograms with/without a cut
    # TODO: save histograms to pickle file
    def plot_with_cuts(self,
                       scaling: Optional[str] = None,
                       bins: Union[tuple, list] = (30, 1, 500),
                       not_log_add: Optional[List[str]] = None,
                       log_x: bool = False
                       ) -> None:
        """
        Plots each variable to cut from _cutfile with each cutgroup applied

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param scaling: either 'xs':     cross section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param not_log_add: Any extra variables that shouldn't be binned in log(x).
                            Currently defaults only '_eta_' and '_phi_'
        :param log_x: log x axis
        """
        # any of the substrings in this list shouldn't be binned logarithmically
        # (may need to double check this as it can cause problems if the substrings appear elsewhere)
        if not_log_add:
            self._not_log += not_log_add

        for var_to_plot in self.vars_to_cut:
            # whether or not bins should be logarithmic bins
            is_logbins, in_bins = self.__getbins(var_to_plot)
            if not in_bins:
                in_bins = bins

            print(f"Generating histogram for {var_to_plot}...")
            plot_1d_overlay_and_acceptance_cutgroups(
                df=self.tree_df,
                var_to_plot=var_to_plot,
                cutgroups=self.cutgroups,
                lumi=self.luminosity,
                dir_path=self.plot_dir,
                cut_label=self.cut_label,
                is_logbins=is_logbins,
                log_x=log_x,
                scaling=scaling,
                bins=in_bins,
            )

    def gen_cutflow_hist(self,
                         event: bool = True,
                         ratio: bool = False,
                         cummulative: bool = False,
                         a_ratio: bool = False,
                         all_plots: bool = False,
                         ) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param event: y-axis is number of events passing each cut
        :param ratio: ratio of each subsequent cut if sequential,
                      else ratio of events passing each cut to inclusive sample
        :param cummulative: ratio of each cut to the previous cut
        :param a_ratio: ratio of cut to inclusive sample
        :param all_plots: it True, plot all
        :return: None
        """
        if all_plots:
            event = ratio = cummulative = a_ratio = True

        if event:
            self.cutflow.print_histogram(self.plot_dir, 'event')
        if ratio:
            self.cutflow.print_histogram(self.plot_dir, 'ratio')
        if cummulative:
            if self.options['sequential']:
                self.cutflow.print_histogram(self.plot_dir, 'cummulative')
            else:
                warn("Sequential cuts cannot generate a cummulative cutflow")
        if a_ratio:
            if self.options['sequential']:
                self.cutflow.print_histogram(self.plot_dir, 'a_ratio')
            else:
                warn("Sequential cuts can't generate cummulative cutflow. "
                     "Ratio of cuts to acceptance will be generated instead.")
                self.cutflow.print_histogram(self.plot_dir, 'ratio')

    def make_all_cutgroup_2dplots(self, bins: Union[tuple, list] = (20, 0, 200)):
        if len(self.vars_to_cut) < 2:
            raise Exception("Need at least two plotting variables to make 2D plot")

        for x_var, y_var in combinations(self.vars_to_cut, 2):
            # binning
            _, xbins = self.__getbins(x_var)
            _, ybins = self.__getbins(y_var)
            if not xbins:
                xbins = bins
            if not ybins:
                ybins = bins
            print(f"Generating 2d histogram for {x_var}-{y_var}...")
            plot_2d_cutgroups(self.tree_df,
                              x_var=x_var, y_var=y_var,
                              xbins=xbins, ybins=ybins,
                              cutgroups=self.cutgroups,
                              dir_path=self.plot_dir,
                              cut_label=self.cut_label,
                              )

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(self) -> None:
        """Prints cutflow table to terminal"""
        self.cutflow.terminal_printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        print(f"\n========== KINEMATICS ===========\n"
              f"cross-section: {self.cross_section:.2f} fb\n"
              f"luminosity   : {self.luminosity:.2f} fb-1\n"
              )

    def print_cutflow_latex_table(self, check_backup: bool = True) -> None:
        """
        Prints a latex table of cutflow. By default first checks if a current backup exists and will not print if
        backup is identical
        :param check_backup: default true. Checks if backup of current cutflow already exists and if so does not print
        :return: None
        """
        if check_backup:
            last_backup = get_last_backup(self.latex_table_dir)
        latex_file = self.cutflow.print_latex_table(self.latex_table_dir, self._cutfile_name)
        if check_backup and \
                identical_to_backup(latex_file, backup_file=last_backup):
            delete_file(latex_file)

    # ===============================
    # ========== PRIVATE ============
    # ===============================
    def __getbins(self, var_to_plot) -> Tuple[bool, Optional[tuple]]:
        """
        Returns special bins if variable input requires it. Returns None if not a special variable
        :param var_to_plot: variable to choose bins from
        :return: tuple (n_bins, start, stop) of bins or None
        """
        is_logbins = not any(map(var_to_plot.__contains__, self._not_log))
        if not is_logbins:
            # set bins for special variables
            sp_var = [sp_var for sp_var in self._not_log if sp_var in var_to_plot]
            if len(sp_var) != 1:
                raise Exception(f"Expected one matching variable for spcial binning. Got {sp_var}")
            return is_logbins, self._special_binning[sp_var[0]]
        else:
            return is_logbins, None


if __name__ == '__main__':

    data_dict = {'truth': {
        'path': '../../data/mc16d_wmintaunu/*',
        'TTree': 'truth',
        'slices': False
        }
    }

    my_analysis = Analysis(data=data_dict,
                           cutfile='../../options/cutfile.txt',
                           lepton='tau',
                           force_rebuild=False
                           )

    # pipeline
    # my_analysis.plot_with_cuts(scaling='xs')
    # my_analysis.make_all_cutgroup_2dplots()
    my_analysis.gen_cutflow_hist(all_plots=True)
    my_analysis.cutflow_printout()
    my_analysis.kinematics_printouts()
    my_analysis.print_cutflow_latex_table()
