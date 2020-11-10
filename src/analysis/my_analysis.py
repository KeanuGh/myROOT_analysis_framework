import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
from itertools import combinations
from warnings import warn

# project imports
import analysis.config as config
from analysis.dataclass import Dataset
import utils.decorators as decs
import utils.file_utils as file_utils
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


class Analysis:
    # ===========================
    # ========= SETUP ===========
    # ===========================

    # filepaths
    backup_cutfiles_dir = config.backup_cutfiles_dir

    def __init__(self, data_dict: Dict[str, Dict],
                 cutfile: str,
                 analysis_label: str = '',
                 force_rebuild: bool = False,
                 grouped_cutflow: bool = True,
                 global_lumi: Optional[float] = None,
                 phibins: Optional[Union[tuple, list]] = None,
                 etabins: Optional[Union[tuple, list]] = None,
                 ):
        """
        TODO:
        | - Currently loops over each substep for each dataframe rather than looping over the entire analysis.
        | - Currently applies the same cuts to ALL dataframes. Instead I could allow each dataframe to read in its own
         cutfile
        | - May want to input your own analysis name to label things rather than just going by the name of the cutfile.
        | - May want to put all methods that act on only a single dataset into the Data class
        | - figure out how to access the a particular dataset by doing analysis.dataname?? (like df.column)
        | - Need to figure out logging
        | - need to differentiate latex filenames

        :param data_dict: Dictionary of dictionaries containing paths to root files and the tree to extract from each.
        The key to the top-level dictionary is the label assigned to the dataset.
        :param cutfile: path to cutfile
        :param force_rebuild: Force rebuild all dataframes.
        :param global_lumi: all data will be scaled to this luminosity
        :param phibins: bins for plotting phi
        :param etabins: bins for plotting eta
        """

        # BUILD DATASETS
        # ============================
        # build Data classes in name:Data dictionary containing initial values
        self.datasets = {name: Dataset(name, **ds) for name, ds in data_dict.items()}

        # set analysis options
        if global_lumi:
            config.lumi = global_lumi
        self._cutfile = cutfile
        self._not_log = [
            '_phi_',
            '_eta_',
        ]

        # variables that require special (default) binning
        if etabins:
            config.etabins = etabins
        if phibins:
            config.phibins = phibins
        self._special_binning = {
            '_eta_': config.etabins,
            '_phi_': config.phibins,
        }

        # PROCESS CUTFILE
        # ============================
        # the name of the cutfile sets directory name for outputs
        self._cutfile_name = file_utils.get_filename(self._cutfile)

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
        for name in self.datasets:
            self.datasets[name].rebuild = if_build_dataframe(self._cutfile,
                                                             self._make_backup,
                                                             self.backup_cutfiles_dir,
                                                             self.datasets[name].pkl_path)

        # SET OUTPUT DIRECTORIES
        # ===========================
        analysis_output_dir_name = self._cutfile_name.rstrip('.txt')

        # place plots in outputs/plots/<cutfile name>
        config.plot_dir = config.plot_dir.format(analysis_output_dir_name) + '/'

        # set where latex tables go
        config.latex_table_dir = config.latex_table_dir.format(analysis_output_dir_name)

        # set where pickled histograms go
        config.pkl_hist_dir = config.pkl_hist_dir.format(analysis_output_dir_name)

        # create directories if they don't exist
        file_utils.makedir([config.plot_dir, config.latex_table_dir, config.pkl_hist_dir])

        # EXTRACT & CLEAN DATA
        # ===============================
        for name in self.datasets:
            if self.datasets[name].rebuild or force_rebuild:
                print(f"\nBuilding {name} dataframe...")
                self.datasets[name].build_df(self.cut_dicts, self.vars_to_cut)
            else:
                print(f"Reading data for {name} dataframe from {self.datasets[name].pkl_path}...")
                self.datasets[name].df = pd.read_pickle(self.datasets[name].pkl_path)

        # MAP WEIGHTS
        # ===============================
        for name in self.datasets:
            self.datasets[name].map_weights()

        # APPLYING CUTS
        # ===============================
        for name in self.datasets:
            self.datasets[name].create_cut_columns(cut_dicts=self.cut_dicts, printout=True)

        # CALCULATING LUMI & XS
        # ===============================
        for name in self.datasets:
            self.datasets[name].gen_cross_section()
            self.datasets[name].gen_luminosity()

        # CUTFLOW
        # ===============================
        for name in self.datasets:
            self.datasets[name].gen_cutflow(cut_dicts=self.cut_dicts,
                                            cutgroups=self.cutgroups if grouped_cutflow else None,
                                            sequential=self.options['sequential'])

    # ===============================
    # =========== PLOTS =============
    # ===============================
    # TODO: save histograms to pickle file
    @decs.check_single_datafile
    def plot_with_cuts(self,
                       ds_name: Optional[str],
                       scaling: Optional[str] = None,
                       bins: Union[tuple, list] = (30, 1, 500),
                       not_log_add: Optional[List[str]] = None,
                       **kwargs
                       ) -> None:
        """
        Plots each variable in specific Dataset to cut from cutfile with each cutgroup applied

        :param ds_name: name of Dataset class to plot
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param scaling: either 'xs':     cross section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param not_log_add: Any extra variables that shouldn't be binned in log(x).
                            Currently defaults only '_eta_' and '_phi_'
        :param kwargs: keyword arguments to pass to plotting_utils.plot_1d_overlay_and_acceptance_cutgroups()
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
                df=self.datasets[ds_name].df,
                lepton=self.datasets[ds_name].lepton,
                var_to_plot=var_to_plot,
                cutgroups=self.cutgroups,
                lumi=self.datasets[ds_name].luminosity,
                is_logbins=is_logbins,
                scaling=scaling,
                bins=in_bins,
                plot_label=ds_name,
                **kwargs
            )

    @decs.check_single_datafile
    def gen_cutflow_hist(self,
                         ds_name: Optional[str],
                         event: bool = True,
                         ratio: bool = False,
                         cummulative: bool = False,
                         a_ratio: bool = False,
                         all_plots: bool = False,
                         ) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param ds_name: Name of dataset to plot
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
            self.datasets[ds_name].cutflow.print_histogram('event')
        if ratio:
            self.datasets[ds_name].cutflow.print_histogram('ratio')
        if cummulative:
            if self.options['sequential']:
                self.datasets[ds_name].cutflow.print_histogram('cummulative')
            else:
                warn("Sequential cuts cannot generate a cummulative cutflow")
        if a_ratio:
            if self.options['sequential']:
                self.datasets[ds_name].cutflow.print_histogram('a_ratio')
            else:
                warn("Sequential cuts can't generate cummulative cutflow. "
                     "Ratio of cuts to acceptance will be generated instead.")
                self.datasets[ds_name].cutflow.print_histogram('ratio')

    @decs.check_single_datafile
    def make_all_cutgroup_2dplots(self, ds_name: Optional[str], bins: Union[tuple, list] = (20, 0, 200), **kwargs):
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
            plot_2d_cutgroups(self.datasets[ds_name].df,
                              lepton=self.datasets[ds_name].lepton,
                              x_var=x_var, y_var=y_var,
                              xbins=xbins, ybins=ybins,
                              cutgroups=self.cutgroups,
                              plot_label=self.datasets[ds_name].name,
                              **kwargs)

    @decs.check_single_datafile
    def plot_mass_slices(self, ds_name: Optional[str], xvar: str, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param ds_name: name of dataset (in slices) to plot
        :param xvar: variable in dataframe to plot
        :param kwargs: keyword args to pass to dataclass.plot_mass_slices()
        """
        if not xvar:
            raise ValueError("xvar must be supplied")
        self.datasets[ds_name].plot_mass_slices(xvar=xvar, **kwargs)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @decs.check_single_datafile
    def cutflow_printout(self, ds_name: Optional[str] = None) -> None:
        """Prints cutflow table to terminal"""
        self.datasets[ds_name].cutflow.terminal_printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        print(f"\n========== KINEMATICS ===========")
        for name in self.datasets:
            print(name + ":\n---------------------------------")
            print(f"cross-section: {self.datasets[name].cross_section:.2f} fb\n"
                  f"luminosity   : {self.datasets[name].luminosity:.2f} fb-1\n"
                  )

    @decs.check_single_datafile
    def print_cutflow_latex_table(self, ds_name: Optional[str] = None, check_backup: bool = True) -> None:
        """
        Prints a latex table of cutflow. By default first checks if a current backup exists and will not print if
        backup is identical
        :param ds_name:
        :param check_backup: default true. Checks if backup of current cutflow already exists and if so does not print
        :return: None
        """
        if check_backup:
            last_backup = file_utils.get_last_backup(config.latex_table_dir)
        latex_file = self.datasets[ds_name].cutflow.print_latex_table(config.latex_table_dir, filename_prefix=ds_name)
        if check_backup and \
                file_utils.identical_to_backup(latex_file, backup_file=last_backup):
            file_utils.delete_file(latex_file)

    # ===============================
    # ========= UTILITIES ===========
    # ===============================

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
    data = {
        'truth_inclusive': {
            'datapath': '../../data/mc16d_wmintaunu/*',
            'TTree_name': 'truth',
            'is_slices': False,
            'lepton': 'tau'
        },
        'truth_slices': {
            'datapath': '../../data/mc16a_wmintaunu_SLICES/*.root',
            'TTree_name': 'truth',
            'is_slices': True,
            'lepton': 'tau'
        }
    }

    my_analysis = Analysis(data_dict=data,
                           cutfile='../../options/cutfile.txt',
                           force_rebuild=False,
                           )

    # pipeline
    my_analysis.plot_mass_slices(ds_name='truth_slices', xvar='MC_WZ_dilep_m_born', logx=True, to_pkl=True)
    my_analysis.plot_with_cuts(scaling='xs', ds_name='truth_inclusive', to_pkl=True)
    my_analysis.make_all_cutgroup_2dplots(ds_name='truth_inclusive', to_pkl=True)
    my_analysis.gen_cutflow_hist(ds_name='truth_inclusive', all_plots=True)
    my_analysis.cutflow_printout(ds_name='truth_inclusive')
    my_analysis.kinematics_printouts()
    my_analysis.print_cutflow_latex_table(ds_name='truth_inclusive')
