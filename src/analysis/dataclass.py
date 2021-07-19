from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Union
from warnings import warn

import pandas as pd

import analysis.config as config
import utils.dataframe_utils as df_utils
import utils.file_utils as file_utils
import utils.plotting_utils as plt_utils
from analysis.cutflow import Cutflow
from utils.cutfile_utils import (
    parse_cutfile,
    gen_cutgroups,
    if_build_dataframe,
    if_make_cutfile_backup,
    backup_cutfile
)


# TODO: change this to a regular class. No need to be dataclass anymore (and write a __repr__)
@dataclass
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.
    Perhaps put all methods that act only on one dataset into here
    TODO: Save histograms as dataset attributes? Ability to perform operations across hisograms? Custom histogram class?
    """
    name: str
    datapath: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    cutfile: str  # path to cutfile
    df: pd.DataFrame = field(default=None, repr=False)  # stores the actual data in a dataframe
    pkl_path: str = None  # where the dataframe pickle file will be stored
    is_slices: bool = False  # whether input data is in mass slices
    lepton: str = 'lepton'  # name of charged DY lepton channel in dataset (if applicable)
    grouped_cutflow: bool = True  # whether cutflow should apply cuts in cutgroups or separately TODO: in cutfile

    def __post_init__(self):
        """Dataset generation pipeline"""
        print(f"\n======== INITIALISING DATASET '{self.name}' =========")
        if not self.pkl_path:
            # initialise pickle filepath with given name
            self.pkl_path = config.pkl_df_filepath + self.name + '_df.pkl'

        # READ AND GET OPTIONS FROM CUTFILE
        # ========================
        self.__cutfile_pipeline()

        # GENERATE DATAFRAME
        # ========================
        self.__dataframe_pipeline()

        # GENERATE CUTFLOW
        # ========================
        self.__gen_cutflow()
        self.cutflow_printout()
        print(f"========= DATASET '{self.name}' INITIALISED =========\n")

    # Builtins
    # ===================
    def __len__(self):
        """Return number of rows in dataframe"""
        return len(self.df.index)

    def __getitem__(self, col):
        return self.df[col]

    def __setitem__(self, col, item):
        self.df[col] = item

    # Variable setting
    # ===================
    @property
    def cross_section(self) -> float:
        """Calculate dataset cross-section"""
        return df_utils.get_cross_section(self.df)

    @property
    def luminosity(self) -> float:
        """Calculate dataset luminosity"""
        return df_utils.get_luminosity(self.df, xs=self.cross_section)

    def __gen_cutflow(self) -> None:
        """Create the cutflow class for this analysis"""
        self.cutflow = Cutflow(self.df, self._cut_dicts,
                               self._cutgroups if self.grouped_cutflow else None,
                               self._cutflow_options['sequential'])

    # ===============================
    # ========= PIPELINES ===========
    # ===============================
    def __cutfile_pipeline(self) -> None:
        """Define the pipeline for parsing cutfile"""
        print(f"Parsting cutfile for {self.name}...")

        self._cut_dicts, self._vars_to_cut, self._cutflow_options = parse_cutfile(self.cutfile)

        # extract cutgroups
        self._cutgroups = gen_cutgroups(self._cut_dicts)

        # check if a backup of the input cutfile should be made
        self._make_backup = if_make_cutfile_backup(self.cutfile, config.backup_cutfiles_dir)

        # if new cutfile, save backup
        if self._make_backup:
            backup_cutfile(config.backup_cutfiles_dir + self.name + '_', self.cutfile)

        self._rebuild = if_build_dataframe(self.cutfile,
                                           self._make_backup,
                                           config.backup_cutfiles_dir,
                                           self.pkl_path)

    def __dataframe_pipeline(self) -> None:
        """Define pipeline for building dataset dataframe"""
        # extract and clean data
        if self._rebuild or config.force_rebuild:
            print(f"Building {self.name} dataframe from {self.datapath}...")
            self.__build_df()
        else:
            print(f"Reading data for {self.name} dataframe from {self.pkl_path}...")
            self.df = pd.read_pickle(self.pkl_path)

        # map appropriate weights
        self.__map_weights()

        # apply cuts to generate cut columns
        self.__create_cut_columns()

    # ===============================
    # ========= DATAFRAME ===========
    # ===============================
    def __build_df(self) -> None:
        """Build dataframe based on 'datapath', 'TTree_name' and 'is_slices'. Print pickle file to 'pkl_path'"""
        self.df = df_utils.build_analysis_dataframe(self, self._cut_dicts, self._vars_to_cut)

    def __create_cut_columns(self) -> None:
        """Create columns in dataframe that contain boolean values corresponding to cuts."""
        print(f"Creating cuts for {self.name}...")
        df_utils.create_cut_columns(self.df, cut_dicts=self._cut_dicts, printout=True)

    def __map_weights(self) -> None:
        """Create weights column in dataset based on dataset type"""
        print(f"Creating weights for {self.name}...")
        if self.is_slices:
            self.df['weight'] = df_utils.gen_weight_column_slices(self.df)
        else:
            self.df['weight'] = df_utils.gen_weight_column(self.df)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(self) -> None:
        """Prints cutflow table to terminal"""
        self.cutflow.terminal_printout()

    def kinematics_printout(self) -> None:
        """Prints some kinematic variables to terminal"""
        print(f"\n========{self.name.upper()} KINEMATICS ===========")
        print(f"cross-section: {self.cross_section:.2f} fb\n"
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
            last_backup = file_utils.get_last_backup(config.latex_table_dir)
            latex_file = self.cutflow.print_latex_table(config.latex_table_dir, self.name)
            if file_utils.identical_to_backup(latex_file, backup_file=last_backup):
                file_utils.delete_file(latex_file)
        else:
            self.cutflow.print_latex_table(config.latex_table_dir, self.name)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    # TODO: single plot function

    def plot_with_cuts(self,
                       scaling: Optional[str] = None,
                       bins: Union[tuple, list] = (30, 1, 500),
                       **kwargs
                       ) -> None:
        """
        Plots each variable in specific Dataset to cut from cutfile with each cutgroup applied

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param scaling: either 'xs':     cross section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param kwargs: keyword arguments to pass to plotting_utils.plot_1d_overlay_and_acceptance_cutgroups()
        """
        for var_to_plot in self._vars_to_cut:
            print(f"Generating histogram for {var_to_plot}...")
            plt_utils.plot_1d_overlay_and_acceptance_cutgroups(
                df=self.df,
                lepton=self.lepton,
                var_to_plot=var_to_plot,
                cutgroups=self._cutgroups,
                lumi=self.luminosity,
                scaling=scaling,
                bins=bins,
                plot_label=self.name,
                **kwargs
            )

    def make_all_cutgroup_2dplots(self, bins: Union[tuple, list] = (20, 0, 200), **kwargs):
        """Plots all cutgroups as 2d plots

        :param bins: bin edges or tuple of (n_bins, start, stop)
        :param kwargs: keyword arguments to pass to plot_utils.plot_2d_cutgroups
        """
        if len(self._vars_to_cut) < 2:
            raise Exception("Need at least two plotting variables to make 2D plot")

        for x_var, y_var in combinations(self._vars_to_cut, 2):
            # binning
            _, xbins = plt_utils.getbins(x_var)
            _, ybins = plt_utils.getbins(y_var)
            if not xbins:
                xbins = bins
            if not ybins:
                ybins = bins
            print(f"Generating 2d histogram for {x_var}-{y_var}...")
            plt_utils.plot_2d_cutgroups(self.df,
                                        lepton=self.lepton,
                                        x_var=x_var, y_var=y_var,
                                        xbins=xbins, ybins=ybins,
                                        cutgroups=self._cutgroups,
                                        plot_label=self.name,
                                        **kwargs)

    def plot_mass_slices(self, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param kwargs: keyword arguments to be passed to plotting_utils.plot_mass_slices()
        """
        if not self.is_slices:
            raise Exception("Dataset does not contain slices.")

        plt_utils.plot_mass_slices(self.df, self.lepton, plot_label=self.name, **kwargs)

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
            self.cutflow.print_histogram('event')
        if ratio:
            self.cutflow.print_histogram('ratio')
        if cummulative:
            if self._cutflow_options['sequential']:
                self.cutflow.print_histogram('cummulative')
            else:
                warn("Sequential cuts cannot generate a cummulative cutflow")
        if a_ratio:
            if self._cutflow_options['sequential']:
                self.cutflow.print_histogram('a_ratio')
            else:
                warn("Sequential cuts can't generate cummulative cutflow. "
                     "Ratio of cuts to acceptance will be generated instead.")
                self.cutflow.print_histogram('ratio')
