import logging
import operator as op
import time
from dataclasses import dataclass, field
from glob import glob
from itertools import combinations
from typing import Optional, Union, List, Dict, OrderedDict
from warnings import warn

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot

import src.config as config
import utils.file_utils as file_utils
import utils.plotting_utils as plt_utils
from src.cutflow import Cutflow
from utils.axis_labels import labels_xs
from utils.cutfile_utils import (
    parse_cutfile,
    gen_cutgroups,
    if_build_dataframe,
    if_make_cutfile_backup,
    backup_cutfile,
    extract_cut_variables,
    # all_vars,
    gen_alt_tree_dict
)
from utils.var_helpers import derived_vars

logger = logging.getLogger('analysis')


@dataclass
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.
    Perhaps put all methods that act only on one dataset into here
    TODO: Save histograms as dataset attributes? Ability to perform operations across hisograms? Custom histogram class?
    """
    name: str
    data_path: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    cutfile_path: str  # path to cutfile
    df: pd.DataFrame = field(init=False, repr=False)  # stores the actual data in a dataframe
    cutflow: Cutflow = field(init=False)
    pkl_path: str = None  # where the dataframe pickle file will be stored
    is_slices: bool = False  # whether input data is in mass slices
    lepton: str = 'lepton'  # name of charged DY lepton channel in dataset (if applicable)
    grouped_cutflow: bool = True  # whether cutflow should apply cuts in cutgroups or separately TODO: in cutfile
    chunksize: int = 1024

    def __post_init__(self):
        """Dataset generation pipeline"""
        logger.info("")
        logger.info("=" * (42 + len(self.name)))
        logger.info(f"======== INITIALISING DATASET '{self.name}' =========")
        logger.info("=" * (42 + len(self.name)))
        # if not file_utils.file_exists(self.datapath):
        #     raise FileExistsError(f"File {self.datapath} not found.")

        if not self.pkl_path:
            # initialise pickle filepath with given name
            self.pkl_path = config.paths['pkl_df_filepath'] + self.name + '_df.pkl'

        # some debug information
        logger.debug("DATASET INPUT OPTIONS: ")
        logger.debug("----------------------------")
        logger.debug(f"Input file(s):  {self.data_path}")
        logger.debug(f"TTree: {self.TTree_name}")
        logger.debug(f"Cutfile: {self.cutfile_path}")
        logger.debug(f"Slices: {self.is_slices}")
        logger.debug(f"grouped cutflow: {self.grouped_cutflow}")
        logger.debug(f"Forced dataset rebuild: {config.force_rebuild}")
        logger.debug("----------------------------")
        logger.debug("")

        if logger.level == logging.DEBUG:
            # log contents of cutfile
            logger.debug("CUTFILE CONTENTS")
            logger.debug('---------------------------')
            with open(self.cutfile_path) as f:
                for line in f.readlines():
                    logger.debug(line.rstrip('\n'))
            logger.debug('---------------------------')

        # READ AND GET OPTIONS FROM CUTFILE
        # ========================
        """Define the pipeline for parsing cutfile"""
        logger.info(f"Parsting cutfile for {self.name}...")

        self._cut_dicts, self._vars_to_cut, self._cutflow_options = parse_cutfile(self.cutfile_path)

        # extract cutgroups
        self._cutgroups = gen_cutgroups(self._cut_dicts)

        # check if a backup of the input cutfile should be made
        self._make_backup = if_make_cutfile_backup(self.cutfile_path, config.paths['backup_cutfiles_dir'])

        # if new cutfile, save backup
        if self._make_backup:
            backup_cutfile(config.paths['backup_cutfiles_dir'] + self.name + '_', self.cutfile_path)

        self._rebuild = if_build_dataframe(self.cutfile_path,
                                           self._make_backup,
                                           config.paths['backup_cutfiles_dir'],
                                           self.pkl_path)

        # GENERATE DATAFRAME
        # ========================
        """Define pipeline for building dataset dataframe"""
        # extract and clean data
        if self._rebuild or config.force_rebuild:
            logger.info(f"Building {self.name} dataframe from {self.data_path}...")
            self.df = self._build_dataframe(datapath=self.data_path,
                                            TTree_name=self.TTree_name,
                                            cut_list_dicts=self._cut_dicts,
                                            vars_to_cut=self._vars_to_cut,
                                            is_slices=self.is_slices)

            # print into pickle file for easier read/write
            if self.pkl_path:
                pd.to_pickle(self.df, self.pkl_path)
                logger.info(f"Dataframe built and saved in {self.pkl_path}")

        else:
            logger.info(f"Reading data for {self.name} dataframe from {self.pkl_path}...")
            self.df = pd.read_pickle(self.pkl_path)
        logger.info(f"Number of events in dataset {self.name}: {len(self.df)}")

        # map appropriate weights
        logger.info(f"Creating weights for {self.name}...")
        if self.is_slices:
            self.df['weight'] = self.__gen_weight_column_slices(self.df)
        else:
            self.df['weight'] = self.__gen_weight_column(self.df)

        # print some dataset ID metadata
        # TODO: avg event weight
        if logger.level == logging.DEBUG:
            if self.is_slices:
                logger.debug("PER-DSID INFO:")
                logger.debug("--------------")
                logger.debug("DSID       n_events   sum_w         x-s fb        lumi fb-1")
                logger.debug("==========================================================")
                for dsid in np.sort(self.df['DSID'].unique()):
                    df_id = self.df[self.df['DSID'] == dsid]
                    logger.debug(f"{int(dsid):<10} "
                                 f"{len(df_id):<10} "
                                 f"{df_id['weight_mc'].sum():<10.6e}  "
                                 f"{self._get_cross_section(df_id):<10.6e}  "
                                 f"{self._get_luminosity(df_id):<10.6e}")
            else:
                logger.debug("INCLUSIVE SAMPLE METADATA:")
                logger.debug("--------------------------")
                logger.debug("n_events   sum_w         x-s fb        lumi fb-1")
                logger.debug("==========================================================")
                logger.debug(f"{len(self.df):<10} "
                             f"{self.df['weight_mc'].sum():<10.6e}  "
                             f"{self._get_cross_section(self.df):<10.6e}  "
                             f"{self._get_luminosity(self.df):.<10.6e}")

        # apply cuts to generate cut columns
        logger.info(f"Creating cuts for {self.name}...")
        self.__create_cut_columns(self.df, cut_dicts=self._cut_dicts)

        # GENERATE CUTFLOW
        # ========================
        self.cutflow = Cutflow(self.df, self._cut_dicts,
                               self._cutgroups if self.grouped_cutflow else None,
                               self._cutflow_options['sequential'])

        logger.info("=" * (42 + len(self.name)))
        logger.info(f"========= DATASET '{self.name}' INITIALISED =========")
        logger.info("=" * (42 + len(self.name)))
        logger.info("")

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
        return self._get_cross_section(self.df)

    @property
    def luminosity(self) -> float:
        """Calculate dataset luminosity"""
        return self._get_luminosity(self.df, xs=self.cross_section)

    @classmethod
    def _get_cross_section(cls, df: pd.DataFrame, n_events=None, weight_mc_col: str = 'weight_mc'):
        """
        Calculates cross-section of data in dataframe
        :param df: input dataframe
        :param n_events: optional: total number of events. Will calculate if not given.
        :param weight_mc_col: column containing monte carlo weights
        :return: cross-section
        """
        if not n_events:
            n_events = len(df.index)
        return df[weight_mc_col].sum() / n_events

    @classmethod
    def _get_luminosity(cls, df: pd.DataFrame, xs=None, weight_col: str = 'weight'):
        """
        Calculates luminosity from dataframe
        :param df: input dataframe
        :param xs: cross-section. If not given, will calculate
        :param weight_col: column of dataframe containing the weights
        :return: luminosity
        """
        if not xs:
            xs = cls._get_cross_section(df)
        return df[weight_col].sum() / xs

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(self) -> None:
        """Prints cutflow table to terminal"""
        self.cutflow.printout()

    def kinematics_printout(self) -> None:
        """Prints some kinematic variables to terminal"""
        logger.info("")
        logger.info(f"========{self.name.upper()} KINEMATICS ===========")
        logger.info(f"cross-section: {self.cross_section:.2f} fb")
        logger.info(f"luminosity   : {self.luminosity:.2f} fb-1")

    def print_cutflow_latex_table(self, check_backup: bool = True) -> None:
        """
        Prints a latex table of cutflow. By default first checks if a current backup exists and will not print if
        backup is identical
        :param check_backup: default true. Checks if backup of current cutflow already exists and if so does not print
        :return: None
        """
        if check_backup:
            last_backup = file_utils.get_last_backup(config.paths['latex_table_dir'])
            latex_file = self.cutflow.print_latex_table(config.paths['latex_table_dir'], self.name)
            if file_utils.identical_to_backup(latex_file, backup_file=last_backup):
                file_utils.delete_file(latex_file)
        else:
            self.cutflow.print_latex_table(config.paths['latex_table_dir'], self.name)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_1d(self,
                x: Union[str, List[str]],
                bins: Union[tuple, list],
                title: str = '',
                to_file: bool = True,
                filename: str = '',
                scaling: str = None,
                **kwargs
                ) -> plt.figure:
        """
        Generate 1D plots of given variables in dataframe. Returns figure object of list of figure objects

        :param x: variable name in dataframe to plot
        :param bins: binnings for x
        :param title: plot title
        :param to_file: bool: to save to file or not
        :param filename: filename to give to output figure
        :param scaling: scaling to apply to plot. Either 'xs' for cross-section or 'widths' for divisiion by bin widths
        :param kwargs: keyword arguments to pass to plotting function
        :return: Figure
        """
        logger.debug(f"Generating histogram {title} for {x} in {self.name}...")

        fig, ax = plt.subplots()
        fig = plt_utils.plot_1d_hist(df=self.df, x=x, bins=bins, fig=fig, ax=ax, scaling=scaling, **kwargs)
        fig.tight_layout()

        if to_file:
            if not filename:
                filename = self.name + '_' + x
            out_png_file = config.paths['plot_dir'] + filename + '.png'
            hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=title)
            fig.savefig(out_png_file, bbox_inches='tight')
            logger.info(f"Figure saved to {out_png_file}")

        return fig

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
            logger.info(f"Generating histogram with cuts for {var_to_plot} in {self.name}...")
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
            logger.info(f"Generating 2d histogram for {x_var}-{y_var} in {self.name}...")
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

    # ===============================
    # ====== DATAFRAME BUILDER ======
    # ===============================
    @classmethod
    def _build_dataframe(cls,
                         datapath: str,
                         TTree_name: str,
                         cut_list_dicts: List[dict],
                         vars_to_cut: List[str],
                         is_slices: bool = False,
                         chunksize: int = 1024,
                         ) -> pd.DataFrame:
        """
        Builds a dataframe from cutfile inputs
        :param datapath: path to root file
        :param TTree_name: name of TTree to extract
        :param cut_list_dicts: list of cut dictionaries
        :param vars_to_cut: list of strings of variables in file to cut on
        :param is_slices: whether or not data is in mass slices
        :param chunksize: chunksize for uproot concat method
        :return: output dataframe containing columns corresponding to necessary variables
        """

        # create list of all necessary values extract
        default_tree_vars = extract_cut_variables(cut_list_dicts, vars_to_cut)
        default_tree_vars.add('weight_mc')
        default_tree_vars.add('eventNumber')

        # get any variables that need to be calculated rather than extracted from ROOT file
        vars_to_calc = {calc_var for calc_var in default_tree_vars if calc_var in derived_vars}
        default_tree_vars -= vars_to_calc

        # get any variables in trees outside the default tree (TTree_name)
        alt_trees = gen_alt_tree_dict(cut_list_dicts)
        if alt_trees:  # remove alt tree variables from variables to extract from default tree
            default_tree_vars -= {var for varlist in alt_trees.values() for var in varlist}

        if logger.level == logging.DEBUG:
            logger.debug(f"Variables to extract from {TTree_name} tree: ")
            for var in default_tree_vars:
                logger.debug(f"  - {var}")
            if alt_trees:
                for tree in alt_trees:
                    logger.debug(f"Variables to extract from {tree} tree: ")
                    for var in alt_trees[tree]:
                        logger.debug(f"  - {var}")
            if vars_to_calc:
                logger.debug("Variables to calculate: ")
                for var in vars_to_calc:
                    logger.debug(f"  - {var}")

        # check that TTree(s) and variables exist in file(s)
        logger.debug(f"Checking TTree and TBranch values in file(s) '{datapath}'...")
        for filepath in glob(datapath):
            with uproot.open(filepath) as file:
                tree_list = [tree.split(';')[0] for tree in file.keys()]  # TTrees are labelled '<name>;<cycle number>'
                if missing_trees := [t for t in [a_t for a_t in alt_trees] + [TTree_name] if t not in tree_list]:
                    raise ValueError(f"TTree(s) '{', '.join(missing_trees)}' not found in file {filepath}")
                else:
                    if missing_branches := [branch for branch in default_tree_vars
                                            if branch not in file[TTree_name].keys()]:
                        raise ValueError(
                            f"Missing TBranch(es) {missing_branches} in TTree '{TTree_name}' of file '{datapath}'.")
            logger.debug(f"All TTrees and variables found in {filepath}...")
        logger.debug("All required TTrees variables found.")

        # check if vars are contained in label dictionary
        if unexpected_vars := [unexpected_var for unexpected_var in default_tree_vars
                               if unexpected_var not in labels_xs]:
            logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                           "Some unexpected behaviour may occur.")

        t1 = time.time()
        # extract pandas dataframe from root file with necessary variables
        if not is_slices:  # If importing inclusive sample
            logger.info(f"Extracting {default_tree_vars} from {datapath}...")
            df = uproot.concatenate(datapath + ':' + TTree_name, default_tree_vars,
                                    library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)
            logger.debug(f"Extracted {len(df)} events.")

            if alt_trees:
                df = cls.__merge_TTrees(df, alt_trees, datapath, TTree_name, chunksize)
            else:
                # pandas merge checks for duplicates, this the only branch with no merge function
                logger.debug("Checking for duplicate events...")
                if (duplicates := df.duplicated(subset='eventNumber')).any():
                    raise ValueError(f"Found {len(duplicates)} duplicate events in datafile {datapath}.")
                logger.debug("No duplicates found.")

        else:  # if importing mass slices
            logger.info(f"Extracting mass slices from {datapath}...")
            default_tree_vars.add('mcChannelNumber')  # to keep track of dataset IDs (DSIDs)

            logger.debug(f"Extracting {default_tree_vars} from {TTree_name} tree...")
            df = uproot.concatenate(datapath + ':' + TTree_name, default_tree_vars, library='pd',
                                    num_workers=config.n_threads, begin_chunk_size=chunksize)
            logger.debug(f"Extracted {len(df)} events.")

            logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
            sumw = uproot.concatenate(datapath + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                      library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)

            logger.debug(f"Calculating sum of weights and merging...")
            sumw = sumw.groupby('dsid').sum()
            df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)
            df.rename(columns={'mcChannelNumber': 'DSID'},
                      inplace=True)  # rename mcChannelNumber to DSID (why are they different)

            if alt_trees:
                df = cls.__merge_TTrees(df, alt_trees, datapath, TTree_name, chunksize)

            if logger.level == logging.DEBUG:
                # sanity check to make sure totalEventsWeighted really is what it says it is
                # also output DSID metadata
                dsids = df['DSID'].unique()
                logger.debug(f"Found {len(dsids)} unique dsid(s).")
                df_id_sub = df[['weight_mc', 'DSID']].groupby('DSID', as_index=False).sum()
                for dsid in dsids:
                    df_id = df[df['DSID'] == dsid]
                    unique_totalEventsWeighted = df_id['totalEventsWeighted'].unique()
                    if len(unique_totalEventsWeighted) != 1:
                        logger.warning("totalEventsWeighted should only have one value per DSID. "
                                       f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} for DSID {dsid}")

                    dsid_weight = df_id_sub[df_id_sub['DSID'] == dsid]['weight_mc'].values[
                        0]  # just checked there's only one value here
                    totalEventsWeighted = df_id['totalEventsWeighted'].values[0]
                    if dsid_weight != totalEventsWeighted:
                        logger.warning(
                            f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                            f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}")

            default_tree_vars.remove('mcChannelNumber')
            default_tree_vars.add('DSID')

        # calculate and combine special derived variables
        if vars_to_calc:

            def row_calc(deriv_var: str, row: pd.Series) -> pd.Series:
                """Helper for applying derived variable calculation function to a dataframe row"""
                row_args = [row[v] for v in derived_vars[deriv_var]['var_args']]
                return derived_vars[deriv_var]['func'](*row_args)

            # save which variables are actually necessary in order to drop extras (keep all columns for now)
            # og_vars = all_vars(cut_list_dicts, vars_to_cut)
            for var in vars_to_calc:
                # compute new column
                temp_cols = derived_vars[var]['var_args']
                logger.info(f"Computing '{var}' column from {temp_cols}...")
                df[var] = df.apply(lambda row: row_calc(var, row), axis=1)

                # # drop unnecessary columns extracted just for calculations
                # to_drop = [var for var in temp_cols if var not in og_vars]
                # logger.debug(f"dropping {to_drop}")
                # df.drop(columns=to_drop, inplace=True)

        t2 = time.time()
        logger.info(f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(t2 - t1))}")

        # properly scale GeV columns
        df = cls.__rescale_to_gev(df)

        return df

    @classmethod
    def __merge_TTrees(cls,
                       df: pd.DataFrame,
                       alt_trees: Dict[str, List[str]],
                       datapath: str,
                       TTree_name: str,
                       chunksize: int = 1024
                       ) -> pd.DataFrame:
        """Merge TTrees across event number"""
        for tree in alt_trees:
            logger.debug(f"Extracting {alt_trees[tree]} from {tree} tree...")
            alt_df = uproot.concatenate(datapath + ":" + tree, alt_trees[tree] + ['eventNumber'],
                                        library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)
            logger.debug(f"Extracted {len(alt_df)} events.")
            logger.debug("Merging with rest of dataframe...")
            try:
                df = pd.merge(df, alt_df, how='left', on='eventNumber', sort=False, copy=False, validate='1:1')
            except pd.errors.MergeError as e:
                err = str(e)
                if err == 'Merge keys are not unique in either left or right dataset; not a one-to-one merge':
                    raise Exception(f"Duplicated events in both '{TTree_name}' and '{tree}' TTrees")
                elif err == 'Merge keys are not unique in left dataset; not a one-to-one merge':
                    raise Exception(f"Duplicated events in '{TTree_name}' TTree")
                elif err == 'Merge keys are not unique in right dataset; not a one-to-one merge':
                    raise Exception(f"Duplicated events in '{tree}' TTree")
                else:
                    raise e
        return df

    @staticmethod
    def _create_cut_columns(df: pd.DataFrame, cut_dicts: List[dict]) -> None:
        """
        Creates boolean columns in dataframe corresponding to each cut
        :param df: input dataframe
        :param cut_dicts: list of dictionaries for each cut to apply
        :return: None, this function applies inplace.
        """
        cut_label = config.cut_label  # get cut label from config

        # use functions of compariton operators in dictionary to make life easier
        # (but maybe a bit harder to read)
        op_dict = {
            '<': op.lt,
            '<=': op.le,
            '=': op.eq,
            '!=': op.ne,
            '>': op.gt,
            '>=': op.ge,
        }

        for cut in cut_dicts:
            if cut['relation'] not in op_dict:
                raise ValueError(f"Unexpected comparison operator: {cut['relation']}.")
            if not cut['is_symmetric']:
                df[cut['name'] + cut_label] = op_dict[cut['relation']](df[cut['cut_var']], cut['cut_val'])
            else:  # take absolute value instead
                df[cut['name'] + cut_label] = op_dict[cut['relation']](df[cut['cut_var']].abs(), cut['cut_val'])

        # print cutflow output
        name_len = max([len(cut['name']) for cut in cut_dicts])
        var_len = max([len(cut['cut_var']) for cut in cut_dicts])
        logger.info('')
        logger.info("========== CUTS USED ============")
        for cut in cut_dicts:
            if not cut['is_symmetric']:
                logger.info(
                    f"{cut['name']:<{name_len}}: {cut['cut_var']:>{var_len}} {cut['relation']} {cut['cut_val']}")
            else:
                logger.info(
                    f"{cut['name']:<{name_len}}: {'|' + cut['cut_var']:>{var_len}}| {cut['relation']} {cut['cut_val']}")
        logger.info('')

    @staticmethod
    def __gen_weight_column(df: pd.DataFrame,
                            weight_mc_col: str = 'weight_mc',
                            global_scale: float = config.lumi
                            ) -> pd.Series:
        """Returns series of weights based off weight_mc column"""
        if weight_mc_col not in df.columns:
            raise KeyError(f"'{weight_mc_col}' column does not exist.")
        return df[weight_mc_col].map(lambda w: global_scale if w > 0 else -1 * global_scale)

    @staticmethod
    def __gen_weight_column_slices(df: pd.DataFrame,
                                   mc_weight_col: str = 'weight_mc',
                                   tot_weighted_events_col: str = 'totalEventsWeighted',
                                   global_scale: float = config.lumi,
                                   ) -> pd.Series:
        """Returns series of weights for mass slices based off weight_mc column and total events weighed"""
        # TODO?: For efficiency, perform batchwise across DSID
        if mc_weight_col not in df.columns:
            raise KeyError(f"'{mc_weight_col}' column not in dataframe.")
        if tot_weighted_events_col not in df.columns:
            raise KeyError(f"'{tot_weighted_events_col}' column not in dataframe.")

        return global_scale * (df[mc_weight_col] * df[mc_weight_col].abs()) / df[tot_weighted_events_col]

    @staticmethod
    def __rescale_to_gev(df: pd.DataFrame) -> pd.DataFrame:
        """rescales to GeV because athena's default output is in MeV for some reason"""
        GeV_columns = [column for column in df.columns
                       if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]
        df[GeV_columns] /= 1000
        if GeV_columns:
            logger.debug(f"Rescaled columns {GeV_columns} to GeV.")
        else:
            logger.debug(f"No columns rescaled to GeV.")
        return df

    @staticmethod
    def _cut_on_cutgroup(df: pd.DataFrame,
                         cutgroups: OrderedDict[str, List[str]],
                         group: str,
                         ) -> pd.DataFrame:
        """Cuts on cutgroup on input dataframe or series"""
        cut_cols = [cut_name + config.cut_label for cut_name in cutgroups[group]]
        cut_data = df[df[cut_cols].all(1)]
        return cut_data
