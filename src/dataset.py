import logging
import operator as op
import pickle as pkl
import time
from dataclasses import dataclass, field
from glob import glob
from itertools import combinations
from typing import Optional, Union, List, Dict, Set, OrderedDict
from warnings import warn

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot

import src.config as config
import utils.file_utils as file_utils
import utils.plotting_utils as plt_utils
from src.cutfile import Cutfile
from src.cutflow import Cutflow
from utils.axis_labels import labels_xs
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
    chunksize: int = 1024  # chunksize for uproot ROOT file import
    _weight_column = 'weight'  # name of weight column in dataframe

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
        self.cutfile = Cutfile(self.cutfile_path)
        self._rebuild = self.cutfile.if_build_dataframe(self.pkl_path)

        # some debug information
        logger.debug("DATASET INPUT OPTIONS: ")
        logger.debug("----------------------------")
        logger.debug(f"Input file(s):  {self.data_path}")
        logger.debug(f"TTree: {self.TTree_name}")
        logger.debug(f"Cutfile: {self.cutfile_path}")
        logger.debug(f"Slices: {self.is_slices}")
        logger.debug(f"grouped cutflow: {self.cutfile.options['grouped cutflow']}")
        logger.debug(f"sequential cutflow: {self.cutfile.options['sequential']}")
        logger.debug(f"Forced dataset rebuild: {config.force_rebuild}")
        logger.debug("----------------------------")
        logger.debug("")

        # GENERATE DATAFRAME
        # ========================
        """Define pipeline for building dataset dataframe"""
        # extract and clean data
        if self._rebuild or config.force_rebuild:
            logger.info(f"Building {self.name} dataframe from {self.data_path}...")
            self.df = self._build_dataframe(datapath=self.data_path,
                                            TTree_name=self.TTree_name,
                                            cut_list_dicts=self.cutfile.cut_dicts,
                                            vars_to_cut=self.cutfile.vars_to_cut,
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
            self.df[self._weight_column] = self.__gen_weight_column_slices(self.df)
        else:
            self.df[self._weight_column] = self.__gen_weight_column(self.df)

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
        self._create_cut_columns(self.df, cut_dicts=self.cutfile.cut_dicts)

        # GENERATE CUTFLOW
        # ========================
        self.cutflow = Cutflow(self.df, self.cutfile.cut_dicts,
                               self.cutfile.cutgroups if self.cutfile.options['grouped cutflow'] else None,
                               self.cutfile.options['sequential'])

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
        return self._get_luminosity(self.df, xs=self.cross_section, weight_col=self._weight_column)

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
    # ====== DATAFRAME BUILDER ======
    # ===============================
    @classmethod
    def _build_dataframe(cls,
                         datapath: str,
                         TTree_name: str,
                         cut_list_dicts: List[dict],
                         vars_to_cut: Set[str],
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

        # get variables to extract from dataframe and their TTrees
        tree_dict, vars_to_calc = Cutfile.extract_cut_variables(cut_list_dicts, derived_vars)

        print(tree_dict)

        # get default tree variables
        default_tree_vars = tree_dict.pop('na')
        default_tree_vars |= tree_dict.pop(TTree_name, set())
        default_tree_vars |= vars_to_cut
        default_tree_vars.add('weight_mc')
        default_tree_vars.add('eventNumber')

        print(default_tree_vars)

        if logger.level == logging.DEBUG:
            logger.debug(f"Variables to extract from {TTree_name} tree: ")
            for var in default_tree_vars:
                logger.debug(f"  - {var}")
            if tree_dict:
                for tree in tree_dict:
                    logger.debug(f"Variables to extract from {tree} tree: ")
                    for var in tree_dict[tree]:
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
                if missing_trees := [t for t in [a_t for a_t in tree_dict] + [TTree_name] if t not in tree_list]:
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

            if tree_dict:
                df = cls.__merge_TTrees(df, tree_dict, datapath, TTree_name, chunksize)
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

            if tree_dict:
                df = cls.__merge_TTrees(df, tree_dict, datapath, TTree_name, chunksize)

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
        cls.__rescale_to_gev(df)

        return df

    @classmethod
    def __merge_TTrees(cls,
                       df: pd.DataFrame,
                       alt_trees: Dict[str, Set[str]],
                       datapath: str,
                       TTree_name: str,
                       chunksize: int = 1024
                       ) -> pd.DataFrame:
        """Merge TTrees across event number"""
        for tree in alt_trees:
            logger.debug(f"Extracting {alt_trees[tree]} from {tree} tree...")
            alt_df = uproot.concatenate(datapath + ":" + tree, alt_trees[tree].add('eventNumber'),
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
    def __rescale_to_gev(df: pd.DataFrame) -> None:
        """rescales to GeV because athena's default output is in MeV for some reason"""
        if GeV_columns := [column for column in df.columns
                           if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]:
            df[GeV_columns] /= 1000
            logger.debug(f"Rescaled columns {GeV_columns} to GeV.")
        else:
            logger.debug(f"No columns rescaled to GeV.")

    @staticmethod
    def _cut_on_cutgroup(df: pd.DataFrame,
                         cutgroups: OrderedDict[str, List[str]],
                         group: str,
                         ) -> pd.DataFrame:
        """Cuts on cutgroup on input dataframe or series"""
        cut_cols = [cut_name + config.cut_label for cut_name in cutgroups[group]]
        cut_data = df[df[cut_cols].all(1)]
        return cut_data

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
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

    def plot_all_with_cuts(self,
                           bins: Union[tuple, list] = (30, 1, 500),
                           scaling: Optional[str] = None,
                           **kwargs
                           ) -> None:
        """
        Plots all variables in Dataset to cut from cutfile with each cutgroup applied

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param scaling: either 'xs':     cross section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param kwargs: keyword arguments to pass to plot_1d_overlay_and_acceptance_cutgroups()
        """
        for var_to_plot in self.cutfile.vars_to_cut:
            logger.info(f"Generating histogram with cuts for {var_to_plot} in {self.name}...")
            self.plot_1d_overlay_and_acceptance_cutgroups(
                var=var_to_plot,
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
        if len(self.cutfile.vars_to_cut) < 2:
            raise Exception("Need at least two plotting variables to make 2D plot")

        for x_var, y_var in combinations(self.cutfile.vars_to_cut, 2):
            # binning
            _, xbins = plt_utils.getbins(x_var)
            _, ybins = plt_utils.getbins(y_var)
            if not xbins:
                xbins = bins
            if not ybins:
                ybins = bins
            logger.info(f"Generating 2d histogram for {x_var}-{y_var} in {self.name}...")
            self.plot_2d_cutgroups(x_var=x_var, y_var=y_var,
                                   xbins=xbins, ybins=ybins,
                                   plot_label=self.name,
                                   **kwargs)

    def plot_1d_overlay_and_acceptance_cutgroups(self,
                                                 var: str,
                                                 bins: Union[tuple, list],
                                                 lumi: Optional[float] = None,
                                                 scaling: Optional[str] = None,
                                                 log_x: bool = False,
                                                 to_pkl: bool = False,
                                                 plot_width=7,
                                                 plot_height=7,
                                                 plot_label: Optional[str] = None,
                                                 ) -> None:
        """Plots overlay of cutgroups and acceptance (ratio) plots"""
        # TODO: write docs for this function
        fig, (fig_ax, accept_ax) = plt.subplots(2, 1,
                                                figsize=(plot_width * 1.2, plot_height),
                                                gridspec_kw={'height_ratios': [3, 1]})
        hists = dict()

        # check if variable needs to be specially binned
        is_logbins, alt_bins = plt_utils.getbins(var)
        if alt_bins:
            bins = alt_bins

        # INCLUSIVE PLOT
        # ================
        h_inclusive = plt_utils.histplot_1d(var_x=self.df[var],
                                            weights=self.df[self._weight_column],
                                            bins=bins, fig_axis=fig_ax,
                                            yerr='sumw2',
                                            lumi=lumi, scaling=scaling,
                                            label='Inclusive',
                                            is_logbins=is_logbins,
                                            color='k', linewidth=2)
        if to_pkl:
            hists['inclusive'] = h_inclusive

        # PLOT CUTS
        # ================
        for cutgroup in self.cutfile.cutgroups.keys():
            logger.info(f"    - generating cutgroup '{cutgroup}'")
            cut_df = Dataset._cut_on_cutgroup(self.df, self.cutfile.cutgroups, cutgroup)
            weight_cut = cut_df[self._weight_column]
            var_cut = cut_df[var]

            h_cut = plt_utils.histplot_1d(var_x=var_cut, weights=weight_cut,
                                          bins=bins, fig_axis=fig_ax,
                                          lumi=lumi, scaling=scaling,
                                          label=cutgroup,
                                          is_logbins=is_logbins)
            if to_pkl:
                hists[cutgroup] = h_cut

            # RATIO PLOT
            # ================
            hep.histplot(h_cut.view().value / h_inclusive.view().value,
                         bins=h_cut.axes[0].edges, ax=accept_ax, label=cutgroup,
                         color=fig_ax.get_lines()[-1].get_color())

        # AXIS FORMATTING
        # ==================
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)

        # figure plot
        fig_ax = plt_utils.set_fig_1d_axis_options(axis=fig_ax, var_name=var, bins=bins,
                                                   scaling=scaling, is_logbins=is_logbins,
                                                   logy=True, logx=log_x, lepton=self.lepton)
        fig_ax.legend()
        fig_ax.axes.xaxis.set_visible(False)

        # ratio plot
        plt_utils.set_fig_1d_axis_options(axis=accept_ax, var_name=var,
                                          bins=bins, is_logbins=is_logbins, lepton=self.lepton)
        accept_ax.set_ylabel("Acceptance")

        # save figure
        if to_pkl:
            with open(config.paths['pkl_hist_dir'] + plot_label + '_' + var + '_1d_cutgroups_ratios.pkl',
                      'wb') as f:
                pkl.dump(hists, f)
                logger.info(f"Saved pickle file to {f.name}")
        hep.atlas.label(llabel="Internal", loc=0, ax=fig_ax, rlabel=plot_label)
        out_png_file = config.paths['plot_dir'] + f"{var}_{str(scaling)}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        logger.info(f"Figure saved to {out_png_file}")
        fig.clf()  # clear for next plot

    def plot_2d_cutgroups(self,
                          x_var: str, y_var: str,
                          xbins: Union[tuple, list], ybins: Union[tuple, list],
                          plot_label: str = '',
                          is_logz: bool = True,
                          to_pkl: bool = False,
                          ) -> None:
        """
        Runs over cutgroups in dictrionary and plots 2d histogram for each group

        :param x_var: column in dataframe to plot on x axis
        :param y_var: column in dataframe to plot on y axis
        :param xbins: binning in x
        :param ybins: binning in y
        :param plot_label: plot title
        :param is_logz: whether display z-axis logarithmically
        :param to_pkl: whether to save histograms as pickle file
        """
        hists = dict()

        # INCLUSIVE
        fig, ax = plt.subplots(figsize=(7, 7))
        weight_cut = self.df[self._weight_column]
        x_vars = self.df[x_var]
        y_vars = self.df[y_var]

        out_path = config.paths['plot_dir'] + f"2d_{x_var}-{y_var}_inclusive.png"
        hist = plt_utils.histplot_2d(
            var_x=x_vars, var_y=y_vars,
            xbins=xbins, ybins=ybins,
            ax=ax, fig=fig,
            weights=weight_cut,
            is_z_log=is_logz,
        )
        if to_pkl:
            hists['inclusive'] = hist

        # get axis labels
        xlabel, _ = plt_utils.get_axis_labels(str(x_var), self.lepton)
        ylabel, _ = plt_utils.get_axis_labels(str(y_var), self.lepton)

        hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - inclusive', loc=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.savefig(out_path, bbox_inches='tight')
        logger.info(f"printed 2d histogram to {out_path}")
        plt.close(fig)

        for cutgroup in self.cutfile.cutgroups:
            logger.info(f"    - generating cutgroup '{cutgroup}'")
            fig, ax = plt.subplots(figsize=(7, 7))

            cut_df = Dataset._cut_on_cutgroup(self.df, self.cutfile.cutgroups, cutgroup)
            weight_cut = cut_df[self._weight_column]
            x_vars = cut_df[x_var]
            y_vars = cut_df[y_var]

            out_path = config.paths['plot_dir'] + f"2d_{x_var}-{y_var}_{cutgroup}.png"
            hist = plt_utils.histplot_2d(
                var_x=x_vars, var_y=y_vars,
                xbins=xbins, ybins=ybins,
                ax=ax, fig=fig,
                weights=weight_cut,
                is_z_log=is_logz,
            )
            if to_pkl:
                hists[cutgroup] = hist

            # get axis labels
            xlabel, _ = plt_utils.get_axis_labels(str(x_var), self.lepton)
            ylabel, _ = plt_utils.get_axis_labels(str(y_var), self.lepton)

            hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - ' + cutgroup, loc=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            fig.savefig(out_path, bbox_inches='tight')
            logger.info(f"printed 2d histogram to {out_path}")
            plt.close(fig)

        if to_pkl:
            with open(config.paths['pkl_hist_dir'] + plot_label + f"_{x_var}-{y_var}_2d.pkl", 'wb') as f:
                pkl.dump(hists, f)
                logger.info(f"Saved pickle file to {f.name}")

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
            if self.cutfile.options['sequential']:
                self.cutflow.print_histogram('cummulative')
            else:
                warn("Sequential cuts cannot generate a cummulative cutflow")
        if a_ratio:
            if self.cutfile.options['sequential']:
                self.cutflow.print_histogram('a_ratio')
            else:
                warn("Sequential cuts can't generate cummulative cutflow. "
                     "Ratio of cuts to acceptance will be generated instead.")
                self.cutflow.print_histogram('ratio')
