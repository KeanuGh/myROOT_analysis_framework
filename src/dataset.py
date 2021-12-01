import logging
import operator as op
import pickle as pkl
import time
from dataclasses import dataclass, field
from glob import glob
from itertools import combinations
from typing import Optional, Union, List, Dict, OrderedDict, Tuple
from warnings import warn

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot
from awkward import to_pandas

import src.config as config
import utils.file_utils as file_utils
import utils.plotting_utils as plt_utils
from src.cutfile import Cutfile
from src.cutflow import Cutflow
from utils.axis_labels import labels_xs
from utils.var_helpers import derived_vars, OtherVar

logger = logging.getLogger('analysis')

# total dataset luminosity per year (pb-1)
lumi_year = {
    '2015': 3219.56,
    '2017': 44307.4,
    '2018': 58450.1,
    '2015+2016': 32988.1 + 3219.56
}


@dataclass
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.
    TODO: Save histograms as dataset attributes? Ability to perform operations across hisograms? Custom histogram class?
    TODO: check branch types and perform dataframe merge in batches
    """
    name: str
    data_path: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    cutfile_path: str  # path to cutfile
    year: str  # data year '2015', '2017', '2018' or '2015+2016'

    lumi: float = field(init=False)  # generated from year
    df: pd.DataFrame = field(init=False, repr=False)  # stores the actual data in a dataframe
    cutflow: Cutflow = field(init=False)

    pkl_path: str = None  # where the dataframe pickle file will be stored
    lepton: str = 'lepton'  # name of charged DY lepton channel in dataset (if applicable)
    chunksize: int = 1024  # chunksize for uproot ROOT file import
    force_rebuild: bool = False  # whether to force rebuild dataset
    validate_missing_events: bool = True  # whether to check for missing events
    validate_duplicated_events: bool = True  # whether to check for duplicated events
    validate_sumofweights: bool = True  # whether to whether to check sumofweights is sum of weight_mc for DSID
    _weight_column = 'weight'  # name of weight column in dataframe

    def __post_init__(self):
        """Dataset generation pipeline"""
        logger.info("")
        logger.info("=" * (42 + len(self.name)))
        logger.info(f"======== INITIALISING DATASET '{self.name}' =========")
        logger.info("=" * (42 + len(self.name)))
        if not file_utils.file_exists(self.data_path):
            raise FileExistsError(f"File {self.data_path} not found.")

        self.lumi = lumi_year[self.year]
        logger.info(f"Year: {self.year}")
        logger.info(f"Data luminosity: {self.lumi}")

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
        logger.info(f"Parsting cutfile '{self.cutfile_path}' for {self.name}...")
        self.cutfile = Cutfile(self.cutfile_path)
        self._build_df = True if self.force_rebuild else self.cutfile.if_build_dataframe(self.pkl_path)
        if self.cutfile.if_make_cutfile_backup():
            self.cutfile.backup_cutfile(self.name)

        # some debug information
        logger.debug("")
        logger.debug("DATASET INPUT OPTIONS: ")
        logger.debug("----------------------------")
        logger.debug(f"Input file(s):  {self.data_path}")
        logger.debug(f"TTree: {self.TTree_name}")
        logger.debug(f"Cutfile: {self.cutfile_path}")
        logger.debug(f"grouped cutflow: {self.cutfile.options['grouped cutflow']}")
        logger.debug(f"sequential cutflow: {self.cutfile.options['sequential']}")
        logger.debug(f"Forced dataset rebuild: {self.force_rebuild}")
        logger.debug(f"Validate missing events: {self.validate_missing_events}")
        logger.debug(f"Validate duplicated events: {self.validate_duplicated_events}")
        logger.debug(f"Validate sum of weights: {self.validate_sumofweights}")
        logger.debug("----------------------------")
        logger.debug("")

        # GENERATE DATAFRAME
        # ========================
        """Define pipeline for building dataset dataframe"""
        # extract and clean data
        if self._build_df:
            logger.info(f"Building {self.name} dataframe from {self.data_path}...")
            self.df = self._build_dataframe(calc_vars_dict=derived_vars,
                                            validate_missing_events=self.validate_missing_events,
                                            validate_duplicated_events=self.validate_duplicated_events,
                                            validate_sumofweights=self.validate_sumofweights)

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
        self.df['total_event_weight'] = self.__total_event_weight()

        # print some dataset ID metadata
        # TODO: avg event weight
        if logger.level == logging.DEBUG:
            logger.info(f"DATASET INFO FOR {self.name}:")
            logger.debug("DSID       n_events   sum_w         x-s fb        lumi fb-1")
            logger.debug("==========================================================")
            for dsid in np.sort(self.df['DSID'].unique()):
                df_id = self.df[self.df['DSID'] == dsid]
                logger.debug(f"{int(dsid):<10} "
                             f"{len(df_id):<10} "
                             f"{df_id['weight_mc'].sum():<10.6e}  "
                             f"{self._get_cross_section(df_id):<10.6e}  "
                             f"{self.lumi:<10.6e}")

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

    def __repr__(self):
        return f'Dataset("{self.name}",TTree:"{self.TTree_name},Events:{self.__len__}'

    def __str__(self):
        return f'{self.name},TTree:"{self.TTree_name}",Events:{self.__len__}'

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
    def _get_cross_section(cls, df: pd.DataFrame, n_events=None, weight_mc_col: str = 'weight_mc') -> float:
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
    def _get_luminosity(cls, df: pd.DataFrame, xs: float = None, weight_col: str = 'weight') -> float:
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
    def _build_dataframe(self,
                         calc_vars_dict: Dict[str, OtherVar],
                         chunksize: int = 1024,
                         validate_missing_events: bool = True,
                         validate_duplicated_events: bool = True,
                         validate_sumofweights: bool = True,
                         ) -> pd.DataFrame:
        """
        Builds a dataframe from cutfile inputs
        :param calc_vars_dict: list of possible calculated variables (see utils.var_helpers)
        :param chunksize: chunksize for uproot concat method
        :param validate_missing_events: whether to check for missing events
        :param validate_duplicated_events: whether to check for duplicated events
        :param validate_sumofweights: whether to check sum of weights against weight_mc
        :return: output dataframe containing columns corresponding to necessary variables
        """

        # get variables to extract from dataframe and their TTrees
        tree_dict, vars_to_calc = self.cutfile.extract_variables(calc_vars_dict)

        # get default tree variables
        default_tree_vars = tree_dict.pop('na')
        default_tree_vars |= tree_dict.pop(self.TTree_name, set())
        default_tree_vars -= vars_to_calc
        default_tree_vars.add('mcChannelNumber')  # to keep track of dataset IDs (DSIDs)

        # pull eventNumber from other trees
        for tree in tree_dict:
            tree_dict[tree].add('eventNumber')

        # required variables for weighting and merging
        default_tree_vars |= {'weight_mc', 'weight_pileup', 'eventNumber'}
        # TODO: Perhaps not the best way of doing this?
        if 'nominal' in self.TTree_name.lower():
            logger.debug("Detected reco dataset, will pull 'weight_leptonSF' and 'weight_KFactor'")
            default_tree_vars |= {'weight_leptonSF', 'weight_KFactor'}
        elif 'truth' in self.TTree_name.lower():
            logger.debug("Detected truth dataset, will pull 'KFactor_weight_truth'")
            default_tree_vars.add('KFactor_weight_truth')
        else:
            logger.info("Neither truth nor reco dataset detected.")

        if logger.level == logging.DEBUG:
            logger.debug(f"Variables to extract from {self.TTree_name} tree: ")
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
        logger.debug(f"Checking TTree and TBranch values in file(s) '{self.data_path}'...")
        for filepath in glob(self.data_path):
            with uproot.open(filepath) as file:
                tree_list = [tree.split(';')[0] for tree in file.keys()]  # TTrees are labelled '<name>;<cycle number>'
                if missing_trees := [t for t in [a_t for a_t in tree_dict] + [self.TTree_name] if t not in tree_list]:
                    raise ValueError(f"TTree(s) '{', '.join(missing_trees)}' not found in file {filepath}")
                else:
                    if missing_branches := [branch for branch in default_tree_vars
                                            if branch not in file[self.TTree_name].keys()]:
                        raise ValueError(f"Missing TBranch(es) {missing_branches} in TTree "
                                         f"'{self.TTree_name}' of file '{self.data_path}'.")
            logger.debug(f"All TTrees and variables found in {filepath}...")
        logger.debug("All required TTrees variables found.")

        # check if vars are contained in label dictionary
        if unexpected_vars := [unexpected_var for unexpected_var in default_tree_vars
                               if unexpected_var not in labels_xs]:
            logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                           "Some unexpected behaviour may occur.")

        t1 = time.time()
        logger.debug(f"Extracting {default_tree_vars} from {self.TTree_name} tree...")
        df = to_pandas(uproot.concatenate(self.data_path + ':' + self.TTree_name, default_tree_vars,
                                          num_workers=config.n_threads, begin_chunk_size=chunksize))
        logger.debug(f"Extracted {len(df)} events.")

        logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = to_pandas(uproot.concatenate(self.data_path + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                            num_workers=config.n_threads, begin_chunk_size=chunksize))

        logger.debug(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)
        df.rename(columns={'mcChannelNumber': 'DSID'}, inplace=True)  # rename mcChannelNumber to DSID

        # TODO: Drop duplicated events, set event number as index?

        # merge TTrees
        if validate_duplicated_events:
            validation = '1:1'
            logger.info("Validating duplicated events")
        else:
            validation = 'm:m'
            logger.info("Skipping dulpacted events validation")

        for tree in tree_dict:
            logger.debug(f"Extracting {tree_dict[tree]} from {tree} tree...")
            alt_df = to_pandas(uproot.concatenate(self.data_path + ":" + tree, tree_dict[tree],
                                                  num_workers=config.n_threads, begin_chunk_size=chunksize))
            logger.debug(f"Extracted {len(alt_df)} events.")

            if validate_missing_events:
                logger.info("Checking for missing events..")
                # test for missing events
                if n_missing := len(df[~df['eventNumber'].isin(alt_df['eventNumber'])]):
                    raise Exception(f"Found {n_missing} events in '{self.TTree_name}' tree not found in '{tree}' tree")
                else:
                    logger.debug(f"All events in {self.TTree_name} tree found in {tree} tree")
            else:
                logger.info("Skipping missing events check")

            logger.debug("Merging with rest of dataframe...")
            try:
                df = pd.merge(df, alt_df, how='left', on='eventNumber', sort=False, copy=False, validate=validation)
            except pd.errors.MergeError as e:
                dup_l = df['eventNumber'][df.duplicated('eventNumber')]
                n_l = len(dup_l)
                evnt_l = dup_l.unique()
                dup_r = df['eventNumber'][alt_df.duplicated('eventNumber')]
                n_r = len(dup_r)
                evnt_r = dup_r.unique()

                err = str(e)
                if err == 'Merge keys are not unique in either left or right dataset; not a one-to-one merge':
                    raise Exception(f"{n_l} duplicated events in '{self.TTree_name}' TTree. Events: {evnt_l}, "
                                    f"and {n_r} in '{tree}' TTree. Events: {evnt_r}")
                elif err == 'Merge keys are not unique in left dataset; not a one-to-one merge':
                    raise Exception(f"{n_l} duplicated events in '{self.TTree_name}' TTree. Events: {evnt_l}")
                elif err == 'Merge keys are not unique in right dataset; not a one-to-one merge':
                    raise Exception(f"{n_r} duplicated events in '{tree}' TTree. Events: {evnt_r}")
                else:
                    raise e

        if validate_sumofweights:
            # sanity check to make sure totalEventsWeighted really is what it says it is
            # also output DSID metadata
            dsids = df['DSID'].unique()
            logger.info(f"Found {len(dsids)} unique dsid(s).")
            df_id_sub = df[['weight_mc', 'DSID']].groupby('DSID', as_index=False).sum()
            for dsid in dsids:
                df_id = df.loc[df['DSID'] == dsid]
                unique_totalEventsWeighted = df_id['totalEventsWeighted'].unique()
                if len(unique_totalEventsWeighted) != 1:
                    logger.warning("totalEventsWeighted should only have one value per DSID. "
                                   f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} for DSID {dsid}")

                dsid_weight = df_id_sub.loc[df_id_sub['DSID'] == dsid]['weight_mc'].values[0]  # just take first value
                totalEventsWeighted = df_id['totalEventsWeighted'].values[0]
                if dsid_weight != totalEventsWeighted:
                    logger.warning(
                        f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                        f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}")

            default_tree_vars.remove('mcChannelNumber')
            default_tree_vars.add('DSID')
        else:
            logger.info("Skipping sum of weights validation")

        # calculate and combine special derived variables
        if vars_to_calc:

            # save which variables are actually necessary in order to drop extras (keep all columns for now)
            # og_vars = all_vars(cut_list_dicts, vars_to_cut)
            for var in vars_to_calc:
                # compute new column
                temp_cols = calc_vars_dict[var]['var_args']
                func = calc_vars_dict[var]['func']
                str_args = calc_vars_dict[var]['var_args']
                logger.info(f"Computing '{var}' column from {temp_cols}...")
                df[var] = func(df, *str_args)

                # # drop unnecessary columns extracted just for calculations
                # to_drop = [var for var in temp_cols if var not in og_vars]
                # logger.debug(f"dropping {to_drop}")
                # df.drop(columns=to_drop, inplace=True)

        t2 = time.time()
        logger.info(f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(t2 - t1))}")

        # properly scale GeV columns
        self.__rescale_to_gev(df)

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

    def __total_event_weight(self,
                             mc_weight: str = 'weight_mc',
                             tot_weighted_events: str = 'totalEventsWeighted',
                             kFactor_reco: str = 'weight_KFactor',
                             kFactor_truth: str = 'KFactor_weight_truth',
                             lepton_SF: str = 'weight_leptonSF',
                             pileup_weight: str = 'weight_pileup',
                             ) -> pd.Series:
        """
        Calculate total event weights

        lumi_data taken from year
        mc_weight = +/-1 * cross_section
        scale factor = weight_leptonSF
        kFactor = weight_kFactor OR kFactor_weight_truth for truth-level
        pileup = weight_pileup
        truth_weight = kFactor * pileup
        recoweight = scalefactors
        lumi_weight = mc_weight * lumi_data / sum of event weights

        total event weight = lumi_weight * truth_weight * reco_weight

        This is done in one line for efficiency with pandas (sorry)
        """
        if kFactor_truth in self.df:
            kFactor = kFactor_truth
            SF = 1.
        elif kFactor_reco in self.df:
            kFactor = kFactor_reco
            SF = self.df[lepton_SF]
        else:
            raise ValueError(f"Missing kFactors in dataset {self.name}")

        return \
            self.lumi * self.df[mc_weight] * abs(self.df[mc_weight]) / self.df[tot_weighted_events] * \
            self.df[kFactor] * self.df[pileup_weight] * SF

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
    # TODO: strip special characters from filepaths
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
            with open(config.paths['pkl_hist_dir'] + plot_label + '_' + var + '_1d_cutgroups_ratios.pkl', 'wb') as f:
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

    def plot_mass_slices(self,
                         var: str,
                         bins: Union[List[float], Tuple[int, float, float]] = (30, 0, 5000),
                         logbins: bool = False,
                         logx: bool = False,
                         logy: bool = True,
                         xlabel: str = '',
                         ylabel: str = '',
                         title: str = '',
                         **kwargs
                         ) -> None:
        logger.info(f'Plotting {var} in {self.name} as slices...')

        bh_axis = plt_utils.get_axis(bins, logbins)

        # per dsid
        for dsid in self.df['DSID'].unique():
            dsid_df = self.df.loc[self.df['DSID'] == dsid]
            hist = bh.Histogram(bh_axis, storage=bh.storage.Weight())
            hist.fill(dsid_df[var], weight=dsid_df['total_event_weight'], threads=config.n_threads)
            hep.histplot(hist, label=dsid, **kwargs)
        # inclusive
        hist = bh.Histogram(bh_axis, storage=bh.storage.Weight())
        hist.fill(self.df[var], weight=self.df['total_event_weight'])
        hep.histplot(hist, label='Inclusive', color='k', **kwargs)

        _xlabel, _ylabel = plt_utils.get_axis_labels(var, self.lepton)
        plt.xlabel(xlabel if xlabel else _xlabel)
        plt.ylabel(ylabel if ylabel else _ylabel)
        plt.legend(fontsize=10, ncol=2)
        if logx:
            plt.semilogx()
        if logy:
            plt.semilogy()
        hep.atlas.label(italic=(True, True), llabel='Internal', rlabel=title)

        filename = config.paths['plot_dir'] + self.name + '_' + var + '_SLICES.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        logger.info(f'Saved mass slice plot of {var} in {self.name} to {filename}')
        plt.clf()

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
