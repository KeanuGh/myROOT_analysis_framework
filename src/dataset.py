import logging
import operator as op
import pickle as pkl
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Union, List, OrderedDict, Tuple, Dict, Iterable

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import uproot
from awkward import to_pandas
from numpy.typing import ArrayLike

import src.config as config
import utils.file_utils as file_utils
import utils.plotting_utils as plt_utils
from src.cutfile import Cutfile
from src.cutflow import Cutflow
from src.histogram import Histogram1D
from utils.axis_labels import labels_xs
from utils.var_helpers import derived_vars

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
    TODO: allow building of dataframe inputting only a TTree dictionary
    TODO: migrate all plotting functions to new histogram
    """
    # INPUT
    name: str
    data_path: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    cutfile_path: str  # path to cutfile
    year: str  # data year '2015', '2017', '2018' or '2015+2016'
    label: str  # label for plot title or legends
    logger: logging.Logger  # a logger to use

    # GENERATED
    lumi: float = field(init=False)  # generated from year
    df: pd.DataFrame = field(init=False, repr=False)  # stores the actual dataset
    cutflow: Cutflow = field(init=False, repr=False)
    cutfile: Cutfile = field(init=False, repr=False)

    # OPTIONS
    paths: Dict[str, str]  # file output paths
    hard_cut: Union[str, List[str]] = None  # name(s) of cut(s) that should be applied to dataframe and not as columns
    reco: bool = False  # whether dataset contains reconstructed data
    truth: bool = False  # whether dataset contains truth data
    to_pkl: bool = True  # whether to output to a pickle file
    pkl_out_dir: str = ''  # where the dataframe pickle file will be stored
    lepton: str = 'lepton'  # name of charged DY lepton channel in dataset (if applicable)
    chunksize: int = 1024  # chunksize for uproot ROOT file import
    force_rebuild: bool = False  # whether to force rebuild dataset
    validate_missing_events: bool = True  # whether to check for events in reco tree that are not in truth tree
    validate_duplicated_events: bool = True  # whether to check for duplicated events
    validate_sumofweights: bool = True  # whether to check sumofweights is sum of weight_mc for DSID
    _weight_column = 'weight'  # name of weight column in dataframe

    def __post_init__(self):
        """Dataset generation pipeline"""
        self.logger.info("")
        self.logger.info("=" * (42 + len(self.name)))
        self.logger.info(f"======== INITIALISING DATASET '{self.name}' =========")
        self.logger.info("=" * (42 + len(self.name)))
        if not file_utils.file_exists(self.data_path):
            raise FileExistsError(f"File {self.data_path} not found.")

        for name, path in self.paths.items():
            # for degbug print output paths
            self.logger.debug(f"{name}: {path}")

        self.lumi = lumi_year[self.year]
        self.logger.info(f"Year: {self.year}")
        self.logger.info(f"Data luminosity: {self.lumi}")

        # initialise pickle filepath with given name
        if not self.pkl_out_dir:
            self.pkl_out_dir = self.paths['pkl_df_dir']
        if self.pkl_out_dir:
            self.pkl_path = self.pkl_out_dir + self.name + '_df.pkl'
        else:
            self.pkl_path = self.paths['pkl_df_dir'] + self.name + '_df.pkl'

        if self.logger.level == logging.DEBUG:
            # log contents of cutfile
            self.logger.debug("CUTFILE CONTENTS")
            self.logger.debug('---------------------------')
            with open(self.cutfile_path) as f:
                for line in f.readlines():
                    self.logger.debug(line.rstrip('\n'))
            self.logger.debug('---------------------------')

        # READ AND GET OPTIONS FROM CUTFILE
        # ========================
        self.logger.info(f"Parsting cutfile '{self.cutfile_path}' for {self.name}...")
        self.cutfile = Cutfile(self.cutfile_path, self.paths['backup_cutfiles_dir'], logger=self.logger)

        self.logger.info('')
        self.logger.info("========== CUTS USED ============")
        self.log_cuts()
        self.logger.info('')

        self._build_df = True if self.force_rebuild else self.cutfile.if_build_dataframe(self.pkl_path)

        self._tree_dict, self._vars_to_calc = self.cutfile.extract_variables(derived_vars)

        # get set unlabeled variables in cutfile as being in default tree
        if self.TTree_name in self._tree_dict:
            self._tree_dict[self.TTree_name] |= self._tree_dict.pop('na', set())
        else:
            self._tree_dict[self.TTree_name] = self._tree_dict.pop('na', set())

        # only add these to 'main tree' to avoid merge issues
        self._tree_dict[self.TTree_name] |= {'weight_mc', 'weight_pileup'}

        reco_flag = False
        truth_flag = False
        for tree in self._tree_dict:
            # add necessary metadata to all trees
            self._tree_dict[tree] |= {'mcChannelNumber', 'eventNumber'}
            self._tree_dict[tree] -= self._vars_to_calc
            if self.reco or 'nominal' in tree.lower():
                self.logger.info(f"Detected {tree} as reco tree, will pull 'weight_leptonSF' and 'weight_KFactor'")
                self._tree_dict[tree] |= {'weight_leptonSF', 'weight_KFactor'}
                reco_flag = True
            elif self.truth or 'truth' in tree.lower():
                self.logger.info(f"Detected {tree} as truth tree, will pull 'KFactor_weight_truth'")
                self._tree_dict[tree].add('KFactor_weight_truth')
                self.truth = True
                truth_flag = True
            else:
                self.logger.info(f"Neither {tree} as truth nor reco dataset detected.")

        self.reco = reco_flag
        self.truth = truth_flag

        # check if vars are contained in label dictionary
        all_vars = {var for var_set in self._tree_dict.values() for var in var_set}
        if unexpected_vars := [unexpected_var for unexpected_var in all_vars
                               if unexpected_var not in labels_xs]:
            self.logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                                "Some unexpected behaviour may occur.")

        # some debug information
        self.logger.debug("")
        self.logger.debug("DATASET INPUT OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug(f"Input file(s):  {self.data_path}")
        self.logger.debug(f"TTree: {self.TTree_name}")
        self.logger.debug(f"Cutfile: {self.cutfile_path}")
        self.logger.debug(f"grouped cutflow: {self.cutfile.options['grouped cutflow']}")
        self.logger.debug(f"Hard cut applied: {self.cutfile.get_cut_string(self.hard_cut)}")
        self.logger.debug(f"Forced dataset rebuild: {self.force_rebuild}")
        self.logger.debug(f"Validate missing events: {self.validate_missing_events}")
        self.logger.debug(f"Validate duplicated events: {self.validate_duplicated_events}")
        self.logger.debug(f"Validate sum of weights: {self.validate_sumofweights}")
        self.logger.debug("----------------------------")
        self.logger.debug("")

        if self.logger.level == logging.DEBUG:
            for tree in self._tree_dict:
                self.logger.debug(f"Variables to extract from {tree} tree: ")
                for var in self._tree_dict[tree]:
                    self.logger.debug(f"  - {var}")
            if self._vars_to_calc:
                self.logger.debug("Variables to calculate: ")
                for var in self._vars_to_calc:
                    self.logger.debug(f"  - {var}")

        # GENERATE DATAFRAME
        # ========================
        """Define pipeline for building dataset dataframe"""
        # extract and clean data
        if self._build_df:
            self.logger.info(f"Building {self.name} dataframe from {self.data_path}...")
            self.df = self._build_dataframe(
                _validate_missing_events=self.validate_missing_events,
                _validate_duplicated_events=self.validate_duplicated_events,
                _validate_sumofweights=self.validate_sumofweights
            )

            # print into pickle file for easier read/write
            if self.to_pkl:
                pd.to_pickle(self.df, self.pkl_path)
                self.logger.info(f"Dataframe built and saved in {self.pkl_path}")

        else:
            self.logger.info(f"Reading data for {self.name} dataframe from {self.pkl_path}...")
            self.df = pd.read_pickle(self.pkl_path)
        self.logger.info(f"Number of events in dataset {self.name}: {len(self.df.index)}")

        # backup cutfile only after having built the dataframe
        if self.cutfile.if_make_cutfile_backup():
            self.cutfile.backup_cutfile(self.name)

        # map appropriate weights
        if self.truth:
            self.logger.info(f"Calculating truth weight for {self.name}...")
            self.df['truth_weight'] = self.__event_weight_truth()
        if self.reco:
            self.logger.info(f"Calculating reco weight for {self.name}...")
            self.df['reco_weight'] = self.__event_weight_reco()

        # print some dataset ID metadata
        # TODO: avg event weight
        if self.df.index.names != ['DSID', 'eventNumber']:
            raise ValueError("Incorrect index")
        if self.logger.level == logging.DEBUG:
            self.logger.info(f"DATASET INFO FOR {self.name}:")
            self.logger.debug("DSID       n_events   sum_w         x-s fb        lumi fb-1")
            self.logger.debug("==========================================================")
            for dsid, df_id in self.df.groupby(level='DSID'):
                self.logger.debug(f"{dsid:<10} "
                                  f"{len(df_id):<10} "
                                  f"{df_id['weight_mc'].sum():<10.6e}  "
                                  f"{self._get_cross_section(df_id):<10.6e}  "
                                  f"{self.lumi:<10.6e}")

        # apply cuts to generate cut columns
        self.logger.info(f"Creating cuts for {self.name}...")
        self.__create_cut_columns()

        # GENERATE CUTFLOW
        # ========================
        self.cutflow = Cutflow(self.df,
                               self.cutfile.cut_dicts,
                               self.logger,
                               self.cutfile.cutgroups if self.cutfile.options['grouped cutflow'] else None)

        if self.hard_cut:
            self.logger.info(f"Applying hard cut(s): {self.hard_cut}: {self.cutflow}...")
            self.apply_cuts(self.hard_cut, inplace=True)

        self.logger.info("=" * (42 + len(self.name)))
        self.logger.info(f"========= DATASET '{self.name}' INITIALISED =========")
        self.logger.info("=" * (42 + len(self.name)))
        self.logger.info("")

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
        return f'Dataset("{self.name}",TTree:"{self.TTree_name}",Events:{len(self)})'

    def __str__(self):
        return f'{self.name},TTree:"{self.TTree_name}",Events:{len(self)}'

    def __add__(self, other) -> pd.DataFrame:
        """Concatenate two dataframes"""
        return pd.concat([self.df, other.df], ignore_index=True, copy=False)

    def __iadd__(self, other):
        """Concatenate dataframe to self.df"""
        self.df.append(other.df, ignore_index=True)

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

    @staticmethod
    def _get_cross_section(df: pd.DataFrame, n_events=None, weight_mc_col: str = 'weight_mc') -> float:
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
        self.logger.info("")
        self.logger.info(f"========{self.name.upper()} KINEMATICS ===========")
        self.logger.info(f"cross-section: {self.cross_section:.2f} fb")
        self.logger.info(f"luminosity   : {self.luminosity:.2f} fb-1")

    def print_latex_table(self) -> None:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        latex_filepath = self.paths['latex_dir'] + self.name + "_cutflow.tex"

        with open(latex_filepath, "w") as f:
            f.write("\\begin{tabular}{|c||c|c|c|}\n"
                    "\\hline\n")
            f.write(f"Cut & Events & Ratio & Cumulative \\\\\\hline\n"
                    f"Inclusive & {len(self)} & — & — \\\\\n")
            for i, cutname in enumerate(self.cutflow.cutflow_labels[1:]):
                f.write(f"{cutname} & "
                        f"{self.cutflow.cutflow_n_events[i + 1]} & "
                        f"{self.cutflow.cutflow_ratio[i + 1]:.3f} & "
                        f"{self.cutflow.cutflow_cum[i + 1]:.3f} "
                        f"\\\\\n")
            f.write("\\hline\n"
                    "\\end{tabular}\n")

        self.logger.info(f"Saved LaTeX cutflow table in {latex_filepath}")

    # ===============================
    # ====== DATAFRAME BUILDER ======
    # ===============================
    def _build_dataframe(self,
                         chunksize: int = 1024,
                         _validate_missing_events: bool = True,
                         _validate_duplicated_events: bool = True,
                         _validate_sumofweights: bool = True,
                         ) -> pd.DataFrame:
        """
        Builds a dataframe from cutfile inputs
        :param chunksize: chunksize for uproot concat method
        :param _validate_duplicated_events: whether to check for duplicated events
        :param _validate_sumofweights: whether to check sum of weights against weight_mc
        :return: output dataframe containing columns corresponding to necessary variables
        """
        # is the default tree a truth tree?
        default_tree_truth = 'truth' in self.TTree_name

        t1 = time.time()
        self.logger.debug(f"Extracting {self._tree_dict[self.TTree_name]} from {self.TTree_name} tree...")
        df = to_pandas(uproot.concatenate(self.data_path + ':' + self.TTree_name, self._tree_dict[self.TTree_name],
                                          num_workers=config.n_threads, begin_chunk_size=chunksize))
        self.logger.debug(f"Extracted {len(df)} events.")

        self.logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = to_pandas(uproot.concatenate(self.data_path + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                            num_workers=config.n_threads, begin_chunk_size=chunksize))

        self.logger.debug(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)

        df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
        df.index.names = ['DSID', 'eventNumber']
        self.logger.debug("Set DSID/eventNumber as index")

        # merge TTrees
        if _validate_duplicated_events:
            validation = '1:1'
            self.logger.info(f"Validating duplicated events in tree {self.TTree_name}...")
            self.__drop_duplicates(df)
        else:
            validation = 'm:m'
            self.logger.info("Skipping duplicted events validation")

        # iterate over TTrees and merge
        for tree in self._tree_dict:
            if tree == self.TTree_name:
                continue

            self.logger.debug(f"Extracting {self._tree_dict[tree]} from {tree} tree...")
            alt_df = to_pandas(uproot.concatenate(self.data_path + ":" + tree, self._tree_dict[tree],
                                                  num_workers=config.n_threads, begin_chunk_size=chunksize))
            self.logger.debug(f"Extracted {len(alt_df)} events.")

            alt_df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
            alt_df.index.names = ['DSID', 'eventNumber']
            self.logger.debug("Set DSID/eventNumber as index")

            if _validate_missing_events:
                self.logger.info(f"Checking for missing events in tree '{tree}'..")
                tree_is_truth = 'truth' in tree

                if tree_is_truth and not default_tree_truth:
                    if n_missing := len(df.index.difference(alt_df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{self.TTree_name}' tree not found in '{tree}' tree")
                    else:
                        self.logger.debug(f"All events in {self.TTree_name} tree found in {tree} tree")
                elif default_tree_truth and not tree_is_truth:
                    if n_missing := len(alt_df.index.difference(df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{tree}' tree not found in '{self.TTree_name}' tree")
                    else:
                        self.logger.debug(f"All events in {tree} tree found in {self.TTree_name} tree")
                else:
                    self.logger.info(f"Skipping missing events check. Not truth/reco tree combination")

            else:
                self.logger.info(f"Skipping missing events check in tree {tree}")

            if _validate_duplicated_events:
                self.logger.info(f"Validating duplicated events in tree {tree}...")
                self.__drop_duplicates(alt_df)

            self.logger.debug("Merging with rest of dataframe...")
            df = pd.merge(df, alt_df, how='left', left_index=True, right_index=True, sort=False, copy=False, validate=validation)

        if _validate_sumofweights:
            # sanity check to make sure totalEventsWeighted really is what it says it is
            # also output DSID metadata
            df_id_sub = df['weight_mc'].groupby(level='DSID').sum()
            for dsid, df_id in df.groupby(level='DSID'):
                unique_totalEventsWeighted = df_id['totalEventsWeighted'].unique()
                if len(unique_totalEventsWeighted) != 1:
                    self.logger.warning("totalEventsWeighted should only have one value per DSID. "
                                        f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} "
                                        f"for DSID {dsid}")

                dsid_weight = df_id_sub[dsid]
                totalEventsWeighted = df_id['totalEventsWeighted'].values[0]
                if dsid_weight != totalEventsWeighted:
                    self.logger.warning(
                        f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                        f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}")
        else:
            self.logger.info("Skipping sum of weights validation")

        # calculate and combine special derived variables
        if self._vars_to_calc:
            # save which variables are actually necessary in order to drop extras (keep all columns for now)
            # og_vars = all_vars(cut_list_dicts, vars_to_cut)
            for var in self._vars_to_calc:
                # compute new column
                temp_cols = derived_vars[var]['var_args']
                func = derived_vars[var]['func']
                str_args = derived_vars[var]['var_args']
                self.logger.info(f"Computing '{var}' column from {temp_cols}...")
                df[var] = func(df, *str_args)

                # # drop unnecessary columns extracted just for calculations
                # to_drop = [var for var in temp_cols if var not in og_vars]
                # logger.debug(f"dropping {to_drop}")
                # df.drop(columns=to_drop, inplace=True)

        # CLEANUP
        df.name = self.name

        self.__rescale_to_gev(df)  # properly scale GeV columns

        # output number of truth and reco events
        if self.truth:
            self.logger.info(f"number of truth events in '{self.name}': {(~df['KFactor_weight_truth'].isna()).sum()}")
        if self.reco:
            self.logger.info(f"number of reco events in '{self.name}': {(~df['weight_KFactor'].isna()).sum()}")

        if self.truth and self.reco:
            # drop truth KFactor as long as values are the same for reco variables
            pd.testing.assert_series_equal(df.loc[pd.notna(df['weight_KFactor']), 'KFactor_weight_truth'],
                                           df['weight_KFactor'].dropna(),
                                           check_exact=True, check_names=False, check_index=False), \
                                                "reco and truth KFactors not equal"
            df.drop(columns='KFactor_weight_truth')
            self.logger.debug("Dropped extra KFactor column")
        
        self.logger.info("Sorting by DSID...")
        df.sort_index(level='DSID', inplace=True)

        self.logger.info(f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}")

        return df

    def __drop_duplicates(self, df: pd.DataFrame) -> None:
        """
        Checks for and drops duplicated events and events with same event numbers for each dataset ID
        """
        b_size = len(df.index)
        df.drop_duplicates(inplace=True)
        self.logger.info(f"{b_size - len(df.index)} duplicate events dropped")
        b_size = len(df.index)
        df.index = df.index.drop_duplicates()
        self.logger.info(f"{b_size - len(df.index)} duplicate event numbers dropped")

    def __create_cut_columns(self) -> None:
        """
        Creates boolean columns in dataframe corresponding to each cut
        """
        label = config.cut_label  # get cut label from config

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

        for cut in self.cutfile.cut_dicts:
            if cut['relation'] not in op_dict:
                raise ValueError(f"Unexpected comparison operator: {cut['relation']}.")
            if not cut['is_symmetric']:
                self.df[cut['name'] + label] = op_dict[cut['relation']](self.df[cut['cut_var']], cut['cut_val'])
            else:  # take absolute value instead
                self.df[cut['name'] + label] = op_dict[cut['relation']](self.df[cut['cut_var']].abs(), cut['cut_val'])

    def log_cuts(self, name: bool = True) -> None:
        """send list of cuts in cutfile to logger"""
        self.cutfile.log_cuts(name=name)

    def __event_weight_reco(self,
                            mc_weight: str = 'weight_mc',
                            tot_weighted_events: str = 'totalEventsWeighted',
                            KFactor: str = 'weight_KFactor',
                            lepton_SF: str = 'weight_leptonSF',
                            pileup_weight: str = 'weight_pileup',
                            ) -> pd.Series:
        """
        Calculate total reco event weights

        lumi_data taken from year
        mc_weight = +/-1 * cross_section
        kFactor = kFactor_weight_truth
        pileup = weight_pileup
        truth_weight = kFactor * pileup
        lumi_weight = mc_weight * lumi_data / sum of event weights

        total event weight = lumi_weight * truth_weight * reco_weight

        This is done in one line for efficiency with pandas (sorry)
        """
        return \
            self.lumi * self.df[mc_weight] * abs(self.df[mc_weight]) / self.df[tot_weighted_events] * \
            self.df[KFactor] * self.df[pileup_weight] * self.df[lepton_SF]

    def __event_weight_truth(self,
                             mc_weight: str = 'weight_mc',
                             tot_weighted_events: str = 'totalEventsWeighted',
                             KFactor: str = 'KFactor_weight_truth',
                             pileup_weight: str = 'weight_pileup',
                             ) -> pd.Series:
        """
        Calculate total truth event weights

        lumi_data taken from year
        mc_weight = +/-1 * cross_section
        scale factor = weight_leptonSF
        KFactor = KFactor_weight_truth
        pileup = weight_pileup
        recoweight = scalefactors * KFactor * pileup
        lumi_weight = mc_weight * lumi_data / sum of event weights

        total event weight = lumi_weight * truth_weight * reco_weight

        This is done in one line for efficiency with pandas (sorry)
        """
        if self.reco:
            KFactor = 'weight_KFactor'
        return \
            self.lumi * self.df[mc_weight] * abs(self.df[mc_weight]) / self.df[tot_weighted_events] * \
            self.df[KFactor] * self.df[pileup_weight]

    def __rescale_to_gev(self, df) -> None:
        """rescales to GeV because athena's default output is in MeV for some reason"""
        if GeV_columns := [
            column for column in df.columns
            if (column in labels_xs)
            and ('[GeV]' in labels_xs[column]['xlabel'])
        ]:
            df[GeV_columns] /= 1000
            self.logger.debug(f"Rescaled columns {GeV_columns} to GeV.")
        else:
            self.logger.debug(f"No columns rescaled to GeV.")

    @staticmethod
    def _cut_on_cutgroup(df: pd.DataFrame,
                         cutgroups: OrderedDict[str, List[str]],
                         group: str,
                         ) -> pd.DataFrame:
        """Cuts on cutgroup on input dataframe or series"""
        cut_cols = [cut_name + config.cut_label for cut_name in cutgroups[group]]
        cut_data = df.loc[df[cut_cols].all(1)]
        return cut_data

    def apply_cuts(self,
                   labels: Union[bool, str, List[str]] = True,
                   inplace: bool = False
                   ) -> Union[pd.DataFrame, None]:
        """
        Apply cut(s) to DataFrame.

        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :param inplace: If True, applies cuts in place to dataframe in self.
                        If False returns DataFrame object
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied and associated cut columns removed.
                 Raises ValueError if cuts do not exist in dataframe
        """
        def __check_cut_cols(c: List[str]) -> None:
            """Check if columns exist in dataframe"""
            if missing_cut_cols := [
                label for label in c
                if label not in self.df.columns
            ]:
                raise ValueError(f"No cut(s) {missing_cut_cols} in dataset {self.name}...")

        if not labels:
            self.logger.debug(f"No cuts applied to {self.name}")
            if inplace:
                return
            else:
                return self.df

        elif isinstance(labels, list):
            self.logger.debug(f"Applying cuts: {labels} to {self.name}...")
            cut_cols = [label + config.cut_label for label in labels]
            __check_cut_cols(cut_cols)

        elif isinstance(labels, str):
            self.logger.debug(f"Applying cut: {labels} to {self.name}...")
            cut_cols = [labels + config.cut_label]
            __check_cut_cols(cut_cols)

        elif labels is True:
            self.logger.debug(f"Applying all cuts to {self.name}...")
            cut_cols = [str(col) for col in self.df.columns if config.cut_label in col]
            __check_cut_cols(cut_cols)

        else:
            raise TypeError("'labels' must be a bool, a string or a list of strings")

        # apply cuts
        if inplace:
            self.df = self.df.loc[self.df[cut_cols].all(1)]
            self.df.drop(columns=cut_cols, inplace=True)

        else:
            return self.df.loc[self.df[cut_cols].all(1)].drop(columns=cut_cols)

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
    def plot_hist(
            self,
            var: Union[str, List[str]],
            bins: Union[tuple, list],
            weight: Union[str, float] = 1.,
            ax: plt.Axes = None,
            yerr: Union[ArrayLike, str] = None,
            normalise: Union[float, bool, str] = 'lumi',
            logbins: bool = False,
            apply_cuts: Union[bool, str, List[str]] = True,
            **kwargs
    ) -> plt.Axes:
        """
        Generate 1D plots of given variables in dataframe. Returns figure object of list of figure objects.

        :param var: variable name to be plotted. must exist in all datasets
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param weight: variable name in dataset to weight by or numeric value to weight all
        :param ax: axis to plot on. Will create new plot if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param normalise: Normalisation value:
                          - int or float
                          - True for normalisation of unity
                          - 'lumi' (default) for normalisation to global_uni variable in analysis
                          - False for no normalisation
        :param apply_cuts: True to apply all cuts to dataset before plotting or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param logbins: whether logarithmic binnings
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Axis
        """
        self.logger.debug(f"Generating {var} histogram in {self.name}...")

        if not ax:
            _, ax = plt.subplots()

        df = self.apply_cuts(apply_cuts)
        weights = (
            df[weight]
            if isinstance(weight, str)
            else weight
        )
        hist = Histogram1D(bins, df[var], weights, logbins)
        hist.plot(
            ax=ax,
            yerr=yerr,
            normalise=normalise,
            **kwargs
        )
        return ax

    def plot_all_with_cuts(self,
                           bins: Union[tuple, list] = (30, 1, 500),
                           scaling: Optional[str] = None,
                           **kwargs
                           ) -> None:
        """
        Plots all variables in Dataset to cut from cutfile with each cutgroup applied

        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param scaling: either 'xs':     cross-section scaling,
                               'widths': divided by bin widths,
                               None:     No scaling
                        y-axis labels set accordingly
        :param kwargs: keyword arguments to pass to plot_1d_overlay_and_acceptance_cutgroups()
        """
        for var_to_plot in self.cutfile.vars_to_cut:
            self.logger.info(f"Generating histogram with cuts for {var_to_plot} in {self.name}...")
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
            self.logger.info(f"Generating 2d histogram for {x_var}-{y_var} in {self.name}...")
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
            self.logger.info(f"    - generating cutgroup '{cutgroup}'")
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
            # ================        :param kwargs: keyword arguments to pass to mplhep.histplot()

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
            with open(self.paths['pkl_hist_dir'] + plot_label + '_' + var + '_1d_cutgroups_ratios.pkl', 'wb') as f:
                pkl.dump(hists, f)
                self.logger.info(f"Saved pickle file to {f.name}")
        hep.atlas.label(llabel="Internal", loc=0, ax=fig_ax, rlabel=plot_label)
        out_png_file = self.paths['plot_dir'] + f"{var}_{str(scaling)}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        self.logger.info(f"Figure saved to {out_png_file}")
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

        :param x_var: column in dataframe to plot on x-axis
        :param y_var: column in dataframe to plot on y-axis
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

        out_path = self.paths['plot_dir'] + f"2d_{x_var}-{y_var}_inclusive.png"
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
        self.logger.info(f"printed 2d histogram to {out_path}")
        plt.close(fig)

        for cutgroup in self.cutfile.cutgroups:
            self.logger.info(f"    - generating cutgroup '{cutgroup}'")
            fig, ax = plt.subplots(figsize=(7, 7))

            cut_df = Dataset._cut_on_cutgroup(self.df, self.cutfile.cutgroups, cutgroup)
            weight_cut = cut_df[self._weight_column]
            x_vars = cut_df[x_var]
            y_vars = cut_df[y_var]

            out_path = self.paths['plot_dir'] + f"2d_{x_var}-{y_var}_{cutgroup}.png"
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
            self.logger.info(f"printed 2d histogram to {out_path}")
            plt.close(fig)

        if to_pkl:
            with open(self.paths['pkl_hist_dir'] + plot_label + f"_{x_var}-{y_var}_2d.pkl", 'wb') as f:
                pkl.dump(hists, f)
                self.logger.info(f"Saved pickle file to {f.name}")

    def plot_mass_slices(self,
                         var: str,
                         weight: str,
                         bins: Union[Iterable[float], Tuple[int, float, float]] = (30, 0, 5000),
                         logbins: bool = False,
                         logx: bool = False,
                         logy: bool = True,
                         xlabel: str = '',
                         ylabel: str = '',
                         title: str = '',
                         apply_cuts: Union[bool, str, List[str]] = True,
                         **kwargs
                         ) -> None:
        """
        Plot single variable in dataset with different DSIDs visible

        :param var: variable in dataset to plot
        :param weight: column in dataset to use as weight
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param logbins: whether logarithmic binnings
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param apply_cuts: True to apply all cuts to dataset before plotting or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param kwargs: keyword arguments to pass to histogram plotting function
        """
        self.logger.info(f'Plotting {var} in {self.name} as slices...')

        fig, ax = plt.subplots()
        df = self.apply_cuts(apply_cuts)

        # per dsid
        for dsid, dsid_df in self.df.groupby(level='DSID', sort=True):
            weights = dsid_df[weight] if isinstance(weight, str) else weight
            hist = Histogram1D(bins, dsid_df[var], weights, logbins)
            hist.plot(ax=ax, label=dsid, **kwargs)
        # inclusive
        weights = df[weight] if isinstance(weight, str) else weight
        hist = Histogram1D(bins, df[var], weights, logbins)
        hist.plot(ax=ax, label='Inclusive', color='k', **kwargs)

        ax.legend(fontsize=10, ncol=2)
        title = self.label if not title else title
        plt_utils.set_axis_options(ax, var, bins, self.lepton, logbins, xlabel, ylabel, title, logx, logy)

        filename = self.paths['plot_dir'] + self.name + '_' + var + '_SLICES.png'
        fig.savefig(filename, bbox_inches='tight')
        self.logger.info(f'Saved mass slice plot of {var} in {self.name} to {filename}')

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
        :param ratio: ratio of events passing each cut to inclusive sample
        :param cummulative: ratio of each cut to the previous cut
        :param a_ratio: ratio of cut to inclusive sample
        :param all_plots: it True, plot all
        :return: None
        """
        if all_plots:
            event = ratio = cummulative = a_ratio = True
        if event:
            self.cutflow.print_histogram(self.paths['plot_dir'], 'event')
        if ratio:
            self.cutflow.print_histogram(self.paths['plot_dir'], 'ratio')
        if cummulative:
            self.cutflow.print_histogram(self.paths['plot_dir'], 'cummulative')
        if a_ratio:
            self.cutflow.print_histogram(self.paths['plot_dir'], 'a_ratio')

    def profile_plot(self, varx: str, vary: str,
                     title: str = '',
                     xlabel: str = '',
                     ylabel: str = '',
                     ax: plt.Axes = None,
                     to_file: bool = True,
                     xlim: Tuple[float, float] = None,
                     ylim: Tuple[float, float] = None,
                     logx: bool = False,
                     logy: bool = False,
                     **kwargs) -> None:

        if not ax:
            fig, ax = plt.subplots()

        ax.scatter(self.df[varx], self.df[vary], **kwargs)

        ax.set_xlabel(xlabel if xlabel else labels_xs[varx]['xlabel'])
        ax.set_ylabel(ylabel if ylabel else labels_xs[vary]['xlabel'])
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if logx: ax.semilogx()
        if logy: ax.semilogy()
        hep.atlas.label(italic=(True, True), loc=0, llabel='Internal', ax=ax, rlabel=title if title else self.label)

        plt.show()
        if to_file:
            plt.savefig(self.paths['plot_dir'] + varx + '_' + vary + '_PROFILE.png', bbox_inches='tight')

        return ax
