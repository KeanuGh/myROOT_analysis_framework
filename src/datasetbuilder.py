import logging
import operator as op
import time
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Iterable, Set, overload

import pandas as pd
import uproot
from awkward import to_pandas

import src.config as config
import utils.file_utils as file_utils
from src.cutfile import Cutfile, CutDict
from src.cutflow import Cutflow
from src.dataset import Dataset
from src.logger import get_logger
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
class DatasetBuilder:
    """
    Dataset builder class.
    Pass dataset build options to initialise class. Use DatasetBuilder.build() with inputs to return Dataset object.

    :param name: dataset name
    :param TTree_name: TTree in datapath to set as default tree
    :param lumi: data luminosity for weight calculation and normalisation
    :param year: data year for luminosty mapping
    :param lumi: data luminosity. Pass either this or year
    :param label: Label to put on plots
    :param lepton: Name of charged DY lepton channel in dataset (if applicable)
    :param logger: logger to write to. Defaults to console output at DEBUG-level
    :param hard_cut: name(s) of cut(s) that should be applied to dataframe immediately
    :param force_rebuild: whether to force build of DataFrame
    :param force_recalc_weights: whether to force recalculating weights, no matter if the column already exists
    :param force_recalc_cuts: whether to force recalculating cuts, whether columns exist or not
    :param force_recalc_vars: whether to force recalculating variables like M^W_T
    :param skip_verify_pkl: verifying pickle file against cutfile
    :param chunksize: chunksize for uproot concat method
    :param validate_missing_events: whether to check for missing events
    :param validate_duplicated_events: whether to check for duplicatedevents
    :param validate_sumofweights: whether to check sum of weights against weight_mc
    :return: output dataframe containing columns corresponding to necessary variables
    """
    name: str = 'data'
    TTree_name: str = 'truth'
    year: str = '2015+2016'
    lumi: float = None
    label: str = 'data',
    lepton: str = 'lepton'
    logger: logging.Logger = field(default_factory=get_logger)
    hard_cut: Union[str, List[str]] = field(default_factory=list)
    force_rebuild: bool = False
    force_recalc_weights: bool = False
    force_recalc_cuts: bool = False
    force_recalc_vars: bool = False
    skip_verify_pkl: bool = False
    chunksize: int = 1024
    validate_missing_events: bool = True
    validate_duplicated_events: bool = True
    validate_sumofweights: bool = True

    def __post_init__(self):
        if self.lumi and self.year:
            raise ValueError("Pass wither lumi or year")
        elif self.year:
            self.lumi = lumi_year[self.year]

    @overload
    def build(self, data_path: str, cutfile_path: str) -> Dataset:
        ...

    @overload
    def build(self, data_path: str, cutfile: Cutfile) -> Dataset:
        ...

    @overload
    def build(self, pkl_path: str, cutfile_path: str) -> Dataset:
        ...

    @overload
    def build(self, pkl_path: str, cutfile: Cutfile) -> Dataset:
        ...

    @overload
    def build(self, data_path: str, pkl_path: str, cutfile_path: str) -> Dataset:
        ...

    @overload
    def build(self, data_path: str, pkl_path: str, cutfile: Cutfile) -> Dataset:
        ...

    @overload
    def build(
            self,
            data_path: str,
            tree_dict: Dict[str, Set[str]],
            vars_to_calc: Set[str],
            cut_list: Optional[List[CutDict]] = None,
    ) -> Dataset:
        ...

    def build(
            self,
            data_path: str = None,
            cutfile: Cutfile = None,
            cutfile_path: str = None,
            pkl_path: str = None,
            tree_dict: Dict[str, Set[str]] = None,
            vars_to_calc: Set[str] = None,
            cut_dicts: Optional[List[CutDict]] = None,
    ) -> Dataset:
        """
        Builds a dataframe from cutfile inputs.

        :param data_path: Path to ROOT file(s) must be passed if not providing a pickle file
        :param cutfile: If passed, uses Cutfile object to build dataframe
        :param cutfile_path: If passed, builds Cutfile object from file and dataframe
        :param pkl_path: If passed opens pickle file, checks against cut and variable options
                         and calculates cuts and weighs if needed.
                         Can be passed instead of ROOT file but then will not be checked against cutfile
        :param tree_dict: If cutfile or cutfile_path not passed,
                          can pass dictionary of tree: variables to extract from Datapath
        :param vars_to_calc: list of variables to calculate to pass to add to DataFrame
        :param cut_dicts: list of cut dictionaries if cuts are to be applied but no cutfile is given
        :return: full built Dataset object
        """
        __build_df = False
        __create_cut_cols = False
        __create_weight_cols = False
        __calculate_vars = False
        __error_at_rebuild = False

        # PRE-PROCESSING
        # ===============================
        # check arguments
        if not data_path and not pkl_path:
            raise ValueError("Either ROOT data or a pickled dataframe must be provided")
        elif not data_path and pkl_path:
            logger.warning("No ROOT data provided so pickled DataFrame cannot be checked")
            __build_df = False
            __error_at_rebuild = True
            __create_weight_cols = False
            __create_cut_cols = False
        elif data_path and not pkl_path:
            __build_df = True

        if data_path and not file_utils.file_exists(data_path):
            raise FileExistsError(f"File {data_path} not found.")

        if (not cutfile) and (not cutfile_path):
            raise ValueError("Cutfile must be provided")

        # Parsing cutfile
        if cutfile_path:
            self.logger.info(f"Parsting cutfile '{cutfile_path}'...")
            cutfile = Cutfile(cutfile_path, logger=self.logger)

            if self.logger.level == logging.DEBUG:
                # log contents of cutfile
                self.logger.debug("CUTFILE CONTENTS")
                self.logger.debug('---------------------------')
                with open(cutfile_path) as f:
                    for line in f.readlines():
                        self.logger.debug(line.rstrip('\n'))
                self.logger.debug('---------------------------')
        if cutfile:
            # get tree dictionary, set of variables to calculate, and whether the dataset will contain truth, reco data
            tree_dict, vars_to_calc = cutfile.extract_var_data(derived_vars, self.TTree_name)
            is_truth, is_reco = cutfile.truth_reco(tree_dict)
            cut_dicts = cutfile.cut_dicts
        elif tree_dict:
            if (vars_to_calc is None) or (cut_dicts is None):
                raise ValueError("Must provide variables to cut and cut dictionary list if not building from cutfile")
            else:
                is_truth, is_reco = Cutfile.truth_reco(tree_dict)
        else:
            raise ValueError("Must provide cutfile or tree dict to build DataFrame")

        # check if vars are contained in label dictionary
        self.__check_axis_labels(tree_dict, vars_to_calc)

        # check if hard cut(s) exists in cutfile. If not, skip
        self.__check_hard_cuts(cutfile)

        # print debug information
        self.logger.debug("")
        self.logger.debug("DEBUG DATASET OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug(f"Input ROOT file(s):  {data_path}")
        self.logger.debug(f"Input pickle file(s): {pkl_path}")
        self.logger.debug(f"TTree: {self.TTree_name}")
        if cutfile_path:
            self.logger.debug(f"Cutfile: {cutfile_path}")
        if cutfile:
            self.logger.debug("Cuts from cutfile:")
            cutfile.log_cuts(debug=True)
        hard_cuts_str = '\n\t'.join([f"{cut}: {cutfile.get_cut_string(cut)}" for cut in self.hard_cut]) \
                        if self.hard_cut else None
        self.logger.debug(f"Hard cut(s) applied: {hard_cuts_str}")
        self.logger.debug(f"Forced dataset rebuild: {self.force_rebuild}")
        self.logger.debug(f"Calculating weights: {__create_weight_cols}")
        self.logger.debug(f"Calculating cuts: {__create_weight_cols}")
        if pkl_path:
            self.logger.debug(f"Pickle datapath: {pkl_path}")
            self.logger.debug(f"Skipping pickle file verification: {self.skip_verify_pkl}")
        self.logger.debug(f"Chunksize for ROOT file import: {self.chunksize}")
        self.logger.debug(f"Validate missing events: {self.validate_missing_events}")
        self.logger.debug(f"Validate duplicated events: {self.validate_duplicated_events}")
        self.logger.debug(f"Validate sum of weights: {self.validate_sumofweights}")
        if self.logger.level == logging.DEBUG:
            for tree in tree_dict:
                self.logger.debug(f"Variables from {tree} tree: ")
                for var in tree_dict[tree]:
                    self.logger.debug(f"  - {var}")
            if vars_to_calc:
                self.logger.debug("Calculated Variables: ")
                for var in vars_to_calc:
                    self.logger.debug(f"  - {var}")
        self.logger.debug("----------------------------")
        self.logger.debug("")

        df = pd.DataFrame()  # literally just to stop pycharm from complaining

        # load dataframe from pickle
        if pkl_path and not self.force_rebuild:
            df = self.__read_pkl_df(pkl_path)

            # Pickle file checks
            if self.skip_verify_pkl:
                self.logger.debug("Skipped pickle verification")
                __build_df = False
            else:
                # verification pipeline
                self.logger.debug("Checking pickled DataFrame...")
                cols = df.columns
                if not self.__check_var_cols(cols, tree_dict, raise_error=__error_at_rebuild):
                    __build_df = True
                elif not self.__check_calc_var_cols(cols, vars_to_calc) and not self.force_recalc_vars:
                    if not self.__check_argument_var_cols(cols, vars_to_calc, raise_error=__error_at_rebuild):
                        __build_df = True
                    else:
                        __calculate_vars = True
                else:
                    self.logger.debug("Found all necessary variables. No rebuild necessary")
                    __build_df = False
                    __create_cut_cols = (
                        True if self.force_recalc_cuts
                        else not self.__check_cut_cols(cols, cut_dicts)
                    )
                    __create_weight_cols = (
                        True if self.force_recalc_weights
                        else not self.__check_weight_cols(cols, truth=is_truth, reco=is_reco)
                    )
        else:
            __build_df = True

        # BUILD
        # ===============================
        if __build_df:
            df = self.build_dataframe(
                data_path=data_path,
                TTree_name=self.TTree_name,
                tree_dict=tree_dict,
                is_truth=is_truth,
                is_reco=is_reco,
                chunksize=self.chunksize,
                validate_missing_events=self.validate_missing_events,
                validate_duplicated_events=self.validate_duplicated_events,
                validate_sumofweights=self.validate_sumofweights
            )
            __create_weight_cols = True
            __create_cut_cols = True
            __calculate_vars = True

        # POST-PROCESSING
        # ===============================
        # calculate variables
        if __calculate_vars or self.force_recalc_vars:
            self.__calculate_vars(df, vars_to_calc)

        # calculate cut columns
        if __create_cut_cols or self.force_recalc_cuts:
            self.__create_cut_columns(df, cutfile)

        # calculate weights
        if __create_weight_cols or self.force_recalc_weights:
            if is_truth:
                self.logger.info(f"Calculating truth weight for {self.name}...")
                df['truth_weight'] = self.__event_weight_truth(df)
                # EVERY event MUST have a truth weight
                self.logger.info(f"Verifying truth weight for {self.name}...")
                if df['truth_weight'].isna().any():
                    raise ValueError("NAN values in truth weights!")
            if is_reco:
                self.logger.info(f"Calculating reco weight for {self.name}...")
                df['reco_weight'] = self.__event_weight_reco(df)
                # every reco event MUST have a reco weight
                self.logger.info(f"Verifying reco weight for {self.name}...")
                if not is_truth:
                    if df['reco_weight'].isna().any():
                        raise ValueError("NAN values in reco weights!")
                else:
                    assert (~df['reco_weight'].isna()).sum() == (~df['weight_leptonSF'].isna()).sum(), \
                        "Different number of events for reco weight and lepton scale factors!"

        # GENERATE CUTFLOW
        # remove hard cut from cutflow:
        cut_dicts = [cut_dict for cut_dict in cut_dicts
                     if cut_dict['name'] not in self.hard_cut]

        cutflow = Cutflow(
            df,
            cut_dicts,
            self.logger,
        )

        # apply hard cut(s) if it exists in cut columns
        hard_cuts_to_apply = []
        for h_cut in self.hard_cut:
            if h_cut + config.cut_label not in df.columns:
                self.logger.debug(f"No cut {h_cut} in DataFrame. Assuming hard cut has already been applied")
            else:
                hard_cuts_to_apply.append(h_cut)
        # apply hard cuts
        if hard_cuts_to_apply:
            self.logger.info(f"Applying hard cut(s): {self.hard_cut}...")
            if isinstance(self.hard_cut, str):
                cuts = [self.hard_cut]
            else:
                cuts = self.hard_cut
            cut_cols = [cut + config.cut_label for cut in cuts]
            df = df.loc[df[cut_cols].all(1)]
            df.drop(columns=cut_cols, inplace=True)

        # BUILD DATASET
        # ===============================
        return Dataset(
            name=self.name,
            df=df,
            cutfile=cutfile,
            cutflow=cutflow,
            lumi=self.lumi,
            label=self.label,
            lepton=self.lepton
        )

    def __read_pkl_df(self, pkl_path: str) -> pd.DataFrame:
        self.logger.info(f"Reading data from {pkl_path}...")
        df: pd.DataFrame = pd.read_pickle(pkl_path)
        assert type(df) == pd.DataFrame, f"Pickle file does not contain a pandas DataFrame. Found type {type(df)}"
        assert df.index.names == ('DSID', 'eventNumber'), f"Pickled DataFrame index incorrect: {df.index.names}"
        return df

    # ===============================
    # =========== CHECKS ============
    # ===============================
    def __check_axis_labels(self, tree_dict: Dict[str, Set[str]], calc_vars: Set[str]) -> None:
        """Check whether variables exist in """
        all_vars = {var for var_set in tree_dict.values() for var in var_set} | calc_vars
        if unexpected_vars := [unexpected_var for unexpected_var in all_vars
                               if unexpected_var not in labels_xs]:
            self.logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                                "Some unexpected behaviour may occur.")

    def __check_hard_cuts(self, cutfile: Cutfile, error: bool = False) -> None:
        """
        Check whether cuts passed as hard cuts exist in cutfile. If not, ignore.
        Raise ValueError if 'error' is True
        """
        if self.hard_cut is not None:
            if not isinstance(self.hard_cut, (str, list)):
                raise TypeError(f"hard_cut parameter must be of type string or list of strings. "
                                f"Got type {type(self.hard_cut)}")
            elif isinstance(self.hard_cut, str):
                self.hard_cut = [self.hard_cut]
            for i, cut in enumerate(self.hard_cut):
                if not cutfile.cut_exists(cut):
                    if error:
                        raise ValueError(f"No cut named '{self.hard_cut}' in cutfile")
                    else:
                        self.logger.debug(f"No cut named '{self.hard_cut}' in cutfile; skipping")
                        self.hard_cut.remove(cut)

    def __check_var_cols(self, df_cols: Iterable, tree_dict: Dict[str, Set[str]], raise_error: bool = False) -> bool:
        """Check whether required variables exist in DataFrame columns"""
        var_set = {var for var_set in tree_dict.values() for var in var_set}
        # variables that get renamed or put in index
        var_set.remove('mcChannelNumber')
        var_set.remove('eventNumber')
        var_set.remove('KFactor_weight_truth')
        var_set.add('weight_KFactor')
        if missing_vars := {var for var in var_set
                            if var not in df_cols}:
            if raise_error:
                raise ValueError(f"Variable(s) {missing_vars} missing from DataFrame")
            else:
                self.logger.info(f"Variable(s) {missing_vars} missing from DataFrame. Will rebuild")
                return False
        else:
            return True

    def __check_calc_var_cols(self, df_cols: Iterable, calc_vars: set) -> bool:
        """
        Check whether calculated variables exist in DataFrame
        """
        if missing_vars := {var for var in calc_vars
                            if var not in df_cols}:
            self.logger.info(f"Calculated variable(s) {missing_vars} missing from DataFrame. "
                             f"Will check its argument variables")
            return False
        else:
            return True

    def __check_argument_var_cols(self, df_cols: Iterable, calc_vars: set, raise_error: bool = False) -> bool:
        """Check whether variables necessary to calculate calc_vars exist in DataFrame"""
        for calc_var in calc_vars:
            if missing_vars := {var for var in derived_vars[calc_var]['var_args']
                                if var not in df_cols}:
                if raise_error:
                    raise ValueError(f"Argument variable(s) {missing_vars} missing from DataFrame")
                else:
                    self.logger.info(f"Argument variable(s) {missing_vars} missing from DataFrame. Will rebuild")
                    return False
        return True

    def __check_cut_cols(self, df_cols: Iterable, cut_dicts: List[CutDict], raise_error: bool = False) -> bool:
        """Check whether all necessary cut columns exist in DataFrame columns not including any hard cut"""
        cut_cols = {cut['name'] + config.cut_label for cut in cut_dicts}
        cut_cols -= {h_cut + config.cut_label for h_cut in self.hard_cut}
        if missing_vars := {var for var in cut_cols
                            if var not in df_cols}:
            if raise_error:
                raise ValueError(f"Cut column(s) {missing_vars} missing from DataFrame")
            else:
                self.logger.info(f"Cut column(s) {missing_vars} missing from DataFrame. Will calculate")
                return False
        else:
            return True

    def __check_weight_cols(self, df_cols: Iterable, truth: bool, reco: bool, raise_error: bool = False) -> bool:
        """Check if necessary weight columns exist in DataFrame"""
        if truth and 'truth_weight' not in df_cols:
            if raise_error:
                raise ValueError(f"Truth weight missing")
            else:
                self.logger.info(f"Truth weight missing, will calculate")
                return False

        if reco and 'reco_weight' not in df_cols:
            if raise_error:
                raise ValueError(f"Reco weight missing")
            else:
                self.logger.info(f"Reco weight missing, will calculate")
                return False
        return True

    # ===============================
    # ===== DATAFRAME FUNCTIONS =====
    # ===============================
    def build_dataframe(
            self,
            data_path: str,
            TTree_name: str,
            tree_dict: Dict[str, Set[str]],
            is_truth: bool,
            is_reco: bool,
            chunksize: int = 1024,
            validate_missing_events: bool = True,
            validate_duplicated_events: bool = True,
            validate_sumofweights: bool = True,
    ) -> pd.DataFrame:
        """
         Builds a dataframe

        :param data_path: path to ROOT datafile(s)
        :param TTree_name: TTree in datapath to set as default tree
        :param tree_dict: dictionary of tree: variables to extract from Datapath
        :param is_truth: whether dataset contains truth data
        :param is_reco: whether dataset contains reco data
        :param chunksize: chunksize for uproot concat method
        :param validate_missing_events: whether to check for missing events
        :param validate_duplicated_events: whether to check for duplicated events
        :param validate_sumofweights: whether to check sum of weights against weight_mc
        :return: output dataframe containing columns corresponding to necessary variables
        """
        self.logger.info(f"Building DataFrame from {data_path} ({file_utils.n_files(data_path)} file(s))...")

        # is the default tree a truth tree?
        default_tree_truth = 'truth' in TTree_name

        t1 = time.time()
        self.logger.debug(f"Extracting {tree_dict[TTree_name]} from {TTree_name} tree...")
        df = to_pandas(uproot.concatenate(data_path + ':' + TTree_name, tree_dict[TTree_name],
                                          num_workers=config.n_threads, begin_chunk_size=chunksize))
        self.logger.debug(f"Extracted {len(df)} events.")

        self.logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = to_pandas(uproot.concatenate(data_path + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                            num_workers=config.n_threads, begin_chunk_size=chunksize))

        self.logger.debug(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)

        df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
        df.index.names = ['DSID', 'eventNumber']
        self.logger.debug("Set DSID/eventNumber as index")

        # merge TTrees
        if validate_duplicated_events:
            validation = '1:1'
            self.logger.info(f"Validating duplicated events in tree {TTree_name}...")
            self.__drop_duplicates(df)
            self.__drop_duplicate_event_numbers(df)
        else:
            validation = 'm:m'
            self.logger.info("Skipping duplicted events validation")

        # iterate over TTrees and merge
        for tree in tree_dict:
            if tree == TTree_name:
                continue

            self.logger.debug(f"Extracting {tree_dict[tree]} from {tree} tree...")
            alt_df = to_pandas(uproot.concatenate(data_path + ":" + tree, tree_dict[tree],
                                                  num_workers=config.n_threads, begin_chunk_size=chunksize))
            self.logger.debug(f"Extracted {len(alt_df)} events.")

            alt_df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
            alt_df.index.names = ['DSID', 'eventNumber']
            self.logger.debug("Set DSID/eventNumber as index")

            if validate_missing_events:
                self.logger.info(f"Checking for missing events in tree '{tree}'..")
                tree_is_truth = 'truth' in tree

                if tree_is_truth and not default_tree_truth:
                    if n_missing := len(df.index.difference(alt_df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{TTree_name}' tree not found in '{tree}' tree")
                    else:
                        self.logger.debug(f"All events in {TTree_name} tree found in {tree} tree")
                elif default_tree_truth and not tree_is_truth:
                    if n_missing := len(alt_df.index.difference(df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{tree}' tree not found in '{TTree_name}' tree")
                    else:
                        self.logger.debug(f"All events in {tree} tree found in {TTree_name} tree")
                else:
                    self.logger.info(f"Skipping missing events check. Not truth/reco tree combination")

            else:
                self.logger.info(f"Skipping missing events check in tree {tree}")

            if validate_duplicated_events:
                self.logger.info(f"Validating duplicated events in tree {tree}...")
                self.__drop_duplicates(alt_df)

            self.logger.debug("Merging with rest of dataframe...")
            df = pd.merge(df, alt_df, how='left', left_index=True, right_index=True, sort=False, copy=False,
                          validate=validation)

        if validate_sumofweights:
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

        # CLEANUP
        self.__rescale_to_gev(df)  # properly scale GeV columns

        # output number of truth and reco events
        if is_truth:
            n_truth = df['KFactor_weight_truth'].notna().sum()
            self.logger.info(f"number of truth events: {n_truth}")
        else:
            n_truth = 0
        if is_reco:
            n_reco = df['weight_KFactor'].notna().sum()
            self.logger.info(f"number of reco events: {n_reco}")
        else:
            n_reco = 0
        # small check
        assert len(df) == max(n_truth, n_reco), \
            f"Length of DataFrame ({len(df.index)}) doesn't match number of truth ({n_truth}) or reco events ({n_reco})"

        if is_truth and is_reco:
            # drop reco KFactor as long as values are the same for reco variables
            pd.testing.assert_series_equal(df.loc[pd.notna(df['weight_KFactor']), 'KFactor_weight_truth'],
                                           df['weight_KFactor'].dropna(),
                                           check_exact=True, check_names=False, check_index=False), \
                                                "reco and truth KFactors not equal"
            df.drop(columns='weight_KFactor', inplace=True)
            self.logger.debug("Dropped extra KFactor column")
        # ensure there is always only one KFactor column and it is named 'weight_KFactor'
        if is_truth:
            df.rename(columns={'KFactor_weight_truth': 'weight_KFactor'}, inplace=True)

        self.logger.info("Sorting by DSID...")
        df.sort_index(level='DSID', inplace=True)

        self.logger.info(f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}")

        return df

    def __drop_duplicates(self, df: pd.DataFrame) -> None:
        """Checks for and drops duplicated events and events with same event numbers for each dataset ID"""
        b_size = len(df.index)
        df.drop_duplicates(inplace=True)
        self.logger.info(f"{b_size - len(df.index)} duplicate events dropped")

    def __drop_duplicate_event_numbers(self, df: pd.DataFrame) -> None:
        b_size = len(df.index)
        df.index = df.index.drop_duplicates()
        self.logger.info(f"{b_size - len(df.index)} duplicate event numbers dropped")

    def __rescale_to_gev(self, df) -> None:
        """rescales to GeV because athena's default output is in MeV for some reason"""
        if GeV_columns := [
            column for column in df.columns
            if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])
        ]:
            df[GeV_columns] /= 1000
            self.logger.debug(f"Rescaled columns {GeV_columns} to GeV.")
        else:
            self.logger.debug(f"No columns rescaled to GeV.")

    def __calculate_vars(self, df: pd.DataFrame, vars_to_calc: set) -> None:
        """Calculate derived variables"""
        # calculate and combine special derived variables
        for var in vars_to_calc:
            # compute new column
            temp_cols = derived_vars[var]['var_args']
            func = derived_vars[var]['func']
            str_args = derived_vars[var]['var_args']
            self.logger.info(f"Computing '{var}' column from {temp_cols}...")
            df[var] = func(df, *str_args)

    def __create_cut_columns(self, df: pd.DataFrame, cutfile: Cutfile) -> None:
        """
        Creates boolean columns in dataframe corresponding to each cut
        """
        label = config.cut_label  # get cut label from config
        self.logger.info(f"Calculating {len(cutfile.cut_dicts)} cut columns...")

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

        for cut in cutfile.cut_dicts:
            if cut['relation'] not in op_dict:
                raise ValueError(f"Unexpected comparison operator: {cut['relation']}.")
            if not cut['is_symmetric']:
                df[cut['name'] + label] = op_dict[cut['relation']](df[cut['cut_var']], cut['cut_val'])
            else:  # take absolute value instead
                df[cut['name'] + label] = op_dict[cut['relation']](df[cut['cut_var']].abs(), cut['cut_val'])

    def __event_weight_reco(self,
                            df: pd.DataFrame,
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
            self.lumi * df[mc_weight] * abs(df[mc_weight]) / df[tot_weighted_events] * \
            df[KFactor] * df[pileup_weight] * df[lepton_SF]

    def __event_weight_truth(self,
                             df: pd.DataFrame,
                             mc_weight: str = 'weight_mc',
                             tot_weighted_events: str = 'totalEventsWeighted',
                             KFactor: str = 'weight_KFactor',
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
        return \
            self.lumi * df[mc_weight] * abs(df[mc_weight]) / df[tot_weighted_events] * \
            df[KFactor] * df[pileup_weight]
