import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, OrderedDict, Set, Iterable, overload

import ROOT
import pandas as pd
import uproot
from awkward import to_pandas

import src.config as config
from src.cutfile import Cutfile, Cut
from src.cutflow import Cutflow
from src.dataset import Dataset
from src.logger import get_logger
from utils import file_utils
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data

# total dataset luminosity per year (fb-1)
lumi_year = {
    '2015': 3.21956,
    '2017': 44.3074,
    '2018': 58.4501,
    '2015+2016': 32.9881 + 3.21956
}
# declare helper function to unravel ROOT vector branches
ROOT.gInterpreter.Declare("""
float getVecVal(ROOT::VecOps::RVec<float> x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<float> x, int i) {
    if (x.size() > i)  return x[i];
    else               return NAN;
}
""")


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
    dataset_type: str = 'DTA'
    force_rebuild: bool = False
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

        self.dataset_type = self.dataset_type.lower()
        if self.dataset_type not in ('dta', 'analysistop'):
            raise ValueError("Known dataset types: 'DTA', 'AnalysisTop'")

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
            cuts: Optional[OrderedDict[str, Cut]] = None,
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
            cuts: OrderedDict[str, Cut] = None,
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
        :param cuts: OrderedDict of Cut objects if cuts are to be applied but no cutfile is given
        :return: full built Dataset object
        """
        __build_df = False
        __create_cut_cols = False
        __calculate_vars = False
        __error_at_rebuild = False

        # PRE-PROCESSING
        # ===============================
        # check arguments
        if not data_path and not pkl_path:
            raise ValueError("Either ROOT data or a pickled dataframe must be provided")
        elif not data_path and pkl_path:
            self.logger.warning("No ROOT data provided so pickled DataFrame cannot be checked")
            __build_df = False
            __error_at_rebuild = True
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
            cutfile = Cutfile(cutfile_path, self.TTree_name, logger=self.logger)

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
            tree_dict = cutfile.tree_dict
            vars_to_calc = cutfile.vars_to_calc
            is_truth, is_reco = cutfile.truth_reco(tree_dict)
            cuts = cutfile.cuts
        elif tree_dict:
            if (vars_to_calc is None) or (cuts is None):
                raise ValueError("Must provide variables to cut and cut dictionary list if not building from cutfile")
            else:
                is_truth, is_reco = Cutfile.truth_reco(tree_dict)
        else:
            raise ValueError("Must provide cutfile or tree dict to build DataFrame")

        # remove variables to calculate from variables to import from ROOT file
        for tree in tree_dict:
            tree_dict[tree] - vars_to_calc

        # check if vars are contained in label dictionary
        self.__check_axis_labels(tree_dict, vars_to_calc)

        # check if hard cut(s) exists in cutfile. If not, skip them
        self.__check_hard_cuts(cuts)

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
        self.logger.debug(f"Calculating cuts: {__create_cut_cols}")
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
        if pkl_path and (file_utils.file_exists(pkl_path)) and (not self.force_rebuild):
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
                        else not self.__check_cut_cols(cols, cuts)
                    )
        else:
            __build_df = True

        # BUILD
        # ===============================
        if __build_df:
            df = self.build_dataframe(
                data_path=data_path,
                tree_dict=tree_dict,
                is_truth=is_truth,
                is_reco=is_reco,
            )
            __create_cut_cols = True
            __calculate_vars = True

        # POST-PROCESSING
        # ===============================
        # calculate variables
        if __calculate_vars or self.force_recalc_vars:
            self.__calculate_vars(df, vars_to_calc)

        # calculate cut columns
        if __create_cut_cols or self.force_recalc_cuts:
            self.__create_cut_columns(df, cuts)

        # apply hard cut(s) if it exists in cut columns
        hard_cuts_to_apply = []
        for h_cut in self.hard_cut:
            if h_cut + config.cut_label not in df.columns:
                self.logger.debug(f"No cut {h_cut} in DataFrame. Assuming hard cut has already been applied")
                self.hard_cut.remove(h_cut)
                cutfile.cuts.pop(h_cut)
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
            # remove cut from cutflow as well
            for h_cut in self.hard_cut:
                cutfile.cuts.pop(h_cut)

        # GENERATE CUTFLOW
        cutflow = Cutflow(
            df,
            cutfile.cuts,
            self.logger,
        )

        # BUILD DATASET
        # ===============================
        dataset = Dataset(
            name=self.name,
            df=df,
            cutfile=cutfile,
            cutflow=cutflow,
            logger=self.logger,
            lumi=self.lumi,
            label=self.label,
            lepton=self.lepton
        )

        # print pickle file if anything is new/changed
        if pkl_path and (
                __build_df or
                __create_cut_cols or
                __calculate_vars
        ):
            dataset.save_pkl_file(pkl_path)
        else:
            self.logger.debug("Pickle file not saved, no changes made.")

        return dataset

    def __read_pkl_df(self, pkl_path: str) -> pd.DataFrame:
        """Read in a dataset pickle file and check its type and index"""
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
                               if unexpected_var not in variable_data]:
            self.logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                                "Some unexpected behaviour may occur.")

    def __check_hard_cuts(self, cuts: OrderedDict[str, Cut], error: bool = False) -> None:
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
            for cut in self.hard_cut:
                if cut not in cuts:
                    if error:
                        raise ValueError(f"No cut named '{self.hard_cut}' in cutfile")
                    else:
                        self.logger.debug(f"No cut named '{self.hard_cut}' in cutfile; skipping this hard cut")
                        self.hard_cut.remove(cut)

    def __check_var_cols(self, df_cols: Iterable, tree_dict: Dict[str, Set[str]], raise_error: bool = False) -> bool:
        """Check whether required variables exist in DataFrame columns"""
        var_set = {var for var_set in tree_dict.values() for var in var_set}
        # variables that get renamed or put in index
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
        """Check whether calculated variables exist in DataFrame"""
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

    def __check_cut_cols(self, df_cols: Iterable, cuts: OrderedDict[str, Cut], raise_error: bool = False) -> bool:
        """Check whether all necessary cut columns exist in DataFrame columns not including any hard cut"""
        cut_cols = {cut_name + config.cut_label for cut_name in cuts}
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

    # ===============================
    # ===== DATAFRAME FUNCTIONS =====
    # ===============================
    def build_dataframe(
            self,
            data_path: str,
            tree_dict: Dict[str, Set[str]],
            is_truth: bool,
            is_reco: bool,
    ) -> pd.DataFrame:
        if self.dataset_type == 'analysistop':
            return self.__build_dataframe_analysistop(
                data_path=data_path,
                tree_dict=tree_dict,
                is_truth=is_truth,
                is_reco=is_reco,
            )
        elif self.dataset_type == 'dta':
            return self.__build_dataframe_dta(
                data_path=data_path,
                tree_dict=tree_dict,
            )

    def __build_dataframe_analysistop(
            self,
            data_path: str,
            tree_dict: Dict[str, Set[str]],
            is_truth: bool,
            is_reco: bool,
    ) -> pd.DataFrame:
        """
         Builds a dataframe

        :param data_path: path to ROOT datafile(s)
        :param tree_dict: dictionary of tree: variables to extract from Datapath
        :param is_truth: whether dataset contains truth data
        :param is_reco: whether dataset contains reco data
        :return: output dataframe containing columns corresponding to necessary variables
        """
        self.logger.info(f"Building DataFrame from {data_path} ({file_utils.n_files(data_path)} file(s))...")

        # is the default tree a truth tree?
        default_tree_truth = 'truth' in self.TTree_name

        # add necessary variables to tree dictionary
        tree_dict = self.__add_necessary_analysistop_variables(tree_dict)

        t1 = time.time()

        # Extract main tree and event weights
        # ---------------------------------------------------------------------------------
        self.logger.debug(f"Extracting {tree_dict[self.TTree_name]} from {self.TTree_name} tree...")
        df = to_pandas(uproot.concatenate(data_path + ':' + self.TTree_name, tree_dict[self.TTree_name],
                                          num_workers=config.n_threads, begin_chunk_size=self.chunksize))
        self.logger.debug(f"Extracted {len(df)} events.")

        self.logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = to_pandas(uproot.concatenate(data_path + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                            num_workers=config.n_threads, begin_chunk_size=self.chunksize))

        self.logger.debug(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)

        df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
        df.index.names = ['DSID', 'eventNumber']
        self.logger.debug("Set DSID/eventNumber as index")

        # validate
        if self.validate_duplicated_events:
            validation = '1:1'
            self.logger.info(f"Validating duplicated events in tree {self.TTree_name}...")
            self.__drop_duplicates(df)
            self.__drop_duplicate_event_numbers(df)
        else:
            validation = 'm:m'
            self.logger.info("Skipping duplicted events validation")
        # -----------------------------------------------------------------------------------

        # iterate over other TTrees, merge & validate
        # -----------------------------------------------------------------------------------
        for tree in tree_dict:
            if tree == self.TTree_name:
                continue

            self.logger.debug(f"Extracting {tree_dict[tree]} from {tree} tree...")
            alt_df = to_pandas(uproot.concatenate(data_path + ":" + tree, tree_dict[tree],
                                                  num_workers=config.n_threads, begin_chunk_size=self.chunksize))
            self.logger.debug(f"Extracted {len(alt_df)} events.")

            alt_df.set_index(['mcChannelNumber', 'eventNumber'], inplace=True)
            alt_df.index.names = ['DSID', 'eventNumber']
            self.logger.debug("Set DSID/eventNumber as index")

            if self.validate_missing_events:
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

            if self.validate_duplicated_events:
                self.logger.info(f"Validating duplicated events in tree {tree}...")
                self.__drop_duplicates(alt_df)

            self.logger.debug("Merging with rest of dataframe...")
            df = pd.merge(df, alt_df, how='left', left_index=True, right_index=True, sort=False, copy=False,
                          validate=validation)
        # -------------------------------------------------------------------------------------

        if self.validate_sumofweights:
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
        # ---------------------------------------------------------------------------------------

        # calculate weights
        # ---------------------------------------------------------------------------------------
        if is_truth:
            self.logger.info(f"Calculating truth weight for {self.name}...")
            df['truth_weight'] = self.__event_weight_truth_analyistop(df)
            # EVERY event MUST have a truth weight
            self.logger.info(f"Verifying truth weight for {self.name}...")
            if df['truth_weight'].isna().any():
                raise ValueError("NAN values in truth weights!")
        if is_reco:
            self.logger.info(f"Calculating reco weight for {self.name}...")
            df['reco_weight'] = self.__event_weight_reco_analysistop(df)
            # every reco event MUST have a reco weight
            self.logger.info(f"Verifying reco weight for {self.name}...")
            if not is_truth:
                if df['reco_weight'].isna().any():
                    raise ValueError("NAN values in reco weights!")
            else:
                assert (~df['reco_weight'].isna()).sum() == (~df['weight_leptonSF'].isna()).sum(), \
                    "Different number of events for reco weight and lepton scale factors!"
        # ---------------------------------------------------------------------------------------

        # CLEANUP
        # ---------------------------------------------------------------------------------------
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
            # make sure KFactor is the same for all reco and truth variables
            pd.testing.assert_series_equal(
                df.loc[pd.notna(df['weight_KFactor']), 'KFactor_weight_truth'],
                df['weight_KFactor'].dropna(),
                check_exact=True, check_names=False, check_index=False
            ), "reco and truth KFactors not equal"
            df.drop(columns='weight_KFactor', inplace=True)
            self.logger.debug("Dropped extra KFactor column")
        # ensure there is always only one KFactor column and it is named 'weight_KFactor'
        if is_truth:
            df.rename(columns={'KFactor_weight_truth': 'weight_KFactor'}, inplace=True)

        self.logger.info("Sorting by DSID...")
        df.sort_index(level='DSID', inplace=True)

        self.logger.info(f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}")

        return df

    def __build_dataframe_dta(
            self,
            data_path: str,
            tree_dict: dict[str, Set[str]],
    ) -> pd.DataFrame:
        self.logger.info(f"Building DataFrame from {data_path} ({file_utils.n_files(data_path)} file(s))...")
        t1 = time.time()

        self.logger.debug("Initialising RDataframe..")
        Rdf = ROOT.RDataFrame(self.TTree_name, data_path)
        Rdf = Rdf.Filter("(passTruth == true) & (passReco == true)")

        # check tree(s)
        for tree in tree_dict:
            if tree != self.TTree_name:
                raise ValueError("Don't import more than one DTA tree at once. Got the following trees: ",
                                 ', '.join(tree_dict.keys()))

        tree_dict = self.__add_necessary_dta_variables(tree_dict)

        # check columns
        import_cols = tree_dict[self.TTree_name]
        all_tree_cols = list(Rdf.GetColumnNames())
        if missing_cols := [
            col for col in import_cols
            if col not in all_tree_cols
        ]:
            raise ValueError(f"No column(s) {missing_cols} in TTree {self.TTree_name}")

        # routine to separate vector branches into separate variables
        badcols = set()  # save old vector column names to avoid extracting them later
        for col_name in import_cols:
            col_type = Rdf.GetColumnType(col_name)

            # unravel vector-type columns
            if "ROOT::VecOps::RVec" in col_type:
                # skip non-numeric vector types
                if col_type == "ROOT::VecOps::RVec<string>":
                    badcols.add(col_name)

                elif 'jet' in str(col_name).lower():
                    # create three new columns for each possible jet
                    for i in range(3):
                        Rdf = Rdf.Define(f"{col_name}{i + 1}", f"getVecVal({col_name},{i})")
                    badcols.add(col_name)

                else:
                    Rdf = Rdf.Redefine(col_name, f"getVecVal({col_name},0)")

        # import needed columns to pandas dataframe
        cols_to_extract = [c for c in import_cols
                           if c not in badcols]
        self.logger.debug(f"Extracting {tree_dict[self.TTree_name]} from {self.TTree_name} tree...")
        df = pd.DataFrame(Rdf.AsNumpy(columns=cols_to_extract))
        self.logger.debug(f"Extracted {len(df)} events.")

        df.set_index(['mcChannel', 'eventNumber'], inplace=True)
        df.index.names = ['DSID', 'eventNumber']
        self.logger.debug("Set DSID/eventNumber as index")

        self.logger.info(f"Time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}")

        # validate
        if self.validate_duplicated_events:
            self.logger.info(f"Validating duplicated events in tree {self.TTree_name}...")
            self.__drop_duplicates(df)
            self.__drop_duplicate_event_numbers(df)
        else:
            self.logger.info("Skipping duplicted events validation")

        # filter events with nan weight values (why do these appear?)
        if (nbad_rows := len(df['weight'].notna())) > 0:
            df = df.loc[df['weight'].notna()]
            self.logger.info(f"Dropped {nbad_rows} rows with missing weight values")

        # rescale GeV columns
        self.__rescale_to_gev(df)

        # calc weights
        self.logger.info("Calculating DTA weights...")
        df['reco_weight'] = df['weight'] * self.lumi / df['mcWeight'].sum()
        df['truth_weight'] = df['mcWeight'] * self.lumi / df['mcWeight'].sum()

        # rename weight col for consistency
        df.rename(columns={'mcWeight': 'weight_mc'}, inplace=True)

        return df

    # ===============================
    # ========== HELPERS ============
    # ===============================
    def __add_necessary_analysistop_variables(self, tree_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Add variables necessary to extract"""
        # only add these to 'main tree' to avoid merge issues
        tree_dict[self.TTree_name] |= {'weight_mc', 'weight_pileup'}

        for tree in tree_dict:
            # add necessary metadata to all trees
            tree_dict[tree] |= {'mcChannelNumber', 'eventNumber'}
            if 'nominal' in tree.lower():
                self.logger.info(f"Detected {tree} as reco tree, "
                                 f"adding 'weight_leptonSF' and 'weight_KFactor' to tree variables")
                tree_dict[tree] |= {'weight_leptonSF', 'weight_KFactor'}
            elif 'truth' in tree.lower():
                self.logger.info(f"Detected {tree} as truth tree, "
                                 f"adding 'KFactor_weight_truth' to tree variables")
                tree_dict[tree].add('KFactor_weight_truth')
            else:
                self.logger.info(f"Neither {tree} as truth nor reco dataset detected.")

        return tree_dict

    def __add_necessary_dta_variables(self, tree_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        for tree in tree_dict:
            tree_dict[tree] |= {
                'weight',
                'mcWeight',
                'mcChannel',
                'mcWeight',
                'runNumber',
                'eventNumber',
                'passTruth',
                'passReco',
            }
        self.logger.debug("Added necessary variables to DTA import")
        return tree_dict

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
            if (column in variable_data) and (variable_data[column]['units'] == 'GeV')
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

    def __create_cut_columns(self, df: pd.DataFrame, cuts: OrderedDict[str, Cut]) -> None:
        """Creates boolean columns in dataframe corresponding to each cut"""
        label = config.cut_label  # get cut label from config
        self.logger.info(f"Calculating {len(cuts)} cut columns...")

        for cut in cuts.values():
            df[cut.name + label] = df.eval(cut.cutstr)

    def __event_weight_reco_analysistop(self,
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

    def __event_weight_truth_analyistop(self,
                                        df: pd.DataFrame,
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
        return \
            self.lumi * df[mc_weight] * abs(df[mc_weight]) / df[tot_weighted_events] * \
            df[KFactor] * df[pileup_weight]
