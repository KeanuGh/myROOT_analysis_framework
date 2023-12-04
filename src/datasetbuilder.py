import glob
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Iterable, overload, Final, List

import ROOT  # type: ignore
import pandas as pd  # type: ignore
import uproot  # type: ignore
from awkward import to_dataframe  # type: ignore

from src.cutfile import Cutfile, Cut
from src.dataset import Dataset, RDataset, CUT_PREFIX  # PDataset
from src.logger import get_logger
from utils import file_utils, ROOT_utils, PMG_tool
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data

# total dataset luminosity per year (pb-1)
lumi_year: Final[dict] = {
    "2015": 3219.56,
    "2017": 44307.4,
    "2018": 5845.01,
    "2015+2016": 32988.1 + 3219.56,
}


def find_variables_in_string(s: str):
    """Find any variables in the variable list that are in given string"""
    split_string = set(re.findall(r"\w+|[.,!?;]", s))
    return split_string & set(variable_data.keys())


@dataclass(slots=True)
class DatasetBuilder:
    """
    Dataset builder class.
    Pass dataset build options to initialise class. Use DatasetBuilder.build() with inputs to return Dataset object.

    :param name: dataset name
    :param ttree: TTree in datapath to set as default tree, or list of trees to merge
    :param lumi: data luminosity for weight calculation and normalisation
    :param year: data year for luminosty mapping
    :param lumi: data luminosity. Pass either this or year
    :param label: Label to put on plots
    :param lepton: Name of charged DY lepton channel in dataset (if applicable)
    :param logger: logger to write to. Defaults to console output at DEBUG-level
    :param hard_cut: string representing a cut that should be applied to dataframe immediately (eg. "TauPt > 20")
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

    name: str = "data"
    ttree: str | Set[str] = ""
    year: str = "2015+2016"
    lumi: float = 139.0
    label: str = "data"
    lepton: str = "lepton"
    logger: logging.Logger = field(default_factory=get_logger)
    hard_cut: str = ""
    dataset_type: str = "dta"
    force_rebuild: bool = False
    force_recalc_cuts: bool = False
    force_recalc_vars: bool = False
    skip_verify_pkl: bool = False
    chunksize: int = 1024
    validate_missing_events: bool = True
    validate_duplicated_events: bool = True
    validate_sumofweights: bool = True

    def __post_init__(self):
        # argument checks
        if self.dataset_type == "dta":
            if not self.ttree:
                raise ValueError(f"No TTree name passed for dataset {self.name}")

            if not isinstance(self.ttree, str):
                self.ttree = set(self.ttree)
            else:
                self.ttree = {self.ttree}

        if self.year:
            self.lumi = lumi_year[self.year]

        self.dataset_type = self.dataset_type.lower()
        if self.dataset_type not in ("dta", "analysistop"):
            raise ValueError("Known dataset types: 'DTA', 'AnalysisTop'")

    @overload
    def build(self, data_path: str, cutfile: str | Path | Cutfile) -> Dataset:
        ...

    @overload
    def build(
        self,
        data_path: str,
        cutfile: str | Path | Cutfile,
    ) -> Dataset:
        ...

    @overload
    def build(
        self,
        data_path: str,
        cutfile: str | Path | Cutfile,
    ) -> Dataset:
        ...

    @overload
    def build(
        self,
        data_path: str,
        cuts: List[Cut],
        extract_vars: Set[str],
    ) -> Dataset:
        ...

    def build(
        self,
        data_path: str,
        cutfile: Cutfile | str | Path | None = None,
        cuts: List[Cut] | None = None,
        extract_vars: Set[str] | None = None,
        *,
        pkl_path: Path | None = None,
        tree_dict: Dict[str, Set[str]] | None = None,
        vars_to_calc: Set[str] | None = None,
    ) -> Dataset:
        """
        Builds a dataframe from cutfile inputs.

        :param data_path: Path to ROOT file(s) must be passed if not providing a pickle file
        :param cutfile: If passed, uses Cutfile object to build dataframe can pass cutfile object or path to cutfile
        :param cuts: List of Cut objects if cuts are to be applied but no cutfile is given
        :param extract_vars: set of branches to extract if cutfile is not given
        :param pkl_path: If passed opens pickle file, checks against cut and variable options
                         and calculates cuts and weighs if needed.
                         Can be passed instead of ROOT file but then will not be checked against cutfile
        :param tree_dict: If cutfile or cutfile_path not passed,
                          can pass dictionary of tree: variables to extract from Datapath
        :param vars_to_calc: list of variables to calculate to pass to add to DataFrame
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
        if self.dataset_type.lower() == "dta":
            self.force_rebuild = True  # TODO: rework this

        if data_path and not file_utils.file_exists(data_path):
            raise FileExistsError(f"File {data_path} not found.")

        # Parsing cutfile
        if isinstance(cutfile, (str, Path)):
            self.logger.info(f"Parsting cutfile '{cutfile}'...")

            cutfile_path = cutfile
            cutfile = Cutfile(cutfile, self.ttree, logger=self.logger)

            if self.logger.level == logging.DEBUG:
                # log contents of cutfile
                self.logger.debug("CUTFILE CONTENTS")
                self.logger.debug("---------------------------")
                with open(cutfile_path) as f:
                    for line in f.readlines():
                        self.logger.debug(line.rstrip("\n"))
                self.logger.debug("---------------------------")
        else:
            cutfile_path = ""

        if cuts and extract_vars:
            # separate variables to calculate and those to extract
            all_vars = {var for cut in cuts for var in cut.var} | extract_vars
            vars_to_calc = {var for var in all_vars if var in derived_vars}
            all_vars -= vars_to_calc
            # add all variables to all ttrees
            tree_dict = {tree: all_vars for tree in self.ttree}
        elif cutfile:
            # get tree dictionary, set of variables to calculate, and whether the dataset will contain truth, reco data
            tree_dict = cutfile.tree_dict
            vars_to_calc = cutfile.vars_to_calc
            all_vars = cutfile.all_vars
            cuts = cutfile.cuts
        elif tree_dict:
            if (vars_to_calc is None) or (cuts is None):
                raise ValueError(
                    "Must provide variables to cut and cut dictionary list if not building from cutfile"
                )
        else:
            raise ValueError(
                "Must provide cutfile, tree dict, or cuts and extract_vars to build DataFrame"
            )

        # check if vars are contained in label dictionary
        self.__check_axis_labels(tree_dict, vars_to_calc)

        # print debug information
        self.logger.debug("")
        self.logger.debug("DEBUG DATASET OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug(f"Input ROOT file(s):  {data_path}")
        self.logger.debug(f"Input pickle file(s): {pkl_path}")
        self.logger.debug(f"TTree: {self.ttree}")
        if cutfile_path:
            self.logger.debug(f"Cutfile: {cutfile_path}")
        if cutfile:
            self.logger.debug("Cuts from cutfile:")
            cutfile.log_cuts(debug=True)
        self.logger.debug(f"Hard cut applied: {self.hard_cut}")
        self.logger.debug(f"Forced dataset rebuild: {self.force_rebuild}")
        self.logger.debug(f"Calculating cuts: {__create_cut_cols or self.force_recalc_cuts}")
        if pkl_path:
            self.logger.debug(f"Pickle datapath: {pkl_path}")
            self.logger.debug(f"Skipping pickle file verification: {self.skip_verify_pkl}")
        self.logger.debug(f"Chunksize for ROOT file import: {self.chunksize}")
        self.logger.debug(f"Validate missing events: {self.validate_missing_events}")
        self.logger.debug(f"Validate duplicated events: {self.validate_duplicated_events}")
        self.logger.debug(f"Validate sum of weights: {self.validate_sumofweights}")
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
        if pkl_path and pkl_path.exists() and (not self.force_rebuild):
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

                elif (
                    not self.__check_calc_var_cols(cols, vars_to_calc)
                    and not self.force_recalc_vars
                ):
                    if not self.__check_argument_var_cols(
                        cols, vars_to_calc, raise_error=__error_at_rebuild
                    ):
                        __build_df = True
                    else:
                        __calculate_vars = True

                if not __build_df:
                    self.logger.debug("Found all necessary variables. No rebuild necessary")
                    __create_cut_cols = (
                        True if self.force_recalc_cuts else not self.__check_cut_cols(cols, cuts)
                    )
        else:
            __build_df = True

        # BUILD
        # ===============================
        if __build_df:
            df = self.build_dataframe(
                data_path=data_path, tree_dict=tree_dict, vars_to_calc=vars_to_calc
            )
            __create_cut_cols = True
            __calculate_vars = True

        # POST-PROCESSING
        # ===============================
        if isinstance(df, pd.DataFrame):
            raise NotImplementedError("No longer support analysistop")
        #     # calculate variables
        #     if __calculate_vars or self.force_recalc_vars:
        #         self.__calculate_vars(df, vars_to_calc)
        #
        #     # calculate cut columns
        #     if __create_cut_cols or self.force_recalc_cuts:
        #         self.__create_cut_columns(df, cuts)
        #
        #     # GENERATE CUTFLOW
        #     cutflow = PCutflow(
        #         df,
        #         cutfile.cuts,
        #         logger=self.logger,
        #     )
        #
        #     # BUILD DATASET
        #     # ===============================
        #     dataset: Dataset = PDataset(
        #         name=self.name,
        #         df=df,
        #         cutfile=cutfile,
        #         cutflow=cutflow,
        #         logger=self.logger,
        #         lumi=self.lumi,
        #         label=self.label,
        #         lepton=self.lepton,
        #     )
        #
        #     # print pickle file if anything is new/changed
        #     if pkl_path and (__build_df or __create_cut_cols or __calculate_vars):
        #         dataset.save_file(str(pkl_path))
        #     else:
        #         self.logger.debug("No pickle file saved.")

        else:
            # BUILD DATASET
            # ===============================
            dataset = RDataset(
                name=self.name,
                df=df,
                cuts=cuts,
                all_vars=all_vars,
                logger=self.logger,
                lumi=self.lumi,
                label=self.label,
                lepton=self.lepton,
            )

        return dataset

    def __read_pkl_df(self, pkl_path: str | Path) -> pd.DataFrame:
        """Read in a dataset pickle file and check its type and index"""
        self.logger.info(f"Reading data from {pkl_path}...")
        df: pd.DataFrame = pd.read_pickle(pkl_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Pickle file does not contain a pandas DataFrame. Found type {type(df)}"
            )
        assert df.index.names == (
            "DSID",
            "eventNumber",
        ), f"Pickled DataFrame index incorrect: {df.index.names}"
        return df

    # ===============================
    # =========== CHECKS ============
    # ===============================
    def __check_axis_labels(self, tree_dict: Dict[str, Set[str]], calc_vars: Set[str]) -> None:
        """Check whether variables exist in"""
        all_vars = {var for var_set in tree_dict.values() for var in var_set} | calc_vars
        if unexpected_vars := [
            unexpected_var for unexpected_var in all_vars if unexpected_var not in variable_data
        ]:
            self.logger.warning(
                f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                "Some unexpected behaviour may occur."
            )

    def __check_var_cols(
        self, df_cols: Iterable, tree_dict: Dict[str, Set[str]], raise_error: bool = False
    ) -> bool:
        """Check whether required variables exist in DataFrame columns"""
        var_set = {var for var_set in tree_dict.values() for var in var_set}
        # variables that get renamed or put in index
        if missing_vars := {var for var in var_set if var not in df_cols}:
            if raise_error:
                raise ValueError(f"Variable(s) {missing_vars} missing from DataFrame")
            else:
                self.logger.info(f"Variable(s) {missing_vars} missing from DataFrame. Will rebuild")
                return False
        else:
            return True

    def __check_calc_var_cols(self, df_cols: Iterable, calc_vars: set) -> bool:
        """Check whether calculated variables exist in DataFrame"""
        if missing_vars := {var for var in calc_vars if var not in df_cols}:
            self.logger.info(
                f"Calculated variable(s) {missing_vars} missing from DataFrame. "
                f"Will check its argument variables"
            )
            return False
        else:
            return True

    def __check_argument_var_cols(
        self, df_cols: Iterable, calc_vars: set, raise_error: bool = False
    ) -> bool:
        """Check whether variables necessary to calculate calc_vars exist in DataFrame"""
        for calc_var in calc_vars:
            if missing_vars := {
                var for var in derived_vars[calc_var]["var_args"] if var not in df_cols
            }:
                if raise_error:
                    raise ValueError(f"Argument variable(s) {missing_vars} missing from DataFrame")
                else:
                    self.logger.info(
                        f"Argument variable(s) {missing_vars} missing from DataFrame. Will rebuild"
                    )
                    return False
        return True

    def __check_cut_cols(
        self, df_cols: Iterable, cuts: List[Cut], raise_error: bool = False
    ) -> bool:
        """Check whether all necessary cut columns exist in DataFrame columns not including any hard cut"""
        cut_cols = {CUT_PREFIX + cut.name for cut in cuts}
        if missing_vars := {var for var in cut_cols if var not in df_cols}:
            if raise_error:
                raise ValueError(f"Cut column(s) {missing_vars} missing from DataFrame")
            else:
                self.logger.info(
                    f"Cut column(s) {missing_vars} missing from DataFrame. Will calculate"
                )
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
        vars_to_calc: Set[str] | None = None,
    ) -> pd.DataFrame | ROOT.RDataFrame:
        """send off dataframe builder to correct dataset type"""
        match self.dataset_type:
            case "analysistop":
                raise NotImplementedError("Using analysistop depreciated")
                # return self.__build_dataframe_analysistop(data_path=data_path, tree_dict=tree_dict)
            case "dta":
                return self.__build_dataframe_dta(
                    data_path=data_path, tree_dict=tree_dict, vars_to_calc=vars_to_calc
                )
            case _:
                raise ValueError(f"Unknown dataset {self.dataset_type}")

    # def __build_dataframe_analysistop(
    #     self,
    #     data_path: str,
    #     tree_dict: Dict[str, Set[str]],
    # ) -> pd.DataFrame:
    #     """
    #      Builds a dataframe
    #
    #     :param data_path: path to ROOT datafile(s)
    #     :param tree_dict: dictionary of tree: variables to extract from Datapath
    #     :return: output dataframe containing columns corresponding to necessary variables
    #     """
    #     self.logger.info(
    #         f"Building DataFrame from {data_path} ({file_utils.n_files(data_path)} file(s))..."
    #     )
    #
    #     if isinstance(self.ttree, set):
    #         raise ValueError("Pass only one default tree for analysistop dataset")
    #
    #     # check hard cut variable(s) exist
    #     vars_to_extract = set().union(*tree_dict.values())
    #     hard_cut_vars = find_variables_in_string(self.hard_cut)
    #     for hard_cut_var in hard_cut_vars:
    #         if hard_cut_var in derived_vars:
    #             assert (
    #                 set(derived_vars[hard_cut_var]) & vars_to_extract
    #             ), f"Missing variables necessary to apply hard cut '{self.hard_cut}': Missing dependencies for {hard_cut_var}"
    #         else:
    #             assert (
    #                 hard_cut_var in vars_to_extract
    #             ), f"Missing variables necessary to apply hard cut '{self.hard_cut}': Missing {hard_cut_var}"
    #
    #     # is the default tree a truth tree?
    #     default_tree_truth = "truth" in self.ttree
    #
    #     # add necessary variables to tree dictionary
    #     tree_dict = self.__add_necessary_analysistop_variables(tree_dict)
    #
    #     t1 = time.time()
    #
    #     # Extract main tree and event weights
    #     # ---------------------------------------------------------------------------------
    #     self.logger.info(f"Extracting {tree_dict[self.ttree]} from {self.ttree} tree...")
    #     df = to_dataframe(
    #         uproot.concatenate(
    #             str(data_path) + ":" + self.ttree,
    #             tree_dict[self.ttree],
    #             num_workers=-1,
    #             begin_chunk_size=self.chunksize,
    #         ),
    #         how="outer",
    #     )
    #     self.logger.debug(f"Extracted {len(df)} events.")
    #
    #     self.logger.info(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
    #     sumw = to_dataframe(
    #         uproot.concatenate(
    #             str(data_path) + ":sumWeights",
    #             ["totalEventsWeighted", "dsid"],
    #             num_workers=-1,
    #             begin_chunk_size=self.chunksize,
    #         ),
    #         how="outer",
    #     )
    #
    #     self.logger.info(f"Calculating sum of weights and merging...")
    #     sumw = sumw.groupby("dsid").sum()
    #     df = pd.merge(df, sumw, left_on="mcChannelNumber", right_on="dsid", sort=False, copy=False)
    #
    #     df.set_index(["mcChannelNumber", "eventNumber"], inplace=True)
    #     df.index.names = ["DSID", "eventNumber"]
    #     self.logger.debug("Set DSID/eventNumber as index")
    #     # -----------------------------------------------------------------------------------
    #
    #     # iterate over other TTrees, merge & validate
    #     # -----------------------------------------------------------------------------------
    #     for tree in tree_dict:
    #         if tree == self.ttree:
    #             continue
    #
    #         self.logger.info(f"Extracting {tree_dict[tree]} from {tree} tree...")
    #         alt_df = to_dataframe(
    #             uproot.concatenate(
    #                 str(data_path) + ":" + tree,
    #                 tree_dict[tree],
    #                 num_workers=-1,
    #                 begin_chunk_size=self.chunksize,
    #             ),
    #             how="outer",
    #         )
    #         self.logger.debug(f"Extracted {len(alt_df)} events.")
    #
    #         alt_df.set_index(["mcChannelNumber", "eventNumber"], inplace=True)
    #         alt_df.index.names = ["DSID", "eventNumber"]
    #         self.logger.debug("Set DSID/eventNumber as index")
    #
    #         if self.validate_missing_events:
    #             self.logger.info(f"Checking for missing events in tree '{tree}'..")
    #             tree_is_truth = "truth" in tree
    #
    #             if tree_is_truth and not default_tree_truth:
    #                 if n_missing := len(df.index.difference(alt_df.index)):
    #                     raise Exception(
    #                         f"Found {n_missing} events in '{self.ttree}' tree not found in '{tree}' tree"
    #                     )
    #                 else:
    #                     self.logger.debug(f"All events in {self.ttree} tree found in {tree} tree")
    #             elif default_tree_truth and not tree_is_truth:
    #                 if n_missing := len(alt_df.index.difference(df.index)):
    #                     raise Exception(
    #                         f"Found {n_missing} events in '{tree}' tree not found in '{self.ttree}' tree"
    #                     )
    #                 else:
    #                     self.logger.debug(f"All events in {tree} tree found in {self.ttree} tree")
    #             else:
    #                 self.logger.info(
    #                     f"Skipping missing events check. Not truth/reco tree combination"
    #                 )
    #
    #         else:
    #             self.logger.info(f"Skipping missing events check in tree {tree}")
    #
    #         self.logger.info("Merging with rest of dataframe...")
    #         df = pd.merge(
    #             df, alt_df, how="left", left_index=True, right_index=True, sort=False, copy=False
    #         )
    #     # -------------------------------------------------------------------------------------
    #
    #     # validate
    #     if self.validate_duplicated_events:
    #         self.logger.info(f"Validating duplicated events...")
    #         # self.__drop_duplicates(df)
    #         df = self.__drop_duplicate_index(df)
    #     else:
    #         self.logger.info("Skipped duplicted events validation")
    #
    #     if self.validate_sumofweights:
    #         # sanity check to make sure totalEventsWeighted really is what it says it is
    #         # also output DSID metadata
    #         df_id_sub = df["weight_mc"].groupby(level="DSID").sum()
    #         for dsid, df_id in df.groupby(level="DSID"):
    #             unique_totalEventsWeighted = df_id["totalEventsWeighted"].unique()
    #             if len(unique_totalEventsWeighted) != 1:
    #                 self.logger.warning(
    #                     "totalEventsWeighted should only have one value per DSID. "
    #                     f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} "
    #                     f"for DSID {dsid}"
    #                 )
    #
    #             dsid_weight = df_id_sub[dsid]
    #             totalEventsWeighted = df_id["totalEventsWeighted"].values[0]
    #             if dsid_weight != totalEventsWeighted:
    #                 self.logger.warning(
    #                     f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
    #                     f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}"
    #                 )
    #     else:
    #         self.logger.info("Skipping sum of weights validation")
    #     # ---------------------------------------------------------------------------------------
    #
    #     # WEIGHTS
    #     # ---------------------------------------------------------------------------------------
    #     if self._is_truth:
    #         """
    #         Calculate total truth event weights
    #
    #         lumi_data taken from year
    #         mc_weight = +/-1 * cross_section
    #         scale factor = weight_leptonSF
    #         KFactor = KFactor_weight_truth
    #         pileup = weight_pileup
    #         recoweight = scalefactors * KFactor * pileup
    #         lumi_weight = mc_weight * lumi_data / sum of event weights
    #
    #         total event weight = lumi_weight * truth_weight * reco_weight
    #         """
    #         self.logger.info(f"Calculating truth weights for {self.name}...")
    #         df["truth_weight"] = (
    #             self.lumi
    #             * df["weight_mc"]
    #             * abs(df["weight_mc"])
    #             / df["totalEventsWeighted"]
    #             * df["KFactor_weight_truth"]
    #             * df["weight_pileup"]
    #         )
    #         df["base_weight"] = df["weight_mc"] * abs(df["weight_mc"]) / df["totalEventsWeighted"]
    #         # EVERY event MUST have a truth weight
    #         self.logger.info(f"Verifying truth weight for {self.name}...")
    #         if df["truth_weight"].isna().any():
    #             raise ValueError("NAN values in truth weights!")
    #
    #     if self._is_reco:
    #         """
    #         Calculate total reco event weights
    #
    #         lumi_data taken from year
    #         mc_weight = +/-1 * cross_section
    #         kFactor = kFactor_weight_truth
    #         pileup = weight_pileup
    #         truth_weight = kFactor * pileup
    #         lumi_weight = mc_weight * lumi_data / sum of event weights
    #
    #         total event weight = lumi_weight * truth_weight * reco_weight
    #         """
    #         self.logger.info(f"Calculating reco weight for {self.name}...")
    #         df["reco_weight"] = (
    #             self.lumi
    #             * df["weight_mc"]
    #             * abs(df["weight_mc"])
    #             * df["weight_KFactor"]
    #             * df["weight_pileup"]
    #             * df["weight_leptonSF"]
    #             / df["totalEventsWeighted"]
    #         )
    #         # every reco event MUST have a reco weight
    #         self.logger.info(f"Verifying reco weight for {self.name}...")
    #         assert (
    #             df["reco_weight"].notna().sum() == df["weight_leptonSF"].notna().sum()
    #         ), "Different number of events between reco weight and lepton scale factors!"
    #
    #     # output number of truth and reco events
    #     if self._is_truth:
    #         n_truth = df["KFactor_weight_truth"].notna().sum()
    #         self.logger.info(f"number of truth events: {n_truth}")
    #     else:
    #         n_truth = 0
    #     if self._is_reco:
    #         n_reco = df["weight_KFactor"].notna().sum()
    #         self.logger.info(f"number of reco events: {n_reco}")
    #     else:
    #         n_reco = 0
    #
    #     # small check
    #     if self._is_truth and self._is_reco:
    #         assert len(df) == max(n_truth, n_reco), (
    #             f"Length of DataFrame ({len(df.index)}) "
    #             f"doesn't match number of truth ({n_truth}) or reco events ({n_reco})"
    #         )
    #
    #     if self._is_truth and self._is_reco:
    #         # make sure KFactor is the same for all reco and truth variables
    #         pd.testing.assert_series_equal(
    #             df.loc[pd.notna(df["weight_KFactor"]), "KFactor_weight_truth"],
    #             df["weight_KFactor"].dropna(),
    #             check_exact=True,
    #             check_names=False,
    #             check_index=False,
    #         ), "reco and truth KFactors not equal"
    #     # ensure there is always only one KFactor column and it is named 'weight_KFactor'
    #     if self._is_truth:
    #         df.rename(columns={"KFactor_weight_truth": "weight_KFactor"}, inplace=True)
    #
    #     # CLEANUP
    #     # ---------------------------------------------------------------------------------------
    #     self.__rescale_to_gev(df)  # properly scale GeV columns
    #
    #     # apply hard cuts
    #     if self.hard_cut:
    #         self.logger.debug(f"Applying hard cut on dataset: {self.hard_cut}")
    #         df.query(self.hard_cut, inplace=True)
    #
    #     self.logger.info("Sorting by DSID...")
    #     df.sort_index(level="DSID", inplace=True)
    #
    #     self.logger.info(
    #         f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}"
    #     )
    #
    #     return df

    def __build_dataframe_dta(
        self,
        data_path: str | Path,
        tree_dict: dict[str, Set[str]],
        vars_to_calc: Set[str] | None = None,
        unravel_vectors: bool = True,
    ) -> ROOT.RDataFrame:
        """Build DataFrame from given files and TTree/branch combinations from DTA output"""

        t1 = time.time()

        # add variables necessary to calculate weights etc
        tree_dict = self.__add_necessary_dta_variables(tree_dict)

        # should always be a set
        if not isinstance(self.ttree, set):
            ttrees = {self.ttree}
        else:
            ttrees = self.ttree

        # check all branches being extracted from each tree are the same, or TChain will throw a fit
        if all(not x == next(iter(tree_dict.values())) for x in tree_dict.values()):
            raise ValueError(
                "Can only extract branches with the same name from multiple trees! "
                "Trying to extract the following branches for each tree:\n\n"
                + "\n".join(tree + ":\n\t" + "\n\t".join(tree_dict[tree]) for tree in tree_dict)
            )
        import_cols = tree_dict[next(iter(tree_dict))]

        self.logger.info(f"Initiating RDataFrame from trees {ttrees} in {data_path}")

        # create c++ map for dataset ID metadatas
        self.logger.debug("Calculating DSID metadata...")
        # each tree /SHOULD/ have the same dsid here
        dsid_metadata = self._get_dsid_values(data_path, list(ttrees)[0])
        self.logger.info(f"Sum of weights per dsid:\n{dsid_metadata['sumOfWeights']}")
        ROOT.gInterpreter.Declare(
            f"""
                std::map<int, float> {self.name}_dsid_sumw{{{','.join(f'{{{t.Index}, {t.sumOfWeights}}}' for t in dsid_metadata.itertuples())}}};
                std::map<int, float> {self.name}_dsid_xsec{{{','.join(f'{{{t.Index}, {t.cross_section}}}' for t in dsid_metadata.itertuples())}}};
                std::map<int, float> {self.name}_dsid_pmgf{{{','.join(f'{{{t.Index}, {t.PMG_factor}}}' for t in dsid_metadata.itertuples())}}};
            """
        )

        # make rdataframe
        Rdf = ROOT_utils.init_rdataframe(self.name, data_path, ttrees)

        # check columns exist in dataframe
        if missing_cols := (set(import_cols) - set(Rdf.GetColumnNames())):
            raise ValueError("Missing column(s) in RDataFrame: \n\t" + "\n\t".join(missing_cols))

        # workaround for broken wmunu file (zero'd out dataset IDs)
        workaround_str = "(mcChannel == 0) ? 700446 : mcChannel"

        # create weights
        Rdf = (
            Rdf.Define(
                "truth_weight",
                f"(mcWeight * rwCorr * {self.lumi} * prwWeight * {self.name}_dsid_pmgf[{workaround_str}]) "
                f"/ {self.name}_dsid_sumw[{workaround_str}]",
            ).Define(
                "reco_weight",
                f"(weight * {self.lumi} * {self.name}_dsid_pmgf[{workaround_str}]) "
                f"/ {self.name}_dsid_sumw[{workaround_str}]",
            )
            # .Define(
            #     "ele_weight_reco",
            #     "reco_weight * Ele_recoSF * Ele_idSF * Ele_isoSF",
            # )
            # .Define(
            #     "muon_weight_reco",
            #     "reco_weight * Muon_recoSF * Muon_isoSF * Muon_ttvaSF",
            # )
            # .Define("tau_weight_reco", "reco_weight * TauRecoSF")
            # .Define("jet_weight_reco", "reco_weight * Jet_btSF")
        )

        # rescale energy columns to GeV
        for gev_column in [
            column
            for column in import_cols
            if (column in variable_data) and (variable_data[column]["units"] == "GeV")
        ]:
            Rdf = Rdf.Redefine(gev_column, f"{gev_column} / 1000")

        if unravel_vectors:
            # routine to separate vector branches into separate variables
            badcols = set()  # save old vector column names to avoid extracting them later
            hard_cut_vars = find_variables_in_string(self.hard_cut)
            self.logger.debug("Shrinking vectors:")
            rdf_cols = import_cols | hard_cut_vars | set(Rdf.GetDefinedColumnNames())
            if vars_to_calc:
                rdf_cols -= vars_to_calc
            for col_name in rdf_cols:
                # unravel vector-type columns
                col_type = Rdf.GetColumnType(col_name)
                debug_str = f"\t- {col_name}: {col_type}"
                if "ROOT::VecOps::RVec" in col_type:
                    # skip non-numeric vector types
                    if col_type == "ROOT::VecOps::RVec<string>":
                        self.logger.debug(f"\t- Skipping string vector column {col_name}")
                        badcols.add(col_name)

                    # elif "jet" in str(col_name).lower():
                    #     # create three new columns for each possible jet
                    #     debug_str += " -> "
                    #     for i in range(3):
                    #         new_name = col_name + str(i + 1)
                    #         Rdf = Rdf.Define(f"{new_name}", f"getVecVal(&{col_name},{i})")
                    #         debug_str += f"{new_name}: {Rdf.GetColumnType(new_name)}, "
                    #         import_cols.add(new_name)
                    #     badcols.add(col_name)

                    # elif "neutrino" in str(col_name).lower():
                    #     Rdf = Rdf.Define(f"{col_name}1", f"(&{col_name})->at(0);")
                    #     Rdf = Rdf.Define(f"{col_name}2", f"(&{col_name})->at(1);")
                    #     Rdf = Rdf.Define(
                    #         f"{col_name}3",
                    #         f"((&{col_name})->size() > 2) ? (&{col_name})->at(2) : NAN;",
                    #     )
                    #     badcols.add(col_name)
                    #     import_cols |= {f"{col_name}1", f"{col_name}2", f"{col_name}3"}

                    else:
                        Rdf = Rdf.Redefine(
                            f"{col_name}",
                            f"((&{col_name})->size() > 0) ? (&{col_name})->at(0) : NAN;",
                        )
                        debug_str += f" -> {Rdf.GetColumnType(col_name)}"
                self.logger.debug(debug_str)

        # calculate derived variables
        if vars_to_calc:
            self.logger.debug(f"calculating variables: {derived_vars}...")
            for derived_var in vars_to_calc:
                function = derived_vars[derived_var]["cfunc"]
                args = derived_vars[derived_var]["var_args"]
                func_str = f"{function}({','.join(args)})"

                Rdf = Rdf.Define(derived_var, func_str)

        # apply any hard cuts
        if self.hard_cut:
            Rdf = Rdf.Filter(self.hard_cut)

        self.logger.info(
            f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}"
        )

        return Rdf

    # ===============================
    # ========== HELPERS ============
    # ===============================
    def _get_dsid_values(self, path: str | Path, ttree_name: str = "") -> pd.DataFrame:
        """Return DataFrame containing sumw, xs and PMG factor per DSID"""
        # PNG factor is cross-section * kfactor * filter eff.
        files_list = glob.glob(str(path))
        dsid_sumw: Dict[int, float] = dict()
        dsid_xs: Dict[int, float] = dict()
        dsid_pmg_factor: Dict[int, float] = dict()
        dsid_phys_short: Dict[int, str] = dict()

        # loop over files and sum sumw values per dataset ID (assuming each file only has one dataset ID value)
        prev_dsid = None
        for file in files_list:
            with ROOT_utils.ROOT_TFile_mgr(file, "read") as tfile:
                if not tfile.GetListOfKeys().Contains(ttree_name):
                    raise ValueError(
                        "Missing key '{}' from file {}\nKeys available: {}".format(
                            ttree_name,
                            tfile,
                            "\n".join([key.GetName() for key in tfile.GetListOfKeys()]),
                        )
                    )

                # read first DSID from branch (there should only be one value)
                tree = tfile.Get(ttree_name)
                tree.GetEntry(0)
                dsid = tree.mcChannel
                sumw = tfile.Get("sumOfWeights").GetBinContent(4)  # bin 4 is AOD sum of weights

                if dsid == 0:
                    self.logger.warning("Passed a '0' DSID")
                    # workaround for broken wmunu samples
                    if "Sh_2211_Wmunu_mW_120_ECMS_BFilter" in str(file):
                        self.logger.warning(
                            f"Working around broken DSID for file {file}, setting DSID to 700446"
                        )
                        dsid = 700446
                    else:
                        self.logger.error(
                            f"Unknown DSID for file {file}, THIS WILL LEAD TO A BROKEN DATASET!!!"
                        )

                self.logger.debug(f"dsid: {dsid}: sumw {sumw} for file {file}")
                if dsid not in dsid_sumw:
                    dsid_sumw[dsid] = sumw
                else:
                    dsid_sumw[dsid] += sumw

            if prev_dsid != dsid:  # do only for one dsid
                if not PMG_tool.check_dsid(dsid):
                    raise self.logger.error(f"Unknown dataset ID: {dsid}")

                xs = PMG_tool.get_crossSection(dsid)
                dsid_xs[dsid] = xs
                dsid_pmg_factor[dsid] = (
                    xs * PMG_tool.get_kFactor(dsid) * PMG_tool.get_genFiltEff(dsid)
                )
                dsid_phys_short[dsid] = PMG_tool.get_physics_short(dsid)

            prev_dsid = dsid

        df = pd.concat(
            [
                pd.DataFrame.from_dict(dsid_sumw, orient="index", columns=["sumOfWeights"]),
                pd.DataFrame.from_dict(dsid_xs, orient="index", columns=["cross_section"]),
                pd.DataFrame.from_dict(dsid_pmg_factor, orient="index", columns=["PMG_factor"]),
            ],
            axis=1,
            join="inner",
        )
        df.index.name = "mcChannel"

        return df

    def __add_necessary_analysistop_variables(
        self, tree_dict: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Add variables necessary to extract for analysistop output"""
        if isinstance(self.ttree, set):
            raise ValueError("Pass only one default tree for analysistop dataset")

        # only add these to 'main tree' to avoid merge issues
        tree_dict[self.ttree] |= {"weight_mc", "weight_pileup"}

        # check for variables in hard cut
        hard_cut_vars = find_variables_in_string(self.hard_cut)
        reco_vars = {var for var in hard_cut_vars if variable_data[var]["tag"] == "RECO"}
        truth_vars = {var for var in hard_cut_vars if variable_data[var]["tag"] == "TRUTH"}

        for tree in tree_dict:
            # add necessary metadata to all trees
            tree_dict[tree] |= {"mcChannelNumber", "eventNumber"}
            if "nominal" in tree.lower():
                self.logger.debug(
                    f"Detected {tree} as reco tree, "
                    f"adding 'weight_leptonSF' and 'weight_KFactor' to tree variables"
                )
                tree_dict[tree] |= {"weight_leptonSF", "weight_KFactor"} | reco_vars
            elif "truth" in tree.lower():
                self.logger.debug(
                    f"Detected {tree} as truth tree, "
                    f"adding 'KFactor_weight_truth' to tree variables"
                )
                tree_dict[tree] |= {"KFactor_weight_truth"} | truth_vars
            else:
                self.logger.debug(f"Neither {tree} as truth nor reco dataset detected.")

        return tree_dict

    def __add_necessary_dta_variables(self, tree_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        necc_vars = {
            "weight",
            "mcWeight",
            "mcChannel",
            # "prwWeight",
            # "rwCorr",
            "runNumber",
            "eventNumber",
            "passTruth",
            "passReco",
        }

        for tree in tree_dict:
            tree_dict[tree] |= necc_vars
        self.logger.debug(f"Added {necc_vars} to DTA import")
        return tree_dict

    def __drop_duplicates(self, df: pd.DataFrame) -> None:
        """Checks for and drops duplicated events and events with same event numbers for each dataset ID"""
        b_size = len(df.index)
        df.drop_duplicates(inplace=True)
        self.logger.info(f"{b_size - len(df.index)} duplicate events dropped")

    def __drop_duplicate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        b_size = len(df.index)
        df = df[~df.index.duplicated(keep="first")]
        self.logger.info(f"{b_size - len(df.index)} duplicate event numbers dropped")
        return df

    def __rescale_to_gev(self, df) -> None:
        """rescales to GeV because athena's default output is in MeV for some reason"""
        if GeV_columns := [
            column
            for column in df.columns
            if (column in variable_data) and (variable_data[column]["units"] == "GeV")
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
            temp_cols = derived_vars[var]["var_args"]
            func = derived_vars[var]["func"]
            str_args = derived_vars[var]["var_args"]
            self.logger.info(f"Computing '{var}' column from {temp_cols}...")
            df[var] = func(df, *str_args)

    def __create_cut_columns(self, df: pd.DataFrame, cuts: List[Cut]) -> None:
        """Creates boolean columns in dataframe corresponding to each cut"""
        label = CUT_PREFIX  # get cut label from config
        self.logger.info(f"Calculating {len(cuts)} cut columns...")

        for cut in cuts:
            try:
                df[label + cut.name] = df.eval(cut.cutstr)
            except ValueError as e:
                raise Exception(f"Error in cut {cut.cutstr}:\n {e}")
