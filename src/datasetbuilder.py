import logging
import re
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, OrderedDict, Set, Iterable, overload, Final

import ROOT
import pandas as pd
import uproot
from awkward import to_pandas

import src.config as config
from src.cutfile import Cutfile, Cut
from src.cutflow import Cutflow
from src.dataset import Dataset
from src.logger import get_logger
from utils import file_utils, ROOT_utils
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
    :param TTree_name: TTree in datapath to set as default tree, or list of trees to merge
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
    TTree_name: str | Set[str] = "truth"
    year: str = "2015+2016"
    lumi: float = None
    label: str = ("data",)
    lepton: str = "lepton"
    logger: logging.Logger = field(default_factory=get_logger)
    hard_cut: str = ""
    dataset_type: str = "DTA"
    force_rebuild: bool = False
    force_recalc_cuts: bool = False
    force_recalc_vars: bool = False
    skip_verify_pkl: bool = False
    chunksize: int = 1024
    validate_missing_events: bool = True
    validate_duplicated_events: bool = True
    validate_sumofweights: bool = True
    _is_reco: bool = field(init=False)
    _is_truth: bool = field(init=False)

    def __post_init__(self):
        # argument checks
        if (self.dataset_type == "analysistop") and (not isinstance(self.TTree_name, str)):
            raise ValueError("Only use one default tree with analysistop ntuples.")
        if self.dataset_type == "dta":
            if not isinstance(self.TTree_name, str):
                self.TTree_name = set(self.TTree_name)
            else:
                self.TTree_name = {self.TTree_name}

        if self.lumi and self.year:
            raise ValueError("Pass either lumi or year")
        elif self.year:
            self.lumi = lumi_year[self.year]

        self.dataset_type = self.dataset_type.lower()
        if self.dataset_type not in ("dta", "analysistop"):
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
        cuts: OrderedDict[str, Cut] | None = None,
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
                self.logger.debug("---------------------------")
                with open(cutfile_path) as f:
                    for line in f.readlines():
                        self.logger.debug(line.rstrip("\n"))
                self.logger.debug("---------------------------")
        if cutfile:
            # get tree dictionary, set of variables to calculate, and whether the dataset will contain truth, reco data
            tree_dict = cutfile.tree_dict
            vars_to_calc = cutfile.vars_to_calc
            self._is_truth, self._is_reco = cutfile.truth_reco(tree_dict)
            cuts = cutfile.cuts
        elif tree_dict:
            if (vars_to_calc is None) or (cuts is None):
                raise ValueError(
                    "Must provide variables to cut and cut dictionary list if not building from cutfile"
                )
        else:
            raise ValueError("Must provide cutfile or tree dict to build DataFrame")

        # remove variables to calculate from variables to import from ROOT file
        for tree in tree_dict:
            tree_dict[tree] - vars_to_calc

        # check if vars are contained in label dictionary
        self.__check_axis_labels(tree_dict, vars_to_calc)

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
                data_path=data_path,
                tree_dict=tree_dict,
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
            lepton=self.lepton,
        )

        # print pickle file if anything is new/changed
        if pkl_path and (__build_df or __create_cut_cols or __calculate_vars):
            dataset.save_pkl_file(pkl_path)
        else:
            self.logger.debug("No pickle file saved.")

        return dataset

    def __read_pkl_df(self, pkl_path: str) -> pd.DataFrame:
        """Read in a dataset pickle file and check its type and index"""
        self.logger.info(f"Reading data from {pkl_path}...")
        df: pd.DataFrame = pd.read_pickle(pkl_path)
        assert (
            type(df) == pd.DataFrame
        ), f"Pickle file does not contain a pandas DataFrame. Found type {type(df)}"
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
        self, df_cols: Iterable, cuts: OrderedDict[str, Cut], raise_error: bool = False
    ) -> bool:
        """Check whether all necessary cut columns exist in DataFrame columns not including any hard cut"""
        cut_cols = {cut_name + config.cut_label for cut_name in cuts}
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
    def build_dataframe(self, data_path: str, tree_dict: Dict[str, Set[str]]) -> pd.DataFrame:
        match self.dataset_type:
            case "analysistop":
                return self.__build_dataframe_analysistop(data_path=data_path, tree_dict=tree_dict)
            case "dta":
                return self.__build_dataframe_dta(data_path=data_path, tree_dict=tree_dict)
            case _:
                raise ValueError(f"Unknown dataset {self.dataset_type}")

    def __build_dataframe_analysistop(
        self,
        data_path: str,
        tree_dict: Dict[str, Set[str]],
    ) -> pd.DataFrame:
        """
         Builds a dataframe

        :param data_path: path to ROOT datafile(s)
        :param tree_dict: dictionary of tree: variables to extract from Datapath
        :return: output dataframe containing columns corresponding to necessary variables
        """
        self.logger.info(
            f"Building DataFrame from {data_path} ({file_utils.n_files(data_path)} file(s))..."
        )

        # is the default tree a truth tree?
        default_tree_truth = "truth" in self.TTree_name

        # add necessary variables to tree dictionary
        tree_dict = self.__add_necessary_analysistop_variables(tree_dict)

        t1 = time.time()

        # Extract main tree and event weights
        # ---------------------------------------------------------------------------------
        self.logger.info(f"Extracting {tree_dict[self.TTree_name]} from {self.TTree_name} tree...")
        df = to_pandas(
            uproot.concatenate(
                str(data_path) + ":" + self.TTree_name,
                tree_dict[self.TTree_name],
                num_workers=config.n_threads,
                begin_chunk_size=self.chunksize,
            )
        )
        self.logger.debug(f"Extracted {len(df)} events.")

        self.logger.info(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = to_pandas(
            uproot.concatenate(
                str(data_path) + ":sumWeights",
                ["totalEventsWeighted", "dsid"],
                num_workers=config.n_threads,
                begin_chunk_size=self.chunksize,
            )
        )

        self.logger.info(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby("dsid").sum()
        df = pd.merge(df, sumw, left_on="mcChannelNumber", right_on="dsid", sort=False, copy=False)

        df.set_index(["mcChannelNumber", "eventNumber"], inplace=True)
        df.index.names = ["DSID", "eventNumber"]
        self.logger.debug("Set DSID/eventNumber as index")
        # -----------------------------------------------------------------------------------

        # iterate over other TTrees, merge & validate
        # -----------------------------------------------------------------------------------
        for tree in tree_dict:
            if tree == self.TTree_name:
                continue

            self.logger.info(f"Extracting {tree_dict[tree]} from {tree} tree...")
            alt_df = to_pandas(
                uproot.concatenate(
                    str(data_path) + ":" + tree,
                    tree_dict[tree],
                    num_workers=config.n_threads,
                    begin_chunk_size=self.chunksize,
                )
            )
            self.logger.debug(f"Extracted {len(alt_df)} events.")

            alt_df.set_index(["mcChannelNumber", "eventNumber"], inplace=True)
            alt_df.index.names = ["DSID", "eventNumber"]
            self.logger.debug("Set DSID/eventNumber as index")

            if self.validate_missing_events:
                self.logger.info(f"Checking for missing events in tree '{tree}'..")
                tree_is_truth = "truth" in tree

                if tree_is_truth and not default_tree_truth:
                    if n_missing := len(df.index.difference(alt_df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{self.TTree_name}' tree not found in '{tree}' tree"
                        )
                    else:
                        self.logger.debug(
                            f"All events in {self.TTree_name} tree found in {tree} tree"
                        )
                elif default_tree_truth and not tree_is_truth:
                    if n_missing := len(alt_df.index.difference(df.index)):
                        raise Exception(
                            f"Found {n_missing} events in '{tree}' tree not found in '{self.TTree_name}' tree"
                        )
                    else:
                        self.logger.debug(
                            f"All events in {tree} tree found in {self.TTree_name} tree"
                        )
                else:
                    self.logger.info(
                        f"Skipping missing events check. Not truth/reco tree combination"
                    )

            else:
                self.logger.info(f"Skipping missing events check in tree {tree}")

            self.logger.info("Merging with rest of dataframe...")
            df = pd.merge(
                df, alt_df, how="left", left_index=True, right_index=True, sort=False, copy=False
            )
        # -------------------------------------------------------------------------------------

        # validate
        if self.validate_duplicated_events:
            self.logger.info(f"Validating duplicated events...")
            # self.__drop_duplicates(df)
            df = self.__drop_duplicate_index(df)
        else:
            self.logger.info("Skipped duplicted events validation")

        if self.validate_sumofweights:
            # sanity check to make sure totalEventsWeighted really is what it says it is
            # also output DSID metadata
            df_id_sub = df["weight_mc"].groupby(level="DSID").sum()
            for dsid, df_id in df.groupby(level="DSID"):
                unique_totalEventsWeighted = df_id["totalEventsWeighted"].unique()
                if len(unique_totalEventsWeighted) != 1:
                    self.logger.warning(
                        "totalEventsWeighted should only have one value per DSID. "
                        f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} "
                        f"for DSID {dsid}"
                    )

                dsid_weight = df_id_sub[dsid]
                totalEventsWeighted = df_id["totalEventsWeighted"].values[0]
                if dsid_weight != totalEventsWeighted:
                    self.logger.warning(
                        f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                        f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}"
                    )
        else:
            self.logger.info("Skipping sum of weights validation")
        # ---------------------------------------------------------------------------------------

        # WEIGHTS
        # ---------------------------------------------------------------------------------------
        if self._is_truth:
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
            """
            self.logger.info(f"Calculating truth weights for {self.name}...")
            df["truth_weight"] = (
                self.lumi
                * df["weight_mc"]
                * abs(df["weight_mc"])
                / df["totalEventsWeighted"]
                * df["KFactor_weight_truth"]
                * df["weight_pileup"]
            )
            df["base_weight"] = df["weight_mc"] * abs(df["weight_mc"]) / df["totalEventsWeighted"]
            # EVERY event MUST have a truth weight
            self.logger.info(f"Verifying truth weight for {self.name}...")
            if df["truth_weight"].isna().any():
                raise ValueError("NAN values in truth weights!")

        if self._is_reco:
            """
            Calculate total reco event weights

            lumi_data taken from year
            mc_weight = +/-1 * cross_section
            kFactor = kFactor_weight_truth
            pileup = weight_pileup
            truth_weight = kFactor * pileup
            lumi_weight = mc_weight * lumi_data / sum of event weights

            total event weight = lumi_weight * truth_weight * reco_weight
            """
            self.logger.info(f"Calculating reco weight for {self.name}...")
            df["reco_weight"] = (
                self.lumi
                * df["weight_mc"]
                * abs(df["weight_mc"])
                * df["weight_KFactor"]
                * df["weight_pileup"]
                * df["weight_leptonSF"]
                / df["totalEventsWeighted"]
            )
            # every reco event MUST have a reco weight
            self.logger.info(f"Verifying reco weight for {self.name}...")
            if not self._is_truth:
                if df["reco_weight"].isna().any():
                    raise ValueError("NAN values in reco weights!")
            else:
                assert (~df["reco_weight"].isna()).sum() == (
                    ~df["weight_leptonSF"].isna()
                ).sum(), "Different number of events between reco weight and lepton scale factors!"

        # output number of truth and reco events
        if self._is_truth:
            n_truth = df["KFactor_weight_truth"].notna().sum()
            self.logger.info(f"number of truth events: {n_truth}")
        else:
            n_truth = 0
        if self._is_reco:
            n_reco = df["weight_KFactor"].notna().sum()
            self.logger.info(f"number of reco events: {n_reco}")
        else:
            n_reco = 0
        # small check
        assert len(df) == max(n_truth, n_reco), (
            f"Length of DataFrame ({len(df.index)}) "
            f"doesn't match number of truth ({n_truth}) or reco events ({n_reco})"
        )

        if self._is_truth and self._is_reco:
            # make sure KFactor is the same for all reco and truth variables
            pd.testing.assert_series_equal(
                df.loc[pd.notna(df["weight_KFactor"]), "KFactor_weight_truth"],
                df["weight_KFactor"].dropna(),
                check_exact=True,
                check_names=False,
                check_index=False,
            ), "reco and truth KFactors not equal"
        # ensure there is always only one KFactor column and it is named 'weight_KFactor'
        if self._is_truth:
            df.rename(columns={"KFactor_weight_truth": "weight_KFactor"}, inplace=True)

        # CLEANUP
        # ---------------------------------------------------------------------------------------
        self.__rescale_to_gev(df)  # properly scale GeV columns

        # apply hard cuts
        if self.hard_cut:
            self.logger.debug(f"Applying hard cut on dataset: {self.hard_cut}")
            df.query(self.hard_cut, inplace=True)

        self.logger.info("Sorting by DSID...")
        df.sort_index(level="DSID", inplace=True)

        self.logger.info(
            f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}"
        )

        return df

    def __build_dataframe_dta(self, data_path: str, tree_dict: dict[str, Set[str]]) -> pd.DataFrame:
        """Build DataFrame from given files and TTree/branch combinations from DTA output"""

        t1 = time.time()
        paths = [str(file) for file in glob(data_path)]

        # add variables necessary to calculate weights etc
        tree_dict = self.__add_necessary_dta_variables(tree_dict)

        # should always be a set
        if not isinstance(self.TTree_name, set):
            ttrees = {self.TTree_name}
        else:
            ttrees = self.TTree_name

        # check all branches being extracted from each tree are the same, or TChain will throw a fit
        if all(not x == next(iter(tree_dict.values())) for x in tree_dict.values()):
            raise ValueError(
                "Can only extract branches with the same name from multiple trees! "
                "Trying to extract the following branches for each tree:\n\n"
                + "\n".join(tree + ":\n\t" + "\n\t".join(tree_dict[tree]) for tree in tree_dict)
            )
        import_cols = tree_dict[next(iter(tree_dict))]

        self.logger.info(
            f"Initiating RDataFrame from trees {ttrees} in {len(paths)} files and {len(import_cols)} branches.."
        )

        # create c++ map for dataset ID metadatas
        # TODO: what's with this tree??
        dsid_metadata = ROOT_utils.get_dsid_values(data_path, "T_s1thv_NOMINAL")
        ROOT.gInterpreter.Declare(
            f"""
                std::map<int, float> dsid_sumw{{{','.join(f'{{{t.Index}, {t.sumOfWeights}}}' for t in dsid_metadata.itertuples())}}};
                std::map<int, float> dsid_xsec{{{','.join(f'{{{t.Index}, {t.cross_section}}}' for t in dsid_metadata.itertuples())}}};
                std::map<int, float> dsid_pmgf{{{','.join(f'{{{t.Index}, {t.PMG_factor}}}' for t in dsid_metadata.itertuples())}}};
            """
        )

        # create TChain in c++ in order for it not to be garbage collected by python
        ROOT.gInterpreter.Declare(
            f"""
                TChain chain;
                void fill_chain() {{
                    std::vector<std::string> paths = {{\"{'","'.join(paths)}\"}};
                    std::vector<std::string> trees = {{\"{'","'.join(ttrees)}\"}};
                    for (const auto& path : paths) {{
                        for (const auto& tree : trees) {{
                            chain.Add((path + "?#" + tree).c_str());
                        }}
                    }}
                }}
            """
        )
        ROOT.fill_chain()

        # create RDataFrame
        Rdf = ROOT.RDataFrame(ROOT.chain)

        # create weights
        Rdf = (
            Rdf.Define(
                "truth_weight",
                f"(mcWeight * rwCorr * {self.lumi} * prwWeight * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel];",
            )
            .Define(
                "base_weight",
                f"(mcWeight * dsid_xsec[mcChannel]) / dsid_sumw[mcChannel];",
            )
            .Define(
                "reco_weight",
                f"(weight * {self.lumi} * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel];",
            )
        )

        # rescale energy columns to GeV
        for gev_column in [
            column
            for column in import_cols
            if (column in variable_data) and (variable_data[column]["units"] == "GeV")
        ]:
            Rdf = Rdf.Redefine(gev_column, f"{gev_column} / 1000")

        # routine to separate vector branches into separate variables
        badcols = set()  # save old vector column names to avoid extracting them later
        hard_cut_vars = find_variables_in_string(self.hard_cut)  # check for variables in hard cut
        for col_name in import_cols | hard_cut_vars:
            # unravel vector-type columns
            col_type = Rdf.GetColumnType(col_name)
            if "ROOT::VecOps::RVec" in col_type:
                # skip non-numeric vector types
                if col_type == "ROOT::VecOps::RVec<string>":
                    print(f"Skipping string vector column {col_name}")
                    badcols.add(col_name)

                elif "jet" in str(col_name).lower():
                    # create three new columns for each possible jet
                    for i in range(3):
                        Rdf = Rdf.Define(f"{col_name}{i + 1}", f"getVecVal({col_name},{i})")
                    badcols.add(col_name)

                else:
                    Rdf = Rdf.Redefine(col_name, f"getVecVal({col_name},0)")

        # apply any hard cuts
        if self.hard_cut:
            Rdf = Rdf.Filter(self.hard_cut)

        # import needed columns to pandas dataframe
        cols_to_extract = [c for c in import_cols if c not in badcols] + [
            "truth_weight",
            "base_weight",
            "reco_weight",
        ]
        self.logger.debug("All columns and types (post vector column shrinking):")
        for col in cols_to_extract:
            self.logger.debug(f"{col}: {Rdf.GetColumnType(col)}")

        self.logger.info(f"Extracting {len(cols_to_extract)} branch(es) from RDataFrame...")
        df = pd.DataFrame(Rdf.AsNumpy(columns=cols_to_extract))
        self.logger.info(f"Extracted {len(df.index)} events.")

        self.logger.debug("Setting DSID/eventNumber as index...")
        df.set_index(["mcChannel", "eventNumber"], inplace=True)
        df.index.names = ["DSID", "eventNumber"]

        self.logger.info("Sorting by DSID...")
        df.sort_index(level="DSID", inplace=True)

        # validate
        if self.validate_duplicated_events:
            self.logger.info(f"Validating duplicated events in tree {self.TTree_name}...")
            # self.__drop_duplicates(df)
            df = self.__drop_duplicate_index(df)
        else:
            self.logger.info("Skipped duplicated event validation")

        # rename mcWeight
        df.rename(columns={"mcWeight": "weight_mc"}, inplace=True)

        self.logger.info(
            f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}"
        )

        return df

    # ===============================
    # ========== HELPERS ============
    # ===============================
    def __add_necessary_analysistop_variables(
        self, tree_dict: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Add variables necessary to extract for analysistop output"""
        # only add these to 'main tree' to avoid merge issues
        tree_dict[self.TTree_name] |= {"weight_mc", "weight_pileup"}

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
            # "passTruth",
            # "passReco",
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

    def __create_cut_columns(self, df: pd.DataFrame, cuts: OrderedDict[str, Cut]) -> None:
        """Creates boolean columns in dataframe corresponding to each cut"""
        label = config.cut_label  # get cut label from config
        self.logger.info(f"Calculating {len(cuts)} cut columns...")

        for cut in cuts.values():
            try:
                df[cut.name + label] = df.eval(cut.cutstr)
            except ValueError as e:
                raise Exception(f"Error in cut {cut.cutstr}:\n {e}")
