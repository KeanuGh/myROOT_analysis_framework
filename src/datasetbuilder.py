import glob
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import ROOT  # type: ignore
import pandas as pd  # type: ignore
from awkward import to_dataframe  # type: ignore

from src.cutfile import Cut
from src.dataset import RDataset
from src.logger import get_logger
from utils import ROOT_utils, PMG_tool
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
    ttree: str | set[str] = ""
    year: str = "2015+2016"
    lumi: float = 139.0
    label: str = "data"
    lepton: str = "lepton"
    logger: logging.Logger = field(default_factory=get_logger)
    hard_cut: str = ""
    dataset_type: str = "dta"
    isMC: bool = True

    # force_rebuild: bool = False
    # force_recalc_cuts: bool = False
    # force_recalc_vars: bool = False
    # skip_verify_pkl: bool = False
    # validate_missing_events: bool = True
    # validate_duplicated_events: bool = True
    # validate_sumofweights: bool = True

    def __post_init__(self):
        # argument checks
        if self.dataset_type == "dta":
            if not self.ttree:
                raise ValueError(f"No TTree name passed for dataset {self.name}")

            if not isinstance(self.ttree, str):
                self.ttree = set(self.ttree)
            else:
                self.ttree = {self.ttree}
        else:
            NotImplementedError("Only allow DTA outputs from now on")
        
        if self.name == "data":
            self.isMC = False

        if self.year:
            self.lumi = lumi_year[self.year]

        self.dataset_type = self.dataset_type.lower()
        if self.dataset_type not in ("dta", "analysistop"):
            raise ValueError("Known dataset types: 'DTA', 'AnalysisTop'")

    def build(
        self,
        data_path: str,
        cuts: list[Cut] | dict[str, list[Cut]] | None = None,
        extract_vars: set[str] | None = None,
    ) -> RDataset:
        """
        Builds a dataframe from cut inputs.

        :param data_path: Path to ROOT file(s) must be passed if not providing a pickle file
        :param cuts: list of Cut objects if cuts are to be applied but no cutfile is given
        :param extract_vars: set of branches to extract if cutfile is not given
        :return: full built Dataset object
        """

        # PRE-PROCESSING
        # ===============================
        if len(glob.glob(str(data_path))) == 0:
            raise FileExistsError(f"No files found in {str(data_path)}!")
        else:
            self.logger.info(f"{len(glob.glob(str(data_path)))} files found in {str(data_path)}")

        # sanitise input
        if isinstance(cuts, list):
            cuts = {"": cuts}

        # Parsing cuts
        # extract all variables in all cuts in all cut sets passed to builder and which to extract
        all_vars = set()
        for cut_list in cuts.values():
            for cut in cut_list:
                all_vars |= cut.var
        all_vars |= extract_vars
        vars_to_calc = {var for var in all_vars if var in derived_vars}
        # add all variables to all ttrees
        tree_dict = {tree: all_vars for tree in self.ttree}

        # remove calculated vars from tree
        for tree in tree_dict:
            tree_dict[tree] -= vars_to_calc

        # check if vars are contained in label dictionary
        self.__check_axis_labels(tree_dict, vars_to_calc)

        # print debug information
        self.logger.debug("")
        self.logger.debug("DEBUG DATASET OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug(f"Input ROOT file(s):  {data_path}")
        self.logger.debug(f"Hard cut applied: {self.hard_cut}")
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

        # BUILD
        # ===============================
        df = self.__build_dataframe_dta(
            data_path=data_path, tree_dict=tree_dict, vars_to_calc=vars_to_calc
        )

        # BUILD DATASET
        # ===============================
        dataset = RDataset(
            name=self.name,
            df=df,
            cuts=cuts,
            all_vars=all_vars | vars_to_calc,
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
    def __check_axis_labels(self, tree_dict: dict[str, set[str]], calc_vars: set[str]) -> None:
        """Check whether variables exist in"""
        vars = {var for var_set in tree_dict.values() for var in var_set} | calc_vars
        if unexpected_vars := [
            unexpected_var for unexpected_var in vars if unexpected_var not in variable_data
        ]:
            self.logger.warning(
                f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                "Some unexpected behaviour may occur."
            )

    # ===============================
    # ===== DATAFRAME FUNCTIONS =====
    # ==============================

    def __build_dataframe_dta(
        self,
        data_path: str | Path,
        tree_dict: dict[str, set[str]],
        vars_to_calc: set[str] | None = None,
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
        if self.isMC:
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
        Rdf = ROOT_utils.init_rdataframe(self.name, str(data_path), ttrees)

        # check columns exist in dataframe
        if missing_cols := (set(import_cols) - set(Rdf.GetColumnNames())):
            raise ValueError("Missing column(s) in RDataFrame: \n\t" + "\n\t".join(missing_cols))

        # workaround for broken wmunu file (zero'd out dataset IDs)
        workaround_str = "(mcChannel == 0) ? 700446 : mcChannel"

        # create weights
        if self.isMC:
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
            )
        else:
            # for consistency's sake
            Rdf = Rdf.Define("reco_weight", "1").Define("truth_weight", "1")

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
            for derived_var in vars_to_calc:
                self.logger.debug(f"calculating variable: {derived_var}")
                function = derived_vars[derived_var]["cfunc"]
                args = derived_vars[derived_var]["var_args"]
                func_str = f"{function}({','.join(args)})"

                Rdf = Rdf.Define(derived_var, func_str)

        # apply any hard cuts
        if self.hard_cut:
            Rdf = Rdf.Filter(self.hard_cut, "Hard cut: " + self.hard_cut)

        self.logger.info(
            f"time to build dataframe: {time.strftime('%H:%M:%S', time.gmtime(time.time() - t1))}"
        )

        filternames = list(Rdf.GetFilterNames())
        self.logger.debug("Filter names:")
        for name in filternames:
            self.logger.debug(f"\t{name}")

        return Rdf

    # ===============================
    # ========== HELPERS ============
    # ===============================
    def _get_dsid_values(self, path: str | Path, ttree_name: str = "") -> pd.DataFrame:
        """Return DataFrame containing sumw, xs and PMG factor per DSID"""
        # PNG factor is cross-section * kfactor * filter eff.
        files_list = glob.glob(str(path))
        dsid_sumw: dict[int, float] = dict()
        dsid_xs: dict[int, float] = dict()
        dsid_pmg_factor: dict[int, float] = dict()
        dsid_phys_short: dict[int, str] = dict()

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

    def __add_necessary_dta_variables(self, tree_dict: dict[str, set[str]]) -> dict[str, set[str]]:
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
