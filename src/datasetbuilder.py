import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import ROOT  # type: ignore

from src.cutting import Cut
from src.dataset import Dataset
from src.logger import get_logger
from utils.ROOT_utils import get_object_names_in_file
from utils.file_utils import multi_glob
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data

# total dataset luminosity per year (pb-1)
LUMI_YEAR: Final[dict] = {
    2015: 3219.56,
    2016: 32988.1 + 3219.56,
    2017: 44307.4,
    2018: 5845.01,
}


def find_variables_in_string(s: str) -> set[str]:
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
    :param logger: logger to write to. Defaults to console output at DEBUG-level
    :param hard_cut: string representing a cut that should be applied to dataframe immediately (eg. "TauPt > 20")
                     this can be applied to the entire dataset, or to a subset:
                     In which case you can pass a dictionary of {subset_name: cut} that should match the subset names
                     in the "data_path" dictionary passed to `build`
    :param is_data: flag for whether dataset is data as opposed to MC
    :param is_signal: whether dataset represents signal sample MC
    :param import_missing_columns_as_nan: whether missing columns are imported as NAN as opposed to throwing an exception
    :param do_systematics: whether to extract systematic uncertainties from data
    """

    name: str
    ttree: str | set[str] = field(default_factory=lambda: {"T_s1hv_NOMINAL"})
    year: int = 2017
    lumi: float = 139.0
    label: str = ""
    logger: logging.Logger = field(default_factory=get_logger)
    hard_cut: str | dict[str, str] = ""
    is_data: bool = False
    is_signal: bool = False
    import_missing_columns_as_nan: bool = False
    do_systematics: bool = False
    nominal_tree_name: str = "T_s1thv_NOMINAL"

    _subsamples: set[str] = field(init=False, default_factory=set)
    _vars_to_calc: set[str] = field(init=False, default_factory=set)
    _vars_to_extract: set[str] = field(init=False, default_factory=set)
    __DEFAULT_SUBSAMPLE_NAME: Final[str] = field(init=False, default="default")

    def __post_init__(self):
        if self.year:
            self.lumi = LUMI_YEAR[self.year]

    def build(
        self,
        data_path: Path | list[Path] | dict[str, list[Path] | Path],
        cuts: list[Cut] | dict[str, list[Cut]],
        extract_vars: set[str] | None = None,
    ) -> Dataset:
        """
        Builds a dataframe from cut inputs.

        :param data_path: Path to ROOT file(s). Either a single path, list of paths to be included to
                          or a sample: path dictionary with defined subsamples for dataset
        :param cuts: list of Cut objects if cuts are to be applied but no cutfile is given
        :param extract_vars: set of branches to extract if cutfile is not given
        :return: full built Dataset object
        """

        # PRE-PROCESSING
        # ===============================

        # sanitise inputs
        if isinstance(cuts, list):
            cuts = {"": cuts}
        if not isinstance(self.ttree, list):
            self.ttree = {self.ttree}
        if not isinstance(data_path, dict):
            sample_paths = {self.__DEFAULT_SUBSAMPLE_NAME: data_path}
        else:
            sample_paths = data_path
        if self.do_systematics and self.is_data:
            self.logger.warning("No systematic uncertainties for data!")
            self.do_systematics = False

        self._subsamples = set(sample_paths.keys())

        # handle subsamples and hard cuts
        if not self.hard_cut:
            self.hard_cut = {}
        if not isinstance(self.hard_cut, dict):
            self.hard_cut = {self.__DEFAULT_SUBSAMPLE_NAME: self.hard_cut}
        hard_cut_samples = set(self.hard_cut.keys())
        if not hard_cut_samples.issubset(self._subsamples):
            raise ValueError("Subsamples declared with hard cuts do not exist in data paths!")

        self._subsamples |= hard_cut_samples
        hard_cut_vars = find_variables_in_string(" ".join(self.hard_cut.values()))

        # Parsing cuts
        # extract all variables in all cuts in all selections passed to builder and which to extract
        all_vars: set[str] = set()
        for cut_list in cuts.values():
            for cut in cut_list:
                all_vars |= cut.included_variables
        all_vars |= extract_vars | hard_cut_vars
        self._vars_to_calc |= {var for var in all_vars if var in derived_vars}
        self._vars_to_extract |= {var for var in all_vars if var not in derived_vars}

        # define sets of systematics
        systematic_tree_sets = {self.nominal_tree_name: self.ttree}
        if self.do_systematics:
            # get dictionary of systematics and trees to merge
            systematic_tree_sets |= self.gen_systematics_map(
                multi_glob(sample_paths[list(sample_paths)[0]])[0]  # single file
            )

        # check if vars are contained in label dictionary
        self.__check_axis_labels(self._vars_to_calc | self._vars_to_extract)

        # print  debug information
        self.logger.debug("")
        self.logger.debug("DEBUG DATASET OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug("Input ROOT file(s):  %s", sample_paths)
        self.logger.debug("Hard cut(s) applied: %s", self.hard_cut)
        self.logger.debug("Defined subsamples:\n\t- %s", "\n\t- ".join(self._subsamples))
        self.logger.debug(
            "Defined systematics:\n\t- %s", "\n\t- ".join(systematic_tree_sets.keys())
        )
        self.logger.debug("Defined TTree(s): %s", self.ttree)
        self.logger.debug(
            "Variables to pull from files:\n\t- %s", "\n\t- ".join(self._vars_to_extract)
        )
        self.logger.debug("Calculated Variables:\n\t- %s", "\n\t- ".join(self._vars_to_calc))
        self.logger.debug("----------------------------")
        self.logger.debug("")

        # BUILD
        # ===============================
        dataframes = {
            systematic_name: self.__build_dataset(
                sample_paths=sample_paths,
                ttrees=tree_set,
                extract_variables=self._vars_to_extract,
                calculate_variables=self._vars_to_calc,
            )
            for systematic_name, tree_set in systematic_tree_sets.items()
        }

        # BUILD DATASET
        # ===============================
        dataset = Dataset(
            name=self.name,
            rdataframes=dataframes,
            selections=cuts,
            all_vars=self._vars_to_calc | self._vars_to_extract | {"truth_weight", "reco_weight"},
            logger=self.logger,
            lumi=self.lumi,
            label=self.label,
            is_signal=self.is_signal,
            is_data=self.is_data,
        )

        return dataset

    # ===============================
    # =========== CHECKS ============
    # ===============================
    def __check_axis_labels(self, variables: set[str]) -> None:
        """Check whether variables exist in"""
        if unexpected_vars := [
            unexpected_var for unexpected_var in variables if unexpected_var not in variable_data
        ]:
            self.logger.warning(
                "Variable(s) '%s' not contained in labels dictionary. "
                "Some unexpected behaviour may occur.",
                unexpected_vars,
            )

    # ===============================
    # ===== DATAFRAME FUNCTIONS =====
    # ==============================
    def __build_dataset(
        self,
        sample_paths: dict[str, list[Path] | Path],
        ttrees: set[str],
        extract_variables: set[str],
        calculate_variables: set[str],
    ) -> dict[str, ROOT.RDataFrame]:
        """Build DataFrame from given files and TTree/branch combinations from DTA output"""

        extract_variables.add("passReco")  # we'll need this later

        # make rdataframe
        self.logger.info(f"Initiating RDataFrame from trees: %s..", ttrees)
        spec = ROOT.RDF.Experimental.RDatasetSpec()
        for sample_name, paths in sample_paths.items():
            for tree in ttrees:
                spec.AddSample(ROOT.RDF.Experimental.RSample(sample_name, tree, multi_glob(paths)))
        Rdf = ROOT.RDataFrame(spec)
        ROOT.RDF.Experimental.AddProgressBar(Rdf)

        # define sample ID column
        if not hasattr(ROOT, f"sampleToId_{self.name}"):  # check whether map exists for sample
            sample_hash_map_repr = ",".join(
                f'{{"{sample_name}", {i}}}' for i, sample_name in enumerate(self._subsamples)
            )
            ROOT.gInterpreter.Declare(
                f"std::map<std::string, unsigned int> sampleToId_{self.name}{{{sample_hash_map_repr}}};"
            )
        Rdf = Rdf.DefinePerSample(
            "SampleID", f"sampleToId_{self.name}[rdfsampleinfo_.GetSampleName()]"
        )

        # define weights
        if self.is_data:
            # for consistency's sake
            Rdf = Rdf.Define("reco_weight", "1.0").Define("truth_weight", "1.0")
        else:
            Rdf = Rdf.Define(
                "truth_weight",
                f"(mcWeight * rwCorr * {self.lumi} * prwWeight * dsid_pmgf[mcChannel]) "
                f"/ dsid_sumw[mcChannel]",
            ).Define(
                "reco_weight",
                f"(weight * {self.lumi} * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel]",
            )

        # rescale energy columns to GeV
        for gev_column in [
            column
            for column in extract_variables
            if (column in variable_data) and (variable_data[column]["units"] == "GeV")
        ]:
            Rdf = Rdf.Redefine(gev_column, f"{gev_column} / 1000")

        # calculate derived variables
        if calculate_variables:
            for derived_var in calculate_variables:
                self.logger.debug("defining variable calculation for: %s", derived_var)
                function = derived_vars[derived_var]["cfunc"]
                args = derived_vars[derived_var]["var_args"]

                is_invalid = False
                for arg in args:
                    if arg not in Rdf.GetColumnNames():
                        if self.import_missing_columns_as_nan:
                            self.logger.warning(
                                "NAN column '%s' passed as argument. "
                                "Will fill derived column '%s' with NAN",
                                arg,
                                derived_var,
                            )
                            Rdf = Rdf.Define(derived_var, "NAN")
                            is_invalid = True
                            break
                        raise ValueError(f"Missing argument column for '{derived_var}': '{arg}'")

                if not is_invalid:
                    func_str = f"{function}({','.join(args)})"
                    Rdf = Rdf.Define(derived_var, func_str)

        # check columns exist in dataframe
        if missing_cols := (extract_variables - set(Rdf.GetColumnNames())):
            if self.import_missing_columns_as_nan:
                self.logger.warning("Importing missing columns as NAN: %s", missing_cols)
                for col in missing_cols:
                    Rdf = Rdf.Define(col, "NAN")
            else:
                raise ValueError("Missing column(s) in RDataFrame:\n\t" + "\n\t".join(missing_cols))

        # apply any hard cuts
        for sample_name, hard_cut in self.hard_cut.items():
            Rdf = Rdf.Filter(
                f'SampleID == sampleToId_{self.name}["{sample_name}"] ? {hard_cut} : true',
                f"Hard cut ({sample_name})" + (f": {hard_cut}" if hard_cut else ""),
            )

        # self.logger.debug("Filter names:\n%s", "\n\t".join(list(map(str, Rdf.GetFilterNames()))))

        return Rdf

    def gen_systematics_map(self, sample_file: str | Path) -> dict[str, set[str]]:
        """Generating mapping of {systematic: set of trees containing systematic for different nominals}"""
        # check for systematics branches
        systematic_trees = set()
        all_systematics = set()

        # gather all systematics trees
        nominal_trees = {tree for tree in self.ttree if tree.endswith("_NOMINAL")}
        for tree in nominal_trees:  # only nominal trees have associated systematics
            tree_prefix = tree.rstrip("NOMINAL")

            systematic_trees |= {
                treename
                for treename in get_object_names_in_file(sample_file, "TTree")
                if treename.startswith(tree_prefix) and treename != tree
            }
            all_systematics |= {
                systematic_tree.removeprefix(tree_prefix) for systematic_tree in systematic_trees
            }
        # bring together systematic trees to be merged
        return {
            systematic: {
                systematic_tree
                for systematic_tree in systematic_trees
                if systematic_tree.endswith(systematic)
            }
            for systematic in all_systematics
        }
