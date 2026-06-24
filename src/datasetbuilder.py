import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import ROOT

from src.cutting import Cut
from src.dataset import Dataset
from src.logger import get_logger
from utils.helper_functions import multi_glob
from utils.ROOT_utils import get_object_names_in_file
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data

# total dataset luminosity per year (pb-1)
LUMI_YEAR: Final[dict] = {
    2015: 3219.56,
    2016: 32988.1 + 3219.56,
    2017: 44307.4,
    2018: 58450.1,
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
    skip_sys: set[str] = field(default_factory=set)
    nominal_tree_name: str = "T_s1thv_NOMINAL"
    nominal_lookup_columns_for_systematics: set[str] = field(default_factory=set)

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
        selections: list[Cut] | dict[str, list[Cut]] | None = None,
        extract_vars: set[str] | None = None,
        systematics_for_selection: set[str] | None = None,
    ) -> Dataset:
        """
        Builds a dataframe from cut inputs.

        :param data_path: Path to ROOT file(s). Either a single path, list of paths to be included to
                          or a sample: path dictionary with defined subsamples for dataset
        :param selections: list of Cut objects if cuts are to be applied but no cutfile is given
        :param extract_vars: set of branches to extract if cutfile is not given
        :return: full built Dataset object
        """

        # PRE-PROCESSING
        # ===============================
        # sanitise inputs
        # Selections are keyed by output selection name.
        if selections is None:
            selections = {}
        elif isinstance(selections, list):
            selections = {"": selections}

        # Trees are always handled as a set, even for a single nominal tree.
        if isinstance(self.ttree, str):
            ttrees = {self.ttree}
        else:
            ttrees = set(self.ttree)

        # Extracted variables are accumulated while building the dataframe.
        if extract_vars is None:
            extract_vars = set()
        else:
            extract_vars = set(extract_vars)

        if self.do_systematics and self.is_data:
            self.logger.warning("No systematic uncertainties for data!")
            self.do_systematics = False

        # Input paths are keyed by subsample name.
        if isinstance(data_path, dict):
            sample_paths = data_path
        else:
            sample_paths = {self.__DEFAULT_SUBSAMPLE_NAME: data_path}
        self._subsamples = set(sample_paths.keys())

        # handle subsamples and hard cuts
        # Hard cuts may apply to all data, or to named subsamples only.
        if isinstance(self.hard_cut, dict):
            hard_cuts = self.hard_cut
        elif self.hard_cut:
            hard_cuts = {self.__DEFAULT_SUBSAMPLE_NAME: self.hard_cut}
        else:
            hard_cuts = {}
        hard_cut_samples = set(hard_cuts.keys())
        if not hard_cut_samples.issubset(self._subsamples):
            raise ValueError("Subsamples declared with hard cuts do not exist in data paths!")

        self._subsamples |= hard_cut_samples
        hard_cut_vars = find_variables_in_string(" ".join(hard_cuts.values()))

        # Parsing cuts
        # extract all variables in all cuts in all selections passed to builder and which to extract
        all_vars: set[str] = set()
        for selection in selections.values():
            for cut in selection:
                all_vars |= cut.included_variables
        all_vars |= extract_vars | hard_cut_vars
        self._vars_to_calc |= {var for var in all_vars if var in derived_vars}
        self._vars_to_extract |= {var for var in all_vars if var not in derived_vars}

        # add dependencies
        for derived_var in self._vars_to_calc:
            self._vars_to_extract |= set(derived_vars[derived_var]["var_args"])

        # define sets of systematics
        systematic_tree_sets = {self.nominal_tree_name: ttrees}
        if self.do_systematics:
            # get dictionary of systematics and trees to merge
            systematic_tree_sets |= self.gen_systematics_map(
                multi_glob(sample_paths[list(sample_paths)[0]])[0],  # single file
                ttrees,
            )

        # check if vars are contained in label dictionary
        self.__check_axis_labels(self._vars_to_calc | self._vars_to_extract)

        # print  debug information
        self.logger.debug("")
        self.logger.debug("DEBUG DATASET OPTIONS: ")
        self.logger.debug("----------------------------")
        self.logger.debug("Input ROOT file(s):  %s", sample_paths)
        self.logger.debug("Hard cut(s) applied: %s", hard_cuts)
        self.logger.debug("Defined subsamples:\n\t- %s", "\n\t- ".join(self._subsamples))
        self.logger.debug("Luminosity %s", self.lumi)
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
        def build_dataframe(
            tree_set: set[str],
            hard_cuts: dict[str, str],
            hard_cut_masks: dict[str, str] | None = None,
            nominal_lookup_maps: dict[str, str] | None = None,
        ) -> ROOT.RDataFrame:
            return self.__build_dataset(
                sample_paths=sample_paths,
                ttrees=tree_set,
                extract_variables=self._vars_to_extract,
                calculate_variables=self._vars_to_calc,
                hard_cuts=hard_cuts,
                hard_cut_masks=hard_cut_masks,
                nominal_lookup_maps=nominal_lookup_maps,
            )

        # Nominal-only: builds one dataframe and apply hard cuts.
        if not self.do_systematics:
            dataframes = {
                self.nominal_tree_name: build_dataframe(ttrees, hard_cuts),
            }

        # Systematic builds may need nominal event masks before shifted trees are built.
        else:
            needs_systematic_hard_cut_masks = bool(hard_cuts) and not self.is_data

            # Build the nominal graph once; omit hard cuts when it must seed masks.
            nominal_rdf = build_dataframe(
                ttrees,
                {} if needs_systematic_hard_cut_masks else hard_cuts,
            )

            # Systematic trees reuse nominal event membership for truth-level hard cuts.
            hard_cut_masks: dict[str, str] = {}
            if needs_systematic_hard_cut_masks:
                self.logger.info("Building nominal hard-cut event masks for systematic trees...")
                hard_cut_masks = self.__build_nominal_hard_cut_masks(
                    nominal_rdf=nominal_rdf,
                    hard_cuts=hard_cuts,
                )

            # Store the nominal dataframe and one dataframe per systematic variation.
            nominal_dataframe = (
                self.__apply_hard_cuts(nominal_rdf, hard_cuts)
                if needs_systematic_hard_cut_masks
                else nominal_rdf
            )
            nominal_lookup_maps = self.__build_nominal_lookup_maps(
                nominal_dataframe,
                self.nominal_lookup_columns_for_systematics,
            )
            dataframes = {
                self.nominal_tree_name: nominal_dataframe,
            }
            dataframes |= {
                systematic_name: build_dataframe(
                    tree_set,
                    hard_cuts,
                    hard_cut_masks=hard_cut_masks,
                    nominal_lookup_maps=nominal_lookup_maps,
                )
                for systematic_name, tree_set in systematic_tree_sets.items()
                if systematic_name != self.nominal_tree_name
            }

        # BUILD DATASET
        # ===============================
        dataset = Dataset(
            name=self.name,
            rdataframes=dataframes,
            selections=selections,
            all_vars=self._vars_to_calc | self._vars_to_extract | {"truth_weight", "reco_weight"},
            logger=self.logger,
            lumi=self.lumi,
            label=self.label,
            is_signal=self.is_signal,
            is_data=self.is_data,
            do_systematics=self.do_systematics,
            skip_sys=self.skip_sys,
            systematics_for_selection=set(systematics_for_selection or set()),
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
    def __declare_sample_id_map(self) -> None:
        """Declare the C++ map used to identify RDatasetSpec subsamples."""
        if hasattr(ROOT, f"sampleToId_{self.name}"):
            return

        sample_hash_map_repr = ",".join(
            f'{{"{sample_name}", {i}}}' for i, sample_name in enumerate(sorted(self._subsamples))
        )
        ROOT.gInterpreter.Declare(
            f"std::map<std::string, unsigned int> sampleToId_{self.name}{{{sample_hash_map_repr}}};"
        )

    @staticmethod
    def __declare_event_mask_functions() -> None:
        """Declare C++ helpers used to match nominal events to systematic trees."""
        if hasattr(ROOT, "makeMcEventKey"):
            return

        ROOT.gInterpreter.Declare(
            """
            #include <cstdint>
            #include <unordered_set>
            #include <vector>

            ULong64_t makeMcEventKey(UInt_t mcChannel, UInt_t eventNumber) {
                return (static_cast<ULong64_t>(mcChannel) << 32) |
                       static_cast<ULong64_t>(eventNumber);
            }

            void fillEventMask(
                std::unordered_set<ULong64_t>& mask,
                const std::vector<ULong64_t>& keys
            ) {
                mask.reserve(mask.size() + keys.size());
                for (const auto key : keys) {
                    mask.insert(key);
                }
            }
            """
        )

    @staticmethod
    def __declare_nominal_lookup_functions() -> None:
        """Declare C++ helpers used to attach nominal values to shifted trees."""

        ROOT.gInterpreter.Declare(
            """
            #ifndef NOMINAL_LOOKUP_HELPERS_DECLARED
            #define NOMINAL_LOOKUP_HELPERS_DECLARED
            #include <limits>
            #include <stdexcept>
            #include <unordered_map>
            #include <vector>

            void fillDoubleLookup(
                std::unordered_map<ULong64_t, double>& lookup,
                const std::vector<ULong64_t>& keys,
                const std::vector<double>& values
            ) {
                if (keys.size() != values.size()) {
                    throw std::runtime_error("Lookup key/value size mismatch");
                }
                lookup.reserve(lookup.size() + keys.size());
                for (std::size_t idx = 0; idx < keys.size(); ++idx) {
                    lookup[keys[idx]] = values[idx];
                }
            }

            double lookupNominalDouble(
                const std::unordered_map<ULong64_t, double>& lookup,
                UInt_t mcChannel,
                UInt_t eventNumber
            ) {
                const auto key = makeMcEventKey(mcChannel, eventNumber);
                const auto found = lookup.find(key);
                if (found == lookup.end()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return found->second;
            }
            #endif
            """
        )

    def __hard_cut_mask_name(self, sample_name: str) -> str:
        safe_name = re.sub(r"\W+", "_", self.name)
        safe_sample = re.sub(r"\W+", "_", sample_name)
        return f"hardCutMask_{safe_name}_{safe_sample}"

    def __nominal_lookup_name(self, column: str) -> str:
        safe_name = re.sub(r"\W+", "_", self.name)
        safe_column = re.sub(r"\W+", "_", column)
        return f"nominalLookup_{safe_name}_{safe_column}"

    def __build_nominal_hard_cut_masks(
        self,
        nominal_rdf: ROOT.RDataFrame,
        hard_cuts: dict[str, str],
    ) -> dict[str, str]:
        """
        Build event-membership masks from the nominal tree for hard-cut subsamples.

        Systematic trees do not reliably carry all truth branches with the same type/content as the
        nominal tree. The mask lets those trees select the same event set without re-evaluating the
        truth-level hard-cut expression on the shifted tree.
        """
        if not hard_cuts:
            return {}

        self.__declare_sample_id_map()
        self.__declare_event_mask_functions()

        mask_expressions: dict[str, str] = {}
        for sample_name, hard_cut in hard_cuts.items():
            mask_name = self.__hard_cut_mask_name(sample_name)
            if not hasattr(ROOT, mask_name):
                ROOT.gInterpreter.Declare(
                    f"#include <unordered_set>\nstd::unordered_set<ULong64_t> {mask_name};"
                )
            else:
                getattr(ROOT, mask_name).clear()

            self.logger.info(
                "Scanning nominal tree to build hard-cut event mask for subsample '%s'...",
                sample_name,
            )
            keys = (
                nominal_rdf.Filter(
                    f'(SampleID == sampleToId_{self.name}["{sample_name}"]) && ({hard_cut})',
                    f"Nominal hard-cut mask ({sample_name})",
                )
                .Define("HardCutEventKey", "makeMcEventKey(mcChannel, eventNumber)")
                .Take["ULong64_t"]("HardCutEventKey")
                .GetValue()
            )
            mask = getattr(ROOT, mask_name)
            self.logger.info(
                "Filling hard-cut event mask for subsample '%s' with %d event keys...",
                sample_name,
                len(keys),
            )
            ROOT.fillEventMask(mask, keys)

            self.logger.info(
                "Built nominal hard-cut event mask for subsample '%s' with %d events.",
                sample_name,
                len(keys),
            )
            mask_expressions[sample_name] = (
                f"{mask_name}.count(makeMcEventKey(mcChannel, eventNumber)) > 0"
            )

        return mask_expressions

    def __build_nominal_lookup_maps(
        self,
        nominal_rdf: ROOT.RDataFrame,
        columns: set[str],
    ) -> dict[str, str]:
        """Build nominal value maps that shifted systematic trees can query by event key."""

        if not columns:
            return {}

        self.__declare_event_mask_functions()
        self.__declare_nominal_lookup_functions()

        available_columns = {str(column) for column in nominal_rdf.GetColumnNames()}
        lookup_maps: dict[str, str] = {}
        for column in sorted(columns):
            if column not in available_columns:
                raise ValueError(
                    f"Cannot build nominal lookup for '{column}' because it is not "
                    "available in the nominal dataframe."
                )

            lookup_name = self.__nominal_lookup_name(column)
            if not hasattr(ROOT, lookup_name):
                ROOT.gInterpreter.Declare(
                    "#include <unordered_map>\n"
                    f"std::unordered_map<ULong64_t, double> {lookup_name};"
                )
            else:
                getattr(ROOT, lookup_name).clear()

            self.logger.info("Building nominal event lookup for column '%s'...", column)
            lookup_rdf = (
                nominal_rdf.Define("NominalLookupEventKey", "makeMcEventKey(mcChannel, eventNumber)")
                .Define(f"{column}_lookup_value", f"static_cast<double>({column})")
                .Filter(f"{column}_lookup_value == {column}_lookup_value")
            )
            keys = lookup_rdf.Take["ULong64_t"]("NominalLookupEventKey").GetValue()
            values = lookup_rdf.Take["double"](f"{column}_lookup_value").GetValue()

            unique_keys = len(set(int(key) for key in keys))
            if unique_keys != len(keys):
                self.logger.warning(
                    "Nominal lookup for '%s' has %d duplicate event keys out of %d entries; "
                    "later values will overwrite earlier values.",
                    column,
                    len(keys) - unique_keys,
                    len(keys),
                )
            if not keys:
                raise ValueError(f"Nominal lookup for '{column}' is empty.")

            ROOT.fillDoubleLookup(getattr(ROOT, lookup_name), keys, values)
            self.logger.info(
                "Built nominal event lookup for column '%s' with %d entries.",
                column,
                len(keys),
            )
            lookup_maps[column] = lookup_name

        return lookup_maps

    def __apply_hard_cuts(
        self,
        rdf: ROOT.RDataFrame,
        hard_cuts: dict[str, str],
        hard_cut_masks: dict[str, str] | None = None,
    ) -> ROOT.RDataFrame:
        hard_cut_masks = hard_cut_masks or {}
        for sample_name, hard_cut in hard_cuts.items():
            cut_expr = hard_cut_masks.get(sample_name, hard_cut)
            rdf = rdf.Filter(
                f'(SampleID != sampleToId_{self.name}["{sample_name}"]) || ({cut_expr})',
                f"Hard cut ({sample_name})" + (f": {hard_cut}" if hard_cut else ""),
            )
        return rdf

    def __build_dataset(
        self,
        sample_paths: dict[str, list[Path] | Path],
        ttrees: set[str],
        extract_variables: set[str],
        calculate_variables: set[str],
        hard_cuts: dict[str, str],
        hard_cut_masks: dict[str, str] | None = None,
        nominal_lookup_maps: dict[str, str] | None = None,
    ) -> ROOT.RDataFrame:
        """Build DataFrame from given files and TTree/branch combinations from DTA output"""

        extract_variables = set(extract_variables)
        calculate_variables = set(calculate_variables)
        nominal_lookup_maps = nominal_lookup_maps or {}
        extract_variables.add("passReco")  # we'll need this later
        is_nominal = all(["nominal" in tree.lower() for tree in ttrees])

        # remove truth variables if passed tree(s) is systematic
        if not is_nominal:
            extract_variables = {v for v in extract_variables if variable_data[v]["tag"] != "truth"}

        # make rdataframe
        self.logger.info("Initiating RDataFrame from trees: %s..", ttrees)
        spec = ROOT.RDF.Experimental.RDatasetSpec()
        for sample_name, paths in sample_paths.items():
            for tree in ttrees:
                spec.AddSample(ROOT.RDF.Experimental.RSample(sample_name, tree, multi_glob(paths)))
        Rdf = ROOT.RDataFrame(spec)
        ROOT.RDF.Experimental.AddProgressBar(Rdf)

        # define sample ID column
        self.__declare_sample_id_map()
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
                f"(mcWeight * {self.lumi} * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel]",
            ).Define(
                "reco_weight",
                f"(weight * {self.lumi} * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel]",
            )

        if (not is_nominal) and nominal_lookup_maps:
            existing_columns = {str(column) for column in Rdf.GetColumnNames()}
            for column, lookup_name in nominal_lookup_maps.items():
                if column in existing_columns:
                    continue
                Rdf = Rdf.Define(
                    column,
                    f"lookupNominalDouble({lookup_name}, mcChannel, eventNumber)",
                )
                existing_columns.add(column)

        # systematics weights
        if is_nominal and self.do_systematics:
            # factor systematic weights with reco weight
            for sys_wgt in [
                str(w)
                for w in Rdf.GetColumnNames()
                if str(w).startswith("weight_TAUS_TRUEHADTAU_EFF_")
                and not any(re.match(p, str(w)) for p in self.skip_sys)
            ]:
                Rdf = Rdf.Redefine(sys_wgt, f"{sys_wgt} * reco_weight / selectionSF")

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
                if derived_var in {str(column) for column in Rdf.GetColumnNames()}:
                    continue
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
                    extract_variables.add(arg)

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
        if hard_cuts:
            Rdf = self.__apply_hard_cuts(Rdf, hard_cuts, hard_cut_masks)

        # self.logger.debug("Filter names:\n%s", "\n\t".join(list(map(str, Rdf.GetFilterNames()))))

        return Rdf

    def gen_systematics_map(
        self,
        sample_file: str | Path,
        nominal_trees: set[str],
    ) -> dict[str, set[str]]:
        """Generating mapping of {systematic: set of trees containing systematic for different nominals}"""
        # check for systematics branches
        systematic_trees = set()
        all_systematics = set()

        # gather all systematics trees
        nominal_trees = {tree for tree in nominal_trees if tree.endswith("_NOMINAL")}
        for tree in nominal_trees:  # only nominal trees have associated systematics
            tree_prefix = tree.removesuffix("NOMINAL")

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
            if not any(re.match(p, systematic) for p in self.skip_sys)
        }
