from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from tabulate import tabulate

from src.cutting import Cut, Cutflow, FilterNode, FilterTree
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import ROOT_utils
from utils.helper_functions import smart_join, get_base_sys_name
from utils.plotting_tools import ProfileOpts, Hist2dOpts
from utils.variable_names import variable_data, VarTag


@dataclass(slots=True)
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.

    :param name: Name of Dataset
    :param rdataframes: Dictionary of systematic: ROOT RDataframes containing data
    :param histograms: Dictionary of systematic: selection: TH1 histograms
    :param cutflow: Dictionary of systematic: dictionary of cutflows mapped to the selection
    :param filters: Dictionary of systematic: dictionary of filter nodes to act as endpoints for selections within dataframes
    :param selections: Dictionary of name: list of cuts to serve as selections/regions for analysis
    :param all_vars: set of all variables to extract from data (taken from branch names in df)
    :param profiles: dictionary of profile_name: ProfileOpts to build TProfile objects from dataframe
    :param hists_2d: dictionary of hist_name: Hist2dOpts to build 2d histograms
    :param lumi: Dataset Luminosity scale
    :param label: Dataset label to put on plots
    :param logger: Logger object to print to. Defaults to console output at DEBUG-level
    :param binnings: dict of variable name : list of bin edges to use for given variables
    :param colour: dataset colour for plotting
    :param is_signal: flag to set if dataset represents signal MC
    :param is_data: flag to set if dataset is NOT MC. This and `is_signal` shouldn't be set at the same time
    :param out_file: file to save histograms to. Will default to "{name}.root".
    """

    name: str
    rdataframes: dict[str, ROOT.RDataFrame]
    histograms: dict[str, dict[str, dict[str, ROOT.TH1]]] = field(init=False, default_factory=dict)
    cutflows: dict[str, dict[str, Cutflow]] = field(init=False, default_factory=dict)
    filters: dict[str, dict[str, FilterNode]] = field(init=False, default_factory=dict)
    selections: dict[str, list[Cut]] = field(default_factory=list)
    all_vars: set[str] = field(default_factory=set)
    profiles: dict[str, ProfileOpts] = field(default_factory=dict)
    hists_2d: dict[str, Hist2dOpts] = field(default_factory=dict)
    lumi: float = 139.0
    label: str = ""
    logger: logging.Logger = field(default_factory=get_logger)
    binnings: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    colour: str | tuple[int, int, int] = field(default_factory=str)
    is_signal: bool = False
    is_data: bool = False
    out_file: Path = ""
    nominal_name: str = field(init=False, default="")
    do_systematics: bool = False
    do_weights: bool = True
    tes_sys_set: set = field(init=False, default_factory=set)
    eff_sys_set: set = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        if not self.out_file:
            self.out_file = Path(self.name + ".root")

        # generate filtered dataframe(s) for cuts if a Dataframe and cuts are passed
        if (self.rdataframes is not None) and (self.selections is not None):
            self.logger.debug("Generating filters trees for dataframes...")
            for sys_name, rdf in self.rdataframes.items():
                self.filters[sys_name] = self.gen_filters(rdf)

        if self.do_systematics:
            self.init_sys()
        else:
            self.nominal_name = next(s for s in self.rdataframes if "NOMINAL" in s)

    def init_sys(self) -> None:
        """initialise systematics"""
        # get list of TES systematics and name of nominal channel from ttrees
        if self.nominal_name:
            find_nominal = False
        else:
            find_nominal = True

        for sys_name in self.rdataframes.keys():
            # skip nominal(s)
            if ("__1up" not in sys_name) and ("__1down" not in sys_name):
                if self.nominal_name and find_nominal:
                    raise ValueError(
                        f"Nominal '{self.nominal_name}' already found. Got another: '{sys_name}'"
                    )
                self.nominal_name = sys_name
                continue
            self.tes_sys_set.add(sys_name)

        # get EFF systematics from rdataframe
        self.eff_sys_set = {
            str(wgt).removeprefix("weight_")
            for wgt in self.rdataframes[self.nominal_name].GetColumnNames()
            if wgt.startswith("weight_TAUS_TRUEHADTAU_EFF_")
        }
        self.logger.info(f"Initialised dataset: {self.name}")

    # Import/Export
    # ===================
    def export_dataset(
            self,
            filepath: str | Path | None = None,
            selections: list | str | None = None,
            systematics: list | str | None = None,
            overwrite: bool = True,
    ) -> None:
        """
        Save data and histograms to ROOT file. File structure goes as:
        - systematic
            - selection
                - TTree
                - Histograms
        """

        if filepath is None:
            filepath = f"{self.name}.root"

        if not isinstance(selections, list):
            selections = [selections]
        if not isinstance(systematics, list):
            selections = [systematics]

        # Snapshotting options
        snapshot_opts = ROOT.RDF.RSnapshotOptions()
        snapshot_opts.fMode = "UPDATE"
        snapshot_opts.fOverwriteIfExists = True

        # delete file if it exists and we want to overwrite
        if Path(filepath).is_file():
            if overwrite:
                self.logger.debug("File exists at %s. Deleting..", filepath)
                Path(filepath).unlink()
            else:
                raise FileExistsError(
                    f"File exists at {filepath}. Pass `tfile_option=recreate'` to overwrite."
                )

        # save data
        n_sys = len(systematics if systematics else self.filters)
        n_sel = len(selections if selections else self.selections)
        self.logger.info(
            "Saving snapshots of %i selections for %i systematics in dataset '%s'...",
            n_sel,
            n_sys,
            self.name,
        )
        t1 = time.time()
        n_ds = 0
        for sys_name, sys_selections in self.filters.items():
            if sys_name not in systematics:
                continue

            for selection in selections:
                if selection not in selections:
                    continue

                self.filters[sys_name][selection].df = self.filters[sys_name][
                    selection
                ].df.Snapshot(
                    f"{sys_name}/data_{selection}",
                    str(filepath),
                    list(self.all_vars),
                    snapshot_opts,
                )
                n_ds += 1
        self.logger.info("Took %.2gs to snapshot %d RDataframes", time.time() - t1, n_ds)

    def export_histograms(self, filepath: str | Path | None = None) -> None:
        """export histograms to output root file"""

        n_hists = sum(len(hists) for hists in self.histograms.values())
        self.logger.info(f"Saving %d histograms...", n_hists)
        t1 = time.time()
        with ROOT.TFile(str(filepath), "UPDATE") as file:
            for sys_name, selection_dict in self.histograms.items():
                for selection, hist_selections in selection_dict.items():
                    if not selection:
                        selection = "No_Selection"

                    path = f"{sys_name}/{selection}"
                    file.mkdir(path, "", True)  # path SHOULD already exist
                    file.cd(path)

                    for hist_name, hist in hist_selections.items():
                        ROOT.gDirectory.WriteObject(hist, hist_name, "Overwrite")
        self.logger.info("Written %d histograms in %.2gs", n_hists, time.time() - t1)
        self.logger.info("Saved dataset file to %s", filepath)

    def import_dataset(self, in_file: Path) -> None:
        """
        Import dataset from file. RDataFrames are imported from TTrees into filtered dataframe class members.
        Due to not having run over full dataset, information relating to subsamples and individual filters is lost.
        These can be obtained from the imported cutflow histogram instead.
        """

        # import per systematic
        self.logger.info("Importing dataset '%s' from file: %s...", self.name, in_file)
        n_hists = 0
        n_df = 0
        with ROOT.TFile(str(in_file), "read") as tfile:
            # look through systematics directories
            for sys_key in tfile.GetListOfKeys():
                if sys_key.GetClassName() != "TDirectoryFile":
                    self.logger.warning(
                        "Non-TDirectoryFile object (%s) " f"in top-level of file: %s",
                        sys_key.GetClassName(),
                        in_file,
                    )
                    continue

                sys_name = str(sys_key.GetName()).removeprefix("data_")
                self.logger.debug("Found systematic: %s", sys_name)
                sys_dir = tfile.Get(sys_key.GetName())

                # loop over selections
                self.histograms[sys_name] = dict()
                imported_trees = set()
                for sel_key in sys_dir.GetListOfKeys():
                    sel_obj_class = sel_key.GetClassName()

                    # import ttrees
                    if sel_obj_class == "TTree":
                        sel_name = str(sel_key.GetName()).removeprefix("data_")
                        if sel_name in imported_trees:
                            raise ValueError(
                                f"'{sel_name}' tree  already found in subdir"
                                f"'{sys_name}/{sel_key.GetName()}' of file: {in_file}"
                            )

                        # import TTree into RDataFrame
                        self.filters[sys_name][sel_name].df = ROOT.RDataFrame(
                            f"{sys_key.GetName()}/{sel_key.GetName()}", str(in_file)
                        )
                        imported_trees.add(sel_name)
                        n_df += 1
                        continue

                    elif sel_obj_class != "TDirectoryFile":
                        self.logger.warning(
                            "Non-TTree or TDirectoryFile object (%s) "
                            "with name '%s' found "
                            "in '%s' directory of file: %s",
                            sel_obj_class,
                            sel_key.getName(),
                            sys_name,
                            in_file,
                        )
                        continue

                    # import histograms
                    sel_dir = sys_dir.Get(sel_key.GetName())
                    sel_name = sel_key.GetName() if sel_key.GetName() != "No_Selection" else ""
                    self.histograms[sys_name][sel_name] = dict()

                    # Look for objects within selections subdirectory
                    for obj_key in sel_dir.GetListOfKeys():
                        obj_class = obj_key.GetClassName()
                        if (hist := obj_key.ReadObj()).InheritsFrom("TH1"):
                            self.histograms[sys_name][sel_name][obj_key.GetName()] = hist
                            n_hists += 1

                        else:
                            raise TypeError(
                                f"Unknown object of type {obj_class} found in "
                                f"'{sys_name}/{sel_key.GetName()}' subdirectory of file: {in_file}"
                            )

        self.logger.info(
            "%d histograms and %d RDataFrames successfully "
            "imported into dataset '%s' from file: %s",
            n_hists,
            n_df,
            self.name,
            in_file,
        )

    def reset_cutflows(self) -> None:
        """(re)set cutflows from cutflow histograms and cuts"""
        if len(self.histograms) == 0:
            raise ValueError("Must generate or load histograms before resetting cutflow")

        for systematic in self.rdataframes.keys():
            self.cutflows[systematic] = dict()

            for selection in self.selections:
                self.cutflows[systematic][selection] = Cutflow(logger=self.logger)
                self.cutflows[systematic][selection].import_cutflow(
                    self.histograms[systematic][selection]["cutflow"],
                    self.selections[selection],
                )

    def gen_filters(self, df: ROOT.RDataFrame) -> dict[str, FilterNode]:
        """Generate end nodes of filter tree from selections for given dataframe"""
        filter_tree = FilterTree(df)
        filter_tree.generate_tree(self.selections)
        return filter_tree.leaves

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def cutflow_printout(
            self, systematic: str = "", selection: str = "", path: Path | None = None
    ) -> None:
        """Prints cutflow table. Pass path to .tex file if you want to print to latex"""

        if not systematic:
            systematics = self.rdataframes.keys()
        else:
            systematics = [systematic]
        if not selection:
            selections = self.selections.keys()
        else:
            selections = [selection]

        if self.cutflows is None:
            raise AttributeError("Must have applied cuts to obtain cutflow")

        for systematic in systematics:
            for selection in selections:
                if path is not None:
                    path /= Path(f"{self.name}_{systematic}_{selection}_cutflow.tex")
                self.cutflows[systematic][selection].print(latex_path=path)

    def histogram_printout(
            self,
            to_file: Literal["txt", "latex", False] = False,
            to_dir: Path | None = None,
    ) -> None:
        """Printout of histogram metadata"""
        rows = []
        header = ["Systematic", "Selection", "Hist name", "Entries", "Bin sum", "Integral"]

        for sys, _ in self.histograms.items():
            for selection, __ in _.items():
                for name, hist in __.items():
                    rows.append(
                        [
                            sys,
                            selection,
                            name,
                            hist.GetEntries(),
                            hist.Integral(),
                            hist.Integral("width"),
                        ]
                    )

        d = to_dir if to_dir else Path(".")
        match to_file:
            case False:
                self.logger.info(tabulate(rows, headers=header))
            case "txt":
                filepath = d / f"{self.name}_histograms.txt"
                with open(filepath, "w") as f:
                    f.write(tabulate(rows, headers=header))
                    self.logger.info(f"Saved histogram table to {filepath}")
            case "latex":
                filepath = d / f"{self.name}_histograms.tex"
                with open(filepath, "w") as f:
                    f.write(tabulate(rows, headers=header, tablefmt="latex_raw"))
                    self.logger.info(f"Saved LaTeX histogram table to {filepath}")

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def gen_cutflows(self, do_print: bool = False, nominal_only: bool = True) -> None:
        """Generate cutflows for each systematic-selection"""

        systematics = (
            [self.nominal_name] if nominal_only else list(self.eff_sys_set | self.tes_sys_set)
        )
        for systematic in systematics:
            self.cutflows[systematic] = dict()
            for selection in self.selections:
                self.logger.info(
                    "Generating cutflow in dataset '%s' "
                    "for selection '%s' with systematic '%s'...",
                    self.name,
                    selection,
                    systematic,
                )
                cutflow = Cutflow(logger=self.logger)
                cutflow.gen_cutflow(self.filters[systematic][selection])

                self.cutflows[systematic][selection] = cutflow
                self.histograms[systematic][selection]["cutflow"] = cutflow.gen_histogram(
                    name=f"{systematic}_{selection}"
                )

        if do_print:
            self.cutflow_printout()

    def define_selection(self, filter_str: str, name: str, systematic: str, selection: str) -> bool:
        """
        Define filter applied from selection.

        :param filter_str: Filter string (eg `var1 < 10`)
        :param name: name of new selection
        :param systematic: name of systematic
        :param selection: Name of selection to apply to if already existing in dataset.
        :return: filtered RDataFrame if `inplace = False`
        """
        if selection not in self.filters[systematic]:
            raise KeyError(f"No selection '{selection}' in dataset '{self.name}'")
        if name in self.filters[systematic]:
            self.logger.warning(
                "Selection %s already exists in dataset '%s'. Skipping definition.", selection, name
            )
            return False

        # define new selection in dictionaries
        self.filters[systematic][name] = self.filters[systematic][selection].create_child(
            Cut(name=name, cutstr=filter_str)
        )
        self.histograms[systematic][name] = dict()
        self.selections[name] = self.filters[systematic][name].get_cuts()
        self.cutflows[systematic][name] = Cutflow(logger=self.logger)
        self.cutflows[systematic][name].gen_cutflow(self.filters[systematic][name])

        # propagate binnings
        if selection in self.binnings:
            self.binnings[name] = dict()
            for var, binning in self.binnings[selection].items():
                self.binnings[name][var] = binning

        return True

    # ===========================================
    # =========== PLOTING FUNCTION(S) ===========
    # ===========================================
    def plot_hist(
            self,
            var: str,
            bins: list[float] | tuple[int, float, float] | None = None,
            ax: plt.Axes = None,
            yerr: ArrayLike | bool = False,
            normalise: float | bool = False,
            systematic: str = "T_s1hv_NOMINAL",
            selection: str = "",
            histtype: Literal["TH1F", "TH1D", "TH1I", "TH1C", "TH1L", "TH1S"] = "TH1F",
            **kwargs,
    ) -> Histogram1D:
        """
        Generate 1D plots of given variables in dataframe. Returns figure object of list of figure objects.

        :param var: variable name to be plotted. must exist in all datasets
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param ax: axis to plot on. Will create new plot if not given
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param normalise: Normalisation value:
                          - int or float
                          - True for normalisation of unity
                          - False (default) for no normalisation
        :param systematic: name of systematic
        :param selection: use selection for histogramming.
                          Pass either name of a selection or True for when there is only one selection
        :param histtype: if generating the histogram, pass as string TH1 type
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histogram
        """
        if selection is True:
            if len(self.filters) > 1:
                raise ValueError("More than one cutflow is present. Must specify.")
            selection = self.selections[list(self.selections.keys())[0]]

        try:
            hist = self.histograms[systematic][selection][var]
        except KeyError as e:
            # create histogram if it doesn't exist
            if bins is None:
                raise ValueError("Must pass bins if histogram not in dictionary") from e

            th1 = self.gen_th1(
                variable=var, systematic=systematic, selection=selection, histtype=histtype
            )

            # convert to boost
            hist = Histogram1D(th1=th1)

        hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)

        return hist

    # ===========================================
    # ============== HISTOGRAMMING ==============
    # ===========================================
    def get_hist(
            self,
            variable,
            systematic: str = "T_s1hv_NOMINAL",
            selection: str = "",
            kind: Literal["th1", "boost"] = "th1",
    ) -> ROOT.TH1 | Histogram1D:
        """Fetch histogram from internal dictionary"""
        try:
            h = self.histograms[systematic][selection][variable]
        except KeyError:
            raise KeyError(f"No histogram in {self.name} for {systematic} {selection} {variable}")

        if kind == "th1":
            return h
        elif kind == "boost":
            return Histogram1D(th1=h, logger=self.logger)
        raise ValueError(f"Unknown histogram type '{kind}'")

    def _match_weight(self, var_) -> str:
        """match variable to weight"""
        if not self.do_weights:
            return ""

        try:
            match variable_data[var_]:
                case {"tag": VarTag.TRUTH}:
                    return "truth_weight"
                case {"tag": VarTag.RECO}:
                    return "reco_weight"
                case {"tag": VarTag.META}:
                    return ""
                case _:
                    raise ValueError(f"Unknown variable tag for variable {var_}")
        except KeyError:
            self.logger.error(f"No known variable {var_}, no weights for you!")
            return ""

    @staticmethod
    def __match_bin_args(var: str) -> dict:
        """Match arguments for plotting bins from variable name"""
        try:
            var_dict = variable_data[var]
        except KeyError as e:
            raise KeyError(f"No known variable {var}") from e

        match var_dict:
            case {"units": "GeV"}:
                return {"bins": (30, 1, 10000), "logbins": True}
            case {"units": ""}:
                if "phi" in var.lower():
                    return {"bins": (30, -np.pi, np.pi), "logbins": False}
                if "eta" in var.lower():
                    return {"bins": (30, -5, 5), "logbins": False}
                if "delta_z0_sintheta" in var.lower():
                    return {"bins": (30, 0, 2 * np.pi), "logbins": False}
                return {"bins": (30, 0, 30), "logbins": False}
            case _:
                return {"bins": (30, 0, 30), "logbins": False}

    def get_binnings(self, variable_name: str, selection: str | None = None) -> dict:
        """Get correct binnings for variable"""

        # look for first matching key in binning dictionary
        for k in self.binnings.keys():
            if not selection:
                continue
            if re.match(k, selection):
                selection = k
                break

        # select bins
        if (
                selection
                and (selection in self.binnings)
                and (variable_name in self.binnings[selection])
        ):
            bin_args = {"bins": self.binnings[selection][variable_name]}
        elif variable_name in self.binnings[""]:
            bin_args = {"bins": self.binnings[""][variable_name]}
        else:
            bin_args = self.__match_bin_args(variable_name)

        return bin_args

    def define_th1(
            self,
            variable: str,
            name: str = "",
            title: str = "",
            histtype: Literal["TH1F", "TH1D", "TH1I", "TH1C", "TH1L", "TH1S"] = "TH1F",
    ) -> ROOT.TH1F:
        """Define 1D histogram from variable with correct binnings"""
        allowed_histtypes = ["TH1F", "TH1D", "TH1I", "TH1C", "TH1L", "TH1S"]
        if histtype.upper() not in allowed_histtypes:
            raise ValueError(
                f"Unknown histogram type: {histtype}. Allowed histogram types: {allowed_histtypes}."
            )
        if variable not in self.all_vars:
            raise ValueError(f"No known variable {variable} in dataset {self.name}")

        return ROOT.__getattr__(histtype)(
            name if name else variable,
            title if title else name if name else variable,
            *ROOT_utils.get_TH1_bin_args(**self.get_binnings(variable)),
        )

    def define_th2(
            self,
            x: str,
            y: str,
            name: str = "",
            title: str = "",
            histtype: Literal["TH2F", "TH2D", "TH2I", "TH2C", "TH2L", "TH2S"] = "TH2F",
    ) -> ROOT.TH1F:
        """Define 2D histogram from variables with correct binnings"""
        allowed_histtypes = ["TH2F", "TH2D", "TH2I", "TH2C", "TH2L", "TH2S"]
        if histtype.upper() not in allowed_histtypes:
            raise ValueError(
                f"Unknown histogram type: {histtype}. Allowed histogram types: {allowed_histtypes}."
            )
        for v in (x, y):
            if v not in self.all_vars:
                raise ValueError(f"No known variable {v} in dataset {self.name}")

        return ROOT.__getattr__(histtype)(
            name if name else f"{x}_{y}",
            title if title else name if name else f"{x}_{y}",
            *ROOT_utils.get_TH1_bin_args(**self.get_binnings(x)),
            *ROOT_utils.get_TH1_bin_args(**self.get_binnings(y)),
        )

    def define_profile(
            self,
            profile_opts: ProfileOpts,
            profile_name: str,
            systematic: str = "T_s1hv_NOMINAL",
            selection: str = "",
    ) -> tuple[ROOT.TProfile1DModel, str, str, str] | tuple[ROOT.TProfile1DModel, str, str]:
        """Return arguments for profile creation from profile options"""
        bin_args = self.get_binnings(profile_opts.x, selection)
        profile_name = smart_join([systematic, selection, profile_name, "PROFILE"])
        profile_model = ROOT.RDF.TProfile1DModel(
            profile_name,
            profile_name,
            *ROOT_utils.get_TH1_bin_args(**bin_args),
            profile_opts.option,
        )
        if profile_opts.weight:
            return profile_model, profile_opts.x, profile_opts.y, profile_opts.weight
        return profile_model, profile_opts.x, profile_opts.y

    def gen_th1(
            self,
            variable: str,
            systematic: str = "T_s1thv_NOMINAL",
            selection: str = "",
            histtype: Literal["TH1F", "TH1D", "TH1I", "TH1C", "TH1L", "TH1S"] = "TH1F",
    ) -> ROOT.TH1:
        """Return TH1 histogram from selection for variable. Binning taken from internal binnings dictionary"""
        weight = self._match_weight(variable)
        fill_cols = [variable, weight] if weight else [variable]

        self.logger.debug(
            "Generating %s histogram in %s " f"for systematic '%s' and selection '%s'...",
            variable,
            self.name,
            systematic,
            selection,
        )
        histname = smart_join([self.name, systematic, selection, variable])
        th1 = self.define_th1(variable=variable, name=histname, title=histname, histtype=histtype)

        if selection:
            h_ptr = self.filters[systematic][selection].df.Fill(th1, fill_cols)
        else:
            h_ptr = self.rdataframes[systematic].Fill(th1, fill_cols)

        return h_ptr.GetValue()

    def gen_all_histograms(self, do_prints: bool = True) -> None:
        """Generate histograms for all variables and cuts."""

        def count(d):
            """Count number of subvalues in nested dict"""
            return sum([count(v) if isinstance(v, dict) else 1 for v in d.values()])

        histograms_dict = dict()  # to store outputs
        output_histogram_variables = self.all_vars
        self.logger.info(f"Defining histograms for dataset {self.name}...")
        if self.logger.getEffectiveLevel() < 20 and do_prints:
            # variable MUST be in scope! (why is ROOT like this)
            verbosity = ROOT.Experimental.RLogScopedVerbosity(
                ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo
            )

            # # print debug information with filter names (from nominal)
            # for selection, dataframe in self.filters[list(self.filters.keys())[0]].items():
            #     if n_cutflows > 1:
            #         self.logger.debug(f"Cutflow {selection}:")
            #     else:
            #         self.logger.debug(f"Cuts applied: ")
            #     filternames = list(dataframe.df.GetFilterNames())
            #     for name in filternames:
            #         self.logger.debug(f"\t%s", name)

        # build histograms
        for sys_name, root_sys_df in self.rdataframes.items():
            # to contain smart pointers to TH1s (instantiate with a "NoSel" for blank selection)
            th1_ptr_map: dict[str, dict[str, ROOT.RResultsPtr]] = dict()

            # # histogram weights
            # for weight_str in ["truth_weight", "reco_weight"]:
            #     wgt_th1 = ROOT.TH1F(weight_str, weight_str, 100, -1000, 1000)
            #     th1_ptr_map["NoSel"] = rdf.Fill(wgt_th1, [weight_str])

            # gather all selections (including no selection)
            selections = [""] + list(self.filters[sys_name].keys())
            rdfs = [root_sys_df] + [node.df for node in self.filters[sys_name].values()]
            for selection, sel_df in zip(selections, rdfs):
                th1_ptr_map[selection] = dict()

                # Define Histograms
                # =======================================================
                for variable_name in output_histogram_variables:
                    weight = self._match_weight(variable_name)
                    fill_cols = [variable_name, weight] if weight else [variable_name]

                    # define histogram
                    hist_name = smart_join([sys_name, selection, variable_name])
                    th1 = self.define_th1(variable_name, hist_name, hist_name)
                    th1_ptr_map[selection][variable_name] = sel_df.Fill(th1, fill_cols)

                    # do systematic weights for reco variables in nominal tree
                    if (
                            (self.eff_sys_set or self.tes_sys_set)
                            and (sys_name == self.nominal_name)
                            and (weight == "reco_weight")
                    ):
                        for sys_wgt in [
                            str(wgt)
                            for wgt in self.rdataframes[self.nominal_name].GetColumnNames()
                            if wgt.startswith("weight_TAUS_TRUEHADTAU_EFF_")
                        ]:
                            eff_sys_name = sys_wgt.removeprefix("weight_")
                            hist_name = smart_join([eff_sys_name, selection, variable_name])
                            th1 = self.define_th1(variable_name, hist_name, hist_name)
                            th1_ptr_map[selection][f"{variable_name}|{eff_sys_name}"] = sel_df.Fill(
                                th1, [variable_name, sys_wgt]
                            )

                # only want regular 1D histograms for systemaics
                if sys_name != self.nominal_name:
                    continue

                # Define profiles
                # =======================================================
                for profile_name, profile_opts in self.profiles.items():
                    if profile_name in output_histogram_variables:
                        self.logger.error(
                            "Histogram for '%s' already exists! Skipping profile creation",
                            profile_name,
                        )
                        continue

                    profile_args = self.define_profile(
                        profile_opts=profile_opts,
                        profile_name=profile_name,
                        systematic=sys_name,
                        selection=selection,
                    )
                    th1_ptr_map[selection][profile_name] = root_sys_df.Profile1D(*profile_args)

                # Define 2D histograms
                # =======================================================
                for hist2d_name, hist2d_opts in self.hists_2d.items():
                    if hist2d_name in output_histogram_variables:
                        self.logger.error(
                            "Histogram for '%s' already exists! Skipping 2D histogram creation",
                            hist2d_name,
                        )
                        continue

                    fill_cols = (
                        [hist2d_opts.x, hist2d_opts.y, hist2d_opts.weight]
                        if hist2d_opts.weight
                        else [hist2d_opts.x, hist2d_opts.y]
                    )

                    # define histogram
                    hist_name = smart_join(
                        [sys_name, selection, f"{hist2d_opts.x}_{hist2d_opts.y}"]
                    )
                    th2 = self.define_th2(hist2d_opts.x, hist2d_opts.y, hist_name, hist_name)
                    th1_ptr_map[selection][f"{hist2d_opts.x}_{hist2d_opts.y}"] = sel_df.Fill(
                        th2, fill_cols
                    )

            # generate histograms
            t = time.time()
            self.logger.info(
                "Producing %s histograms for %s in %s...", count(th1_ptr_map), sys_name, self.name
            )

            histograms_dict[sys_name] = dict()
            for selection, hist_ptrs in th1_ptr_map.items():
                histograms_dict[sys_name][selection] = dict()
                for hist_name, hist_ptr in hist_ptrs.items():
                    # put EFF. sys in their own systematics key
                    if "TAUS_TRUEHADTAU_EFF_" in hist_name:
                        var_name, eff_sys_name = hist_name.split("|")
                        if eff_sys_name not in histograms_dict:
                            histograms_dict[eff_sys_name] = dict()
                        if selection not in histograms_dict[eff_sys_name]:
                            histograms_dict[eff_sys_name][selection] = dict()
                        histograms_dict[eff_sys_name][selection][var_name] = hist_ptr.GetValue()

                    else:
                        histograms_dict[sys_name][selection][hist_name] = hist_ptr.GetValue()

            self.logger.info(
                "Took %.3fs to produce %d histograms over %d run(s).",
                time.time() - t,
                count(histograms_dict),
                root_sys_df.GetNRuns(),
            )

        self.logger.info("Produced %d histograms.", count(histograms_dict))
        self.histograms = histograms_dict

        # gen uncertainties
        if self.do_systematics:
            self.calculate_systematic_uncertainties()

    # ===========================================
    # ============== UNCERTAINTIES ==============
    # ===========================================
    def get_systematic_uncertainty(
            self,
            val: str,
            selection: str = "",
            symmetric: bool = True,
    ) -> tuple[np.typing.NDArray[1] | Literal[0], np.typing.NDArray[1] | Literal[0]]:
        """Get systematic uncertainty for single variable in dataset. Returns 0s if not found"""
        nominal = self.histograms[self.nominal_name][selection][val]

        try:
            if symmetric:
                uncert_halved = self.histograms[self.nominal_name][selection][
                    f"{val}_tot_uncert"
                ].Clone()
                uncert_halved.Scale(0.5)
                sys_up = uncert_halved
                sys_down = uncert_halved
            else:
                sys_up = ROOT_utils.th1_abs(
                    self.histograms[self.nominal_name][selection][f"{val}_1up_uncert"] - nominal
                )
                sys_down = ROOT_utils.th1_abs(
                    self.histograms[self.nominal_name][selection][f"{val}_1down_uncert"] - nominal
                )
        except KeyError:
            self.logger.debug(
                "No systematic for histogram: v:%s, ds: %s, sel: %s", val, self.name, selection
            )
            return 0, 0

        return ROOT_utils.get_th1_bin_values(sys_down), ROOT_utils.get_th1_bin_values(sys_up)

    def calculate_systematic_uncertainties(self) -> None:
        """
        Calculate systematic uncertainties from variables in systematic trees

        Calculates:
            - diff between sys and nominal for each sys: {var}_{sys}_{1up/1down}_diff
            - diff between up and down for each sys: {var}_{sys}_tot_uncert
            - percentage sys uncertainty for each sys: {var}_{sys}_pct_uncert
        """
        self.logger.info("Calculating systematic uncertainties...")

        def handle_sys(var: str, sel: str, sys: str, sys_hist: ROOT.TH1) -> None:
            """
            calculate systematics:
                - "{var}_1(up/down)_uncert": the sum of every systematic uncertainty for var
                - "{var}_1(up/down)_{sys}_diff": absolute difference between nominal and systematic
                - "{var}_1(up/down)_{sys}_pct": percentage difference between nominal and systematic
            """
            if (var not in variable_data) or (variable_data[var]["tag"] != VarTag.RECO):
                return  # only systematics for reco variables

            nonlocal sys_pairs
            base_sys = get_base_sys_name(sys)

            if isinstance(sys_hist, ROOT.TProfile):
                return  # don't want profiles
            if variable_data[var]["tag"] == VarTag.TRUTH:
                return  # no truth systematics

            if var not in sys_pairs[base_sys][sel]:
                sys_pairs[base_sys][sel][var] = dict()

            # save individual histograms for diff later + sum for total
            if sys.endswith("_1up"):
                sys_pairs[base_sys][sel][var]["1up"] = sys_hist
                full_sys_name = f"{var}_1up_uncert"
            elif sys.endswith("_1down"):
                sys_pairs[base_sys][sel][var]["1down"] = sys_hist
                full_sys_name = f"{var}_1down_uncert"
            else:
                raise ValueError(f"Systematic '{sys}' isn't 1up or 1down?")

            # save diff between nominal and systematic
            nominal = self.histograms[self.nominal_name][sel][var]
            diff_hist = sys_hist - nominal
            self.histograms[self.nominal_name][sel][f"{var}_{sys}_diff"] = diff_hist

            # save pct between nominal and systematic
            pct_hist = 100 * diff_hist / nominal
            self.histograms[self.nominal_name][sel][f"{var}_{sys}_pct"] = pct_hist

            # sum for full systematics
            if full_sys_name not in self.histograms[self.nominal_name][sel][var]:
                self.histograms[self.nominal_name][sel][full_sys_name] = diff_hist
            else:
                self.histograms[self.nominal_name][sel][full_sys_name] += diff_hist

        # hold sys: selection: variable: (1up, 1down)
        sys_pairs: dict[str, dict[str, dict[str, dict[str, ROOT.TH1]]]] = dict()

        # SYSTEMATICS LOOP FOR INDIVIDUAL SYSTEMATICS
        # ================================================================
        for sys_name in self.eff_sys_set | self.tes_sys_set:
            sel_dict = self.histograms[sys_name]

            # start collecting systemtics pairs
            base_sys_name = get_base_sys_name(sys_name)
            if base_sys_name not in sys_pairs:
                sys_pairs[base_sys_name] = dict()

            # SELECTION LOOP
            for selection, selection_dict in sel_dict.items():
                if selection not in sys_pairs[base_sys_name]:
                    sys_pairs[base_sys_name][selection] = dict()

                # VARIABLE LOOP
                for variable, syst_histogram in selection_dict.items():
                    handle_sys(variable, selection, sys_name, syst_histogram)

        # FULL UNCERTAINTY
        # ==================================
        # final loop for full uncertainty
        for sys_name, _ in sys_pairs.items():
            for selection, __ in _.items():
                for variable, pair in __.items():
                    nominal_hist = self.histograms[self.nominal_name][selection][variable]
                    uncert_up = ROOT_utils.th1_abs(pair["1up"] - nominal_hist)
                    undert_down = ROOT_utils.th1_abs(pair["1down"] - nominal_hist)
                    tot_uncert = undert_down + uncert_up
                    pct_uncert = (tot_uncert / nominal_hist) * 100

                    # total uncert for systematic
                    self.histograms[self.nominal_name][selection][
                        f"{variable}_{sys_name}_tot_uncert"
                    ] = tot_uncert

                    # percentage uncert
                    self.histograms[self.nominal_name][selection][
                        f"{variable}_{sys_name}_pct_uncert"
                    ] = pct_uncert

                    # overall summed uncertainty
                    if (
                            f"{variable}_tot_uncert"
                            not in self.histograms[self.nominal_name][selection]
                    ):
                        self.histograms[self.nominal_name][selection][
                            f"{variable}_tot_uncert"
                        ] = tot_uncert
                    else:
                        self.histograms[self.nominal_name][selection][
                            f"{variable}_tot_uncert"
                        ] += tot_uncert

                    # overall percent uncertainty
                    if (
                            f"{variable}_pct_uncert"
                            not in self.histograms[self.nominal_name][selection]
                    ):
                        self.histograms[self.nominal_name][selection][
                            f"{variable}_pct_uncert"
                        ] = pct_uncert
                    else:
                        self.histograms[self.nominal_name][selection][
                            f"{variable}_pct_uncert"
                        ] += pct_uncert

        self.logger.info("Done.")
