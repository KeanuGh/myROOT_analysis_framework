from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike

from src.cutting import Cut, Cutflow, FilterNode, FilterTree
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools
from utils.file_utils import smart_join
from utils.variable_names import variable_data, VarTag


@dataclass(slots=True)
class ProfileOpts:
    """
    Options for building ROOT profile from RDataFrame columns

    :param x: x-axis column name.
    :param y: y-axis column name.
    :param weight: name of column to apply as weight.
    :param option: option paramter to pass to `TProfile1DModel()`
        (see https://root.cern.ch/doc/master/classTProfile.html#a1ff9340284c73ce8762ab6e7dc0e6725)
    """

    x: str
    y: str
    weight: str = ""
    option: str = ""


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
    lumi: float = 139.0
    label: str = ""
    logger: logging.Logger = field(default_factory=get_logger)
    binnings: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    colour: str | tuple[int, int, int] = field(default_factory=str)
    is_signal: bool = False
    is_data: bool = False
    out_file: Path = ""

    def __post_init__(self) -> None:
        if not self.out_file:
            self.out_file = Path(self.name + ".root")

        # generate filtered dataframe(s) for cuts if a Dataframe and cuts are passed
        if (self.rdataframes is not None) and (self.selections is not None):
            self.logger.debug("Generating filters trees for dataframes...")
            for sys_name, rdf in self.rdataframes.items():
                self.filters[sys_name] = self.gen_filters(rdf)

    # Import/Export
    # ===================
    def export_dataset(self, filepath: str | Path | None = None, overwrite: bool = True) -> None:
        """
        Save data and histograms to ROOT file. File structure goes as:
        - systematic
            - selection
                - TTree
                - Histograms
        """

        if filepath is None:
            filepath = f"{self.name}.root"

        # Snapshotting options
        snapshot_opts = ROOT.RDF.RSnapshotOptions()
        snapshot_opts.fMode = "UPDATE"
        snapshot_opts.fOverwriteIfExists = True

        # delete file if it exists and we want to overwrite
        if Path(filepath).is_file():
            if overwrite:
                self.logger.debug(f"File exists at {filepath}. Deleting..")
                Path(filepath).unlink()
            else:
                raise FileExistsError(
                    f"File exists at {filepath}. Pass `tfile_option=recreate'` to overwrite."
                )

        # save data
        self.logger.info(
            f"Saving snapshots of {len(self.filters)} systematics in dataset '{self.name}'..."
        )
        t1 = time.time()
        n_ds = 0
        for sys_name, sys_selections in self.filters.items():
            for selection in sys_selections:
                self.filters[sys_name][selection].df = self.filters[sys_name][
                    selection
                ].df.Snapshot(
                    f"{sys_name}/data_{selection}",
                    str(filepath),
                    list(self.all_vars),
                    snapshot_opts,
                )
                n_ds += 1
        self.logger.info(f"Took {time.time() - t1:.2g}s to snapshot {n_ds} RDataframes")

        # save histograms
        n_hists = sum(len(hists) for hists in self.histograms.values())
        self.logger.info(f"Saving {n_hists} histograms...")
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
        self.logger.info(f"Written {n_hists} histograms in {time.time() - t1:.2g}s")
        self.logger.info(f"Saved dataset file to {filepath}")

    def import_dataset(self, in_file: Path) -> None:
        """
        Import dataset from file. RDataFrames are imported from TTrees into filtered dataframe class members.
        Due to not having run over full dataset, information relating to subsamples and individual filters is lost.
        These can be obtained from the imported cutflow histogram instead.
        """

        # import per systematic
        self.logger.info(f"Importing dataset '{self.name}' from file: {in_file}...")
        n_hists = 0
        n_df = 0
        with ROOT.TFile(str(in_file), "read") as tfile:
            # look through systematics directories
            for sys_key in tfile.GetListOfKeys():
                if sys_key.GetClassName() != "TDirectoryFile":
                    self.logger.warning(
                        f"Non-TDirectoryFile object ({sys_key.GetClassName()}) "
                        f"in top-level of file: {in_file}"
                    )
                    continue

                sys_name = str(sys_key.GetName()).removeprefix("data_")
                self.logger.debug(f"Found systematic: {sys_name}")
                sys_dir = tfile.Get(sys_key.GetName())

                # loop over selections
                self.histograms[sys_name] = dict()
                imported_trees = set()
                for sel_key in sys_dir.GetListOfKeys():
                    sel_obj_class = sel_key.GetClassName()

                    # import ttrees
                    if sel_obj_class == "TTree":
                        sys_tree_name = str(sel_key.GetName()).removeprefix("data_")
                        if sys_tree_name in imported_trees:
                            raise ValueError(
                                f"'{sys_tree_name}' tree  already found in subdir"
                                f"'{sys_name}/{sel_key.GetName()}' of file: {in_file}"
                            )

                        # import TTree into RDataFrame
                        self.filters[sys_name][sys_tree_name].df = ROOT.RDataFrame(
                            f"{sys_key.GetName()}/{sel_key.GetName()}", str(in_file)
                        )
                        imported_trees.add(sys_tree_name)
                        n_df += 1
                        continue

                    elif sel_obj_class != "TDirectoryFile":
                        self.logger.warning(
                            f"Non-TTree or TDirectoryFile object ({sel_obj_class}) "
                            f"with name '{sel_key.getName()}' found "
                            f"in '{sel_name}' directory of file: {in_file}"
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
            f"{n_hists} histograms and {n_df} RDataFrames successfully "
            f"imported into dataset '{self.name}' from file: {in_file}"
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

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def gen_cutflows(self, do_print: bool = True) -> None:
        """Generate cutflows for each systematic-selection"""

        for systematic in self.rdataframes.keys():
            self.cutflows[systematic] = dict()
            for selection, cuts in self.selections.items():
                self.logger.info(
                    f"Generating cutflow in dataset '{self.name}' "
                    f"for selection '{selection}' with systematic '{systematic}'..."
                )
                cutflow = Cutflow(logger=self.logger)
                cutflow.gen_cutflow(self.filters[systematic][selection])

                self.cutflows[systematic][selection] = cutflow
                self.histograms[systematic][selection]["cutflow"] = cutflow.gen_histogram(
                    name=f"{systematic}_{selection}"
                )

        if do_print:
            self.cutflow_printout()

    def define_selection(self, filter_str: str, name: str, systematic: str, selection: str) -> None:
        """
        Define filter applied from selection.

        :param filter_str: Filter string (eg `var1 < 10`)
        :param name: name of new selection
        :param systematic: name of systematic
        :param selection: Name of selection to apply to if already existing in dataset.
        :return: filtered RDataFrame if `inplace = False`
        """
        if selection not in self.filters:
            raise KeyError(f"No selection '{selection}' in dataset '{self.name}'")

        self.filters[systematic][name] = self.filters[systematic][selection].create_child(
            Cut(name=name, cutstr=filter_str)
        )

        # propagate binnings
        if selection in self.binnings:
            self.binnings[name] = dict()
            for var, binning in self.binnings[selection].items():
                self.binnings[name][var] = binning

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
        histtype: str = "TH1F",
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
        except KeyError:
            # create histogram if it doesn't exist
            if bins is None:
                raise ValueError(f"Must pass bins if histogram not in dictionary")

            th1 = self.gen_histogram(
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
        else:
            raise ValueError(f"Unknown histogram type '{kind}'")

    @staticmethod
    def _match_weight(var_) -> str:
        """match variable to weight"""
        match variable_data[var_]:
            case {"tag": VarTag.TRUTH}:
                return "truth_weight"
            case {"tag": VarTag.RECO}:
                return "reco_weight"
            case {"tag": VarTag.META}:
                return ""
            case _:
                raise ValueError(f"Unknown variable tag for variable {var_}")

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
                elif "eta" in var.lower():
                    return {"bins": (30, -5, 5), "logbins": False}
                elif "delta_z0_sintheta" in var.lower():
                    return {"bins": (30, 0, 2 * np.pi), "logbins": False}
                else:
                    return {"bins": (30, 0, 30), "logbins": False}
            case _:
                return {"bins": (30, 0, 30), "logbins": False}

    def get_binnings(self, variable_name: str, selection: str | None = None) -> dict:
        """Get correct binnings for variable"""

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
        self, variable: str, name: str = "", title: str = "", histtype="TH1F"
    ) -> ROOT.TH1F:
        """Define histogram from variable with correct binnings"""
        allowed_histtypes = [
            "TH1F",
            "TH1D",
            "TH1I",
            "TH1C",
            "TH1L",
            "TH1S",
        ]
        if histtype.upper() not in allowed_histtypes:
            raise ValueError(
                f"Unknown histogram type: {histtype}. Allowed histogram types: {allowed_histtypes}."
            )
        if variable not in self.all_vars:
            raise ValueError(f"No known variable {variable} in dataset {self.name}")

        bin_args = self.get_binnings(variable)
        return ROOT.__getattr__(histtype)(
            name if name else variable,
            title if title else name if name else variable,
            *plotting_tools.get_TH1_bins(**bin_args),
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
            *plotting_tools.get_TH1_bins(**bin_args),
            profile_opts.option,
        )
        if profile_opts.weight:
            return profile_model, profile_opts.x, profile_opts.y, profile_opts.weight
        else:
            return profile_model, profile_opts.x, profile_opts.y

    def gen_histogram(
        self,
        variable: str,
        systematic: str = "T_s1thv_NOMINAL",
        selection: str = "",
        histtype: str = "TH1F",
    ) -> ROOT.TH1:
        """Return TH1 histogram from selection for variable. Binning taken from internal binnings dictionary"""
        weight = self._match_weight(variable)
        fill_cols = [variable, weight] if weight else [variable]

        self.logger.debug(
            f"Generating {variable} histogram in {self.name} "
            f"for systematic '{systematic}' and selection '{selection}'..."
        )
        histname = smart_join([self.name, systematic, selection, variable])
        th1 = self.define_th1(variable=variable, name=histname, title=histname, histtype=histtype)

        if selection:
            h_ptr = self.filters[systematic][selection].df.Fill(th1, fill_cols)
        else:
            h_ptr = self.df.Fill(th1, fill_cols)

        return h_ptr.GetValue()

    def gen_all_histograms(self, do_prints: bool = True) -> None:
        """Generate histograms for all variables and cuts."""

        histograms_dict = dict()  # to store outputs
        output_histogram_variables = self.all_vars
        n_cutflows = len(self.selections)
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
        for sys_name, rdf in self.rdataframes.items():
            # to contain smart pointers to TH1s (instantiate with a "NoSel" for blank selection)
            th1_ptr_map: dict[str, dict[str, ROOT.RResultsPtr]] = dict()

            # # histogram weights
            # for weight_str in ["truth_weight", "reco_weight"]:
            #     wgt_th1 = ROOT.TH1F(weight_str, weight_str, 100, -1000, 1000)
            #     th1_ptr_map["NoSel"] = rdf.Fill(wgt_th1, [weight_str])

            # get all systematics (including none!)
            selections = [""] + list(self.filters[sys_name].keys())
            for selection in selections:
                th1_ptr_map[selection] = dict()

                # Define histograms from listed variables
                for variable_name in output_histogram_variables:
                    weight = self._match_weight(variable_name)
                    fill_cols = [variable_name, weight] if weight else [variable_name]

                    # define histogram
                    hist_name = smart_join([sys_name, selection, variable_name])
                    th1 = self.define_th1(variable_name, hist_name, hist_name)
                    th1_ptr_map[selection][variable_name] = rdf.Fill(th1, fill_cols)

                # Define profiles
                for profile_name, profile_opts in self.profiles.items():
                    if profile_name in output_histogram_variables:
                        self.logger.error(
                            f"Histogram for {profile_name} already exists! Skipping profile creation"
                        )
                        continue

                    profile_args = self.define_profile(
                        profile_opts=profile_opts,
                        profile_name=profile_name,
                        systematic=sys_name,
                        selection=selection,
                    )
                    th1_ptr_map[selection][profile_name] = rdf.Profile1D(*profile_args)

            # generate histograms
            t = time.time()
            self.logger.info(
                f"Producing {len(th1_ptr_map)} histograms for {sys_name} in {self.name}..."
            )
            histograms_dict[sys_name] = dict()
            for selection, hist_ptrs in th1_ptr_map.items():
                histograms_dict[sys_name][selection] = dict()
                for name, hist_ptr in hist_ptrs.items():
                    histograms_dict[sys_name][selection][name] = hist_ptr.GetValue()
            self.logger.info(
                f"Took {time.time() - t:.3f}s to produce {len(histograms_dict)} histograms over {rdf.GetNRuns()} run(s)."
            )

        self.logger.info(f"Producted {len(histograms_dict)} histograms.")

        self.histograms = histograms_dict
