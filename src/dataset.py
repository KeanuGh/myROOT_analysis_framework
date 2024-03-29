from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike

from src.cutfile import Cut
from src.cutflow import RCutflow
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools
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
    :param df: ROOT RDataframe containing data
    :param selections: dictionary of name: list of cuts to serve as selections/regions for analysis
    :param all_vars: set of all variables to extract from data (taken from branch names in df
    :param profiles: dictionary of profile_name: ProfileOpts to build TProfile objects from dataframe
    :param lumi: Dataset Luminosity scale
    :param label: Dataset label to put on plots
    :param logger: Logger object to print to. Defaults to console output at DEBUG-level
    :param binnings: dict of variable name : list of bin edges to use for given variables
    :param colour: dataset colour for plotting
    :param is_merged: whether dataset is formed from merge of multiple datasets.
                      If set to true `df` will be a dummy emptry RDataframe and histograms are used instead.
    :param is_signal: flag to set if dataset represents signal MC
    :param is_data: flag to set if dataset is NOT MC. This and `is_signal` shouldn't be set at the same time
    :param out_file: file to save histograms to. Will default to "{name].root".
    """

    name: str = ""
    df: pd.DataFrame | ROOT.RDataFrame = None
    selections: dict[str, list[Cut]] = field(default_factory=list)
    all_vars: set[str] = field(default_factory=set)
    profiles: dict[str, ProfileOpts] = field(default_factory=dict)
    lumi: float = 139.0
    label: str = "data"
    logger: logging.Logger = field(default_factory=get_logger)
    binnings: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    colour: str | tuple = field(default_factory=str)
    is_merged: bool = False
    is_signal: bool = False
    is_data: bool = False
    out_file: Path = field(default="")
    histograms: dict[str, ROOT.TH1 | ROOT.TProfile] = field(init=False, default_factory=dict)
    cutflows: dict[str, RCutflow] = field(init=False, default_factory=dict)
    filtered_df: dict[str, ROOT.RDataFrame] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.out_file:
            self.out_file = Path(self.name + ".root")

        # generate filtered dataframe(s) for cuts if an RDataframe and cuts are passed
        if (self.df is not None) and (self.selections is not None):

            def __sanitise_str(string: str) -> str:
                """sanitise latex-like string to stop ROOT from interpreting them as escape sequences"""
                return string.replace("\\", "\\\\")

            # find the minimum shared cuts between each cutflow in order to avoid having to repeat computations
            n_shared_cuts = 0
            if len(self.selections) > 1:
                max_shared_cuts = min(len(selections) for selections in self.selections.values())
                for i in range(max_shared_cuts):
                    first_element = self.selections[list(self.selections.keys())[0]][i]

                    if not all(cuts[i] == first_element for cuts in self.selections.values()):
                        n_shared_cuts = i
                        break

            # Base "Filter" that passes everything for inclusive label
            base_filter = self.df.Filter("true", "Inclusive")
            if n_shared_cuts > 0:
                # take the first n shared cuts from the first list of cuts as the base filter
                shared_cuts = self.selections[list(self.selections.keys())[0]][:n_shared_cuts]
                for cut in shared_cuts:
                    base_filter = base_filter.Filter(cut.cutstr, __sanitise_str(cut.name))

            for cuts_name, cuts in self.selections.items():
                # add all cuts
                # take the base filter for the first cut to make sure all filters share a base dataframe
                try:
                    self.filtered_df[cuts_name] = base_filter.Filter(
                        cuts[n_shared_cuts].cutstr, __sanitise_str(cuts[n_shared_cuts].name)
                    )
                except IndexError as e:
                    raise IndexError(
                        f"Two or more identical cutflows were found in dataset {self.name}. Check passed cuts"
                    ) from e

                for cut in cuts[n_shared_cuts + 1 :]:
                    self.filtered_df[cuts_name] = self.filtered_df[cuts_name].Filter(
                        cut.cutstr, __sanitise_str(cut.name)
                    )

    def __len__(self) -> int:
        if (self.histograms is not None) and ("cutflow" in self.histograms):
            return int(self.histograms["cutflow"].GetBinContent(1))

        else:
            raise ValueError("Must have run cutflow before getting number of events")

    # Variable setting/getting
    # ===================
    @property
    def columns(self) -> list[str]:
        """Get names of columns in dataframe"""
        return list(self.df.GetColumnNames())

    @property
    def variables(self) -> set[str]:
        """Get column names that do not contain a cut label"""
        return set(self.columns)

    def reset_cutflows(self) -> None:
        """(re)set cutflows from cutflow histograms and cuts"""
        if len(self.histograms) == 0:
            raise ValueError("Must generate or load histograms before resetting cutflow")

        for cutset_name in self.selections:
            self.cutflows[cutset_name] = RCutflow(logger=self.logger)
            self.cutflows[cutset_name].import_cutflow(
                self.histograms["cutflow" + (("_" + cutset_name) if cutset_name else "")],
                self.selections[cutset_name],
            )

    def import_dataframes(self, in_file: Path) -> None:
        """
        Import RDataFrames from TTrees into filtered dataframe class members.
        Due to not having run over full dataset, information relating to subsamples and individual filters is lost.
        These can be obtained from the imported cutflow histogram instead.
        """
        for selection in self.selections.keys():
            self.filtered_df[selection] = ROOT.RDataFrame(f"{self.name}/{selection}", str(in_file))
        self.logger.info(f"Selection RDataFrames successfully imported from {in_file}")

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def save_file(
        self, path: str | Path | None = None, ttree: str = "default", tfile_option: str = "recreate"
    ) -> None:
        """Save dataset to ROOT file"""
        if not path:
            path = Path(self.out_file)
        self.logger.info("Saving to ROOT file...")
        with ROOT.TFile(path, tfile_option):
            self.df.Snapshot(ttree, path, tfile_option)
        self.logger.info(f"Saved to {path}")

    def cutflow_printout(self, path: Path | None = None) -> None:
        """Prints cutflow table. Pass path to .tex file if you want to print to latex"""
        if self.cutflows is not None:
            for cutflow_name, cutflow in self.cutflows.items():
                if path is not None:
                    tex_path = path / f"{self.name}_{cutflow_name}_cutflow.tex"
                    cutflow.print(latex_path=tex_path)
                else:
                    self.logger.info("%s: ", cutflow_name)
                    cutflow.print()
        else:
            raise AttributeError("Must have applied cuts to obtain cutflow")

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def gen_cutflows(self) -> None:
        """Generate cutflow, optionally with specific cuts applied"""
        for cuts_name, cuts in self.selections.items():
            self.logger.info(f"Generating cutflow {cuts_name}...")
            cutflow = RCutflow(logger=self.logger)
            cutflow.gen_cutflow(self.filtered_df[cuts_name], cuts)

            self.cutflows[cuts_name] = cutflow

    # ===========================================
    # =========== PLOTING FUNCTION(S) ===========
    # ===========================================
    def plot_hist(
        self,
        var: str,
        bins: list[float] | tuple[int, float, float] | None = None,
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
        cut: bool | str = False,
        name: str = "",
        title: str = "",
        **kwargs,
    ) -> Histogram1D:
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
                          - False (default) for no normalisation
        :param logbins: whether logarithmic binnings
        :param cut: use cuts for histogramming.
                    Pass either name of a cutflow or True for when there is only one cutflow
        :param name: histogram name
        :param title: histgoram title
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histogram
        """
        if cut is True:
            if len(self.filtered_df) > 1:
                raise ValueError("More than one cutflow is present. Must specify.")
            cut = self.selections[list(self.selections.keys())[0]]
        if isinstance(cut, str):
            if cut in self.cutflows.keys():
                histname = var + "_" + cut
            else:
                raise ValueError(f"Unknown cutflow '{cut}'")
        else:
            histname = var

        # if histogram already exists
        if histname in self.histograms:
            hist = Histogram1D(th1=self.histograms[histname])

        else:
            if bins is None:
                raise ValueError(f"Must pass bins if histogram not in dictionary")
            if var not in self.columns:
                raise ValueError(f"No column named {var} in RDataFrame.")

            self.logger.debug(f"Generating {var} histogram in {self.name}...")

            if not ax:
                _, ax = plt.subplots()

            # handle weight
            if weight:
                fill_args = [var, weight]
            else:
                fill_args = [var]

            th1 = ROOT.TH1F(name, title, *plotting_tools.get_TH1_bins(bins, logbins=logbins))

            if cut:
                th1 = self.filtered_df[cut].Fill(th1, fill_args).GetPtr()
            else:
                th1 = self.df.Fill(th1, fill_args).GetPtr()

            # convert to boost
            hist = Histogram1D(th1=th1)

        hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)

        return hist

    # ===========================================
    # ============== HISTOGRAMMING ==============
    # ===========================================

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

    def gen_histograms(self, to_file: bool | str | Path = True) -> dict[str, ROOT.TH1]:
        """Generate histograms for all variables and cuts."""

        output_histogram_variables = self.all_vars
        n_cutflows = len(self.selections)
        if self.logger.getEffectiveLevel() < 20:
            # print debug information with filter names
            for cutflow_name, filtered_df in self.filtered_df.items():
                if n_cutflows > 1:
                    self.logger.debug(f"Cutflow {cutflow_name}:")
                else:
                    self.logger.debug(f"Cuts applied: ")
                filternames = list(filtered_df.GetFilterNames())
                for name in filternames:
                    self.logger.debug(f"\t{name}")

        # to contain smart pointers to TH1s
        th1_histograms: dict[str, ROOT.RResultsPtr] = dict()

        # # histogram weights
        # for weight_str in ["truth_weight", "reco_weight"]:
        #     wgt_th1 = ROOT.TH1F(weight_str, weight_str, 100, -1000, 1000)
        #     th1_histograms[weight_str] = self.df.Fill(wgt_th1, [weight_str])

        def match_weight(var) -> str:
            """match variable to weight"""
            match variable_data[var]:
                case {"tag": VarTag.TRUTH}:
                    return "truth_weight"
                case {"tag": VarTag.RECO}:
                    return "reco_weight"
                case {"tag": VarTag.META}:
                    return ""
                case _:
                    raise ValueError(f"Unknown variable tag for variable {var}")

        # build histograms
        for variable_name in output_histogram_variables:
            # which binning?
            bin_args = self.get_binnings(variable_name)
            weight = match_weight(variable_name)
            fill_cols = [variable_name, weight] if weight else [variable_name]

            th1 = ROOT.TH1F(
                variable_name,
                variable_name,
                *plotting_tools.get_TH1_bins(**bin_args),
            )
            th1_histograms[variable_name] = self.df.Fill(th1, fill_cols)

            for cutflow_name, filtered_df in self.filtered_df.items():
                # which binning?
                bin_args = self.get_binnings(variable_name, cutflow_name)
                cut_hist_name = (
                    variable_name + (("_" + cutflow_name) if cutflow_name else "") + "_cut"
                )
                cut_th1 = ROOT.TH1F(
                    cut_hist_name,
                    variable_name,
                    *plotting_tools.get_TH1_bins(**bin_args),
                )
                th1_histograms[cut_hist_name] = filtered_df.Fill(cut_th1, fill_cols)

        # build profiles
        if self.profiles:
            for profile_name, options in self.profiles.items():
                binning_var = options.x
                bin_args = self.get_binnings(binning_var)
                profile_args = [
                    ROOT.RDF.TProfile1DModel(
                        profile_name + "_PROFILE",
                        profile_name + "_PROFILE",
                        *plotting_tools.get_TH1_bins(**bin_args),
                        options.option,
                    ),
                    options.x,
                    options.y,
                ]
                if options.weight:
                    profile_args.append(options.weight)
                th1_histograms[profile_name + "_PROFILE"] = self.df.Profile1D(*profile_args)

                for cutflow_name, filtered_df in self.filtered_df.items():
                    # which binning?
                    bin_args = self.get_binnings(binning_var, cutflow_name)
                    cut_profile_name = (
                        profile_name
                        + (("_" + cutflow_name) if cutflow_name else "")
                        + "_cut_PROFILE"
                    )
                    profile_args[0] = ROOT.RDF.TProfile1DModel(
                        cut_profile_name,
                        cut_profile_name,
                        *plotting_tools.get_TH1_bins(**bin_args),
                        options.option,
                    )
                    th1_histograms[cut_profile_name] = filtered_df.Profile1D(*profile_args)

        # generate histograms
        t = time.time()
        self.logger.info(f"Producing {len(th1_histograms)} histograms for {self.name}...")
        for hist_name, hist in th1_histograms.items():
            self.histograms[hist_name] = hist.GetValue()
        self.logger.info(
            f"Took {time.time() - t:.3f}s to produce {len(self.histograms)} histograms over {self.df.GetNRuns()} run(s)."
        )

        # generate cutflow
        self.gen_cutflows()
        for cutflow_name, cutflow in self.cutflows.items():
            self.histograms[
                "cutflow" + (("_" + cutflow_name) if cutflow_name else "")
            ] = cutflow.gen_histogram(name=cutflow_name)
        self.cutflow_printout()

        self.logger.info(f"Producted {len(self.histograms)} histograms.")

        if to_file:
            if to_file is True:
                to_file = Path(self.name + "_histograms.root")
            self.export_histograms(filepath=to_file)

        return self.histograms

    def export_histograms(
        self,
        filepath: str | Path | None = None,
        tfile_option: str = "Recreate",
        write_option: str = "Overwrite",
    ) -> None:
        """Save histograms in histogram dictionary to ROOT file containing TH1 objects"""
        if filepath is None:
            filepath = self.name + "_histograms.root"
        with ROOT.TFile(str(filepath), tfile_option) as file:
            for name, hist in self.histograms.items():
                file.WriteObject(
                    hist.TH1 if isinstance(hist, Histogram1D) else hist, name, write_option
                )
        self.logger.info(f"Written {len(self.histograms)} histograms to {filepath}")

    def import_histograms(self, in_file: Path | str) -> None:
        """Import histograms from root file into histogram dictionary"""
        histograms: OrderedDict[str, ROOT.TH1] = OrderedDict()

        with ROOT.TFile(str(in_file), "read") as file:
            for obj in file.GetListOfKeys():
                obj_class = obj.GetClassName()
                if ("TH1" not in obj_class) and ("Profile" not in obj_class):
                    self.logger.warning(f"Non-TH1 object {obj_class} found in file")
                    continue

                th1 = obj.ReadObj()
                histograms[th1.GetName()] = th1

        self.histograms = histograms
        self.logger.info(
            f"Successfully mported {len(self.histograms)} histogram(s) from file {in_file}"
        )

    def get_binnings(self, variable_name: str, selection: str | None = None) -> dict:
        """Get correct binnings for variable"""

        if (
            (selection is not None)
            and (selection in self.binnings)
            and (variable_name in self.binnings[selection])
        ):
            bin_args = {"bins": self.binnings[selection][variable_name]}
        elif variable_name in self.binnings[""]:
            bin_args = {"bins": self.binnings[""][variable_name]}
        else:
            bin_args = self.__match_bin_args(variable_name)

        return bin_args
