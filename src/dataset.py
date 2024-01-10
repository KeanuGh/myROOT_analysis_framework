from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Final, Set, Dict

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike
from tabulate import tabulate  # type: ignore

from src.cutfile import Cut
from src.cutflow import RCutflow
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools, ROOT_utils
from utils.variable_names import variable_data, VarTag

CUT_PREFIX: Final = "PASS_"


@dataclass
class Dataset(ABC):
    name: str = ""
    df: pd.DataFrame | ROOT.RDataFrame = None
    cuts: List[Cut] = field(default_factory=list)
    all_vars: Set[str] = field(default_factory=set)
    cutflow: RCutflow = None
    lumi: float = 139.0
    label: str = "data"
    logger: logging.Logger = field(default_factory=get_logger)
    lepton: str = "lepton"
    file: Path = field(init=False)
    histograms: OrderedDict[str, ROOT.TH1] = field(init=False, default_factory=OrderedDict)
    binnings: Dict[str, List[float]] = field(default_factory=dict)

    @abstractmethod
    def __len__(self):
        ...

    def __repr__(self):
        return f"{type(self)}(name={self.name}, label={self.label}, file={self.file}, ...)"

    # Variable setting/getting
    # ===================
    @property
    @abstractmethod
    def columns(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def variables(self) -> Set[str]:
        ...

    def set_filepath(self, filepath: Path | str) -> None:
        self.file = Path(filepath)

    @property
    def is_truth(self) -> bool:
        """Does dataset contain truth data?"""
        return bool(VarTag.TRUTH in self.__var_tags)

    @property
    def is_reco(self) -> bool:
        """Does dataset contain reco data?"""
        return bool(VarTag.RECO in self.__var_tags)

    @property
    def meta_vars(self) -> list[str]:
        """Get meta variables in dataset"""
        return self.__get_var_tag(VarTag.META)

    @property
    def truth_vars(self) -> list[str]:
        """Get truth variables in dataset"""
        return self.__get_var_tag(VarTag.TRUTH)

    @property
    def reco_vars(self) -> list[str]:
        """Get reconstructed variables in dataset"""
        return self.__get_var_tag(VarTag.RECO)

    @property
    def __var_tags(self) -> list[str]:
        """Get tags for all variables"""
        return [variable_data[col]["tag"] for col in self.columns if col in variable_data]

    def __get_var_tag(self, tag: VarTag | str) -> list[str]:
        """Get all variables in dataset with given tag"""
        return [
            col
            for col in self.df.columns
            if col in variable_data and variable_data[col]["tag"] == VarTag(tag)
        ]

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @abstractmethod
    def save_file(self, path: str | Path | None = None) -> None:
        ...

    def cutflow_printout(self, latex_path: Path | None = None) -> None:
        """Prints cutflow table. Pass path to .tex file if you want to print to latex"""
        if self.cutflow is not None:
            self.cutflow.print(latex_path=latex_path)
        else:
            raise AttributeError("Must have applied cuts to obtain cutflow")

    @abstractmethod
    def dsid_metadata_printout(self, truth: bool = True, reco: bool = False) -> None:
        ...

    # ===============================
    # ========== CUTTING ============
    # ===============================
    @abstractmethod
    def apply_cuts(
        self,
        labels: bool | str | List[str] = True,
        reco: bool = False,
        truth: bool = False,
    ) -> None:
        ...

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
    def match_bin_args(self, var: str) -> dict:
        """Match arguments for plotting bins from variable name"""
        try:
            var_dict = variable_data[var]
        except KeyError:
            raise KeyError(f"No known variable {var}")

        if var in self.binnings:
            return {"bins": self.binnings[var]}

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

    @abstractmethod
    def plot_hist(
        self,
        var: str,
        bins: List[float] | Tuple[int, float, float] | None = None,
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
        cut: bool = True,
        name: str = "",
        title: str = "",
        **kwargs,
    ) -> Histogram1D:
        ...

    @abstractmethod
    def gen_histograms(
        self, cut: bool = True, to_file: bool | str | Path = True
    ) -> OrderedDict[str, ROOT.TH1]:
        ...

    def export_histograms(
        self,
        filepath: str | Path | None = None,
        tfile_option: str = "Recreate",
        write_option: str = "Overwrite",
    ) -> None:
        """Save histograms in histogram dictionary to ROOT file containing TH1 objects"""
        if filepath is None:
            filepath = self.name + "_histograms.root"
        with ROOT_utils.ROOT_TFile_mgr(filepath, tfile_option) as file:
            for name, hist in self.histograms.items():
                file.WriteObject(
                    hist.TH1 if isinstance(hist, Histogram1D) else hist, name, write_option
                )
        self.logger.info(f"Written {len(self.histograms)} histograms to {filepath}")

    def import_histograms(self, in_file: Path | str, inplace=True) -> None | Dict[str, ROOT.TH1]:
        """Import histograms from root file into histogram dictionary"""
        histograms: OrderedDict[str, ROOT.TH1] = OrderedDict()

        with ROOT_utils.ROOT_TFile_mgr(in_file, "read") as file:
            for obj in file.GetListOfKeys():
                if "TH1" not in (obj_class := obj.GetClassName()):
                    self.logger.warning(f"Non-TH1 Object {obj_class} found in file")
                    continue

                th1 = obj.ReadObj()
                histograms[th1.GetName()] = th1

        if inplace:
            self.histograms = histograms
            return None
        else:
            return histograms


@dataclass(slots=True)
class RDataset(Dataset):
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.

    :param name: Name of Dataset
    :param df: pandas DataFrame containing data
    :param cuts: list of cuts
    :param all_vars: set of all variables
    :param filtered_df: view of dataframe with cuts applied. Must be generated with gen_cutflow() or apply_cuts() first
    :param lumi: Dataset Luminosity
    :param label: Label to put on plots
    :param logger: Logger object to print to. Defaults to console output at DEBUG-level
    :param lepton: Name of charged DY lepton channel in dataset (if applicable)
    :param plot_dir: directory to save plots to. Defaults to current directory
    :param binnings: dict of variable name : list of bin edges to use for given variables
    """

    cutflow: RCutflow = field(init=False)
    filtered_df: ROOT.RDataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.file = Path(self.name + ".root")

        # generate filtered dataframe for cuts
        if self.df is not None:
            self.filtered_df: ROOT.RDataFrame = self.df.Filter("true", "Inclusive")
        if self.cuts:
            for cut in self.cuts:
                sanitised_name = cut.name.replace(
                    "\\", "\\\\"
                )  # stop root interpreting escape for latex
                self.filtered_df = self.filtered_df.Filter(cut.cutstr, sanitised_name)

    def __len__(self) -> int:
        if (self.histograms is not None) and ("cutflow" in self.histograms):
            return int(self.histograms["cutflow"].GetBinContent(1))

        else:
            raise ValueError("Must have run cutflow before getting len")

    def __getattr__(self, item):
        if self.df is not None:
            return getattr(self.df, item)
        else:
            return None

    # Variable setting/getting
    # ===================
    @property
    def columns(self) -> List[str]:
        return list(self.df.GetColumnNames())

    @property
    def variables(self) -> Set[str]:
        """Column names that do not contain a cut label"""
        return set(self.columns)

    def reset_cutflow(self) -> None:
        """(re)set cutflow from cutflow histogram and cuts"""
        if len(self.histograms) == 0:
            raise ValueError("Must generate or load histograms before resetting cutflow")

        self.cutflow = RCutflow(logger=self.logger)
        self.cutflow.import_cutflow(self.histograms["cutflow"], self.cuts)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def save_file(
        self, path: str | Path | None = None, ttree: str = "default", tfile_option: str = "recreate"
    ) -> None:
        """Saves pickle"""
        if not path:
            path = Path(self.file)
        self.logger.info(f"Saving to ROOT file...")
        with ROOT_utils.ROOT_TFile_mgr(path, tfile_option):
            self.df.Snapshot(ttree, path, tfile_option)
        self.logger.info(f"Saved to {path}")

    def dsid_metadata_printout(self, truth: bool = True, reco: bool = False) -> None:
        # TODO
        raise NotImplementedError()

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def apply_cuts(
        self,
        reco: bool = False,
        truth: bool = False,
        inplace: bool = True,
    ) -> ROOT.RDataFrame | None:
        """
        Apply specific cut(s) to DataFrame.

        :param reco: cut on reco cuts
        :param truth: cut on truth cuts
        :param inplace: If True, applies cuts in place to dataframe in self.
                        If False returns DataFrame object
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied.
                 Raises ValueError if cuts do not exist
        """
        # handle which cuts to apply depending on what is passed as 'labels'
        cuts_to_apply: List[Cut] = []
        if truth:
            self.logger.debug(f"Applying truth cuts to {self.name}...")
            cuts_to_apply += [cut for cut in self.cuts if cut.is_reco is False]
        if reco:
            self.logger.debug(f"Applying reco cuts to {self.name}...")
            cuts_to_apply += [cut for cut in self.cuts if cut.is_reco is True]

        # apply spefic cuts
        return self.gen_cutflow(cuts_to_apply, inplace=inplace)

    def gen_cutflow(
        self, cuts: List[Cut] | None = None, inplace: bool = True
    ) -> None | Tuple[ROOT.RDataFrame, RCutflow]:
        """Generate cutflow, optionally with specific cuts applied"""
        if cuts is None:
            # use all cuts already in dataset
            cuts = self.cuts
            filtered_df = self.filtered_df
        else:
            # create new filters for dataset
            filtered_df = self.df.Filter("true", "Inclusive")
            for cut in cuts:
                filtered_df = filtered_df.Filter(cut.cutstr, cut.name)

        cutflow = RCutflow(logger=self.logger)
        cutflow.gen_cutflow(filtered_df, cuts)

        if inplace:
            self.filtered_df = filtered_df
            self.cutflow = cutflow
            return None
        else:
            return filtered_df, cutflow

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
    def plot_hist(
        self,
        var: str,
        bins: List[float] | Tuple[int, float, float] | None = None,
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
        cut: bool = False,
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
        :param cut: use cuts for histogramming
        :param name: histogram name
        :param title: histgoram title
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histogram
        """
        if cut:
            histname = var + "_cut"
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
                th1 = self.filtered_df.Fill(th1, fill_args).GetPtr()
            else:
                th1 = self.df.Fill(th1, fill_args).GetPtr()

            # convert to boost
            hist = Histogram1D(th1=th1)

        hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)

        return hist

    def gen_histograms(
        self, cut: bool = True, to_file: bool | str | Path = True
    ) -> OrderedDict[str, ROOT.TH1]:
        """Generate histograms for all variables mentioned in cuts. Optionally with cuts applied as well"""

        output_histogram_variables = self.all_vars
        n_hists = len(output_histogram_variables)
        self.logger.info(f"Generating {n_hists} histograms...")
        if cut:
            n_hists *= 2
            self.logger.info(f"Cuts applied: ")
            filternames = list(self.filtered_df.GetFilterNames())
            for name in filternames:
                self.logger.info(f"\t{name}")

        # to contain smart pointers to TH1s
        th1_histograms: Dict[str, ROOT.RResultsPtr] = dict()

        def match_weight(var) -> str:
            match variable_data[var]:
                case {"tag": VarTag.TRUTH}:
                    return "truth_weight"
                case {"tag": VarTag.RECO}:
                    return "reco_weight"
                case {"tag": VarTag.META}:
                    return ""
                case _:
                    raise ValueError(f"Unknown variable tag for variable {var}")

        # histogram weights
        for weight_str in ["truth_weight", "reco_weight"]:
            wgt_th1 = ROOT.TH1F(weight_str, weight_str, 100, -1000, 1000)
            th1_histograms[weight_str] = self.df.Fill(wgt_th1, [weight_str])

        for variable_name in output_histogram_variables:
            # which binning?
            bin_args = self.match_bin_args(variable_name)
            weight = match_weight(variable_name)
            fill_cols = [variable_name, weight] if weight else [variable_name]

            th1 = ROOT.TH1F(
                variable_name,
                variable_name,
                *plotting_tools.get_TH1_bins(**bin_args),
            )
            th1_histograms[variable_name] = self.df.Fill(th1, fill_cols)

            if cut:
                cut_hist_name = variable_name + "_cut"
                cut_th1 = ROOT.TH1F(
                    cut_hist_name,
                    variable_name,
                    *plotting_tools.get_TH1_bins(**bin_args),
                )
                th1_histograms[cut_hist_name] = self.filtered_df.Fill(cut_th1, fill_cols)

        # generate histograms
        t = time.time()
        self.logger.info(f"Producing {len(th1_histograms)} histograms...")
        for hist_name in th1_histograms:
            self.histograms[hist_name] = th1_histograms[hist_name].GetValue()
        self.logger.info(
            f"Took {time.time() - t:.3f}s to produce {len(self.histograms)} histograms over {self.df.GetNRuns()} run(s)."
        )

        if cut:
            # generate cutflow
            self.gen_cutflow()
            self.histograms["cutflow"] = self.cutflow.gen_histogram()
            self.cutflow_printout()

        self.logger.info(f"Producted {len(self.histograms)} histograms.")

        if to_file:
            if to_file is True:
                to_file = Path(self.name + "_hisograms.root")
            self.export_histograms(filepath=to_file)

        return self.histograms
