from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Iterable, Final, Set, Dict

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike
from tabulate import tabulate  # type: ignore

from src.cutfile import Cutfile, Cut
from src.cutflow import PCutflow, RCutflow
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools, PMG_tool, ROOT_utils
from utils.variable_names import variable_data, VarTag

CUT_PREFIX: Final = "PASS_"


@dataclass
class Dataset(ABC):
    name: str = ""
    df: pd.DataFrame | ROOT.RDataFrame = None
    cutfile: Cutfile = None
    cutflow: PCutflow | RCutflow = None
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

    @abstractmethod
    def __getattr__(self, item):
        ...

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
    def cuts(self) -> List[Cut]:
        return self.cutfile.cuts

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
class PDataset(Dataset):
    """
    PDataset class. Contains/will contain all the variables needed for a singular analysis dataset.

    :param name: Name of Dataset
    :param df: pandas DataFrame containing data
    :param cutfile: Cutfile object containg cuts applied to dataset
    :param cutflow: Cutflow object containing cutflow variables
    :param lumi: Dataset Luminosity
    :param label: Label to put on plots
    :param logger: Logger object to print to. Defaults to console output at DEBUG-level
    :param lepton: Name of charged DY lepton channel in dataset (if applicable)
    :param plot_dir: directory to save plots to. Defaults to current directory
    :param file: File containing pickled DataFrame. Defaults to '<name>.pkl' in current directory
    """

    def __post_init__(self):
        self.file = Path(self.name + "_df.pkl")

    # Builtins
    # ===================
    def __len__(self):
        """Return number of rows in dataframe"""
        return self.df.index.__len__()

    def __getitem__(self, col):
        return self.df[col]

    def __setitem__(self, col, item):
        self.df[col] = item

    def __getattr__(self, item):
        return getattr(self.df, item)

    def __repr__(self):
        return f'Dataset("{self.name}",Variables:{self.variables},Cuts:{self.cut_cols},Events:{len(self)})'

    def __str__(self):
        return f"{self.name},Variable:{self.variables},Cuts:{self.cut_cols},Events:{len(self)}"

    # Variable setting/getting
    # ===================
    @property
    def columns(self) -> List[str]:
        return list(self.df.columms)

    @property
    def variables(self) -> set:
        """Column names that do not contain a cut label"""
        return {col for col in self.df.columns if CUT_PREFIX not in col}

    @property
    def cut_cols(self) -> set:
        """Column names that contain a cut label"""
        return {col for col in self.df.columns if CUT_PREFIX in col}

    def get_dsid(self, dsid: str | int) -> pd.DataFrame:
        """Get events from partcular DSID"""
        return self.df.loc[dsid, :]

    @property
    def n_truth_events(self) -> int:
        """How many truth events in dataset"""
        if self.is_truth:
            return self.df["truth_weight"].notna().sum()
        else:
            return 0

    @property
    def n_reco_events(self) -> int:
        """How many reco events in dataset"""
        if self.is_reco:
            return self.df["reco_weight"].notna().sum()
        else:
            return 0

    def get_truth_events(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Retrun view of truth events"""
        if df is None:
            df = self.df
        return df.loc[df["truth_weight"].notna()]

    def get_reco_events(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Retrun view of reco events"""
        if df is None:
            df = self.df
        return df.loc[df["reco_weight"].notna()]

    @property
    def cross_section(self) -> float:
        """Calculate dataset cross-section"""
        return self.get_cross_section(self.df)

    @property
    def luminosity(self) -> float:
        """Calculate dataset luminosity"""
        return self.get_luminosity(self.df, xs=self.cross_section)

    @staticmethod
    def get_cross_section(
        df: pd.DataFrame, n_events=None, weight_mc_col: str = "weight_mc"
    ) -> float:
        """
        Calculates cross-section of data in dataframe
        :param df: input dataframe
        :param n_events: optional: total number of events. Will calculate if not given.
        :param weight_mc_col: column containing monte carlo weights
        :return: cross-section
        """
        if not n_events:
            n_events = len(df.index)
        return df[weight_mc_col].sum() / n_events

    @classmethod
    def get_luminosity(
        cls, df: pd.DataFrame, xs: float | None = None, weight_col: str = "weight"
    ) -> float:
        """
        Calculates luminosity from dataframe
        :param df: input dataframe
        :param xs: cross-section. If not given, will calculate
        :param weight_col: name of weight column in DAtaFrame
        :return: luminosity
        """
        if not xs:
            xs = cls.get_cross_section(df)
        return df[weight_col].sum() / xs

    # ===============================
    # ========== SUBSETS ============
    # ===============================
    def subset(self, args) -> Dataset:
        """Create new dataset that is subset of this dataset"""
        return PDataset(
            name=self.name,
            df=self.df.loc[args],
            cutfile=self.cutfile,
            cutflow=self.cutflow,
            logger=self.logger,
            lumi=self.lumi,
            label=self.label,
            lepton=self.lepton,
        )

    def subset_dsid(self, dsid: str | int) -> Dataset:
        """Get DSID subset of Dataset"""
        return self.subset(dsid)

    def subset_cut(self, cut_name: str) -> Dataset:
        """Get cut subset of Dataset"""
        return self.subset(self.df[[cut_name + CUT_PREFIX]].all(1))

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    def save_file(self, path: str | Path | None = None) -> None:
        """Saves pickled dataframe to file"""
        if not path:
            path = Path(self.file)
        self.logger.info(f"Saving pickle file...")
        self.df.to_pickle(path)
        self.logger.info(f"Saved pickled DataFrame to {path}")

    def kinematics_printout(self) -> None:
        """Prints some kinematic variables to terminal"""
        self.logger.info("")
        self.logger.info(f"========{self.name.upper()} KINEMATICS ===========")
        self.logger.info(f"cross-section: {self.cross_section:.2f} fb")
        self.logger.info(f"luminosity   : {self.luminosity:.2f} fb-1")

    def dsid_metadata_printout(self, truth: bool = True, reco: bool = False) -> None:
        """print some dataset ID metadata"""
        if self.df.index.names != ["DSID", "eventNumber"]:
            raise ValueError("Incorrect index")

        print_truth = self.is_truth and truth
        print_reco = self.is_reco and reco

        self.logger.info(f"DATASET INFO FOR {self.name}:")

        # per dsid
        rows = []
        for dsid in self.df.index.unique(level="DSID"):
            dsid_slice = self.df.loc[slice(dsid)]
            phys_short = PMG_tool.get_physics_short(dsid)
            row = [
                dsid,
                phys_short,
                len(dsid_slice),
                PMG_tool.get_crossSection(dsid),
                self.lumi,
                PMG_tool.get_genFiltEff(dsid),
                dsid_slice["weight_mc"].sum(),
            ]
            if print_truth:
                row += [dsid_slice["truth_weight"].notna().sum(), dsid_slice["truth_weight"].mean()]
            if print_reco:
                row += [dsid_slice["reco_weight"].notna().sum(), dsid_slice["reco_weight"].mean()]

            rows.append(row)

        header = [
            "DSID",
            "phys_short",
            "n_events",
            "x-s pb",
            "data lumi pb-1",
            "filter eff.",
            "sumw",
        ]
        if print_truth:
            header += ["truth events", "avg truth wgt"]
        if print_reco:
            header += ["reco events", "avg reco wgt"]

        # print it all
        self.logger.info(tabulate(rows, headers=header))

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def apply_cuts(
        self,
        labels: bool | str | List[str] = True,
        reco: bool = False,
        truth: bool = False,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Apply cut(s) to DataFrame.

        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :param reco: cut on reco cuts
        :param truth: cut on truth cuts
        :param inplace: If True, applies cuts in place to dataframe in self.
                        If False returns DataFrame object
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied and associated cut columns removed.
                 Raises ValueError if cuts do not exist in dataframe
        """
        if not labels and (not reco) and (not truth):
            raise ValueError("No cuts supplied")
        if (reco or truth) and isinstance(labels, (str, list)):
            raise ValueError("Supply either named cut labels, truth or reco")
        if truth or reco:
            labels = False

        cut_cols = []

        if truth:
            self.logger.debug(f"Applying truth cuts to {self.name}...")
            cut_cols += [CUT_PREFIX + cut.name for cut in self.cuts if cut.is_reco is False]
        if reco:
            self.logger.debug(f"Applying reco cuts to {self.name}...")
            cut_cols += [CUT_PREFIX + cut.name for cut in self.cuts if cut.is_reco is True]

        if isinstance(labels, list):
            self.logger.debug(f"Applying cuts: {labels} to {self.name}...")
            cut_cols += [CUT_PREFIX + label for label in labels]

        elif isinstance(labels, str):
            self.logger.debug(f"Applying cut: '{labels}' to {self.name}...")
            cut_cols += [CUT_PREFIX + labels]

        elif labels is True:
            self.logger.debug(f"Applying all cuts to {self.name}...")
            cut_cols += [str(col) for col in self.df.columns if CUT_PREFIX in col]

        elif labels is not False:
            raise TypeError("'labels' must be a bool, a string or a list of strings")

        # apply cuts
        self.__check_cut_cols(cut_cols)
        if inplace:
            self.df = self.df.loc[self.df[cut_cols].all(1)]
            self.df.drop(columns=cut_cols, inplace=True)
            return self.df
        else:
            return self.df.loc[self.df[cut_cols].all(1)].drop(columns=cut_cols)

    def __check_cut_cols(self, cuts: List[str]) -> None:
        """Check if cut columns exist in dataframe"""
        if missing_cut_cols := [label for label in cuts if label not in self.df.columns]:
            raise ValueError(f"No cut(s) {missing_cut_cols} in dataset {self.name}")

    def dropna(self, col: str | List[str], drop_inf: bool = False) -> None:
        """Drop rows with missing (and optionally infinite) values in column(s) with a message"""
        if nbad_rows := self.df[col].isna().sum():
            self.df.dropna(subset=col, inplace=True)
            self.logger.debug(f"Dropped {nbad_rows} rows with missing '{col}' values")
        else:
            self.logger.debug(f"No missing values in '{col}' column")

        if drop_inf:
            infrows = np.isinf(self.df[col])
            if nbad_rows := infrows.sum():
                self.df = self.df.loc[~infrows]
                self.logger.debug(f"Dropped {nbad_rows} rows with infinite '{col}' values")
            else:
                self.logger.debug(f"No infinite values in '{col}' column")

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
    def plot_hist(
        self,
        var: str,
        bins: Tuple[int, float, float] | List[float] | None = None,
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
        """
        Generate 1D plots of given variables in dataframe. Returns figure object of list of figure objects.
        Checks to see if histogram already exists in histogram dictionary

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
        :param title: histogram title
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histgoram
        """
        if cut:
            histname = var + "_cut"
        else:
            histname = var

        if histname in self.histograms:
            hist = Histogram1D(th1=self.histograms[histname])
        else:
            self.logger.debug(f"Generating {histname} histogram in {self.name}...")

            if not ax:
                _, ax = plt.subplots()

            weights = self.df[weight] if isinstance(weight, str) else weight
            hist = Histogram1D(
                var=self.df[var],
                bins=bins,
                weight=weights,
                logbins=logbins,
                name=name,
                title=title,
                logger=self.logger,
            )
        hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)
        return hist

    def plot_cut_overlays(
        self,
        var: str,
        bins: List[float] | Tuple[int, float, float],
        weight: str | float = 1.0,
        yerr: ArrayLike | bool = True,
        w2: bool = False,
        normalise: float | bool | str = "lumi",
        logbins: bool = False,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        lepton: str = "lepton",
        **kwargs,
    ) -> None:
        """Plots overlay of cut and acceptance (ratio) plots"""
        self.logger.info(f"Plotting cuts on {var}...")

        fig, (fig_ax, accept_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        if normalise == "lumi":
            normalise = self.lumi
        elif isinstance(normalise, str):
            raise ValueError(f"Unknown normalisation factor '{normalise}'")

        # INCLUSIVE PLOT
        # ================
        h_inclusive = self.plot_hist(
            var=var,
            bins=bins,
            weight=weight,
            ax=fig_ax,
            yerr=yerr,
            normalise=normalise,
            logbins=logbins,
            label=self.label,
            w2=w2,
            color="k",
            linewidth=2,
            name=self.name,
            **kwargs,
        )

        # PLOT CUTS
        # ================
        for cut in self.cutfile.cuts.values():
            h_cut = self.plot_hist(
                var=var,
                bins=bins,
                weight=weight,
                ax=fig_ax,
                apply_cuts=cut.name,
                yerr=yerr,
                normalise=normalise,
                logbins=logbins,
                label=cut.name,
                w2=w2,
                linewidth=2,
                **kwargs,
            )

            # RATIO PLOT
            # ================
            h_cut.plot_ratio(
                h_inclusive,
                ax=accept_ax,
                yerr=yerr,
                label=cut.name,
                color=fig_ax.get_lines()[-1].get_color(),
            )

        # AXIS FORMATTING
        # ==================
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0)

        # figure plot
        plotting_tools.set_axis_options(
            axis=fig_ax,
            var_name=var,
            lepton=lepton,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            logx=logx,
            logy=logy,
        )
        fig_ax.legend()
        fig_ax.get_xaxis().set_visible(False)

        # ratio plot
        plotting_tools.set_axis_options(
            axis=accept_ax,
            var_name=var,
            lepton=lepton,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            logx=logx,
            logy=False,
            label=False,
        )
        accept_ax.set_ylabel("Acceptance")

        out_png_file = f"{self.plot_dir}{var}_CUTS{'_NORMED' if normalise else ''}.png"
        fig.savefig(out_png_file, bbox_inches="tight")
        self.logger.info(f"Figure saved to {out_png_file}")

    def plot_dsid(
        self,
        var: str,
        weight: str,
        bins: Iterable[float] | Tuple[int, float, float] = (30, 0, 5000),
        logbins: bool = False,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        **kwargs,
    ) -> None:
        """
        Plot single variable in dataset with different DSIDs visible

        :param var: variable in dataset to plot
        :param weight: column in dataset to use as weight
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param logbins: whether logarithmic binnings
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param kwargs: keyword arguments to pass to histogram plotting function
        """
        self.logger.info(f"Plotting {var} in {self.name} as slices...")

        fig, ax = plt.subplots()

        # per dsid
        for dsid, dsid_df in self.df.groupby(level="DSID"):
            self.logger.debug(f"Plotting DSID {dsid}...")
            weights = dsid_df[weight] if isinstance(weight, str) else weight
            hist = Histogram1D(
                dsid_df[var], bins=bins, weight=weights, logbins=logbins, logger=self.logger  # type: ignore
            )
            hist.plot(ax=ax, label=dsid, **kwargs)
        # inclusive
        self.logger.debug(f"Plotting inclusive histogram...")
        weights = self.df[weight] if isinstance(weight, str) else weight
        hist = Histogram1D(
            self.df[var], bins=bins, weight=weights, logbins=logbins, logger=self.logger  # type: ignore
        )
        hist.plot(ax=ax, label="Inclusive", color="k", **kwargs)

        ax.legend(fontsize=10, ncol=2)
        title = self.label if not title else title
        plotting_tools.set_axis_options(
            ax, var, bins, self.lepton, xlabel, ylabel, title, logx, logy  # type: ignore
        )

        filename = f"{self.plot_dir}{self.name}_{var}_SLICES.png"
        fig.savefig(filename, bbox_inches="tight")
        self.logger.info(f"Saved dsid slice plot of {var} in {self.name} to {filename}")

    def profile_plot(
        self,
        varx: str,
        vary: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        ax: plt.Axes = None,
        to_file: bool = True,
        xlim: Tuple[float, float] | None = None,
        ylim: Tuple[float, float] | None = None,
        logx: bool = False,
        logy: bool = False,
        lepton: str = "lepton",
        **kwargs,
    ) -> None:
        if not ax:
            fig, ax = plt.subplots()

        self.logger.debug(f"Making profile plot of {varx}-{vary} in {self.name}...")

        ax.scatter(self.df[varx], self.df[vary], c="k", s=0.2, **kwargs)

        ax.set_xlabel(xlabel if xlabel else plotting_tools.get_axis_labels(varx, lepton)[0])
        ax.set_ylabel(ylabel if ylabel else plotting_tools.get_axis_labels(vary, lepton)[0])
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if logx:
            ax.semilogx()
        if logy:
            ax.semilogy()
        hep.atlas.label(
            italic=(True, True),
            loc=0,
            llabel="Internal",
            ax=ax,
            rlabel=title if title else self.label,
        )

        if to_file:
            filename = f"{self.plot_dir}{self.name}_{varx}_{vary}_PROFILE.png"
            plt.savefig(filename, bbox_inches="tight")
            self.logger.info(f"Figure saved as {filename}")
        else:
            plt.show()

        return ax

    def gen_histograms(
        self, cut: bool = True, to_file: bool | str | Path = True
    ) -> OrderedDict[str, Histogram1D]:
        """Generate histograms for all variables mentioned in cutfile. Optionally with cuts applied as well"""

        output_histogram_variables = self.cutfile.all_vars
        n_hists = len(output_histogram_variables)
        if cut:
            n_hists *= 2

        def match_weight(var) -> str:
            match variable_data[var]:
                case {"tag": VarTag.TRUTH}:
                    return "truth_weight"
                case {"tag": VarTag.RECO}:
                    return "reco_weight"
                case _:
                    return "truth_weight"

        i = 1
        for variable_name in output_histogram_variables:
            print(f"Producing histogram {i}/{n_hists}: {variable_name}", end="\r")
            i += 1

            # which binning?
            bin_args = self.match_bin_args(variable_name)
            weight = match_weight(variable_name)

            self.histograms[variable_name] = Histogram1D(
                self[variable_name],
                name=variable_name,
                **bin_args,
                weight=self[weight],
                title=variable_name,
            ).TH1

        if cut:
            cut_df = self.apply_cuts(inplace=False)

            for variable_name in output_histogram_variables:
                print(f"Producing histogram {i}/{n_hists}: {variable_name}", end="\r")

                # which binning?
                bin_args = self.match_bin_args(variable_name)
                weight = match_weight(variable_name)

                i += 1
                cut_hist_name = variable_name + "_cut"
                self.histograms[cut_hist_name] = Histogram1D(
                    cut_df[variable_name],
                    name=cut_hist_name,
                    **bin_args,
                    weight=cut_df[weight],
                    title=variable_name,
                ).TH1

        self.logger.info(f"Producted {n_hists} histograms.")

        if to_file:
            self.export_histograms(filepath=to_file if isinstance(to_file, (str, Path)) else None)

        return self.histograms


@dataclass(slots=True)
class RDataset(Dataset):
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.

    :param name: Name of Dataset
    :param df: pandas DataFrame containing data
    :param cutfile: Cutfile object containg cuts applied to dataset
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
        if self.cutfile is not None:
            for cut in self.cutfile.cuts:
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
        """(re)set cutflow from cutflow histogram and cutfile"""
        if len(self.histograms) == 0:
            raise ValueError("Must generate or load histograms before resetting cutflow")

        self.cutflow = RCutflow(logger=self.logger)
        self.cutflow.import_cutflow(self.histograms["cutflow"], self.cutfile)

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
        """Generate histograms for all variables mentioned in cutfile. Optionally with cuts applied as well"""

        output_histogram_variables = self.cutfile.all_vars
        n_hists = len(output_histogram_variables)
        if cut:
            n_hists *= 2
        self.logger.info(f"Generating {n_hists} histograms...")

        output_histogram_variables = self.cutfile.all_vars

        # to contain smart pointers to TH1s
        th1_histograms: Dict[str, ROOT.RResultsPtr] = dict()

        def match_weight(var) -> str:
            match variable_data[var]:
                case {"tag": VarTag.TRUTH}:
                    return "truth_weight"
                case {"tag": VarTag.RECO}:
                    return "reco_weight"
                case _:
                    return "truth_weight"

        # histogram weights
        for weight_str in ["truth_weight", "reco_weight"]:
            wgt_th1 = ROOT.TH1F(weight_str, weight_str, 500, -50, 50)
            th1_histograms[weight_str] = self.df.Fill(wgt_th1, [weight_str])

        for variable_name in output_histogram_variables:
            # which binning?
            bin_args = self.match_bin_args(variable_name)
            weight = match_weight(variable_name)

            th1 = ROOT.TH1F(
                variable_name,
                variable_name,
                *plotting_tools.get_TH1_bins(**bin_args),
            )
            th1_histograms[variable_name] = self.df.Fill(th1, [variable_name, weight])

            if cut:
                cut_hist_name = variable_name + "_cut"
                cut_th1 = ROOT.TH1F(
                    cut_hist_name,
                    variable_name,
                    *plotting_tools.get_TH1_bins(**bin_args),
                )
                th1_histograms[cut_hist_name] = self.filtered_df.Fill(
                    cut_th1, [variable_name, weight]
                )

        # generate histograms
        t = time.time()
        for i, (hist_name, th1_ptr) in enumerate(th1_histograms.items()):
            self.logger.debug(f"Producing histogram {hist_name}...")
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
