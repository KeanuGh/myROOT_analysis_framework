from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Iterable, OrderedDict, Final, Set

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
    name: str
    df: pd.DataFrame | ROOT.RDataFrame
    cutfile: Cutfile
    cutflow: PCutflow | RCutflow
    lumi: float = 139.0
    label: str = "data"
    logger: logging.Logger = field(default_factory=get_logger)
    lepton: str = "lepton"
    file: Path = field(init=False)
    histograms: OrderedDict[str, Histogram1D] = field(init=False)

    @abstractmethod
    def __len__(self):
        ...

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
    def cuts(self) -> OrderedDict[str, Cut]:
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
    @abstractmethod
    def plot_hist(
        self,
        var: str,
        bins: List[float] | Tuple[int, float, float],
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
        name: str = "",
        title: str = "",
        **kwargs,
    ) -> Histogram1D:
        ...

    # def gen_histograms(self, to_file: bool = True) -> Dict[str, Histogram1D]:
    #     output_histograms = self.cutfile.all_vars()


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
    ) -> None:
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
            cut_cols += [
                CUT_PREFIX + cut.name for cut in self.cuts.values() if cut.is_reco is False
            ]
        if reco:
            self.logger.debug(f"Applying reco cuts to {self.name}...")
            cut_cols += [CUT_PREFIX + cut.name for cut in self.cuts.values() if cut.is_reco is True]

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
            return None
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
        var: str | List[str],
        bins: tuple | list,
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
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
        :param name: histogram name
        :param title: histogram title
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histgoram
        """
        self.logger.debug(f"Generating {var} histogram in {self.name}...")

        if not ax:
            _, ax = plt.subplots()

        weights = self.df[weight] if isinstance(weight, str) else weight
        hist = Histogram1D(df[var], bins, weights, logbins, name=name, title=title, logger=self.logger)  # type: ignore
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
            fig_ax, var, bins, lepton, xlabel, ylabel, title, logx, logy
        )
        fig_ax.legend()
        fig_ax.get_xaxis().set_visible(False)

        # ratio plot
        plotting_tools.set_axis_options(
            accept_ax, var, bins, lepton, xlabel, ylabel, title, logx, False, label=False
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
    """

    cutflow: RCutflow = field(init=False)
    filtered_df: ROOT.RDataFrame | None = None

    def __post_init__(self) -> None:
        self.file = Path(self.name + ".root")

        # generate filtered dataframe for cuts
        self.filtered_df: ROOT.RDataFrame = self.df.Filter("true", "Inclusive")
        for cut in self.cutfile.cuts.values():
            self.filtered_df = self.filtered_df.Filter(cut.cutstr, cut.name)

    def __len__(self) -> int:
        return self.cutflow.total_events

    def __getattr__(self, item):
        return getattr(self.df, item)

    @property
    def columns(self) -> List[str]:
        return list(self.df.GetColumnNames())

    # Variable setting/getting
    # ===================
    @property
    def variables(self) -> Set[str]:
        """Column names that do not contain a cut label"""
        return {col for col in self.columns if CUT_PREFIX not in col}

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
        labels: bool | str | List[str] = True,
        reco: bool = False,
        truth: bool = False,
        inplace: bool = True,
    ) -> ROOT.RDataFrame | None:
        """
        Apply specific cut(s) to DataFrame.

        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :param reco: cut on reco cuts
        :param truth: cut on truth cuts
        :param inplace: If True, applies cuts in place to dataframe in self.
                        If False returns DataFrame object
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied.
                 Raises ValueError if cuts do not exist
        """
        if not labels and (not reco) and (not truth):
            raise ValueError("No cuts supplied")
        if (reco or truth) and isinstance(labels, (str, list)):
            raise ValueError("Supply either named cut labels, truth or reco")
        if truth or reco:
            labels = False

        # handle which cuts to apply depending on what is passed as 'labels'
        cuts_to_apply: List[Cut] = []
        if truth:
            self.logger.debug(f"Applying truth cuts to {self.name}...")
            cuts_to_apply += [cut for cut in self.cuts.values() if cut.is_reco is False]
        if reco:
            self.logger.debug(f"Applying reco cuts to {self.name}...")
            cuts_to_apply += [cut for cut in self.cuts.values() if cut.is_reco is True]

        if isinstance(labels, list):
            self.logger.debug(f"Applying cuts: {labels} to {self.name}...")
            for cut_name in labels:
                try:
                    cuts_to_apply.append(self.cuts[cut_name])
                except KeyError:
                    raise ValueError(f"No cut named {cut_name} in cuts ")

        elif isinstance(labels, str):
            self.logger.debug(f"Applying cut: '{labels}' to {self.name}...")
            try:
                cuts_to_apply.append(self.cuts[labels])
            except KeyError:
                raise ValueError(f"No cut named {labels} in cuts ")

        elif labels is True:
            self.logger.debug(f"Applying all cuts to {self.name}...")
            cuts_to_apply = list(self.cuts.values())

        elif labels is not False:
            raise TypeError("'labels' must be a bool, a string or a list of strings")

        # apply spefic cuts
        return self.gen_cutflow(cuts_to_apply, inplace=inplace)

    def gen_cutflow(
        self, cuts: List[Cut] | None = None, inplace: bool = True
    ) -> None | Tuple[ROOT.RDataFrame, RCutflow]:
        """Generate cutflow, optionally with specific cuts applied"""
        if cuts is None:
            # use all cuts already in dataset
            cuts = list(self.cuts.values())
            filtered_df = self.filtered_df
        else:
            # create new filters for dataset
            filtered_df = self.df.Filter("true", "Inclusive")
            for cut in cuts:
                filtered_df = filtered_df.Filter(cut.cutstr, cut.name)

        cutflow = RCutflow(filtered_df)
        cutflow.gen_cutflow(cuts)

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
        bins: List[float] | Tuple[int, float, float],
        weight: str | float = 1.0,
        ax: plt.Axes = None,
        yerr: ArrayLike | bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
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
        :param name: histogram name
        :param title: histgoram title
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histgoram
        """

        if var not in self.columns:
            raise ValueError(f"No column named {var} in RDataFrame.")

        self.logger.debug(f"Generating {var} histogram in {self.name}...")

        if not ax:
            _, ax = plt.subplots()

        if logbins:
            if not isinstance(bins, tuple) or (len(bins) != 3):
                raise ValueError(
                    "Must pass tuple of (nbins, xmin, xmax) as bins to calculate logarithmic bins"
                )
            bins = np.geomspace(bins[1], bins[2], bins[0] + 1)  # type: ignore

        # handle weight
        if weight:
            fill_args = [var, weight]
        else:
            fill_args = [var]

        th1 = ROOT.TH1F(name, title, *plotting_tools.get_TH1_bins(bins))
        th1 = self.df.Fill(th1, fill_args).GetPtr()

        # convert to boost
        hist = Histogram1D(th1=th1)
        hist = hist.plot(ax=ax, yerr=yerr, normalise=normalise, **kwargs)

        return hist
