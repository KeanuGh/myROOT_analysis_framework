import logging
import pickle as pkl
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Iterable

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from numpy.typing import ArrayLike

import src.config as config
import utils.plotting_utils as plt_utils
from src.cutfile import Cutfile
from src.cutflow import Cutflow
from src.histogram import Histogram1D
from src.logger import get_logger
from utils.axis_labels import labels_xs


@dataclass
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.
    TODO: migrate all plotting functions to new histogram

    :param name: Name of Dataset
    :param df: pandas DataFrame containing data
    :param cutfile: Cutfile object containg cuts applied to dataset
    :param cutflow: Cutflow object containing cutflow variables
    :param lumi: Dataset Luminosity
    :param label: Label to put on plots
    :param logger: Logger object to print to. Defaults to console output at DEBUG-level
    :param lepton: Name of charged DY lepton channel in dataset (if applicable)
    :param pkl_dir: directory to save plots to. Defaults to current directory
    :param pkl_file: File containing pickled DataFrame. Defaults to '<name>.pkl' in current directory
    """
    name: str
    df: pd.DataFrame
    cutfile: Cutfile
    cutflow: Cutflow
    lumi: float = 139.
    label: str = 'data'
    logger: logging.Logger = get_logger(log_out='console', log_level=10)
    lepton: str = 'lepton'
    plot_dir: str = '.'
    pkl_file: str = field(init=False)

    def __post_init__(self):
        self.pkl_file = self.name + '_df.pkl'

    # Builtins
    # ===================
    def __len__(self):
        """Return number of rows in dataframe"""
        return len(self.df.index)

    def __getitem__(self, col):
        return self.df[col]

    def __setitem__(self, col, item):
        self.df[col] = item

    def __repr__(self):
        return f'Dataset("{self.name}",Variables:{self.variables},Cuts:{self.cut_cols},Events:{len(self)})'

    def __str__(self):
        return f'{self.name},Variable:{self.variables},Cuts:{self.cut_cols},Events:{len(self)}'

    def __add__(self, other) -> pd.DataFrame:
        """Concatenate two dataframes"""
        return pd.concat([self.df, other.df], ignore_index=True, copy=False)

    def __iadd__(self, other):
        """Concatenate dataframe to self.df"""
        self.df.append(other.df, ignore_index=True)

    # Variable setting/getting
    # ===================
    def set_plot_dir(self, path: str):
        self.plot_dir = path

    def set_pkl_path(self, filepath: str):
        self.pkl_file = filepath

    @property
    def variables(self):
        """Column names that do not contain a cut label"""
        return {
            col for col in self.df.columns
            if config.cut_label not in col
        }

    @property
    def cut_cols(self) -> set:
        """Column names that contain a cut label"""
        return {
            col - config.cut_label
            for col in self.df.columns
            if config.cut_label in col
        }

    @property
    def is_truth(self) -> bool:
        """Does dataset contain truth data?"""
        return 'truth_weight' in self.df.columns

    @property
    def is_reco(self) -> bool:
        """Does dataset contain reco data?"""
        return 'is_reco' in self.df.columns

    @property
    def n_truth_events(self) -> int:
        """How many truth events in dataset"""
        if self.is_truth:
            return df['truth_weight'].notna().sum()
        else:
            return 0

    @property
    def n_reco_events(self) -> int:
        """How many reco events in dataset"""
        if self.is_reco:
            return df['reco_weight'].notna().sum()
        else:
            return 0

    def get_truth_events(self) -> pd.DataFrame:
        """Retrun view of truth events"""
        return self.df.loc[self.df['truth_weight'].notna()]

    def get_reco_events(self) -> pd.DataFrame:
        """Retrun view of reco events"""
        return self.df.loc[self.df['reco_weight'].notna()]

    @property
    def cross_section(self) -> float:
        """Calculate dataset cross-section"""
        return self.get_cross_section(self.df)

    @property
    def luminosity(self) -> float:
        """Calculate dataset luminosity"""
        return self.get_luminosity(self.df, xs=self.cross_section)

    @staticmethod
    def get_cross_section(df: pd.DataFrame, n_events=None, weight_mc_col: str = 'weight_mc') -> float:
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
    def get_luminosity(cls, df: pd.DataFrame, xs: float = None, weight_col: str = 'weight') -> float:
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
    # ========= PRINTOUTS ===========
    # ===============================
    def save_pkl_file(self, path: str = None) -> None:
        """Saves pickle"""
        if not path: path = self.pkl_file
        self.df.to_pickle(path)
        self.logger.info(f"Saved pickled DataFrame to {path}")

    def cutflow_printout(self) -> None:
        """Prints cutflow table to terminal"""
        self.cutflow.printout()

    def kinematics_printout(self) -> None:
        """Prints some kinematic variables to terminal"""
        self.logger.info("")
        self.logger.info(f"========{self.name.upper()} KINEMATICS ===========")
        self.logger.info(f"cross-section: {self.cross_section:.2f} fb")
        self.logger.info(f"luminosity   : {self.luminosity:.2f} fb-1")

    def print_latex_table(self, filepath: str) -> None:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        self.cutflow.print_latex_table(filepath)
        self.logger.info(f"Saved LaTeX cutflow table in {filepath}")

    # ===============================
    # ========== CUTTING ============
    # ===============================
    def cut_on_cutgroup(self, group: str) -> pd.DataFrame:
        """Cuts on cutgroup on input dataframe or series"""
        return self.df.loc[df[self.cutfile.get_cutgroup(group)].all(1)]

    def apply_cuts(self,
                   labels: Union[bool, str, List[str]] = True,
                   inplace: bool = False
                   ) -> Union[pd.DataFrame, None]:
        """
        Apply cut(s) to DataFrame.

        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :param inplace: If True, applies cuts in place to dataframe in self.
                        If False returns DataFrame object
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied and associated cut columns removed.
                 Raises ValueError if cuts do not exist in dataframe
        """

        def __check_cut_cols(c: List[str]) -> None:
            """Check if columns exist in dataframe"""
            if missing_cut_cols := [
                label for label in c
                if label not in self.df.columns
            ]:
                raise ValueError(f"No cut(s) {missing_cut_cols} in dataset {self.name}...")

        if not labels:
            if inplace:
                return
            else:
                return self.df

        elif isinstance(labels, list):
            self.logger.debug(f"Applying cuts: {labels} to {self.name}...")
            cut_cols = [label + config.cut_label for label in labels]
            __check_cut_cols(cut_cols)

        elif isinstance(labels, str):
            self.logger.debug(f"Applying cut: {labels} to {self.name}...")
            cut_cols = [labels + config.cut_label]
            __check_cut_cols(cut_cols)

        elif labels is True:
            self.logger.debug(f"Applying all cuts to {self.name}...")
            cut_cols = [str(col) for col in self.df.columns if config.cut_label in col]
            __check_cut_cols(cut_cols)

        else:
            raise TypeError("'labels' must be a bool, a string or a list of strings")

        # apply cuts
        if inplace:
            self.df = self.df.loc[self.df[cut_cols].all(1)]
            self.df.drop(columns=cut_cols, inplace=True)

        else:
            return self.df.loc[self.df[cut_cols].all(1)].drop(columns=cut_cols)

    # ===========================================
    # =========== PLOTING FUNCTIONS =============
    # ===========================================
    def plot_hist(
            self,
            var: Union[str, List[str]],
            bins: Union[tuple, list],
            weight: Union[str, float] = 1.,
            ax: plt.Axes = None,
            yerr: Union[ArrayLike, str] = None,
            normalise: Union[float, bool] = False,
            logbins: bool = False,
            apply_cuts: Union[bool, str, List[str]] = True,
            **kwargs
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
        :param apply_cuts: True to apply all cuts to dataset before plotting or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param logbins: whether logarithmic binnings
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        :return: Histgoram
        """
        self.logger.debug(f"Generating {var} histogram in {self.name}...")

        if not ax:
            _, ax = plt.subplots()

        df = self.apply_cuts(apply_cuts)
        weights = (
            df[weight]
            if isinstance(weight, str)
            else weight
        )
        hist = Histogram1D(bins, df[var], weights, logbins)
        hist.plot(
            ax=ax,
            yerr=yerr,
            normalise=normalise,
            **kwargs
        )
        return hist

    def plot_cut_overlays(
            self,
            var: str,
            bins: Union[List[float], Tuple[int, float, float]],
            weight: Union[str, float] = 1.,
            yerr: Union[ArrayLike, str] = None,
            w2: bool = False,
            normalise: Union[float, bool, str] = 'lumi',
            logbins: bool = False,
            logx: bool = False,
            logy: bool = True,
            xlabel: str = '',
            ylabel: str = '',
            title: str = '',
            lepton: str = 'lepton',
            **kwargs
    ) -> None:
        """Plots overlay of cutgroups and acceptance (ratio) plots"""
        self.logger.info(f"Plotting cuts on {var}...")

        fig, (fig_ax, accept_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

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
            color='k',
            linewidth=2,
            **kwargs
        )

        # PLOT CUTS
        # ================
        for cutgroup in self.cutfile.cutgroups.keys():
            self.logger.debug(f"    - generating cutgroup '{cutgroup}'")
            h_cut = self.plot_hist(
                var=var,
                bins=bins,
                weight=weight,
                ax=fig_ax,
                apply_cuts=cutgroup,
                yerr=yerr,
                normalise=normalise,
                logbins=logbins,
                label=cutgroup,
                w2=w2,
                linewidth=2,
                **kwargs
            )

            # RATIO PLOT
            # ================
            hep.histplot(h_cut.bin_values / h_inclusive.bin_values,
                         bins=h_cut.bin_edgesn, ax=accept_ax, label=cutgroup,
                         color=fig_ax.get_lines()[-1].get_color())

        # AXIS FORMATTING
        # ==================
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)

        # figure plot
        plt_utils.set_axis_options(fig_ax, var, bins, lepton, xlabel, ylabel, title, logx, logy)
        fig_ax.legend()
        fig_ax.axes.xaxis.set_visible(False)

        # ratio plot
        plt_utils.set_axis_options(accept_ax, var, bins, lepton, xlabel, ylabel, title, logx, logy)
        accept_ax.set_ylabel("Acceptance")

        out_png_file = f"{self.plot_dir}{var}_cuts_{'_NORMED' if normalise else ''}.png"
        fig.savefig(out_png_file, bbox_inches='tight')
        self.logger.info(f"Figure saved to {out_png_file}")

    def plot_2d_cutgroups(self,
                          x_var: str, y_var: str,
                          xbins: Union[tuple, list], ybins: Union[tuple, list],
                          weight: str,
                          plot_label: str = '',
                          is_logz: bool = True,
                          to_pkl: bool = False,
                          ) -> None:
        """
        Runs over cutgroups in dictrionary and plots 2d histogram for each group

        :param x_var: column in dataframe to plot on x-axis
        :param y_var: column in dataframe to plot on y-axis
        :param weight: weight column
        :param xbins: binning in x
        :param ybins: binning in y
        :param plot_label: plot title
        :param is_logz: whether display z-axis logarithmically
        :param to_pkl: whether to save histograms as pickle file
        """
        hists = dict()

        # INCLUSIVE
        fig, ax = plt.subplots(figsize=(7, 7))
        x_vars = self.df[x_var]
        y_vars = self.df[y_var]

        out_path = self.plot_dir + f"2d_{x_var}-{y_var}_inclusive.png"
        hist = plt_utils.histplot_2d(
            var_x=x_vars, var_y=y_vars,
            xbins=xbins, ybins=ybins,
            ax=ax, fig=fig,
            weights=self.df[weight],
            is_z_log=is_logz,
        )
        if to_pkl:
            hists['inclusive'] = hist

        # get axis labels
        xlabel, _ = plt_utils.get_axis_labels(str(x_var), self.lepton)
        ylabel, _ = plt_utils.get_axis_labels(str(y_var), self.lepton)

        hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - inclusive', loc=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.savefig(out_path, bbox_inches='tight')
        self.logger.info(f"printed 2d histogram to {out_path}")
        plt.close(fig)

        for cutgroup in self.cutfile.cutgroups:
            self.logger.info(f"    - generating cutgroup '{cutgroup}'")
            fig, ax = plt.subplots(figsize=(7, 7))

            cut_df = self.cut_on_cutgroup(cutgroup)
            weight_cut = cut_df[weight]
            x_vars = cut_df[x_var]
            y_vars = cut_df[y_var]

            out_path = self.plot_dir + f"2d_{x_var}-{y_var}_{cutgroup}.png"
            hist = plt_utils.histplot_2d(
                var_x=x_vars, var_y=y_vars,
                xbins=xbins, ybins=ybins,
                ax=ax, fig=fig,
                weights=weight_cut,
                is_z_log=is_logz,
            )
            if to_pkl:
                hists[cutgroup] = hist

            # get axis labels
            xlabel, _ = plt_utils.get_axis_labels(str(x_var), self.lepton)
            ylabel, _ = plt_utils.get_axis_labels(str(y_var), self.lepton)

            hep.atlas.label(italic=(True, True), ax=ax, llabel='Internal', rlabel=plot_label + ' - ' + cutgroup, loc=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            fig.savefig(out_path, bbox_inches='tight')
            self.logger.info(f"printed 2d histogram to {out_path}")
            plt.close(fig)

        if to_pkl:
            with open(self.plot_dir + plot_label + f"_{x_var}-{y_var}_2d.pkl", 'wb') as f:
                pkl.dump(hists, f)
                self.logger.info(f"Saved pickle file to {f.name}")

    def plot_mass_slices(
            self,
            var: str,
            weight: str,
            bins: Union[Iterable[float], Tuple[int, float, float]] = (30, 0, 5000),
            logbins: bool = False,
            logx: bool = False,
            logy: bool = True,
            xlabel: str = '',
            ylabel: str = '',
            title: str = '',
            apply_cuts: Union[bool, str, List[str]] = True,
            **kwargs
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
        :param apply_cuts: True to apply all cuts to dataset before plotting or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param kwargs: keyword arguments to pass to histogram plotting function
        """
        self.logger.info(f'Plotting {var} in {self.name} as slices...')

        fig, ax = plt.subplots()
        df = self.apply_cuts(apply_cuts)

        # per dsid
        for dsid, dsid_df in self.df.groupby(level='DSID'):
            weights = dsid_df[weight] if isinstance(weight, str) else weight
            hist = Histogram1D(bins, dsid_df[var], weights, logbins)
            hist.plot(ax=ax, label=dsid, **kwargs)
        # inclusive
        weights = df[weight] if isinstance(weight, str) else weight
        hist = Histogram1D(bins, df[var], weights, logbins)
        hist.plot(ax=ax, label='Inclusive', color='k', **kwargs)

        ax.legend(fontsize=10, ncol=2)
        title = self.label if not title else title
        plt_utils.set_axis_options(ax, var, bins, self.lepton, xlabel, ylabel, title, logx, logy)

        filename = f"{self.plot_dir}{self.name}_{var}_SLICES.png"
        fig.savefig(filename, bbox_inches='tight')
        self.logger.info(f'Saved mass slice plot of {var} in {self.name} to {filename}')

    def gen_cutflow_hist(self,
                         event: bool = True,
                         ratio: bool = False,
                         cummulative: bool = False,
                         a_ratio: bool = False,
                         all_plots: bool = False,
                         ) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param event: y-axis is number of events passing each cut
        :param ratio: ratio of events passing each cut to inclusive sample
        :param cummulative: ratio of each cut to the previous cut
        :param a_ratio: ratio of cut to inclusive sample
        :param all_plots: it True, plot all
        :return: None
        """
        if all_plots:
            event = ratio = cummulative = a_ratio = True
        if event:
            self.cutflow.print_histogram(self.plot_dir, 'event')
        if ratio:
            self.cutflow.print_histogram(self.plot_dir, 'ratio')
        if cummulative:
            self.cutflow.print_histogram(self.plot_dir, 'cummulative')
        if a_ratio:
            self.cutflow.print_histogram(self.plot_dir, 'a_ratio')

    def profile_plot(
            self,
            varx: str,
            vary: str,
            title: str = '',
            xlabel: str = '',
            ylabel: str = '',
            ax: plt.Axes = None,
            to_file: bool = True,
            xlim: Tuple[float, float] = None,
            ylim: Tuple[float, float] = None,
            logx: bool = False,
            logy: bool = False,
            **kwargs
    ) -> None:
        if not ax:
            fig, ax = plt.subplots()

        ax.scatter(self.df[varx], self.df[vary], **kwargs)

        ax.set_xlabel(xlabel if xlabel else labels_xs[varx]['xlabel'])
        ax.set_ylabel(ylabel if ylabel else labels_xs[vary]['xlabel'])
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if logx: ax.semilogx()
        if logy: ax.semilogy()
        hep.atlas.label(italic=(True, True), loc=0, llabel='Internal', ax=ax, rlabel=title if title else self.label)

        plt.show()
        if to_file:
            plt.savefig(f"{self.plot_dir}{varx}_{vary}_PROFILE.png", bbox_inches='tight')

        return ax
