import inspect
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Callable, Any, Sequence, Generator

import ROOT
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep
import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike
from tabulate import tabulate  # type: ignore

from src.dataset import Dataset, RDataset  # PDataset
from src.datasetbuilder import DatasetBuilder, lumi_year
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools, ROOT_utils
from utils.context import handle_dataset_arg, redirect_stdout


@dataclass(slots=True)
class AnalysisPath:
    """Container class for paths needed by analyses"""

    plot_dir: Path
    latex_dir: Path
    log_dir: Path

    def create_paths(self):
        for p in (
            self.plot_dir,
            self.latex_dir,
            self.log_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


class Analysis:
    """
    Analysis class acts as a container for the src.dataset.Dataset class. Contains methods to apply either to
    single datasets or across multiple datasets.
    Access datasets in class with analysis.dataset_name or analysis['dataset_name']. Can set by key but not by attribute
    When calling a method that applies to only one dataset, naming the dataset in argument ds_name is optional.
    """

    __slots__ = (
        "name",
        "paths",
        "histograms",
        "logger",
        "datasets",
        "global_lumi",
        "_output_dir",
        "cmap",
    )

    def __init__(
        self,
        data_dict: dict[str, dict],
        analysis_label: str,
        global_lumi: float | None = 139.0,
        output_dir: Path | str | None = None,
        log_level: int = 20,
        log_out: str = "both",
        timedatelog: bool = True,
        separate_loggers: bool = False,
        regen_histograms: bool = False,
        **kwargs,
    ):
        """
        :param data_dict: Dictionary of dictionaries containing paths to root files and the tree to extract from each.
               The key to the top-level dictionary is the label assigned to the dataset.
        :param global_lumi: All data will be scaled to this luminosity (fb-1)
        :param output_dir: Root directory for outputs
        :param log_level: Logging level. Default INFO. See https://docs.python.org/3/library/logging.html#logging-levels
        :param log_out: Where to set log output: 'FILE', 'CONSOLE' or 'BOTH'. (case-insensitive)
        :param timedatelog: Whether to output log filename with timedate
               (useful to turn off for testing or you'll be flooded with log files)
        :param separate_loggers: Whether each dataset should output logs to separate log files
        :param regen_histograms: Whether to regenerate all histograms for all datasets (can be applied separately)
        :param kwargs: Options arguments to pass to all dataset builders
        """
        self.name = analysis_label
        if self.name in data_dict:
            raise SyntaxError("Analysis must have different name to any dataset")
        self.histograms: dict[str, ROOT.TH1] = dict()

        # SET OUTPUT DIRECTORIES
        # ===========================
        if not output_dir:
            # root in the directory above this one
            output_dir = Path(__file__).absolute().parent.parent
        self._output_dir = Path(output_dir) / "outputs" / analysis_label  # where outputs go
        self.paths = AnalysisPath(
            plot_dir=Path(self._output_dir) / "plots",
            latex_dir=Path(self._output_dir) / "LaTeX",  # where to print latex cutflow table(s)
            log_dir=Path(self._output_dir) / "logs",
        )
        self.paths.create_paths()

        # LOGGING
        # ============================
        self.logger = get_logger(
            name=self.name,
            log_path=self.paths.log_dir,
            log_level=log_level,
            log_out=log_out,
            timedatelog=timedatelog,
        )

        # SET OTHER GLOBAL OPTIONS
        # ============================
        self.name = analysis_label
        if "year" in kwargs:
            try:
                self.global_lumi = lumi_year[kwargs["year"]]
            except KeyError as e:
                raise KeyError(
                    f"Unknown data-year: {kwargs['year']}. Known data-years: {lumi_year.keys()}"
                ) from e
        else:
            self.global_lumi = global_lumi
        self.logger.debug(f"Set global luminosity scale to {self.global_lumi} pb-1")

        # BUILD DATASETS
        # ============================
        self.datasets: dict[str, Dataset] = dict()
        for dataset_name, data_args in data_dict.items():
            self.logger.info("")
            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info(f"======== INITIALISING DATASET '{dataset_name}' =========")
            self.logger.info("=" * (42 + len(dataset_name)))

            # get dataset build arguments out of options passed to analysis
            if "regen_histograms" in data_args and "regen_histograms" in kwargs:
                raise SyntaxError(
                    f"Got multiple values for argument 'regen_histograsms' for dataset {dataset_name}"
                )
            if dup_args := set(data_args) & set(kwargs):
                raise SyntaxError(
                    f"Got multiple values for argument(s) {dup_args} for dataset {dataset_name}"
                )
            args = data_args | kwargs

            # this argument should be handled separately
            if "regen_histograms" in args:
                indiv_regen_hists = args.pop("regen_histograms")
            else:
                indiv_regen_hists = regen_histograms

            # make dataset
            builder = DatasetBuilder(
                name=dataset_name,
                **self.__match_params(args, DatasetBuilder.__init__),
                logger=(
                    self.logger
                    if not separate_loggers  # use single logger
                    else get_logger(  # if seperate, make new logger for each Dataset
                        name=dataset_name,
                        log_path=self.paths.log_dir,
                        log_level=log_level,
                        log_out=log_out,
                        timedatelog=timedatelog,
                    )
                ),
            )
            dataset = builder.build(**self.__match_params(args, DatasetBuilder.build))

            # apply some manual settings
            if "binnings" in args:
                dataset.binnings = args["binnings"]
            if "is_data" in args:
                dataset.is_data = args["is_data"]
            if "is_signal" in args:
                dataset.is_signal = args["is_signal"]

            if separate_loggers:
                # set new logger to append to analysis logger
                dataset.logger = self.logger
                dataset.logger.debug(f"{dataset_name} log handler returned to analysis.")  # test

            # histogramming
            histogram_file = self._output_dir / f"{dataset_name}_histograms.root"
            if not indiv_regen_hists and histogram_file.exists():
                # just read in previous histogram file if it exists
                dataset.import_histograms(histogram_file)
                dataset.reset_cutflows()
                dataset.logger.info(
                    f"Imported {len(dataset.histograms)} histogram(s) from file {histogram_file}"
                )
            else:
                dataset.gen_histograms(to_file=histogram_file)

            # integrate into own histogram dictionary
            for hist_name, hist in dataset.histograms.items():
                self.histograms[dataset_name + "_" + hist_name] = hist

            # merge histograms if label is given
            if "merge_into" in args:
                merged_ds = args["merge_into"]

                # create a "dummy" dataset with only the label and cuts of the first dataset
                if merged_ds not in self.datasets:
                    self[merged_ds] = RDataset(
                        name=merged_ds, label=dataset.label, cuts=dataset.cuts, is_merged=True
                    )
                else:
                    # verify cuts are the same
                    if self[merged_ds].cuts != dataset.cuts:
                        raise ValueError(
                            f"Cuts in merged dataset are not the same. Got:"
                            f"\nMerged: {self[merged_ds].cuts}"
                            f"\nand"
                            f"\nOriginal: {dataset.cuts}"
                        )

                for hist_name, hist in dataset.histograms.items():
                    hist_name_merged = merged_ds + "_" + hist_name
                    if hist_name_merged not in self.histograms:
                        self.histograms[hist_name_merged] = dataset.histograms[hist_name].Clone()
                        self.logger.debug(
                            f"Added {hist_name_merged} to histograms from {dataset_name}"
                        )
                    else:
                        # check bin edges are the same
                        edges_1 = [
                            self.histograms[hist_name_merged].GetBinLowEdge(i + 1)
                            for i in range(self.histograms[hist_name_merged].GetNbinsX())
                        ]
                        edges_2 = [
                            dataset.histograms[hist_name].GetBinLowEdge(i + 1)
                            for i in range(dataset.histograms[hist_name].GetNbinsX())
                        ]
                        if edges_1 != edges_2:
                            raise ValueError(
                                f"Histograms cannot be merged, bin edges do not match."
                                f"Got \n{edges_1} and \n{edges_2}"
                            )

                        # merge but capture any ROOT error messages
                        with redirect_stdout(in_stream="stderr") as root_msg:
                            self.histograms[hist_name_merged].Add(dataset.histograms[hist_name])
                        if "Attempt to add histograms with different labels" in root_msg.getvalue():
                            self.logger.warning(
                                f"Histograms {hist_name_merged} and {dataset_name}_{hist_name} have different labels."
                                f"Cannot add directly. Attempting TH1::Merge.."
                            )

                        self.logger.debug(
                            f"Merged {dataset_name}_{hist_name} into {hist_name_merged}"
                        )

                    # add/replace to dummy dataset
                    self[merged_ds].histograms[hist_name] = self.histograms[hist_name_merged]

                # sum cutflows
                if self[merged_ds].cutflows:
                    for cutflow_name, cutflow in self[merged_ds].cutflows.items():
                        self[merged_ds].cutflows[cutflow_name] += cutflow
                else:
                    self[merged_ds].cutflows = deepcopy(dataset.cutflows)

            try:
                dataset.dsid_metadata_printout()
            except NotImplementedError:
                pass

            self[dataset_name] = dataset  # save to analysis

            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info(f"========= DATASET '{dataset_name}' INITIALISED =========")
            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info("")

        self.logger.info("=" * (len(analysis_label) + 23))
        self.logger.info(f"ANALYSIS '{analysis_label}' INITIALISED")

    @staticmethod
    def __match_params(params: dict[str, Any], func: Callable) -> dict[str, Any]:
        """Return parameters matching passed function signature"""
        args = dict()
        for arg in inspect.signature(func).parameters:
            if str(arg) == "self":
                continue
            if arg in params:
                args[arg] = params[arg]
        return args

    # ===============================
    # ========== BUILTINS ===========
    # ===============================
    def __getitem__(self, key: str) -> Dataset:
        return self.datasets[key]

    def __setitem__(self, ds_name: str, dataset: Dataset) -> None:
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Analysis dataset must be of type {Dataset}")
        self.datasets[ds_name] = dataset

    def __delitem__(self, key: str) -> None:
        del self.datasets[key]

    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self) -> Generator[Dataset, None, None]:
        yield from self.datasets.values()

    def __repr__(self) -> str:
        return f'Analysis("{self.name}", Datasets:{{{", ".join([f"{name}: {len(d)}" for name, d in self.datasets.items()])}}}'

    def __str__(self) -> str:
        return f'"{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}" for name, d in self.datasets.items()])}}}'

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_hist(
        self,
        var: str | Sequence[str] | Histogram1D,
        datasets: str | Sequence[str] | None = None,
        yerr: ArrayLike | str = True,
        labels: list[str] | None = None,
        w2: bool = False,
        normalise: float | bool = False,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        lepton: str = "lepton",
        scale_by_bin_width: bool = False,
        stats_box: bool = False,
        x_axlim: tuple[float, float] | None = None,
        y_axlim: tuple[float, float] | None = None,
        gridopts: bool | tuple[bool | None, str | None, str | None] = False,
        ratio_plot: bool = True,
        ratio_fit: bool = False,
        ratio_axlim: float | tuple[float, float] | None = None,
        ratio_label: str = "Ratio",
        ratio_err: str = "sumw2",
        filename: str | Path | None = None,
        cut: bool = False,
        suffix: str = "",
        prefix: str = "",
        **kwargs,
    ) -> None:
        """
        Plot same variable from different datasets.
        If one dataset is passed but multiple variables, will plot overlays of the variable for that one dataset
        Checks to see if histogram exists in dataset histogram dictionary first, to avoid refilling histogram

        :param datasets: string or list of strings corresponding to datasets in the analysis
        :param var: variable name to be plotted. Either a string that exists in all datasets
                    or a list one for each dataset, or histograms themselves
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param labels: list of labels for plot legend corresponding to each dataset
        :param w2: Whether to do a poissonian interval error calculation based on weights
        :param normalise: Normalisation value:
                          - int or float
                          - True for normalisation of unity
                          - 'lumi' (default) for normalisation to global_uni variable in analysis
                          - False for no normalisation
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param lepton: lepton to fill variable label
        :param scale_by_bin_width: divide histogram bin values by bin width
        :param stats_box: display stats box
        :param x_axlim: x-axis limits. If None matplolib decides
        :param y_axlim: x-axis limits. If None matplolib decides
        :param gridopts: arguments to pass to plt.grid()
        :param ratio_plot: If True, adds ratio of the first plot with each subseqent plot below
        :param ratio_fit: If True, fits ratio plot to a 0-degree polynomial and display line, chi-square and p-value
        :param ratio_axlim: pass to yax_lim in rato plotter
        :param ratio_label: y-axis label for ratio plot
        :param ratio_err: yerr for ratio plot. Either "sumw2", "binom", or "carry"
        :param filename: name of output
        :param cut: applies cuts before plotting
        :param suffix: suffix to add at end of histogram/file name
        :param prefix: prefix to add at start of histogram/file
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        """
        self.logger.info(
            f"Plotting %s%s...", var, f"in {datasets}" in datasets if datasets else None
        )
        _passed_hists = False

        # naming template for file/histogram name
        name_template = (
            ((prefix + "_") if prefix else "")  # prefix
            + "{short}"
            + ("_NORMED" if normalise else "")  # normalisation flag
            + (("_" + suffix) if suffix else "")  # suffix
        )
        name_template_short = "{dataset}_{variable}"  # name of dataset(s)  # name of variable(s)

        if datasets and isinstance(datasets, str):
            datasets = [datasets]
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(var, (str, Histogram1D)):
            var = [var]

        # handle how cuts are going to be done
        if cut is False or cut is None:
            cutsets_to_loop = [False]
        elif cut is True:
            # separate plot for EACH set of cuts
            if datasets and len(datasets) > 1:
                # check that all datasets to be plot have the same sets of cuts
                first_cutflow = self[datasets[0]].cuts.keys()
                if not all(ds.cuts.keys() == first_cutflow for ds in self.datasets.values()):
                    raise ValueError("Datasets do not have the same cuts")
            cutsets_to_loop = self[datasets[0]].cuts
        elif isinstance(cut, str):
            cutsets_to_loop = [cut]
        elif isinstance(cut, list):
            cutsets_to_loop = cut
        else:
            raise ValueError(f"Unknown cut {cut}")

        # figure out if we should loop over datasets or variables or both
        varloop = False
        datasetloop = False
        if datasets is None:
            varloop = (True,)
            n_overlays = len(var)
        elif len(datasets) > 1:
            if (len(datasets) != len(var)) and (len(var) > 1):
                raise ValueError(
                    "Number of datasets and variables must match if passing multiple variables."
                )
            datasetloop = True
            if len(var) > 1:
                varloop = True
            n_overlays = len(datasets)
        elif len(var) > 1:
            n_overlays = len(var)
            varloop = True
        else:
            n_overlays = 1

        # no ratio if just one thing being plotted
        if n_overlays == 1:
            ratio_plot = False

        if isinstance(normalise, str):
            if normalise == "lumi":
                normalise = self.global_lumi
            else:
                raise ValueError("Only 'lumi' allowed for string value normalisation")

        if n_overlays > 2:
            self.logger.warning("Not enough space to display stats box. Will not display")
            stats_box = False

        if labels:
            assert len(labels) == n_overlays, (
                f"Labels iterable (length: {len(labels)}) "
                f"must be of same length as number of overlaid plots ({n_overlays})"
            )

        # plot for each cut
        for cut in cutsets_to_loop:
            if ratio_plot:
                fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
            else:
                fig, ax = plt.subplots()
                ratio_ax = None  # just so IDE doesn't complain about missing variable

            # plotting loop
            hists = []  # add histograms to be overlaid in this list
            for i in range(n_overlays):
                if datasets:
                    dataset = datasets[i] if datasetloop else datasets[0]
                else:
                    dataset = None
                varname = var[i] if varloop else var[0]

                try:
                    label = labels[i] if labels else self[dataset].label
                except KeyError:
                    label = None

                # plot
                if isinstance(varname, Histogram1D):
                    hist = varname
                    _passed_hists = True
                else:
                    hist = self.get_hist(varname, dataset, cut)
                hist.plot(
                    ax=ax,
                    yerr=yerr,
                    normalise=normalise,
                    stats_box=stats_box,
                    scale_by_bin_width=scale_by_bin_width,
                    label=None if n_overlays == 1 else label,
                    w2=w2,
                    **kwargs,
                )

                # save
                hists.append(hist)

                if ratio_plot and len(hists) > 1:
                    # ratio of first dataset to this one
                    try:
                        label = (
                            f"{labels[-1]}/{labels[0]}"
                            if labels
                            else f"{self[dataset].label}/{self[datasets[0]].label}"
                        )
                    except KeyError:
                        label = None
                    # match ratio colour to plot
                    color = ax.get_lines()[-1].get_color() if (n_overlays > 2) else "k"
                    ratio_hist_name = (
                        name_template.format(
                            short=name_template_short.format(
                                dataset=f"{dataset}_{datasets[0]}" if dataset else None,
                                variable=varname.name if _passed_hists else varname,
                            )
                        )
                        + "_ratio"
                    )
                    ratio_hist = hists[0].plot_ratio(
                        hists[-1],
                        ax=ratio_ax,
                        yerr=ratio_err,
                        label=label,
                        normalise=bool(normalise),
                        color=color,
                        fit=ratio_fit,
                        yax_lim=ratio_axlim,
                        name=ratio_hist_name,
                        display_stats=n_overlays <= 3,  # display results if there are <2 fits
                    )
                    self.histograms[ratio_hist_name] = ratio_hist.TH1

            if n_overlays > 1:
                ax.legend(fontsize=10, loc="upper right")
            plotting_tools.set_axis_options(
                axis=ax,
                var_name=var,
                lepton=lepton,
                xlim=(hist.bin_edges[0], hist.bin_edges[-1]),
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                logx=logx,
                logy=logy,
                diff_xs=scale_by_bin_width,
            )
            if x_axlim:
                ax.set_xlim(*x_axlim)
            if y_axlim:
                ax.set_ylim(*y_axlim)
            if gridopts:
                ax.grid(*gridopts)
            if ratio_plot:
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.1, wspace=0)
                ax.set_xticklabels([])
                ax.set_xlabel("")

                if n_overlays > 2:  # don't show legend if there's only two plots
                    ratio_ax.legend(fontsize=10, loc=1)

                plotting_tools.set_axis_options(
                    axis=ratio_ax,
                    var_name=var,
                    lepton=lepton,
                    xlim=(ratio_hist.bin_edges[0], ratio_hist.bin_edges[-1]),
                    xlabel=xlabel,
                    ylabel=ratio_label,
                    title="",
                    logx=logx,
                    logy=False,
                    label=False,
                )

            if filename:
                filepath = self.paths.plot_dir / filename
            else:
                if isinstance(var, str):
                    varname = var
                elif _passed_hists:
                    if isinstance(var, Histogram1D):
                        varname = var.name
                    else:
                        varname = "_".join([h.name for h in var])
                else:
                    varname = "_".join(var)
                filepath = self.paths.plot_dir / (
                    name_template.format(
                        short=name_template_short.format(
                            dataset="_".join(datasets) if dataset else None, variable=varname
                        )
                    )
                    + (f"_{cut}_cut" if cut else "")
                    + ".png"
                )

            fig.savefig(filepath, bbox_inches="tight")
            self.logger.info(f"Saved plot of {var} to {filepath}")
            plt.close(fig)

    def stack_plot(
        self,
        datasets: str | Sequence[str],
        var: str | Sequence[str],
        yerr: ArrayLike | str = True,
        labels: list[str] | None = None,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        scale_by_bin_width: bool = False,
        x_axlim: tuple[float, float] | None = None,
        y_axlim: tuple[float, float] | None = None,
        filename: str | Path | None = None,
        cut: bool | str | list[str] = False,
        histtype="fill",
        sort: bool = True,
        ratio_plot: bool = False,
        ratio_axlim: float | tuple[float, float] | None = (0.5, 1.5),
        suffix: str = "",
        prefix: str = "",
        **kwargs,
    ) -> None:
        """
        Plot stacked histograms for datasets/variables.
        If one dataset is passed but multiple variables, will plot overlays of the variable for that one dataset

        :param datasets: string or list of strings corresponding to datasets in the analysis
        :param var: variable name to be plotted. Either a string that exists in all datasets
                    or a list one for each dataset
        :param yerr: Histogram uncertainties. Following modes are supported:
                - True: will calculate sum of histogram stack errors
                - shape(N) array of for one-sided errors or list thereof
                - shape(Nx2) array of for two-sided errors or list thereof
        :param labels: list of labels for plot legend corresponding to each dataset
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param scale_by_bin_width: divide histogram bin values by bin width
        :param x_axlim: x-axis limits. If None matplolib decides
        :param y_axlim: x-axis limits. If None matplolib decides
        :param filename: name of output
        :param cut: applies cuts before plotting
        :param histtype: type of stack plot to be passed to mplhep
        :param sort: whether to sort histograms from lowest to highest integral before plotting.
        :param ratio_plot: add data/mc ratio plot to the bottom of figure
        :param ratio_axlim: pass to yax_lim in rato plotter
        :param suffix: suffix to add at end of histogram/file name
        :param prefix: prefix to add at start of histogram/file
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        """
        self.logger.info(f"Plotting stack plot of {var} in {datasets}")

        # naming template for file/histogram name
        name_template = (
            ((prefix + "_") if prefix else "")  # prefix
            + "{variable}"
            + ("_BIN_SCALED" if scale_by_bin_width else "")
            + "_STACKED"
        )

        if isinstance(datasets, str):
            datasets = [datasets]
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(var, str):
            var = [var]

        if (len(datasets) > 1) and (len(var) > 1) and (len(datasets) != len(var)):
            raise ValueError(
                f"If both multiple datasets and variables passed, lengths must match."
                f"Got {len(datasets)} datasets and {len(var)} variables"
            )

        # handle how cuts are going to be done
        if cut is False or cut is None:
            cutsets_to_loop = [None]
        elif cut is True:
            # separate plot for EACH set of cuts
            if len(datasets) > 1:
                self.__verify_same_cuts(datasets)
            cutsets_to_loop = list(self[datasets[0]].cuts.keys())
        elif isinstance(cut, str):
            cutsets_to_loop = [cut]
        elif isinstance(cut, list):
            cutsets_to_loop = cut
        else:
            raise ValueError(f"Unknown cut {cut}")

        # figure out if we should loop over datasets or variables or both
        _varloop = False
        _datasetloop = False
        if len(datasets) > 1:
            if (len(datasets) != len(var)) and (len(var) > 1):
                raise ValueError(
                    "Number of datasets and variables must match if passing multiple variables."
                )
            _datasetloop = True
            if len(var) > 1:
                _varloop = True
            n_stacks = len(datasets)
        elif len(var) > 1:
            n_stacks = len(var)
            _varloop = True
        else:
            n_stacks = 1

        if n_stacks < 1:
            raise ValueError("Nothing to plot!")

        # plot for each cut
        for cut in cutsets_to_loop:
            # work out histograms to stack
            hist_list: list[Histogram1D] = []
            label_list: list[str | None] = []
            colours_list: list[str | None] = []

            signal_ds = data_ds = ""
            for i in range(n_stacks):
                # save signal for the end if only plotting one variable and many datasets
                if self[datasets[i]].is_signal and _datasetloop:
                    signal_ds = datasets[i]
                    continue
                elif self[datasets[i]].is_data and _datasetloop:
                    data_ds = datasets[i]
                    continue

                dataset = datasets[i] if _datasetloop else datasets[0]
                varname = var[i] if _varloop else var[0]

                label = labels[i] if labels else self[dataset].label
                label_list.append(label)

                colours_list.append(self[dataset].colour)

                hist = self.get_hist(varname, dataset, cut)
                if scale_by_bin_width:
                    hist /= hist.bin_widths

                # check bins
                if len(hist_list) > 1:
                    assert np.allclose(
                        hist.bin_edges, hist_list[-1].bin_edges
                    ), f"Bins {hist} and {hist_list[-1]} not equal!"

                hist_list.append(hist)

            if sort:
                # Sort lists based on integral of histograms so smallest histograms sit at bottom
                all_lists = zip(hist_list, label_list, colours_list)
                sorted_lists = sorted(all_lists, key=lambda l: l[0].integral)
                hist_list, label_list, colours_list = [list(l) for l in zip(*sorted_lists)]

            # plot
            if ratio_plot:
                fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
            else:
                fig, ax = plt.subplots()
                ratio_ax = None  # just so IDE doesn't complain about missing variable

            alpha_list = [0.8] * len(hist_list)
            edgecolour_list = ["k"] * len(hist_list)
            hep.histplot(
                H=[h.bin_values() for h in hist_list],
                bins=hist_list[-1].bin_edges,
                ax=ax,
                color=colours_list if colours_list else None,
                alpha=alpha_list if alpha_list else None,
                edgecolor=edgecolour_list if edgecolour_list else None,
                linewidth=1 if edgecolour_list else 0,
                label=label_list,
                stack=True,
                histtype=histtype,
                zorder=reversed(range(len(hist_list))),  # mplhep plots in wrong order
                **kwargs,
            )

            # handle signal seperately
            if signal_ds:
                bkg_sum = reduce((lambda x, y: x + y), hist_list)
                sig_var = var[datasets.index(signal_ds)] if _varloop else var[0]
                sig_hist = self.get_hist(sig_var, signal_ds, cut)
                if scale_by_bin_width:
                    sig_hist /= sig_hist.bin_widths
                hist_list.append(sig_hist)

                sig_stack = sig_hist + bkg_sum

                sig_stack.plot(ax=ax, yerr=None, color="r", label=self[signal_ds].label)

            edges = hist_list[0].bin_edges
            if yerr:
                # MC error propagation
                errs = np.array([hist.error() for hist in hist_list])
                errs = np.sum(errs, axis=0)

                # top of histogram stack
                stack = np.array([hist.bin_values() for hist in hist_list])
                stack = np.sum(stack, axis=0)
                err_top = stack + (errs / 2)
                err_bottom = stack - (errs / 2)

                # add error as clear hatch
                ax.fill_between(
                    x=edges,
                    y1=np.append(err_top, err_top[-1]),
                    y2=np.append(err_bottom, err_bottom[-1]),
                    alpha=0.3,
                    color="grey",
                    hatch="/",
                    label="MC error",
                    step="post",
                )

            # handle data separately
            if data_ds:
                # figure out which variable we're meant to plot
                if len(var) < 2:
                    varname = var[0]
                else:
                    varname = var[datasets.index(data_ds)]

                # get histogram and plot
                data_hist = self.get_hist(varname, data_ds, cut)
                hist_list.append(data_hist)

                if scale_by_bin_width:
                    data_hist /= data_hist.bin_widths
                ax.errorbar(
                    data_hist.bin_centres,
                    data_hist.bin_values(),
                    xerr=data_hist.bin_widths / 2,
                    yerr=data_hist.error(),
                    linestyle="None",
                    color="black",
                    marker=".",
                    label=self["data"].label,
                )

                # TODO: fix
                if ratio_plot:
                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.1, wspace=0)
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

                    # MC errors
                    if yerr:
                        err_bottom = 1 - errs
                        err_top = 1 + errs
                        ratio_ax.fill_between(
                            x=edges,
                            y1=np.append(err_top, err_top[-1]),
                            y2=np.append(err_bottom, err_bottom[-1]),
                            alpha=0.3,
                            color="grey",
                            hatch="/",
                            label="MC error",
                            step="post",
                        )

                    # add line for MC
                    ratio_ax.hlines(
                        y=1,
                        xmin=edges[0],
                        xmax=edges[-1],
                        colors="r",
                    )

                    # plot data/mc ratio
                    all_mc_hist = reduce((lambda x, y: x.__add__(y)), hist_list)

                    ratio_hist = data_hist.plot_ratio(
                        all_mc_hist,
                        ax=ratio_ax,
                        yerr="binom",
                        label=label,
                        color="k",
                        yax_lim=ratio_axlim,
                        display_unity=False,
                    )

                    plotting_tools.set_axis_options(
                        axis=ratio_ax,
                        var_name=var,
                        xlim=(ratio_hist.bin_edges[0], ratio_hist.bin_edges[-1]),
                        ylim=ratio_axlim,
                        xlabel=xlabel,
                        ylabel="Data / Simulation",
                        title="",
                        logx=logx,
                        logy=False,
                        label=False,
                    )

            plotting_tools.set_axis_options(
                axis=ax,
                var_name=var,
                xlim=(x_axlim if x_axlim else (hist.bin_edges[0], hist.bin_edges[-1])),
                ylim=y_axlim,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                logx=logx,
                logy=logy,
                diff_xs=scale_by_bin_width,
            )
            if x_axlim:
                ax.set_xlim(*x_axlim)
            if y_axlim:
                ax.set_ylim(*y_axlim)

            # limit to 4 rows and reverse order (so more important samples go in front)
            ncols = len(hist_list) + bool(yerr)
            ncols //= 4
            handles, labels_ = ax.get_legend_handles_labels()
            ax.legend(
                reversed(handles), reversed(labels_), fontsize=10, loc="upper right", ncols=ncols
            )

            if filename:
                filepath = self.paths.plot_dir / filename
            else:
                if isinstance(var, str):
                    varname = var
                else:
                    varname = "_".join(var)
                filepath = self.paths.plot_dir / (
                    name_template.format(variable=varname)
                    + (f"_{cut}_cut" if cut else "")
                    + (f"_{suffix}" if suffix else "")
                    + ".png"
                )

            fig.savefig(filepath, bbox_inches="tight")
            self.logger.info(f"Saved plot of {var} to {filepath}")
            plt.close(fig)

    def get_hist(
        self,
        variable: str,
        dataset: str | None = None,
        cut: str | None = None,
        TH1: bool = False,
    ) -> Histogram1D | ROOT.TH1:
        """Get TH1 histogram from histogram dict or internal dataset"""
        # if passing a histogram name directly as the variable
        if variable in self.histograms:
            hist_name_internal = variable

        elif cut and f"{variable}_{cut}_cut" in self.histograms:
            hist_name_internal = f"{variable}_{cut}_cut"

        elif dataset is None:
            raise ValueError(
                f"No variable '{variable}' for cut '{cut}' found in analysis: {self.name}"
            )

        elif cut and f"{variable}_{cut}_cut" in self[dataset].histograms:
            hist_name_internal = f"{dataset}_{variable}_{cut}_cut"

        elif cut:
            raise ValueError(f"No cut {cut} found for {variable} in {dataset}")

        elif variable in self[dataset].histograms:
            hist_name_internal = dataset + "_" + variable

        else:
            raise ValueError(
                f"No histogram for {variable} in {dataset}."
                "\nHistograms in analysis:"
                + "\n".join(self.histograms.keys())
                + "\n Histograms in dataset: "
                + "\n".join(self[dataset].histograms.keys())
            )

        if TH1:
            return self.histograms[hist_name_internal]
        else:
            return Histogram1D(th1=self.histograms[hist_name_internal], logger=self.logger)

    def __verify_same_cuts(self, datasets: list[str]):
        """check that all datasets to be plotted have the same sets of cuts"""
        first_cutflow = self[datasets[0]].cuts
        if not all(ds.cuts == first_cutflow for ds in list(self.datasets.values())[1:]):
            raise ValueError("Datasets do not have the same cuts")
        return True

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @handle_dataset_arg
    def cutflow_printout(self, datasets: str, latex: bool = False) -> None:
        """Prints cutflow table to terminal"""
        self[datasets].cutflow_printout(self.paths.latex_dir if latex else None)

    def full_cutflow_printout(
        self,
        datasets: list[str],
        cutsets: list[str] | str | None = None,
        filename: str | None = None,
    ) -> None:
        """Prints full cutflows for all passed datasets"""

        self.__verify_same_cuts(datasets)

        # for each cut set, create new set of rows in table
        if isinstance(cutsets, str):
            cutsets = [cutsets]
        elif cutsets is None:
            cutsets = list(self[datasets[0]].cuts)

        # table build loop
        latex_str = r"\begin{tabular}{" + "l" * (len(datasets) + 1) + "}\n"

        for cutset in cutsets:
            sanitised_str = cutset.replace(r"_", r"\_")
            # header
            latex_str += r"\hline" + "\n"
            latex_str += (
                " & ".join(
                    [f"Cut ({sanitised_str})"] + [self[dataset].label for dataset in datasets]
                )
                + r"\\"
                + "\n"
            )
            latex_str += r"\hline" + "\n"

            cut_names = [
                cutflow_item.cut.name for cutflow_item in self[datasets[0]].cutflows[cutset]
            ]
            for i, cut_name in enumerate(cut_names):
                passes_list = [
                    str(int(self[dataset].cutflows[cutset][i].npass)) for dataset in datasets
                ]
                latex_str += cut_name + " & " + " & ".join(passes_list) + r"\\" + "\n"

        latex_str += r"\hline" + "\n" + r"\end{tabular}"

        # print to file
        if not filename:
            filename = self.paths.latex_dir / f"{self.name}_full_cutflows.tex"

        with open(filename, "w") as f:
            f.write(latex_str)

    def save_histograms(
        self,
        filename: str | Path | None = None,
        tfile_option: str = "Update",
        write_option: str = "Overwrite",
        clear_hists: bool = False,
    ) -> None:
        """
        Saves current histograms into root file

        :param filename: Should end in '.root'. if not given will set as '<analysis_name>_histograms.root'
        :param tfile_option: TFile option.
                             See: https://root.cern.ch/doc/master/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
        :param write_option: WriteObject() option.
                             See: https://root.cern.ch/doc/master/classTDirectoryFile.html#ae1bb32dcbb69de7f06a3b5de9d22e852
        :param clear_hists: clears histograms in dictionary
        """
        if not filename:
            filename = self._output_dir / f"{self.name}_histograms.root"

        self.logger.info(f"Saving {len(self.histograms)} histograms to file {filename}...")
        with ROOT_utils.ROOT_TFile_mgr(str(filename), tfile_option) as file:
            for name, histo in self.histograms.items():
                file.WriteObject(histo, name, write_option)

        if clear_hists:
            self.histograms = dict()

    def histogram_printout(self, to_latex: bool = False) -> None:
        """Printout of histogram metadata"""
        rows = []
        header = ["Hist name", "Entries", "Bin sum", "Integral"]

        for name, h in self.histograms.items():
            rows.append([name, h.GetEntries(), h.Integral(), h.Integral("width")])

        if not to_latex:
            self.logger.info(tabulate(rows, headers=header))
        else:
            filepath = self.paths.latex_dir / f"{self.name}_histograms.tex"
            with open(filepath, "w") as f:
                f.write(tabulate(rows, headers=header, tablefmt="latex_raw"))
                self.logger.info(f"Saved LaTeX histogram table to {filepath}")
