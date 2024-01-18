import inspect
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
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
from utils.context import check_single_dataset, handle_dataset_arg, redirect_stdout


@dataclass(slots=True)
class AnalysisPath:
    """Container class for paths needed by analyses"""

    plot_dir: Path
    pkl_dir: Path
    latex_dir: Path
    log_dir: Path

    def create_paths(self):
        for p in (
            self.plot_dir,
            self.pkl_dir,
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
        data_dir: Path | str | None = None,
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
        :param data_dir: Root directory for pickle data in/out
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
        self.histograms: OrderedDict[str, ROOT.TH1] = OrderedDict()

        # SET OUTPUT DIRECTORIES
        # ===========================
        if not output_dir:
            # root in the directory above this one
            output_dir = Path(__file__).absolute().parent.parent
        self._output_dir = Path(output_dir) / "outputs" / analysis_label  # where outputs go
        self.paths = AnalysisPath(
            plot_dir=Path(self._output_dir) / "plots",
            pkl_dir=Path(data_dir if data_dir else (self._output_dir / "pickles")),
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
            except KeyError:
                raise KeyError(
                    f"Unknown data-year: {kwargs['year']}. Known data-years: {lumi_year.keys()}"
                )
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

            # set correct pickle path if not passed as a build argument
            if "pkl_path" not in args:
                args["pkl_path"] = self.paths.pkl_dir / (dataset_name + "_df.pkl")

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

            if "binnings" in args:
                dataset.binnings = args["binnings"]

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
                if self[merged_ds].cutflows is not None:
                    for cutflow in self[merged_ds].cutflows:
                        self[merged_ds].cutflows[cutflow] += dataset.cutflow
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

    def merge_datasets(
        self,
        *datasets: str,
        new_name: str | None = None,
        delete: bool = True,
        to_pkl: bool = False,
        verify: bool = False,
        delete_file: bool = False,
        sort: bool = True,
    ) -> None:
        """
        Merge datasets by concatenating one or more into the other

        :param datasets: strings of datasets to merge. First dataset will be merged into.
        :param apply_cuts: True to apply all cuts to datasets before merging or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param new_name: new name to given to merged dataset
        :param delete: whether to delete datasets internally
        :param to_pkl: whether to print new dataset to a pickle file (will replace original pickle file)
        :param verify: whether to check for duplicated events
        :param delete_file: whether to delete pickle files of merged datasets (not the one that is merged into)
        :param sort: whether to sort output dataset
        """
        for n in datasets:
            if n not in self.datasets:
                raise ValueError(f"No dataset named {n} found in analysis {self.name}")

        # check for columns missing from any dataset
        first_cols = set(self[datasets[0]].df.columns)
        for d in datasets[0:]:
            curr_cols = set(self[d].df.columns)
            if first_cols ^ curr_cols:
                if cols_missing_from_first := first_cols - curr_cols:
                    self.logger.warning(
                        f"Missing column(s) from dataset '{d}' but are in '{datasets[0]}': "
                        f"{cols_missing_from_first}"
                    )
                if cols_missing_from_curr := curr_cols - first_cols:
                    self.logger.warning(
                        f"Missing column(s) from dataset '{datasets[0]}' but are in '{d}': "
                        f"{cols_missing_from_curr}"
                    )

        # check duplicate indices
        for d in datasets:
            if self[d].df.index.duplicated().sum() > 0:
                self.logger.warning(f"Duplicate indexes in dataset {d}!")

        self.logger.info(f"Merging dataset(s) {datasets[1:]} into dataset {datasets[0]}...")

        try:
            self[datasets[0]].df = pd.concat(
                [self[dataset].df for dataset in datasets],
                verify_integrity=verify,
                copy=False,
            )
        except pd.errors.InvalidIndexError:
            err_str = ""
            for dataset in datasets:
                common_index = self[datasets[0]].df.index.intersection(self[dataset].df.index)
                err_str += f"Index common between {datasets[0]} and {dataset}: {common_index}\n"
            raise pd.errors.InvalidIndexError(err_str)

        self[datasets[0]].name = datasets[0]

        if sort:
            self[datasets[0]].df.sort_index(level="DSID", inplace=True)

        if new_name:
            self[new_name] = self.datasets.pop(datasets[0])

        for n in datasets[1:]:
            if delete:
                self.__delete_dataset(n)
            if delete_file:
                self.__delete_dataset_file(n)

        if to_pkl:
            pd.to_pickle(self[datasets[0]].df, self[datasets[0]].file)
            self.logger.info(f"Saved merged dataset to file {self[datasets[0]].file}")

    @check_single_dataset
    def __delete_dataset(self, ds_name: str) -> None:
        self.logger.info(f"Deleting dataset {ds_name} from analysis {self.name}")
        del self[ds_name]

    @check_single_dataset
    def __delete_dataset_file(self, ds_name: str) -> None:
        self.logger.info(f"Deleting pickled dataset {ds_name}")
        self[ds_name].file.unlink()

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_hist(
        self,
        datasets: str | Sequence[str],
        var: str | Sequence[str],
        bins: list[float | int] | tuple[int, float, float] | None = None,
        weight: list[str | float] | str | float = 1.0,
        yerr: ArrayLike | str = True,
        labels: list[str] | None = None,
        w2: bool = False,
        normalise: float | bool = False,
        logbins: bool = False,
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
                    or a list one for each dataset
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param weight: variable name in dataset to weight by or numeric value to weight all
                       can pass list for separate weights to
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
        :param logbins: whether logarithmic binnings
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
        self.logger.info(f"Plotting {var} in {datasets}...")

        # naming template for file/histogram name
        name_template = (
            ((prefix + "_") if prefix else "")  # prefix
            + "{short}"
            + ("_NORMED" if normalise else "")  # normalisation flag
            + (("_" + suffix) if suffix else "")  # suffix
        )
        name_template_short = (
            "{dataset}" + "_{variable}"  # name of dataset(s)  # name of variable(s)
        )

        if isinstance(datasets, str):
            datasets = [datasets]
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(var, str):
            var = [var]

        # handle how cuts are going to be done
        if cut is False or cut is None:
            cutsets_to_loop = [False]
        elif cut is True:
            # separate plot for EACH set of cuts
            if len(datasets) > 1:
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
        if len(datasets) > 1:
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
                dataset = datasets[i] if datasetloop else datasets[0]
                varname = var[i] if varloop else var[0]

                label = labels[i] if labels else self[dataset].label
                hist_name = name_template.format(
                    short=name_template_short.format(dataset=dataset, variable=varname)
                )

                # if passing a histogram name directly as the variable
                if cut and f"{varname}_{cut}_cut" in self.histograms:
                    hist_name_internal = f"{varname}_{cut}_cut"
                elif cut and f"{varname}_{cut}_cut" in self[dataset].histograms:
                    hist_name_internal = f"{dataset}_{varname}_{cut}_cut"
                elif cut:
                    raise ValueError(f"No cut {cut} found for {varname} in {dataset}")

                elif varname in self.histograms:
                    hist_name_internal = varname
                elif varname in self[dataset].histograms:
                    hist_name_internal = dataset + "_" + varname
                elif name_template_short in self.histograms:
                    hist_name_internal = varname
                else:
                    hist_name_internal = None

                # plot
                if hist_name_internal:
                    hist = Histogram1D(th1=self.histograms[hist_name_internal], logger=self.logger)
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
                else:
                    self.logger.warning(
                        f"WARNING: Histogram '{varname}' not found. Will try to generate."
                    )
                    if bins is None:
                        raise ValueError("Must provide bins if histogram is not yet generated.")

                    hist = self[dataset].plot_hist(
                        var=varname,
                        bins=bins,
                        weight=weight[i] if isinstance(weight, list) else weight,
                        ax=ax,
                        yerr=yerr,
                        normalise=normalise,
                        logbins=logbins,
                        name=hist_name,
                        label=None if n_overlays == 1 else label,
                        w2=w2,
                        stats_box=stats_box,
                        scale_by_bin_width=scale_by_bin_width,
                        cut=cut,
                        **kwargs,
                    )

                # save
                hists.append(hist)
                self.histograms[hist_name] = hist.TH1

                if ratio_plot and len(hists) > 1:
                    # ratio of first dataset to this one
                    label = (
                        f"{labels[-1]}/{labels[0]}"
                        if labels
                        else f"{self[dataset].label}/{self[datasets[0]].label}"
                    )
                    # match ratio colour to plot
                    color = ax.get_lines()[-1].get_color() if (n_overlays > 2) else "k"
                    ratio_hist_name = (
                        name_template.format(
                            short=name_template_short.format(
                                dataset=f"{dataset}_{datasets[0]}", variable=varname
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

                if len(datasets) > 2:  # don't show legend if there's only two datasets
                    ratio_ax.legend(fontsize=10, loc=1)

                plotting_tools.set_axis_options(
                    axis=ratio_ax,
                    var_name=var,
                    lepton=lepton,
                    xlim=(ratio_hist.bin_edges[0], ratio_hist.bin_edges[-1]),
                    diff_xs=scale_by_bin_width,
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
                else:
                    varname = "_".join(var)
                filepath = self.paths.plot_dir / (
                    name_template.format(
                        short=name_template_short.format(
                            dataset="_".join(datasets), variable=varname
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
        data: bool = False,
        yerr: ArrayLike | str = True,
        labels: list[str] | None = None,
        normalise: float | bool = False,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        scale_by_bin_width: bool = False,
        x_axlim: tuple[float, float] | None = None,
        y_axlim: tuple[float, float] | None = None,
        filename: str | Path | None = None,
        cut: bool = False,
        histtype="fill",
        sort: bool = True,
        suffix: str = "",
        prefix: str = "",
        **kwargs,
    ) -> None:
        self.logger.info(f"Plotting stack plot of {var} in {datasets}")

        if data and "data" not in self.datasets:
            raise ValueError("No 'data' dataset found in analysis")

        # naming template for file/histogram name
        name_template = (
            ((prefix + "_") if prefix else "")  # prefix
            + "{short}"
            + ("_NORMED" if normalise else "")  # normalisation flag
            + ("_BIN_SCALED" if scale_by_bin_width else "")
            + "_STACKED"
            + (("_" + suffix) if suffix else "")  # suffix
        )
        name_template_short = (
            "{dataset}"  # name of dataset(s)
            + "_{variable}"  # name of variable(s)
            + ("_cut" if cut else "")  # cut flag
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
            cutsets_to_loop = [False]
        elif cut is True:
            # separate plot for EACH set of cuts
            if len(datasets) > 1:
                # check that all datasets to be plot have the same sets of cuts
                first_cutflow = self[datasets[0]].cuts
                if not all(ds.cuts == first_cutflow for ds in list(self.datasets.values())[1:]):
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
        if len(datasets) > 1:
            if (len(datasets) != len(var)) and (len(var) > 1):
                raise ValueError(
                    "Number of datasets and variables must match if passing multiple variables."
                )
            datasetloop = True
            if len(var) > 1:
                varloop = True
            n_stacks = len(datasets)
        elif len(var) > 1:
            n_stacks = len(var)
            varloop = True
        else:
            n_stacks = 1

        # do the datasets have defined colours?
        is_colours = all(self[ds].colour != "" for ds in datasets)
        if not is_colours:
            self.logger.debug("Colours aren't specified for all datasets, will generate.")

        # plot for each cut
        for cut in cutsets_to_loop:
            # work out histograms to stack
            hist_list = []
            label_list = []
            err_list = []
            colours_list = []

            for i in range(n_stacks):
                dataset = datasets[i] if datasetloop else datasets[0]
                varname = var[i] if varloop else var[0]

                label = labels[i] if labels else self[dataset].label
                label_list.append(label)

                if is_colours:
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
                err_list.append(hist.error())

            if sort:
                # Sort lists based on integral of histograms so smallest histograms sit at bottom
                all_lists = zip(hist_list, err_list, label_list, colours_list)
                sorted_lists = sorted(all_lists, key=lambda l: l[0].integral)
                hist_list, err_list, label_list, colours_list = zip(*sorted_lists)

            # plot
            fig, ax = plt.subplots()
            hep.histplot(
                H=[h.bin_values() for h in hist_list],
                bins=hist_list[-1].bin_edges,
                ax=ax,
                yerr=err_list if yerr is True else yerr,
                color=colours_list if colours_list else None,
                label=label_list,
                stack=True,
                histtype=histtype,
                **kwargs,
            )

            if data:
                hist = self.get_hist(varname, "data", cut)
                if scale_by_bin_width:
                    hist /= hist.bin_widths
                ax.errorbar(
                    hist.bin_centres,
                    hist.bin_values(),
                    xerr=hist.bin_widths / 2,
                    yerr=hist.error(),
                    linestyle="None",
                    color="black",
                    marker=".",
                    label=self["data"].label,
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
            ax.legend(fontsize=10, loc="upper right")

            if filename:
                filepath = self.paths.plot_dir / filename
            else:
                if isinstance(var, str):
                    varname = var
                else:
                    varname = "_".join(var)
                filepath = self.paths.plot_dir / (
                    name_template.format(
                        short=name_template_short.format(
                            dataset="_".join(datasets), variable=varname
                        )
                    )
                    + (f"{cut}_cut" if cut else "")
                    + ".png"
                )

            fig.savefig(filepath, bbox_inches="tight")
            self.logger.info(f"Saved plot of {var} to {filepath}")
            plt.close(fig)

    def get_hist(self, variable: str, dataset: str, cut: str) -> Histogram1D:
        # if passing a histogram name directly as the variable
        if cut and f"{variable}_{cut}_cut" in self.histograms:
            hist_name_internal = f"{variable}_{cut}_cut"
        elif cut and f"{variable}_{cut}_cut" in self[dataset].histograms:
            hist_name_internal = f"{dataset}_{variable}_{cut}_cut"
        elif cut:
            raise ValueError(f"No cut {cut} found for {variable} in {dataset}")

        elif variable in self.histograms:
            hist_name_internal = variable
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

        return Histogram1D(th1=self.histograms[hist_name_internal], logger=self.logger)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @handle_dataset_arg
    def cutflow_printout(self, datasets: str, latex: bool = False) -> None:
        """Prints cutflow table to terminal"""
        self[datasets].cutflow_printout(self.paths.latex_dir if latex else None)

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
            self.histograms = OrderedDict()

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
