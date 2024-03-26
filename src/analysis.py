import inspect
import os
from dataclasses import dataclass, fields
from functools import reduce
from itertools import islice
from pathlib import Path
from typing import Callable, Any, Sequence, Generator

import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from numpy.typing import ArrayLike
from tabulate import tabulate

from src.dataset import Dataset
from src.datasetbuilder import DatasetBuilder, lumi_year
from src.dsid_meta import DatasetMetadata
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools
from utils.context import handle_dataset_arg


@dataclass(slots=True)
class FakesOpts:
    """
    Options for basic fakes estimation

    :param fakes_source_var: variable from which to calculate fake factors from
    :param apply_fakes_to: fakes histograms will be calculated in these variables from source
    :param CR_passID_data: selection name for control region pass-ID for data (must exist in analysis)
    :param CR_failID_data: selection name for control region fail-ID for data (must exist in analysis)
    :param CR_failID_data: selection name for signal region fail-ID for data (must exist in analysis)
    :param CR_passID_mc: selection name for control region pass-ID for "true" tau MC (must exist in analysis)
    :param CR_failID_mc: selection name for control region fail-ID for "true" tau MC (must exist in analysis)
    :param CR_failID_mc: selection name for signal region fail-ID for "true" tau MC (must exist in analysis)
    """

    fakes_source_var: str
    apply_fakes_to: set[str]
    CR_passID_data: str = "CR_passID"
    CR_failID_data: str = "CR_failID"
    SR_failID_data: str = "SR_failID"
    CR_passID_mc: str = "CR_passID_trueTau"
    CR_failID_mc: str = "CR_failID_trueTau"
    SR_failID_mc: str = "SR_failID_trueTau"


@dataclass(slots=True)
class AnalysisPath:
    """
    Container class for paths needed by analyses

    :param plot_dir: directory to save plots to
    :param latex_dir: directory to save latex tables to
    :param root_dir: directory to save root files to
    :param log_dir: directory to save log files to
    """

    plot_dir: Path
    latex_dir: Path
    root_dir: Path
    log_dir: Path

    def create_paths(self):
        """Create paths needed by analyses"""
        for field in fields(self):
            if not issubclass(field.type, Path):
                raise ValueError(f"Non-Path attribute in {self}!")
            getattr(self, field.name).mkdir(parents=True, exist_ok=True)


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
        "year",
        "metadata",
        "mc_samples",
        "data_sample",
        "signal_sample",
        "binnings",
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
        regen_metadata: bool = False,
        snapshot: bool = True,
        year: int = 2017,
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
        :param regen_metadata: Whether to regenerate DSID metadata (requires connection to pyami)
        :param snapshot: Whether to save a snapshot of datasets to disk
        :param year: Data-year. One of 2016, 2017, 2018
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
            root_dir=Path(self._output_dir) / "root",
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
        self.year = year
        if self.year:
            try:
                self.global_lumi = lumi_year[self.year]
            except KeyError as e:
                raise KeyError(
                    f"Unknown data-year: {self.year}. Known data-years: {list(lumi_year.keys())}"
                ) from e
        else:
            self.global_lumi = global_lumi
        self.logger.debug(f"Set global luminosity scale to {self.global_lumi} pb-1")

        # HANDLE METADATA
        # ============================
        self.metadata = DatasetMetadata(logger=self.logger)
        dsid_metadata_cache = self._output_dir / "dsid_meta_cache.json"

        if (not dsid_metadata_cache.is_file()) or regen_metadata:
            # fetch and save metadata
            if "ttree" in kwargs:
                self.metadata.fetch_metadata(
                    datasets=data_dict, ttree=kwargs["ttree"], data_year=self.year
                )
            else:
                self.metadata.fetch_metadata(datasets=data_dict, data_year=self.year)

            self.metadata.save_metadata(dsid_metadata_cache)
            self.logger.debug(f"Saved metadata cache in %s", dsid_metadata_cache)

        else:
            # load metadata
            self.metadata.read_metadata(dsid_metadata_cache)
            self.logger.debug(f"Loaded metadata cache from %s", dsid_metadata_cache)

        # create c++ maps for calculation of weights in datasets
        sumws = [(dsid, meta.sumw) for (dsid, meta) in self.metadata]
        pmgfs = [
            (dsid, meta.cross_section * meta.kfactor * meta.filter_eff)
            for (dsid, meta) in self.metadata
        ]
        ROOT.gInterpreter.Declare(
            f"""
                std::map<int, float> dsid_sumw{{{','.join(f'{{{dsid}, {sumw}}}' for (dsid, sumw) in sumws)}}};
                std::map<int, float> dsid_pmgf{{{','.join(f'{{{dsid}, {pmgf}}}' for (dsid, pmgf) in pmgfs)}}};
            """
        )
        self.logger.debug("Declared metadata maps in ROOT")

        # BUILD DATASETS
        # ===============================
        self.mc_samples: list[str] = []
        self.data_sample: str = ""
        self.signal_sample: str = ""
        self.datasets: dict[str, Dataset] = dict()
        for dataset_name, data_args in data_dict.items():
            self.logger.info("")
            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info(f"======== INITIALISING DATASET '{dataset_name}' =========")
            self.logger.info("=" * (42 + len(dataset_name)))

            if "is_data" in data_args:
                if self.data_sample != "":
                    raise ValueError("Can't have more than one data sample!")
                self.data_sample = dataset_name
            elif "is_signal" in data_args:
                if self.signal_sample != "":
                    raise ValueError("Can't have more than one signal sample!")
                self.signal_sample = dataset_name
                self.mc_samples.append(dataset_name)
            else:
                self.mc_samples.append(dataset_name)

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
                **self._match_params(args, DatasetBuilder.__init__),
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
            dataset = builder.build(**self._match_params(args, DatasetBuilder.build))

            # apply some manual settings
            for manual_setting in [
                "binnings",
                "profiles",
            ]:
                if manual_setting in args:
                    dataset.__setattr__(manual_setting, args[manual_setting])

            if separate_loggers:
                # set new logger to append to analysis logger
                dataset.logger = self.logger
                dataset.logger.debug(f"{dataset_name} log handler returned to analysis.")  # test

            # load/gen histograms
            histogram_file = self.paths.root_dir / f"{dataset_name}_histograms.root"
            if not indiv_regen_hists and histogram_file.exists():
                # just read in previous histogram file if it exists
                dataset.import_histograms(histogram_file)
                dataset.import_dataframes(self.paths.root_dir / f"{self.name}.root")
                dataset.reset_cutflows()
            else:
                dataset.gen_histograms(to_file=histogram_file)

            # integrate into own histogram dictionary
            for hist_name, hist in dataset.histograms.items():
                self.histograms[dataset_name + "_" + hist_name] = hist

            self[dataset_name] = dataset  # save to analysis

            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info(f"========= DATASET '{dataset_name}' INITIALISED =========")
            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info("")

        self.logger.info("=" * (len(analysis_label) + 23))
        self.logger.info(f"ANALYSIS '{analysis_label}' INITIALISED")

        if snapshot and regen_histograms:
            self.snapshot(self.paths.root_dir / f"{self.name}.root")

    @staticmethod
    def _match_params(params: dict[str, Any], func: Callable) -> dict[str, Any]:
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
    def plot(
        self,
        var: str | Histogram1D | Sequence[str | Histogram1D],
        datasets: str | Sequence[str | None] | None = None,
        labels: list[str] | None = None,
        colours: list[str] | None = None,
        yerr: ArrayLike | str = True,
        logx: bool = False,
        logy: bool = True,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        scale_by_bin_width: bool = False,
        stats_box: bool = False,
        x_axlim: tuple[float, float] | None = None,
        y_axlim: tuple[float, float] | None = None,
        ratio_plot: bool = False,
        ratio_fit: bool = False,
        ratio_axlim: float | tuple[float, float] | None = None,
        ratio_label: str = "Ratio",
        ratio_err: str = "sumw2",
        filename: str | Path | None = None,
        sort: bool = True,
        cut: bool | str | Sequence[str] = False,
        kind: str = "overlay",
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
        :param labels: list of labels for plot legend corresponding to each line
        :param colours: list of colours for histograms
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
                     - shape(N) array of for one-sided errors or list thereof
                     - shape(Nx2) array of for two-sided errors or list thereof
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param scale_by_bin_width: divide histogram bin values by bin width
        :param stats_box: display stats box
        :param x_axlim: x-axis limits. If None matplolib decides
        :param y_axlim: x-axis limits. If None matplolib decides
        :param ratio_plot: If True, adds ratio of the first plot with each subseqent plot below
        :param ratio_fit: If True, fits ratio plot to a 0-degree polynomial and display line, chi-square and p-value
        :param ratio_axlim: pass to yax_lim in rato plotter
        :param ratio_label: y-axis label for ratio plot
        :param ratio_err: yerr for ratio plot. Either "sumw2", "binom", or "carry"
        :param filename: name of output
        :param sort: sort stacks by size so smallest histogram is at the bottom
        :param cut: applies cuts before plotting
        :param kind: "overlay" for overlays of line histograms, "stack" for stacks
        :param suffix: suffix to add at end of histogram/file name
        :param prefix: prefix to add at start of histogram/file
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        """

        # PREAMBLE
        # ============================
        # check options
        if kind not in {"stack", "overlay"}:
            raise ValueError("Histogram types are either either 'stack' or 'overlay'.")
        do_stack = True if kind == "stack" else False

        # normalise inputs that could be single strings
        if not isinstance(var, (list, tuple)):
            var = [var]
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]

        # check length of per-histogram inputs
        opt_lens = [
            len(opt) for opt in [var, datasets, labels, colours] if opt is not None and len(opt) > 1
        ]
        if len(set(opt_lens)) > 1:
            raise ValueError(
                f"Lengths for options passed per histogram must all be of the same length if not None. Got:\n"
                + f"var: {len(var)}\n"
                + (f"datasets: {len(datasets)}\n" if datasets else "")
                + (f"labels: {len(labels)}\n" if labels else "")
                + (f"colours: {len(colours)}\n" if colours else "")
            )

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
            n_hists = len(datasets)
        elif len(var) > 1:
            n_hists = len(var)
            _varloop = True
        else:
            n_hists = 1
        if n_hists == 0:
            raise Exception("Nothing to plot!")

        # handle how many plots there are actually going to be
        if cut is False or cut is None:
            selections_to_loop: list[str | None] = [None]
        elif cut is True:
            # separate plot for EACH set of cuts
            if _datasetloop:
                # check that all datasets to be plot have the same sets of cuts
                first_cutflow = self[datasets[0]].selections.keys()
                if not all(ds.selections.keys() == first_cutflow for ds in self.datasets.values()):
                    raise ValueError("Datasets do not have the same cuts")
            selections_to_loop = list(self[datasets[0]].selections.keys())
        elif isinstance(cut, str):
            selections_to_loop = [cut]
        elif isinstance(cut, list):
            selections_to_loop = cut
        else:
            raise ValueError(f"I don't know what you mean by cut '{cut}'")

        # handle which labels and colours are to be used
        if not labels:
            if n_hists > 1:
                try:
                    labels = [self[ds].label for ds in datasets]
                except KeyError:
                    raise ValueError(
                        "Labels must be passed manually if a histogram is passed with no dataset"
                    )
            else:
                labels = [labels]
        if not colours:
            if _datasetloop:
                colours = [self[ds].colour for ds in datasets]
            else:
                # just use default prop cycle colours in matplotib
                c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                colours = list(islice(c_iter, n_hists))

        # no options that depend on multiple histograms
        if n_hists == 1:
            if ratio_plot:
                self.logger.warning("Cannot generate ratio plot for plot with only one histogram!")
            ratio_plot = False
        if n_hists > 2:
            if stats_box:
                self.logger.warning("Not enough space to display stats box. Will not display.")
            stats_box = False

        # PER-SELECTION LOOP
        # ============================
        for selection in selections_to_loop:
            if ratio_plot:
                fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
            else:
                fig, ax = plt.subplots()

            hist_list: list[Histogram1D] = []
            label_list: list[str | None] = []
            colours_list: list[str | None] = []

            for i in range(n_hists):
                variable = var[i] if _varloop else var[0]
                dataset = datasets[i] if _datasetloop else datasets[0]

                if do_stack and (dataset is not None):
                    # save signal & data for the end if only stacking one variable and many datasets
                    if _datasetloop and (datasets[i] in (self.data_sample, self.signal_sample)):
                        continue

                # save options for this histogram
                if isinstance(variable, Histogram1D):
                    hist = variable
                elif isinstance(variable, ROOT.TH1):
                    hist = Histogram1D(th1=variable)
                else:
                    hist = self.get_hist(variable, dataset, selection)
                if scale_by_bin_width:
                    hist /= hist.bin_widths
                hist_list.append(hist)
                label_list.append(labels[i])
                colours_list.append(colours[i])

                # check bins
                if len(hist_list) > 1:
                    assert np.allclose(
                        hist.bin_edges, hist_list[-1].bin_edges
                    ), f"Bins {hist} and {hist_list[-1]} not equal!"

            bin_range = (hist_list[0].bin_edges[0], hist_list[0].bin_edges[-1])

            # STACK PLOT
            # ============================
            if do_stack:
                # Sort lists based on integral of histograms so smallest histograms sit at bottom
                if sort:
                    all_lists = zip(hist_list, label_list, colours_list)
                    sorted_lists = sorted(all_lists, key=lambda ls: ls[0].integral)
                    hist_list, label_list, colours_list = [list(ls) for ls in zip(*sorted_lists)]

                alpha_list = [0.8] * len(hist_list)
                edgecolour_list = ["k"] * len(hist_list)
                hep.histplot(
                    H=[h.bin_values() for h in hist_list],
                    bins=hist_list[-1].bin_edges,
                    ax=ax,
                    color=colours_list,
                    alpha=alpha_list if alpha_list else None,
                    edgecolor=edgecolour_list if edgecolour_list else None,
                    linewidth=1 if edgecolour_list else 0,
                    label=label_list,
                    stack=True,
                    histtype="fill",
                    zorder=reversed(range(len(hist_list))),  # mplhep plots in wrong order
                    **kwargs,
                )

                # handle signal seperately
                if self.signal_sample in datasets:
                    bkg_sum = reduce((lambda x, y: x + y), hist_list)
                    sig_var = var[datasets.index(self.signal_sample)] if _varloop else var[0]
                    sig_hist = self.get_hist(sig_var, self.signal_sample, selection)
                    if scale_by_bin_width:
                        sig_hist /= sig_hist.bin_widths
                    hist_list.append(sig_hist)
                    sig_stack = sig_hist + bkg_sum
                    sig_stack.plot(
                        ax=ax, yerr=None, color="r", label=self[self.signal_sample].label
                    )

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
                if self.data_sample in datasets:
                    # figure out which variable we're meant to plot
                    if len(var) < 2:
                        varname = var[0]
                    else:
                        varname = var[datasets.index(self.data_sample)]

                    # get histogram and plot
                    data_hist = self.get_hist(varname, self.data_sample, selection)

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

                if ratio_plot:
                    if not self.data_sample:
                        raise ValueError("Ratio of what?")

                    fig.tight_layout()
                    fig.subplots_adjust(hspace=0.1, wspace=0)
                    all_mc_hist = reduce((lambda x, y: x + y), hist_list)
                    all_mc_bin_vals = all_mc_hist.bin_values()

                    # MC errors
                    if yerr:
                        err_bottom = (all_mc_bin_vals - errs) / all_mc_bin_vals
                        err_top = (all_mc_bin_vals + errs) / all_mc_bin_vals
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

                    all_mc_hist.plot_ratio(
                        data_hist,
                        ax=ratio_ax,
                        yerr=True,
                        color="k",
                        yax_lim=ratio_axlim,
                        display_unity=False,
                    )
                    plotting_tools.set_axis_options(
                        axis=ratio_ax,
                        var_name=var,
                        xlim=bin_range,
                        xlabel=xlabel,
                        ylabel="Data / MC",
                        logx=not ratio_plot,
                        label=False,
                    )

            # OVERLAY PLOT
            # ============================
            else:  # overlays
                for i, hist in enumerate(hist_list):
                    hist.plot(
                        ax=ax,
                        yerr=yerr,
                        stats_box=stats_box,
                        scale_by_bin_width=scale_by_bin_width,
                        label=label_list[i],
                        color=colours_list[i],
                        **kwargs,
                    )

                    if ratio_plot and (len(hist_list) > 1) and (i > 0):
                        # ratio of first histogram to this one
                        hist_list[0].plot_ratio(
                            hist,
                            ax=ratio_ax,
                            yerr=ratio_err,
                            label=f"{label_list[i]}/{label_list[0]}",
                            color=colours_list[i],
                            fit=ratio_fit,
                            yax_lim=ratio_axlim,
                            display_stats=len(hist_list) <= 3,
                        )

            # CLEANUP
            # ============================
            if n_hists > 1:
                # limit to 4 rows and reverse order (so more important samples go in front)
                ncols = len(hist_list) + bool(yerr) + bool(self.data_sample in datasets)
                ncols = max(ncols // 4, 1)  # need at least one column!
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                ax.legend(
                    reversed(legend_handles),
                    reversed(legend_labels),
                    fontsize=10,
                    loc="upper right",
                    ncols=ncols,
                )
            plotting_tools.set_axis_options(
                axis=ax,
                var_name=var,
                xlim=bin_range,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                logx=not ratio_plot,
                logy=logy,
                diff_xs=scale_by_bin_width,
            )
            if x_axlim:
                ax.set_xlim(*x_axlim)
            if y_axlim:
                ax.set_ylim(*y_axlim)
            if ratio_plot:
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.1, wspace=0)

                if n_hists > 2:  # don't show legend if there's only two plots
                    ratio_ax.legend(fontsize=10, loc=1)

                plotting_tools.set_axis_options(
                    axis=ratio_ax,
                    var_name=var,
                    xlim=bin_range,
                    ylim=ratio_axlim,
                    xlabel=xlabel,
                    ylabel=ratio_label,
                    label=False,
                    logx=logx,
                )
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if filename:
                filepath = self.paths.plot_dir / filename
            else:
                # naming template for file/histogram name
                filename_template = (
                    (f"{prefix}_" if prefix else "")
                    + "_".join(var)
                    + (
                        ("_" + "_".join([ds for ds in datasets if ds is not None]))
                        if datasets
                        else ""
                    )
                    + ("_BIN_SCALED" if scale_by_bin_width else "")
                    + ("_STACKED" if kind == "stack" else "")
                    + (f"_{selection}_cut" if cut else "")
                    + (f"_{suffix}" if suffix else "")
                )
                filepath = self.paths.plot_dir / (filename_template + ".png")

            fig.savefig(filepath, bbox_inches="tight")
            self.logger.info(f"Saved plot to {filepath}")
            plt.close(fig)

    def get_hist(
        self,
        variable: str,
        dataset: str | None = None,
        selection: str | None | bool = None,
        TH1: bool = False,
    ) -> Histogram1D | ROOT.TH1:
        """Get TH1 histogram from histogram dict or internal dataset"""
        hist_name_internal = self.get_hist_name(variable, dataset, selection)
        if TH1:
            return self.histograms[hist_name_internal]
        else:
            return Histogram1D(th1=self.histograms[hist_name_internal], logger=self.logger)

    def get_hist_name(
        self,
        variable: str,
        dataset: str | None = None,
        selection: str | None | bool = None,
    ) -> str:
        """Get name of histogram saved in histogram dict"""
        if variable in self.histograms:
            return variable

        elif selection and f"{variable}_{selection}_cut" in self.histograms:
            return f"{variable}_{selection}_cut"

        elif dataset is None:
            raise ValueError(
                f"No variable '{variable}' for selection '{selection}' found in analysis: {self.name}"
            )

        elif selection and f"{variable}_{selection}_cut" in self[dataset].histograms:
            return f"{dataset}_{variable}_{selection}_cut"

        elif selection:
            raise ValueError(f"No selection {selection} found for {variable} in {dataset}")

        elif variable in self[dataset].histograms:
            return dataset + "_" + variable

        else:
            raise ValueError(
                f"No histogram for {variable} in {dataset}."
                "\nHistograms in analysis:"
                + "\n".join(self.histograms.keys())
                + "\n Histograms in dataset: "
                + "\n".join(self[dataset].histograms.keys())
            )

    def sum_hists(self, hists: list[str], inplace_name: str | None = None) -> ROOT.TH1 | None:
        """
        Sum together internal histograms.
        Optionally pass inplace_name to save automatically to internal histogram dictionary
        """
        h = self.histograms[hists[0]].Clone()
        for hist_name in hists[1:]:
            hist = self.histograms[hist_name]
            h.Add(hist)

        if inplace_name:
            self.histograms[inplace_name] = h
        return h

    def __verify_same_cuts(self, datasets: list[str]):
        """check that all datasets to be plotted have the same sets of cuts"""
        first_cutflow = self[datasets[0]].selections
        if not all(ds.selections == first_cutflow for ds in list(self.datasets.values())[1:]):
            raise ValueError("Datasets do not have the same cuts")
        return True

    def do_fakes_estimate(
        self,
        fakes_source_var: str,
        fakes_target_vars: Sequence[str],
        CR_passID_data: str = "CR_passID",
        CR_failID_data: str = "CR_failID",
        SR_passID_data: str = "SR_passID",
        SR_failID_data: str = "SR_failID",
        CR_passID_mc: str = "CR_passID_trueTau",
        CR_failID_mc: str = "CR_failID_trueTau",
        SR_passID_mc: str = "SR_passID_trueTau",
        SR_failID_mc: str = "SR_failID_trueTau",
    ) -> None:
        """Perform fakes estimate"""
        ff_var = fakes_source_var

        self.logger.info("Calculating fake factors for %s...", ff_var)

        # data histograms
        hCR_passID_data = self.get_hist(ff_var, "data", CR_passID_data, TH1=True)
        hCR_failID_data = self.get_hist(ff_var, "data", CR_failID_data, TH1=True)
        hSR_failID_data = self.get_hist(ff_var, "data", SR_failID_data, TH1=True)

        # mc truth matched histograms
        mc_ds = self.mc_samples
        hCR_passID_mc = self.sum_hists([f"{ds}_{ff_var}_{CR_passID_mc}_cut" for ds in mc_ds])
        hCR_failID_mc = self.sum_hists([f"{ds}_{ff_var}_{CR_failID_mc}_cut" for ds in mc_ds])
        hSR_failID_mc = self.sum_hists([f"{ds}_{ff_var}_{SR_failID_mc}_cut" for ds in mc_ds])
        self.histograms[f"all_mc_{ff_var}_{CR_passID_mc}_cut"] = hCR_passID_mc
        self.histograms[f"all_mc_{ff_var}_{CR_failID_mc}_cut"] = hCR_failID_mc
        self.histograms[f"all_mc_{ff_var}_{SR_failID_mc}_cut"] = hSR_failID_mc

        # FF calculation
        h_FF = (hCR_passID_data - hCR_passID_mc) / (hCR_failID_data - hCR_failID_mc)
        h_FF.SetName(f"{ff_var}_FF")
        h_SR_data_fakes = (hSR_failID_data - hSR_failID_mc) * h_FF

        self.histograms[f"{ff_var}_FF"] = h_FF
        self.histograms[f"{ff_var}_fakes_bkg_{ff_var}_src"] = h_SR_data_fakes

        # define ff_weights in MC
        ROOT.gInterpreter.Declare(
            f"TH1F* FF_hist_{ff_var} = reinterpret_cast<TH1F*>({ROOT.addressof(h_FF)});"
        )
        ff_weight = (
            f"reco_weight * FF_hist_{ff_var}->GetBinContent(FF_hist_{ff_var}->FindBin({ff_var}))"
        )
        ff_weight_col = f"FF_weight_{ff_var}"
        for mc in mc_ds:
            self[mc].filtered_df[SR_passID_mc] = (
                self[mc].filtered_df[SR_passID_mc].Define(ff_weight_col, ff_weight)
            )

        # background estimation in target variables
        mc_ff_hists: dict[str, dict[str, ROOT.RDF.RResultsPtr]] = dict()
        for target_var in fakes_target_vars:
            # define target histogram
            h_name = f"{target_var}_fakes_bkg_{ff_var}_src"
            h_bins = self[self.data_sample].get_binnings(target_var, SR_passID_data)
            h_target_var_ff = ROOT.TH1F(h_name, h_name, *plotting_tools.get_TH1_bins(**h_bins))

            # need to fill histograms for each MC sample
            ptrs: dict[str, ROOT.RDF.RResultsPtr] = {}
            for mc in mc_ds:
                ptrs[mc] = (
                    self[mc]
                    .filtered_df[SR_passID_mc]
                    .Fill(h_target_var_ff, [target_var, ff_weight_col])
                )
            mc_ff_hists[target_var] = ptrs

        # rerun over dataframes (must be its own loop to avoid separating the runs)
        for target_var, hists in mc_ff_hists.items():
            self.logger.info(f"Calculating fake background estimate for '%s'...", target_var)
            self.histograms[f"{target_var}_fakes_bkg_{ff_var}_src"] = reduce(
                lambda x, y: x + y, [ptr.GetValue() for ptr in mc_ff_hists[target_var].values()]
            )

        self.logger.info("Completed fakes estimate")

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @staticmethod
    def __sanitise_for_latex(s: Any) -> str:
        return str(s).replace(r"_", r"\_")

    @handle_dataset_arg
    def cutflow_printout(self, datasets: str, latex: bool = False) -> None:
        """Prints cutflow table to terminal"""
        self[datasets].cutflow_printout(self.paths.latex_dir if latex else None)

    def snapshot(self, filepath: Path = "", recreate: bool = True) -> None:
        """
        Save snapshot of all datasets to ROOT file, where each tree contains the ntuple representation of the dataset
        """
        self.logger.info(f"Saving snapshot of datasets...")

        if not filepath:
            filepath = self.paths.root_dir / f"{self.name}.root"

        if recreate and os.path.isfile(filepath):
            self.logger.debug(f"Found file at {filepath}. Removing..")
            os.remove(filepath)

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fMode = "UPDATE"
        opts.fOverwriteIfExists = True
        for dataset in self.datasets.values():
            for selection in dataset.filtered_df:
                self.logger.info(f"Snapshoting '{selection}' selection in {dataset.name}...")
                dataset.filtered_df[selection] = dataset.filtered_df[selection].Snapshot(
                    f"{dataset.name}/{selection}", str(filepath), list(dataset.all_vars), opts
                )
        self.logger.info(f"Full snapshot sucessfully saved.")

    def full_cutflow_printout(
        self,
        datasets: list[str],
        cutsets: list[str] | str | None = None,
        filename: str | Path | None = None,
    ) -> None:
        """Prints full cutflows for all passed datasets"""

        self.__verify_same_cuts(datasets)

        # for each cut set, create new set of rows in table
        if isinstance(cutsets, str):
            cutsets = [cutsets]
        elif cutsets is None:
            cutsets = list(self[datasets[0]].selections)

        # table build loop
        latex_str = f"\\begin{{tabular}}{{{'l' * (len(datasets) + 1)}}}\n"

        for cutset in cutsets:
            sanitised_str = self.__sanitise_for_latex(cutset)
            # header
            latex_str += "\\hline\n"
            latex_str += (
                " & ".join(
                    [f"Cut ({sanitised_str})"] + [self[dataset].label for dataset in datasets]
                )
                + "\\\\\n"
            )
            latex_str += "\\hline\n"

            cut_names = [
                cutflow_item.cut.name for cutflow_item in self[datasets[0]].cutflows[cutset]
            ]
            for i, cut_name in enumerate(cut_names):
                passes_list = [
                    str(int(self[dataset].cutflows[cutset][i].npass)) for dataset in datasets
                ]
                latex_str += f"{cut_name} & {' & '.join(passes_list)}\\\\\n"

        latex_str += "\\hline\n\\end{tabular}"

        # print to file
        if filename is None:
            filename = self.paths.latex_dir / f"{self.name}_full_cutflows.tex"

        with open(filename, "w") as f:
            f.write(latex_str)

    def print_metadata_table(
        self,
        datasets: list[str] | None = None,
        columns: list[str] | str = "all",
        filename: str | Path | None = None,
    ) -> None:
        """Print a latex table containing metadata for all datasets"""

        # Which datasets to run over?
        if datasets is None:
            datasets = list(self.datasets.keys())

        # possible headings
        header_names: dict[str, str] = {
            "dsid": "Dataset ID",
            "phys_short": "Physics short",
            "total_events": "Total Events",
            "sumw": "Sum of weights (post pre-selection)",
            "cross_section": "Cross-section (pb)",
            "kfactor": "K-Factor",
            "filter_eff": "Filter Efficiency",
            "generator_name": "Generator",
            "ptag": "p-tag",
            # "total_size": "Total size",
        }
        if columns == "all":
            columns = list(header_names.keys())
        else:
            if unexpected_column := [col for col in columns if col not in header_names.keys()]:
                self.logger.error(
                    "Metadata column(s) %s not contained in labels dictionary. "
                    "Possble column names: %s",
                    unexpected_column,
                    list(header_names.keys()),
                )

        # table build loop
        latex_str = f"\\begin{{tabular}}{{{'l' * (len(columns) + 1)}}}\n"
        latex_str += (
            " & ".join(["Dataset"] + [header_names[col] for col in columns])
            + "\\\\\n\\hline\\hline\n"
        )

        # loop over wanted datasets
        for dataset in datasets:
            dataset_dsids = self.metadata.dataset_dsids[dataset]
            latex_str += f"{self.__sanitise_for_latex(dataset)} & "

            for i, dsid in enumerate(dataset_dsids):
                if i != 0:
                    # blank space in first column under dataset name
                    latex_str += " & "

                row_values = [
                    self.__sanitise_for_latex(self.metadata[dsid][s]) for s in header_names
                ]
                latex_str += " & ".join(row_values) + "\\\\\n"
            latex_str += "\\hline\n"

        latex_str += "\\end{tabular}"

        # print to file
        if filename is None:
            filename = self.paths.latex_dir / f"{self.name}_metadata.tex"

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
            filename = self.paths.root_dir / f"{self.name}_histograms.root"

        self.logger.info(f"Saving {len(self.histograms)} histograms to file {filename}...")
        with ROOT.TFile(str(filename), tfile_option) as file:
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
