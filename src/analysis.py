import copy
import inspect
import itertools
import os
from dataclasses import dataclass, field, fields
from functools import reduce
from pathlib import Path
from typing import Callable, Any, Sequence, Generator, TypedDict

import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import ticker
from numpy.typing import ArrayLike
from tabulate import tabulate

from src.dataset import Dataset
from src.datasetbuilder import DatasetBuilder, lumi_year
from src.dsid_meta import DatasetMetadata
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import plotting_tools, variable_names
from utils.context import handle_dataset_arg


class PlotOpts(TypedDict):
    """Per-histogram options for plotting"""

    vals: list[str | Histogram1D | ROOT.TH1]
    hists: list[Histogram1D]
    datasets: list[str]
    selections: list[str]
    labels: list[str]
    colours: list[str]


@dataclass(slots=True)
class AnalysisPath:
    """
    Container class for paths needed by analyses
    """

    output_dir: Path

    plot_dir: Path = field(init=False, default_factory=Path)
    latex_dir: Path = field(init=False, default_factory=Path)
    root_dir: Path = field(init=False, default_factory=Path)
    log_dir: Path = field(init=False, default_factory=Path)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.plot_dir = Path(self.output_dir) / "plots"
        self.root_dir = Path(self.output_dir) / "root"
        self.latex_dir = Path(self.output_dir) / "LaTeX"
        self.log_dir = Path(self.output_dir) / "logs"

    def create_paths(self):
        """Create paths needed by analyses"""
        for field_ in fields(self):
            if not issubclass(field_.type, Path):
                raise ValueError(f"Non-Path attribute in {self}!")
            getattr(self, field_.name).mkdir(parents=True, exist_ok=True)


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
            output_dir = Path(__file__).absolute().parent.parent / "outputs" / analysis_label
        self.paths = AnalysisPath(output_dir)
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
        dsid_metadata_cache = output_dir / "dsid_meta_cache.json"

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
        self.data_sample: str | None = None
        self.signal_sample: str | None = None
        self.datasets: dict[str, Dataset] = dict()
        for dataset_name, data_args in data_dict.items():
            self.logger.info("")
            self.logger.info("=" * (42 + len(dataset_name)))
            self.logger.info(f"======== INITIALISING DATASET '{dataset_name}' =========")
            self.logger.info("=" * (42 + len(dataset_name)))

            if "is_data" in data_args:
                if self.data_sample is not None:
                    raise ValueError("Can't have more than one data sample!")
                self.data_sample = dataset_name
            elif "is_signal" in data_args:
                if self.signal_sample is not None:
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
                dataset.histograms = dataset.gen_all_histograms(dataset.df)
                if builder.do_systematics:
                    for systematic, systematics_df in dataset.systematics_df.items():
                        dataset.histograms_systematics[systematic] = dataset.gen_all_histograms(
                            systematics_df, sys_name=systematic, do_prints=False
                        )
            # generate cutflow
            dataset.gen_cutflows()
            dataset.export_histograms(f"{dataset_name}_histograms.root")

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
        val: str | Histogram1D | ROOT.TH1 | Sequence[str | Histogram1D | ROOT.TH1],
        dataset: str | Sequence[str | None] | None = None,
        selection: str | None | Sequence[str | None] = False,
        label: str | None | Sequence[str | None] = None,
        colour: str | None | Sequence[str | None] = None,
        yerr: ArrayLike | str | bool = True,
        logx: bool = False,
        logy: bool = False,
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
        kind: str = "overlay",
        flow: bool = False,
        suffix: str = "",
        prefix: str = "",
        **kwargs,
    ) -> None:
        """
        Plot same variable from different datasets.
        If one dataset is passed but multiple variables, will plot overlays of the variable for that one dataset
        Checks to see if histogram exists in dataset histogram dictionary first, to avoid refilling histogram

        :param val: variable name to be plotted. Either a string that exists in all datasets
                    or a list one for each dataset, or histograms themselves
        :param dataset: string or list of strings corresponding to datasets in the analysis
        :param selection: string or list of strings corresponding to selection(s) applied to variable
        :param label: list of labels for plot legend corresponding to each line
        :param colour: list of colours for histograms
        :param yerr: Histogram uncertainties. Following modes are supported:
                     - 'rsumw2', sqrt(SumW2) errors
                     - 'sqrtN', sqrt(N) errors or poissonian interval when w2 is specified
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
        :param sort: sort stacks by size so that smallest histogram is at the bottom
        :param kind: "overlay" for overlays of line histograms, "stack" for stacks
        :param flow: whether to show over/underflow bins
        :param suffix: suffix to add at end of histogram/file name
        :param prefix: prefix to add at start of histogram/file
        :param kwargs: keyword arguments to pass to `mplhep.histplot`
        """

        # PREAMBLE
        # ============================

        # check options
        _allowed_options: dict[str, set[str]] = {
            "kind": {"stack", "overlay"},
            "yerr": {True, False, "rsumw2", "sqrtN"},
            "ratio_err": {"sumw2", "binom", "carry"},
        }
        _opt_err_msg = "Valid options for '{}' are: {}. Got '{}'."
        _plot_vars = locals()
        for arg, opts in _allowed_options.items():
            assert _plot_vars[arg] in opts, _opt_err_msg.format(arg, opts, _plot_vars[arg])

        # listify and verify per-histogram variables
        n_plottables, per_hist_vars = self._process_plot_variables(
            {
                "vals": copy.copy(val),
                "datasets": copy.copy(dataset),
                "selections": copy.copy(selection),
                "labels": copy.copy(label),
                "colours": copy.copy(colour),
            }
        )
        if scale_by_bin_width:
            per_hist_vars["hists"] = [h / h.bin_widths for h in per_hist_vars["hists"]]

        # remove data & signal histogram if stacking, so it can be handled separately
        data_plot_args = {}
        signal_plot_args = {}
        if (kind == "stack") and (self.data_sample in per_hist_vars["datasets"]):
            idx = per_hist_vars["datasets"].index(self.data_sample)
            for v in per_hist_vars.keys():
                data_plot_args[v] = per_hist_vars[v].pop(idx)  # type: ignore
        if (kind == "stack") and (self.signal_sample in per_hist_vars["datasets"]):
            idx = per_hist_vars["datasets"].index(self.signal_sample)
            for v in per_hist_vars:
                signal_plot_args[v] = per_hist_vars[v].pop(idx)  # type: ignore

        # unset options that depend on multiple histograms
        if n_plottables == 1:
            if ratio_plot:
                self.logger.warning("Cannot generate ratio plot for plot with only one histogram!")
            ratio_plot = False
        if n_plottables > 2:
            if stats_box:
                self.logger.warning("Not enough space to display stats box. Will not display.")
            stats_box = False

        if ratio_plot:
            fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.1, wspace=0)
        else:
            fig, ax = plt.subplots()
            ratio_ax = None

        # it'll error out when plotting if the histogram edges aren't equal
        bin_range = (
            per_hist_vars["hists"][0].bin_edges[0],
            per_hist_vars["hists"][0].bin_edges[-1],
        )

        # STACK PLOT
        # ============================
        if kind == "stack":
            self._plot_stack(
                ax=ax,
                per_hist_vars=per_hist_vars,
                signal_hist=signal_plot_args["hists"] if signal_plot_args else None,
                data_hist=data_plot_args["hists"] if data_plot_args else None,
                sort=sort,
                yerr=yerr,
                flow=flow,
                **kwargs,
            )

            # handle ratio plot options
            if ratio_plot:
                all_mc_hist = reduce(
                    (lambda x, y: x + y), per_hist_vars["hists"] + [signal_plot_args["hists"]]
                )
                all_mc_bin_vals = all_mc_hist.bin_values()

                # MC errors
                if yerr is True:
                    # errors are sum of MC errors
                    err = np.array(
                        [hist.error() for hist in per_hist_vars["hists"]]
                        + [signal_plot_args["hists"].error()]
                    )
                    err = np.sum(err, axis=0)
                    err_bottom = (all_mc_bin_vals - err) / all_mc_bin_vals
                    err_top = (all_mc_bin_vals + err) / all_mc_bin_vals
                    ratio_ax.fill_between(
                        x=per_hist_vars["hists"][0].bin_edges,
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
                    xmin=bin_range[0],
                    xmax=bin_range[1],
                    colors="r",
                )

                all_mc_hist.plot_ratio(
                    data_plot_args["hists"],
                    ax=ratio_ax,
                    yerr=True,
                    yax_lim=ratio_axlim,
                    display_unity=False,
                )
                ratio_label = "Data / MC"

        # OVERLAY PLOT
        # ============================
        else:  # overlays
            for i, hist in enumerate(per_hist_vars["hists"]):
                hist.plot(
                    ax=ax,
                    yerr=yerr,
                    stats_box=stats_box,
                    label=per_hist_vars["labels"][i],
                    color=per_hist_vars["colours"][i],
                    **kwargs,
                )

                if ratio_plot and (n_plottables > 1) and (i > 0):
                    # ratio of first histogram to this one
                    per_hist_vars["hists"][0].plot_ratio(
                        hist,
                        ax=ratio_ax,
                        yerr=ratio_err,
                        label=(
                            f"{per_hist_vars['labels'][i]}/{per_hist_vars['labels'][0]}"
                            if n_plottables > 2
                            else None
                        ),
                        colour=per_hist_vars["colours"][i],
                        fit=ratio_fit,
                        yax_lim=ratio_axlim,
                        display_stats=len(per_hist_vars["hists"]) <= 3,
                    )
                    if n_plottables > 2:
                        ratio_ax.legend()

        # AXIS OPTIONS SETTING
        # ============================
        # limit to 4 rows and reverse order (so more important samples go in front)
        ncols = (
            len(per_hist_vars["hists"])
            + bool(yerr is not False)
            + bool(data_plot_args)
            + bool(signal_plot_args)
        )
        ncols = max(ncols // 4, 1)  # need at least one column!
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            reversed(legend_handles),
            reversed(legend_labels),
            fontsize=10,
            loc="upper right",
            ncols=ncols,
        )

        hep.atlas.label(italic=(True, True, False), ax=ax, loc=0, llabel="Internal", rlabel=title)

        # set axis options
        # get axis labels from variable names if possible
        if (
            all([isinstance(val, str) for val in per_hist_vars["vals"]])
            and len(val_name := set(per_hist_vars["vals"])) == 1
        ):
            val_name = next(iter(val_name))
            if val_name in variable_names.variable_data:
                _xlabel, _ylabel = plotting_tools.get_axis_labels(
                    val_name, diff_xs=scale_by_bin_width
                )
                if not xlabel:
                    xlabel = _xlabel
                if not ylabel:
                    ylabel = ylabel

        # Main plot yaxis options
        if y_axlim:
            ax.set_ylim(*y_axlim)
        if logy:
            ax.semilogy()
        else:
            # if there are no negative yvalues, set limit at origin
            ax.set_ylim(bottom=0)
        if ylabel:
            ax.set_ylabel(ylabel)

        if x_axlim:
            ax.set_xlim(*x_axlim)
        else:
            ax.set_xlim(*bin_range)

        # set some xaxis options only on bottom xaxis (ie ratio plot if it exists else regular plot)
        axis_ = ratio_ax if ratio_plot else ax
        if logx:
            axis_.semilogx()
            axis_.xaxis.set_minor_formatter(ticker.LogFormatter())
            axis_.xaxis.set_major_formatter(ticker.LogFormatter())
        if xlabel:
            axis_.set_xlabel(xlabel)

        if ratio_plot:
            if n_plottables > 2:  # don't show legend if there's only two plots
                ratio_ax.legend(fontsize=10, loc=1)
            ratio_ax.set_ylabel(ratio_label)

            if x_axlim:
                ratio_ax.set_xlim(*x_axlim)
            else:
                ratio_ax.set_xlim(*bin_range)

            # just in case (I do not trust matplotlib)
            ax.set_xticklabels([])
            ax.set_xlabel("")

        # SAVE
        # ============================
        if filename:
            filepath = self.paths.plot_dir / filename
        else:
            # naming template for file/histogram name
            def _srep(s: str, init_: bool = True) -> str:
                """String rep. of combinations of histogram definitions"""
                out = [
                    el for el in per_hist_vars[s] if (el is not None) and isinstance(el, str)
                ]  # type: ignore
                init = "_" if init_ else ""
                if out:
                    all_el = [el for el in per_hist_vars[s] if el is not None]  # type: ignore
                    if len(set(all_el)) == 1:
                        all_el = all_el[0]
                        return init + all_el
                    else:
                        return init + "_".join(all_el)
                else:
                    return ""

            filename_template = (
                (f"{prefix}_" if prefix else "")
                + _srep("vals", init_=False)
                + _srep("datasets")
                + _srep("selections")
                + ("_BIN_SCALED" if scale_by_bin_width else "")
                + ("_STACKED" if kind == "stack" else "")
                + (f"_{suffix}" if suffix else "")
            )
            filepath = self.paths.plot_dir / (filename_template + ".png")

        fig.savefig(filepath, bbox_inches="tight")
        self.logger.info(f"Saved plot to {filepath}")
        plt.close(fig)

    def _plot_stack(
        self,
        ax: plt.Axes,
        per_hist_vars: PlotOpts,
        signal_hist: Histogram1D | None = None,
        data_hist: Histogram1D | None = None,
        sort: bool = False,
        yerr: ArrayLike | bool = False,
        flow: bool = False,
        **kwargs,
    ) -> None:
        # Sort lists based on integral of histograms so smallest histograms sit at bottom
        if sort:
            for val in per_hist_vars:
                per_hist_vars[val] = sorted(  # type: ignore
                    per_hist_vars[val],  # type: ignore
                    key=lambda ls: per_hist_vars["hists"][per_hist_vars[val].index(ls)].integral,
                    # type: ignore
                )

        hist_list = per_hist_vars["hists"]
        full_stack = reduce((lambda x, y: x + y), hist_list)
        edges = full_stack.bin_edges
        alpha_list = [0.8] * len(hist_list)
        edgecolour_list = ["k"] * len(hist_list)

        hep.histplot(
            H=[h.bin_values(flow) for h in hist_list],
            bins=hist_list[-1].bin_edges,
            ax=ax,
            color=per_hist_vars["colours"],
            alpha=alpha_list if alpha_list else None,
            edgecolor=edgecolour_list if edgecolour_list else None,
            linewidth=1 if edgecolour_list else 0,
            label=per_hist_vars["labels"],
            stack=True,
            histtype="fill",
            zorder=reversed(range(len(hist_list))),  # mplhep plots in wrong order
            flow="show" if flow else None,
            **kwargs,
        )

        # handle signal seperately
        if signal_hist:
            full_stack += signal_hist
            full_stack.plot(ax=ax, yerr=None, color="r", label=self[self.signal_sample].label)

        if yerr is True:
            full_stack_vals = full_stack.bin_values(flow)
            full_stack_errs = full_stack.error(flow)
            err_top = full_stack_vals + (full_stack_errs / 2)
            err_bottom = full_stack_vals - (full_stack_errs / 2)

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
        if data_hist:
            ax.errorbar(
                data_hist.bin_centres,
                data_hist.bin_values(flow),
                xerr=data_hist.bin_widths / 2,
                yerr=data_hist.error(flow),
                linestyle="None",
                color="black",
                marker=".",
                label=self["data"].label,
            )

    def _process_plot_variables(self, var_dict: dict[str, Any]) -> tuple[int, PlotOpts]:
        """
        Make sure per-plottable variables in `plot()` are either all the same length,
        or be exactly 1 object to apply to each histogram.

        :return number of plottables, and cleaned variable dictionary
        """

        # make sure
        n_plottables = 1
        for var, val in var_dict.items():
            if isinstance(val, list):
                n_val = len(val)
                if n_val == 0:
                    raise ValueError(f"Empty value for '{var}'! Check arguments.")
                elif n_val > 1:
                    if n_plottables != 1 and n_val != n_plottables:
                        d_ = {k: len(v) for k, v in var_dict.items()}
                        raise ValueError(
                            f"Lengths of variables: {list(var_dict.keys())} must match in length"
                            f" or be single values to apply to all histograms.\n"
                            f"Got: {d_}"
                        )
                    n_plottables = len(val)

            else:
                var_dict[var] = [val]

        # second loop to duplicate and make sure everything is normalised
        for var, val in var_dict.items():
            if len(val) == 1:
                var_dict[var] = val * n_plottables

        # check variables individually
        hists = []
        c_iter = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for i in range(n_plottables):
            hists.append(
                self._process_val_args(
                    var_dict["vals"][i],
                    var_dict["datasets"][i],
                    var_dict["selections"][i],
                )
            )

            if var_dict["colours"][i] is None:
                if all([c is None for c in var_dict["colours"]]):
                    # just do all at once
                    var_dict["colours"] = list(itertools.islice(c_iter, len(var_dict["colours"])))

                elif var_dict["datasets"][i] and (len(set(var_dict["datasets"])) > 1):
                    var_dict["colours"][i] = self[var_dict["datasets"][i]].colour

                else:
                    # make sure colours don't repeat
                    c = next(c_iter)
                    while c in var_dict["colours"]:
                        try:
                            c = next(c_iter)
                        except StopIteration as e:
                            raise StopIteration(
                                "Run out of colours! Try a larger colour prop_cycle"
                            ) from e
                    var_dict["colours"][i] = c

            if var_dict["labels"][i] is None:
                if var_dict["datasets"][i]:
                    var_dict["labels"][i] = self[var_dict["datasets"][i]].label
                elif n_plottables > 1:
                    raise ValueError(f"Missing label for {i}th plot! Check arguments to plot()")

        var_dict["hists"] = hists

        return n_plottables, var_dict

    def _process_val_args(
        self,
        val: str | Histogram1D | ROOT.TH1,
        dataset: str | None = None,
        selection: str | None = None,
    ) -> Histogram1D:
        """Get Histogram1D object from val argument in plot"""
        if isinstance(val, Histogram1D):
            return val
        if isinstance(val, ROOT.TH1):
            return Histogram1D(th1=val)
        else:
            return self.get_hist(val, dataset, selection, TH1=False)

    # ===============================
    # ===== HISTOGRAM HANDLING ======
    # ===============================
    def gen_histogram(
        self,
        variable: str,
        dataset: str,
        selection: str | None = None,
        name: str = "",
        title: str = "",
        histtype: str = "TH1F",
        save: bool = True,
    ) -> ROOT.TH1:
        """
        Generate histogram on-the-fly from given options

        :param variable: variable in dataset to plot. Binning will be taken from internal binning dictionary
        :param dataset: dataset to fetch data from
        :param selection: selection to be applied to data
        :param name: name of histogram
        :param title: title of histogram
        :param histtype: TH1 type name
        :param save: whether to save in internal histogram dictionary
        :return: generated histogram
        """
        h = self[dataset].gen_histogram(variable, selection, name, title, histtype)

        if save:
            self.histograms[f"{dataset}_{variable}_{selection}"] = h

        return h

    def get_hist(
        self,
        variable: str,
        dataset: str | None = None,
        selection: str | None = None,
        allow_generation: bool = False,
        TH1: bool = True,
    ) -> Histogram1D | ROOT.TH1:
        """Get TH1 histogram from histogram dict or internal dataset"""
        try:
            hist_name_internal = self.get_hist_name(variable, dataset, selection)
        except ValueError as e:
            if allow_generation:
                self.logger.info(
                    f"Generating histogram for {variable} in {dataset} with selection: {selection}.."
                )
                h = self[dataset].gen_histogram(variable, selection)
                self.histograms[f"{dataset}_{variable}_{selection}"] = h
                self[dataset].histograms[f"{variable}_{selection}"] = h
                return h
            else:
                raise ValueError(
                    "No histogram found. "
                    "Set `allow_generation=True` to generate histograms that do not yet exist."
                ) from e

        if TH1:
            return self.histograms[hist_name_internal]
        else:
            return Histogram1D(th1=self.histograms[hist_name_internal], logger=self.logger)

    def get_hist_name(
        self,
        variable: str,
        dataset: str | None = None,
        selection: str | None = None,
    ) -> str:
        """Get name of histogram saved in histogram dict"""
        if variable in self.histograms:
            return variable

        elif selection and f"{variable}_{selection}" in self.histograms:
            return f"{variable}_{selection}"

        elif dataset is None:
            raise ValueError(
                f"No variable '{variable}' for selection '{selection}' found in analysis: {self.name}"
            )

        elif selection and f"{variable}_{selection}" in self[dataset].histograms:
            return f"{dataset}_{variable}_{selection}"

        elif selection:
            raise ValueError(f"No selection {selection} found for {variable} in {dataset}")

        elif variable in self[dataset].histograms:
            return dataset + "_" + variable

        else:
            raise ValueError(
                f"No histogram for {variable} in {dataset} for selection {selection}."
                "\nHistograms in analysis:"
                + "\n".join(self.histograms.keys())
                + "\n Histograms in dataset: "
                + "\n".join(self[dataset].histograms.keys())
            )

    def sum_hists(
        self, hists: list[str | ROOT.TH1], inplace_name: str | None = None
    ) -> ROOT.TH1 | None:
        """
        Sum together internal histograms.
        Optionally pass inplace_name to save automatically to internal histogram dictionary
        """
        if isinstance(hists[0], str):
            h = self.histograms[hists[0]].Clone()
        else:
            h = hists[0].Clone()

        for hist_to_sum in hists[1:]:
            if isinstance(hist_to_sum, ROOT.TH1):
                h.Add(hist_to_sum)
            else:
                h.Add(self.get_hist(hist_to_sum))

        if inplace_name:
            self.histograms[inplace_name] = h
        return h

    def __verify_same_cuts(self, datasets: list[str]):
        """check that all datasets to be plotted have the same sets of cuts"""
        first_cutflow = self[datasets[0]].selections
        if not all(ds.selections == first_cutflow for ds in list(self.datasets.values())[1:]):
            raise ValueError("Datasets do not have the same cuts")
        return True

    # ===============================
    # ========== ANALYSES ===========
    # ===============================
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
        name: str = "",
        save_intermediates: bool = False,
    ) -> None:
        """
        Perform fakes estimate

        :param fakes_source_var: variable to perform fakes estimate binning in
        :param fakes_target_vars: variables to apply fakes estimate to
        :param CR_passID_data: Control region passing ID in data
        :param CR_failID_data: Control region failing ID in data
        :param SR_passID_data: Signal region passing ID in data
        :param SR_failID_data: Signal region failing ID in data
        :param CR_passID_mc: Control region passing ID in mc
        :param CR_failID_mc: Control region failing ID in mc
        :param SR_passID_mc: Signal region passing ID in mc
        :param SR_failID_mc: Signal region failing ID in mc
        :param name: prefix to histogram naming for estimation
        :param save_intermediates: Whether to save intermediate fakes calculation histograms
        """
        ff_var = fakes_source_var
        if name:
            prefix = name + "_"
        else:
            prefix = ""

        if not self.data_sample:
            raise ValueError(
                "No data sample in analysis! "
                "How are you going to perform a data-driven fakes estimate?"
            )

        info_msg = "Calculating fake factors for %s"
        if name:
            info_msg += " with name: '%s'"
            msg_args = (ff_var, name)
        else:
            msg_args = (ff_var,)
        info_msg += "..."
        self.logger.info(info_msg, *msg_args)

        # data histograms
        hCR_passID_data = self.get_hist(
            ff_var, self.data_sample, CR_passID_data, allow_generation=True
        )
        hCR_failID_data = self.get_hist(
            ff_var, self.data_sample, CR_failID_data, allow_generation=True
        )
        hSR_failID_data = self.get_hist(
            ff_var, self.data_sample, SR_failID_data, allow_generation=True
        )

        # mc truth matched histograms
        hCR_passID_mc = self.sum_hists(
            [
                self.get_hist(ff_var, mc_ds, CR_failID_mc, allow_generation=True)
                for mc_ds in self.mc_samples
            ]
        )
        hCR_failID_mc = self.sum_hists(
            [
                self.get_hist(ff_var, mc_ds, CR_passID_mc, allow_generation=True)
                for mc_ds in self.mc_samples
            ]
        )
        hSR_failID_mc = self.sum_hists(
            [
                self.get_hist(ff_var, mc_ds, SR_passID_mc, allow_generation=True)
                for mc_ds in self.mc_samples
            ]
        )

        # FF calculation
        numerator = hCR_passID_data - hCR_passID_mc
        denominator = hCR_failID_data - hCR_failID_mc
        fakes_data_est = hSR_failID_data - hSR_failID_mc

        h_FF = numerator / denominator
        h_FF.SetName(f"{prefix}{ff_var}_FF")
        h_SR_data_fakes = fakes_data_est * h_FF

        self.histograms[f"{prefix}{ff_var}_FF"] = h_FF
        self.histograms[f"{prefix}{ff_var}_fakes_bkg_{ff_var}"] = h_SR_data_fakes

        # define ff_weights in MC
        ROOT.gInterpreter.Declare(
            f"TH1F* FF_hist_{prefix}{ff_var} = reinterpret_cast<TH1F*>({ROOT.addressof(h_FF)});"
        )
        ff_weight = f"reco_weight * FF_hist_{prefix}{ff_var}->GetBinContent(FF_hist_{prefix}{ff_var}->FindBin({ff_var}))"
        ff_weight_col = f"FF_weight_{prefix}{ff_var}"
        for mc in self.mc_samples:
            self[mc].filters[SR_passID_mc].df = (
                self[mc].filters[SR_passID_mc].df.Define(ff_weight_col, ff_weight)
            )

        # background estimation in target variables
        mc_ff_hists: dict[str, dict[str, ROOT.RDF.RResultsPtr]] = dict()
        for target_var in fakes_target_vars:
            # define target histogram
            h_name = f"{prefix}{target_var}_fakes_bkg_{ff_var}_src"
            h_bins = self[self.data_sample].get_binnings(target_var, SR_passID_data)
            h_target_var_ff = ROOT.TH1F(h_name, h_name, *plotting_tools.get_TH1_bins(**h_bins))

            # need to fill histograms for each MC sample
            ptrs: dict[str, ROOT.RDF.RResultsPtr] = {}
            for mc in self.mc_samples:
                ptrs[mc] = (
                    self[mc]
                    .filters[SR_passID_mc]
                    .df.Fill(h_target_var_ff, [target_var, ff_weight_col])
                )
            mc_ff_hists[target_var] = ptrs

        # rerun over dataframes (must be its own loop to avoid separating the runs)
        for target_var, hists in mc_ff_hists.items():
            info_msg = "Calculating fake background estimate for %s"
            if name:
                info_msg += " with name: '%s'"
                msg_args = (target_var, name)
            else:
                msg_args = (target_var,)
            info_msg += "..."
            self.logger.info(info_msg, *msg_args)
            self.histograms[f"{prefix}{target_var}_fakes_bkg_{ff_var}_src"] = reduce(
                lambda x, y: x + y, [ptr.GetValue() for ptr in mc_ff_hists[target_var].values()]
            )

        if save_intermediates:
            self.histograms[f"{prefix}all_mc_{ff_var}_{CR_passID_mc}"] = hCR_passID_mc
            self.histograms[f"{prefix}all_mc_{ff_var}_{CR_failID_mc}"] = hCR_failID_mc
            self.histograms[f"{prefix}all_mc_{ff_var}_{SR_failID_mc}"] = hSR_failID_mc
            self.histograms[f"{prefix}{ff_var}_FF_numerator"] = numerator
            self.histograms[f"{prefix}{ff_var}_FF_denominator"] = denominator
            self.histograms[f"{prefix}{ff_var}_FF_fakes_data_est"] = fakes_data_est

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
            for selection in dataset.filters:
                self.logger.info(f"Snapshoting '{selection}' selection in {dataset.name}...")
                # point dataframes to snapshot instead to avoid having to rerun on all the data later
                dataset.filters[selection].df = dataset.filters[selection].df.Snapshot(
                    f"{dataset.name}/{selection}", str(filepath), list(dataset.all_vars), opts
                )
            if dataset.filters_systematics:
                for sys_name, sys_selections in dataset.filters_systematics.items():
                    for selection in sys_selections:
                        dataset.filters_systematics[sys_name][
                            selection
                        ].df = dataset.filters_systematics[sys_name][selection].df.Snapshot(
                            f"{dataset.name}/{sys_name}/{selection}",
                            str(filepath),
                            list(dataset.all_vars),
                            opts,
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
