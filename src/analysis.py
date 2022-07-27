import inspect
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import ArrayLike

import src.config as config
from src.dataset import Dataset
from src.datasetbuilder import DatasetBuilder
from src.histogram import Histogram1D
from src.logger import get_logger
from utils import file_utils, plotting_utils, ROOT_utils
from utils.context import check_single_dataset, handle_dataset_arg


class Analysis:
    """
    Analysis class acts as a container for the src.dataset.Dataset class. Contains methods to apply either to
    single datasets or across multiple datasets.
    Access datasets in class with analysis.dataset_name or analsis['dataset_name']. Can set by key but not by attribute
    When calling a method that applies to only one dataset, naming the dataset in argument ds_name is optional.
    """
    __slots__ = "name", "paths", "histograms", "logger", "datasets", "global_lumi", "__output_dir"

    def __init__(
            self,
            data_dict: Dict[str, Dict],
            analysis_label: str,
            global_lumi: float | None = 139.,
            output_dir: str = None,
            data_dir: str = None,
            log_level: int = 20,
            log_out: str = 'both',
            timedatelog: bool = True,
            separate_loggers: bool = False,
            **kwargs
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
        :param kwargs: Options arguments to pass to all dataset builders
        """
        self.name = analysis_label
        if self.name in data_dict:
            raise SyntaxError("Analysis must have different name to any dataset")
        self.histograms: OrderedDict[str, Histogram1D] = OrderedDict()

        # SET OUTPUT DIRECTORIES
        # ===========================
        if not output_dir:
            # root in the directory above this one
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__output_dir = output_dir + '/outputs/' + analysis_label + '/'  # where outputs go
        self.paths = {
            'plot_dir': self.__output_dir + '/plots/',  # where plots go
            'pkl_df_dir': data_dir if data_dir else output_dir + '/data/',  # pickle file directory
            'latex_dir': self.__output_dir + '/LaTeX/',  # where to print latex cutflow table
            'log_dir': self.__output_dir + '/logs/',
        }
        for path in self.paths:
            file_utils.makedir(self.paths[path])

        # LOGGING
        # ============================
        self.logger = get_logger(
            name=self.name,
            log_dir=self.paths['log_dir'],
            log_level=log_level,
            log_out=log_out,
            timedatelog=timedatelog
        )

        # SET OTHER GLOBAL OPTIONS
        # ============================
        self.name = analysis_label
        self.global_lumi = global_lumi

        # BUILD DATASETS
        # ============================
        self.datasets: Dict[str, Dataset] = dict()
        for name, data_args in data_dict.items():
            self.logger.info("")
            self.logger.info("=" * (42 + len(name)))
            self.logger.info(f"======== INITIALISING DATASET '{name}' =========")
            self.logger.info("=" * (42 + len(name)))

            # get dataset build arguments out of options passed to analysis
            if dup_args := set(data_args) & set(kwargs):
                raise SyntaxError(f"Got multiple values for argument(s) {dup_args} for dataset {name}")

            builder_args = dict()  # arguments to pass to dataset builder
            for builder_arg in inspect.signature(DatasetBuilder.__init__).parameters:
                if str(builder_arg) == 'self': continue
                if builder_arg in data_args:
                    builder_args[builder_arg] = data_args[builder_arg]
                if builder_arg in kwargs:
                    builder_args[builder_arg] = kwargs[builder_arg]

            build_args = dict()  # arguments to pass to build()
            for build_arg in inspect.signature(DatasetBuilder.build).parameters:
                if str(build_arg) == 'self': continue
                if build_arg in data_args:
                    build_args[build_arg] = data_args[build_arg]
                if build_arg in kwargs:
                    build_args[build_arg] = kwargs[build_arg]

            # set correct pickle path if not passed as a build argument
            if 'pkl_path' not in build_args:
                build_args['pkl_path'] = f"{self.paths['pkl_df_dir']}{name}_df.pkl"

            # check if a pickle file already exists if not already given
            # avoids rebuilding dataset unnecessarily
            if file_utils.file_exists(build_args['pkl_path']):
                self.logger.debug(f"Found pickle file at {build_args['pkl_path']}. Passing to builder")

            # make dataset
            builder = DatasetBuilder(
                name=name,
                **builder_args,
                logger=(
                    self.logger  # use single logger
                    if not separate_loggers
                    else get_logger(  # if seperate, make new logger for each Dataset
                        name=name,
                        log_dir=self.paths['log_dir'],
                        log_level=log_level,
                        log_out=log_out,
                        timedatelog=timedatelog
                    )
                )
            )
            dataset = builder.build(**build_args)
            if separate_loggers:
                # set new logger to append to analysis logger
                dataset.logger = self.logger
                dataset.logger.debug(f"{name} log handler returned to analysis.")  # test

            dataset.dsid_metadata_printout()

            dataset.set_plot_dir(self.paths['plot_dir'])
            dataset.set_pkl_path(build_args['pkl_path'])

            self[name] = dataset  # save to analysis

            self.logger.info("=" * (42 + len(name)))
            self.logger.info(f"========= DATASET '{name}' INITIALISED =========")
            self.logger.info("=" * (42 + len(name)))
            self.logger.info("")

        self.logger.info("=" * (len(analysis_label) + 23))
        self.logger.info(f"ANALYSIS '{analysis_label}' INITIALISED")

    # ===============================
    # ========== BUILTINS ===========
    # ===============================
    def __getitem__(self, key: str):
        self.__check_ds(key)
        return self.datasets[key]

    def __setitem__(self, ds_name: str, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Analysis dataset must be of type {Dataset}")
        self.datasets[ds_name] = dataset

    def __delitem__(self, key: str):
        self.__check_ds(key)
        del self.datasets[key]

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        yield from self.datasets.values()

    def __repr__(self):
        return f'Analysis("{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.df.columns)}" for name, d in self.datasets.items()])}}}'

    def __str__(self):
        return f'"{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.df.columns)}" for name, d in self.datasets.items()])}}}'

    def __or__(self, other):
        return self.datasets | other.datasets

    def __ror__(self, other):
        return other.datasets | self.datasets

    def __ior__(self, other):
        self.datasets |= other.datasets

    # ===============================
    # ====== DATASET FUNCTIONS ======
    # ===============================
    def create_subdataset(self, dataset: str, name: str, args) -> None:
        """
        Create new dataset from subset of other dataset

        :param dataset: Dataset to subset
        :param name: Name of new dataset
        :param args: argument to pass to pd.DataFrame.loc[]
        """
        self[name] = self[dataset].subset(args)

    def create_dsid_subdataset(self, dataset: str, name: str, dsid: str | int) -> None:
        """
        Create new dataset from DSID of other dataset

        :param dataset: Dataset to subset
        :param name: Name of new dataset
        :param dsid: Dataset ID
        """
        self[name] = self[dataset].subset_dsid(dsid)

    def create_cut_subdataset(self, dataset: str, name: str, cut: str) -> None:
        """
        Create new dataset from cut in other dataset

        :param dataset: Dataset to subset
        :param name: Name of new dataset
        :param cut: name of cut
        """
        self[name] = self[dataset].subset_cut(cut)

    def __check_ds(self, key: str) -> None:
        """Does dataset exist?"""
        if key not in self.datasets:
            raise ValueError(f"No dataset named {key} found in analysis {self.name}")

    def merge_datasets(
            self,
            *datasets: str,
            apply_cuts: bool | str | List[str] = False,
            new_name: str = None,
            delete: bool = True,
            to_pkl: bool = False,
            verify: bool = False,
            delete_pkl: bool = False,
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
        :param delete_pkl: whether to delete pickle files of merged datasets (not the one that is merged into)
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
                    self.logger.warning(f"Missing column(s) from dataset '{d}' but are in '{datasets[0]}': "
                                        f"{cols_missing_from_first}")
                if cols_missing_from_curr := curr_cols - first_cols:
                    self.logger.warning(f"Missing column(s) from dataset '{datasets[0]}' but are in '{d}': "
                                        f"{cols_missing_from_curr}")

        if apply_cuts:
            self.apply_cuts(list(datasets), labels=apply_cuts)

        self.logger.info(f"Merging dataset(s) {datasets[1:]} into dataset {datasets[0]}...")

        self[datasets[0]].df = pd.concat([self[n].df for n in datasets], verify_integrity=verify, copy=False)
        self[datasets[0]].name = datasets[0]

        if sort:
            self[datasets[0]].df.sort_index(level='DSID', inplace=True)

        if new_name:
            self[new_name] = self.datasets.pop(datasets[0])

        for n in datasets[1:]:
            if delete:
                self.__delete_dataset(n)
            if delete_pkl:
                self.__delete_pickled_dataset(n)

        if to_pkl:
            pd.to_pickle(self[datasets[0]].df, self[datasets[0]].pkl_file)
            self.logger.info(f"Saved merged dataset to file {self[datasets[0]].pkl_file}")

    @handle_dataset_arg
    def apply_cuts(self,
                   datasets: str | Iterable[str],
                   labels: bool | str | List[str] = True,
                   reco: bool = False,
                   truth: bool = False,
                   ) -> None:
        """
        Apply cuts to dataset dataframes. Skip cuts that do not exist in dataset, logging in debug.

        :param datasets: list of datasets or single dataset name. If not given applies to all datasets.
        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :param reco: cut on reco cuts
        :param truth: cut on truth cuts
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied and associated cut columns removed.
                 Raises ValueError if cuts do not exist in dataframe
        """
        # skip cuts that don't exist in dataset
        if isinstance(labels, str):
            if labels + config.cut_label not in self[datasets].df.columns:
                self.logger.debug(f"No cut '{labels}' in dataset '{datasets}'; skipping.")
                return

        elif isinstance(labels, list):
            if missing_cuts := [
                label for label in labels
                if label + config.cut_label not in self[datasets].df.columns
            ]:
                self.logger.debug(f"No cuts {missing_cuts} in dataset '{datasets}'; skipping.")
                # remove missing cuts from list
                labels = [
                    label for label in labels
                    if label + config.cut_label not in missing_cuts
                ]

        self[datasets].apply_cuts(labels, reco, truth, inplace=True)

    @check_single_dataset
    def __delete_dataset(self, ds_name: str) -> None:
        self.logger.info(f"Deleting dataset {ds_name} from analysis {self.name}")
        del self[ds_name]

    @check_single_dataset
    def __delete_pickled_dataset(self, ds_name: str) -> None:
        self.logger.info(f"Deleting pickled dataset {ds_name}")
        file_utils.delete_file(self[ds_name].pkl_file)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_hist(
            self,
            datasets: str | ArrayLike | List[str],
            var: str | ArrayLike | List[str],
            bins: List[float] | ArrayLike | Tuple[int, float, float],
            weight: List[str | float] | str | float = 1.,
            yerr: ArrayLike | str = True,
            labels: List[str] = None,
            w2: bool = False,
            normalise: float | bool | str = False,
            apply_cuts: bool | str | List[str] = False,
            logbins: bool = False,
            logx: bool = False,
            logy: bool = True,
            xlabel: str = '',
            ylabel: str = '',
            title: str = '',
            lepton: str = 'lepton',
            scale_by_bin_width: bool = False,
            stats_box: bool = False,
            ratio_plot: bool = True,
            ratio_fit: bool = False,
            ratio_axlim: float = None,
            ratio_label: str = 'Ratio',
            filename: str = None,
            name_suffix: str = '',
            name_prefix: str = '',
            **kwargs
    ) -> List[Histogram1D]:
        """
        Plot same variable from different datasets

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
        :param apply_cuts: True to apply all cuts to dataset before plotting or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param logbins: whether logarithmic binnings
        :param logx: whether log scale x-axis
        :param logy: whether log scale y-axis
        :param xlabel: x label
        :param ylabel: y label
        :param title: plot title
        :param lepton: lepton to fill variable label
        :param scale_by_bin_width: divide histogram bin values by bin width
        :param stats_box: display stats box
        :param ratio_plot: If True, adds ratio of the first plot with each subseqent plot below
        :param ratio_fit: If True, fits ratio plot to a 0-degree polynomial and display line, chi-square and p-value
        :param ratio_axlim: pass to yax_lim in rato plotter
        :param ratio_label: y-axis label for ratio plot
        :param filename: name of output
        :param name_suffix: suffix to add at end of histogram/file name
        :param name_prefix: prefix to add at start of histogram/file
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        """
        self.logger.info(f'Plotting {var} in as overlay in {datasets}...')

        # naming template for file/histogram name
        name_template = (
            ((name_prefix + '_') if name_prefix else '') +  # prefix
            "{dataset}" +                                   # name of dataset(s)
            "_{variable}" +                                 # name of variable(s)
            ('_NORMED' if normalise else '') +              # normalisation flag
            (('_' + name_suffix) if name_suffix else '')    # suffix
        )

        if isinstance(datasets, str):
            datasets = [datasets]
        if isinstance(labels, str):
            labels = [labels]

        # no ratio if just one thing being plotted
        if len(datasets) == 1:
            ratio_plot = False

        if isinstance(normalise, str):
            if normalise == 'lumi':
                normalise = self.global_lumi
            else:
                raise ValueError("Only 'lumi' allowed for string value normalisation")

        if labels:
            assert len(labels) == len(datasets), \
                f"Labels iterable (length: {len(labels)}) must be of same length as number of datasets ({len(datasets)})"

        if ratio_plot:
            fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax = plt.subplots()
            ratio_ax = None  # just so IDE doesn't complain about missing variable

        if len(datasets) > 2:
            self.logger.warning("Not enough space to display stats box. Will not display")
            stats_box = False

        hists = []  # add histograms to be overlaid in this list
        for i, dataset in enumerate(datasets):
            varname = var if isinstance(var, str) else var[i]
            label = labels[i] if isinstance(labels, list) else self[dataset].label
            hist_name = name_template.format(dataset=dataset, variable=var)
            hist = self[dataset].plot_hist(
                    var=varname,
                    bins=bins,
                    weight=weight[i] if isinstance(weight, list) else weight,
                    ax=ax,
                    yerr=yerr,
                    normalise=normalise,
                    logbins=logbins,
                    name=hist_name,
                    label=label,
                    apply_cuts=apply_cuts,
                    w2=w2,
                    stats_box=stats_box,
                    scale_by_bin_width=scale_by_bin_width,
                    **kwargs
                )
            hists.append(hist)
            self.histograms[hist_name] = hist

            if ratio_plot and len(hists) > 1:
                # ratio of first dataset to this one
                label = f"{labels[-1]}/{labels[0]}" if labels else f"{self[dataset].label}/{self[datasets[0]].label}"
                color = 'k' if (len(datasets) == 2) else ax.get_lines()[-1].get_color()
                ratio_hist_name = name_template.format(dataset=f"{dataset}_{datasets[0]}", variable=varname) + '_ratio'
                ratio_hist = hists[0].plot_ratio(
                    hists[-1],
                    ax=ratio_ax,
                    yerr=yerr,
                    label=label,
                    normalise=bool(normalise),
                    color=color,
                    fit=ratio_fit,
                    yax_lim=ratio_axlim,
                    name=ratio_hist_name,
                    display_stats=len(datasets) <= 3  # ony display fit results if there are two fits or less
                )
                self.histograms[ratio_hist_name] = ratio_hist

        ax.legend(fontsize=10, loc='upper right')
        plotting_utils.set_axis_options(ax, var, bins, lepton, xlabel, ylabel, title, logx, logy,
                                        diff_xs=scale_by_bin_width)
        if ratio_plot:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.1, wspace=0)
            ax.set_xticklabels([])
            ax.set_xlabel('')

            if len(datasets) > 2:  # don't show legend if there's only two datasets
                ratio_ax.legend(fontsize=10, loc=1)

            plotting_utils.set_axis_options(axis=ratio_ax, var_name=var, bins=bins, lepton=lepton,
                                            diff_xs=scale_by_bin_width, xlabel=xlabel, ylabel=ratio_label,
                                            title='', logx=logx, logy=False, label=False)

        if filename:
            filename = self.paths['plot_dir'] + '/' + filename
        else:
            if isinstance(var, list):
                varname = '_'.join(var)
            else:
                varname = var
            filename = (
                    self.paths['plot_dir'] +
                    name_template.format(dataset='_'.join(datasets), variable=varname) +
                    '.png'
            )

        fig.savefig(filename, bbox_inches='tight')
        self.logger.info(f'Saved overlay plot of {var} to {filename}')
        plt.close(fig)
        return hists

    @check_single_dataset
    def gen_cutflow_hist(self, ds_name: str | None, **kwargs) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param ds_name: Name of dataset to plot
        :return: None
        """
        self[ds_name].gen_cutflow_hist(**kwargs)

    @handle_dataset_arg
    def plot_mass_slices(self, datasets: str | Iterable[str], xvar: str, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param datasets: name(s) of dataset to plot
        :param xvar: variable in dataframe to plot
        :param kwargs: keyword args to pass to Dataset.plot_mass_slices()
        """
        self[datasets].plot_dsid(var=xvar, **kwargs)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @handle_dataset_arg
    def cutflow_printout(self, datasets: str | Iterable[str]) -> None:
        """Prints cutflow table to terminal"""
        self[datasets].cutflow.printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        self.logger.info(f"========== KINEMATICS ===========")
        for name in self.datasets:
            self.logger.info(name + ":")
            self.logger.info("---------------------------------")
            self.logger.info(f"cross-section: {self[name].cross_section:.2f} fb")
            self.logger.info(f"luminosity   : {self[name].luminosity:.2f} fb-1")

    def save_histograms(self, filename: str = None,
                        tfile_option: str = 'Update',
                        write_option: str = 'Overwrite',
                        clear_hists: bool = False,
                        ) -> None:
        """
        Saves current histograms into root file

        :param filename: Should end in '.root'. if not given will set as '<analysis_name>.root'
        :param tfile_option: TFile option.
                             See: https://root.cern.ch/doc/master/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
        :param write_option: WriteObject() option.
                             See: https://root.cern.ch/doc/master/classTDirectoryFile.html#ae1bb32dcbb69de7f06a3b5de9d22e852
        :param clear_hists: clears histograms in dictionary
        """
        if not filename:
            filename = f"{self.__output_dir}/{self.name}.root"

        self.logger.info(f"Saving {len(self.histograms)} histograms to file {filename}...")
        with ROOT_utils.ROOT_TFile_mgr(filename, tfile_option) as file:
            for name, histo in self.histograms.items():
                file.WriteObject(histo.TH1, name, write_option)

        if clear_hists:
            self.histograms = OrderedDict()

    @handle_dataset_arg
    def print_latex_table(self, datasets: str | Iterable[str]) -> None:
        """
        Prints a latex table(s) of cutflow.

        :param datasets: list of datasets or single dataset name. If not given applies to all datasets.
        :return: None
        """
        self[datasets].print_latex_table(f"{self.paths['latex_dir']}{self[datasets].name}_cutflow.tex")
