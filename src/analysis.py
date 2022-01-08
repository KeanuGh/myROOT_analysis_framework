import os
from typing import Optional, Union, Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import ArrayLike

import src.config as config
from src.dataset import Dataset
from src.datasetbuilder import DatasetBuilder
from src.logger import get_logger
from utils import file_utils, plotting_utils
from utils.decorators import check_single_dataset, handle_dataset_arg


class Analysis:
    """
    Analysis class acts as a container for the src.dataset.Dataset class. Contains methods to apply either to
    single datasets or across multiple datasets.
    Access datasets in class with analysis.dataset_name or analsis['dataset_name']. Can set by key but not by attribute
    When calling a method that applies to only one dataset, naming the dataset in argument ds_name is optional.
    TODO: apply method to ALL datasets if ds_name not provided?
    """

    def __init__(
            self,
            data_dict: Dict[str, Dict],
            analysis_label: str,
            global_lumi: Optional[float] = 139.,
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

        # SET OUTPUT DIRECTORIES
        # ===========================
        # TODO: change how this is done
        if not output_dir:
            # root in the directory above this one
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = output_dir + '/outputs/' + analysis_label + '/'  # where outputs go
        self.paths = {
            'plot_dir': output_dir + '/plots/',  # where plots go
            'pkl_df_dir': data_dir if data_dir else output_dir + '/data/',  # pickle file directory
            'pkl_hist_dir': output_dir + '/histograms/',  # pickle file to place histograms into
            'backup_cutfiles_dir': output_dir + '/cutfiles/',  # cutfile backups
            'latex_dir': output_dir + '/LaTeX/',  # where to print latex cutflow table
            'log_dir': output_dir + '/logs/',
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
            self.logger.info("=" * (42 + len(self.name)))
            self.logger.info(f"======== INITIALISING DATASET '{self.name}' =========")
            self.logger.info("=" * (42 + len(self.name)))

            if dup_args := set(data_args) & set(kwargs):
                raise SyntaxError(f"Got multiple values for argument(s) {dup_args} for dataset {name}")

            # get dataset build arguments out of options passed to analysis
            build_args = dict()
            for arg in [
                'data_path',
                'pkl_path',
                'cutfile_path',
                'cutfile',
                'tree_dict',
                'vars_to_calc',
                'cut_dicts'
            ]:
                build_args[arg] = data_args.pop(arg, None)

            # set correct pickle path if passed as a build argument
            if build_args['pkl_path'] is None:
                build_args['pkl_path'] = f"{self.paths['pkl_df_dir']}{name}_df.pkl"

            # check if a pickle file already exists if not already given
            # avoids rebuilding dataset unnecessarily
            if file_utils.file_exists(build_args['pkl_path']):
                self.logger.debug(f"Found pickle file at {build_args['pkl_path']}. Passing to builder")

            # make dataset
            builder = DatasetBuilder(
                name=name,
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
                ),
                **data_args,
                **kwargs
            )
            dataset = builder.build(**build_args)
            dataset.set_plot_dir(self.paths['plot_dir'])
            dataset.set_pkl_path(build_args['pkl_path'])

            if separate_loggers:
                # set new logger to append to analysis logger
                dataset.logger = self.logger
                dataset.logger.debug(f"{name} log handler returned to analysis.")  # test

            # save dataset no matter what
            dataset.save_pkl_file()

            self[name] = dataset  # save to analysis

            self.logger.info("=" * (42 + len(self.name)))
            self.logger.info(f"========= DATASET '{self.name}' INITIALISED =========")
            self.logger.info("=" * (42 + len(self.name)))
            self.logger.info("")

        self.logger.info("=" * (len(analysis_label) + 23))
        self.logger.info(f"ANALYSIS '{analysis_label}' INITIALISED")

    # ===============================
    # ========== BUILTINS ===========
    # ===============================
    def __getitem__(self, ds_name):
        return self.datasets[ds_name]

    def __setitem__(self, ds_name, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Analysis dataset must be of type {Dataset}")
        self.datasets[ds_name] = dataset

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        yield from self.datasets.values()

    def __repr__(self):
        return f'Analysis("{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.df.columns)}" for name, d in self.datasets.items()])}}}'

    def __str__(self):
        return f'"{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.df.columns)}" for name, d in self.datasets.items()])}}}'

    def __iadd__(self, other):
        self.datasets |= other.datasets

    # ===============================
    # ====== DATASET FUNCTIONS ======
    # ===============================
    def merge_datasets(
            self,
            *datasets: str,
            apply_cuts: Union[bool, str, Iterable[str]] = False,
            delete: bool = True,
            to_pkl: bool = False,
            verify: bool = False,
            delete_pkl: bool = False
    ) -> None:
        """
        Merge datasets by concatenating one or more into the other

        :param datasets: strings of datasets to merge. First dataset will be merged into.
        :param apply_cuts: True to apply all cuts to datasets before merging or False for no cuts
                           pass a string or list of strings of the cut label(s) to apply just those cuts
        :param delete: whether to delete datasets internally
        :param to_pkl: whether to print new dataset to a pickle file (will replace original pickle file)
        :param verify: whether to check for duplicated events
        :param delete_pkl: whether to delete pickle files of merged datasets (not the one that is merged into)
        """
        for n in datasets:
            if n not in self.datasets:
                raise ValueError(f"No dataset named {n} found in analysis {self.name}")

        if apply_cuts:
            self.apply_cuts(list(datasets), labels=apply_cuts)

        self.logger.info(f"Merging dataset(s) {datasets[1:]} into dataset {datasets[0]}...")

        self.datasets[datasets[0]].df = pd.concat([self.datasets[n].df for n in datasets],
                                                  verify_integrity=verify, copy=False)

        for n in datasets[1:]:
            if delete:
                self.__delete_dataset(n)
            if delete_pkl:
                self.__delete_pickled_dataset(n)

        if to_pkl:
            pd.to_pickle(self.datasets[datasets[0]].df, self.datasets[datasets[0]].pkl_file)
            self.logger.info(f"Saved merged dataset to file {self.datasets[datasets[0]].pkl_file}")

    @handle_dataset_arg
    def apply_cuts(self,
                   datasets: Union[str, Iterable[str]],
                   labels: Union[bool, str, List[str]] = True,
                   ) -> None:
        """
        Apply cuts to dataset dataframes. Skip cuts that do not exist in dataset, logging in debug.

        :param datasets: list of datasets or single dataset name. If not given applies to all datasets.
        :param labels: list of cut labels or single cut label. If True applies all cuts. Skips if logical false.
        :return: None if inplace is True.
                 If False returns DataFrame with cuts applied and associated cut columns removed.
                 Raises ValueError if cuts do not exist in dataframe
        """
        # skip cuts that don't exist in dataset
        if isinstance(labels, str):
            if labels + config.cut_label not in self.datasets[datasets].df.columns:
                self.logger.debug(f"No cut '{labels}' in dataset '{datasets}'; skipping.")
                return

        elif isinstance(labels, list):
            if missing_cuts := [
                label for label in labels
                if label + config.cut_label not in self.datasets[datasets].df.columns
            ]:
                self.logger.debug(f"No cuts {missing_cuts} in dataset '{datasets}'; skipping.")
                # remove missing cuts from list
                labels = [
                    label for label in labels
                    if label + config.cut_label not in missing_cuts
                ]

        self[datasets].apply_cuts(labels, inplace=True)

    @check_single_dataset
    def __delete_dataset(self, ds_name: str) -> None:
        self.logger.info(f"Deleting dataset {ds_name} from analysis {self.name}")
        del self.datasets[ds_name]

    @check_single_dataset
    def __delete_pickled_dataset(self, ds_name: str) -> None:
        self.logger.info(f"Deleting pickled dataset {ds_name}")
        file_utils.delete_file(self.datasets[ds_name].pkl_file)

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_hist(
            self,
            datasets: Union[str, Iterable[str]],
            var: str,
            bins: Union[List[float], Tuple[int, float, float]],
            weight: Union[str, float] = 1.,
            yerr: Union[ArrayLike, str] = None,
            labels: List[str] = None,
            w2: bool = False,
            normalise: Union[float, bool, str] = 'lumi',
            apply_cuts: Union[bool, str, List[str]] = False,
            logbins: bool = False,
            logx: bool = False,
            logy: bool = True,
            xlabel: str = '',
            ylabel: str = '',
            title: str = '',
            lepton: str = 'lepton',
            **kwargs
    ) -> None:
        """
        Plot same variable from different datasets

        :param datasets: string or list of strings corresponding to datasets in the analysis
        :param var: variable name to be plotted. must exist in all datasets
        :param bins: tuple of bins in x (n_bins, start, stop) or list of bin edges.
                     In the first case returns an axis of type Regular(), otherwise of type Variable().
                     Raises error if not formatted in one of these ways.
        :param weight: variable name in dataset to weight by or numeric value to weight all
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
        :param kwargs: keyword arguments to pass to mplhep.histplot()
        """
        self.logger.info(f'Plotting {var} in as overlay in {datasets}...')

        if isinstance(normalise, str):
            if normalise == 'lumi':
                normalise = self.global_lumi
            else:
                raise ValueError("Only 'lumi' allowed for string value normalisation")

        if labels:
            assert len(labels) == len(datasets), \
                f"Labels iterable (length: {len(labels)}) must be of same length as number of datasets ({len(datasets)})"

        fig, ax = plt.subplots()
        if isinstance(datasets, str):
            datasets = [datasets]
        for i, dataset in enumerate(datasets):
            self.datasets[dataset].plot_hist(
                var=var,
                bins=bins,
                weight=weight,
                ax=ax,
                yerr=yerr,
                normalise=normalise,
                logbins=logbins,
                label=labels[i] if labels else self.datasets[dataset].label,
                apply_cuts=apply_cuts,
                w2=w2,
                **kwargs
            )

        ax.legend(fontsize=10)
        plotting_utils.set_axis_options(ax, var, bins, lepton, xlabel, ylabel, title, logx, logy)

        filename = f"{self.paths['plot_dir']}{'_'.join(datasets)}_{var}{'_NORMED' if normalise else ''}.png"
        fig.savefig(filename, bbox_inches='tight')
        self.logger.info(f'Saved overlay plot of {var} to {filename}')

    @check_single_dataset
    def gen_cutflow_hist(self, ds_name: Optional[str], **kwargs) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param ds_name: Name of dataset to plot
        :return: None
        """
        self.datasets[ds_name].gen_cutflow_hist(**kwargs)

    @handle_dataset_arg
    def plot_mass_slices(self, datasets: Union[str, Iterable[str]], xvar: str, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param datasets: name(s) of dataset to plot
        :param xvar: variable in dataframe to plot
        :param kwargs: keyword args to pass to Dataset.plot_mass_slices()
        """
        self.datasets[datasets].plot_mass_slices(var=xvar, **kwargs)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @handle_dataset_arg
    def cutflow_printout(self, datasets: Union[str, Iterable[str]]) -> None:
        """Prints cutflow table to terminal"""
        self[datasets].cutflow.printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        self.logger.info(f"========== KINEMATICS ===========")
        for name in self.datasets:
            self.logger.info(name + ":")
            self.logger.info("---------------------------------")
            self.logger.info(f"cross-section: {self.datasets[name].cross_section:.2f} fb")
            self.logger.info(f"luminosity   : {self.datasets[name].luminosity:.2f} fb-1")

    @handle_dataset_arg
    def print_latex_table(self, datasets: Union[str, Iterable[str]]) -> None:
        """
        Prints a latex table(s) of cutflow.

        :param datasets: list of datasets or single dataset name. If not given applies to all datasets.
        :return: None
        """
        self[datasets].print_latex_table(f"{self.paths['latex_dir']}{self[datasets].name}_cutflow.tex")
