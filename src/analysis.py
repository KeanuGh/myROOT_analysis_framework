import logging
import os
import sys
import time
from typing import Optional, Union, Dict, Iterable, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import boost_histogram as bh
import mplhep as hep

# project imports
import src.config as config
from src.dataset import Dataset
from utils import file_utils, decorators, plotting_utils


class Analysis:
    """
    Analysis class acts as a container for the src.dataset.Dataset class. Contains methods to apply either to
    single datasets or across multiple datasets.
    Access datasets in class with analysis.dataset_name or analsis['dataset_name']. Can set by key but not by attribute
    When calling a method that applies to only one dataset, naming the dataset in argument ds_name is optional.
    TODO: apply method to ALL datasets if ds_name not provided?
    """
    def __init__(self, data_dict: Dict[str, Dict],
                 analysis_label: str,
                 global_lumi: Optional[float] = None,
                 phibins: Optional[Union[tuple, list]] = None,
                 etabins: Optional[Union[tuple, list]] = None,
                 output_dir: str = None,
                 log_level: int = logging.INFO,
                 log_out: str = 'both',
                 ):
        """
        :param data_dict: Dictionary of dictionaries containing paths to root files and the tree to extract from each.
        The key to the top-level dictionary is the label assigned to the dataset.
        :param global_lumi: all data will be scaled to this luminosity
        :param phibins: bins for plotting phi
        :param etabins: bins for plotting eta
        :param log_level: logging level. See https://docs.python.org/3/library/logging.html#logging-levels
        :param log_out: where to set log output: 'FILE', 'CONSOLE' or 'BOTH'. (case-insensitive)
        """

        # SET OUTPUT DIRECTORIES
        # ===========================
        if not output_dir:
            # root in the directory above this one
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = output_dir + '/outputs/' + analysis_label + '/'  # where outputs go
        self.paths = {
            'plot_dir': output_dir + '/plots/',  # where plots go
            'pkl_df_filepath': output_dir + '/data/',  # pickle file containing extracted data, format to used dataset
            'pkl_hist_dir': output_dir + '/histograms/',  # pickle file to place histograms into
            'backup_cutfiles_dir': output_dir + '/cutfiles/',  # _cutfile backups
            'latex_table_dir': output_dir + '/LaTeX_cutflow_table/',  # where to print latex cutflow table
            'log_dir': output_dir + '/logs/',
        }
        for path in self.paths:
            file_utils.makedir(self.paths[path])

        # LOGGING
        # ============================
        if log_out not in ('file', 'both', 'console', None):
            raise ValueError("Accaptable values for 'log_out' parameter: 'file', 'both', 'console', None.")
        self.logger = logging.getLogger('analysis')
        self.logger.setLevel(log_level)
        if log_out.lower() in ('file', 'both'):
            filehandler = logging.FileHandler(f"{self.paths['log_dir']}/{analysis_label}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
            filehandler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-10s %(message)s'))
            self.logger.addHandler(filehandler)
        if log_out.lower() in ('console', 'both'):
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        logging.captureWarnings(True)

        self.logger.info(f"INITIALISING ANALYSIS '{analysis_label}'...")
        self.logger.info("="*(len(analysis_label)+27))

        # SET OTHER GLOBAL OPTIONS
        # ============================
        self.name = analysis_label
        
        if global_lumi:
            config.lumi = global_lumi

        # variables that require special (default) binning
        if etabins:
            config.etabins = etabins
        if phibins:
            config.phibins = phibins
        config.special_binning = {
            '_eta_': config.etabins,
            '_phi_': config.phibins,
            'w_y': config.phibins
        }
        
        # BUILD DATASETS
        # ============================
        self.datasets: Dict[str, Dataset] = {name: Dataset(name, paths=self.paths, **kwargs)
                                             for name, kwargs in data_dict.items()}

        self.logger.info("="*(len(analysis_label)+23))
        self.logger.info(f"ANALYSIS '{analysis_label}' INITIALISED")

    # ===============================
    # ========== BUILTINS ===========
    # ===============================
    # keys and attributes access containing datasets
    def __getitem__(self, ds_name):
        return self.datasets[ds_name]

    def __setitem__(self, ds_name, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Analysis dataset must be of type {Dataset}")
        self.datasets[ds_name] = dataset

    def __getattr__(self, name):
        if name in self.datasets:
            return self.datasets[name]
        else:
            raise AttributeError(f"No attrbute or dataset {name} found in analysis {self.name}")

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        yield from self.datasets.values()
        
    def __repr__(self):
        return f'Analysis("{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.columns)}" for name, d in self.datasets.items()])}}}'
            
    def __str__(self):
        return f'"{self.name}",Datasets:{{{", ".join([f"{name}: {len(d)}, {list(d.columns)}" for name, d in self.datasets.items()])}}}'
        
    def __iadd__(self, other):
        self.datasets |= other.datasets
    
    # ===============================
    # ====== DATASET FUNCTIONS ======
    # ===============================
    def merge(self, 
              *names: str, 
              delete: bool = True, 
              to_pkl: bool = False, 
              delete_pkl: bool = False) -> None:
        """
        Merge datasets by concatenating one or more into the other
        
        :param names: strings of datasets to merge. First dataset will be merged into.
        :param delete: whether to delete datasets internally
        :param to_pkl: whether to print new dataset to a pickle file (will replace original pickle file)
        :param delete_pkl: whether to delete pickle files of merged datasets (not the one that is merged into)
        """
        for n in names:
            if n not in self.datasets:
                raise ValueError(f"No dataset named {n} found in analysis {self.name}")
        
        self.logger.info(f"Merging dataset(s) {names[1:]} into dataset{names[0]}...")
        self.datasets[names[0]].df.append([self.datasets[n].df for n in names[1:]], ignore_index=True)
        
        for n in names[1:]:
            if delete:
                self.logger.debug(f"Internally deleting dataset {n}")
                del self.datasets[n]
            if delete_pkl:
                self.logger.info(f"Deleting pickled dataset {self.datasets[n].pkl_path}")
                file_utils.delete_file(self.datasets[n].pkl_path)
                    
        if to_pkl:
            pd.to_pickle(self.datasets[names[0]].df, self.datasets[names[0]].pkl_path)
            self.logger.info(f"Saved merged dataset to file {self.datasets[names[0]].pkl_path}")

    # ===============================
    # =========== PLOTS =============
    # ===============================
    def plot_hist_overlay(self, 
                          datasets = Union[str, Iterable[str]],
                          var = str,
                          bins=Union[Iterable[float], Tuple[int, float, float]],
                          labels: Iterable[str] = None,
                          weight: str = '',
                          logbins: bool = False,
                          logx: bool = False,
                          logy: bool = True,
                          xlabel: str = '',
                          ylabel: str = '',
                          title: str = '',
                          lepton: str = 'lepton',
                          **kwargs
                          ) -> None:
        """Plot overlaid variables in separate datasets"""
        self.logger.info(f'Plotting {var} in as overlay in {datasets}...')
        
        bh_axis = plotting_utils.get_axis(bins, logbins)
        
        if labels:
            assert len(labels) == len(datasets), \
               f"Labels iterable (length: {len(labels)}) must be of same length as number of datasets ({len(datasets)})"
    
        for i, dataset in enumerate(datasets):
            hist = bh.Histogram(bh_axis, storage=bh.storage.Weight())
            hist.fill(self.datasets[dataset][var], weight=self.datasets[dataset][weight] if weight else None)
            hep.histplot(hist, label=labels[i] if labels else self.datasets[dataset].label, **kwargs)
        
        _xlabel, _ylabel = plotting_utils.get_axis_labels(var, lepton)
        plt.xlabel(xlabel if xlabel else _xlabel)
        plt.ylabel(ylabel if ylabel else _ylabel)
        plt.legend(fontsize=10)
        if logx:
            plt.semilogx()
        if logy:
            plt.semilogy()
        hep.atlas.label(italic=(True, True), loc=0, llabel='Internal', rlabel=title)
        
        filename = self.paths['plot_dir'] + '_'.join(datasets) + '_' + var + '_overlay.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        self.logger.info(f'Saved overlay plot of {var} to {filename}')
        plt.clf()
    
    @decorators.check_single_datafile
    def plot_1d(self, ds_name: Optional[str], **kwargs) -> None:
        """
        Plots variable in specific Dataset. Simple plotter.

        :param ds_name: name of Dataset class to plot
        :param kwargs: keyword arguments to pass to method in dataset
        """
        self.datasets[ds_name].plot_1d(**kwargs)

    @decorators.check_single_datafile
    def plot_with_cuts(self, ds_name: Optional[str], **kwargs) -> None:
        """
        Plots each variable in specific Dataset to cut from cutfile with each cutgroup applied

        :param ds_name: name of Dataset class to plot
        :param kwargs: keyword arguments to pass to method in dataset
        """
        self.datasets[ds_name].plot_all_with_cuts(**kwargs)

    @decorators.check_single_datafile
    def gen_cutflow_hist(self, ds_name: Optional[str], **kwargs) -> None:
        """
        Generates and saves cutflow histograms. Choose which cutflow histogram option to print. Default: only by-event.

        :param ds_name: Name of dataset to plot
        :return: None
        """
        self.datasets[ds_name].gen_cutflow_hist(**kwargs)

    @decorators.check_single_datafile
    def make_all_cutgroup_2dplots(self, ds_name: Optional[str], **kwargs) -> None:
        """Plots all cutgroups as 2d plots

        :param ds_name: name of dataset to plot
        :param kwargs: keyword arguments to pass to plot_utils.plot_2d_cutgroups
        """
        self.datasets[ds_name].make_all_cutgroup_2dplots(**kwargs)

    @decorators.check_single_datafile
    def plot_mass_slices(self, ds_name: Optional[str], xvar: str, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param ds_name: name of dataset (in slices) to plot
        :param xvar: variable in dataframe to plot
        :param kwargs: keyword args to pass to dataclass.plot_mass_slices()
        """
        self.datasets[ds_name].plot_mass_slices(var=xvar, **kwargs)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @decorators.check_single_datafile
    def cutflow_printout(self, ds_name: Optional[str] = None) -> None:
        """Prints cutflow table to terminal"""
        self.datasets[ds_name].cutflow.printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        self.logger = logging.getLogger('analysis')
        self.ogger.info(f"========== KINEMATICS ===========")
        for name in self.datasets:
            self.logger.info(name + ":")
            self.logger.info("---------------------------------")
            self.logger.info(f"cross-section: {self.datasets[name].cross_section:.2f} fb")
            self.logger.info(f"luminosity   : {self.datasets[name].luminosity:.2f} fb-1")

    @decorators.check_single_datafile
    def print_cutflow_latex_table(self, ds_name: Optional[str] = None, check_backup: bool = True) -> None:
        """
        Prints a latex table of cutflow. By default, first checks if a current backup exists and will not print if
        backup is identical
        :param ds_name:
        :param check_backup: default true. Checks if backup of current cutflow already exists and if so does not print
        :return: None
        """
        if check_backup:
            last_backup = file_utils.get_last_backup(self.paths['latex_table_dir'])
            latex_file = self.datasets[ds_name].cutflow.print_latex_table(self.paths['latex_table_dir'], ds_name)
            if file_utils.identical_to_backup(latex_file, backup_file=last_backup):
                file_utils.delete_file(latex_file)
        else:
            self.datasets[ds_name].cutflow.print_latex_table(self.paths['latex_table_dir'], ds_name)
