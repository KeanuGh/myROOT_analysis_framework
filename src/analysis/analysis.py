from typing import Optional, Union, Dict

# project imports
import analysis.config as config
from analysis.dataclass import Dataset
from utils import file_utils, decorators


class Analysis:
    def __init__(self, data_dict: Dict[str, Dict],
                 analysis_label: str,
                 force_rebuild: bool = False,
                 global_lumi: Optional[float] = None,
                 phibins: Optional[Union[tuple, list]] = None,
                 etabins: Optional[Union[tuple, list]] = None,
                 ):
        """
        Analysis class acts as a container for the analysis.dataclass.Dataset class. Contains methods to apply either to
        single datasets or across multiple datasets.
        Access datasets in class with analysis.dataset_name or analysis['dataset_name']. Can set by key but not by attribute
        When calling a method that applies to only one dataset, naming the dataset in argument ds_name is optional.
        TODO: logging, apply method to ALL datasets if ds_name not provided?

        :param data_dict: Dictionary of dictionaries containing paths to root files and the tree to extract from each.
        The key to the top-level dictionary is the label assigned to the dataset.
        :param force_rebuild: Force rebuild all dataframes.
        :param global_lumi: all data will be scaled to this luminosity
        :param phibins: bins for plotting phi
        :param etabins: bins for plotting eta
        """

        # SET OUTPUT DIRECTORIES
        # ===========================
        # set and create output directories in outputs/<analysis_label>/
        for path_var in (
            'pkl_df_filepath',
            'plot_dir',
            'latex_table_dir',
            'pkl_hist_dir',
            'backup_cutfiles_dir'
        ):
            config.__dict__[path_var] = config.__dict__[path_var].format(analysis_label)
            file_utils.makedir(config.__dict__[path_var])

        # SET OTHER GLOBAL OPTIONS
        # ============================
        config.force_rebuild = force_rebuild
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
        self.datasets = {name: Dataset(name, **ds) for name, ds in data_dict.items()}

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
        elif name not in self.__dict__:
            raise AttributeError(f"No attribute {name} in {self}")
        else:
            getattr(self, name)

    # len() is the number of contained datasets
    def __len__(self):
        return len(self.datasets)

    # can iterate over datasets
    def __iter__(self):
        yield from self.datasets.values()

    # ===============================
    # =========== PLOTS =============
    # ===============================
    @decorators.check_single_datafile
    def plot_with_cuts(self, ds_name: Optional[str], **kwargs) -> None:
        """
        Plots each variable in specific Dataset to cut from cutfile with each cutgroup applied

        :param ds_name: name of Dataset class to plot
        :param kwargs: keyword arguments to pass to method in dataset
        """
        self.datasets[ds_name].plot_with_cuts(**kwargs)

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
        self.datasets[ds_name].plot_mass_slices(xvar=xvar, **kwargs)

    # ===============================
    # ========= PRINTOUTS ===========
    # ===============================
    @decorators.check_single_datafile
    def cutflow_printout(self, ds_name: Optional[str] = None) -> None:
        """Prints cutflow table to terminal"""
        self.datasets[ds_name].cutflow.terminal_printout()

    def kinematics_printouts(self) -> None:
        """Prints some kinematic variables to terminal"""
        print(f"\n========== KINEMATICS ===========")
        for name in self.datasets:
            print(name + ":\n---------------------------------")
            print(f"cross-section: {self.datasets[name].cross_section:.2f} fb\n"
                  f"luminosity   : {self.datasets[name].luminosity:.2f} fb-1\n"
                  )

    @decorators.check_single_datafile
    def print_cutflow_latex_table(self, ds_name: Optional[str] = None, check_backup: bool = True) -> None:
        """
        Prints a latex table of cutflow. By default first checks if a current backup exists and will not print if
        backup is identical
        :param ds_name:
        :param check_backup: default true. Checks if backup of current cutflow already exists and if so does not print
        :return: None
        """
        if check_backup:
            last_backup = file_utils.get_last_backup(config.latex_table_dir)
            latex_file = self.datasets[ds_name].cutflow.print_latex_table(config.latex_table_dir, ds_name)
            if file_utils.identical_to_backup(latex_file, backup_file=last_backup):
                file_utils.delete_file(latex_file)
        else:
            self.datasets[ds_name].cutflow.print_latex_table(config.latex_table_dir, ds_name)
