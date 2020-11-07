from dataclasses import dataclass, field
import pandas as pd
from typing import List, Dict, Optional, OrderedDict, Union

from analysis.cutflow import Cutflow
import analysis.config as config
import utils.dataframe_utils as df_utils
from utils.plotting_utils import plot_mass_slices


@dataclass
class Dataset:
    """
    My Data class. (Will) contain(s) all the variables needed for a singular analysis dataset.
    Perhaps put all methods that act only on one dataset into here
    """
    name: str
    datapath: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    # field(repr=False) stops these vars showing up when .__repr__() is called
    cutflow: Cutflow = field(default=None, repr=False)  # stores cutflow object
    df: pd.DataFrame = field(default=None, repr=False)  # stores the actual data in a dataframe
    pkl_path: str = None  # where the dataframe pickle file will be stored
    cross_section: float = None
    luminosity: float = None
    is_slices: bool = False  # whether input data is in mass slices
    rebuild: bool = False  # whether dataframe is rebuilt
    lepton: str = 'lepton'  # name of lepton in dataset (if applicable)

    def __post_init__(self):
        """ this all runs after the __init__ """
        # initialise pickle filepath with given name
        if not self.pkl_path:
            self.pkl_path = config.pkl_df_filepath.format(self.name)

    # Variable setting
    # ===================
    def gen_cross_section(self) -> None:
        """Calculates dataset cross-section"""
        self.__check_df()
        self.cross_section = df_utils.get_cross_section(self.df)

    def gen_luminosity(self) -> None:
        """Calculates dataset luminosity"""
        self.__check_df()
        self.luminosity = df_utils.get_luminosity(self.df, xs=self.cross_section)

    def gen_cutflow(self, cut_dicts: List[Dict],
                    cutgroups: Optional[OrderedDict[str, List[str]]],
                    sequential: bool = True,
                    ) -> None:
        """Creates the cutflow class for this analysis"""
        self.cutflow = Cutflow(self.df,
                               cut_dicts=cut_dicts,
                               cutgroups=cutgroups,
                               sequential=sequential)

    # Dataframe methods
    # ===================
    def build_df(self, cut_list_dicts: List[dict], vars_to_cut: List[str],
                 extra_vars: Optional[List[str]] = None) -> None:
        """Builds dataframe based on 'datapath', 'TTree_name' and 'is_slices'. Prints pickle file to 'pkl_path'"""
        self.df = df_utils.build_analysis_dataframe(self, cut_list_dicts, vars_to_cut, extra_vars)

    def create_cut_columns(self, cut_dicts: List[Dict], printout=True) -> None:
        """Creates columns in dataframe that contain boolean values corresponding to cuts. """
        df_utils.create_cut_columns(self.df, cut_dicts, printout)

    # Plotting methods
    # ===================
    def plot_mass_slices(self, xvar: str, xbins: Union[tuple, list] = (100, 300, 10000),
                         logbins: bool = True, logx: bool = False, plot_label: bool = True
                         ) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param xvar: variable in dataframe to plot
        :param xbins: x-axis binning
        :param logbins: tuple of bins in x (n_bins, start, stop) or list of bin edges
        :param logx: whether to apply log bins (only if logbins was passed as tuple)
        :param plot_label: Whether to add dataset name as label to plot
        """
        if not self.is_slices:
            raise Exception("Dataset does not contain slices.")

        label = self.name if plot_label else None
        plot_mass_slices(self.df, self.lepton, xvar, xbins, logbins, logx, plot_label=label)

    # Private
    # ===================
    def __check_df(self) -> None:
        """Makes sure dataframe exists before trying to apply methods to it"""
        if self.df is None:
            raise Exception(f"Dataframe for {self.name} has yet to be generated.")
