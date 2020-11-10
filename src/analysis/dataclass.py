from dataclasses import dataclass, field
import pandas as pd

from analysis.cutflow import Cutflow
import analysis.config as config
import utils.dataframe_utils as df_utils
from utils.plotting_utils import plot_mass_slices


@dataclass
class Dataset:
    """
    Dataset class. Contains/will contain all the variables needed for a singular analysis dataset.
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

    def gen_cutflow(self, **kwargs) -> None:
        """Creates the cutflow class for this analysis"""
        self.__check_df()
        self.cutflow = Cutflow(self.df, **kwargs)

    # Dataframe methods
    # ===================
    def build_df(self, **kwargs) -> None:
        """Builds dataframe based on 'datapath', 'TTree_name' and 'is_slices'. Prints pickle file to 'pkl_path'"""
        self.df = df_utils.build_analysis_dataframe(self, **kwargs)

    def create_cut_columns(self, **kwargs) -> None:
        """Creates columns in dataframe that contain boolean values corresponding to cuts."""
        self.__check_df()
        print(f"Creating cuts for {self.name}...")
        df_utils.create_cut_columns(self.df, **kwargs)

    def map_weights(self) -> None:
        """Creates weights column in dataset based on dataset type"""
        self.__check_df()
        print(f"Creating weights for {self.name}...")
        if self.is_slices:
            self.df['weight'] = df_utils.gen_weight_column_slices(self.df)
        else:
            self.df['weight'] = df_utils.gen_weight_column(self.df)

    # Plotting methods
    # ===================
    def plot_mass_slices(self, **kwargs) -> None:
        """
        Plots mass slices for input variable xvar if dataset is_slices

        :param kwargs: keyword arguments to be passed to plotting_utils.plot_mass_slices()
        """
        self.__check_df()
        if not self.is_slices:
            raise Exception("Dataset does not contain slices.")

        plot_mass_slices(self.df, self.lepton, plot_label=self.name, **kwargs)

    # Private
    # ===================
    def __check_df(self) -> None:
        """Makes sure dataframe exists before trying to apply methods to it"""
        if self.df is None:
            raise Exception(f"Dataframe for {self.name} has yet to be generated.")
