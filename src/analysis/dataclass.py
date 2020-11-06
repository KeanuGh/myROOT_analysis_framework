from dataclasses import dataclass, field
import pandas as pd
from typing import List, Dict, Optional, OrderedDict

from analysis.cutflow import Cutflow
import analysis.config as config
import utils.dataframe_utils as df_utils


@dataclass
class Data:
    """
    My Data class. (Will) contain(s) all the variables needed for a singular analysis dataset.
    Perhaps put all methods that act only on one dataset into here
    """
    name: str
    datapath: str  # path to root file(s)
    TTree_name: str  # name of TTree to extract
    is_slices: bool = False  # whether input data is in mass slices
    rebuild: bool = False  # whether dataframe should be rebuilt
    # field(repr=False) stops these vars showing up when .__repr__() is called
    cutflow: Cutflow = field(default=None, repr=False)  # stores cutflow object
    df: pd.DataFrame = field(default=None, repr=False)  # stores dataframe
    pkl_path: str = None  # where the dataframe pickle file will be stored
    cross_section: float = None
    luminosity: float = None

    def __post_init__(self):
        """ this all runs after the __init__ """
        # initialise pickle filepath
        if not self.pkl_path:
            self.pkl_path = config.pkl_df_filepath.format(self.name)

    # Variable setting
    # ===================
    def gen_cross_section(self) -> None:
        self.__check_df()
        self.cross_section = df_utils.get_cross_section(self.df)

    def gen_luminosity(self) -> None:
        self.__check_df()
        self.luminosity = df_utils.get_luminosity(self.df, xs=self.cross_section)

    def gen_cutflow(self, cut_dicts: List[Dict],
                    cutgroups: Optional[OrderedDict[str, List[str]]],
                    cut_label: str = ' CUT', sequential: bool = True,
                    ) -> None:
        """Creates the cutflow class for this analysis"""
        self.cutflow = Cutflow(self.df,
                               cut_dicts=cut_dicts, cutgroups=cutgroups, cut_label=cut_label,
                               sequential=sequential)

    # Dataframe utils
    # ===================
    def create_cut_columns(self, cut_dicts: List[Dict], cut_label: str = ' CUT', printout=True) -> None:
        df_utils.create_cut_columns(self.df, cut_dicts, cut_label, printout)

    # Private
    # ===================
    def __check_df(self) -> None:
        """Makes sure dataframe exists before trying to apply methods to it"""
        if self.df is None:
            raise Exception(f"Dataframe for {self.name} has yet to be generated.")
