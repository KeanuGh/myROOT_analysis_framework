import pandas as pd
from typing import Optional
from utils.axis_labels import labels_xs
import uproot4 as uproot
from utils.cutfile_parser import extract_cut_variables
from utils.axis_labels import labels_xs
from typing import List
from warnings import warn


def build_analysis_dataframe(cut_list_dicts: List[dict], vars_to_cut: List[str],
                             input_root_file: str, TTree_name: str, pkl_filepath: str) -> pd.DataFrame:
    """
    :param cut_list_dicts:
    :param vars_to_cut:
    :param input_root_file:
    :param TTree_name:
    :param pkl_filepath:
    :return: output dataframe containing columns
    """
    # create list of all necessary values extract
    vars_to_extract = extract_cut_variables(cut_list_dicts, vars_to_cut)
    # strictly necessary variables
    vars_to_extract.append('weight_mc')

    # extract pandas dataframe from root file with necessary variables
    tree = uproot.open(input_root_file)[TTree_name]

    # check vars exist in file
    unexpected_vars = [unexpected_var for unexpected_var in vars_to_extract if unexpected_var not in tree.keys()]
    if unexpected_vars:
        raise ValueError(f"Variable not found in root file '{input_root_file}' {TTree_name} tree: {unexpected_vars}")

    # check if vars are contained in label dictionary
    unexpected_vars = [unexpected_var for unexpected_var in vars_to_extract if unexpected_var not in labels_xs]
    if unexpected_vars:
        warn(f"Warning: variable(s) {unexpected_vars} not contained in labels dictionary. "
             f"Some undexpected behaviour may occur")

    # extract variables from tree
    tree_df = tree.arrays(library='pd', filter_name=vars_to_extract)

    # print into pickle file for easier read/write
    pd.to_pickle(tree_df, pkl_filepath)
    print(f"Dataframe built and saved in {pkl_filepath}")

    return tree_df


# TODO: WRAP THESE INTO A CUSTOM DATAFRAME CLASS?
def gen_weight_column(df: pd.DataFrame, weight_mc_col: str = 'weight_mc', scale: float = 1) -> pd.Series:
    """Returns series of weights based off weight_mc column"""
    if weight_mc_col not in df.columns:
        raise ValueError(f"'{weight_mc_col}' column does not exist.")
    return df[weight_mc_col].map(lambda w: scale if w > 0 else -1 * scale)


def rescale_to_GeV(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """rescales to GeV because athena's default output is in MeV for some reason"""
    GeV_columns = [column for column in df.columns
                   if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]
    df[GeV_columns] /= 1000
    if not inplace:
        return df


def get_crosssection(df: pd.DataFrame, n_events = None, weight_mc_col: str = 'weight_mc'):
    """
    Calculates cross-section of data in dataframe
    :param df: input dataframe
    :param n_events: optional: total number of events. Will calculate if not given.
    :param weight_mc_col: column containing monte carlo weights
    :return: cross-section
    """
    if not n_events:
        n_events = len(df.index)
    return df[weight_mc_col].abs().sum() / n_events


def get_luminosity(df: pd.DataFrame, xs=None, weight_col: str = 'weight'):
    """
    Calculates luminosity from dataframe
    :param df: input dataframe
    :param xs: cross-section. If not given, will calculate
    :param weight_col: column of dataframe containing the weights
    :return: luminosity
    """
    if not xs:
        if 'weight_mc' not in df.columns:
            raise Exception("weight_mc column missing in dataframe")
        xs = get_crosssection(df)

    return df[weight_col].sum() / xs
