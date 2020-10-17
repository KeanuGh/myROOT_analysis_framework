import uproot4 as uproot
from utils.cutfile_parser import extract_cut_variables
import pandas as pd
from typing import List


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

    # extract variables from tree
    tree_df = tree.arrays(library='pd', filter_name=vars_to_extract)

    # print into pickle file for easier read/write
    pd.to_pickle(tree_df, pkl_filepath)
    print(f"Dataframe built and saved in {pkl_filepath}")

    return tree_df
