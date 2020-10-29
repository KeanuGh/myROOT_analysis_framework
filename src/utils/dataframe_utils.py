import pandas as pd
from typing import Optional
import uproot4 as uproot
from utils.cutfile_utils import extract_cut_variables
from utils.axis_labels import labels_xs
from typing import List, OrderedDict, Union
from warnings import warn


# TODO: write custom dataframe class that can be either pandas or dask
def build_analysis_dataframe(cut_list_dicts: List[dict],
                             vars_to_cut: List[str],
                             root_filepath: str,
                             TTree: str,
                             pkl_filepath: Optional[str] = None,
                             extra_vars: Optional[List[str]] = None
                             ) -> pd.DataFrame:
    """
    Builds a dataframe from _cutfile inputs
    :param cut_list_dicts: list of cut dictionaries
    :param vars_to_cut: list of strings of variables in file to cut on
    :param root_filepath: path to input root file
    :param TTree: name of TTree to extract ntuples from
    :param pkl_filepath: path of pickle file to save dataframe. if None, does not save
    :param extra_vars: list of any extra variables wanting to extract
    :return: output dataframe containing columns corresponding to necessary variables
    """
    print("Building Dataframe...")
    # create list of all necessary values extract
    vars_to_extract = extract_cut_variables(cut_list_dicts, vars_to_cut)
    # strictly necessary variable(s)
    vars_to_extract.append('weight_mc')
    # extras
    if extra_vars:
        vars_to_extract += extra_vars

    # extract pandas dataframe from root file with necessary variables
    tree_gen = uproot.iterate(root_filepath+':'+TTree, branches=vars_to_extract, library='pd')

    # extract variables from tree
    tree_df = pd.concat([data for data in tree_gen])

    # check vars exist in file
    if unexpected_vars := [unexpected_var for unexpected_var in vars_to_extract
                           if unexpected_var not in tree_df.columns]:
        raise ValueError(f"Variable(s) not found in root file '{root_filepath}' {TTree} tree: {unexpected_vars}")

    # check if vars are contained in label dictionary
    if unexpected_vars := [unexpected_var for unexpected_var in vars_to_extract
                           if unexpected_var not in labels_xs]:
        warn(f"Warning: variable(s) {unexpected_vars} not contained in labels dictionary. "
             f"Some unexpected behaviour may occur")

    # print into pickle file for easier read/write
    if pkl_filepath:
        pd.to_pickle(tree_df, pkl_filepath)
        print(f"Dataframe built and saved in {pkl_filepath}")

    return tree_df


# TODO: WRAP THESE INTO A CUSTOM DATAFRAME CLASS?
def gen_weight_column(df: pd.DataFrame,
                      weight_mc_col: str = 'weight_mc',
                      scale: float = 1
                      ) -> pd.Series:
    """Returns series of weights based off weight_mc column"""
    if weight_mc_col not in df.columns:
        raise ValueError(f"'{weight_mc_col}' column does not exist.")
    return df[weight_mc_col].map(lambda w: scale if w > 0 else -1 * scale)


def rescale_to_gev(df: pd.DataFrame) -> pd.DataFrame:
    """rescales to GeV because athena's default output is in MeV for some reason"""
    GeV_columns = [column for column in df.columns
                   if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]
    df[GeV_columns] /= 1000
    return df


def get_cross_section(df: pd.DataFrame, n_events=None, weight_mc_col: str = 'weight_mc'):
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
        xs = get_cross_section(df)

    return df[weight_col].sum() / xs


def create_cut_columns(df: pd.DataFrame,
                       cut_dicts: List[dict],
                       cut_label: str = ' CUT',
                       printout=True
                       ) -> None:
    """
    Creates boolean columns in dataframe corresponding to each cut
    :param df: input dataframe
    :param cut_dicts: list of dictionaries for each cut to apply
    :param cut_label: label to be added to column names for boolean columns
    :return: None, this function applies inplace.
    :param printout: whether to print a summary of cuts
    """
    print("applying cuts...")
    for cut in cut_dicts:
        if not cut['is_symmetric']:
            if cut['moreless'] == '>':
                df[cut['name'] + cut_label] = df[cut['cut_var']] > cut['cut_val']
            elif cut['moreless'] == '<':
                df[cut['name'] + cut_label] = df[cut['cut_var']] < cut['cut_val']
            else:
                raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")

        else:
            # take absolute value instead
            if cut['moreless'] == '>':
                df[cut['name'] + cut_label] = df[cut['cut_var']].abs() > cut['cut_val']
            elif cut['moreless'] == '<':
                df[cut['name'] + cut_label] = df[cut['cut_var']].abs() < cut['cut_val']
            else:
                raise ValueError(f"Unexpected comparison operator: {cut['moreless']}. Currently accepts '>' or '<'.")

    if printout:
        # print cutflow output
        name_len = max([len(cut['name']) for cut in cut_dicts])
        var_len = max([len(cut['cut_var']) for cut in cut_dicts])
        print("\n========== CUTS USED ============")
        for cut in cut_dicts:
            if not cut['is_symmetric']:
                print(f"{cut['name']:<{name_len}}: "
                      f"{cut['cut_var']:>{var_len}} {cut['moreless']} {cut['cut_val']}")
            else:
                print(f"{cut['name']:<{name_len}}: "
                      f"{cut['cut_var']:>{var_len}} {cut['moreless']} |{cut['cut_val']}|")
        print('\n')


def cut_on_cutgroup(df: pd.DataFrame,
                    cutgroups: OrderedDict[str, List[str]],
                    group: str,
                    cut_label: str,
                    ) -> pd.DataFrame:
    """Cuts on cutgroup on input dataframe or series"""
    cut_rows = [cut_name + cut_label for cut_name in cutgroups[group]]
    cut_data = df[df[cut_rows].any(1)]
    return cut_data
