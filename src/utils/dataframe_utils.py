import pandas as pd
from typing import Optional
import uproot4 as uproot
import analysis.config as config
from utils.cutfile_utils import extract_cut_variables
from utils.axis_labels import labels_xs
from typing import List, OrderedDict
from warnings import warn
import time


def build_analysis_dataframe(data,  # This type hint is left blank to avoid a circular import
                             cut_list_dicts: List[dict],
                             vars_to_cut: List[str],
                             extra_vars: Optional[List[str]] = None
                             ) -> pd.DataFrame:
    """
    Builds a dataframe from cutfile inputs
    :param data: Dataset class containing is_slices, TTree_name, datapath and pkl_path
    :param cut_list_dicts: list of cut dictionaries
    :param vars_to_cut: list of strings of variables in file to cut on
    :param extra_vars: list of any extra variables wanting to extract
    :return: output dataframe containing columns corresponding to necessary variables
    """
    print("Extracting variables:\n{}".format('\n'.join(vars_to_cut)))
    # create list of all necessary values extract
    vars_to_extract = extract_cut_variables(cut_list_dicts, vars_to_cut)
    # strictly necessary variable(s)
    vars_to_extract.append('weight_mc')
    # extras
    if extra_vars:
        vars_to_extract += extra_vars

    t1 = time.time()
    # If importing inclusive sample
    if not data.is_slices:
        # extract pandas dataframe from root file with necessary variables
        tree_df = uproot.concatenate(data.datapath + ':' + data.TTree_name,
                                     filter_name=vars_to_extract,
                                     library='pd')
    # if importing mass slices
    else:
        vars_to_extract += ['mcChannelNumber']  # to keep track of dataset IDs (DSIDs)
        tree_df = uproot.concatenate(data.datapath + ':' + data.TTree_name, filter_name=vars_to_extract, library='pd')
        sumw = uproot.concatenate(data.datapath + ':sumWeights', filter_name=['totalEventsWeighted', 'dsid'], library='pd')
        sumw = sumw.groupby('dsid').sum()
        tree_df = pd.merge(tree_df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False)

        # properly name DSID column
        tree_df.rename(columns={'mcChannelNumber': 'DSID'}, inplace=True)
        vars_to_extract = vars_to_extract[:-1] + ['DSID']
    t2 = time.time()
    print(f"time to build dataframe: {t2 - t1:.2f}s")

    # check vars exist in file
    if unexpected_vars := [unexpected_var for unexpected_var in vars_to_extract
                           if unexpected_var not in tree_df.columns]:
        raise ValueError(f"Variable(s) not found in root file '{data.datapath}' {data.TTree_name} tree: {unexpected_vars}")

    # check if vars are contained in label dictionary
    if unexpected_vars := [unexpected_var for unexpected_var in vars_to_extract
                           if unexpected_var not in labels_xs]:
        warn(f"Warning: variable(s) {unexpected_vars} not contained in labels dictionary. "
             f"Some unexpected behaviour may occur.")

    tree_df = rescale_to_gev(tree_df)  # cleanup

    # print into pickle file for easier read/write
    if data.pkl_path:
        pd.to_pickle(tree_df, data.pkl_path)
        print(f"Dataframe built and saved in {data.pkl_path}")

    return tree_df


def gen_weight_column(df: pd.DataFrame,
                      weight_mc_col: str = 'weight_mc',
                      global_scale: float = 1.
                      ) -> pd.Series:
    """Returns series of weights based off weight_mc column"""
    if weight_mc_col not in df.columns:
        raise KeyError(f"'{weight_mc_col}' column does not exist.")
    return df[weight_mc_col].map(lambda w: global_scale if w > 0 else -1 * global_scale)


def gen_weight_column_slices(df: pd.DataFrame,
                             weight_mc_col: str = 'weight_mc',
                             tot_weighted_events_col: str = 'totalEventsWeighted',
                             global_scale: float = 1.,
                             ) -> pd.Series:
    """Returns series of weights for mass slices based off weight_mc column and total events weighed"""
    if weight_mc_col not in df.columns:
        raise KeyError(f"'{weight_mc_col}' column not in dataframe.")
    if tot_weighted_events_col not in df.columns:
        raise KeyError(f"'{tot_weighted_events_col}' column not in dataframe.")
    return global_scale * (df['weight_mc'] * df['weight_mc'].abs()) / df[tot_weighted_events_col]


def rescale_to_gev(df: pd.DataFrame) -> pd.DataFrame:
    """rescales to GeV because athena's default output is in MeV for some reason"""
    GeV_columns = [column for column in df.columns
                   if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]
    print(f"Rescaling {len(GeV_columns)} columns to GeV...")
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
        xs = get_cross_section(df)
    return df[weight_col].sum() / xs


def create_cut_columns(df: pd.DataFrame,
                       cut_dicts: List[dict],
                       printout=True
                       ) -> None:
    """
    Creates boolean columns in dataframe corresponding to each cut
    :param df: input dataframe
    :param cut_dicts: list of dictionaries for each cut to apply
    :return: None, this function applies inplace.
    :param printout: whether to print a summary of cuts
    """
    cut_label = config.cut_label  # get cut label from config

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
                    ) -> pd.DataFrame:
    """Cuts on cutgroup on input dataframe or series"""
    cut_rows = [cut_name + config.cut_label for cut_name in cutgroups[group]]
    cut_data = df[df[cut_rows].all(1)]
    return cut_data
