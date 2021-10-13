import operator as op
import time
from glob import glob
from typing import List, OrderedDict
from warnings import warn

import pandas as pd
import uproot

import src.config as config
from utils.axis_labels import labels_xs
from utils.cutfile_utils import extract_cut_variables, all_vars
from utils.var_helpers import derived_vars


def build_analysis_dataframe(datapath: str,
                             TTree_name: str,
                             cut_list_dicts: List[dict],
                             vars_to_cut: List[str],
                             is_slices: bool = False,
                             pkl_path: str = None,
                             ) -> pd.DataFrame:
    """
    Builds a dataframe from cutfile inputs
    :param datapath: path to root file
    :param TTree_name: name of TTree to extract
    :param cut_list_dicts: list of cut dictionaries
    :param vars_to_cut: list of strings of variables in file to cut on
    :param is_slices: whether or not data is in mass slices
    :param pkl_path: path to output pickle file (optional)
    :return: output dataframe containing columns corresponding to necessary variables
    """

    # create list of all necessary values extract
    vars_to_extract = extract_cut_variables(cut_list_dicts, vars_to_cut)
    vars_to_extract.add('weight_mc')
    print("Variables to extract:{}".format('\n  -'.join(['', *vars_to_extract])))

    # check that TTree and variables exist in file(s)
    print(f"Checking TTree and TBranch values in file(s) '{datapath}'...")
    for filepath in glob(datapath):
        with uproot.open(filepath) as file:
            tree_list = [tree.split(';')[0] for tree in file.keys()]  # TTrees are labelled '<name>;<cycle number>'
            if TTree_name not in tree_list:
                raise ValueError(f"TTree '{TTree_name}' not found in file {filepath}.")
            else:
                if missing_branches := [branch for branch in vars_to_extract
                                        if branch not in file[TTree_name].keys()]:
                    raise ValueError(f"Missing TBranch(es) {missing_branches} in TTree '{TTree_name}' of file '{datapath}'.")
    print("All required variables found.")

    # check if vars are contained in label dictionary
    if unexpected_vars := [unexpected_var for unexpected_var in vars_to_extract
                           if unexpected_var not in labels_xs]:
        warn(f"Warning: variable(s) {unexpected_vars} not contained in labels dictionary."
             "Some unexpected behaviour may occur.")

    t1 = time.time()
    # extract pandas dataframe from root file with necessary variables
    print(f"Extracting data from {datapath}...")
    if not is_slices:  # If importing inclusive sample
        df = uproot.concatenate(datapath + ':' + TTree_name, vars_to_extract,
                                library='pd', num_workers=config.n_threads)

    else:  # if importing mass slices
        print("Extracting in slices...")
        vars_to_extract.add('mcChannelNumber')  # to keep track of dataset IDs (DSIDs)
        df = uproot.concatenate(datapath + ':' + TTree_name, vars_to_extract, library='pd',
                                num_workers=config.n_threads)
        sumw = uproot.concatenate(datapath + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                  library='pd', num_workers=config.n_threads)
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False)
        df.rename(columns={'mcChannelNumber': 'DSID'},
                  inplace=True)  # rename mcChannelNumber to DSID (why are they different)

        # sanity check to make sure totalEventsWeighted really is what it says it is
        # TODO: wrap this in a debug mode
        dsids = df['DSID'].unique()
        print(f"Found {len(dsids)} unique dsid(s).")
        df_id_sub = df[['weight_mc', 'DSID']].groupby('DSID', as_index=False).sum()
        for dsid in dsids:
            unique_totalEventsWeighted = df[df['DSID'] == dsid]['totalEventsWeighted'].unique()
            if len(unique_totalEventsWeighted) != 1:
                warn("totalEventsWeighted should only have one value per DSID. "
                     f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} for DSID {dsid}")

            dsid_weight = df_id_sub[df_id_sub['DSID'] == dsid]['weight_mc'].values[0]
            totalEventsWeighted = df[df['DSID'] == dsid]['totalEventsWeighted'].values[0]
            if dsid_weight != totalEventsWeighted:
                warn(f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                     f"'weight_mc' ({dsid_weight}) for DISD {dsid}.")

        vars_to_extract.remove('mcChannelNumber')
        vars_to_extract.add('DSID')
    t2 = time.time()
    print(f"time to build dataframe: {t2 - t1:.2g}s")

    # calculate and combine special derived variables
    if to_calc := [calc_var for calc_var in vars_to_cut if calc_var in derived_vars]:

        def row_calc(deriv_var: str, row: pd.Series) -> pd.Series:
            """Helper for applying derived variable calculation function to a dataframe row"""
            row_args = [row[v] for v in derived_vars[deriv_var]['var_args']]
            return derived_vars[deriv_var]['func'](*row_args)

        # save which variables are actually necessary in order to drop extras
        og_vars = all_vars(cut_list_dicts, vars_to_cut)
        for var in to_calc:
            # compute new column
            temp_cols = derived_vars[var]['var_args']
            print(f"Computing '{var}' column from {temp_cols}...")
            df[var] = df.apply(lambda row: row_calc(var, row), axis=1)

            # drop unnecessary columns extracted just for calculations
            to_drop = [var for var in temp_cols if var not in og_vars]
            print(f"dropping {to_drop}")
            df.drop(columns=to_drop, inplace=True)

    # properly scale GeV columns
    df = rescale_to_gev(df)

    # print into pickle file for easier read/write
    if pkl_path:
        pd.to_pickle(df, pkl_path)
        print(f"Dataframe built and saved in {pkl_path}")

    return df


def gen_weight_column(df: pd.DataFrame,
                      weight_mc_col: str = 'weight_mc',
                      global_scale: float = config.lumi
                      ) -> pd.Series:
    """Returns series of weights based off weight_mc column"""
    if weight_mc_col not in df.columns:
        raise KeyError(f"'{weight_mc_col}' column does not exist.")
    return df[weight_mc_col].map(lambda w: global_scale if w > 0 else -1 * global_scale)


def gen_weight_column_slices(df: pd.DataFrame,
                             mc_weight_col: str = 'weight_mc',
                             tot_weighted_events_col: str = 'totalEventsWeighted',
                             global_scale: float = 1.,
                             ) -> pd.Series:
    """Returns series of weights for mass slices based off weight_mc column and total events weighed"""
    # TODO: For efficiency, perform batchwise across DSID
    if mc_weight_col not in df.columns:
        raise KeyError(f"'{mc_weight_col}' column not in dataframe.")
    if tot_weighted_events_col not in df.columns:
        raise KeyError(f"'{tot_weighted_events_col}' column not in dataframe.")
    return global_scale * (df[mc_weight_col] * df[mc_weight_col].abs()) / df[tot_weighted_events_col]


def rescale_to_gev(df: pd.DataFrame) -> pd.DataFrame:
    """rescales to GeV because athena's default output is in MeV for some reason"""
    GeV_columns = [column for column in df.columns
                   if (column in labels_xs) and ('[GeV]' in labels_xs[column]['xlabel'])]
    df[GeV_columns] /= 1000
    if GeV_columns:
        print(f"Rescaled columns {GeV_columns} to GeV.")
    else:
        print(f"No columns rescaled to GeV.")
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

    # use functions of compariton operators in dictionary to make life easier
    # (but maybe a bit harder to read)
    op_dict = {
        '<': op.lt,
        '<=': op.le,
        '=': op.eq,
        '!=': op.ne,
        '>': op.gt,
        '>=': op.ge,
    }

    for cut in cut_dicts:
        if cut['relation'] not in op_dict:
            raise ValueError(f"Unexpected comparison operator: {cut['relation']}.")
        if not cut['is_symmetric']:
            df[cut['name'] + cut_label] = op_dict[cut['relation']](df[cut['cut_var']], cut['cut_val'])
        else:  # take absolute value instead
            df[cut['name'] + cut_label] = op_dict[cut['relation']](df[cut['cut_var']].abs(), cut['cut_val'])

    if printout:
        # print cutflow output
        name_len = max([len(cut['name']) for cut in cut_dicts])
        var_len = max([len(cut['cut_var']) for cut in cut_dicts])
        print("\n========== CUTS USED ============")
        for cut in cut_dicts:
            if not cut['is_symmetric']:
                print(f"{cut['name']:<{name_len}}: {cut['cut_var']:>{var_len}} {cut['relation']} {cut['cut_val']}")
            else:
                print(f"{cut['name']:<{name_len}}: {cut['cut_var']:>{var_len}} {cut['relation']} |{cut['cut_val']}|")
        print('')


def cut_on_cutgroup(df: pd.DataFrame,
                    cutgroups: OrderedDict[str, List[str]],
                    group: str,
                    ) -> pd.DataFrame:
    """Cuts on cutgroup on input dataframe or series"""
    cut_cols = [cut_name + config.cut_label for cut_name in cutgroups[group]]
    cut_data = df[df[cut_cols].all(1)]
    return cut_data
