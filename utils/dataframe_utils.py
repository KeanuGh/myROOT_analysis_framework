import logging
import operator as op
import time
from glob import glob
from typing import List, OrderedDict

import pandas as pd
import uproot

import src.config as config
from utils.axis_labels import labels_xs
from utils.cutfile_utils import extract_cut_variables, all_vars, gen_alt_tree_dict
from utils.var_helpers import derived_vars

logger = logging.getLogger('analysis')


def build_analysis_dataframe(datapath: str,
                             TTree_name: str,
                             cut_list_dicts: List[dict],
                             vars_to_cut: List[str],
                             is_slices: bool = False,
                             pkl_path: str = None,
                             event_n_col: str = 'eventNumber'
                             ) -> pd.DataFrame:
    """
    Builds a dataframe from cutfile inputs
    :param datapath: path to root file
    :param TTree_name: name of TTree to extract
    :param cut_list_dicts: list of cut dictionaries
    :param vars_to_cut: list of strings of variables in file to cut on
    :param is_slices: whether or not data is in mass slices
    :param pkl_path: path to output pickle file (optional)
    :param event_n_col: name of event number variable in root file for merging across TTrees if necessary
    :return: output dataframe containing columns corresponding to necessary variables
    """
    chunksize = 2560

    # create list of all necessary values extract
    default_tree_vars = extract_cut_variables(cut_list_dicts, vars_to_cut)
    default_tree_vars.add('weight_mc')

    # get any variables that need to be calculated rather than extracted from ROOT file
    vars_to_calc = {calc_var for calc_var in default_tree_vars if calc_var in derived_vars}
    default_tree_vars -= vars_to_calc

    # get any variables in trees outside the default tree (TTree_name)
    alt_trees = gen_alt_tree_dict(cut_list_dicts)
    if alt_trees:
        default_tree_vars.add(event_n_col)  # need event number to merge across
        # remove alt tree variables from variables to extract from default tree
        default_tree_vars -= {var for varlist in alt_trees.values() for var in varlist}

    if logger.level == logging.DEBUG:
        logger.debug(f"Variables to extract from {TTree_name} tree: ")
        for var in default_tree_vars:
            logger.debug(f"  - {var}")
        if alt_trees:
            for tree in alt_trees:
                logger.debug(f"Variables to extract from {tree} tree: ")
                for var in alt_trees[tree]:
                    logger.debug(f"  - {var}")
        if vars_to_calc:
            logger.debug("Variables to calculate: ")
            for var in vars_to_calc:
                logger.debug(f"  - {var}")

    # check that TTree(s) and variables exist in file(s)
    logger.debug(f"Checking TTree and TBranch values in file(s) '{datapath}'...")
    for filepath in glob(datapath):
        with uproot.open(filepath) as file:
            tree_list = [tree.split(';')[0] for tree in file.keys()]  # TTrees are labelled '<name>;<cycle number>'
            if missing_trees := [t for t in [a_t for a_t in alt_trees] + [TTree_name] if t not in tree_list]:
                raise ValueError(f"TTree(s) {', '.join(missing_trees)} not found in file {filepath}")
            else:
                if missing_branches := [branch for branch in default_tree_vars
                                        if branch not in file[TTree_name].keys()]:
                    raise ValueError(f"Missing TBranch(es) {missing_branches} in TTree '{TTree_name}' of file '{datapath}'.")
        logger.debug(f"All TTrees and variables found in {filepath}...")
    logger.debug("All required TTrees variables found.")

    # check if vars are contained in label dictionary
    if unexpected_vars := [unexpected_var for unexpected_var in default_tree_vars
                           if unexpected_var not in labels_xs]:
        logger.warning(f"Variable(s) {unexpected_vars} not contained in labels dictionary. "
                       "Some unexpected behaviour may occur.")

    t1 = time.time()
    # extract pandas dataframe from root file with necessary variables
    if not is_slices:  # If importing inclusive sample
        logger.info(f"Extracting {default_tree_vars} from {datapath}...")
        df = uproot.concatenate(datapath + ':' + TTree_name, default_tree_vars,
                                library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)
        logger.debug(f"Extracted {len(df)} events.")

        if alt_trees:
            for tree in alt_trees:
                logger.debug(f"Extracting {alt_trees[tree]} from TTree {tree}...")
                alt_df = uproot.concatenate(datapath + ":" + tree, alt_trees[tree] + [event_n_col],
                                            library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)

                logger.debug("Merging with rest of dataframe...")
                df = pd.merge(df, alt_df, how='left', on=event_n_col, sort=False)

    else:  # if importing mass slices
        logger.info(f"Extracting mass slices from {datapath}...")
        default_tree_vars.add('mcChannelNumber')  # to keep track of dataset IDs (DSIDs)

        logger.debug(f"Extracting {default_tree_vars} from {TTree_name} tree...")
        df = uproot.concatenate(datapath + ':' + TTree_name, default_tree_vars, library='pd',
                                num_workers=config.n_threads, begin_chunk_size=chunksize)
        logger.debug(f"Extracted {len(df)} events.")

        logger.debug(f"Extracting ['total_EventsWeighted', 'dsid'] from 'sumWeights' tree...")
        sumw = uproot.concatenate(datapath + ':sumWeights', ['totalEventsWeighted', 'dsid'],
                                  library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)

        logger.debug(f"Calculating sum of weights and merging...")
        sumw = sumw.groupby('dsid').sum()
        df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False)
        df.rename(columns={'mcChannelNumber': 'DSID'}, inplace=True)  # rename mcChannelNumber to DSID (why are they different)

        if alt_trees:
            for tree in alt_trees:
                logger.debug(f"Extracting {alt_trees[tree]} from {tree} tree...")
                alt_df = uproot.concatenate(datapath + ":" + tree, alt_trees[tree] + [event_n_col],
                                            library='pd', num_workers=config.n_threads, begin_chunk_size=chunksize)

                logger.debug("Merging with rest of dataframe...")
                df = pd.merge(df, alt_df, how='left', on=event_n_col, sort=False)

        if logger.level == logging.DEBUG:
            # sanity check to make sure totalEventsWeighted really is what it says it is
            # also output DSID metadata
            dsids = df['DSID'].unique()
            logger.debug(f"Found {len(dsids)} unique dsid(s).")
            df_id_sub = df[['weight_mc', 'DSID']].groupby('DSID', as_index=False).sum()
            for dsid in dsids:
                df_id = df[df['DSID'] == dsid]
                unique_totalEventsWeighted = df_id['totalEventsWeighted'].unique()
                if len(unique_totalEventsWeighted) != 1:
                    logger.warning("totalEventsWeighted should only have one value per DSID. "
                                   f"Got {len(unique_totalEventsWeighted)}, of values {unique_totalEventsWeighted} for DSID {dsid}")

                dsid_weight = df_id_sub[df_id_sub['DSID'] == dsid]['weight_mc'].values[0]  # just checked there's only one value here
                totalEventsWeighted = df_id['totalEventsWeighted'].values[0]
                if dsid_weight != totalEventsWeighted:
                    logger.warning(f"Value of 'totalEventsWeighted' ({totalEventsWeighted}) is not the same as the total summed values of "
                                   f"'weight_mc' ({dsid_weight}) for DISD {dsid}. Ratio = {totalEventsWeighted / dsid_weight:.2g}")

        default_tree_vars.remove('mcChannelNumber')
        default_tree_vars.add('DSID')
    t2 = time.time()
    logger.info(f"time to build dataframe: {t2 - t1:.2g}s")

    # calculate and combine special derived variables
    if vars_to_calc:

        def row_calc(deriv_var: str, row: pd.Series) -> pd.Series:
            """Helper for applying derived variable calculation function to a dataframe row"""
            row_args = [row[v] for v in derived_vars[deriv_var]['var_args']]
            return derived_vars[deriv_var]['func'](*row_args)

        # save which variables are actually necessary in order to drop extras
        og_vars = all_vars(cut_list_dicts, vars_to_cut)
        for var in vars_to_calc:
            # compute new column
            temp_cols = derived_vars[var]['var_args']
            logger.info(f"Computing '{var}' column from {temp_cols}...")
            df[var] = df.apply(lambda row: row_calc(var, row), axis=1)

            # drop unnecessary columns extracted just for calculations
            to_drop = [var for var in temp_cols if var not in og_vars]
            logger.debug(f"dropping {to_drop}")
            df.drop(columns=to_drop, inplace=True)

    # properly scale GeV columns
    df = rescale_to_gev(df)

    # print into pickle file for easier read/write
    if pkl_path:
        pd.to_pickle(df, pkl_path)
        logger.info(f"Dataframe built and saved in {pkl_path}")

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
                             global_scale: float = config.lumi,
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
        logger.debug(f"Rescaled columns {GeV_columns} to GeV.")
    else:
        logger.debug(f"No columns rescaled to GeV.")
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
    return df[weight_mc_col].sum() / n_events


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
                       ) -> None:
    """
    Creates boolean columns in dataframe corresponding to each cut
    :param df: input dataframe
    :param cut_dicts: list of dictionaries for each cut to apply
    :return: None, this function applies inplace.
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

    # print cutflow output
    name_len = max([len(cut['name']) for cut in cut_dicts])
    var_len = max([len(cut['cut_var']) for cut in cut_dicts])
    logger.info('')
    logger.info("========== CUTS USED ============")
    for cut in cut_dicts:
        if not cut['is_symmetric']:
            logger.info(f"{cut['name']:<{name_len}}: {cut['cut_var']:>{var_len}} {cut['relation']} {cut['cut_val']}")
        else:
            logger.info(f"{cut['name']:<{name_len}}: {'|'+cut['cut_var']:>{var_len}}| {cut['relation']} {cut['cut_val']}")
    logger.info('')


def cut_on_cutgroup(df: pd.DataFrame,
                    cutgroups: OrderedDict[str, List[str]],
                    group: str,
                    ) -> pd.DataFrame:
    """Cuts on cutgroup on input dataframe or series"""
    cut_cols = [cut_name + config.cut_label for cut_name in cutgroups[group]]
    cut_data = df[df[cut_cols].all(1)]
    return cut_data
