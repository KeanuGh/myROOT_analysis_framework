import pandas as pd


def get_cross_section(df: pd.DataFrame, n_events=None, weight_mc_col: str = "weight_mc"):
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


def get_luminosity(df: pd.DataFrame, xs=None, weight_col: str = "weight"):
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
