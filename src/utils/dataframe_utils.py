import pandas as pd
from typing import Optional
from utils.axis_labels import labels_xs


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

# TODO: Check that variables being extracted are contained in the axis labels dictionary, or some weird behaviour
#  will occur.
