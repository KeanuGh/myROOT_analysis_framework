"""
Defines helper functions to calculate various kinematic/other physical variables
"""
from typing import Dict, TypedDict, Callable, List

import numpy as np
import pandas as pd


# VARIABLE FUNCTIONS
# ================================
def calc_mt(df: pd.DataFrame, l1_pt: str, l2_pt: str, l1_phi: str, l2_phi: str) -> pd.Series:
    """Calculate transverse mass of vector boson in Drell-Yan process"""
    dphi = abs(df[l1_phi] - df[l2_phi])
    dphi.loc[dphi > np.pi] = 2 * np.pi - dphi.loc[dphi > np.pi]
    return np.sqrt(2. * df[l1_pt] * df[l2_pt] * (1 - np.cos(dphi)))


def calc_vy(df: pd.DataFrame, x1: str, x2: str) -> pd.Series:
    """Calculate boson rapidity"""
    return .5 * np.log(df[x1] / df[x2])


# VARIABLE BUILDING DICTIONARY
# ================================
class OtherVar(TypedDict):
    """Define type hint for other_vars nested dictionary"""
    var_args: List[str]
    tree: str
    func: Callable


# this dictionary of special variables aren't in the usual root ntuples
# var_args is a list of the ntuple variables needed to calculate, to be passed to the 'calc' function
derived_vars: Dict[str, OtherVar] = {
    'mt': {  # boson mt
        # 'var_args': ['e_pt', 'met_met', 'e_phi', 'met_phi'],
        'var_args': ['MC_WZmu_el_pt_born', 'MC_WZneutrino_pt_born', 'MC_WZmu_el_phi_born', 'MC_WZneutrino_phi_born'],
        'tree': 'truth',
        'func': calc_mt,
    },
    'w_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'tree': 'truth',
        'func': calc_vy,
    },
    'z_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'tree': 'truth',
        'func': calc_vy,
    },
    'v_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'tree': 'truth',
        'func': calc_vy,
    },
}
