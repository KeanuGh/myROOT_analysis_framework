"""
Defines helper functions to calculate various kinematic/other physical variables
"""
from typing import Dict, TypedDict, Callable, List

import numpy as np


# VARIABLE FUNCTIONS
# ================================
def calc_mt(l1_pt: float, l2_pt: float, l1_phi: float, l2_phi: float) -> float:
    """Calculate transverse mass of vector boson in Drell-Yan process"""
    return np.sqrt(2. * l1_pt * l2_pt * (1 - np.cos(l1_phi - l2_phi)))


def calc_vy(x1: float, x2: float) -> float:
    """Calculate boson rapidity"""
    return .5 * np.log(x1 / x2)


# VARIABLE BUILDING DICTIONARY
# ================================
class OtherVar(TypedDict):
    """Define type hint for other_vars nested dictionary"""
    var_args: List[str]
    func: Callable


# this dictionary of special variables aren't in the usual root ntuples
# var_args is a list of the ntuple variables needed to calculate, to be passed to the 'calc' function
derived_vars: Dict[str, OtherVar] = {
    'mu_mt': {  # boson pt from muon decay
        'var_args': ['mu_pt', 'met_met', 'mu_phi', 'met_phi'],
        'func': calc_mt,
    },
    'e_mt': {  # boson pt from electron decay
        'var_args': ['e_pt', 'met_met', 'e_phi', 'met_phi'],
        'func': calc_mt,
    },
    'w_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'func': calc_vy,
    },
    'z_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'func': calc_vy,
    },
    'v_y': {  # boson rapidity
        'var_args': ['PDFinfo_X1', 'PDFinfo_X2'],
        'func': calc_vy,
    },
}
