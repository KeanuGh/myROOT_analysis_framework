"""
Defines helper functions to calculate various kinematic/other physical variables
"""
from typing import Dict, TypedDict, Callable, List

import numpy as np


# VARIABLE FUNCTIONS
# ================================
def calc_mt(l1_pt: float, l2_pt: float, l1_phi: float, l2_phi: float) -> float:
    """Calculate transverse mass of vector boson in Drell-Yan process"""
    dphi = abs(l2_phi - l1_phi)
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi
    return np.sqrt(2. * l1_pt * l2_pt * (1 - np.cos(dphi)))


def calc_vy(x1: float, x2: float) -> float:
    """Calculate boson rapidity"""
    return .5 * np.log(x1 / x2)


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
