"""
Defines helper functions to calculate various kinematic/other physical variables
"""
from typing import Dict, TypedDict, Callable, List

import numpy as np
import pandas as pd  # type: ignore


# VARIABLE FUNCTIONS
# ================================
def calc_mt(df: pd.DataFrame, l1_pt: str, l2_pt: str, l1_phi: str, l2_phi: str) -> pd.Series:
    """Calculate transverse mass of vector boson in Drell-Yan process"""
    dphi = abs(df[l1_phi] - df[l2_phi])
    dphi.loc[dphi > np.pi] = 2 * np.pi - dphi.loc[dphi > np.pi]
    return np.sqrt(2.0 * df[l1_pt] * df[l2_pt] * (1 - np.cos(dphi)))


def calc_vy(df: pd.DataFrame, x1: str, x2: str) -> pd.Series:
    """Calculate boson rapidity"""
    return 0.5 * np.log(df[x1] / df[x2])


def calc_delta_z0_sintheta(df: pd.DataFrame, z0: str, eta: str) -> pd.Series:
    return df[z0] * np.sin(2 * np.arctan(np.exp(-df[eta])))


def calc_dilep_m(df: pd.DataFrame, m1: str, m2: str) -> pd.Series:
    return df[m1] + df[m2]


# VARIABLE BUILDING DICTIONARY
# ================================
class OtherVar(TypedDict):
    """Define type hint for other_vars nested dictionary"""

    var_args: List[str]
    tree: str
    func: Callable
    cfunc: str


# this dictionary of special variables aren't in the usual root ntuples
# var_args is a list of the ntuple variables needed to calculate, to be passed to the 'calc' function
derived_vars: Dict[str, OtherVar] = {
    # analysistop
    "e_mt_reco": {
        "var_args": [
            "e_pt",
            "met_met",
            "e_phi",
            "met_phi",
        ],
        "tree": "nominal_Loose",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "mu_mt_reco": {
        "var_args": [
            "mu_pt",
            "met_met",
            "mu_phi",
            "met_phi",
        ],
        "tree": "nominal_Loose",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "mt_born": {  # boson mt
        "var_args": [
            "MC_WZmu_el_pt_born",
            "MC_WZneutrino_pt_born",
            "MC_WZmu_el_phi_born",
            "MC_WZneutrino_phi_born",
        ],
        "tree": "truth",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "mt_bare": {  # boson mt
        "var_args": [
            "MC_WZmu_el_pt_bare",
            "MC_WZneutrino_pt_bare",
            "MC_WZmu_el_phi_bare",
            "MC_WZneutrino_phi_bare",
        ],
        "tree": "truth",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "mt_dres": {  # boson mt
        "var_args": [
            "MC_WZmu_el_pt_dres",
            "MC_WZneutrino_pt_dres",
            "MC_WZmu_el_phi_dres",
            "MC_WZneutrino_phi_dres",
        ],
        "tree": "truth",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "w_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
        "func": calc_vy,
        "cfunc": "vy",
    },
    "z_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
        "func": calc_vy,
        "cfunc": "vy",
    },
    "v_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
        "func": calc_vy,
        "cfunc": "vy",
    },
    # DTA
    "MTW": {
        "var_args": [
            "TauPt",
            "MET_met",
            "TauPhi",
            "MET_phi",
        ],
        "tree": "",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "TruthMTW": {  # boson mt
        "var_args": [
            "TruthTauPt",
            "TruthNeutrinoPt",
            "TruthTauPhi",
            "TruthNeutrinoPhi",
        ],
        "tree": "",
        "func": calc_mt,
        "cfunc": "mt",
    },
    "Muon_delta_z0_sintheta": {
        "var_args": [
            "Muon_delta_z0",
            "MuonEta",
        ],
        "tree": "",
        "func": calc_delta_z0_sintheta,
        "cfunc": "delta_z0_sintheta",
    },
    "Ele_delta_z0_sintheta": {
        "var_args": [
            "Ele_delta_z0",
            "EleEta",
        ],
        "tree": "",
        "func": calc_delta_z0_sintheta,
        "cfunc": "delta_z0_sintheta",
    },
    "DilepM": {
        "var_args": [
            "TruthTauE",
            "TruthNeutrinoE",
        ],
        "tree": "",
        "func": calc_dilep_m,
        "cfunc": "dilep_m",
    },
}
