"""
Defines helper functions to calculate various kinematic/other physical variables
"""
from typing import Dict, TypedDict, List


# VARIABLE BUILDING DICTIONARY
# ================================
class OtherVar(TypedDict):
    """Define type hint for other_vars nested dictionary"""

    var_args: List[str]
    tree: str
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
        "cfunc": "mt",
    },
    "w_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
        "cfunc": "vy",
    },
    "z_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
        "cfunc": "vy",
    },
    "v_y": {  # boson rapidity
        "var_args": [
            "PDFinfo_X1",
            "PDFinfo_X2",
        ],
        "tree": "truth",
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
        "cfunc": "mt",
    },
    "TruthMTW": {  # truth boson mt
        "var_args": [
            "TruthTauPt",
            "TruthNeutrinoPt",
            "TruthTauPhi",
            "TruthNeutrinoPhi",
        ],
        "tree": "",
        "cfunc": "mt",
    },
    "Muon_delta_z0_sintheta": {
        "var_args": [
            "Muon_delta_z0",
            "MuonEta",
        ],
        "tree": "",
        "cfunc": "delta_z0_sintheta",
    },
    "Ele_delta_z0_sintheta": {
        "var_args": [
            "Ele_delta_z0",
            "EleEta",
        ],
        "tree": "",
        "cfunc": "delta_z0_sintheta",
    },
    "DilepM": {
        "var_args": [
            "TruthTauE",
            "TruthNeutrinoE",
        ],
        "tree": "",
        "cfunc": "dilep_m",
    },
    "DeltaR_tau_mu": {
        "var_args": [
            "TauEta",
            "MuonEta",
            "TauPhi",
            "MuonPhi",
        ],
        "tree": "",
        "cfunc": "delta_r",
    },
    "DeltaR_tau_e": {
        "var_args": [
            "TauEta",
            "EleEta",
            "TauPhi",
            "ElePhi",
        ],
        "tree": "",
        "cfunc": "delta_r",
    },
    "DeltaR_e_mu": {
        "var_args": [
            "EleEta",
            "MuonEta",
            "ElePhi",
            "TauPhi",
        ],
        "tree": "",
        "cfunc": "delta_r",
    },
    "TauPt_div_MET": {
        "var_args": [
            "TauPt",
            "MET_met",
        ],
        "tree": "",
        "cfunc": "calc_div",
    },
    "TruthTauPt_div_MET": {
        "var_args": [
            "TruthTauPt",
            "TruthNeutrinoPt",
        ],
        "tree": "",
        "cfunc": "calc_div",
    },
    "DeltaPhi_tau_met": {
        "var_args": [
            "TauPhi",
            "MET_phi",
        ],
        "tree": "",
        "cfunc": "calc_dphi",
    },
    "TruthDeltaPhi_tau_met": {
        "var_args": [
            "TruthTauPhi",
            "TruthNeutrinoPhi",
        ],
        "tree": "",
        "cfunc": "calc_dphi",
    },
    "TauPt_res": {
        "var_args": [
            "MatchedTruthParticlePt",
            "TauPt",
        ],
        "tree": "",
        "cfunc": "calc_frac",
    },
    "TauPt_diff": {
        "var_args": [
            "MatchedTruthParticlePt",
            "TauPt",
        ],
        "tree": "",
        "cfunc": "calc_diff",
    },
    "MatchedTruthParticle_isJet": {
        "var_args": [
            "MatchedTruthParticle_isTau",
            "MatchedTruthParticle_isElectron",
            "MatchedTruthParticle_isMuon",
            "MatchedTruthParticle_isPhoton",
        ],
        "tree": "",
        "cfunc": "is_not",
    },
    "nJets": {
        "var_args": ["JetPt"],
        "tree": "",
        "cfunc": "n_vec",
    },
    "LeadingJetPt": {"var_args": ["JetPt"], "tree": "", "cfunc": "lead_val"},
}
