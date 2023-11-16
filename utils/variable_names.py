"""
This file contains metadata about named branches in NTuples,
useful for the logic behind cuts and generating axis labels
"""
from enum import Enum
from typing import Dict, TypedDict


class VarTag(str, Enum):
    """Variable tag. A variable should be either truth, reconstructed or 'metadata'"""

    META = "meta"
    TRUTH = "truth"
    RECO = "reco"


class Branch(TypedDict):
    """Formatting for named branches in ROOT NTuples"""

    name: str
    units: str
    tag: VarTag


# labels for cross-sections
variable_data: Dict[str, Branch] = {
    # DATASET METADATA
    # =======================================================
    "weight_mc": {
        "name": r"MC weight",
        "units": "",
        "tag": VarTag.META,
    },
    "weight": {
        "name": r"weight",
        "units": "",
        "tag": VarTag.META,
    },
    "weight_KFactor": {
        "name": r"KFactor weight",
        "units": "",
        "tag": VarTag.META,
    },
    "KFactor_weight_truth": {
        "name": r"KFactor weight",
        "units": "",
        "tag": VarTag.META,
    },
    "weight_pileup": {
        "name": r"Pileup weight",
        "units": "",
        "tag": VarTag.META,
    },
    "weight_leptonSF": {
        "name": r"Lepton scale factors",
        "units": "",
        "tag": VarTag.META,
    },
    "totalEventsWeighted": {
        "name": r"Total DSID weight",
        "units": "",
        "tag": VarTag.META,
    },
    "DSID": {
        "name": r"Dataset ID",
        "units": "",
        "tag": VarTag.META,
    },
    "mcChannelNumber": {
        "name": r"Dataset ID",
        "units": "",
        "tag": VarTag.META,
    },
    "eventNumber": {
        "name": r"Event number",
        "units": "",
        "tag": VarTag.META,
    },
    "runNumber": {
        "name": "Run number",
        "units": "",
        "tag": VarTag.META,
    },
    "truth_weight": {
        "name": "Truth event weight",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "reco_weight": {
        "name": "Reconstructed event weight",
        "units": "",
        "tag": VarTag.META,
    },
    # DTA-specific
    "mcWeight": {
        "name": "mc weight",
        "units": "",
        "tag": VarTag.META,
    },
    "prwWeight": {
        "name": "pileup weight",
        "units": "",
        "tag": VarTag.META,
    },
    "rwCorr": {
        "name": "rwCorr",
        "units": "",
        "tag": VarTag.META,
    },
    "passTrigger": {
        "name": "pass trigger",
        "units": "",
        "tag": VarTag.RECO,
    },
    "passMetTrigger": {
        "name": "pass MET trigger",
        "units": "",
        "tag": VarTag.RECO,
    },
    "passTruth": {
        "name": "pass truth",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "passReco": {
        "name": "pass reco",
        "units": "",
        "tag": VarTag.RECO,
    },
    "nVtx": {
        "name": "number of vertices",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthTau_decay_mode": {
        "name": "Truth Tau decay mode",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthTau_isHadronic": {
        "name": "Truth Tau is hadronic",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "FJVT_SF": {
        "name": "Forward Jet Vertex Tagger Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "JVT_SF": {
        "name": "Jet Vertex Tagger Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "TauSF_LooseWP": {
        "name": "Tau Loose WP Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "TauSF_MediumWP": {
        "name": "Tau Medium WP Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "TauSF_TightWP": {
        "name": "Tau Tight WP Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Muon_isoSF": {
        "name": "Muon Isolation Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Muon_recoSF": {
        "name": "Muon Reco Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Muon_ttvaSF": {
        "name": "Muon TTVA Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Ele_isoSF": {
        "name": "Electron Isolation Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Ele_recoSF": {
        "name": "Electron Reco Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    "Ele_ttvaSF": {
        "name": "Electron TTVA Scale Factor",
        "units": "",
        "tag": VarTag.META,
    },
    # DERIVED KINEMATIC VARIABLES
    # =======================================================
    "mt_born": {  # boson pt
        "name": r"Born $M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "mt_bare": {  # boson pt
        "name": r"Bare $M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "mt_dres": {  # boson pt
        "name": r"Dressed $M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "mu_mt_reco": {  # boson pt
        "name": r"$M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "e_mt_reco": {  # boson pt
        "name": r"$M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "w_y": {  # boson rapidity
        "name": r"W rapidity $y$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "z_y": {  # boson rapidity
        "name": r"Z rapidity $y$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "v_y": {  # boson rapidity
        "name": r"Vector boson rapidity $y$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MTW": {
        "name": r"Reco $M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "TruthMTW": {
        "name": r"Truth $M^{W}_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "DilepM": {
        "name": r"$m_{ll}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthDilepM": {
        "name": r"$m_{ll}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    # RECO-LEVEL KINEMATIC VARIABLES
    # =======================================================
    # analysistop
    "el_pt": {
        "name": r"Electron $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "el_eta": {
        "name": r"Electron $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "el_phi": {
        "name": r"Electron $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "el_e": {
        "name": r"Electron $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "el_d0sig": {
        "name": r"Electron $d_0$ significance",
        "units": "",
        "tag": VarTag.RECO,
    },
    "el_delta_z0_sintheta": {
        "name": r"Electron $\Delta z_0\sin\theta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "mu_pt": {
        "name": r"Muon $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "mu_eta": {
        "name": r"Muon $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "mu_phi": {
        "name": r"Muon $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "mu_e": {
        "name": r"Muon $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "mu_d0sig": {
        "name": r"Muon $d_0$ significance",
        "units": "",
        "tag": VarTag.RECO,
    },
    "mu_delta_z0_sintheta": {
        "name": r"Muon $\Delta z_0\sin\theta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "jet_pt": {
        "name": r"Jet $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "jet_eta": {
        "name": r"Jet $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "jet_phi": {
        "name": r"Jet $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "jet_e": {
        "name": r"Jet $E$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "met_met": {
        "name": r"$E_{T}^{\mathrm{miss}}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "met_phi": {
        "name": r"$E_T^{\mathrm{miss}} \phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "mu_isTight": {
        "name": r"Tight Muon",
        "units": "",
        "tag": VarTag.RECO,
    },
    "passTightQuality": {
        "name": "Is Tight",
        "units": "",
        "tag": VarTag.RECO,
    },
    # from DTA
    "TauBaselineWP": {
        "name": r"Tau Baseline WP",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauLooseWP": {
        "name": "Tau Loose WP",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauMediumWP": {
        "name": "Tau Medium WP",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauTightWP": {
        "name": "Tau Tight WP",
        "units": "",
        "tag": VarTag.RECO,
    },
    "ElePt": {
        "name": r"Electron $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "EleEta": {
        "name": r"Electron $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "ElePhi": {
        "name": r"Electron $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "EleE": {
        "name": r"Electron $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "Ele_d0sig": {
        "name": r"Electron $d_0$ significance",
        "units": "",
        "tag": VarTag.RECO,
    },
    "Ele_delta_z0": {
        "name": r"Electron $\Delta z_0$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "Ele_delta_z0_sintheta": {
        "name": r"Electron $\Delta z_0 \sin\theta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "MuonPt": {
        "name": r"Muon $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "MuonEta": {
        "name": r"Muon $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "MuonPhi": {
        "name": r"Muon $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "MuonE": {
        "name": r"Muon $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "Muon_d0sig": {
        "name": r"Muon $d_0$ significance",
        "units": "",
        "tag": VarTag.RECO,
    },
    "Muon_delta_z0": {
        "name": r"Muon $\Delta z_0$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "Muon_delta_z0_sintheta": {
        "name": r"Muon $\Delta z_0 \sin\theta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauCharge": {
        "name": "Tau Charge",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauPt": {
        "name": r"Tau $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "TauEta": {
        "name": r"Tau $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauPhi": {
        "name": r"Tau $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "TauE": {
        "name": r"Tau $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "JetPt": {
        "name": r"Jet $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "JetEta": {
        "name": r"Jet $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "JetPhi": {
        "name": r"Jet $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "JetE": {
        "name": r"Jet $E$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "Jet_btag": {
        "name": r"Jet b-tag",
        "units": "",
        "tag": VarTag.RECO,
    },
    "PhotonPt": {
        "name": r"Photon $p_{T}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "PhotonEta": {
        "name": r"Photon $\eta$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "PhotonPhi": {
        "name": r"Photon $\phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    "PhotonE": {
        "name": r"Photon $E$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "MET_etx": {
        "name": r"$E_{Tx}^{\mathrm{miss}}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "MET_ety": {
        "name": r"$E_{Ty}^{\mathrm{miss}}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "MET_met": {
        "name": r"$E_{T}^{\mathrm{miss}}$",
        "units": "GeV",
        "tag": VarTag.RECO,
    },
    "MET_phi": {
        "name": r"$E_T^{\mathrm{miss}} \phi$",
        "units": "",
        "tag": VarTag.RECO,
    },
    # TRUTH-LEVEL KINEMATIC VARIABLES
    # =======================================================
    # analysistop
    "PDFinfo_X1": {
        "name": r"$x$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "PDFinfo_X2": {
        "name": r"$x$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "PDFinfo_Q": {
        "name": r"$Q$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_phi_born": {
        "name": r"Born dilepton $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_phi_bare": {
        "name": r"bare dilepton $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_phi_dres": {
        "name": r"dressed dilepton $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_eta_born": {
        "name": r"Born dilepton $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_eta_bare": {
        "name": r"bare dilepton $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_eta_dres": {
        "name": r"dressed dilepton $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_pt_born": {
        "name": r"Born dilepton $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_pt_bare": {
        "name": r"bare dilepton $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_pt_dres": {
        "name": r"dressed dilepton $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_m_born": {
        "name": r"Born $m_{ll}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_m_bare": {
        "name": r"bare $m_{ll}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_dilep_m_dres": {
        "name": r"dressed $m_{ll}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_phi_born": {
        "name": r"Born neutrino $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_phi_bare": {
        "name": r"bare neutrino $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_phi_dres": {
        "name": r"dressed neutrino $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_eta_born": {
        "name": r"Born neutrino $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_eta_bare": {
        "name": r"bare neutrino $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_eta_dres": {
        "name": r"dressed neutrino $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_m_born": {
        "name": r"Born neutrino $m$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_m_bare": {
        "name": r"bare neutrino $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_m_dres": {
        "name": r"dressed neutrino $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_pt_born": {
        "name": r"Born neutrino $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_pt_bare": {
        "name": r"bare neutrino $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZneutrino_pt_dres": {
        "name": r"dressed neutrino $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_phi_born": {
        "name": r"Born %s $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_phi_bare": {
        "name": r"bare %s $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_phi_dres": {
        "name": r"dressed %s $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_eta_born": {
        "name": r"Born %s $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_eta_bare": {
        "name": r"bare %s $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_eta_dres": {
        "name": r"dressed %s $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_pt_born": {
        "name": r"Born %s $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_pt_bare": {
        "name": r"bare %s $p_{T}$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_pt_dres": {
        "name": r"dressed %s $p{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_m_born": {
        "name": r"Born %s $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_m_bare": {
        "name": r"bare %s $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZmu_el_m_dres": {
        "name": r"dressed %s $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_pt": {
        "name": r"dressed W $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_phi": {
        "name": r"dressed W $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_eta": {
        "name": r"dressed W $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "MC_WZ_m": {
        "name": r"dressed $m^W$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    # for DTA
    "TruthJetEta": {
        "name": r"Truth jet $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthJetPhi": {
        "name": r"Truth jet $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthJetPt": {
        "name": r"Truth jet $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthJetE": {
        "name": r"Truth jet $E$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthNeutrinoEta": {
        "name": r"Truth neutrino $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthNeutrinoE": {
        "name": r"Truth neutrino $E$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthNeutrinoPhi": {
        "name": r"Truth neutrino $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthNeutrinoPt": {
        "name": r"Truth neutrino $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthMetPt": {
        "name": r"Truth neutrino $p_T$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthMetPhi": {
        "name": r"Truth neutrino $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthMetEta": {
        "name": r"Truth neutrino $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "ImplicitMetPt": {
        "name": r"Implicit MET $p_T$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "ImplicitMetPhi": {
        "name": r"Implicit MET $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "ImplicitMetEta": {
        "name": r"Implicit MET $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthMuonEta": {
        "name": r"Truth muon $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthMuonPhi": {
        "name": r"Truth muon $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthMuonPt": {
        "name": r"Truth muon $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthMuonE": {
        "name": r"Truth muon $E$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthEleEta": {
        "name": r"Truth electron $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthElePhi": {
        "name": r"Truth electron $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthElePt": {
        "name": r"Truth electron $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthEleE": {
        "name": r"Truth electron $E$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthTauCharge": {
        "name": "Truth Tau Charge",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthTauEta": {
        "name": r"Truth tau $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthTauPhi": {
        "name": r"Truth tau $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthTauPt": {
        "name": r"Truth tau $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthTauE": {
        "name": r"Truth tau $E$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthTauM": {
        "name": r"Truth tau $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthBosonM": {
        "name": r"Truth boson $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthBosonE": {
        "name": r"Truth boson $E$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthBosonPt": {
        "name": r"Truth boson $p_T$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "TruthBosonPhi": {
        "name": r"Truth boson $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "TruthBosonEta": {
        "name": r"Truth boson $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "VisTruthTauPhi": {
        "name": r"Visible truth tau products $\phi$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "VisTruthTauPt": {
        "name": r"Visible truth tau products $p_{T}$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    "VisTruthTauEta": {
        "name": r"Visible truth tau products $\eta$",
        "units": "",
        "tag": VarTag.TRUTH,
    },
    "VisTruthTauM": {
        "name": r"Visible truth tau products $m$",
        "units": "GeV",
        "tag": VarTag.TRUTH,
    },
    # DEFAULT & TESTING
    # =======================================================
    "testvartruth": {
        "name": "testvar",
        "units": "s",
        "tag": VarTag.TRUTH,
    },
    "testvarreco": {
        "name": "testvar",
        "units": "s",
        "tag": VarTag.RECO,
    },
}
