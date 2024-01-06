from src.cutfile import Cut

cuts_truth_full = [
    Cut(
        r"$p_T^e > 7$",
        r"(isnan(TruthElePt) || TruthElePt > 7)",
    ),
    Cut(
        r"$|\eta_\mathrm{truth}^e| < 1.37 || 1.52 < |\eta_\mathrm{truth}^e| < 2.47$",
        r"(isnan(TruthEleEta) || (abs(TruthEleEta) < 1.37 || 1.52 < abs(TruthEleEta) < 2.47))",
    ),
    Cut(
        r"$p_T^e > 7$",
        r"(isnan(TruthMuonPt) || TruthMuonPt > 7)",
    ),
    Cut(
        r"$|\eta_\mathrm{truth}^\mu| < 2.5$",
        r"(isnan(TruthMuonEta) || abs(TruthMuonEta) < 2.5)",
    ),
    Cut(
        r"$p_T^\tau > 20$",
        r"(isnan(TruthTauPt) || TruthTauPt > 20)",
    ),
    Cut(
        r"$|\eta^\tau| < 1.37 || 1.52 < |\eta^\tau| < 2.47$",
        r"(isnan(TruthTauEta) || (abs(TruthTauEta) < 1.37 || 1.52 < abs(TruthTauEta) < 2.47))",
    ),
    Cut(
        r"\mathrm{pass trigger}",
        r"passTrigger",
    ),
    Cut(
        r"\mathrm{medium tau}",
        r"TauMediumWP",
    ),
    Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    ),
    Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    ),
]
cuts_reco_full = [
    Cut(
        r"\mathrm{pass trigger}",
        r"passTrigger",
    ),
        Cut(
        r"\mathrm{}",
        r"TauMediumWP",
    ),
    Cut(
        r"\mathrm{medium tau}",
        r"TauMediumWP",
    ),
    Cut(
        r"$p_T^e > 20$",
        r"(isnan(ElePt) || ElePt > 20)",
    ),
    Cut(
        r"$|\eta_\mathrm{reco}^e| < 1.37 || 1.52 < |\eta_\mathrm{reco}^e| < 2.47$",
        r"(isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))",
    ),
    Cut(
        r"$p_T^\mu > 20$",
        r"(isnan(MuonPt) || MuonPt > 20)",
    ),
    Cut(
        r"$|\eta_\mathrm{reco}^\mu| < 2.5$",
        r"(isnan(MuonEta) || (abs(MuonEta) < 2.5))",
    ),
    Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    ),
    Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    ),
]

# HAD
# ===========================
cuts_truth_had = [
    Cut(
        r"\tau\rightarrow \textrm{had}",
        r"TruthTau_decay_mode == 0",
    ),
    Cut(
        r"$p_T^\tau > 20$",
        r"(isnan(TruthTauPt) || TruthTauPt > 20)",
    ),
    Cut(
        r"$|\eta^\tau| < 1.37 || 1.52 < |\eta^\tau| < 2.47$",
        r"(isnan(TruthTauEta) || (abs(TruthTauEta) < 1.37 || 1.52 < abs(TruthTauEta) < 2.47))",
    ),
]
cuts_reco_had = [
    Cut(
        r"\mathrm{pass trigger}",
        r"passTrigger",
    ),
    Cut(
        r"\mathrm{medium tau}",
        r"TauMediumWP",
    ),
    Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    ),
    Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    ),
]

# E
# ===========================
cuts_truth_e = [
    Cut(
        r"\tau\rightarrow e",
        r"TruthTau_decay_mode == 2",
    ),
    Cut(
        r"$p_T^e > 7$",
        r"(isnan(TruthElePt) || TruthElePt > 7)",
    ),
    Cut(
        r"$|\eta_\mathrm{truth}^e| < 1.37 || 1.52 < |\eta_\mathrm{truth}^e| < 2.47$",
        r"(isnan(TruthEleEta) || (abs(TruthEleEta) < 1.37 || 1.52 < abs(TruthEleEta) < 2.47))",
    ),
    Cut(
        r"$p_T^\tau > 20$",
        r"(isnan(TruthTauPt) || TruthTauPt > 20)",
    ),
    Cut(
        r"$|\eta^\tau| < 1.37 || 1.52 < |\eta^\tau| < 2.47$",
        r"(isnan(TruthTauEta) || (abs(TruthTauEta) < 1.37 || 1.52 < abs(TruthTauEta) < 2.47))",
    ),
]
cuts_reco_e = [
    Cut(
        r"\mathrm{pass trigger}",
        r"passTrigger",
    ),
    Cut(
        r"\mathrm{medium tau}",
        r"TauMediumWP",
    ),
    Cut(
        r"$p_T^e > 20$",
        r"(isnan(ElePt) || ElePt > 20)",
    ),
    Cut(
        r"$|\eta_\mathrm{reco}^e| < 1.37 || 1.52 < |\eta_\mathrm{reco}^e| < 2.47$",
        r"(isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))",
    ),
    Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    ),
    Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    ),
    Cut(
        r"$\Delta R_{\tau---e}  0.5$",
        r"DeltaR_tau_e < 0.5",
    )
]

# MU
# ===========================
cuts_truth_mu = [
    Cut(
        r"\tau\rightarrow\mu",
        r"TruthTau_decay_mode == 1",
    ),
    Cut(
        r"$p_T^e > 7$",
        r"(isnan(TruthMuonPt) || TruthMuonPt > 7)",
    ),
    Cut(
        r"$|\eta_\mathrm{truth}^\mu| < 2.5$",
        r"(isnan(TruthMuonEta) || abs(TruthMuonEta) < 2.5)",
    ),
    Cut(
        r"$p_T^\tau > 20$",
        r"(isnan(TruthTauPt) || TruthTauPt > 20)",
    ),
    Cut(
        r"$|\eta^\tau| < 1.37 || 1.52 < |\eta^\tau| < 2.47$",
        r"(isnan(TruthTauEta) || (abs(TruthTauEta) < 1.37 || 1.52 < abs(TruthTauEta) < 2.47))",
    ),
]
cuts_reco_mu = [
    Cut(
        r"\mathrm{pass trigger}",
        r"passTrigger",
    ),
    Cut(
        r"\mathrm{medium tau}",
        r"TauMediumWP",
    ),
    Cut(
        r"$p_T^\mu > 20$",
        r"(isnan(MuonPt) || MuonPt > 20)",
    ),
    Cut(
        r"$|\eta_\mathrm{reco}^\mu| < 2.5$",
        r"(isnan(MuonEta) || (abs(MuonEta) < 2.5))",
    ),
    Cut(
        r"$p_T^\tau > 170$",
        r"TauPt > 170",
    ),
    Cut(
        r"$m_T^W > 150$",
        r"MTW > 150",
    ),
    Cut(
        r"$\Delta R_{\tau---\mu} < 0.5$",
        r"DeltaR_tau_mu < 0.5",
    )
]

# Variables
import_vars_truth = {
    "TruthTauPt",
    "TruthTauEta",
    "TruthTauPhi",
    "VisTruthTauPt",
    "VisTruthTauEta",
    "VisTruthTauPhi",
    "TruthNeutrinoPt",
    "TruthNeutrinoEta",
    "TruthNeutrinoPhi",
    "TruthJetEta",
    "TruthJetPhi",
    "TruthElePt",
    "TruthEleEta",
    "TruthElePhi",
    "TruthMuonPt",
    "TruthMuonEta",
    "TruthMuonPhi",
    "TruthBosonM",
}

import_vars_reco = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "JetEta",
    "JetPhi",
    "JetPt",
    "MuonEta",
    "MuonPhi",
    "MuonPt",
    "Muon_d0sig",
    "Muon_delta_z0",
    "EleEta",
    "ElePhi",
    "ElePt",
    "EleE",
    "Ele_d0sig",
    "Muon_delta_z0",
    "PhotonEta",
    "PhotonPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "DeltaR_tau_e",
    "DeltaR_tau_mu",
}
