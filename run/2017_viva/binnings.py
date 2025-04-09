import numpy as np

mtw_bins = np.array([350, 375, 400, 430, 465, 500, 550, 600, 700, 850, 1000, 2000], dtype="double")
taupt_bins = np.array([170, 200, 250, 300, 350, 425, 500, 600, 1000], dtype="double")
nedges = 16
BINNINGS = {
    "MTW": mtw_bins,
    "TruthMTW": mtw_bins,
    "TauPt": taupt_bins,
    "VisTruthTauPt": taupt_bins,
    "TauEta": np.linspace(-2.5, 2.5, nedges),
    "TauPhi": np.linspace(-np.pi, np.pi, nedges),
    "VisTruthTauEta": np.linspace(-2.5, 2.5, nedges),
    "VisTruthTauPhi": np.linspace(-np.pi, np.pi, nedges),
    "TruthNeutrinoPhi": np.linspace(-np.pi, np.pi, nedges),
    "AbsDeltaPhi_tau_met": np.linspace(0, np.pi, nedges),
    "TruthAbsDeltaPhi_tau_met": np.linspace(0, np.pi, nedges),
    "MET_phi": np.linspace(-np.pi, np.pi, nedges),
    "MET_met": taupt_bins,
    "TruthNeutrinoPt": taupt_bins,
    "TruthTauPt": taupt_bins,
    "TruthTau_nChargedTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "TruthTau_nNeutralTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    "TruthTauEta": np.linspace(-2.5, 2.5, nedges),
    "TruthTauPhi": np.linspace(-2.5, 2.5, nedges),
    "TruthNeutrinoEta": np.linspace(-2.5, 2.5, nedges),
    "MET_eta": np.linspace(-2.5, 2.5, nedges),
    "DeltaPhi_tau_met": np.linspace(0, 2 * np.pi, nedges),
    "TruthDeltaPhi_tau_met": np.linspace(0, 3.5, nedges),
    "TauPt_div_MET": np.linspace(0, 3, nedges),
    "TruthTauPt_div_MET": np.linspace(0, 3, nedges),
    "TauRNNJetScore": np.linspace(0, 1, 36),
    "TauBDTEleScore": np.linspace(0, 1, 36),
    "TauNCoreTracks": np.linspace(0, 4, 5),
    "TauPt_res_frac": np.linspace(-1, 1, nedges),
    "TauPt_res": np.linspace(-300, 300, nedges),
}
TRUTHS = {
    "MTW": "TruthMTW",
    "TauPt": "VisTruthTauPt",
    "TauEta": "VisTruthTauEta",
    "TauPhi": "VisTruthTauPhi",
    "MET_met": "TruthNeutrinoPt",
    "MET_phi": "TruthNeutrinoPhi",
    "AbsDeltaPhi_tau_met": "TruthAbsDeltaPhi_tau_met",
    "TauPt_div_MET": "TruthTauPt_div_MET",
}
