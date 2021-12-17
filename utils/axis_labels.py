from typing import Dict, TypedDict


class BranchLabel(TypedDict):
    xlabel: str
    ylabel: str


# labels for cross-sections
labels_xs: Dict[str, BranchLabel] = {
    # DATASET VARIABLES
    # =======================================================
    'weight_mc': {
        'xlabel': r'MC weight',
        'ylabel': r'Entries',
    },
    'weight_KFactor': {
        'xlabel': r'KFactor weight',
        'ylabel': r'Entries'
    },
    'KFactor_weight_truth': {
        'xlabel': r'KFactor weight',
        'ylabel': r'Entries'
    },
    'weight_pileup': {
        'xlabel': r'Pileup weight',
        'ylabel': r'Entries'
    },
    'weight_leptonSF': {
        'xlabel': r'Lepton scale factors',
        'ylabel': r'Entries'
    },
    'totalEventsWeighted': {
        'xlabel': r'Total DSID weight',
        'ylabel': r'Entries'
    },
    'DSID': {
        'xlabel': r'Dataset ID',
        'ylabel': r'Entries',
    },
    'eventNumber': {
        'xlabel': r'Event number',
        'ylabel': r'Entries'
    },
    'runNumber': {
        'xlabel': 'Run number',
        'ylabel': 'Entries'
    },

    # DERIVED KINEMATIC VARIABLES
    # =======================================================
    'mt_born': {  # boson pt
        'xlabel': r'Born $M^{W}_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dM_{T}}$ [fb GeV$^{-1}$]',
    },
    'mt_bare': {  # boson pt
        'xlabel': r'Bare $M^{W}_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dM_{T}}$ [fb GeV$^{-1}$]',
    },
    'mt_dres': {  # boson pt
        'xlabel': r'Dressed $M^{W}_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dM_{T}}$ [fb GeV$^{-1}$]',
    },
    'mt_reco': {  # boson pt
        'xlabel': r'$M^{W}_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dM_{T}}$ [fb GeV$^{-1}$]',
    },
    'w_y': {  # boson rapidity
        'xlabel': r'W rapidity $y$',
        'ylabel': r'$\frac{d\sigma}{dy}$ [fb]',
    },
    'z_y': {  # boson rapidity
        'xlabel': r'Z rapidity $y$',
        'ylabel': r'$\frac{d\sigma}{dy}$ [fb]',
    },
    'v_y': {  # boson rapidity
        'xlabel': r'Vector boson rapidity $y$',
        'ylabel': r'$\frac{d\sigma}{dy}$ [fb]',
    },

    # RECO-LEVEL KINEMATIC VARIABLES
    # =======================================================
    'el_pt': {
      'xlabel': r'Electron $p_{T}$ [GeV]',
      'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]'
    },
    'el_eta': {
      'xlabel': r'Electron $\eta$',
      'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]'
    },
    'el_phi': {
        'xlabel': r'Electron $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]'
    },
    'el_e': {
        'xlabel': r'Electron $E$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dE}$ [fb GeV$^{-1}$]'
    }, 
    'e_d0sig': {
        'xlabel': r'Electron $d_0$ significance',
        'ylabel': r'Entries' 
    },
    'e_delta_z0_sintheta': {
        'xlabel': r'Electron $\Delta z_0\sin\theta$',
        'ylabel': r'Entries'
    },
    'mu_pt': {
        'xlabel': r'Muon $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb]'
    },
    'mu_eta': {
        'xlabel': r'Muon $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]'
    },
    'mu_phi': {
        'xlabel': r'Muon $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]'
    },
    'mu_e': {
        'xlabel': r'Muon $E$',
        'ylabel': r'$\frac{d\sigma}{dE}$ [fb GeV$^{-1}$]'
    },
    'mu_d0sig': {
        'xlabel': r'Muon $d_0$ significance',
        'ylabel': r'Entries' 
    },
    'mu_delta_z0_sintheta': {
        'xlabel': r'Muon $\Delta z_0\sin\theta$',
        'ylabel': r'Entries'
    },
    'jet_pt': {
        'xlabel': r'Jet $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]'
    },
    'jet_eta': {
        'xlabel': r'Jet $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]'
    },
    'jet_phi': {
        'xlabel': r'Jet $\phi$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]'
    },
    'jet_e': {
        'xlabel': r'Jet $E$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dE}$ [fb GeV$^{-1}$]'
    },
    'met_met': {
        'xlabel': r'$E_{T}^{\mathrm{miss}}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dE_{T}^{\mathrm{miss}}}$ [fb GeV$^{-1}$]'
    },
    'met_phi': {
        'xlabel': r'$\phi^{\mathrm{miss}$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]'
    },

    # TRUTH-LEVEL KINEMATIC VARIABLES
    # =======================================================
    'PDFinfo_X1': {
        'xlabel': r'x',
        'ylabel': r'$\frac{d\sigma}{dx}$ [fb]',
    },
    'PDFinfo_X2': {
        'xlabel': r'x',
        'ylabel': r'$\frac{d\sigma}{dx}$ [fb]',
    },
    'PDFinfo_Q': {
        'xlabel': r'Q [GeV]',
        'ylabel': r'$\frac{d\sigma}{dQ}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_phi_born': {
        'xlabel': r'Born dilepton $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZ_dilep_phi_bare': {
        'xlabel': r'bare dilepton $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZ_dilep_phi_dres': {
        'xlabel': r'dressed dilepton $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZ_dilep_eta_born': {
        'xlabel': r'Born dilepton $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZ_dilep_eta_bare': {
        'xlabel': r'bare dilepton $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZ_dilep_eta_dres': {
        'xlabel': r'dressed dilepton $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZ_dilep_pt_born': {
        'xlabel': r'Born dilepton $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_pt_bare': {
        'xlabel': r'bare dilepton $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_pt_dres': {
        'xlabel': r'dressed dilepton $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_m_born': {
        'xlabel': r'Born dilepton mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_m_bare': {
        'xlabel': r'bare dilepton mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_dilep_m_dres': {
        'xlabel': r'dressed dilepton mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_phi_born': {
        'xlabel': r'Born neutrino $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_phi_bare': {
        'xlabel': r'bare neutrino $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_phi_dres': {
        'xlabel': r'dressed neutrino $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_eta_born': {
        'xlabel': r'Born neutrino $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_eta_bare': {
        'xlabel': r'bare neutrino $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_eta_dres': {
        'xlabel': r'dressed neutrino $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZneutrino_m_born': {
        'xlabel': r'Born neutrino mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_m_bare': {
        'xlabel': r'bare neutrino mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_m_dres': {
        'xlabel': r'dressed neutrino mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_pt_born': {
        'xlabel': r'Born neutrino $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_pt_bare': {
        'xlabel': r'bare neutrino $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZneutrino_pt_dres': {
        'xlabel': r'dressed neutrino $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_phi_born': {
        'xlabel': r'Born %s $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZmu_el_phi_bare': {
        'xlabel': r'bare %s $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZmu_el_phi_dres': {
        'xlabel': r'dressed %s $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZmu_el_eta_born': {
        'xlabel': r'Born %s $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZmu_el_eta_bare': {
        'xlabel': r'bare %s $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZmu_el_eta_dres': {
        'xlabel': r'dressed %s $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZmu_el_pt_born': {
        'xlabel': r'Born %s $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_pt_bare': {
        'xlabel': r'bare %s $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_pt_dres': {
        'xlabel': r'dressed %s $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_m_born': {
        'xlabel': r'Born %s mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_m_bare': {
        'xlabel': r'bare %s mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZmu_el_m_dres': {
        'xlabel': r'dressed %s mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_pt': {
        'xlabel': r'dressed W $p_{T}$ [GeV]',
        'ylabel': r'$\frac{d\sigma}{dp_{T}}$ [fb GeV$^{-1}$]',
    },
    'MC_WZ_phi': {
        'xlabel': r'dressed W $\phi$',
        'ylabel': r'$\frac{d\sigma}{d\phi}$ [fb]',
    },
    'MC_WZ_eta': {
        'xlabel': r'dressed W $\eta$',
        'ylabel': r'$\frac{d\sigma}{d\eta}$ [fb]',
    },
    'MC_WZ_m': {
        'xlabel': r'dressed W mass [GeV]',
        'ylabel': r'$\frac{d\sigma}{dm}$ [fb GeV$^{-1}$]',
    },

    # DEFAULT & TESTING
    # =======================================================
    'testvar': {
        'xlabel': 'testxlabel',
        'ylabel': 'testylabel',
    },
}
