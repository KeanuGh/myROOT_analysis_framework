from typing import Dict, TypedDict


class Branch(TypedDict):
    name: str
    units: str
    tag: str


# labels for cross-sections
variable_data: Dict[str, Branch] = {
    # DATASET METADATA
    # =======================================================
    'weight_mc': {
        'name': r'MC weight',
        'units': '',
        'tag': 'meta',
    },
    'weight_KFactor': {
        'name': r'KFactor weight',
        'units': '',
        'tag': 'meta',
    },
    'KFactor_weight_truth': {
        'name': r'KFactor weight',
        'units': '',
        'tag': 'meta',
    },
    'weight_pileup': {
        'name': r'Pileup weight',
        'units': '',
        'tag': 'meta',
    },
    'weight_leptonSF': {
        'name': r'Lepton scale factors',
        'units': '',
        'tag': 'meta'
    },
    'totalEventsWeighted': {
        'name': r'Total DSID weight',
        'units': '',
        'tag': 'meta',
    },
    'DSID': {
        'name': r'Dataset ID',
        'units': '',
        'tag': 'meta'
    },
    'mcChannelNumber': {
        'name': r'Dataset ID',
        'units': '',
        'tag': 'meta'
    },
    'eventNumber': {
        'name': r'Event number',
        'units': '',
        'tag': 'meta'
    },
    'runNumber': {
        'name': 'Run number',
        'units': '',
        'tag': 'meta',
    },
    'truth_weight': {
        'name': 'Truth event weight',
        'units': '',
        'tag': 'truth',
    },
    'reco_weight': {
        'name': 'Reconstructed event weight',
        'units': '',
        'tag': 'reco'
    },

    # DERIVED KINEMATIC VARIABLES
    # =======================================================
    'mt_born': {  # boson pt
        'name': r'Born $M^{W}_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'mt_bare': {  # boson pt
        'name': r'Bare $M^{W}_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'mt_dres': {  # boson pt
        'name': r'Dressed $M^{W}_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'mu_mt_reco': {  # boson pt
        'name': r'$M^{W}_{T}$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'e_mt_reco': {  # boson pt
        'name': r'$M^{W}_{T}$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'w_y': {  # boson rapidity
        'name': r'W rapidity $y$',
        'units': '',
        'tag': 'truth',
    },
    'z_y': {  # boson rapidity
        'name': r'Z rapidity $y$',
        'units': '',
        'tag': 'truth',
    },
    'v_y': {  # boson rapidity
        'name': r'Vector boson rapidity $y$',
        'units': '',
        'tag': 'truth',
    },

    # RECO-LEVEL KINEMATIC VARIABLES
    # =======================================================
    'el_pt': {
        'name': r'Electron $p_{T}$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'el_eta': {
        'name': r'Electron $\eta$',
        'units': '',
        'tag': 'reco',
    },
    'el_phi': {
        'name': r'Electron $\phi$',
        'units': '',
        'tag': 'reco',
    },
    'el_e': {
        'name': r'Electron $E$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'e_d0sig': {
        'xlabel': r'Electron $d_0$ significance',
        'units': '',
        'tag': 'reco',
    },
    'e_delta_z0_sintheta': {
        'name': r'Electron $\Delta z_0\sin\theta$',
        'units': '',
        'tag': 'reco',
    },
    'mu_pt': {
        'name': r'Muon $p_{T}$',
        'units': '',
        'tag': 'reco',
    },
    'mu_eta': {
        'name': r'Muon $\eta$',
        'units': '',
        'tag': 'reco',
    },
    'mu_phi': {
        'name': r'Muon $\phi$',
        'units': '',
        'tag': 'reco',
    },
    'mu_e': {
        'name': r'Muon $E$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'mu_d0sig': {
        'name': r'Muon $d_0$ significance',
        'units': '',
        'tag': 'reco'
    },
    'mu_delta_z0_sintheta': {
        'name': r'Muon $\Delta z_0\sin\theta$',
        'units': '',
        'tag': 'reco'
    },
    'jet_pt': {
        'name': r'Jet $p_{T}$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'jet_eta': {
        'name': r'Jet $\eta$',
        'units': '',
        'tag': 'reco',
    },
    'jet_phi': {
        'name': r'Jet $\phi$',
        'units': '',
        'tag': 'reco'
    },
    'jet_e': {
        'name': r'Jet $E$',
        'units': '',
        'tag': 'reco'
    },
    'met_met': {
        'name': r'$E_{T}^{\mathrm{miss}}$',
        'units': 'GeV',
        'tag': 'reco',
    },
    'met_phi': {
        'name': r'$\phi^{\mathrm{miss}$',
        'units': '',
        'tag': 'reco'
    },

    # TRUTH-LEVEL KINEMATIC VARIABLES
    # =======================================================
    'PDFinfo_X1': {
        'name': r'$x$',
        'units': '',
        'tag': 'truth'
    },
    'PDFinfo_X2': {
        'name': r'$x$',
        'units': '',
        'tag': 'truth',
    },
    'PDFinfo_Q': {
        'name': r'$Q$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZ_dilep_phi_born': {
        'name': r'Born dilepton $\phi$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_dilep_phi_bare': {
        'name': r'bare dilepton $\phi$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_dilep_phi_dres': {
        'name': r'dressed dilepton $\phi$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZ_dilep_eta_born': {
        'name': r'Born dilepton $\eta$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZ_dilep_eta_bare': {
        'name': r'bare dilepton $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_dilep_eta_dres': {
        'name': r'dressed dilepton $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_dilep_pt_born': {
        'name': r'Born dilepton $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZ_dilep_pt_bare': {
        'name': r'bare dilepton $p_{T}$',
        'units' 'GeV'
        'tag': 'truth',
    },
    'MC_WZ_dilep_pt_dres': {
        'name': r'dressed dilepton $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZ_dilep_m_born': {
        'name': r'Born $m_{ll}$',
        'units': 'GeV',
        'tag': 'truth'
    },
    'MC_WZ_dilep_m_bare': {
        'name': r'bare $m_{ll}$',
        'units': 'GeV',
        'tag': 'truth'
    },
    'MC_WZ_dilep_m_dres': {
        'name': r'dressed $m_{ll}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZneutrino_phi_born': {
        'name': r'Born neutrino $\phi$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZneutrino_phi_bare': {
        'name': r'bare neutrino $\phi$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZneutrino_phi_dres': {
        'name': r'dressed neutrino $\phi$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZneutrino_eta_born': {
        'name': r'Born neutrino $\eta$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZneutrino_eta_bare': {
        'name': r'bare neutrino $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZneutrino_eta_dres': {
        'name': r'dressed neutrino $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZneutrino_m_born': {
        'name': r'Born neutrino $m$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZneutrino_m_bare': {
        'name': r'bare neutrino $m$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZneutrino_m_dres': {
        'name': r'dressed neutrino $m$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZneutrino_pt_born': {
        'name': r'Born neutrino $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZneutrino_pt_bare': {
        'name': r'bare neutrino $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZneutrino_pt_dres': {
        'name': r'dressed neutrino $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZmu_el_phi_born': {
        'name': r'Born %s $\phi$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_phi_bare': {
        'name': r'bare %s $\phi$',
        'units': '',
        'tag': 'truth'
    },
    'MC_WZmu_el_phi_dres': {
        'name': r'dressed %s $\phi$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_eta_born': {
        'name': r'Born %s $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_eta_bare': {
        'name': r'bare %s $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_eta_dres': {
        'name': r'dressed %s $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_pt_born': {
        'name': r'Born %s $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZmu_el_pt_bare': {
        'name': r'bare %s $p_{T}$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZmu_el_pt_dres': {
        'name': r'dressed %s $p{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZmu_el_m_born': {
        'name': r'Born %s $m$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZmu_el_m_bare': {
        'name': r'bare %s $m$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZmu_el_m_dres': {
        'name': r'dressed %s $m$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZ_pt': {
        'name': r'dressed W $p_{T}$',
        'units': 'GeV',
        'tag': 'truth',
    },
    'MC_WZ_phi': {
        'name': r'dressed W $\phi$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_eta': {
        'name': r'dressed W $\eta$',
        'units': '',
        'tag': 'truth',
    },
    'MC_WZ_m': {
        'name': r'dressed $m^W$',
        'units': 'GeV',
        'tag': 'truth',
    },

    # DEFAULT & TESTING
    # =======================================================
    'testvartruth': {
        'name': 'testvar',
        'units': 's',
        'tag': 'truth',
    },
    'testvarreco': {
        'name': 'testvar',
        'units': 's',
        'tag': 'reco',
    },
}
