import pandas as pd
import uproot

from src.histogram import Histogram1D

filepath = '/home/keanu/Uni_Stuff_Queen_Mary/DTA/histograms_singletau_H.root'
treename = 'T_s1thv_NOMINAL'
wanted_cols = [
    'weight',
    'mcChannel',
    'runNumber',
    'eventNumber',
    'nVtx',
    'passTruth',
    'passReco',
    'TauEta', 'TauPhi', 'TauPt', 'TauE',
    'JetEta', 'JetPhi', 'JetE', 'Jet_btag',
    'MuonEta', 'MuonPhi', 'MuonPt', 'MuonE', 'Muon_d0sig', 'Muon_delta_z0',
    'EleEta', 'ElePhi', 'ElePt', 'EleE', 'Ele_d0sig', 'Ele_delta_z0',
    'PhotonEta', 'PhotonPhi', 'PhotonPt', 'PhotonE',
    'TruthJetE', 'TruthJetPhi', 'TruthJetPt', 'TruthJetE',
    'TruthNeutrinoEta', 'TruthNeutrinoPhi', 'TruthNeutrinoPt', 'TruthNeutrinoE',
    'TruthMuonEta', 'TruthMuonPhi', 'TruthMuonPt', 'TruthMuonE',
    'TruthEleEta', 'TruthElePhi', 'TruthElePt', 'TruthEleE',
    'TruthTauEta', 'TruthTauPhi', 'TruthTauPt', 'TruthTauM',
    'VisTruthTauEta', 'VisTruthTauPhi', 'VisTruthTauPt', 'VisTruthTauM',
    'TruthTau_isHadronic',
    'MET_etx', 'MET_ety', 'MET_met', 'MET_phi',
]
bins = (30, 0, 1000)

df = pd.DataFrame(uproot.concatenate(f'{filepath}:{treename}', expressions=wanted_cols, library='np'))
df.set_index(['mcChannel', 'eventNumber'], inplace=True)

df_pass = df[df['passReco'] & df['passTruth']]

hTauPt = Histogram1D(df_pass['TauPt'], bins, df_pass['weight'])
hTauPt.plot()
