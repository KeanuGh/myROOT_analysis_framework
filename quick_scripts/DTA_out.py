import matplotlib.pyplot as plt
import pandas as pd
import uproot

from src.datasetbuilder import lumi_year
from src.histogram import Histogram1D
from utils import plotting_utils

filepath = '/home/keanu/Uni_Stuff_Queen_Mary/DTA/histograms_singletau_H.root'
treename = 'T_s1thv_NOMINAL'
wanted_cols = [
    'weight',
    'mcChannel',
    'mcWeight',
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
bins = (30, 20, 1000)

df = pd.DataFrame(uproot.concatenate(f'{filepath}:{treename}', expressions=wanted_cols, library='np'))
df.set_index(['mcChannel', 'eventNumber'], inplace=True)
df.index.names = ['DSID', 'eventNumber']

df_pass = df[df['passReco'] & df['passTruth']]
weight = df_pass['weight'] * lumi_year['2015+2016'] / df['mcWeight'].sum()

TauPt = df_pass['TauPt'].map(lambda x: x[0]) / 1000
hTauPt = Histogram1D(TauPt, bins, weight, logbins=True)
ax = hTauPt.plot()
plotting_utils.set_axis_options(ax, 'TauPt', bins, lepton='Tau', logx=True, logy=True)
plt.show()
