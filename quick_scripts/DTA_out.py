import ROOT
import matplotlib.pyplot as plt
import pandas as pd

from src.datasetbuilder import lumi_year
from src.histogram import Histogram1D
from utils import plotting_utils
from utils.variable_names import variable_data

filepath = '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto.MC16a.v1.2022-04-01_histograms.root/*.root'
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

ROOT.gInterpreter.Declare("""
float getVecVal(ROOT::VecOps::RVec<float> x, int i = 0);

float getVecVal(ROOT::VecOps::RVec<float> x, int i) {
    if (x.size() > i)  return x[i];
    else               return NAN;
}
""")

Rdf = ROOT.RDataFrame(treename, filepath)
Rdf = Rdf.Filter("(passTruth == true) & (passReco == true)")

badcols = set()  # save old column names to avoid extracting them later
for col_name in list(Rdf.GetColumnNames()):
    col_type = Rdf.GetColumnType(col_name)

    # unravel vector-type columns
    if "ROOT::VecOps::RVec" in col_type:
        # skip non-numeric vector types
        if col_type == "ROOT::VecOps::RVec<string>":
            badcols.add(col_name)

        elif 'jet' in str(col_name).lower():
            # create three new columns for each possible jet
            for i in range(3):
                Rdf = Rdf.Define(f"{col_name}{i+1}", f"getVecVal({col_name},{i})")
            badcols.add(col_name)

        else:
            Rdf = Rdf.Redefine(col_name, f"getVecVal({col_name},0)")


cols_to_extract = [c for c in list(Rdf.GetColumnNames())
                   if c not in badcols]
for c in cols_to_extract:
    if "ROOT::VecOps::RVec" in Rdf.GetColumnType(c):
        print(f"Column {c} is of type {Rdf.GetColumnType(c)}")

# import needed columns to pandas dataframe
df = pd.DataFrame(Rdf.AsNumpy(columns=[c for c in list(Rdf.GetColumnNames()) if c not in badcols]))
df.set_index(['mcChannel', 'eventNumber'], inplace=True)
df.index.names = ['DSID', 'eventNumber']

# rescale GeV columns
GeV_columns = [
    column for column in df.columns
    if (column in variable_data) and (variable_data[column]['units'] == 'GeV')
]
df[GeV_columns] /= 1000

# calc weights
weight = df['weight'] * lumi_year['2015+2016'] / df['mcWeight'].sum()

# plot
hTauPt = Histogram1D(df['TauPt'], bins, weight, logbins=True)
ax = hTauPt.plot()
plotting_utils.set_axis_options(ax, 'TauPt', bins, lepton='Tau', logx=True, logy=True)
plt.show()
