import glob

import ROOT
import pandas as pd

from utils import ROOT_utils

filepath = '/mnt/D/data/DTA_outputs/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_CVetoBVeto.MC16a.v1.2022-04-01_histograms.root/*.root'
treename = 'T_s1tlv_NOMINAL'
wanted_cols = [
    'weight',
    'mcWeight',
    'mcChannel',
    'rwCorr',
    'prwWeight',
    'mcWeight',
    'runNumber',
    'eventNumber',
    # 'nVtx',
    # 'passTruth',
    # 'passReco',
    # 'TauEta', 'TauPhi', 'TauPt', 'TauE',
    # 'JetEta', 'JetPhi', 'JetE', 'Jet_btag',
    'Muon_recoSF', 'Muon_isoSF', 'Muon_ttvaSF',
    'MuonEta', 'MuonPhi', 'MuonPt', 'MuonE', 'Muon_d0sig', 'Muon_delta_z0',
    # 'EleEta', 'ElePhi', 'ElePt', 'EleE', 'Ele_d0sig', 'Ele_delta_z0',
    # 'PhotonEta', 'PhotonPhi', 'PhotonPt', 'PhotonE',
    # 'TruthJetE', 'TruthJetPhi', 'TruthJetPt',
    'TruthNeutrinoEta', 'TruthNeutrinoPhi', 'TruthNeutrinoPt', 'TruthNeutrinoE',
    # 'TruthMuonEta', 'TruthMuonPhi', 'TruthMuonPt', 'TruthMuonE',
    # 'TruthEleEta', 'TruthElePhi', 'TruthElePt', 'TruthEleE',
    'TruthTauEta', 'TruthTauPhi', 'TruthTauPt', 'TruthTauM',
    # 'VisTruthTauEta', 'VisTruthTauPhi', 'VisTruthTauPt', 'VisTruthTauM',
    # 'TruthTau_isHadronic', 'TruthTau_decay_mode',
    # 'MET_etx', 'MET_ety', 'MET_met', 'MET_phi',
    'TruthTau_decay_mode'
]

chain = ROOT_utils.glob_chain(treename, filepath)
Rdf = ROOT.RDataFrame(chain)
# Rdf = Rdf.Filter("(passTruth == true) & (passReco == true)")

# get sumweights
files_list = glob.glob(filepath)
sum_of_weights = 0
for file in files_list:
    with ROOT_utils.ROOT_file(file, 'read') as f:
        sum_of_weights += f.Get("sumOfWeights").GetBinContent(4)
print("summed sumOfWeights: ", sum_of_weights)

with ROOT_utils.ROOT_file('/mnt/D/data/DTA_outputs/CVetoBVeto_H.root', 'read') as f:
    h = f.Get("sumOfWeights")
    sum_of_weights = h.GetBinContent(4)
print("hadd sumOfWeights: ", sum_of_weights)

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


cols_to_extract = [c for c in wanted_cols
                   if c not in badcols]
for c in cols_to_extract:
    if "ROOT::VecOps::RVec" in Rdf.GetColumnType(c):
        print(f"Column {c} is of type {Rdf.GetColumnType(c)}")

# import needed columns to pandas dataframe
df = pd.DataFrame(Rdf.AsNumpy(columns=cols_to_extract))

df.set_index(['mcChannel', 'eventNumber'], inplace=True)
df.index.names = ['DSID', 'eventNumber']

df.dropna(subset='weight', inplace=True)

print("sum of mcWeight col: ", df['mcWeight'].sum())

# # rescale GeV columns
# GeV_columns = [
#     column for column in df.columns
#     if (column in variable_data) and (variable_data[column]['units'] == 'GeV')
# ]
# df[GeV_columns] /= 1000
#
# # calc weights
# df['reco_weight'] = df['weight'] * lumi_year['2015+2016'] / sum_of_weights
# df['truth_weight'] = df['mcWeight'] * lumi_year['2015+2016'] * df['rwCorr'] * df['prwWeight'] / sum_of_weights
# df['muon_reco_weight'] = df['reco_weight'] * df['Muon_recoSF'] * df['Muon_isoSF'] * df['Muon_ttvaSF']
#
# df.dropna(subset='truth_weight', inplace=True)
# df = df.loc[~np.isinf(df['muon_reco_weight'])]
#
# df = df.loc[df['TruthTauPt'] > 25]
# df = df.loc[df['TruthTauEta'].abs() < 2.47]
# df = df.loc[(df['TruthTauEta'].abs() < 1.37) | (df['TruthTauEta'].abs() > 1.52)]
#
# BR = df.loc[df['TruthTau_decay_mode'] == 1].sum() / df.loc[df['TruthTau_decay_mode'] == 2].sum()
# print("tau->munu / tau->enu: ", BR)
#
# # plot
# bins = (30, 1, 50000)
# hTauPt = Histogram1D(
#     df['MuonPt'],
#     bins,
#     weight=df['muon_reco_weight'],
#     logbins=True
# )
# ax = hTauPt.plot(normalise=False)
# plotting_utils.set_axis_options(ax, 'MuonPt', bins, lepton='Tau', logx=True, logy=True, title='36.2 fb$^{-1}$')
# plt.show()
