import os
import time

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot
from awkward import to_pandas

hep.style.use('ATLAS')

N_THREADS = os.cpu_count()
OUT_DIR = '../outputs/quick_script_outputs/'
DATAFILE = '../data/mc16a_wmintaunu_SLICES/*.root'
LUMI_DATA = 32988.1 + 3219.56
BRANCHES_NOMINAL = [
    'mcChannelNumber',
    'weight_mc',
    'weight_leptonSF',
    'weight_KFactor',
    'weight_pileup',
    'eventNumber',
    'mu_eta',
    'mu_pt',
    'mu_phi',
    'met_met',
]
BRANCHES_TRUTH = [
    'MC_WZmu_el_pt_born',
    'MC_WZneutrino_pt_born',
    'MC_WZmu_el_phi_born',
    'MC_WZneutrino_phi_born',
    'MC_WZ_dilep_m_born',
    'mcChannelNumber',
    'weight_mc',
    'KFactor_weight_truth',
    'weight_pileup',
    'eventNumber',
]

# pull root data
t = time.time()
nominal_df = to_pandas(uproot.concatenate(DATAFILE + ':nominal_Loose', BRANCHES_NOMINAL, num_workers=N_THREADS))
print(f"Importing nominal from ROOT: {time.time() - t:.3f}s")

t = time.time()
truth_df = to_pandas(uproot.concatenate(DATAFILE + ':truth', BRANCHES_TRUTH, num_workers=N_THREADS))
print(f"Importing truth from ROOT: {time.time() - t:.3f}s")

# merge
t = time.time()
df = pd.merge(nominal_df, truth_df, how='left', on='eventNumber', sort=False, copy=False, suffixes=('_reco', '_truth'), indicator=True)
print(f"Merging truth and nominal: {time.time() - t:.3f}s")

del nominal_df, truth_df

# calculate sum of weights
t = time.time()
sumw = to_pandas(uproot.concatenate(DATAFILE + ':sumWeights', ['dsid', 'totalEventsWeighted'], num_workers=N_THREADS))
sumw = sumw.groupby('dsid').sum()
df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)
print(f"Calculating sum of weights: {time.time() - t:.3f}s")

# scale GeV
GeV_cols = ['MC_WZ_dilep_m_born', 'mu_pt', 'met_met', 'MC_WZneutrino_pt_born', 'MC_WZmu_el_pt_born']
t = time.time()
df.loc[GeV_cols] /= 1000
# df['MC_WZ_dilep_m_born'] /= 1000
# df['MC_WZmu_el_pt_born'] /= 1000
# df['MC_WZneutrino_pt_born'] /= 1000
# df['mu_pt'] /= 1000
# df['met_met'] /= 1000
print(f"Rescaling to GeV: {time.time() - t:.3f}s")

# calculate total truth weight
t = time.time()
df['total_truth_weight'] = LUMI_DATA * df['weight_mc'] * abs(df['weight_mc']) / df['totalEventsWeighted'] \
                                 * df['KFactor_weight_truth'] * df['weight_pileup']
print(f"Calculating truth event weight {time.time() - t:.3f}s")

# calculate total reco weight
t = time.time()
df['total_truth_weight'] = LUMI_DATA * df['weight_mc'] * abs(df['weight_mc']) / df['totalEventsWeighted'] \
                                 * df['weight_kFactor'] * df['weight_pileup'] * df['weight_leptonSF']
print(f"Calculating reco event weight {time.time() - t:.3f}s")

# calculate mt
t = time.time()
dphi = abs(df['MC_WZneutrino_phi_born'] - df['MC_WZmu_el_phi_born'])
dphi.loc[dphi > np.pi] = 2 * np.pi - dphi.loc[dphi > np.pi]
df['mt'] = np.sqrt(2. * df['MC_WZmu_el_pt_born'] * df['MC_WZneutrino_pt_born'] * (1 - np.cos(dphi)))
print(f"Calculating mt {time.time() - t:.3f}s")


def plot(var: str, xlabel: str, weight: str) -> None:
    # plot
    t0 = time.time()
    for dsid in df['mcChannelNumber'].unique():
        # per dsid
        df_dsid = df.loc[df['mcChannelNumber'] == dsid]
        hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
        hist.fill(df_dsid[var], weight=df_dsid[weight], threads=N_THREADS // 2)
        hep.histplot(hist, label=dsid)
    # inclusive
    hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
    hist.fill(truth_df[var], weight=truth_df[weight])
    hep.histplot(hist, label='Inclusive', color='k')
    plt.xlabel(xlabel)
    plt.ylabel("Entries")
    plt.semilogy()
    plt.legend(fontsize=10, ncol=2)
    hep.atlas.label(italic=(True, True), llabel='Internal', rlabel=r'$W^+\rightarrow\tau\nu$ 13TeV')
    plt.savefig(OUT_DIR + 'wplustaunu_' + var + '.png')
    plt.show()
    print(f"Making {var} plot: {time.time() - t0:.3f}s")
    plt.clf()


plot('MC_WZ_dilep_m_born', r"Born $M_{ll}$ [GeV]", weight='total_truth_weight')
plot('mt', r"$M_T^W$ [GeV]", weight='total_truth_weight')
plot('mu_phi', r"$\tau \phi$ [GeV]", weight='total_truth_weight')
plot('mu_pt', r"$\tau p_T$ [GeV]", weight='total_reco_weight')
