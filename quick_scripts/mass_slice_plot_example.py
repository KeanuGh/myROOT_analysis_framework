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
BRANCHES = [
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
truth_df = to_pandas(uproot.concatenate(DATAFILE + ':truth', BRANCHES, num_workers=N_THREADS))
print(f"Importing from ROOT: {time.time() - t:.3f}s")\

# # delete duplicate events
# t = time.time()
# len_before = len(truth_df.index)
# truth_df.drop_duplicates('eventNumber', keep='first', inplace=True)
# print(f"Dropping duplicates: {time.time() - t:.3f}s ({len_before - len(truth_df.index)} duplicates found)")

# calculate sum of weights
t = time.time()
sumw = to_pandas(uproot.concatenate(DATAFILE + ':sumWeights', ['dsid', 'totalEventsWeighted'], num_workers=N_THREADS))
sumw = sumw.groupby('dsid').sum()
truth_df = pd.merge(truth_df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False, copy=False)
print(f"Calculating sum of weights: {time.time() - t:.3f}s")

# scale GeV
truth_df['MC_WZ_dilep_m_born'] /= 1000
truth_df['MC_WZmu_el_pt_born'] /= 1000
truth_df['MC_WZneutrino_pt_born'] /= 1000

# calculate total event weight
t = time.time()
truth_df['total_w'] = LUMI_DATA * truth_df['weight_mc'] * abs(truth_df['weight_mc']) / truth_df['totalEventsWeighted'] \
                      * truth_df['KFactor_weight_truth'] * truth_df['weight_pileup']
print(f"Calculating total event weight {time.time() - t:.3f}s")

# calculating cuts


# calculate mt
t = time.time()
dphi = abs(truth_df['MC_WZneutrino_phi_born'] - truth_df['MC_WZmu_el_phi_born'])
dphi.loc[dphi > np.pi] = 2 * np.pi - dphi.loc[dphi > np.pi]
truth_df['mt'] = np.sqrt(2. * truth_df['MC_WZmu_el_pt_born'] * truth_df['MC_WZneutrino_pt_born'] * (1 - np.cos(dphi)))
print(f"Calculating mt {time.time() - t:.3f}s")


def plot(var: str, xlabel: str) -> None:
    # plot
    t0 = time.time()
    for dsid in truth_df['mcChannelNumber'].unique():
        # per dsid
        truth_df_dsid = truth_df.loc[truth_df['mcChannelNumber'] == dsid]
        hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
        hist.fill(truth_df_dsid[var], weight=truth_df_dsid['total_w'], threads=N_THREADS // 2)
        hep.histplot(hist, label=dsid)
    # inclusive
    hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
    hist.fill(truth_df[var], weight=truth_df['total_w'])
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


plot('MC_WZ_dilep_m_born', "Born $M_{ll}$ [GeV]")
plot('mt', "$M_T^W$ [GeV]")
_weight_truth'] * truth_df['weight_pileup']
print(f"Calculating total event weight {time.time() - t:.3f}s")

# calculating cuts


# calculate mt
t = time.time()
dphi = abs(truth_df['MC_WZneutrino_phi_born'] - truth_df['MC_WZmu_el_phi_born'])
dphi.loc[dphi > np.pi] = 2 * np.pi - dphi.loc[dphi > np.pi]
truth_df['mt'] = np.sqrt(2. * truth_df['MC_WZmu_el_pt_born'] * truth_df['MC_WZneutrino_pt_born'] * (1 - np.cos(dphi)))
print(f"Calculating mt {time.time() - t:.3f}s")


def plot(var: str, xlabel: str) -> None:
    # plot
    t0 = time.time()
    for dsid in truth_df['mcChannelNumber'].unique():
        # per dsid
        truth_df_dsid = truth_df.loc[truth_df['mcChannelNumber'] == dsid]
        hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
        hist.fill(truth_df_dsid[var], weight=truth_df_dsid['total_w'], threads=N_THREADS // 2)
        hep.histplot(hist, label=dsid)
    # inclusive
    hist = bh.Histogram(bh.axis.Regular(50, 0, 5000))
    hist.fill(truth_df[var], weight=truth_df['total_w'])
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


plot('MC_WZ_dilep_m_born', "Born $M_{ll}$ [GeV]")
plot('mt', "$M_T^W$ [GeV]")
