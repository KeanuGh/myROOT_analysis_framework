import uproot4 as uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import boost_histogram as bh
import mplhep as hep

# styles
plt.style.use([hep.style.ATLAS])

filename = '../data/wminmunu_MC.root'

# extract
truth = uproot.open(filename)["truth"]

# into pandas
truth_df = truth.arrays(library='pd',
                        filter_name=[
                            'MC_WZmu_el_pt_born',
                            'MC_WZneutrino_pt_born',
                            'weight_mc',
                            'MC_WZ_pt',
                        ])

# map weight column
truth_df['weight'] = truth_df['weight_mc'].map(lambda w: 1 if w > 0 else -1)

# to GeV
GeV_cols = [
    'MC_WZneutrino_pt_born',
    'MC_WZmu_el_pt_born',
    'MC_WZ_pt',
]
truth_df[GeV_cols] = truth_df[GeV_cols].applymap(lambda val: val / 1000)

# add columns of cut booleans
truth_df['WpT_lo'] = truth_df['MC_WZ_pt'] < 50
truth_df['WpT_hi'] = truth_df['MC_WZ_pt'] > 100

# plot
fig, (ax1, ax2) = plt.subplots(1, 2)

# WpT low
WpT_lo = truth_df[truth_df['WpT_lo']]
hWpT_lo = bh.Histogram(bh.axis.Regular(20, 0, 200),
                       bh.axis.Regular(20, 0, 200),
                       )
hWpT_lo.fill(WpT_lo['MC_WZneutrino_pt_born'].to_numpy(),
             WpT_lo['MC_WZmu_el_pt_born'].to_numpy(),
             weight=WpT_lo['weight'].to_numpy(),
             threads=6,
             )
mesh1 = ax1.pcolormesh(*hWpT_lo.axes.edges.T, hWpT_lo.view().T, norm=LogNorm())
fig.colorbar(mesh1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_aspect('equal')
ax1.set_title('W $p_{T}$ < 50 GeV')
ax1.set_xlabel("neutrino $p_{T}$ [GeV]")
ax1.set_ylabel("muon $p_{T}$ [GeV]")

# WpT high
WpT_hi = truth_df[truth_df['WpT_hi']]
hWpT_hi = bh.Histogram(bh.axis.Regular(20, 0, 200),
                       bh.axis.Regular(20, 0, 200),
                       )
hWpT_hi.fill(WpT_hi['MC_WZneutrino_pt_born'].to_numpy(),
             WpT_hi['MC_WZmu_el_pt_born'].to_numpy(),
             weight=WpT_hi['weight'].to_numpy(),
             threads=6
             )
mesh2 = ax2.pcolormesh(*hWpT_hi.axes.edges.T, hWpT_hi.view().T, norm=LogNorm())
fig.colorbar(mesh2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_aspect('equal')
ax2.set_title('W $p_{T}$ > 100 GeV')
ax2.set_xlabel("neutrino $p_{T}$ [GeV]")
ax2.set_ylabel("muon $p_{T}$ [GeV]")

fig.set_figheight(15)
fig.set_figwidth(15)
plt.savefig("WCuts.png")
plt.show()
