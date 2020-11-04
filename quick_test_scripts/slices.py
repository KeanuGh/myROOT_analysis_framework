import uproot4 as uproot
import time
import pandas as pd
import boost_histogram as bh
import mplhep as hep
import matplotlib.pyplot as plt


plt.style.use([hep.style.ATLAS,
               {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
               {'axes.labelsize': 23},
               {'axes.labelpad': 23},
               ])


def scale_and_plot(frame, c=None, linewidth=1):
    xs = frame['weight_mc'].abs().sum() / len(mass_slice.index)
    lumi = frame['weight'].sum() / xs

    hist = bh.Histogram(bh.axis.Regular(100, 300, 10000, transform=bh.axis.transform.log), storage=bh.storage.Weight())
    hist.fill(frame['MC_WZ_dilep_m_born'], weight=frame['weight'], threads=6)

    # scale cross-section
    hist /= hist.axes[0].widths

    hep.histplot(hist.view().value, bins=hist.axes[0].edges, color=c, linewidth=linewidth)
    return xs, lumi


paths = '../data/mc16a_wmintaunu_SLICES/*.root'
new_lumi = 140

cols = ['MC_WZ_dilep_m_born', 'weight_mc', 'mcChannelNumber']

ti = time.time()
df = uproot.concatenate(paths + ':truth', filter_name=cols, library='pd')
print(len(df.index))
sumw = uproot.concatenate(paths + ':sumWeights', filter_name=['totalEventsWeighted', 'dsid'], library='pd')
sumw.groupby('dsid').sum()
df = pd.merge(df, sumw, left_on='mcChannelNumber', right_on='dsid', sort=False)  # THIS IS WRONG
print(len(df.index))
tl = time.time()
print(f"time to extract: {tl - ti:.2f}s")

# calculating the weight
df['weight'] = (df['weight_mc'] * df['weight_mc'].abs()) / df['totalEventsWeighted']

# rescale to GeV
df['MC_WZ_dilep_m_born'] /= 1000

for i, mass_slice in df.groupby('mcChannelNumber'):
    xs, lumi = scale_and_plot(mass_slice)
    print(f"DSIS: {i}, luminosity: {lumi:.3g}, xs: {xs:.3g}")

xs, lumi = scale_and_plot(df, c='k', linewidth=2)
print(f"ALL: luminosity: {lumi:.3g}, xs: {xs:.3g}")

plt.semilogy()
plt.semilogx()
plt.xlabel("Born dilep m [GeV]")
plt.ylabel(r"$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]")
plt.show()
