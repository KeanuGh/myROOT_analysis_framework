import time

import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import uproot4 as uproot
from mplhep import label as label_base

plt.style.use([hep.style.ATLAS,
               {'font.sans-serif': ['Tex Gyre Heros']},  # use when helvetica isn't installed
               {'axes.labelsize': 23},
               {'axes.labelpad': 23},
               ])


def scale_and_plot(frame, new_lumi: float = 1., c=None, linewidth=1, label=None):
    xs = frame['weight_mc'].abs().sum() / len(frame.index)
    lumi = frame['weight'].sum() / xs

    hist = bh.Histogram(bh.axis.Regular(100, 300, 10000, transform=bh.axis.transform.log), storage=bh.storage.Weight())
    hist.fill(frame['MC_WZ_dilep_m_born'], weight=new_lumi*frame['weight'], threads=6)

    # scale cross-section
    hist /= hist.axes[0].widths

    hep.histplot(hist.view().value, bins=hist.axes[0].edges,
                 color=c, linewidth=linewidth, label=label)
    return xs, lumi


def main():
    paths = '../data/mc16a_wmintaunu_SLICES/*.root'
    new_lumi = 140

    cols = ['MC_WZ_dilep_m_born', 'weight_mc', 'mcChannelNumber']

    ti = time.time()
    df = uproot.concatenate(paths + ':truth', filter_name=cols, library='pd')
    sumw = uproot.concatenate(paths + ':sumWeights', filter_name=['totalEventsWeighted', 'dsid'], library='pd')
    df = pd.merge(df, sumw.groupby('dsid').sum(), left_on='mcChannelNumber', right_on='dsid', sort=False)
    del sumw
    df.rename(columns={'mcChannelNumber': 'DSID'}, inplace=True)
    tl = time.time()
    print(f"time to build dataframe: {tl - ti:.2f}s")

    # calculating the weight
    df['weight'] = (df['weight_mc'] * df['weight_mc'].abs()) / df['totalEventsWeighted']

    # rescale to GeV
    df['MC_WZ_dilep_m_born'] /= 1000

    for dsid, mass_slice in df.groupby('DSID'):
        xs, lumi = scale_and_plot(mass_slice, label=dsid)
        print(f"DSID: {dsid}, luminosity: {lumi:.3g}, xs: {xs:.3g}")

    xs, lumi = scale_and_plot(df, c='k', linewidth=2, label='All DIDs')
    print(f"ALL: luminosity: {lumi:.3g}, xs: {xs:.3g}")

    plt.legend(fontsize=10, ncol=2, loc='upper right')
    plt.semilogy()
    plt.semilogx()
    label_base._exp_label(exp='ATLAS', data=True, paper=True, italic=(True, True),
                          llabel='Internal', rlabel=r"$W^-\rightarrow\tau\bar{\nu}$ mass slices")
    plt.xlabel("Born dilep m [GeV]")
    plt.ylabel(r"$\frac{d\sigma}{dm_{ll}}$ [fb GeV$^{-1}$]")
    plt.show()


if __name__ == '__main__':
    main()
