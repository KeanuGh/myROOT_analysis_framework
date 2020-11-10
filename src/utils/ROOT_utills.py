import ROOT
from typing import Type
import boost_histogram as bh
from numpy import sqrt

# this dictionary decides which ROOT constructor needs to be called based on hist type and dimension
# call it with TH1_constructor[type][n_dims](name, title, *n_bins, *bin_edges)
TH1_constructor = {
    'F': {  # histograms with one float per channel. Maximum precision 7 digits
        1: ROOT.TH1F,
        2: ROOT.TH2F,
        3: ROOT.TH3F
    },
    'D': {  # histograms with one double per channel. Maximum precision 14 digits
        1: ROOT.TH1D,
        2: ROOT.TH2D,
        3: ROOT.TH3D
    },
    'I': {  # histograms with one int per channel. Maximum bin content = 2147483647
        1: ROOT.TH1I,
        2: ROOT.TH2I,
        3: ROOT.TH3I
    },
    'C': {  # histograms with one byte per channel. Maximum bin content = 127
        1: ROOT.TH1C,
        2: ROOT.TH2C,
        3: ROOT.TH3C
    },
    'S': {  # histograms with one short per channel. Maximum bin content = 32767
        1: ROOT.TH1S,
        2: ROOT.TH2S,
        3: ROOT.TH3S
    },
}


def bh_to_root(bh_hist: bh.Histogram, name: str, title: str, hist_type='F') -> Type[ROOT.TH1]:
    """Converts a boost-histogram histogram into a ROOT TH1 histogram."""

    # check type
    TH1_types = {'F', 'D', 'I', 'C', 'S'}
    if hist_type.upper() not in TH1_types:
        raise ValueError(f"TH1 types: {', '.join(TH1_types)}. {hist_type.upper()} was input.")

    # check dims
    hist_dim = len(bh_hist.axes)
    if hist_dim > 3:
        raise Exception("ROOT cannot handle Histograms with more than 3 dimensions.")

    # TH1 constructor
    h_root = TH1_constructor[hist_type.upper()][hist_dim](name, title,
                                                          *[len(ax) for ax in bh_hist.axes],
                                                          *[ax.edges for ax in bh_hist.axes])

    # filling bins contents
    # TODO: what about histograms without weight storage?
    if hist_dim == 1:
        h_root.SetBinContent(0, bh_hist[bh.underflow].value)  # underflow bin
        h_root.SetBinContent(-1, bh_hist[bh.overflow].value)  # overflow bin
        for i, bin_cont in enumerate(bh_hist.view()):
            h_root.SetBinContent(i + 1, bin_cont.value)  # bin value
            h_root.SetBinError(i + 1, sqrt(bin_cont.variance))  # root sum of weights
    elif hist_dim == 2:
        for i in (0, 1):
            h_root.SetBinContent(0, i, bh_hist.axes[i][bh.underflow][1])  # underflow bin
            h_root.SetBinContent(-1, i, bh_hist.axes[i][bh.overflow][0])  # overflow bin
            for j, bin_cont in enumerate(bh_hist.axes[i].view()):
                h_root.SetBinContent(j + 1, i, bin_cont.value)  # bin value
                h_root.SetBinError(j + 1, i, sqrt(bin_cont.variance))  # root sum of weights

    return h_root
