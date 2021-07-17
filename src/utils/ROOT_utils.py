import pickle as pkl
from contextlib import contextmanager
from typing import Type, Optional

import ROOT
import boost_histogram as bh
import numpy as np

# import uproot

# this dictionary decides which ROOT constructor needs to be called based on hist type and dimension
# call it with TH1_constructor[type][n_dims](name, title, *n_bins, *bin_edges) where 'type' is in {'F','D','I','C','S'}
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


def bh_to_TH1(h_bh: bh.Histogram, name: str, title: str, hist_type: str = 'F') -> Type[ROOT.TH1]:
    """
    Converts a boost-histogram histogram into a ROOT TH1 histogram. Only works for histograms with numeric axes.
    """

    # check type
    if hist_type.upper() not in TH1_constructor:
        raise ValueError(f"TH1 types: {', '.join(TH1_constructor.keys())}. {hist_type.upper()} was input.")

    # check dims
    n_dims = len(h_bh.axes)
    if n_dims > 3:
        raise Exception("ROOT cannot handle Histograms with more than 3 dimensions.")

    # TH1 constructor
    binargs = [arg for t in [(len(ax), ax.edges) for ax in h_bh.axes] for arg in t]  # extracts n_bins and bin edges
    try:
        h_root = TH1_constructor[hist_type.upper()][n_dims](str(name), str(title), *binargs)
    except TypeError as e:
        print("Input Arguments: ", name, title, *binargs)
        raise e

    # filling bins contents n-dimensionally for different storage types
    # not very pythonic but shouldn't take too long as long as the histogram isn't too big
    if hasattr(h_bh.view(), 'value'):
        # for Weighted/Mean/WeightedMean Accumulator storages
        for idx, bin_cont in np.ndenumerate(h_bh.view(flow=True)):
            h_root.SetBinContent(*idx, bin_cont[0])  # bin value
            h_root.SetBinError(*idx, np.sqrt(bin_cont[1]))  # root sum of weights
    else:
        # Sum storage
        for idx, bin_cont in np.ndenumerate(h_bh.view(flow=True)):
            h_root.SetBinContent(*idx, bin_cont)  # bin value

    return h_root


# def TH1_to_bh(h_root: Type[ROOT.TH1]) -> bh.Histogram:
#     """Converts a ROOT TH1 into a boost-histogram"""
#
#     # construct axes
#     edges = rnp.hist2array(h_root, include_overflow=True, return_edges=True)[1]
#     axes = [bh.axis.Variable(ax_bins) for ax_bins in edges]
#
#     # filling bins contents n-dimensionally for different storage types
#     # not very pythonic but shouldn't take too long as long as the histogram isn't too big
#     h_bh = bh.Histogram(*axes, storage=bh.storage.Weight())
#     for idx, _ in np.ndenumerate(h_bh.view(flow=True)):
#         h_bh.view(flow=True).value[idx] = h_root.GetBinContent(*idx)  # bin value
#         h_bh.view(flow=True).variance[idx] = h_root.GetBinError(*idx) ** 2
#
#     return h_bh


def convert_pkl_to_root(filename: str, histname: Optional[str] = None) -> None:
    """
    Converts pickled histogram file to root file.

    Reads pickle files containing either a boost-histogram Histogram object, a dictionary or other iterable containing
    boost histograms
    """
    with open(filename, 'rb') as f:
        obj = pkl.load(f)

    rootfilename = filename.replace('.pkl', '.root')
    TH1s = []

    if isinstance(obj, bh.Histogram):
        print(f"Printing {histname} to {rootfilename}")
        with ROOT_file(rootfilename):
            bh_to_TH1(obj, name=histname, title=histname).Write()
        return
    elif isinstance(obj, dict):
        TH1s = [bh_to_TH1(hist, name, name) for name, hist in obj.items() if isinstance(hist, bh.Histogram)]
    elif hasattr(obj, '__iter__'):
        TH1s = [bh_to_TH1(hist, 'hist'+i, 'hist'+i) for i, hist in obj if isinstance(hist, bh.Histogram)]
    else:
        print(f"No boost histogram objects found in {obj}")

    if len(TH1s) > 0:
        with ROOT_file(rootfilename):
            [h.Write() for h in TH1s]
            print(f"Histograms saved to {rootfilename}")
        return

    raise ValueError(f"No Histogram found in file {filename}")


@contextmanager
def ROOT_file(filename: str, TFile_arg='RECREATE'):
    """Context manager for opening root files"""
    file = ROOT.TFile(filename, TFile_arg)
    try:
        yield file
    finally:
        file.Close()

