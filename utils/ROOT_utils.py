import glob
import logging
import pathlib
import pickle as pkl
from contextlib import contextmanager
from os import PathLike
from typing import Type

import ROOT
import boost_histogram as bh
import numpy as np


# ROOT settings
def load_ROOT_settings(set_batch: bool = True, gui: bool = False, th1_dir: bool = False, optstat: int = 1111):
    ROOT.gROOT.SetBatch(set_batch)  # Prevents TCanvas popups
    ROOT.PyConfig.StartGUIThread = gui  # disables polling for ROOT GUI events
    ROOT.TH1.AddDirectory(th1_dir)  # stops TH1s from being saved and prevents overwrite warnings
    ROOT.TH1.SetDefaultSumw2()  # Sets weighted binning in all ROOT histograms by default
    ROOT.EnableImplicitMT()  # enable multithreading
    ROOT.gStyle.SetOptStat(optstat)

    # load custom C++ functions
    ROOT.gSystem.Load(f'{pathlib.Path(__file__).parent}/rootfuncs.h')
    ROOT.gInterpreter.Declare(f'#include "{pathlib.Path(__file__).parent}/rootfuncs.h"')


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

    # check dimensions
    n_dims = len(h_bh.axes)
    if n_dims > 3:
        raise Exception("ROOT cannot handle Histograms with more than 3 dimensions.")

    # TH1 constructor
    binargs = [arg for t in [(len(ax), ax.edges) for ax in h_bh.axes] for arg in t]  # extracts n_bins and bin edges
    try:
        h_root = TH1_constructor[hist_type.upper()][n_dims](str(name), str(title), *binargs)
    except TypeError as e:
        raise e

    for idx, bin_cont in np.ndenumerate(h_bh.view(flow=True)):
        h_root.SetBinContent(*idx, bin_cont[0])  # bin value
        h_root.SetBinError(*idx, np.sqrt(bin_cont[1]))  # root sum of weights

    return h_root


def TH1_to_bh(h_root: Type[ROOT.TH1]) -> bh.Histogram:
    """Converts a ROOT TH1 into a boost-histogram"""

    # construct axes
    edges = [h_root.GetBinLowEdge(i) for i in range(h_root.GetNbinsX() + 2)]
    axes = [bh.axis.Variable(ax_bins) for ax_bins in edges]

    # filling bins contents n-dimensionally for different storage types
    # not very pythonic but shouldn't take too long as long as the histogram isn't too big
    h_bh = bh.Histogram(*axes, storage=bh.storage.Weight())
    for idx, _ in np.ndenumerate(h_bh.view(flow=True)):
        h_bh.view(flow=True).value[idx] = h_root.GetBinContent(*idx)  # bin value
        h_bh.view(flow=True).variance[idx] = h_root.GetBinError(*idx) ** 2

    return h_bh


def convert_pkl_to_root(filename: str, histname: str | None = None) -> None:
    """
    Converts pickled histogram file to root file.

    Reads pickle files containing either a boost-histogram Histogram object, a dictionary or other iterable containing
    boost histograms
    """
    with open(filename, 'rb') as f:
        obj = pkl.load(f)

    rootfilename = filename.replace('.pkl', '.root')

    if isinstance(obj, bh.Histogram):
        logging.info(f"Printing {histname} to {rootfilename}")
        with ROOT_TFile_mgr(rootfilename):
            bh_to_TH1(obj, name=histname, title=histname).Write()
        return
    elif isinstance(obj, dict):
        TH1s = [bh_to_TH1(hist, name, name) for name, hist in obj.items() if isinstance(hist, bh.Histogram)]
    elif hasattr(obj, '__iter__'):
        TH1s = [bh_to_TH1(hist, 'hist'+i, 'hist'+i) for i, hist in obj if isinstance(hist, bh.Histogram)]
    else:
        raise ValueError(f"No boost-histogram objects found in object {obj}.")

    if len(TH1s) > 0:
        with ROOT_TFile_mgr(rootfilename):
            for h in TH1s: h.Write()
            logging.info(f"Histograms saved to {rootfilename}")
        return


@contextmanager
def ROOT_TFile_mgr(filename: str | PathLike, TFile_arg='RECREATE'):
    """Context manager for opening root files"""
    file = ROOT.TFile(filename, TFile_arg)
    try:
        yield file
    finally:
        file.Close()


def glob_chain(TTree: str, path: PathLike | str) -> ROOT.TChain:
    """Return TChain with glob'd files because ROOT can't glob by itself"""
    chain = ROOT.TChain(TTree)
    for file in glob.glob(path):
        chain.Add(file)
    return chain

# # not sure why this doesn't work
# class RDataFrame(ROOT.RDataFrame):
#     __slots__ = 'chain'
#
#     def __init__(self, path: PathLike | str, tree: str):
#         """Allow RDataFrames to be constructed with a wildcard in the filepath"""
#         self.chain = glob_chain(tree, path)
#         super().__init__(self.chain)


def get_dta_sumw(path: str) -> float:
    """get total xAOD sum of weights for sample in DTA"""
    files_list = glob.glob(path)
    sum_of_weights = 0.
    for file in files_list:
        with ROOT_TFile_mgr(file, 'read') as f:
            sum_of_weights += f.Get("sumOfWeights").GetBinContent(4)
    return sum_of_weights
