"""
tests. Put this into actual tests once you get pytest working again lmao
please test with empty histograms, multidimensional histograms and histograms with log binning
"""
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import boost_histogram as bh
import ROOT
import root_numpy as rnp
from utils.ROOT_utils import bh_to_TH1, TH1_to_bh

out_dir = '../outputs/test_outputs/'
ROOT.TH1.SetDefaultSumw2()  # Sets weighted binning in all ROOT histograms by default
ROOT.gROOT.SetBatch()  # Prevents TCanvas popups

data = np.random.normal(0, 0.3, 1000)
data2 = np.random.normal(0, 0.2, 1000)


def test_bh_to_root_with_weights_1d():
    bh_hist = bh.Histogram(bh.axis.Regular(20, -1, 1), storage=bh.storage.Weight())
    bh_hist.fill(data, weight=2)
    hep.histplot(bh_hist.view().value, bins=bh_hist.axes[0].edges, yerr=np.sqrt(bh_hist.view().variance))
    plt.savefig(out_dir+'bh_to_root_bh_hist_1d.png')
    plt.clf()

    root_hist = bh_to_TH1(bh_hist, 'test', 'test;x;entries')
    c1 = ROOT.TCanvas()
    c1.cd()
    root_hist.Draw("histE")
    c1.Print(out_dir+'bh_to_root_root_hist_1d.png')

    for i, binval in enumerate(bh_hist.view(flow=True)):
        assert binval.value == root_hist.GetBinContent(i)
        assert np.sqrt(binval.variance) == root_hist.GetBinError(i)


def test_bh_to_root_with_weights_2d():
    fig, ax = plt.subplots()
    bh_hist = bh.Histogram(bh.axis.Regular(20, -1, 1),
                           bh.axis.Regular(20, -1, 1),
                           storage=bh.storage.Weight())

    bh_hist.fill(data, data2, weight=2, threads=6)
    mesh = ax.pcolormesh(*bh_hist.axes.edges.T, bh_hist.view().value.T)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect(1 / ax.get_data_ratio())
    plt.savefig(out_dir+'bh_to_root_bh_hist_2d.png')
    plt.clf()

    root_hist = bh_to_TH1(bh_hist, 'test', 'test;x;entries')
    c2 = ROOT.TCanvas()
    c2.cd()
    root_hist.Draw("Colz")
    c2.Print(out_dir+'bh_to_root_root_hist_2d.png')

    for i, binval in np.ndenumerate(bh_hist.view(flow=True)):
        assert binval[0] == root_hist.GetBinContent(*i)
        assert np.sqrt(binval[1]) == root_hist.GetBinError(*i)


def test_root_to_bh_with_weights_1d():
    root_hist = ROOT.TH1F('test', 'test;x;entries', 20, -1, 1)

    rnp.fill_hist(root_hist, data, weights=np.full(data.shape[0], 2))  # fill
    c3 = ROOT.TCanvas()
    c3.cd()
    root_hist.Draw("histE")
    c3.Print(out_dir+'root_to_bh_root_hist_1d.png')

    bh_hist = TH1_to_bh(root_hist)
    hep.histplot(bh_hist.view().value, bins=bh_hist.axes[0].edges, yerr=np.sqrt(bh_hist.view().variance))
    plt.savefig(out_dir+'root_to_bh_hist_1d.png')
    plt.clf()

    for i, binval in enumerate(bh_hist.view(flow=True)):
        assert binval.value == root_hist.GetBinContent(i)
        assert np.sqrt(binval.variance) == root_hist.GetBinError(i)


def test_root_to_bh_with_weights_2d():
    root_hist = ROOT.TH2F('test', 'test;x;entries', 20, -1, 1, 20, -1, 1)

    data2d = np.array([data, data2]).T
    rnp.fill_hist(root_hist, data2d, weights=np.full(data2d.shape[0], 2))  # fill
    c4 = ROOT.TCanvas()
    c4.cd()
    root_hist.Draw("Colz")
    c4.Print(out_dir+'root_to_bh_root_hist_2d.png')

    fig, ax = plt.subplots()
    bh_hist = TH1_to_bh(root_hist)
    mesh = ax.pcolormesh(*bh_hist.axes.edges.T, bh_hist.view().value.T)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect(1 / ax.get_data_ratio())
    plt.savefig(out_dir+'root_to_bh_hist_2d.png')
    plt.clf()

    for i, binval in np.ndenumerate(bh_hist.view(flow=True)):
        assert binval[0] == root_hist.GetBinContent(*i)
        assert np.sqrt(binval[1]) == root_hist.GetBinError(*i)


test_bh_to_root_with_weights_1d()
test_bh_to_root_with_weights_2d()
test_root_to_bh_with_weights_1d()
test_root_to_bh_with_weights_2d()
