import ROOT
import numpy as np
import pytest

from src.histogram import Histogram1D

ROOT.TH1.SetDefaultSumw2()
ROOT.gROOT.SetBatch()
np.random.seed(42)

nbins = 20
xmin = -1.
xmax = 1.

# regular
a = np.random.normal(0, 1, 1000)
w = np.random.random(1000)
mine_h = Histogram1D(a, (nbins, xmin, xmax), weight=w)
root_h = ROOT.TH1F('test', 'test;test', nbins, xmin, xmax)
for ia, iw in zip(a, w):
    root_h.Fill(ia, iw)

# extra histograms for operations
b = np.random.normal(0.5, 1.1, 1000)
w2 = np.random.random(1000)
mine_h2 = Histogram1D(b, (nbins, xmin, xmax), weight=w2)
root_h2 = ROOT.TH1F('test', 'test;test', nbins, xmin, xmax)
for ib, iw2 in zip(b, w2):
    root_h2.Fill(ib, iw2)

# multiply by 10
mine_h_10 = mine_h * 10
root_h_10 = root_h * 10

# multiply by 10
mine_h_01 = mine_h / 10
root_h_01 = root_h.Clone()
root_h_01.Scale(0.1)

# normalised
mine_h_normed = mine_h.normalised()
root_h_normed = root_h.Clone()
root_h_normed.Scale(1 / root_h.Integral())

# ratio
mine_h_ratio = mine_h / mine_h2
root_h_ratio = root_h.Clone()
root_h_ratio.Divide(root_h2)

# product
mine_h_prod = mine_h * mine_h2
root_h_prod = root_h.Clone()
root_h_prod.Multiply(root_h2)


@pytest.mark.parametrize(
    'my_hist,root_hist',
    [
        pytest.param(mine_h, root_h, id='regular'),
        pytest.param(mine_h_10, root_h_10, id='* 10'),
        pytest.param(mine_h_01, root_h_01, id='/ 10'),
        pytest.param(mine_h_normed, root_h_normed, id='normalised'),
        pytest.param(mine_h_ratio, root_h_ratio, id='ratio'),
        pytest.param(mine_h_prod, root_h_prod, id='product'),
    ]
)
class TestHistogram1D:
    def test_bin_edges(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_edges = [root_hist.GetBinLowEdge(i + 1) for i in range(root_hist.GetNbinsX())]
        TH1_edges.append(TH1_edges[-1] + root_hist.GetBinWidth(root_hist.GetNbinsX()))
        np.testing.assert_almost_equal(my_hist.bin_edges, TH1_edges, decimal=4)

    def test_bin_values(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_values = [root_hist.GetBinContent(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_almost_equal(my_hist.bin_values, TH1_values, decimal=4)

    def test_bin_widths(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_widths = [root_hist.GetBinWidth(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_almost_equal(my_hist.bin_widths, TH1_widths, decimal=4)

    def test_bin_centres(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_centres = [root_hist.GetBinCenter(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_almost_equal(my_hist.bin_centres, TH1_centres, decimal=4)

    def test_bin_errors(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_errors = [root_hist.GetBinError(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_almost_equal(my_hist.root_sumw2, TH1_errors, decimal=4)

    def test_integral(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_almost_equal(my_hist.integral, root_hist.Integral("width"), decimal=4)

    def test_bin_sum(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_almost_equal(my_hist.bin_sum, root_hist.Integral(), decimal=4)
