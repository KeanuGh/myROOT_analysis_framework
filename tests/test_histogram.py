import ROOT
import numpy as np
import pytest

from src.histogram import Histogram1D

ROOT.TH1.SetDefaultSumw2()  # default to weighted TH1s
ROOT.TH1.AddDirectory(False)  # avoid TH1 overwrite warning
np.random.seed(42)

nbins = 20
xmin = -1.
xmax = 1.

a = np.random.normal(0, 1, 100000)
b = np.random.normal(0.5, 1.1, 100000)
w = .4 * np.random.random(100000) + 0.8  # between 0.8 and 0.2

# regular
mine_h = Histogram1D(a, (nbins, xmin, xmax), weight=w)
root_h = ROOT.TH1F('test', 'test;test', nbins, xmin, xmax)
for ia, iw in zip(a, w):
    root_h.Fill(ia, iw)

# extra histograms for operations
mine_h2 = Histogram1D(b, (nbins, xmin, xmax), weight=w)
root_h2 = ROOT.TH1F('test', 'test;test', nbins, xmin, xmax)
for ib, iw2 in zip(b, w):
    root_h2.Fill(ib, iw2)

# multiply by 10
mine_h_10 = mine_h * 10
root_h_10 = root_h * 10

# divide by 10
mine_h_01 = mine_h / 10
root_h_01 = root_h.Clone()
root_h_01.Scale(0.1)

# normalised
mine_h_normed = mine_h.normalised()
root_h_normed = root_h.Clone()
root_h_normed.Scale(1 / root_h.Integral())

# normalised to
mine_h_normed_10 = mine_h.normalised_to(10)
root_h_normed_10 = root_h.Clone()
root_h_normed_10.Scale(10 / root_h.Integral())

# ratio
mine_h_ratio = mine_h / mine_h2
root_h_ratio = root_h.Clone()
root_h_ratio.Divide(root_h2)
root_h_ratio.ResetStats()

# product
mine_h_prod = mine_h * mine_h2
root_h_prod = root_h.Clone()
root_h_prod.Multiply(root_h2)


@pytest.mark.parametrize(
    'my_hist,root_hist',
    [
        pytest.param(mine_h, root_h, id='regular'),
        pytest.param(mine_h2, root_h2, id='regular2'),
        pytest.param(mine_h_10, root_h_10, id='* 10'),
        pytest.param(mine_h_01, root_h_01, id='/ 10'),
        pytest.param(mine_h_normed, root_h_normed, id='normalised'),
        pytest.param(mine_h_normed_10, root_h_normed_10, id='normalised to 10'),
        pytest.param(mine_h_ratio, root_h_ratio, id='ratio'),
        pytest.param(mine_h_prod, root_h_prod, id='product'),

        pytest.param(mine_h, mine_h.TH1, id='regular - bh-selfTH1'),
        pytest.param(mine_h2, mine_h2.TH1, id='regular2 - bh-selfTH1'),
        pytest.param(mine_h_10, mine_h_10.TH1, id='* 10 - bh-selfTH1'),
        pytest.param(mine_h_01, mine_h_01.TH1, id='/ 10 - bh-selfTH1'),
        pytest.param(mine_h_normed, mine_h_normed.TH1, id='normalised - bh-selfTH1'),
        pytest.param(mine_h_normed_10, mine_h_normed_10.TH1, id='normalised to 10 - bh-selfTH1'),
        pytest.param(mine_h_ratio, mine_h_ratio.TH1, id='ratio - bh-selfTH1'),
        pytest.param(mine_h_prod, mine_h_prod.TH1, id='product - bh-selfTH1'),
    ]
)
class TestHistogram1DTH1Comparison:
    def test_bin_edges(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_edges = [root_hist.GetBinLowEdge(i + 1) for i in range(root_hist.GetNbinsX() + 1)]
        np.testing.assert_allclose(my_hist.bin_edges, TH1_edges, rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_bin_widths(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_widths = [root_hist.GetBinWidth(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_allclose(my_hist.bin_widths, TH1_widths, rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_bin_centres(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_centres = [root_hist.GetBinCenter(i + 1) for i in range(root_hist.GetNbinsX())]
        np.testing.assert_allclose(my_hist.bin_centres, TH1_centres, rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_bin_values(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_values = [root_hist.GetBinContent(i) for i in range(root_hist.GetNbinsX() + 2)]
        np.testing.assert_allclose(my_hist.bin_values(flow=True), TH1_values, rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_bin_errors(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        TH1_errors = [root_hist.GetBinError(i) for i in range(root_hist.GetNbinsX() + 2)]
        np.testing.assert_allclose(my_hist.root_sumw2(flow=True), TH1_errors, rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_integral(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.integral, root_hist.Integral("width"), rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_bin_sum(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.bin_sum(), root_hist.Integral(), rtol=1e-05)
        assert isinstance(my_hist, Histogram1D)

    def test_entries(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.n_entries, root_hist.GetEntries(), rtol=1e-05)

    def test_eff_enties(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.eff_entries, root_hist.GetEffectiveEntries(), rtol=1e-05)

    def test_mean_bh_root(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.mean, root_hist.GetMean(), rtol=1e-05)

    def test_std(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.std, root_hist.GetStdDev(), rtol=1e-05)

    def test_mean_error(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.mean_error, root_hist.GetMeanError(), rtol=1e-05)

    def test_std_error(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        np.testing.assert_allclose(my_hist.std_error, root_hist.GetStdDevError(), rtol=1e-05)

    def test_convert(self, my_hist: Histogram1D, root_hist: ROOT.TH1F):
        # NOTE: mean and std will not pass

        new_th1 = my_hist.to_TH1()

        assert isinstance(new_th1, ROOT.TH1)

        # Edges
        TH1_edges = [root_hist.GetBinLowEdge(i) for i in range(root_hist.GetNbinsX() + 2)]
        new_TH1_edges = [new_th1.GetBinLowEdge(i) for i in range(new_th1.GetNbinsX() + 2)]
        np.testing.assert_allclose(new_TH1_edges, TH1_edges, rtol=1e-05)

        # values
        TH1_values = [root_hist.GetBinContent(i + 1) for i in range(root_hist.GetNbinsX())]
        new_TH1_values = [new_th1.GetBinContent(i + 1) for i in range(new_th1.GetNbinsX())]
        np.testing.assert_allclose(new_TH1_values, TH1_values, rtol=1e-05)

        # Centres
        TH1_centres = [root_hist.GetBinCenter(i + 1) for i in range(root_hist.GetNbinsX())]
        new_TH1_centres = [new_th1.GetBinCenter(i + 1) for i in range(new_th1.GetNbinsX())]
        np.testing.assert_allclose(new_TH1_centres, TH1_centres, rtol=1e-05)

        # widths
        TH1_widths = [root_hist.GetBinWidth(i + 1) for i in range(root_hist.GetNbinsX())]
        new_TH1_widths = [new_th1.GetBinWidth(i + 1) for i in range(new_th1.GetNbinsX())]
        np.testing.assert_allclose(new_TH1_widths, TH1_widths, rtol=1e-05)

        # errors
        TH1_errors = [root_hist.GetBinError(i + 1) for i in range(root_hist.GetNbinsX())]
        new_TH1_errors = [new_th1.GetBinError(i + 1) for i in range(new_th1.GetNbinsX())]
        np.testing.assert_allclose(new_TH1_errors, TH1_errors, rtol=1e-05)

        # integral
        np.testing.assert_allclose(new_th1.Integral("width"), root_hist.Integral("width"), rtol=1e-05)
        np.testing.assert_allclose(new_th1.Integral(), root_hist.Integral(), rtol=1e-05)
