import ROOT
import numpy as np

from src.histogram import Histogram1D

ROOT.gROOT.LoadMacro("../utils/AtlasStyle/AtlasStyle.C")
ROOT.SetAtlasStyle()
ROOT.TH1.SetDefaultSumw2()

np.random.seed(42)
a = np.random.normal(0, 1, 1000)
b = np.random.normal(0, 1, 1000)

w = .4 * np.random.random_sample(1000) + 0.8  # random sample from [-0.8, 1.2)
nbins = 30
xmin = -1
xmax = 1

h = Histogram1D(a, (nbins, xmin, xmax), w, name='test', title='test')
h2 = Histogram1D(b, (nbins, xmin, xmax), w, name='test2', title='test2')
h.plot(out_filename='bh_hist.png', color='k')
h2.plot(out_filename='bh_hist2.png', color='k')
h.plot_ratio(h2, out_filename='bh_ratio.png', color='k', fit=True)

h3 = h / h2
h3.Rplot(stats_box=True, out_filename='Rplot.png')

th = ROOT.TH1F('test', 'test', nbins, xmin, xmax)
th2 = ROOT.TH1F('test2', 'test2', nbins, xmin, xmax)
for ia, ib, iw in zip(a, b, w):
    th.Fill(ia, iw)
    th2.Fill(ib, iw)
c = ROOT.TCanvas()
th.Draw("Hist E")
c.Print("root_hist.png")
th2.Draw("Hist E")
c.Print("root_hist2.png")

th.Divide(th2)
th.Fit('pol0', 'F')
# th.Draw("Hist E")
c.Print("root_ratio.png")
