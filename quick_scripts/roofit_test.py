import ROOT
import matplotlib.pyplot as plt
import numpy as np

from src.histogram import Histogram1D

np.random.seed(0)

bins = (10, 0, 1)
data1 = np.random.uniform(0, 1, 1000)
data2 = np.random.uniform(0, 1, 1000)

h1 = Histogram1D(data1, bins)
h2 = Histogram1D(data2, bins)

h1.plot_ratio(h2, fit=True)
plt.show()

# roofit
print("ROOFIT:\n--------------------------------------------------------------------------------")

x = ROOT.RooRealVar("x", "x", 0, 1)
c = ROOT.RooRealVar("c", "c", 0, 2)
linear = ROOT.RooPolynomial("linear", "linear ratio fit", x, c, lowestOrder=0)

h1 = ROOT.TH1F("", "", *bins)
for i in data1:
    h1.Fill(i)
h2 = ROOT.TH1F("", "", *bins)
for i in data2:
    h2.Fill(i)
h3 = h2.Clone()
h3.Divide(h1)
data = ROOT.RooDataHist("", "", [x], ROOT.RooFit.Import(h3))

fit_result = linear.fitTo(data,
                          # ROOT.RooFit.NumCPU(12),
                          ROOT.RooFit.SumW2Error(True),
                          ROOT.RooFit.Minimizer("Minuit"),
                          ROOT.RooFit.Verbose(True),
                          ROOT.RooFit.Hesse(False),
                          Save=True,
                          )

xframe = x.frame(ROOT.RooFit.Title("linear fit with data"))
data.plotOn(xframe)
linear.plotOn(xframe)

fit_result.Print()

canvas = ROOT.TCanvas("Cool fit", "cool fit")
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetLeftMargin(0.15)
xframe.GetYaxis().SetTitleOffset(1.6)
xframe.Draw()

canvas.SaveAs("coolplot.png")
