import ROOT
import matplotlib.pyplot as plt
import numpy as np

from src.histogram import Histogram1D

file = "~/histograms_H_cvbv.root"
tree = "T_s1thv_NOMINAL"

rdf = ROOT.RDataFrame(tree, file)

th_histograms = dict()
for var in ["TruthTau", "ImplicitMet", "TruthMet"]:
    th_histograms[var + "Pt"] = ROOT.TH1F(var + "Pt", var + "Pt", 50, np.geomspace(1, 5000, 51))
    th_histograms[var + "Pt"] = rdf.Fill(th_histograms[var + "Pt"], [var + "Pt"])
    th_histograms[var + "Eta"] = ROOT.TH1F(var + "Eta", var + "Eta", 30, np.linspace(-5, 5, 31))
    th_histograms[var + "Eta"] = rdf.Fill(th_histograms[var + "Eta"], [var + "Eta"])

# convert to boost
b_histograms = {name: Histogram1D(th1=hist.GetPtr()) for name, hist in th_histograms.items()}

# plot Pt
fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
b_histograms["TruthTauPt"].plot(ax=ax, label="Truth Tau")
b_histograms["ImplicitMetPt"].plot(ax=ax, label="Implicit MET")
b_histograms["TruthMetPt"].plot(ax=ax, label="Truth MET")
ax.semilogy(True)
ax.semilogx(True)
ax.legend()
ax.set_xlabel("Pt [MeV]")
ax.set_ylabel("Entries")
plt.show()
