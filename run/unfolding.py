from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import ROOT

from datasetbuilder import LUMI_YEAR
from histogram import Histogram1D
from utils.ROOT_utils import load_ROOT_settings

load_ROOT_settings()

path_root = Path("/home/keanu/Uni_Stuff_Queen_Mary/Python_Projects/myFramework/outputs/")
YEAR = 2017
NOMINAL_NAME = "T_s1thv_NOMINAL"
var = "MTW"
var_truth = "TruthMTW"
wp = "loose"
lumi = LUMI_YEAR[2017]

# get response
with ROOT.TFile(str(path_root / "efficiency_and_acceptance/root/wtaunu.root")) as file:
    hist_reco = file[f"{NOMINAL_NAME}/{wp}_reco_tau"].Get(var)
    hist_reco.SetDirectory(0)
    hist_truth = file[f"{NOMINAL_NAME}/truth_tau"].Get(var_truth)
    hist_truth.SetDirectory(0)
    h_response = file[f"{NOMINAL_NAME}/{wp}_truth_reco_tau"].Get(f"{var}_{var_truth}")
    h_response.SetDirectory(0)
response = ROOT.RooUnfoldResponse(hist_reco, hist_truth, h_response)

with ROOT.TFile(
        str(path_root / "efficiency_and_acceptance/root/efficiency_and_acceptance.root")
) as file:
    eff_hist = file.Get(f"{wp}_{var}_efficiency")
    eff_hist.SetDirectory(0)
    acc_hist = file.Get(f"{wp}_{var}_acceptance")
    acc_hist.SetDirectory(0)
    fake_rate_hist = file.Get(f"{wp}_{var}_fake_rate")
    fake_rate_hist.SetDirectory(0)

hist_truth = Histogram1D(th1=hist_truth)
hist_reco = Histogram1D(th1=hist_reco)

# plot
plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()

hep.histplot(
    H=[hist_truth, hist_reco],
    ax=ax,
    label=["Truth", "Reco"],
    color=["k", "r"],
)
plt.xlim(hist_truth.bin_edges[0], hist_truth.bin_edges[-1])
plt.semilogx()
plt.xlabel(r"$m^W_\mathrm{T}$ [GeV]")
ax.set_ylabel("Events")
ax.legend()
fig.savefig("truth_reco.png", bbox_inches="tight")
plt.show()

# UNFOLD
# ===============================================================================
print(f"Performing unfolding for {var} in {wp}")


def unfold_bayes(h: ROOT.TH1, i: int) -> ROOT.TH1:
    return ROOT.RooUnfoldBayes(response, h, i).Hunfold()


# unfolded mc
reco_unfolded_bin = Histogram1D(th1=ROOT.RooUnfoldBinByBin(response, hist_reco.TH1).Hunfold())
reco_unfolded_svd0 = Histogram1D(th1=ROOT.RooUnfoldSvd(response, hist_reco.TH1, 0).Hunfold())
reco_unfolded_svd2 = Histogram1D(th1=ROOT.RooUnfoldSvd(response, hist_reco.TH1, 2).Hunfold())
reco_unfolded_svd4 = Histogram1D(th1=ROOT.RooUnfoldSvd(response, hist_reco.TH1, 4).Hunfold())
reco_unfolded_svd6 = Histogram1D(th1=ROOT.RooUnfoldSvd(response, hist_reco.TH1, 6).Hunfold())
reco_unfolded_svd8 = Histogram1D(th1=ROOT.RooUnfoldSvd(response, hist_reco.TH1, 8).Hunfold())
# reco_unfolded1 = Histogram1D(th1=unfold_bayes(hist_reco, 1))
reco_unfolded2 = Histogram1D(th1=unfold_bayes(hist_reco.TH1, 2))
reco_unfolded4 = Histogram1D(th1=unfold_bayes(hist_reco.TH1, 4))
reco_unfolded6 = Histogram1D(th1=unfold_bayes(hist_reco.TH1, 6))
# reco_unfolded8 = Histogram1D(th1=unfold_bayes(hist_reco, 8))

eff_hist = Histogram1D(th1=eff_hist)
acc_hist = Histogram1D(th1=acc_hist)
fake_rate_hist = Histogram1D(th1=fake_rate_hist)

plt.style.use(hep.style.ATLAS)
fig, ax = plt.subplots()

width = reco_unfolded_bin.bin_widths
hep.histplot(
    # [reco_unfolded_bin, reco_unfolded2, reco_unfolded4, reco_unfolded6, hist_truth],
    [
        reco_unfolded2 / (lumi * width),
        reco_unfolded4 / (lumi * width),
        reco_unfolded6 / (lumi * width),
        # reco_unfolded_svd0 * acc_hist / (lumi * width),
        # reco_unfolded_svd2 * acc_hist / (lumi * width),
        # reco_unfolded_svd4 * acc_hist / (lumi * width),
        # reco_unfolded_svd6 * acc_hist / (lumi * width),
        # reco_unfolded_svd8 * acc_hist / (lumi * width),
        hist_truth / (lumi * width),
    ],
    label=[
        # "0",
        "2",
        "4",
        "6",
        # "8",
        "truth",
    ],
    yerr=True,
)

# hep.histplot(
#     # [reco_unfolded_bin, reco_unfolded2, reco_unfolded4, reco_unfolded6, hist_truth],
#     [
#         (reco_unfolded2 * (1 - fake_rate_hist)) / (lumi * width),
#         reco_unfolded2 / (lumi * width),
#         hist_truth / (lumi * width),
#     ],
#     label=[
#         "2_fake_rate",
#         "2",
#         "truth",
#     ],
#     yerr=True,
# )
hep.histplot(
    reco_unfolded_bin / (lumi * width),
    label="bin-by-bin",
    yerr=True,
    color="k",
    histtype="errorbar",
    binticks=True,
    ax=ax,
    capsize=5,
    xerr=width / 2,
)
plt.xlim(hist_truth.bin_edges[0], hist_truth.bin_edges[-1])
plt.xlabel(r"$m^W_\mathrm{T}$ [GeV]")
plt.semilogx()
plt.legend()
fig.savefig("unfolding_tests.png", bbox_inches="tight")
plt.show()

hep.histplot(fake_rate_hist)
plt.show()
