import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ROOT
from utils import PMG_tool
from utils.ROOT_utils import ROOT_TFile_mgr, get_dsid_values
from histogram import Histogram1D


if __name__ == "__main__":
    data_dir = "/data/DTA_outputs/2023-01-31/**/*.root"
    ttrees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}
    branches = [
        "Met_met",
        "TruthNeutrinoPt",
        "TruthTauPt",
        "TruthTauPhi",
        "TruthNeutrinoPhi",
        "TruthBosonM",
    ]
    mass_bins = np.array(
        [
            130,
            140.3921,
            151.6149,
            163.7349,
            176.8237,
            190.9588,
            206.2239,
            222.7093,
            240.5125,
            259.7389,
            280.5022,
            302.9253,
            327.1409,
            353.2922,
            381.5341,
            412.0336,
            444.9712,
            480.5419,
            518.956,
            560.4409,
            605.242,
            653.6246,
            705.8748,
            762.3018,
            823.2396,
            889.0486,
            960.1184,
            1036.869,
            1119.756,
            1209.268,
            1305.936,
            1410.332,
            1523.072,
            1644.825,
            1776.311,
            1918.308,
            2071.656,
            2237.263,
            2416.107,
            2609.249,
            2817.83,
            3043.085,
            3286.347,
            3549.055,
            3832.763,
            4139.151,
            4470.031,
            4827.361,
            5213.257,
        ]
    )
    dsid_metadata = get_dsid_values(data_dir, "T_s1thv_NOMINAL")

    histo_args = ("taupt", "Tau $p_T$", len(mass_bins) - 1, mass_bins)
    h_taupt = ROOT.TH1F(*histo_args)
    print(h_taupt)

    snapshot_options = ROOT.RDF.RSnapshotOptions()
    snapshot_options.fMode = "Update"

    for dsid in dsid_metadata.index:
        sumw = dsid_metadata.loc[dsid, "sumOfWeights"]
        xsec = dsid_metadata.loc[dsid, "cross-section"]
        pmgf = dsid_metadata.loc[dsid, "PMG_factor"]

        for ttree in ttrees:
            paths = [str(file) for file in glob.glob(data_dir)]
            Rdf = ROOT.RDataFrame(ttree, paths)

            # create columns
            Rdf = Rdf.Define(
                "truth_weight", f"(mcWeight * rwCorr * prwWeight * {pmgf} * {xsec}) / {sumw}"
            )
            Rdf = Rdf.Redefine("TauPt", f"TauPt / 1000")

            # fill
            print(f"filling histogram for {ttree} tree in {PMG_tool.get_physics_short(dsid)}...")
            h_tree = Rdf.Histo1D(histo_args, "TauPt", "truth_weight").GetPtr()
            h_taupt.Add(h_tree)

            # to file
            Rdf.Snapshot("outTree", "out.root", ["TauPt", "truth_weight"], snapshot_options)

    print(f"{h_taupt.GetEntries()=}")

    # plot ROOT
    h_taupt.SetStats(True)
    c = ROOT.TCanvas()
    h_taupt.Draw("E Hist")
    c.Print("TauPt.png")

    print(f"{h_taupt.GetEntries()=}")

    # into boost
    print("convert to boost...")
    hist = Histogram1D(th1=h_taupt)
    hist.plot()
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel("TauPt [GeV]")
    plt.ylabel("Entries")
    plt.show()

    print(f"{h_taupt.GetEntries()=}")

    # trying again with full file
    Rdf = ROOT.RDataFrame("outTree", "out.root")
    h = Rdf.Histo1D(histo_args, "TauPt", "truth_weight").GetPtr()
    h.SetStats(True)
    c = ROOT.TCanvas()
    h.Draw("E Hist")
    c.Print("TauPt_full.png")
