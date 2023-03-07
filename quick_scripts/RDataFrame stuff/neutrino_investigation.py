import glob

import ROOT
import numpy as np

from utils.ROOT_utils import init_rdataframe, ROOT_TFile_mgr

ROOT.EnableImplicitMT()
vector_col_name = "TruthNeutrino"
bins = np.geomspace(1, 5000, 50)
nbins = len(bins) - 1

files = "/data/DTA_outputs/2023-01-31/**/*.root"
trees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}

rdf = init_rdataframe("test", glob.glob(files), trees)

hists = dict()
for var in ("Pt", "Eta", "Phi"):
    rdf = rdf.Define(f"{vector_col_name}{var}1", f"(&{vector_col_name}{var})->at(0);")
    rdf = rdf.Define(f"{vector_col_name}{var}2", f"(&{vector_col_name}{var})->at(1);")
    rdf = rdf.Define(
        f"{vector_col_name}{var}3",
        f"((&{vector_col_name}{var})->size() > 2) ? (&{vector_col_name}{var})->at(2) : NAN;",
    )

    hists[f"{vector_col_name}{var}1"] = ROOT.TH1F(
        f"{vector_col_name}{var}1", f"{vector_col_name}{var}1", nbins, bins
    )
    hists[f"{vector_col_name}{var}2"] = ROOT.TH1F(
        f"{vector_col_name}{var}2", f"{vector_col_name}{var}2", nbins, bins
    )
    hists[f"{vector_col_name}{var}3"] = ROOT.TH1F(
        f"{vector_col_name}{var}3", f"{vector_col_name}{var}3", nbins, bins
    )

    hists[f"{vector_col_name}{var}1"] = rdf.Fill(
        hists[f"{vector_col_name}{var}1"], [f"{vector_col_name}{var}1"]
    )
    hists[f"{vector_col_name}{var}2"] = rdf.Fill(
        hists[f"{vector_col_name}{var}2"], [f"{vector_col_name}{var}2"]
    )
    hists[f"{vector_col_name}{var}3"] = rdf.Fill(
        hists[f"{vector_col_name}{var}3"], [f"{vector_col_name}{var}3"]
    )

print("Writing...")
with ROOT_TFile_mgr("truth_neutrinos.root", "RECREATE") as file:
    for col, hist in hists.items():
        file.WriteObject(hist.GetPtr(), col, "OVERWRITE")
