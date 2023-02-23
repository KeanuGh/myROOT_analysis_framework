import glob

import ROOT

from utils.ROOT_utils import init_rdataframe, ROOT_TFile_mgr

ROOT.EnableImplicitMT()

files = "/data/DTA_outputs/2023-01-31/**/*.root"
trees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}

rdf = init_rdataframe("test", glob.glob(files), trees)

vector_columns = [
    str(col) for col in rdf.GetColumnNames() if "ROOT::VecOps::RVec" in rdf.GetColumnType(col)
]
print("Vector columns: \n\t{}".format("\n\t".join(vector_columns)))

hists = dict()
n_bins = 5
for col in vector_columns:
    # define new columns containing number of elements in vector columns
    rdf = rdf.Define(f"{col}_nelements", f"{col}.size();")
    hists[col] = ROOT.TH1F(col, f"number of elements in {col} vector", n_bins, 0, n_bins)
    hists[col] = rdf.Fill(hists[col], [f"{col}_nelements"])

print("Writing...")
with ROOT_TFile_mgr("element_hists.root", "RECREATE") as file:
    for col, hist in hists.items():
        file.WriteObject(hist.GetPtr(), col, "OVERWRITE")
