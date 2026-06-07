import os
from functools import reduce

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.StartGUIThread = False
ROOT.TH1.AddDirectory(False)
ROOT.TH1.SetDefaultSumw2()
ROOT.EnableImplicitMT()

out_file = "snapshot_test.root"

if os.path.exists(out_file):
    os.remove(out_file)

dfs = {}
print("building dataframes....")
for i in range(1, 4):
    dfs[f"df{i}"] = ROOT.RDataFrame(100)
    dfs[f"df{i}"] = dfs[f"df{i}"].Define("x", "gRandom->Rndm()")

opts = ROOT.RDF.RSnapshotOptions()
opts.fMode = "UPDATE"
opts.fOverwriteIfExists = True
print("snapshotting....")
for dataset in dfs:
    dfs[dataset].Snapshot(f"{dataset}/{dataset}", out_file, ["x"], opts)

# import and fill
dfs2 = {}
print("import and fill...")
h = ROOT.TH1D("test", "test", 10, 0, 1)
ptrs = []
for i in range(1, 4):
    dfs2[f"df{i}"] = ROOT.RDataFrame(f"df{i}/df{i}", out_file)
    ptrs.append(dfs2[f"df{i}"].Fill(h, ["x"]))

hs = [ptr.GetValue() for ptr in ptrs]
h = reduce(lambda x, y: x + y, hs)

c = ROOT.TCanvas()
h.Draw()
c.SaveAs("snapshotting.png")
