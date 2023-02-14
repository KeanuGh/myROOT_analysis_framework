import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ROOT
from utils import PMG_tool
from utils.ROOT_utils import ROOT_TFile_mgr, get_dsid_values
from histogram import Histogram1D
import time


if __name__ == "__main__":
    data_dir = "/data/DTA_outputs/2023-01-31/**/*.root"
    trees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}
    cols_out = ["TauPt", "eventNumber", "truth_weight", "mcChannel"]
    dsid_metadata = get_dsid_values(data_dir, "T_s1thv_NOMINAL")

    # create c++ map for dataset ID metadatas
    ROOT.gInterpreter.Declare(
        f"""
            std::map<int, float> dsid_sumw{{{','.join(f'{{{t.Index}, {t.sumOfWeights}}}' for t in dsid_metadata.itertuples())}}};
            std::map<int, float> dsid_xsec{{{','.join(f'{{{t.Index}, {t.cross_section}}}' for t in dsid_metadata.itertuples())}}};
            std::map<int, float> dsid_pmgf{{{','.join(f'{{{t.Index}, {t.PMG_factor}}}' for t in dsid_metadata.itertuples())}}};
        """
    )

    # create TChain in c++ in order for it not to be garbage collected by python
    paths = [str(file) for file in glob.glob(data_dir)]
    ROOT.gInterpreter.Declare(
        f"""
            TChain chain;
            void fill_chain() {{
                std::vector<std::string> paths = {{\"{'","'.join(paths)}\"}};
                std::vector<std::string> trees = {{\"{'","'.join(trees)}\"}};
                for (const auto& path : paths) {{
                    for (const auto& tree : trees) {{
                        chain.Add((path + "?#" + tree).c_str());
                    }}
                }}
            }}
        """
    )
    ROOT.fill_chain()

    # create RDataFrame
    Rdf = ROOT.RDataFrame(ROOT.chain)

    # create columns
    Rdf = Rdf.Define(
        "truth_weight",
        f"(mcWeight * rwCorr * prwWeight * dsid_xsec[mcChannel] * dsid_pmgf[mcChannel]) / dsid_sumw[mcChannel];",
    )
    Rdf = Rdf.Redefine("TauPt", f"TauPt / 1000")
    print(f"{Rdf.GetNRuns()=}")

    # plot ROOT
    histo_args = ("taupt", "Tau $p_T$", 30, 1, 5000)
    h_taupt_vector = Rdf.Histo1D(histo_args, "TauPt", "truth_weight").GetPtr()
    h_taupt_vector.SetStats(True)
    c = ROOT.TCanvas()
    h_taupt_vector.Draw("E Hist")
    c.Print("TauPt_vector.png")

    print(f"{h_taupt_vector.GetEntries()=}")
    print(f"{Rdf.GetNRuns()=}")

    # routine to separate vector branches into separate variables
    badcols = set()  # save old vector column names to avoid extracting them later
    for col_name in cols_out:
        # unravel vector-type columns
        col_type = Rdf.GetColumnType(col_name)
        if "ROOT::VecOps::RVec" in col_type:
            # skip non-numeric vector types
            if col_type == "ROOT::VecOps::RVec<string>":
                print(f"Skipping string vector column {col_name}")
                badcols.add(col_name)

            elif "jet" in str(col_name).lower():
                # create three new columns for each possible jet
                for i in range(3):
                    Rdf = Rdf.Define(f"{col_name}{i + 1}", f"getVecVal({col_name},{i})")
                badcols.add(col_name)

            else:
                Rdf = Rdf.Redefine(col_name, f"getVecVal({col_name},0)")
    # plot ROOT
    h_taupt = Rdf.Histo1D(histo_args, "TauPt", "truth_weight").GetPtr()
    h_taupt.SetStats(True)
    c = ROOT.TCanvas()
    h_taupt.Draw("E Hist")
    c.Print("TauPt.png")

    print(f"{h_taupt.GetEntries()=}")
    print(f"{Rdf.GetNRuns()=}")

    # # print to file
    # print("to root file...")
    # t = time.time()
    # snapshot_options = ROOT.RDF.RSnapshotOptions()
    # snapshot_options.fMode = "Recreate"
    # snapshot_options.fOverwriteIfExists = True
    # Rdf.Snapshot("outTree", "out.root", cols_out, snapshot_options)
    # print(f"time: {time.time() - t:.3g}s")
    #
    # # to pandas
    # print("to pandas...")
    # t = time.time()
    # df = pd.DataFrame(Rdf.AsNumpy(columns=cols_out))
    # print("Setting DSID/eventNumber as index...")
    # df.set_index(["mcChannel", "eventNumber"], inplace=True)
    # df.index.names = ["DSID", "eventNumber"]
    # print(f"time: {time.time() - t:.3g}s")
    #
    # print("to pickle...")
    # t = time.time()
    # df.to_pickle("out.pkl")
    # print(f"time: {time.time() - t:.3g}s")
