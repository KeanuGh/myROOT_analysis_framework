from pathlib import Path
import pandas as pd

import ROOT
from utils.ROOT_utils import load_ROOT_settings, ROOT_TFile_mgr, get_dsod_values

if __name__ == "__main__":
    load_ROOT_settings()
    data_dir = Path("/data/DTA_outputs/2023-01-31/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_BFilter.e8351.MC16a.v1.2023-01-31_histograms.root/")
    ttrees = {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"}
    branches = {
        "rwCorr",
        "prwWeight",
        "Met_met",
        "TruthNeutrinoPt",
        "TruthTauPt",
        "TruthTauPhi",
        "TruthNeutrinoPhi",
        "TruthBosonM",
    }

    # per tree
    for ttree in ttrees:
        print(f"{ttree}...")
        paths = [str(file) for file in data_dir.glob("*.root")]
        Rdf = ROOT.RDataFrame(ttree, paths)

        # create columns
        ROOT.gInterpreter.Declare(
            f'std::map<int, float> dsid_map = get_sumw("{data_dir}", "{ttree}");'
        )
        Rdf = Rdf.Define("sumOfWeights", f"dsid_map[mcWeight]")

        print("to file...")
        Rdf.Snapshot("outTree", f"{ttree}.root", ["sumOfWeights", "mcWeight", "TauPt"])
