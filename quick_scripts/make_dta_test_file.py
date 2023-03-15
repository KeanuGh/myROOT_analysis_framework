import ROOT

out_file = "wtaunu_h_cvbv_1000.root"

rdf = ROOT.RDataFrame(
    "T_s1tev_NOMINAL",
    "/data/DTA_outputs/2023-01-31/user.kghorban.Sh_2211_Wtaunu_L_maxHTpTV2_CVetoBVeto.e8351.MC16a.v1.2023-01-31_histograms.root/user.kghorban.32164992._000001.histograms.root",
)
rdf = rdf.Range(5000)

output_columns = [
    str(col)
    for col in rdf.GetColumnNames()
    if col
    not in (
        "TauWP",
        "TauMatchedTriggers",
        "MuonMatchedTriggers",
        "EleMatchedTriggers",
        "PassedTriggers",
    )
]  # problematic and unnecessary
print("printing file...")
rdf.Snapshot("T_s1thv_NOMINAL", out_file, output_columns)

# make sum of weights histogram
sum_of_weights_hist = ROOT.TH1F("sumOfWeights", "700346", 7, -0.5, 6.5)
sum_of_weights_hist.GetXaxis().SetBinLabel(1, "DxAOD")
sum_of_weights_hist.GetXaxis().SetBinLabel(2, "DxAOD squared")
sum_of_weights_hist.GetXaxis().SetBinLabel(3, "DxAOD events")
sum_of_weights_hist.GetXaxis().SetBinLabel(4, "xAOD")
sum_of_weights_hist.GetXaxis().SetBinLabel(5, "xAOD squared")
sum_of_weights_hist.GetXaxis().SetBinLabel(6, "xAOD events")
sum_of_weights_hist.SetBinContent(1, 5.03982e14)
sum_of_weights_hist.SetBinContent(2, 4.30309e23)
sum_of_weights_hist.SetBinContent(3, 3.095e06)
sum_of_weights_hist.SetBinContent(4, 5.03982e14)
sum_of_weights_hist.SetBinContent(5, 4.30309e23)
sum_of_weights_hist.SetBinContent(6, 3.095e06)
sum_of_weights_hist.SetBinError(1, 5.80411e13)
sum_of_weights_hist.SetBinError(2, 4.96243e22)
sum_of_weights_hist.SetBinError(3, 356383)
sum_of_weights_hist.SetBinError(4, 5.80411e13)
sum_of_weights_hist.SetBinError(5, 4.96243e22)
sum_of_weights_hist.SetBinError(6, 356383)
sum_of_weights_hist.SetEntries(462)

file = ROOT.TFile(out_file, "UPDATE")
file.WriteObject(sum_of_weights_hist, "sumOfWeights")
file.Close()
