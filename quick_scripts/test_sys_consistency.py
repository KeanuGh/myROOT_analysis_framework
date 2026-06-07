import ROOT

ROOT.EnableImplicitMT()  # enable multithreading

filepath = "/home/keanu/Uni_Stuff_Queen_Mary/Python_Projects/myFramework/outputs/analysis_full/root/wtaunu.root"
check_sys_up = "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1up"
check_sys_down = "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt__1down"
nominal = "T_s1thv_NOMINAL"
check_sel = "SR_passID"

file2 = "/data/DTA_outputs/2024-08-28/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_BFilter.e8351.MC16d.v1.2024-08-28_histograms.root/user.kghorban.40997756._000001.histograms.root"
df_sys_up = ROOT.RDataFrame(check_sys_up, file2)
df_sys_down = ROOT.RDataFrame(check_sys_down, file2)
df_nom = ROOT.RDataFrame(nominal, file2)
