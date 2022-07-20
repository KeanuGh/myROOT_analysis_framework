# from src.histogram import Histogram1D
import glob

# import pandas as pd
import ROOT
import numpy as np

# import matplotlib.pyplot as plt

ROOT.TH1.SetDefaultSumw2()
ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(False)

verbosity = ROOT.Experimental.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo)

FILEPATH_PEAK = '/data/analysistop_out/mc16a/wmintaunu/*.root'
FILEPATH_SLICE = '/data/analysistop_out/mc16a/wmintaunu_*/*.root'
TREENAME = 'truth'
BRANCH = 'MC_WZ_dilep_m_born'
LUMI_SAMPLE = 32.9881 + 3.21956

# ROOT.gSystem.Load(f'../utils/rootfuncs.h')
# ROOT.gInterpreter.Declare(f'#include "../utils/rootfuncs.h"')

# get sum of weights
all_files = glob.glob(FILEPATH_PEAK) + glob.glob(FILEPATH_SLICE)
sumw_Rdf = ROOT.RDataFrame('sumWeights', all_files)
sumw = sumw_Rdf.Sum("totalEventsWeighted").GetValue()


def build_rdataframe(filepath: str, tree: str) -> ROOT.RDataFrame:
    """Return Rdataframe with calculated weights"""
    # get full tree
    files = glob.glob(filepath)
    Rdf = ROOT.RDataFrame(tree, files)

    # # define weights
    Rdf = Rdf.Redefine(BRANCH, f"{BRANCH} / 1000") \
             .Define("weight", f"weight_mc * {LUMI_SAMPLE} / {sumw}") \
             .Define("truth_weight", "weight * KFactor_weight_truth * weight_pileup") \

    return Rdf  # bug in ROOT requres chain to be returned with the dataframe


Rdf_slice = build_rdataframe(FILEPATH_PEAK, TREENAME)
Rdf_peak = build_rdataframe(FILEPATH_SLICE, TREENAME)

# filter peak
Rdf_peak = Rdf_peak.Filter("MC_WZ_dilep_m_born < 120")

# define and fill histogram
bins = np.array(
    [130, 140.3921, 151.6149, 163.7349, 176.8237, 190.9588, 206.2239, 222.7093, 240.5125, 259.7389, 280.5022,
     302.9253, 327.1409, 353.2922, 381.5341, 412.0336, 444.9712, 480.5419, 518.956, 560.4409, 605.242, 653.6246,
     705.8748, 762.3018, 823.2396, 889.0486, 960.1184, 1036.869, 1119.756, 1209.268, 1305.936, 1410.332, 1523.072,
     1644.825, 1776.311, 1918.308, 2071.656, 2237.263, 2416.107, 2609.249, 2817.83, 3043.085, 3286.347, 3549.055,
     3832.763, 4139.151, 4470.031, 4827.361, 5213.257])
h = ROOT.TH1F("Dilep m", "Dilep m", len(bins) - 1, bins)

h_filled = Rdf_slice.Fill(h, [BRANCH, 'truth_weight'])
h_filled = Rdf_peak.Fill(h_filled.GetPtr(), [BRANCH, 'truth_weight'])

c = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(1111)
c.SetLogx()
c.SetLogy()
ROOT.gPad.Update()

h_filled.Draw()

# print("With pandas...")
# df_slice = pd.DataFrame(Rdf_slice.AsNumpy(columns=['MC_WZ_dilep_m_born', 'truth_weight']))
# df_peak = pd.DataFrame(Rdf_peak.AsNumpy(columns=['MC_WZ_dilep_m_born', 'truth_weight']))
#
# df = pd.concat([df_slice, df_peak])
#
# fig, ax = plt.subplots()
# h = Histogram1D(df['MC_WZ_dilep_m_born'], bins, df['truth_weight'])
# h.plot(ax=ax)
# ax.semilogx()
# ax.semilogy()
# plt.show()
