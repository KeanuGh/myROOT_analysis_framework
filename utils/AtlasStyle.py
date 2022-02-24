import os

import ROOT


def set_atlas_style():
    ROOT.gROOT.LoadMacro(os.path.dirname(os.path.abspath(__file__)) + "/AtlasStyle/AtlasStyle.C")
    ROOT.SetAtlasStyle()
