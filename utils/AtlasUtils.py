import os

import ROOT


def set_atlas_style():
    """Set ATLAS style in root plots"""
    ROOT.gROOT.LoadMacro(os.path.dirname(os.path.abspath(__file__)) + "/AtlasStyle/AtlasStyle.C")
    ROOT.SetAtlasStyle()
