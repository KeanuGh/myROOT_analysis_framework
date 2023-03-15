from utils import plotting_tools


class TestPlottingUtils:
    def test_get_axis_labels_str(self):
        output = plotting_tools.get_axis_labels("TauPt")
        assert output == ("Tau $p_{T}$ [GeV]", "Entries")
