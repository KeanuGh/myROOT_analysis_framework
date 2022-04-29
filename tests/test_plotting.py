import pytest

from utils import plotting_utils


class TestPlottingUtils:
    def test_get_axis_labels_str(self):
        output = plotting_utils.get_axis_labels('TauPt')
        assert output == ('Tau $p_{T}$ [GeV]', 'Entries')

    @pytest.mark.filterwarnings('ignore:UserWarning')
    def test_get_axis_labels_list(self):
        output = plotting_utils.get_axis_labels(['TauPt', 'met_met'])
        assert output == ('Tau $p_{T}$ [GeV]', 'Entries')
