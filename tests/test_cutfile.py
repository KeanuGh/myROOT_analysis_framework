import pytest

from src.cutting import Cut


class TestCut:
    def test_str(self):
        test_cut = Cut("tau pt", "TauPt > 50")
        assert str(test_cut) == "tau pt: TauPt > 50"

    def test_included_variables(self):
        test_cut = Cut("truth mass", "TruthMTW > 350 && TruthTauPt > 170")
        assert test_cut.included_variables == {"TruthMTW", "TruthTauPt"}
        assert not test_cut.is_reco

    def test_reco_cut_detection(self):
        test_cut = Cut("reco met", "MET_met > 170")
        assert test_cut.included_variables == {"MET_met"}
        assert test_cut.is_reco

    def test_unknown_variable_cut_raises(self):
        with pytest.raises(ValueError, match="No known variable"):
            Cut("bad", "NotAFrameworkVariable > 0")
