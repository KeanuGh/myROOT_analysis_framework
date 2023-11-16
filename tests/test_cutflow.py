from src.cutfile import Cut
from src.cutflow import RCutflow, CutflowItem


class TestCutflow:
    cutflow1 = RCutflow()
    cutflow1._cutflow = [
        CutflowItem(npass=100, eff=100, ceff=100, cut=Cut("1", "1", set(), set())),
        CutflowItem(npass=50, eff=50, ceff=50, cut=Cut("2", "2", set(), set())),
        CutflowItem(npass=10, eff=10, ceff=20, cut=Cut("3", "3", set(), set())),
    ]

    cutflow2 = RCutflow()
    cutflow2._cutflow = [
        CutflowItem(npass=200, eff=100, ceff=100, cut=Cut("1", "1", set(), set())),
        CutflowItem(npass=150, eff=75, ceff=75, cut=Cut("2", "2", set(), set())),
        CutflowItem(npass=10, eff=100 / 15, ceff=5, cut=Cut("3", "3", set(), set())),
    ]

    cutflow_merged = [
        CutflowItem(npass=300, eff=100, ceff=100, cut=Cut("1", "1", set(), set())),
        CutflowItem(npass=200, eff=200 / 3, ceff=200 / 3, cut=Cut("2", "2", set(), set())),
        CutflowItem(npass=20, eff=10, ceff=100 / 15, cut=Cut("3", "3", set(), set())),
    ]

    def test_names(self):
        assert self.cutflow1[0].cut.name == "1"
        assert self.cutflow1[-1].cut.name == "3"

    def test_merge_add(self):
        assert (self.cutflow1 + self.cutflow2)._cutflow == self.cutflow_merged

    def test_merge_iadd(self):
        self.cutflow1 += self.cutflow2

        assert self.cutflow1._cutflow == self.cutflow_merged
