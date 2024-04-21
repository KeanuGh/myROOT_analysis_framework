import ROOT

from src.cutting import Cut, Cutflow, CutflowItem, FilterNode, FilterTree

cut1 = Cut("cut1", "TauPt == 1", set(), set())
cut2 = Cut("cut2", "TauPt == 0", set(), set())
cut3 = Cut("cut3", "TauPt == 3", set(), set())
cut4 = Cut("cut4", "TauPt == 90", set(), set())
cut5 = Cut("cut5", "TauPt == 234", set(), set())


class TestCutflow:
    cutflow1 = Cutflow()
    cutflow1._cutflow = [
        CutflowItem(npass=100, eff=100, ceff=100, cut=cut1),
        CutflowItem(npass=50, eff=50, ceff=50, cut=cut2),
        CutflowItem(npass=10, eff=10, ceff=20, cut=cut3),
    ]

    cutflow2 = Cutflow()
    cutflow2._cutflow = [
        CutflowItem(npass=200, eff=100, ceff=100, cut=cut1),
        CutflowItem(npass=150, eff=75, ceff=75, cut=cut2),
        CutflowItem(npass=10, eff=100 / 15, ceff=5, cut=cut3),
    ]

    cutflow_merged = [
        CutflowItem(npass=300, eff=100, ceff=100, cut=cut1),
        CutflowItem(npass=200, eff=200 / 3, ceff=200 / 3, cut=cut2),
        CutflowItem(npass=20, eff=10, ceff=100 / 15, cut=cut3),
    ]

    def test_names(self):
        assert self.cutflow1[0].cut.name == "cut1"
        assert self.cutflow1[-1].cut.name == "cut3"

    def test_merge_add(self):
        assert (self.cutflow1 + self.cutflow2)._cutflow == self.cutflow_merged

    def test_merge_iadd(self):
        self.cutflow1 += self.cutflow2

        assert self.cutflow1._cutflow == self.cutflow_merged


class TestFilterTree:
    rdf = ROOT.RDataFrame(100)
    rdf = rdf.Define("TauPt", "3")

    def test_parents(self):
        node1 = FilterNode(self.rdf, cut1)
        node2 = node1.create_child(cut2)

        assert node2.get_parents() == [node1]
        assert node2.get_chain() == [node1, node2]

    def test_branch(self):
        selections = {
            "selection1": [cut1, cut2, cut3],
            "selection2": [cut1, cut4, cut5],
        }
        tree = FilterTree(self.rdf)
        tree.add_path(selections["selection1"], "selection1")
        tree.add_path(selections["selection2"], "selection2")

        tree_string = tree.tree_string_repr()
        assert tree_string == "root\n\tcut1\n\t\tcut2\n\t\t\tcut3\n\t\tcut4\n\t\t\tcut5"

    def test_graph(self):
        selections = {
            "selection1": [cut1, cut2, cut3],
            "selection2": [cut1, cut4, cut5],
        }
        output_tree = FilterTree(self.rdf)
        output_tree.generate_tree(selections)

        tree_string = output_tree.tree_string_repr()
        assert tree_string == "root\n\tcut1\n\t\tcut2\n\t\t\tcut3\n\t\tcut4\n\t\t\tcut5"
