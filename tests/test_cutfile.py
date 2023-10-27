import pytest

from src.cutfile import Cut, Cutfile


class TestCut:
    def test_str(self):
        test_cut = Cut("met phi", "met_phi > 50", {"met_phi"}, {"truth"}, is_reco=False)
        assert str(test_cut) == "met phi: met_phi > 50"


class TestCutfile:
    @pytest.fixture(scope="session")
    def cutfile(self, tmp_cutfile):
        yield Cutfile(tmp_cutfile, default_tree="truth")

    def test_cuts(self, cutfile):
        expected_cuts = [
            Cut(
                "tight muon",
                "testvartruth == 1",
                {"testvartruth"},
                {"truth"},
                is_reco=False,
            ),
            Cut(
                "lepton pt",
                "MC_WZneutrino_pt_born > 25 and MC_WZmu_el_pt_born > 25",
                {"MC_WZmu_el_pt_born", "MC_WZneutrino_pt_born"},
                {"truth"},
                False,
            ),
            Cut(
                "muon eta",
                "MC_WZmu_el_eta_bare.abs() < 25",
                {"MC_WZmu_el_eta_bare"},
                {"truth"},
                is_reco=False,
            ),
            Cut(
                "exclude crack region",
                "MC_WZmu_el_eta_bare.abs() < 1.37 or MC_WZmu_el_eta_bare.abs() > 1.57",
                {"MC_WZmu_el_eta_bare"},
                {"truth"},
                is_reco=False,
            ),
            Cut("met phi", "met_phi > 200", {"met_phi"}, {"reco"}, is_reco=True),
        ]

        assert cutfile.cuts == expected_cuts

    def test_multiple_default_trees(self, tmp_cutfile):
        """Look for any unlabeled variables in all tree dictionaries"""
        cutfile = Cutfile(tmp_cutfile, default_tree={"truth", "truth2", "truth3"})
        expected_treedict = {
            "truth": {
                "testvartruth",
                "MC_WZmu_el_eta_bare",
                "MC_WZneutrino_pt_born",
                "MC_WZmu_el_pt_born",
                "PDFinfo_Q",
                "MC_WZ_dilep_pt_born",
            },
            "reco": {
                "mu_d0sig",
                "met_phi",
                "jet_e",
                # varibles to calculate mu_mt
                "met_phi",
                "met_met",
                "mu_phi",
                "mu_pt",
            },
            "truth2": {
                "testvartruth",
                "MC_WZmu_el_eta_bare",
                "MC_WZneutrino_pt_born",
                "MC_WZmu_el_pt_born",
                "PDFinfo_Q",
            },
            "truth3": {
                "testvartruth",
                "MC_WZmu_el_eta_bare",
                "MC_WZneutrino_pt_born",
                "MC_WZmu_el_pt_born",
                "PDFinfo_Q",
            },
        }
        assert cutfile.tree_dict == expected_treedict

    def test_cutfile_output(self, cutfile):
        expected_treedict = {
            "truth": {
                "testvartruth",
                "MC_WZmu_el_eta_bare",
                "MC_WZneutrino_pt_born",
                "MC_WZmu_el_pt_born",
                "PDFinfo_Q",
                "MC_WZ_dilep_pt_born",
            },
            "reco": {
                "mu_d0sig",
                "met_phi",
                "jet_e",
                # varibles to calculate mu_mt
                "met_phi",
                "met_met",
                "mu_phi",
                "mu_pt",
            },
        }
        expected_vars_to_cut = {"mu_mt_reco"}

        assert cutfile.tree_dict == expected_treedict
        assert cutfile.vars_to_calc == expected_vars_to_cut

    def test_all_vars(self, cutfile):
        expected_output = {
            "testvartruth",
            "PDFinfo_Q",
            "MC_WZneutrino_pt_born",
            "MC_WZmu_el_pt_born",
            "MC_WZmu_el_eta_bare",
            "met_phi",
            "MC_WZ_dilep_pt_born",
            "mu_d0sig",
            "mu_mt_reco",
            "jet_e",
            "met_met",
            "met_phi",
            "mu_pt",
            "mu_phi",
        }
        assert cutfile.all_vars == expected_output

    def test_reco_cut_in_wrong_place(self, tmp_path):
        wrong_cutfile = """
[CUTS]
# Name \t cutstr \t tree
tight muon\t testvartruth == 1
lepton pt \t MC_WZneutrino_pt_born > 25 and MC_WZmu_el_pt_born > 25
met phi \t met_phi > 200 \t reco
muon eta \t MC_WZmu_el_eta_bare.abs() < 25
exclude crack region \t MC_WZmu_el_eta_bare.abs() < 1.37 or MC_WZmu_el_eta_bare.abs() > 1.57\ttruth

[OUTPUTS]
# truth
PDFinfo_Q \t truth
MC_WZ_dilep_pt_born \t truth

# reco
mu_d0sig \t reco
mu_mt_reco \t reco
jet_e \t reco
"""
        tmp_filepath = tmp_path / "wrong_cutfile"
        with open(tmp_filepath, "w") as f:
            f.write(wrong_cutfile)

        with pytest.raises(ValueError) as e:
            _ = Cutfile(str(tmp_filepath), default_tree="truth")
        assert (
            str(e.value) == "Truth cut after reco cut!\n\tmuon eta: MC_WZmu_el_eta_bare.abs() < 25"
        )
