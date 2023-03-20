import pytest


@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory):
    yield tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def tmp_cutfile(tmp_path_factory):
    """Gen temporary cutfile"""
    test_input = """
[CUTS]
# truth
# Name \t cutstr \t tree
tight muon\t testvartruth == 1
lepton pt \t MC_WZneutrino_pt_born > 25 and MC_WZmu_el_pt_born > 25
muon eta \t MC_WZmu_el_eta_bare.abs() < 25
exclude crack region \t MC_WZmu_el_eta_bare.abs() < 1.37 or MC_WZmu_el_eta_bare.abs() > 1.57\ttruth

# reco cut
met phi \t met_phi > 200 \t reco

[OUTPUTS]
# truth
PDFinfo_Q
MC_WZ_dilep_pt_born \t truth

# reco
mu_d0sig \t reco
mu_mt_reco \t reco
jet_e \t reco
"""

    tmp_cutfile_path = tmp_path_factory.mktemp("options") / "test_cutfile.txt"
    with open(tmp_cutfile_path, "w") as f:
        f.write(test_input)
    yield str(tmp_cutfile_path)
