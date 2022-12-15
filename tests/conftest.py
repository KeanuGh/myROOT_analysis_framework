import numpy as np
import pytest
import uproot


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


@pytest.fixture(scope="session")
def tmp_root_datafile(tmp_path_factory):
    """Generate test root file to read in"""
    datapath = tmp_path_factory.mktemp("data") / "test_data.root"
    with uproot.recreate(datapath) as test_file:
        test_file["tree1"] = {
            "testvar1": np.arange(1000),
            "testvar2": np.arange(1000) * 1.1,
            "testvar3": np.arange(1000) * 3,
            "weight_mc": np.append(np.ones(990), -1 * np.ones(10)),
            # FIXME
            "mcChannelNumber": np.append(np.ones(500), np.full(500, 2)),
            "eventNumber": np.arange(1000),
            "weight_pileup": np.ones(1000),
        }
        test_file["tree2"] = {"testvar4": np.arange(1500) * -1, "eventNumber": np.arange(1500)}
        test_file["sumWeights"] = {
            "totalEventsWeighted": np.array([500, 480]),
            "dsid": np.array([1, 2]),
        }
    yield str(datapath)


@pytest.fixture(scope="session")
def tmp_root_datafiles(tmp_path_factory):
    """Generate 3 test root file to read in"""
    data_prefix = "test_data"
    tmpdir = tmp_path_factory.mktemp("data")
    a_n = 0
    for i in [1, 2, 3]:
        datapath = tmpdir / f"{data_prefix}{i}.root"
        with uproot.recreate(datapath) as test_file:
            test_file["tree1"] = {
                "testvar1": np.arange(1000 * i),
                "testvar2": np.arange(1000 * i) * 1.1 * i,
                "testvar3": np.arange(1000 * i) * i,
                "weight_mc": np.append(np.ones(990 * i), -1 * np.ones(10 * i)),
                "mcChannelNumber": np.full(1000 * i, i),
                "eventNumber": np.arange(
                    1000 * a_n, 1000 * (a_n := a_n + i)
                ),  # a_{n+1}=a_n+n+1, a_0=0
                "weight_pileup": np.ones(1000 * i),
            }
            test_file["tree2"] = {
                "testvar4": np.arange(1500 * i) * -1,
                "eventNumber": np.arange(1500 * i),
            }
            test_file["sumWeights"] = {"totalEventsWeighted": np.array([980 * i]), "dsid": [i]}
    yield str(tmpdir / f"{data_prefix}*.root")


@pytest.fixture(scope="function")
def tmp_root_datafile_duplicate_events(tmp_path_factory):
    """Generate test root file with duplicate events to read in"""
    datapath = tmp_path_factory.mktemp("data") / "test_data_duplicated_events.root"
    with uproot.recreate(datapath) as test_file:
        test_file["tree1"] = {
            "testvar1": np.arange(1000),
            "testvar2": np.arange(1000) * 1.1,
            "testvar3": np.arange(1000) * 3,
            "weight_mc": np.append(np.ones(990), -1 * np.ones(10)),
            "mcChannelNumber": np.append(np.ones(500), np.full(500, 2)),
            "eventNumber": np.append(np.array([12, 112, 500]), np.arange(3, 1000)),
            "weight_pileup": np.ones(1000),
        }
        test_file["tree2"] = {"testvar4": np.arange(1500) * -1, "eventNumber": np.arange(1500)}
        test_file["sumWeights"] = {"totalEventsWeighted": np.array([980]), "dsid": np.array([1])}
    yield str(datapath)


@pytest.fixture(scope="function")
def tmp_root_datafile_missing_events(tmp_path_factory):
    """Generate test root file with missing events to read in"""
    datapath = tmp_path_factory.mktemp("data") / "test_data_duplicated_events.root"
    with uproot.recreate(datapath) as test_file:
        test_file["tree1"] = {
            "testvar1": np.arange(1000),
            "testvar2": np.arange(1000) * 1.1,
            "testvar3": np.arange(1000) * 3,
            "weight_mc": np.append(np.ones(990), -1 * np.ones(10)),
            "mcChannelNumber": np.append(np.ones(500), np.full(500, 2)),
            "eventNumber": np.arange(1000),
            "weight_pileup": np.ones(1000),
        }
        test_file["tree2"] = {
            "testvar4": np.arange(5, 1500) * -1,
            "eventNumber": np.arange(5, 1500),
        }
        test_file["sumWeights"] = {"totalEventsWeighted": np.array([980]), "dsid": np.array([1])}
    yield str(datapath)
