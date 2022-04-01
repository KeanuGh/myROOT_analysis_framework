import pytest
from src.cutfile import *

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
PDFinfo_Q \t truth
MC_WZ_dilep_pt_born \t truth

# reco
mu_d0sig \t reco
mu_mt_reco \t reco
jet_e \t reco
"""


@pytest.fixture(scope='session')
def tmp_cutfile(tmp_path_factory):
    """Gen temporary cutfile"""
    tmp_cutfile_path = tmp_path_factory.mktemp('options') / 'test_cutfile.txt'
    with open(tmp_cutfile_path, 'w') as f:
        f.write(test_input)
    yield str(tmp_cutfile_path)


class TestCut:
    def test_str(self):
        test_cut = Cut('met phi', 'met_phi > 50', 'truth')
        assert str(test_cut) == 'met phi: met_phi > 50'


class TestCutfile:
    @pytest.fixture(scope='session')
    def cutfile(self, tmp_cutfile):
        cutfile = Cutfile(tmp_cutfile, default_tree='truth')
        yield cutfile

    def test_cuts(self, cutfile):
        expected_cuts = OrderedDict()
        expected_cuts['tight muon'] = Cut('tight muon', 'testvartruth == 1', 'truth')
        expected_cuts['lepton pt'] = Cut('lepton pt', 'MC_WZneutrino_pt_born > 25 and MC_WZmu_el_pt_born > 25', 'truth')
        expected_cuts['muon eta'] = Cut('muon eta', 'MC_WZmu_el_eta_bare.abs() < 25', 'truth')
        expected_cuts['exclude crack region'] = Cut('exclude crack region', 'MC_WZmu_el_eta_bare.abs() < 1.37 or MC_WZmu_el_eta_bare.abs() > 1.57', 'truth')
        expected_cuts['met phi'] = Cut('met phi', 'met_phi > 200', 'reco')

        assert cutfile.cuts == expected_cuts

    def test_cutfile_output(self, cutfile):
        expected_treedict = {
            'truth': {'testvartruth', 'MC_WZmu_el_eta_bare', 'MC_WZneutrino_pt_born', 'MC_WZmu_el_pt_born', 'PDFinfo_Q',
                      'MC_WZ_dilep_pt_born'},
            'reco': {'mu_d0sig', 'met_phi', 'jet_e',
                     # varibles to calculate mu_mt
                     'met_phi', 'met_met', 'mu_phi', 'mu_pt'}
        }
        expected_vars_to_cut = {'mu_mt_reco'}

        assert cutfile.tree_dict == expected_treedict
        assert cutfile.vars_to_calc == expected_vars_to_cut