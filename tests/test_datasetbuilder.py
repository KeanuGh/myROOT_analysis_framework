import pytest


class TestBuildDataframe:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_normal_input(self, tmp_root_datafile):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_missing_tree(self, tmp_root_datafile):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_missing_branch(self, tmp_root_datafile):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_multifile(self, tmp_root_datafiles):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_mass_slices(self, tmp_root_datafiles):
        """Test input as 'mass slices'"""
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_alt_trees(self, tmp_root_datafile):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_events_no_alt_tree(self, tmp_root_datafile_duplicate_events):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_events_alt_tree(self, tmp_root_datafile_duplicate_events):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_missing_events_alt_tree(self, tmp_root_datafile_missing_events):
        ...

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_derived_variable(self, tmp_root_datafile):
        ...


class TestCreateCutColumns(object):
    def test_morethan_lessthan(self):
        """Test more/less than cuts"""
        ...

    def test_equals_nequals(self):
        """Test (not) equals cuts"""
        ...
