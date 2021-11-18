import numpy as np
import pandas as pd
import pytest

import src.config as config
from src.dataset import Dataset


class TestBuildDataframe(object):
    test_cut_dicts = [
        {'name': 'cut 1', 'cut_var': 'testvar1', 'relation': '<=', 'cut_val': 100, 'group': 'var1cut',
         'is_symmetric': True},
        {'name': 'cut 2', 'cut_var': 'testvar1', 'relation': '>', 'cut_val': 1, 'group': 'var1cut',
         'is_symmetric': False}
    ]
    test_vars_to_cut = {'testvar1', 'testvar3'}
    expected_output = pd.DataFrame({
        'testvar1': np.arange(1000),
        'testvar3': np.arange(1000) * 3,
        'weight_mc': np.append(np.ones(990), -1 * np.ones(10)),
        'eventNumber': np.arange(1000),
    })
    default_TTree = 'tree1'

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_normal_input(self, tmp_root_datafile):
        output = Dataset._build_dataframe(tmp_root_datafile,
                                          TTree_name=self.default_TTree,
                                          cut_list_dicts=self.test_cut_dicts,
                                          vars_to_cut=self.test_vars_to_cut
                                          )
        # test column names are the same
        assert set(output.columns) == set(self.expected_output.columns)
        # test contents are the same
        for col in output.columns:
            assert np.array_equal(output[col], self.expected_output[col])

    def test_missing_tree(self, tmp_root_datafile):
        with pytest.raises(ValueError) as e:
            _ = Dataset._build_dataframe(tmp_root_datafile,
                                         TTree_name='missing',
                                         cut_list_dicts=self.test_cut_dicts,
                                         vars_to_cut=self.test_vars_to_cut
                                         )
        assert str(e.value) == f"TTree(s) 'missing' not found in file {tmp_root_datafile}"

    def test_missing_branch(self, tmp_root_datafile):
        missing_branches = {'missing1', 'missing2'}
        with pytest.raises(ValueError) as e:
            _ = Dataset._build_dataframe(tmp_root_datafile,
                                         TTree_name=self.default_TTree,
                                         cut_list_dicts=self.test_cut_dicts,
                                         vars_to_cut=missing_branches
                                         )
        assert e.match(r"Missing TBranch\(es\) .* in TTree 'tree1' of file .*")

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_multifile(self, tmp_root_datafiles):
        expected_output = pd.DataFrame({'testvar1': np.concatenate((np.arange(3000),
                                                                    np.arange(2000),
                                                                    np.arange(1000))),
                                        'testvar3': np.concatenate((np.arange(3000) * 3,
                                                                    np.arange(2000) * 2,
                                                                    np.arange(1000))),
                                        'weight_mc': np.concatenate((np.append(np.ones(2970), -1 * np.ones(30)),
                                                                     np.append(np.ones(1980), -1 * np.ones(20)),
                                                                     np.append(np.ones(990), -1 * np.ones(10)),)),
                                        'eventNumber': np.concatenate((np.arange(3000, 6000),
                                                                       np.arange(1000, 3000),
                                                                       np.arange(1000)))
                                        })
        output = Dataset._build_dataframe(tmp_root_datafiles,
                                          TTree_name=self.default_TTree,
                                          cut_list_dicts=self.test_cut_dicts,
                                          vars_to_cut=self.test_vars_to_cut
                                          )
        # test column names are the same
        assert set(output.columns) == set(expected_output.columns)
        # test contents are the same
        for col in output.columns:
            assert np.array_equal(output[col], expected_output[col]), \
                f"Dataframe builder failed in column {col};\n" \
                f"Expected: \n{expected_output[col]},\n" \
                f"Got: \n{output[col]}"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_mass_slices(self, tmp_root_datafiles):
        """Test input as 'mass slices'"""
        expected_output = pd.DataFrame({'testvar1': np.concatenate((np.arange(3000),
                                                                    np.arange(2000),
                                                                    np.arange(1000))),
                                        'testvar3': np.concatenate((np.arange(3000) * 3,
                                                                    np.arange(2000) * 2,
                                                                    np.arange(1000))),
                                        'weight_mc': np.concatenate((np.append(np.ones(2970), -1 * np.ones(30)),
                                                                     np.append(np.ones(1980), -1 * np.ones(20)),
                                                                     np.append(np.ones(990), -1 * np.ones(10)),)),
                                        'eventNumber': np.concatenate((np.arange(3000, 6000),
                                                                       np.arange(1000, 3000),
                                                                       np.arange(1000))),
                                        # dataset IDs
                                        'DSID': np.concatenate((np.full(3000, 3),
                                                                np.full(2000, 2),
                                                                np.full(1000, 1)
                                                                )),
                                        # sum of weights for events with same dataset IDs
                                        'totalEventsWeighted': np.concatenate((np.full(3000, 2940),
                                                                               np.full(2000, 1960),
                                                                               np.full(1000, 980)))
                                        })
        output = Dataset._build_dataframe(tmp_root_datafiles,
                                          TTree_name=self.default_TTree,
                                          cut_list_dicts=self.test_cut_dicts,
                                          vars_to_cut=self.test_vars_to_cut,
                                          is_slices=True,
                                          )
        # test column names are the same
        assert set(output.columns) == set(expected_output.columns)
        # test contents are the same
        for col in output.columns:
            assert np.array_equal(output[col], expected_output[col]), \
                f"Dataframe builder failed in column {col};\n" \
                f"Expected: \n{expected_output[col]},\n" \
                f"Got: \n{output[col]}"
        # TODO: check lumi

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_alt_trees(self, tmp_root_datafile):
        newcut = {
            'name': 'cut 3',
            'cut_var': 'testvar4',
            'relation': '<',
            'cut_val': -10,
            'group': 'var4cut',
            'is_symmetric': False,
            'tree': 'tree2'
        }
        list_of_dicts = self.test_cut_dicts.copy()
        list_of_dicts += [newcut]
        expected_output = self.expected_output.copy()
        expected_output['testvar4'] = np.arange(1000) * -1
        expected_output['eventNumber'] = np.arange(1000)
        output = Dataset._build_dataframe(tmp_root_datafile,
                                          TTree_name=self.default_TTree,
                                          cut_list_dicts=list_of_dicts,
                                          vars_to_cut=self.test_vars_to_cut,
                                          is_slices=False)
        assert set(output.columns) == set(expected_output.columns)
        # test contents are the same
        for col in output.columns:
            assert np.array_equal(output[col], expected_output[col]), \
                f"Dataframe builder failed in column {col};\n" \
                f"Expected: \n{expected_output[col]},\n" \
                f"Got: \n{output[col]}"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_events_no_alt_tree(self, tmp_root_datafile_duplicate_events):
        with pytest.raises(ValueError) as e:
            _ = Dataset._build_dataframe(tmp_root_datafile_duplicate_events,
                                         TTree_name=self.default_TTree,
                                         cut_list_dicts=self.test_cut_dicts,
                                         vars_to_cut=self.test_vars_to_cut,
                                         is_slices=False)
        assert str(e.value) == f"Found 1000 duplicate events in datafile {tmp_root_datafile_duplicate_events}."

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_events_alt_tree(self, tmp_root_datafile_duplicate_events):
        with pytest.raises(Exception) as e:
            newcut = {
                'name': 'cut 3',
                'cut_var': 'testvar4',
                'relation': '<',
                'cut_val': -10,
                'group': 'var4cut',
                'is_symmetric': False,
                'tree': 'tree2'
            }
            newlist = self.test_cut_dicts.copy()
            newlist += [newcut]
            _ = Dataset._build_dataframe(tmp_root_datafile_duplicate_events,
                                         TTree_name=self.default_TTree,
                                         cut_list_dicts=newlist,
                                         vars_to_cut=self.test_vars_to_cut,
                                         is_slices=False)
        assert str(e.value) == "Duplicated events in both 'tree1' and 'tree2' TTrees"

    def test_derived_variable(self, tmp_root_datafile):
        derived_vars = {
            'dev_var1': {
                'var_args': ['testvar1', 'testvar2'],
                'func': lambda x, y: x + y
            }
        }
        # TODO


class TestCreateCutColumns(object):
    def test_morethan_lessthan(self):
        """Test more/less than cuts"""
        test_df = pd.DataFrame({
            'testvar1': [0, 1, 2, 300, 10, 100, -22, -10000, 0.0001, 12.2],
        })
        test_cut_dicts = [{'name': 'cut 1', 'cut_var': 'testvar1', 'relation': '<=', 'cut_val': 100, 'group': 'var1cut',
                           'is_symmetric': True},
                          {'name': 'cut 2', 'cut_var': 'testvar1', 'relation': '>', 'cut_val': 1, 'group': 'var1cut',
                           'is_symmetric': False}
                          ]
        cut_label = config.cut_label

        Dataset._create_cut_columns(test_df, test_cut_dicts)
        out_column1 = pd.Series(data=[True, True, True, False, True, True, True, False, True, True],
                                name='cut 1' + cut_label)
        out_column2 = pd.Series(data=[False, False, True, True, True, True, False, False, False, True],
                                name='cut 2' + cut_label)

        assert pd.Series.equals(test_df['cut 1' + cut_label], out_column1), \
            f"Expected {out_column1}, got {test_df['cut 1' + cut_label]}"
        assert pd.Series.equals(test_df['cut 2' + cut_label], out_column2), \
            f"Expected {out_column2}, got {test_df['cut 2' + cut_label]}"

    def test_equals_nequals(self):
        """Test (not) equals cuts"""
        test_df = pd.DataFrame({
            'testvar1': [1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        })
        test_cut_dicts = [{'name': 'cut 1', 'cut_var': 'testvar1', 'relation': '=', 'cut_val': 1, 'group': 'var1cut',
                           'is_symmetric': True},
                          {'name': 'cut 2', 'cut_var': 'testvar1', 'relation': '!=', 'cut_val': 1, 'group': 'var1cut',
                           'is_symmetric': False}
                          ]
        cut_label = config.cut_label

        Dataset._create_cut_columns(test_df, test_cut_dicts)
        out_column1 = pd.Series(data=[True, False, True, False, False, False, True, True, True, False],
                                name='cut 1' + cut_label)
        out_column2 = pd.Series(data=[False, True, False, True, True, True, False, False, False, True],
                                name='cut 2' + cut_label)

        assert pd.Series.equals(test_df['cut 1' + cut_label], out_column1), \
            f"Expected {out_column1}, got {test_df['cut 1' + cut_label]}"
        assert pd.Series.equals(test_df['cut 2' + cut_label], out_column2), \
            f"Expected {out_column2}, got {test_df['cut 2' + cut_label]}"
