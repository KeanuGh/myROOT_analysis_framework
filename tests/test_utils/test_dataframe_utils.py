import numpy as np
import pytest

from utils.dataframe_utils import *


@pytest.fixture
def tmp_root_datafile(tmpdir):
    """Generate test root file to read in"""
    datapath = tmpdir.join('test_data.root')
    with uproot.recreate(datapath) as test_file:
        test_file['tree1'] = {'testvar1': np.arange(1000),
                              'testvar2': np.arange(1000) * 1.1,
                              'testvar3': np.arange(1000) * 3,
                              'weight_mc': np.append(np.ones(990), -1*np.ones(10)),
                              'mcChannelNumber': np.append(np.ones(500), np.full(500, 2)),
                              }
        test_file['sumWeights'] = {'totalEventsWeighted': np.arange(4),
                                   'dsid': np.array([1, 1, 2, 2])}
    yield str(datapath)


class TestBuildAnalysisDataframe(object):
    test_cut_dicts = [{'name': 'cut 1', 'cut_var': 'testvar1', 'relation': '<=', 'cut_val': 100, 'group': 'var1cut',
                       'is_symmetric': True},
                      {'name': 'cut 2', 'cut_var': 'testvar1', 'relation': '>', 'cut_val': 1, 'group': 'var1cut',
                       'is_symmetric': False}
                      ]
    test_vars_to_cut = ['testvar1', 'testvar3']
    expected_output = pd.DataFrame({'testvar1': np.arange(1000),
                                    'testvar3': np.arange(1000) * 3,
                                    'weight_mc': np.append(np.ones(990), -1*np.ones(10)),
                                    })

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_normal_input(self, tmp_root_datafile):
        output = build_analysis_dataframe(tmp_root_datafile,
                                          TTree_name='tree1',
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
            _ = build_analysis_dataframe(tmp_root_datafile,
                                         TTree_name='missing',
                                         cut_list_dicts=self.test_cut_dicts,
                                         vars_to_cut=self.test_vars_to_cut
                                         )
            assert e.match(f"TTree 'missing' not found in file {tmp_root_datafile}")

    def test_missing_branch(self, tmp_root_datafile):
        with pytest.raises(ValueError) as e:
            missing_branches = ['missing1', 'missing2']
            _ = build_analysis_dataframe(tmp_root_datafile,
                                         TTree_name='tree1',
                                         cut_list_dicts=self.test_cut_dicts,
                                         vars_to_cut=missing_branches
                                         )
            assert e.match(f"Missing TBranch(es) {missing_branches} in TTree 'tree1' of file '{tmp_root_datafile}'.'")

    # TODO: Run this on MULTIPLE root files instead of just one
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_mass_slices(self, tmp_root_datafile):
        """Test input as 'mass slices'"""
        expected_output = pd.DataFrame({'testvar1': np.arange(1000),
                                        'testvar3': np.arange(1000) * 3,
                                        'weight_mc': np.append(np.ones(990), -1*np.ones(10)),
                                        # dataset IDs (half 1, half 2)
                                        'DSID': np.append(np.ones(500), np.full(500, 2)),
                                        # sum of weights for events with same dataset IDs
                                        'totalEventsWeighted': np.append(np.full(500, 500), np.full(500, 1800)),
                                        })
        output = build_analysis_dataframe(tmp_root_datafile,
                                          TTree_name='tree1',
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

    # TODO: test on derived variables
    # TODO: test on multiple files
    # TODO: test on large files(?)


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

        create_cut_columns(test_df, test_cut_dicts, printout=False)
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

        create_cut_columns(test_df, test_cut_dicts, printout=False)
        out_column1 = pd.Series(data=[True, False, True, False, False, False, True, True, True, False],
                                name='cut 1' + cut_label)
        out_column2 = pd.Series(data=[False, True, False, True, True, True, False, False, False, True],
                                name='cut 2' + cut_label)

        assert pd.Series.equals(test_df['cut 1' + cut_label], out_column1), \
            f"Expected {out_column1}, got {test_df['cut 1' + cut_label]}"
        assert pd.Series.equals(test_df['cut 2' + cut_label], out_column2), \
            f"Expected {out_column2}, got {test_df['cut 2' + cut_label]}"
