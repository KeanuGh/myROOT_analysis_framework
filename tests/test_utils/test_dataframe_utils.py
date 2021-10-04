from utils.dataframe_utils import *


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

