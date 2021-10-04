from collections import OrderedDict

from utils.cutfile_utils import *

# test list of dicts
test_cut_list_of_dicts = [
    {
        'name': 'cut_1',
        'cut_var': 'var1',
        'relation': '>',
        'cut_val': 1,
        'suffix': '_var1',
        'group': 'group1',
        'is_symmetric': True,
    },
    {
        'name': 'cut_2',
        'cut_var': 'var2',
        'relation': '>',
        'cut_val': 1,
        'suffix': '_var2',
        'group': 'group2',
        'is_symmetric': True,
    },
    {
        'name': 'cut_3',
        'cut_var': 'var3',
        'relation': '>',
        'cut_val': 1,
        'suffix': '_var3',
        'group': 'group2',
        'is_symmetric': True,
    },
    {
        'name': 'cut_4',
        'cut_var': 'var4',
        'relation': '>',
        'cut_val': 1,
        'suffix': '_var4',
        'group': 'group3',
        'is_symmetric': True,
    },
    {
        'name': 'cut_5',
        'cut_var': 'var5',
        'relation': '>',
        'cut_val': 1,
        'suffix': '_var5',
        'group': 'group3',
        'is_symmetric': True,
    },
]
test_output_list = [
    'out_var1',
    'out_var2',
    'out_var3',
]


class TestExtractCutVariables(object):
    def test_cutvars_input(self):
        expected_output = {'var1', 'var2', 'var3', 'var4', 'var5', 'out_var1', 'out_var2', 'out_var3'}
        actual_output = extract_cut_variables(test_cut_list_of_dicts, test_output_list)
        assert expected_output == actual_output, \
            f"Expected: {expected_output}. Actual: {actual_output}"


# TODO: MAKE SURE CUTS CUTGROUPS ARE APPLIED SEQUENTIALLY, OR IT DOESN'T MAKE SENSE

class TestGenCutroups(object):
    def test_cutgroups(self):
        # must be ordered dict because cuts need to be applied in the same order as in cutfile
        expected_output = OrderedDict([
            ('group1', ['cut_1']),
            ('group2', ['cut_2', 'cut_3']),
            ('group3', ['cut_4', 'cut_5']),
        ])
        actual_output = gen_cutgroups(test_cut_list_of_dicts)

        assert expected_output == actual_output, \
            f"Expected: {expected_output}. Actual: {actual_output}"
        assert type(expected_output) == type(actual_output), \
            f"Cutgroup output is of unexpected type. Expected: {type(expected_output)}. Actual: {type(actual_output)}"
