from collections import OrderedDict

from utils.cutfile_utils import *

# test list of dicts
test_cut_list_of_dicts = [
    {
        'name': 'cut_1',
        'cut_var': 'var1',
        'relation': '=',
        'cut_val': 1,
        'group': 'group1',
        'is_symmetric': False,
    },
    {
        'name': 'cut_2',
        'cut_var': 'var2',
        'relation': '>=',
        'cut_val': 1,
        'group': 'group1',
        'is_symmetric': False,
    },
    {
        'name': 'cut_3',
        'cut_var': 'var1',
        'relation': '<=',
        'cut_val': 2,
        'group': 'group2',
        'is_symmetric': True,
    },
    {
        'name': 'cut_4',
        'cut_var': 'var3',
        'relation': '!=',
        'cut_val': 0,
        'group': 'group2',
        'is_symmetric': False,
    },
    {
        'name': 'cut_5',
        'cut_var': 'var4',
        'relation': '<',
        'cut_val': 1,
        'group': 'group3',
        'is_symmetric': False,
    },
]
test_output_list = [
    'out_var1',
    'out_var2',
    'out_var3',
]
test_options_dict = {
    'sequential': True,
    'grouped cutflow': True,
}


class TestExtractCutVariables(object):
    def test_cutvars_input(self):
        expected_output = {'var1', 'var2', 'var3', 'var4', 'out_var1', 'out_var2', 'out_var3'}
        actual_output = extract_cut_variables(test_cut_list_of_dicts, test_output_list)
        assert expected_output == actual_output, \
            f"Expected: {expected_output}. Actual: {actual_output}"


class TestGenCutroups(object):
    def test_cutgroups(self):
        # must be ordered dict because cuts need to be applied in the same order as in cutfile
        expected_output = OrderedDict([
            ('group1', ['cut_1', 'cut_2']),
            ('group2', ['cut_3', 'cut_4']),
            ('group3', ['cut_5']),
        ])
        actual_output = gen_cutgroups(test_cut_list_of_dicts)

        assert expected_output == actual_output, \
            f"Expected: {expected_output}. Actual: {actual_output}"
        assert type(expected_output) == type(actual_output), \
            f"Cutgroup output is of unexpected type. Expected: {type(expected_output)}. Actual: {type(actual_output)}"


class TestParseCutfile(object):
    # 'parse_cutfile' outputs a list of
    output = parse_cutfile('tests/test_cutfile.txt')

    def test_list_of_cut_dicts(self):
        for i, test_cut in enumerate(test_cut_list_of_dicts):
            assert self.output[0][i] == test_cut, \
                f"Cutfile parser failed in cut '{test_cut['name']}';\n" \
                f"Expected: \n{test_cut},\n" \
                f"Got: \n{self.output[0][i]}"

    def test_output_vars(self):
        assert self.output[1] == test_output_list, \
            f"Cutfile parser failed outputs. Expected: {test_output_list}. Got: {self.output[1]}"

    def test_option_dict(self):
        assert self.output[2] == test_options_dict, \
            f"Cutfile parser failed options. Expected: {test_options_dict}. Got: {self.output[2]}"
