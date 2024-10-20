from collections import OrderedDict

import pytest

from src.cutfile import Cutfile

# test list of dicts
test_cut_list_of_dicts = [
    {
        "name": "cut 1",
        "cut_var": "testvar1",
        "relation": "<=",
        "cut_val": 100,
        "group": "var1cut",
        "is_symmetric": True,
    },
    {
        "name": "cut 2",
        "cut_var": "testvar1",
        "relation": ">",
        "cut_val": 1,
        "group": "var1cut",
        "is_symmetric": False,
    },
    {
        "name": "cut 3",
        "cut_var": "testvar4",
        "relation": "<",
        "cut_val": -10,
        "group": "var4cut",
        "is_symmetric": False,
        "tree": "tree2",
    },
]
test_uncut_set = {
    "testvar1",
    "testvar3",
}
test_options_dict = {
    "sequential": False,
    "grouped cutflow": False,
}


class TestExtractCutVariables(object):
    def test_calculated_variables(self):
        derived_vars = {
            "dev_var1": {
                "var_args": ["testvar1", "testvar2"],
                "tree": "tree1",
                "func": lambda x, y: x + y,
            },
            "dev_var2": {"var_args": ["testvar4"], "tree": "tree2", "func": lambda x: 2 * x},
        }
        new_cut_list = test_cut_list_of_dicts.copy()
        new_cut_list.append(
            {
                "name": "cut 4",
                "cut_var": "dev_var1",
                "relation": ">",
                "cut_val": 2,
                "group": "devcut",
                "is_symmetric": False,
            }
        )
        new_uncut_set = test_uncut_set.copy()
        new_uncut_set.add("dev_var2")
        exp_tree_dict = {
            "na": {"testvar1"},
            "tree1": {"testvar1", "testvar2"},
            "tree2": {"testvar4"},
        }
        exp_calc_vars = {"dev_var1", "dev_var2"}
        act_tree_dict, act_calc_vars = Cutfile.extract_cut_variables(
            new_cut_list, new_uncut_set, derived_vars
        )
        assert exp_tree_dict == act_tree_dict, f"Expected: {exp_tree_dict}, Actual: {act_tree_dict}"
        assert exp_calc_vars == act_calc_vars, f"Expected: {exp_calc_vars}, Actual: {act_calc_vars}"


class TestGenCutroups(object):
    def test_cutgroups(self):
        # must be ordered dict because cuts need to be applied in the same order as in cutfile
        expected_output = OrderedDict(
            [
                ("var1cut", ["cut 1", "cut 2"]),
                ("var4cut", ["cut 3"]),
            ]
        )
        actual_output = Cutfile.gen_cutgroups(test_cut_list_of_dicts)

        assert (
            expected_output == actual_output
        ), f"Expected: {expected_output}. Actual: {actual_output}"
        assert type(expected_output) == type(
            actual_output
        ), f"Cutgroup output is of unexpected type. Expected: {type(expected_output)}. Actual: {type(actual_output)}"


class TestParseCutfile(object):
    @pytest.fixture(scope="session")
    def output(self, tmp_cutfile):
        output = Cutfile.parse_cutfile(tmp_cutfile)
        yield output

    def test_list_of_cut_dicts(self, output):
        for i, test_cut in enumerate(test_cut_list_of_dicts):
            assert output[0][i] == test_cut, (
                f"Cutfile parser failed in cut '{test_cut['name']}';\n"
                f"Expected: \n{test_cut},\n"
                f"Got: \n{output[0][i]}"
            )

    def test_output_vars(self, output):
        assert (
            output[1] == test_uncut_set
        ), f"Cutfile parser failed outputs. Expected: {test_uncut_set}. Got: {self.output[1]}"

    def test_option_dict(self, output):
        assert (
            output[2] == test_options_dict
        ), f"Cutfile parser failed options. Expected: {test_options_dict}. Got: {self.output[2]}"


class TestParseCutline(object):
    sep = "\t"

    def test_no_tree(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}var1cut{0}true".format(self.sep)
        expected_output = {
            "name": "cut 1",
            "cut_var": "testvar1",
            "relation": "<=",
            "cut_val": 100,
            "group": "var1cut",
            "is_symmetric": True,
        }
        actual_output = Cutfile.parse_cutline(line)
        assert expected_output == actual_output

    def test_tree(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}var1cut{0}true{0}tree2".format(self.sep)
        expected_output = {
            "name": "cut 1",
            "cut_var": "testvar1",
            "relation": "<=",
            "cut_val": 100,
            "group": "var1cut",
            "is_symmetric": True,
            "tree": "tree2",
        }
        actual_output = Cutfile.parse_cutline(line)
        assert expected_output == actual_output

    def test_missing_value(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}true".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert (
            str(e.value)
            == f"Check cutfile. Line {line} is badly formatted. Got {line.split(self.sep)}."
        )

    def test_extra_value(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}var1cut{0}true{0}tree2{0}woah".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert (
            str(e.value)
            == f"Check cutfile. Line {line} is badly formatted. Got {line.split(self.sep)}."
        )

    def test_incorrect_cutval(self):
        line = "cut 1{0}testvar1{0}<={0}woops{0}var1cut{0}true{0}tree2".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert str(e.value) == f"Check 'cut_val' argument in line {line}. Got 'woops'."

    def test_blank_value(self):
        line = "cut 1{0}{0}<={0}100{0}var1cut{0}true{0}tree2{0}woah".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert (
            str(e.value)
            == f"Check cutfile. Line {line} is badly formatted. Got {line.split(self.sep)}."
        )

    def test_trailing_separator_with_tree(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}var1cut{0}true{0}tree2{0}".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert (
            str(e.value)
            == f"Check cutfile. Line {line} is badly formatted. Got {line.split(self.sep)}."
        )

    def test_trailing_separator_without_tree(self):
        line = "cut 1{0}testvar1{0}<={0}100{0}var1cut{0}true{0}".format(self.sep)
        with pytest.raises(SyntaxError) as e:
            _ = Cutfile.parse_cutline(line)
        assert (
            str(e.value)
            == f"Check cutfile. Blank value given in line {line}. Got {line.split(self.sep)}."
        )

    def test_trailing_space(self, caplog):
        line = "cut 1{0}testvar1 {0}<={0}100{0}var1cut{0}true".format(self.sep)
        _ = Cutfile.parse_cutline(line)
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert (
            f"Found trailing space in option cutfile line {line}: Variable 'testvar1 '.\n"
            in caplog.text
        )


class TestGenAltTreeDict(object):
    def test_default_input(self):
        expected = {"tree2": {"testvar4"}, "na": {"testvar1"}}
        actual = Cutfile.gen_tree_dict(test_cut_list_of_dicts)
        assert actual == expected

    def test_no_alt_trees(self):
        expected = {"na": {"testvar1"}}
        actual = Cutfile.gen_tree_dict(test_cut_list_of_dicts[:2])
        assert actual == expected

    def test_multiple_alt_trees(self):
        expected = {"tree2": {"testvar4", "testvar5"}, "tree3": {"testvar6"}, "na": {"testvar1"}}
        new_cuts = [
            {
                "name": "cut 4",
                "cut_var": "testvar5",
                "relation": ">",
                "cut_val": 1,
                "group": "var5cut",
                "is_symmetric": False,
                "tree": "tree2",
            },
            {
                "name": "cut 5",
                "cut_var": "testvar6",
                "relation": "<",
                "cut_val": -10,
                "group": "var6cut",
                "is_symmetric": True,
                "tree": "tree3",
            },
        ]
        new_cutlist = test_cut_list_of_dicts + new_cuts
        actual = Cutfile.gen_tree_dict(new_cutlist)
        assert actual == expected
