import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Dict, Set, Union

from src.logger import get_logger
from utils.file_utils import get_filename
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data

# all variables known by this framework
all_vars = set(derived_vars.keys()) | set(variable_data.keys())


@dataclass
class Cut:
    """Cut class containing info for each cut"""
    name: str
    cutstr: str
    var: Union[str, Set[str]]
    tree: str

    def __str__(self) -> str:
        return f"{self.name}: {self.cutstr}"


class Cutfile:
    """Handles importing cutfiles and extracting variables"""
    def __init__(self, file_path: str, default_tree: str = '0:NONE', logger: logging.Logger = None, sep='\t'):
        """
        Read and pull variables and cuts from cutfile.

        :param file_path: cutfile
        :param default_tree: name of TTree to assume if not given in file path. default value is to avoid overlap with
                             any possible TTree names
        :param logger: logger to output messages to. Will default to console output if not given
        :param sep: separator for values in cutfile. Default is TAB
        """
        self.sep = sep
        self.logger = logger if logger is not None else get_logger()
        self.name = get_filename(file_path)
        self._path = file_path
        self.given_tree = default_tree
        self.cuts, self.__vars_to_cut = self.parse_cutfile()
        self.tree_dict, self.vars_to_calc = self.extract_variables()

    def __repr__(self):
        return f'Cutfile("{self._path}")'

    def parse_cutline(self, cutline: str) -> Cut:
        """Processes each line of cuts into dictionary of cut options. with separator <sep>"""
        # strip trailing and leading spaces
        cutline_split = [i.strip() for i in cutline.split(self.sep)]

        # if badly formatted
        if len(cutline_split) not in (2, 3):
            raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}.")
        for v in cutline_split:
            if len(v) == 0:
                raise SyntaxError(f"Check cutfile. Blank value given in line {cutline}. Got {cutline_split}.")

        name = cutline_split[0]
        cut_str = cutline_split[1]

        try:
            tree = cutline_split[2]  # if an alternate TTree is given
        except IndexError:
            tree = self.given_tree

        if tree == '':
            raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}.")

        # check for variables in the cut string
        split_string = set(re.findall(r"[\w]+|[.,!?;]", cut_str))
        cutvars = all_vars & split_string  # find all known variables in cutstring

        if len(cutvars) == 1:
            var = cutvars.pop()
        elif len(cutvars) > 1:
            var = cutvars
        else:
            raise ValueError(f"No known variable in string '{cut_str}' for line '{cutline}'\n"
                             f"Read {cutline_split}")

        return Cut(name=name, cutstr=cut_str, var=var, tree=tree)

    def parse_cutfile(self, path: str = None, sep='\t') -> Tuple[OrderedDict[str, Cut], Set[Tuple[str, str]]]:
        """
        | Generates pythonic outputs from input cutfile
        | Cutfile should be formatted with headers [CUTS] and [OUTPUTS]
        | Each line under [CUTS] header contains the 'sep'-separated values (detault: tab):
        | - name: name of cut to be printed and used in plot labels
        | - cut_var: variable in root file to cut on
        | - relation: '<' or '>'
        | - cut_val: value of cut on variable
        |
        | Each line under [OUTPUTS] should be a variable in root file

        :param path: path to an alternative cutfile
        :param sep: cutfile separator. Default is TAB
        :return list of cut dictionary, variables to output
        """
        if not path:
            path = self._path

        with open(path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

            # get cut lines
            cuts = OrderedDict()
            for cutline in lines[lines.index('[CUTS]') + 1: lines.index('[OUTPUTS]')]:
                if cutline.startswith('#') or len(cutline) < 2:
                    continue
                cut = self.parse_cutline(cutline)
                if cut.name in cuts:
                    raise ValueError(f"Duplicate cut name in cutfile: {cut.name}")
                cuts[cut.name] = cut

            # get output variables
            output_vars = set()
            for output_var in lines[lines.index('[OUTPUTS]') + 1:]:
                if output_var.startswith('#') or len(output_var) < 2:
                    continue

                var_tree = [i.strip() for i in output_var.split(sep)]
                if len(var_tree) > 2:
                    raise SyntaxError(f"Check line '{output_var}'. Should be variable and tree (optional). "
                                      f"Got '{var_tree}'.")
                elif len(var_tree) == 2:
                    out_var, tree = var_tree
                    output_vars.add((out_var, tree))
                elif len(var_tree) == 1:
                    out_var = var_tree[0]
                    output_vars.add((out_var, self.given_tree))
                else:
                    raise Exception("This should never happen")

        return cuts, output_vars

    def extract_variables(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Get which variables are needed to extract from root file based on cutfile parser output
        uses outputs from parse_cutfile()

        :return Tuple[{Dictionary of trees and its variables to extract}, {set of variables to calculate}]
        """
        # generate initial dict with given (default) TTree
        tree_dict = {self.given_tree: set()}
        extracted_vars = dict()  # keep all extracted variables here

        for cut in self.cuts.values():
            # cut could have multiple variables in it
            if isinstance(cut.var, str):
                the_var = {cut.var}
                extracted_vars[cut.var] = cut.tree
            elif isinstance(cut.var, set):
                the_var = cut.var
                for var in the_var:
                    extracted_vars[var] = cut.tree
            else:
                raise ValueError("This should never happen")

            if cut.tree not in tree_dict:
                tree_dict[cut.tree] = the_var
            else:
                tree_dict[cut.tree] |= the_var

        # add variables not cut on to tree dict
        for var, tree in self.__vars_to_cut:
            extracted_vars[var] = tree
            if tree in tree_dict:
                tree_dict[tree] |= {var}
            else:
                tree_dict[tree] = {var}

        # work out which variables to calculate and which to extract from ROOT file
        calc_vars = {  # cut variables that are in derived vars
            (var, tree) for var, tree in extracted_vars.items()
            if var in derived_vars
        }

        # remove variables to calculate from tree dict
        for tree in tree_dict:
            tree_dict[tree] -= {var for var, _ in calc_vars}

        # add any variables needed from which trees for calculating derived variables
        for calc_var, tree in calc_vars:
            if tree == self.given_tree:
                tree = derived_vars[calc_var]['tree']
            if tree in tree_dict:
                tree_dict[tree] |= set(derived_vars[calc_var]['var_args'])
            else:
                tree_dict[tree] = set(derived_vars[calc_var]['var_args'])

        # only return the actual variable names to calculate. Which tree to extract from will be handled by tree_dict
        return tree_dict, {var for var, _ in calc_vars}

    @classmethod
    def all_vars(cls, cuts: OrderedDict[str, Cut], vars_set: Set[Tuple[str, str]]) -> Set[str]:
        """Return all variables mentioned in cutfile"""
        return {cut.var for cut in cuts.values()} | {var for var, _ in vars_set}

    @staticmethod
    def truth_reco(tree_dict: Dict[str, Set[str]]) -> Tuple[bool, bool]:
        """Does cutfile ask for truth data? Does cutfile ask for reco data?"""
        is_truth = is_reco = False
        for var_ls in tree_dict.values():
            for var in var_ls:
                if var in variable_data:
                    if variable_data[var]['tag'] == 'truth':
                        is_truth = True
                    elif variable_data[var]['tag'] == 'reco':
                        is_reco = True
        return is_truth, is_reco

    def get_cut_string(self, cut_label: str, name: bool = False, align: bool = False) -> str:
        """
        Get displayed string of cut.

        :param cut_label: name of cut to print
        :param name: if starting with the name of the cut
        :param align: if aligning to all other cuts
        :return: string of style "cool name: var_1 > 10" if name is True else "var_1 > 10"
                 optionally returns the string 'None' if cut_label is None
        """
        if cut_label is None:
            return 'None'

        # get cut dict with name cut_label
        cut = next((c for c in self.cuts.values() if c.name == cut_label), None)
        if cut is None:
            raise ValueError(f"No cut named '{cut_label}' in cutfile {self.name}")

        name_len = max([len(cut_name) for cut_name in self.cuts]) if align else 0

        return (f"{cut.name:<{name_len}}: " if name else '') + cut.cutstr

    def cut_exists(self, cut_name: str) -> bool:
        """check if cut exists in cutfile"""
        return cut_name in self.cuts

    def log_cuts(self, name: bool = True, debug: bool = False) -> None:
        """send list of cuts in cutfile to logger"""
        for cut_name in self.cuts:
            if debug:
                self.logger.debug(self.get_cut_string(cut_name, name=name, align=True))
            else:
                self.logger.info(self.get_cut_string(cut_name, name=name, align=True))
