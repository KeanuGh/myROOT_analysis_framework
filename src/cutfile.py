import logging
import time
from distutils.util import strtobool
from shutil import copyfile
from typing import Tuple, List, Dict, Set, TypedDict

from utils.file_utils import identical_to_backup, get_last_backup, get_filename
from utils.var_helpers import OtherVar


class CutDict(TypedDict):
    """Define type hint for cut dictionary"""
    name: str
    cut_var: str
    relation: str
    cut_val: float
    is_symmetric: bool
    tree: str


class Cutfile:
    """
    Handles importing cutfiles and extracting variables
    # TODO: create Cut class (will simpify getting cuts SO much)
    """
    def __init__(self, file_path: str, logger: logging.Logger, backup_path: str = None, sep='\t'):
        self.sep = sep
        self.__na_tree = '0:NONE'
        self.logger = logger
        self.name = get_filename(file_path)
        self._path = file_path
        self.backup_path = backup_path
        self.cut_dicts, self.vars_to_cut = self.parse_cutfile()

    def __repr__(self):
        return f'Cutfile("{self._path}")'

    def parse_cutline(self, cutline: str) -> CutDict:
        """
        Processes each line of cuts into dictionary of cut options. with separator sep
        """
        cutline_split = cutline.split(self.sep)

        # if badly formatted
        if len(cutline_split) not in (5, 6):
            raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}.")
        for v in cutline_split:
            if len(v) == 0:
                raise SyntaxError(f"Check cutfile. Blank value given in line {cutline}. Got {cutline_split}.")
            if v[0] == ' ' or v[-1] == ' ':
                self.logger.warning(f"Found trailing space in option cutfile line {cutline}: Variable '{v}'.")

        name = cutline_split[0]
        cut_var = cutline_split[1]
        relation = cutline_split[2]
        try:
            cut_val = float(cutline_split[3])
        except ValueError:  # make sure the cut value is actually a number
            raise SyntaxError(f"Check 'cut_val' argument in line {cutline}. Got '{cutline_split[3]}'.")

        try:
            is_symmetric = bool(strtobool(cutline_split[4].lower()))  # converts string to boolean
        except ValueError as e:
            raise ValueError(f"Incorrect formatting for 'is_symmetric' in line {cutline} \n"
                             f"Got: {e}")

        try:
            tree = cutline_split[5]  # if an alternate TTree is given
        except IndexError:
            tree = self.__na_tree

        if tree == '':
            raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}.")

        # check values
        if relation not in ('>', '<', '<=', '>=', '=', '!='):
            raise SyntaxError(f"Unexpected comparison operator: {cutline_split[2]}")

        # fill dictionary
        cut_dict = {
            'name': name,
            'cut_var': cut_var,
            'relation': relation,
            'cut_val': cut_val,
            'is_symmetric': is_symmetric,
            'tree': tree
        }

        return cut_dict

    def parse_cutfile(self, path: str = None, sep='\t') -> Tuple[List[CutDict], Set[Tuple[str, str]]]:
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
            cuts_list_of_dicts = []
            for cutline in lines[lines.index('[CUTS]') + 1: lines.index('[OUTPUTS]')]:
                if cutline.startswith('#') or len(cutline) < 2:
                    continue
                cuts_list_of_dicts.append(self.parse_cutline(cutline))

            # get output variables
            output_vars = set()
            for output_var in lines[lines.index('[OUTPUTS]') + 1:]:
                if output_var.startswith('#') or len(output_var) < 2:
                    continue
                elif len(var_tree := output_var.split(sep)) > 2:
                    raise SyntaxError(f"Check line '{output_var}'. Should be variable and tree (optional). "
                                      f"Got '{var_tree}'.")
                elif len(var_tree) == 2:
                    out_var, tree = var_tree
                    output_vars.add((out_var, tree))
                elif len(var_tree) == 1:
                    out_var = var_tree[0]
                    output_vars.add((out_var, self.__na_tree))
                else:
                    raise Exception("This should never happen")

        return cuts_list_of_dicts, output_vars

    def extract_variables(self,
                          derived_vars: Dict[str, OtherVar],
                          list_of_cut_dicts: List[CutDict]
                          ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Get which variables are needed to extract from root file based on cutfile parser output
        uses outputs from parse_cutfile()

        :return Tuple[{Dictionary of trees and its variables to extract}, {set of variables to calculate}]
        """
        # generate initial dict. Fill 'blank' tree with self.__na_tree to avoid possible tree name overlap
        tree_dict = {self.__na_tree: set()}
        for cut_dict in list_of_cut_dicts:
            if cut_dict['tree'] not in tree_dict:
                tree_dict[cut_dict['tree']] = {cut_dict['cut_var']}
            else:
                tree_dict[cut_dict['tree']].add(cut_dict['cut_var'])

        # work out which variables to calculate and which to extract from ROOT file
        calc_vars = {  # cut variables that are in derived vars
            (var, tree) for var, tree in {(_['cut_var'], _['tree']) for _ in self.cut_dicts}
            if var in derived_vars
        }
        calc_vars |= {  # add output variables in derived vars
            (var, tree) for var, tree in self.vars_to_cut
            if var in derived_vars
        }
        self.vars_to_cut -= {  # remove derived variables from variables to cut on
            (var, tree) for var, tree in self.vars_to_cut
            if var in derived_vars
        }
        tree_dict[self.__na_tree] = {  # get default tree variables and remove variables to calculate from default tree
            cut_dict['cut_var'] for cut_dict in self.cut_dicts
            if cut_dict['tree'] == self.__na_tree
        } - {var for var, _ in calc_vars}

        for var, tree in self.vars_to_cut:
            if tree in tree_dict:
                tree_dict[tree] |= {var}
            else:
                tree_dict[tree] = {var}

        # add any variables needed from which trees for calculating derived variables
        for calc_var, alt_tree in calc_vars:
            if alt_tree == self.__na_tree:  # if no tree provided use the one in the derived_vars dictionary
                alt_tree = derived_vars[calc_var]['tree']
            if alt_tree in tree_dict:
                tree_dict[alt_tree] |= set(derived_vars[calc_var]['var_args'])
            else:
                tree_dict[alt_tree] = set(derived_vars[calc_var]['var_args'])

        # only return the actual variable names to calculate. Which tree to extract from will be handled by tree_dict
        return tree_dict, {var for var, _ in calc_vars}

    @classmethod
    def all_vars(cls, cut_dicts: List[CutDict], vars_set: Set[Tuple[str, str]]) -> Set[str]:
        """Return all variables mentioned in cutfile"""
        return {cut_dict['cut_var'] for cut_dict in cut_dicts} | {var for var, _ in vars_set}

    def extract_var_data(self,
                         derived_vars: Dict[str, OtherVar],
                         default_tree_name: str,
                         ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        generate full tree dictionary that a Dataset object might need
        returns the tree dictionary, set of variables to calculate,
        and whether the dataset will contain truth, reco data
        """
        tree_dict, vars_to_calc = self.extract_variables(derived_vars, self.cut_dicts)

        # get set unlabeled variables in cutfile as being in default tree
        if default_tree_name in tree_dict:
            tree_dict[default_tree_name] |= tree_dict.pop(self.__na_tree, set())
        else:
            tree_dict[default_tree_name] = tree_dict.pop(self.__na_tree, set())

        # only add these to 'main tree' to avoid merge issues
        tree_dict[default_tree_name] |= {'weight_mc', 'weight_pileup'}

        for tree in tree_dict:
            # add necessary metadata to all trees
            tree_dict[tree] |= {'mcChannelNumber', 'eventNumber'}
            tree_dict[tree] -= vars_to_calc
            if 'nominal' in tree.lower():
                self.logger.info(f"Detected {tree} as reco tree, "
                                 f"adding 'weight_leptonSF' and 'weight_KFactor' to tree variables")
                tree_dict[tree] |= {'weight_leptonSF', 'weight_KFactor'}
            elif 'truth' in tree.lower():
                self.logger.info(f"Detected {tree} as truth tree, "
                                 f"adding 'KFactor_weight_truth' to tree variables")
                tree_dict[tree].add('KFactor_weight_truth')
            else:
                self.logger.info(f"Neither {tree} as truth nor reco dataset detected.")

        return tree_dict, vars_to_calc

    @staticmethod
    def truth_reco(tree_dict: Dict[str, Set[str]]) -> Tuple[bool, bool]:
        """Does cutfile ask for truth data? Does cutfile ask for reco data?"""
        is_truth = is_reco = False
        for tree in tree_dict.keys():
            if 'nominal' in tree:
                is_reco = True
            elif 'truth' in tree:
                is_truth = True
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
        cut_dict = next((d for d in self.cut_dicts if d['name'] == cut_label), None)
        if cut_dict is None:
            raise ValueError(f"No cut named '{cut_label}' in cutfile {self.name}")

        name_len = max([len(cut['name']) for cut in self.cut_dicts]) if align else 0
        var_len = max([len(cut['cut_var']) for cut in self.cut_dicts]) if align else 0

        if not cut_dict['is_symmetric']:
            return (f"{cut_dict['name']:<{name_len}}: " if name else '') + \
                   f"{cut_dict['cut_var']:>{var_len}} {cut_dict['relation']} {cut_dict['cut_val']}"
        else:
            return (f"{cut_dict['name']:<{name_len}}: " if name else '') + \
                   f"{'|' + cut_dict['cut_var']:>{max(var_len - 1, 0)}}| {cut_dict['relation']} {cut_dict['cut_val']}"

    def cut_exists(self, cut_name: str) -> bool:
        """check if cut exists in cutfile"""
        return cut_name in [cut['name'] for cut in self.cut_dicts]

    def log_cuts(self, name: bool = True, debug: bool = False) -> None:
        """send list of cuts in cutfile to logger"""
        for cut_name in [cut['name'] for cut in self.cut_dicts]:
            if debug:
                self.logger.debug(self.get_cut_string(cut_name, name=name, align=True))
            else:
                self.logger.info(self.get_cut_string(cut_name, name=name, align=True))

    # LEGACY FUNCTIONS (no longer used)
    def if_make_cutfile_backup(self) -> bool:
        """Decides if a backup cutfile should be made"""
        if self.backup_path is None:
            self.logger.info("No backup path given. Skipping backup check")
        elif get_last_backup(self.backup_path, self.name):
            return not identical_to_backup(self._path, backup_dir=self.backup_path, name=self.name, logger=self.logger)
        else:
            return True

    def backup_cutfile(self, name: str) -> None:
        if self.backup_path is None:
            self.logger.info("No cutfile backup path, skipping backup")
        cutfile_backup_filepath = f"{self.backup_path}{self.name}_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        copyfile(self._path, cutfile_backup_filepath)
        self.logger.info(f"Backup cutfile saved in {cutfile_backup_filepath}")
