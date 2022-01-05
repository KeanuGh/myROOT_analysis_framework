import collections
import logging
import os
import time
from distutils.util import strtobool
from shutil import copyfile
from typing import Tuple, List, OrderedDict, Dict, Set

import pandas as pd

from utils.file_utils import identical_to_backup, get_last_backup, is_dir_empty, get_filename
from utils.var_helpers import OtherVar


class Cutfile:
    """
    Handles importing cutfiles and extracting variables
    """    
    def __init__(self, file_path: str, backup_path: str, logger: logging.Logger, sep='\t'):
        self.sep = sep
        self.__na_tree = '0:NONE'
        self.logger = logger
        self.name = get_filename(file_path)
        self._path = file_path
        self.backup_path = backup_path
        self.cut_dicts, self.vars_to_cut, self.options = self.parse_cutfile()
        self.cutgroups = self.gen_cutgroups(self.cut_dicts)
        self.has_reco = False
        self.has_truth = False

        self.logger.info('')
        self.logger.info("========== CUTS USED ============")
        self.log_cuts()
        self.logger.info('')

    def __repr__(self):
        return f'Cutfile("{self._path}")'

    def parse_cutline(self, cutline: str) -> dict:
        """
        Processes each line of cuts into dictionary of cut options. with separator sep
        """
        cutline_split = cutline.split(self.sep)

        # if badly formatted
        if len(cutline_split) not in (6, 7):
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
        
        group = cutline_split[4]
        
        try:
            is_symmetric = bool(strtobool(cutline_split[5].lower()))  # converts string to boolean
        except ValueError as e:
            raise ValueError(f"Incorrect formatting for 'is_symmetric' in line {cutline} \n"
                             f"Got: {e}")
            
        try:
            tree = cutline_split[6]  # if an alternate TTree is given
        except IndexError:
            tree = None
            
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
            'group': group,
            'is_symmetric': is_symmetric,
            'tree': tree if tree else self.__na_tree
        }

        return cut_dict

    def parse_cutfile(self, path: str = None, sep='\t') -> Tuple[List[dict], Set[Tuple[str, str]], Dict[str, bool]]:
        """
        | Generates pythonic outputs from input cutfile
        | Cutfile should be formatted with headers [CUTS], [OUTPUTS] and [OPTIONS]
        | Each line under [CUTS] header contains the 'sep'-separated values (detault: tab):
        | - name: name of cut to be printed and used in plot labels
        | - cut_var: variable in root file to cut on
        | - relation: '<' or '>'
        | - cut_val: value of cut on variable
        | - group: each cut with same group number will be applied all at once.
        |          !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!
        |
        | Each line under [OUTPUTS] should be a variable in root file
        |
        | Each line under [OPTIONS] header should be '[option]<sep>[value]'

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
            for output_var in lines[lines.index('[OUTPUTS]') + 1: lines.index('[OPTIONS]')]:
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

            # global cut options
            options_dict = {}
            for option in lines[lines.index('[OPTIONS]') + 1:]:
                if option.startswith('#') or len(option) < 2:
                    continue
                option = option.split(sep)

                # options should be formatted as '[option]<sep>[value]'
                if len(option) != 2:
                    raise Exception(f'Badly Formatted option: {option}')

                options_dict[option[0]] = bool(strtobool(option[1].lower()))  # converts string to boolean

            # Options necessary for the analysis to run (remember to add to this when adding new options)
            necessary_options = [
                'grouped cutflow',
            ]
            if missing_options := [opt for opt in necessary_options if opt not in options_dict.keys()]:
                raise Exception(f"Missing option(s) in cutfile: {', '.join(missing_options)}")

        return cuts_list_of_dicts, output_vars, options_dict

    def extract_variables(self,
                          derived_vars: Dict[str, OtherVar],
                          list_of_cut_dicts: List[dict]
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
    def all_vars(cls, cut_dicts: List[dict], vars_set: Set[Tuple[str, str]]) -> Set[str]:
        """Return all variables mentioned in cutfile"""
        return {cut_dict['cut_var'] for cut_dict in cut_dicts} | {var for var, _ in vars_set}

    @classmethod
    def gen_cutgroups(cls, cut_list_of_dicts: List[dict]) -> OrderedDict[str, List[str]]:
        """
        Creates an ordererd dictionary, where the keys are strings containing the name of the group,
        and the values are a list of all the cuts to be applied at once (the cutgroup)
        """
        cutgroups = []

        for cut_dict in cut_list_of_dicts:
            # if group exists, add cut name to group
            curr_groups = [group[0] for group in cutgroups]
            if cut_dict['group'] in curr_groups:
                cutgroups[curr_groups.index(cut_dict['group'])][1].append(cut_dict['name'])
            # else make new group and add group label as first element
            else:
                cutgroups.append((cut_dict['group'], [cut_dict['name']]))

        return collections.OrderedDict(cutgroups)

    def extract_var_data(self,
                         derived_vars: Dict[str, OtherVar],
                         default_tree_name: str,
                         ) -> Tuple[Dict[str, Set[str]], Set[str], bool, bool]:
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

        reco = False
        truth = False
        for tree in tree_dict:
            # add necessary metadata to all trees
            tree_dict[tree] |= {'mcChannelNumber', 'eventNumber'}
            tree_dict[tree] -= vars_to_calc
            if 'nominal' in tree.lower():
                self.logger.info(f"Detected {tree} as reco tree, will pull 'weight_leptonSF' and 'weight_KFactor'")
                tree_dict[tree] |= {'weight_leptonSF', 'weight_KFactor'}
                reco = True
            elif 'truth' in tree.lower():
                self.logger.info(f"Detected {tree} as truth tree, will pull 'KFactor_weight_truth'")
                tree_dict[tree].add('KFactor_weight_truth')
                truth = True
            else:
                self.logger.info(f"Neither {tree} as truth nor reco dataset detected.")

        return tree_dict, vars_to_calc, truth, reco

    def if_make_cutfile_backup(self) -> bool:
        """Decides if a backup cutfile should be made"""
        if get_last_backup(self.backup_path, self.name):
            return not identical_to_backup(self._path, backup_dir=self.backup_path, name=self.name, logger=self.logger)
        else:
            return True

    def if_build_dataframe(self, pkl_filepath: str) -> bool:
        """
        compares current cutfile and dataframe to backups and decides whether to rebuild dataframe
        cutfile
        :param pkl_filepath: pickle file containing data in pandas dataframe
        :return: whether to build new dataframe
        """
        is_pkl_file = os.path.isfile(pkl_filepath)
        if is_pkl_file:
            self.logger.debug(f"Previous datafile found in {pkl_filepath}.")
        else:
            return True

        # if cutfile backup exists, check for new variables
        if not is_dir_empty(self.backup_path):
            latest_backup = get_last_backup(self.backup_path, self.name)
            self.logger.debug(f"Found backup cutfile in {latest_backup}")

            BACKUP_cutfile_dicts, BACKUP_cutfile_outputs, _ = self.parse_cutfile(latest_backup)
            current_variables = self.all_vars(self.cut_dicts, self.vars_to_cut)
            backup_variables = self.all_vars(BACKUP_cutfile_dicts, BACKUP_cutfile_outputs)

            # check whether variables in current cutfile are in previous cutfile
            if not current_variables <= backup_variables:
                self.logger.debug(f"New variables found; dataframe will be rebuilt")
                self.logger.debug(f" Current cutfile variables: {current_variables}. Previous: {backup_variables}")
                return True
            else:
                self.logger.debug(f"All needed variables already contained in previous cutfile")
                self.logger.debug(f"previous cutfile variables: {backup_variables}")
                self.logger.debug(f"current cutfile variables: {current_variables}")

        # if backup doesn't exit, make backup and check if there is already a pickle file
        else:
            # if pickle file already exists
            self.logger.debug(f"No cutfile backup found in {self.backup_path}")
            if is_pkl_file:
                old_df = pd.read_pickle(pkl_filepath)
                old_cols = set(old_df.columns)
                current_variables = self.all_vars(self.cut_dicts, self.vars_to_cut)
                if current_variables <= old_cols:
                    self.logger.debug("All variables found in old pickle file.")
                    return False
                else:
                    self.logger.info("New variables found in cutfile. Will rebuild dataframe")
                    new_vars = {v for v in current_variables if v not in old_cols}
                    self.logger.debug(f"New cutfile variable(s): {new_vars}")
                    return True

            # if no backup or pickle file, rebuild
            else:
                self.logger.info("No pickle file found. Will rebuild dataframe.")
                return True

        return False

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

    def log_cuts(self, name: bool = True) -> None:
        """send list of cuts in cutfile to logger"""
        for cut_name in [cut['name'] for cut in self.cut_dicts]:
            self.logger.info(self.get_cut_string(cut_name, name=name, align=True))

    def backup_cutfile(self, name: str) -> None:
        cutfile_backup_filepath = f"{self.backup_path}{self.name}_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        copyfile(self._path, cutfile_backup_filepath)
        self.logger.info(f"Backup cutfile saved in {cutfile_backup_filepath}")
