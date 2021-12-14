import collections
import logging
import os
import time
from distutils.util import strtobool
from shutil import copyfile
from typing import Tuple, List, OrderedDict, Dict, Set
from warnings import warn

import pandas as pd

from utils.file_utils import identical_to_backup, get_last_backup, is_dir_empty, get_filename
from utils.var_helpers import OtherVar


class Cutfile:
    """
    Handles importing cutfiles and extracting variables
    """
    def __init__(self, file_path: str, backup_path: str, logger: logging.Logger):
        self.logger = logger
        self.name = get_filename(file_path)
        self._path = file_path
        self.backup_path = backup_path
        self.cut_dicts, self.vars_to_cut, self.options = self.parse_cutfile(file_path)
        self.cutgroups = self.gen_cutgroups(self.cut_dicts)
        self.alt_trees = self.gen_tree_dict(self.cut_dicts)

    def __repr__(self):
        return f'Cutfile("{self._path}")'

    @classmethod
    def parse_cutline(cls, cutline: str, sep='\t') -> dict:
        """
        Processes each line of cuts into dictionary of cut options. with separator sep
        """
        cutline_split = cutline.split(sep)

        # if badly formatted
        if len(cutline_split) not in (6, 7):
            raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}.")
        for v in cutline_split:
            if len(v) == 0:
                raise SyntaxError(f"Check cutfile. Blank value given in line {cutline}. Got {cutline_split}.")
            if v[0] == ' ' or v[-1] == ' ':
                warn(f"Found trailing space in option cutfile line {cutline}: Variable '{v}'.")

        name = cutline_split[0]
        cut_var = cutline_split[1]
        relation = cutline_split[2]
        try:
            cut_val = float(cutline_split[3])
        except ValueError:  # make sure the cut value is actually a number
            raise SyntaxError(f"Check 'cut_val' argument in line {cutline}. Got '{cutline_split[3]}'.")
        group = cutline_split[4]
        is_symmetric = bool(strtobool(cutline_split[5].lower()))  # converts string to boolean
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
        }
        if tree:
            cut_dict['tree'] = tree

        return cut_dict

    @classmethod
    def parse_cutfile(cls, file: str, sep='\t') -> Tuple[List[dict], Set[Tuple[str, str]], Dict[str, bool]]:
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
        | - sequential: (bool) whether each cut should be applied sequentially so a cutflow can be generated
        """
        # open file
        with open(file, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

            # get cut lines
            cuts_list_of_dicts = []
            for cutline in lines[lines.index('[CUTS]') + 1: lines.index('[OUTPUTS]')]:
                if cutline.startswith('#') or len(cutline) < 2:
                    continue
                cuts_list_of_dicts.append(cls.parse_cutline(cutline, sep=sep))

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
                    output_vars.add((out_var, 'na'))
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
                    raise Exception(f'Badly Formatted option: {sep.join(option)}')

                options_dict[option[0]] = bool(strtobool(option[1].lower()))  # converts string to boolean

            # Options necessary for the analysis to run (remember to add to this when adding new options)
            necessary_options = [
                'sequential',
                'grouped cutflow',
            ]
            if missing_options := [opt for opt in necessary_options if opt not in options_dict.keys()]:
                raise Exception(f"Missing option(s) in cutfile: {', '.join(missing_options)}")

        return cuts_list_of_dicts, output_vars, options_dict

    def extract_variables(self, derived_vars: Dict[str, OtherVar]) -> Tuple[Dict[str, set[str]], Set[str]]:
        """
        Get which variables are needed to extract from root file based on cutfile parser output
        uses outputs from parse_cutfile()

        :return Tuple[{Dictionary of trees and its variables to extract}, {set of variables to calculate}]
        """
        tree_dict = self.gen_tree_dict(self.cut_dicts)
        calc_vars = {var for var in {cut_dict['cut_var'] for cut_dict in self.cut_dicts} if var in derived_vars}
        calc_vars |= {var for var, _ in self.vars_to_cut if var in derived_vars}
        self.vars_to_cut -= {(var, tree) for var, tree in self.vars_to_cut if var in derived_vars}

        # for variables in the 'default' tree (when tree not specified)
        tree_dict['na'] = {cut_dict['cut_var'] for cut_dict in self.cut_dicts if 'tree' not in cut_dict} - calc_vars

        for var, tree in self.vars_to_cut:
            if tree in tree_dict:
                tree_dict[tree] |= {var}
            else:
                tree_dict[tree] = {var}

        # add any variables needed from which trees for calculating derived variables
        for calc_var in calc_vars:
            if (alt_tree := derived_vars[calc_var]['tree']) in tree_dict:
                tree_dict[alt_tree] |= set(derived_vars[calc_var]['var_args'])
            else:
                tree_dict[alt_tree] = set(derived_vars[calc_var]['var_args'])

        return tree_dict, calc_vars

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

    @classmethod
    def gen_tree_dict(cls, list_of_cut_dicts: List[dict]) -> Dict[str, set[str]]:
        """
        generate dictionary like {'tree': {'var', ...}, ...} in order to extract variables from other trees in root file
        'na' when tree not given (this is the default tree given in dataset)
        """
        out = {'na': set()}
        for cut_dict in list_of_cut_dicts:
            if 'tree' in cut_dict:
                if cut_dict['tree'] not in out:
                    out[cut_dict['tree']] = {cut_dict['cut_var']}
                else:
                    out[cut_dict['tree']].add(cut_dict['cut_var'])
            else:
                out['na'].add(cut_dict['cut_var'])
        return out

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
                old_df = pd.load_pickle(pkl_filepath)
                old_cols = set(old_df.columns)
                current_variables = self.all_vars(self.cut_dicts, {var[0] for var in self.vars_to_cut})
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

    def backup_cutfile(self, name: str) -> None:
        cutfile_backup_filepath = self.backup_path + self.name + '_' + name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        copyfile(self._path, cutfile_backup_filepath)
        self.logger.info(f"Backup cutfile saved in {cutfile_backup_filepath}")
