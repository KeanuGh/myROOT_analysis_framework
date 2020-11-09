import sys
import os
import time
from shutil import copyfile
from typing import Tuple, List, OrderedDict, Dict
import collections
from distutils.util import strtobool
from utils.file_utils import identical_to_backup, get_last_backup, is_dir_empty, get_filename


def parse_cutline(cutline: str, sep='\t') -> dict:
    """
    | processes each line of cuts into dictionary of cut options. with separator sep
    | For a cut range add each less/more than as separate cuts and add into same group
    |
    | name: name of cut to be printed and used in plot labels
    | cut_var: variable in root file to cut on
    | moreless: < or >
    | cut_val: value of cut on variable
    | suffix: suffix to be added onto plot names
    | group: each cut with same group label will be applied all at once.
    |        group labels will be printed in plot legends if sequential, as title if not.
    |        !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!
    | is_symmetric: either 'true' or false, take abs value of cut (eg for eta or phi)
    """
    cutline_split = cutline.split(sep)

    # if badly formatted
    if len(cutline_split) != 7:
        raise SyntaxError(f"Check cutfile. Line {cutline} is badly formatted.")

    name = cutline_split[0]
    cut_var = cutline_split[1]
    moreless = cutline_split[2]
    cut_val = float(cutline_split[3])
    suffix = cutline_split[4]
    group = cutline_split[5]
    is_symmetric = bool(strtobool(cutline_split[6].lower()))  # converts string to boolean

    # check values
    if moreless not in ('>', '<'):
        raise SyntaxError(f"Unexpected comparison operator: {cutline_split[2]}. Currently accepts '>' or '<'.")

    # fill dictionary
    cut_dict = {
        'name': name,
        'cut_var': cut_var,
        'moreless': moreless,
        'cut_val': cut_val,
        'suffix': suffix,
        'group': group,
        'is_symmetric': is_symmetric,
    }
    return cut_dict


def parse_cutfile(file: str, sep='\t') -> Tuple[List[dict], List[str], Dict[str, bool]]:
    """
    | generates pythonic outputs from input cutfile
    | Cutfile should be formatted with headers [CUTS], [OUTPUTS] and [OPTIONS]
    | Each line under [CUTS] header contains the 'sep'-separated values (detault: tab):
    | - name: name of cut to be printed and used in plot labels
    | - cut_var: variable in root file to cut on
    | - moreless: '<' or '>'
    | - cut_val: value of cut on variable
    | - suffix: suffix to be added onto plot names
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
            # ignore comments and blank lines
            if cutline.startswith('#') or len(cutline) < 2:
                continue

            # append cut dictionary
            cuts_list_of_dicts.append(parse_cutline(cutline, sep=sep))

        # get output variables
        output_vars_list = []
        for output_var in lines[lines.index('[OUTPUTS]') + 1: lines.index('[OPTIONS]')]:
            # ignore comments and blank lines
            if output_var.startswith('#') or len(output_var) < 2:
                continue

            # append cut dictionary
            output_vars_list.append(output_var)

        # global cut options
        options_dict = {}
        for option in lines[lines.index('[OPTIONS]') + 1:]:
            # ignore comments and blank lines
            if option.startswith('#') or len(option) < 2:
                continue

            option = option.split(sep)

            # options should be formatted as '[option]>sep>[value]'
            if len(option) != 2:
                raise Exception(f'Badly Formatted option {sep.join(option)}')

            # fill dict
            options_dict[option[0]] = bool(strtobool(option[1].lower()))  # converts string to boolean

            # Options necessary for the analysis to run (remember to add to this when adding new options)
            necessary_options = [
                'sequential',
            ]
            if missing_options := [opt for opt in necessary_options if opt not in options_dict.keys()]:
                raise Exception(f"Missing option(s) in cutfile: {', '.join(missing_options)}")

    return cuts_list_of_dicts, output_vars_list, options_dict


def extract_cut_variables(cut_dicts: List[dict], vars_list: List[str]) -> List[str]:
    """
    gets which variables are needed to extract from root file based on cutfile parser output
    uses outputs from parse_cutfile()
    """
    extract_vars = [cut_dict['cut_var'] for cut_dict in cut_dicts]
    return extract_vars + [variable for variable in vars_list if variable not in extract_vars]


# create cut_groups
def gen_cutgroups(cut_list_of_dicts: List[dict]) -> OrderedDict[str, List[str]]:
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


def if_make_cutfile_backup(current_cutfile: str, backup_dirpath: str) -> bool:
    """Decides if a backup cutfile should be made"""
    if not is_dir_empty(backup_dirpath):
        make_backup = not identical_to_backup(current_cutfile, backup_dir=backup_dirpath)
    else:
        make_backup = True
    return make_backup


def if_build_dataframe(current_cutfile: str,
                       will_backup_cutfile: bool,
                       backup_dirpath: str,
                       pkl_filepath: str
                       ) -> bool:
    """
    compares current cutfile to backups and decides whether to rebuild dataframe and save new backup cutfile
    :param current_cutfile: current cutfile
    :param will_backup_cutfile: whether or not the cutfile will be backed up
    :param backup_dirpath: path to dir of backups
    :param pkl_filepath: pickle file containing data in pandas dataframe
    :return: tuple of bools: (whether to rebuild dataframe, whether to save cutfile backup)
    """

    cut_list_dicts, vars_to_cut, options_dict = parse_cutfile(current_cutfile)

    is_pkl_file = os.path.isfile(pkl_filepath)
    if is_pkl_file:
        print("Datafile found")

    # default behaviour: don't build if you don't need to (it's slow and painful)
    build_dataframe = False

    # check if backup exists
    if not is_dir_empty(backup_dirpath):
        # if cutfiles are different, check if dataframe variables need an update
        if will_backup_cutfile:
            print("New cutfile, will save backup.")
            latest_backup = get_last_backup(backup_dirpath)

            # check if variables to extract from root file are the same as before. If yes, use previous pkl file.
            # if not, extract again from root file.
            BACKUP_cutfile_dicts, BACKUP_cutfile_outputs, _ = parse_cutfile(latest_backup)

            current_variables = extract_cut_variables(cut_list_dicts, vars_to_cut)
            backup_variables = extract_cut_variables(BACKUP_cutfile_dicts, BACKUP_cutfile_outputs)

            # if _cutfile contains different variables, extract necessary variables from root file and put into pickle
            # file for quicker read/write in pandas
            if not set(current_variables) == set(backup_variables):
                print(f"New variables found; dataframe will be rebuilt.\n"
                      f" Current cutfile variables: {current_variables}. Previous: {backup_variables}")
                build_dataframe = True

    # if backup doesn't exit, make backup and check if there is already a pickle file
    else:
        # if pickle file already exists
        if is_pkl_file:
            yn = input(f"No cutfile backups found in {backup_dirpath}. Continue with current pickle file? (y/n) ")
            while True:
                if yn.lower() in ('yes', 'y'):
                    print(f"Using dataframe {pkl_filepath}")
                    break
                elif yn.lower() in ('no', 'n'):
                    yn = input("Rebuild dataframe? (y/n) ")
                    while True:
                        if yn.lower() in ('no', 'n'):
                            sys.exit("Exiting")
                        elif yn.lower() in ('yes', 'y'):
                            build_dataframe = True
                            break
                        else:
                            yn = input("yes or no ")
                    break
                else:
                    yn = input("yes or no ")
        # if no backup or pickle file, rebuild
        else:
            print("No picke file found. Will rebuild dataframe")
            build_dataframe = True

    # check pickle file is actually there before trying to read from it
    if not build_dataframe and not is_pkl_file:
        yn = input(f"Pickle datafile not found at {pkl_filepath}. Rebuild? (y/n) ")
        while True:
            if yn.lower() in ('yes', 'y'):
                build_dataframe = True
                break
            elif yn.lower() in ('no', 'n'):
                sys.exit("Exiting")
            else:
                yn = input("yes or no ")

    return build_dataframe


def backup_cutfile(path: str, cutfile: str) -> None:
    curr_filename = get_filename(cutfile).rstrip('.txt')
    cutfile_backup_filepath = path + curr_filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    copyfile(cutfile, cutfile_backup_filepath)
    print(f"Backup _cutfile saved in {cutfile_backup_filepath}")
