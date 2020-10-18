import sys
import os
from filecmp import cmp
from glob import glob
from typing import Tuple, List, OrderedDict
import collections


def parse_cutlines(cutline: str, sep='\t') -> dict:
    """
    processes each line of cuts into dictionary of cut options. with separator sep
    For a cut range add each less/more than as separate cuts and add into same group

    name: name of cut to be printed and used in plot labels
    cut_var: variable in root file to cut on
    moreless: < or >
    cut_val: value of cut on variable
    suffix: suffix to be added onto plot names
    group: each cut with same group label will be applied all at once.
           group labels will be printed in plot legends if sequential, as title if not.
           !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!
    is_symmetric: either 'true' or false, take abs value of cut (eg for eta or phi)
    """
    cutline_split = cutline.split(sep)

    # if badly formatted
    if len(cutline_split) != 7:
        raise Exception(f"Check cutfile. Line {cutline} is badly formatted.")

    # check values
    if cutline_split[2] not in ('>', '<'):
        raise ValueError(f"Unexpected comparison operator: {cutline_split[2]}. Currently accepts '>' or '<'.")

    # fill dictionary
    cut_dict = {
        'name': cutline_split[0],
        'cut_var': cutline_split[1],
        'moreless': cutline_split[2],
        'cut_val': float(cutline_split[3]),
        'suffix': cutline_split[4],
        'group': cutline_split[5],
        'is_symmetric': True if cutline_split[6].lower() == 'true' else False,
    }
    return cut_dict


def parse_cutfile(file: str, sep='\t') -> Tuple[List[dict], List[str], dict]:
    """
    generates pythonic outputs from input cutfile
    Cutfile should be formatted with headers [CUTS], [OUTPUTS] and [OPTIONS]

    Each line under [CUTS] header contains the 'sep'-separated values (detault: tab):
    - name: name of cut to be printed and used in plot labels
    - cut_var: variable in root file to cut on
    - moreless: '<' or '>'
    - cut_val: value of cut on variable
    - suffix: suffix to be added onto plot names
    - group: each cut with same group number will be applied all at once.
             !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!

    Each line under [OUTPUTS] should be a variable in root file

    Each line under [OPTIONS] header should be '[option]<sep>[value]'
    - sequential: (bool) whether each cut should be applied sequentially so a cutflow can be generated
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
            cuts_list_of_dicts.append(parse_cutlines(cutline, sep=sep))

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

            # convert booleans
            if option[1].lower() == 'true':
                option[1] = True
            elif option[1].lower() == 'false':
                option[1] = False
            else:
                raise ValueError(f"Option '{option[0]}' should be true or false (case insensitive)")

            # fill dict
            options_dict[option[0]] = option[1]

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


def compare_backup(current_cutfile: str, backup_filepath: str, pkl_filepath: str) -> Tuple[bool, bool]:
    """
    compares current cutfile to backups and decides whether to rebuild dataframe and save new backup cutfile
    :param current_cutfile: current cutfile
    :param backup_filepath: path to dir of backups
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
    if len(os.listdir(backup_filepath)) != 0:
        print("Cutfile backups found")
        latest_backup = max(glob(backup_filepath + '*'), key=os.path.getctime)
        print(f"Latest backup: {latest_backup}")

        # make backup if new cutfile
        make_backup = not cmp(current_cutfile, latest_backup)

        # if cutfiles are different, check if dataframe variables need an update
        if make_backup:
            print("New cutfile, will save backup.")
            # check if variables to extract from root file are the same as before. If yes, use previous pkl file.
            # if not, extract again from root file.
            BACKUP_cutfile_dicts, BACKUP_cutfile_outputs, _ = parse_cutfile(latest_backup)

            current_variables = extract_cut_variables(cut_list_dicts, vars_to_cut)
            backup_variables = extract_cut_variables(BACKUP_cutfile_dicts, BACKUP_cutfile_outputs)

            # if cutfile contains different variables, extract necessary variables from root file and put into pickle
            # file for quicker read/write in pandas
            if not set(current_variables) == set(backup_variables):
                build_dataframe = True

    # if backup doesn't exit, make backup and check if there is already a pickle file
    else:
        # make backup
        make_backup = True
        # if pickle file already exists
        if is_pkl_file:
            yn = input(f"No cutfile backups found in {backup_filepath}. Continue with current pickle file? (y/n) ")
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
                            print("Rebuilding dataframe...")
                            break
                        else:
                            yn = input("yes or no ")
                    break
                else:
                    yn = input("yes or no ")
        # if no backup or pickle file, rebuild
        else:
            build_dataframe = True
            print("Building dataframe...")

    # check pickle file is actually there before trying to read from it
    if not build_dataframe and not is_pkl_file:
        yn = input(f"Pickle datafile not found at {pkl_filepath}. Rebuild? (y/n) ")
        while True:
            if yn.lower() in ('yes', 'y'):
                build_dataframe = True
                print("Rebuilding dataframe...")
                break
            elif yn.lower() in ('no', 'n'):
                sys.exit("Exiting")
            else:
                yn = input("yes or no ")

    return build_dataframe, make_backup


if __name__ == '__main__':
    cutfile = '../options/cutfile.txt'
    cuts, outputs, options = parse_cutfile(cutfile)
    print(cuts)
    print(outputs)
    print(options)
