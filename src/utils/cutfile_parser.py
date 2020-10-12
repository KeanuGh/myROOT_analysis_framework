# file containing cuts
cutfile = '../options/cutfile.txt'


def parse_cutlines(cutline: str) -> dict:
    """
    processes each line of cuts into dictionary of cut options.
    For a cut range add each less/more than as separate cuts and add into same group

    name: name of cut to be printed and used in plot labels
    cut_var: variable in root file to cut on
    moreless: < or >
    cut_val: value of cut on variable
    suffix: suffix to be added onto plot names
    group: each cut with same group number will be applied all at once.
           !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!
    """
    cutline = cutline.split('\t')

    # if badly formatted
    if len(cutline) != 6:
        raise Exception(f"Check cutfile. Line {cutline} is badly formatted.")

    # fill dictionary
    cut_dict = {
        'name': cutline[0],
        'cut_var': cutline[1],
        'moreless': cutline[2],
        'cut_val': float(cutline[3]),
        'suffix': cutline[4],
        'group': int(cutline[5]),
    }
    return cut_dict


def parse_cutfile(file: str):
    """
    generates pythonic outputs from input cutfile
    Cutfile should be formatted with headers [CUTS], [OUTPUTS] and [OPTIONS]

    Each line under [CUTS] header contains the tab-separated values:
    - name: name of cut to be printed and used in plot labels
    - cut_var: variable in root file to cut on
    - moreless: '<' or '>'
    - cut_val: value of cut on variable
    - suffix: suffix to be added onto plot names
    - group: each cut with same group number will be applied all at once.
             !!!SUFFIXES FOR CUTS IN GROUP MUST BE THE SAME!!!

    Each line under [OUTPUTS] should be a variable in root file

    Each line under [OPTIONS] header should be '[option]\t[value]'
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
            cuts_list_of_dicts.append(parse_cutlines(cutline))

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
        for option in lines[lines.index('[OPTIONS]')+1:]:
            # ignore comments and blank lines
            if option.startswith('#') or len(option) < 2:
                continue

            option = option.split('\t')

            # options should be formatted as '[option]\t[value]'
            if len(option) != 2:
                raise Exception(f'Badly Formatted option {option}')

            # convert booleans
            if option[1].lower() == 'true':
                option[1] = True
            elif option[1].lower() == 'false':
                option[1] = False

            # fill dict
            options_dict[option[0]] = option[1]

    return cuts_list_of_dicts, output_vars_list, options_dict


if __name__ == '__main__':
    cuts, outputs, options = parse_cutfile(cutfile)
    print(cuts)
    print(outputs)
    print(options)
