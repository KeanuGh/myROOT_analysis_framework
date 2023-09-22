import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Set, List

from src.logger import get_logger
from utils.file_utils import get_filename
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data, VarTag

# all variables known by this framework
all_known_vars = set(derived_vars.keys()) | set(variable_data.keys())


@dataclass
class Cut:
    """Cut class containing info for each cut"""

    name: str
    cutstr: str = ""
    var: Set[str] = field(default_factory=set)
    tree: Set[str] = field(default_factory=set)
    is_reco: bool = False

    def __str__(self) -> str:
        return f"{self.name}: {self.cutstr}"


class Cutfile:
    """Handles importing cutfiles and extracting variables"""

    __slots__ = (
        "sep",
        "logger",
        "name",
        "_path",
        "given_tree",
        "cuts",
        "all_vars",
        "output_vars",
        "tree_dict",
        "vars_to_calc",
    )

    def __init__(
        self,
        file_path: str | Path,
        default_tree: str | Set[str] = "0:NONE",
        logger: logging.Logger | None = None,
        sep="\t",
    ):
        """
        Read and pull variables and cuts from cutfile.

        :param file_path: cutfile
        :param default_tree: name of TTree or list of names to assume if not given in file path. Default value is to
                         avoid overlap with any possible TTree names
        :param logger: logger to output messages to. Will default to console output if not given
        :param sep: separator for values in cutfile. Default is TAB
        """
        self.sep = sep
        self.logger = logger if logger is not None else get_logger()
        self.name = get_filename(file_path)
        self._path = Path(file_path)
        # make sure the default tree is a set of strings
        if isinstance(default_tree, str):
            self.given_tree = {default_tree}
        else:
            self.given_tree = set(default_tree)
        self.cuts, self.output_vars = self.parse_cutfile()

        # make sure truth cuts always come first
        recoflag = False
        for cut in self.cuts:
            if recoflag and not cut.is_reco:
                raise ValueError(f"Truth cut after reco cut!\n\t{cut.name}: {cut.cutstr}")
            elif cut.is_reco:
                recoflag = True

        self.tree_dict, self.vars_to_calc = self.extract_variables()
        self.all_vars = self._all_vars()

    def __repr__(self):
        return f'Cutfile("{self._path}")'

    def parse_cut(self, cutline: str) -> Cut:
        """Processes each line of cuts into dictionary of cut options. with separator <sep>"""
        # strip trailing and leading spaces
        cutline_split = [i.strip() for i in cutline.split(self.sep)]

        # if badly formatted
        if len(cutline_split) not in (2, 3):
            raise SyntaxError(
                f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}."
            )
        for v in cutline_split:
            if len(v) == 0:
                raise SyntaxError(
                    f"Check cutfile. Blank value given in line {cutline}. Got {cutline_split}."
                )

        name = cutline_split[0]
        cut_str = cutline_split[1]

        try:
            tree_value = cutline_split[2]  # if alternate TTree(s) are given
            # if multiple, separate (assuming separated by a space
            tree = {tree for tree in tree_value.split()}
        except IndexError:
            tree = self.given_tree

        if tree == "":
            raise SyntaxError(
                f"Check cutfile. Line {cutline} is badly formatted. Got {cutline_split}."
            )

        # check for variables in the cut string
        split_string = set(re.findall(r"\w+|[.,!?;]", cut_str))
        cutvars = all_known_vars & split_string  # find all known variables in cutstring

        if len(cutvars) == 1:
            is_reco = variable_data[next(iter(cutvars))]["tag"] == VarTag.RECO

        elif len(cutvars) > 1:  # make sure all variables have the same tag (truth or reco)
            tags = {variable_data[v]["tag"] for v in cutvars}
            if VarTag.META in tags:
                raise Exception(f"Meta variable cut {cutline}")
            elif len(tags) > 1:
                raise Exception(f"Mixing reco/truth variables in cut {cutline}")
            else:
                is_reco = VarTag.RECO in tags

        else:
            raise ValueError(
                f"No known variable in string '{cut_str}' for line '{cutline}'\n"
                f"Read {cutline_split}"
            )

        return Cut(name=name, cutstr=cut_str, var=cutvars, tree=tree, is_reco=is_reco)

    def parse_cutfile(
        self, path: str | Path | None = None, sep="\t"
    ) -> Tuple[List[Cut], Dict[str, Set[str]]]:
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

        with open(path, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

            if "[CUTS]" not in lines:
                raise ValueError("Missing [CUTS] section!")
            if "[OUTPUTS]" not in lines:
                raise ValueError("Missing [OUTPUTS] section!")

            # get cut lines
            cuts: List[Cut] = []
            for cutline in lines[lines.index("[CUTS]") + 1 : lines.index("[OUTPUTS]")]:
                if cutline.startswith("#") or len(cutline) < 2:
                    continue
                cut = self.parse_cut(cutline)
                cuts.append(cut)

            # get output variables
            output_vars: Dict[str, Set[str]] = dict()
            for output_var in lines[lines.index("[OUTPUTS]") + 1 :]:
                if output_var.startswith("#") or len(output_var) < 2:
                    continue

                var_tree = [i.strip() for i in output_var.split(sep)]
                if len(var_tree) > 2:
                    raise SyntaxError(
                        f"Check line '{output_var}'. Should be variable and tree (optional). "
                        f"Got '{var_tree}'."
                    )
                elif len(var_tree) == 2:
                    out_var, tree = var_tree
                    output_vars[out_var] = {tree}
                elif len(var_tree) == 1:
                    out_var = var_tree[0]
                    output_vars[out_var] = self.given_tree
                else:
                    raise Exception("This should never happen")

        return cuts, output_vars

    def extract_variables(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Get which variables are needed to extract from root file based on cutfile parser output
        uses outputs from parse_cutfile()

        :return Tuple[{Dictionary of trees and its variables to extract}, {set of variables to calculate}]
        """
        # generate initial dict with given (default) TTree(s)
        tree_dict: Dict[str, set] = dict()
        for given_tree in self.given_tree:
            tree_dict[given_tree] = set()
        extracted_vars = dict()  # keep all extracted variables here

        for cut in self.cuts:
            trees = cut.tree

            cut_variables = cut.var
            for var in cut_variables:
                extracted_vars[var] = trees

            for tree in trees:
                if tree not in tree_dict:
                    tree_dict[tree] = set()
                tree_dict[tree] |= cut_variables

        # add variables not cut on to tree dict
        for var, trees in self.output_vars.items():
            if var in extracted_vars:
                extracted_vars[var] |= trees
            else:
                extracted_vars[var] = trees
            for tree in trees:
                if tree in tree_dict:
                    tree_dict[tree] |= {var}
                else:
                    tree_dict[tree] = {var}

        # work out which variables to calculate and which to extract from ROOT file
        calc_vars = [  # cut variables that are in derived vars
            (var, trees) for var, trees in extracted_vars.items() if var in derived_vars
        ]

        # remove variables to calculate from tree dict
        for tree in tree_dict:
            tree_dict[tree] -= {var for var, _ in calc_vars}

        # add any variables needed from which trees for calculating derived variables
        for calc_var, trees in calc_vars:
            for tree in trees:
                if tree == self.given_tree and derived_vars[calc_var]["tree"]:
                    tree = derived_vars[calc_var]["tree"]

                if tree in tree_dict:
                    tree_dict[tree] |= set(derived_vars[calc_var]["var_args"])
                else:
                    tree_dict[tree] = set(derived_vars[calc_var]["var_args"])

        # only return the actual variable names to calculate. Which tree to extract from will be handled by tree_dict
        return tree_dict, {var for var, _ in calc_vars}

    def _all_vars(self) -> Set[str]:
        """Return all variables mentioned in cutfile"""
        all_vars_set = set()
        for cut in self.cuts:
            all_vars_set.update(cut.var)
        for var in self.vars_to_calc:
            all_vars_set.update(set(derived_vars[var]["var_args"]))
        return all_vars_set | set(self.output_vars.keys()) | self.vars_to_calc

    @staticmethod
    def truth_reco(tree_dict: Dict[str, Set[str]]) -> Tuple[bool, bool]:
        """Does cutfile ask for truth data? Does cutfile ask for reco data?"""
        is_truth = is_reco = False
        for var_ls in tree_dict.values():
            for var in var_ls:
                if var in variable_data:
                    if variable_data[var]["tag"] == "truth":
                        is_truth = True
                    elif variable_data[var]["tag"] == "reco":
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
            return "None"

        # get cut dict with name cut_label
        cut = next((c for c in self.cuts if c.name == cut_label), None)
        if cut is None:
            raise ValueError(f"No cut named '{cut_label}' in cutfile {self.name}")

        name_len = max([len(cut.name) for cut in self.cuts]) if align else 0

        return (f"{cut.name:<{name_len}}: " if name else "") + cut.cutstr

    def cut_exists(self, cut_name: str) -> bool:
        """check if cut exists in cutfile"""
        return cut_name in self.cuts

    def log_cuts(self, name: bool = True, debug: bool = False) -> None:
        """send list of cuts in cutfile to logger"""
        for cut in self.cuts:
            if debug:
                self.logger.debug(self.get_cut_string(cut.name, name=name, align=True))
            else:
                self.logger.info(self.get_cut_string(cut.name, name=name, align=True))
