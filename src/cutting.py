"""
Module containing classes related to cutting/filtering:
    - Cut: defining a single cut
    - FilterNode: defining one computational filtering node on an RDataFrame
    - Cutflow: defining the flow of data through filters
    - Cutfile: (depreciated) for importing lists of cuts from a file (just do these in code - much simpler)
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import List, Generator
from typing import Tuple, Dict, Set

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore

from src.logger import get_logger
from utils.file_utils import get_filename
from utils.var_helpers import derived_vars
from utils.variable_names import variable_data, VarTag

# all variables known by this framework
_all_known_vars: set[str] = set(derived_vars.keys()) | set(variable_data.keys())


@dataclass(slots=True)
class Cut:
    """Cut class containing info for each cut"""

    name: str
    cutstr: str = ""
    included_variables: Set[str] = field(default_factory=set)
    tree: Set[str] = field(default_factory=set)
    is_reco: bool = False

    def __post_init__(self):
        # check for variables in the cut string
        split_string = set(re.findall(r"\w+|[.,!?;]", self.cutstr))
        # find all known variables in cutstring
        self.included_variables = _all_known_vars & split_string

        if len(self.included_variables) == 1:
            self.is_reco = variable_data[next(iter(self.included_variables))]["tag"] == VarTag.RECO

        elif len(self.included_variables) > 1:
            # make sure all variables have the same tag (truth or reco)
            tags = {variable_data[v]["tag"] for v in self.included_variables}
            if VarTag.META in tags:
                raise Exception(f"Meta variable cut {self.name}")
            elif len(tags) > 1:
                raise Exception(f"Mixing reco/truth variables in cut {self.name}")
            else:
                self.is_reco = VarTag.RECO in tags

        else:
            raise ValueError(f"No known variable in string '{self.cutstr}' for cut: '{self.name}'")

    def __str__(self) -> str:
        return f"{self.name}: {self.cutstr}"


@dataclass(slots=True, repr=False)
class FilterNode:
    """
    Single filter node of an RDataFrame computation graph.
    Only the root node can be instantiated
    """

    df: ROOT.RDF.RInterFace
    cut: Cut | None = None
    parent: FilterNode | None = None
    children: list[FilterNode] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if (self.parent is not None) and (self.cut is None):
            raise AttributeError("Only the root node can have no cut")

    def __repr__(self) -> str:
        n_parents = len(self.get_parents())
        n_children = len(self.children)
        return f"FilterNode(cut={self.cut},n_parents={n_parents},n_children={n_children})"

    def __len__(self) -> int:
        return len(self.get_chain())

    def add_child(self, child: FilterNode) -> None:
        """Add child"""
        self.children.append(child)

    def create_child(self, cut: Cut) -> FilterNode:
        """
        Create child node from this node.
        Creates node, adds to children of current node, then returns new child node
        """
        child_node = FilterNode(
            self.df.Filter(cut.cutstr, self._sanitise_str(cut.name)), cut=cut, parent=self
        )
        self.add_child(child_node)
        return child_node

    def create_branch(self, cuts: list[Cut]) -> FilterNode:
        """
        Create branch originating from current node from list of filters.
        """
        node = self
        for cut in cuts:
            node = node.create_child(cut)
        return node

    def get_children_names(self) -> list[str]:
        """Return list of child names"""
        return [child.name for child in self.children]

    def get_parents(self) -> list[FilterNode]:
        """Return list of parent chain from this node backwards"""
        node = self.parent
        parents = []
        while node is not None:
            parents.append(node)
            node = node.parent
        parents.reverse()
        return parents

    def get_chain(self) -> list[FilterNode]:
        """Get full chain from root to here"""
        return self.get_parents() + [self]

    def get_cuts(self) -> list[Cut]:
        """Return list of cuts in current chain"""
        return [
            parent.cut
            for parent in self.get_chain()
            if parent.cut is not None  # don't include root
        ]

    def has_cuts(self, cuts: list[Cut]) -> bool:
        """Return whether passed list of cuts matches cuts currently in node"""
        return cuts == [parent.cut for parent in self.get_chain()]

    @staticmethod
    def _sanitise_str(string: str) -> str:
        """sanitise latex-like string to stop ROOT from interpreting them as escape sequences"""
        return string.replace("\\", "\\\\")


class FilterTree:
    """
    Implementation of a RDataFrame computational graph.
    Pass RDataFrame to generate a root node.
    filter_tree[name] to access leaves of the tree (these are the end-point selections).
    """

    __slots__ = ("root", "leaves")

    def __init__(self, df: ROOT.RDataFrame):
        self.root = FilterNode(df)
        self.leaves: dict[str, FilterNode] = dict()

    def __getitem__(self, item) -> FilterNode:
        return self.leaves[item]

    def __len__(self) -> int:
        return len(self.leaves)

    def __repr__(self) -> str:
        return self.tree_string_repr()

    def get_paths(self):
        """Get all paths through the tree"""
        return self._get_subtree_paths(self.root)

    def _get_subtree_paths(self, node: FilterNode, path=None):
        if path is None:
            path = []
        paths = [path + [node.name]]
        for child in node.children:
            paths.extend(self._get_subtree_paths(child, path + [node.name]))
        return paths

    def add_path(self, selection: list[Cut], name: str) -> None:
        """Adds a branch to the tree taking into account location"""
        if not all(isinstance(cut, Cut) for cut in selection):
            raise ValueError("Path must be a list of strings")

        # look for last matching node
        current_node = self.root
        cut_idx = 0
        for i in range(len(selection)):
            child_nodes = current_node.children
            wanted_cut = selection[i]

            # look through child nodes for a matching node
            node_found = False
            for child_node in child_nodes:
                if wanted_cut == child_node.cut:
                    current_node = child_node
                    cut_idx = i + 1
                    node_found = True
                    break

            if not node_found:
                break

        # create branch
        self.leaves[name] = current_node.create_branch(selection[cut_idx:])

    def add_leaf(self, cut: Cut, name: str, from_selection: str | None = None) -> None:
        """
        Create new leaf. Either from the root node or from an existing selection.
        Does not remove existing selection
        """
        if from_selection:
            try:
                new_node = self.leaves[from_selection].create_child(cut)
            except KeyError as e:
                raise KeyError(f"No selection '{from_selection}' in dataset '{self.name}'") from e
        else:
            new_node = self.root.create_child(cut)

        self.leaves[name] = new_node

    def generate_tree(self, selections: dict[str, list[Cut]]) -> None:
        """Generate linked filter graphs from dictionary of selections"""
        for selection_name, selection in selections.items():
            self.add_path(selection, selection_name)

    def tree_string_repr(self) -> str:
        """Return string representation of current tree"""
        tree_str = "root"
        for child in self.root.children:
            tree_str += self._subtree_string_repr(child, prefix="\t")
        return tree_str

    def _subtree_string_repr(self, node: FilterNode, prefix: str = "") -> str:
        """Return string representation of subtree"""
        subtree_str = f"\n{prefix}{node.cut.name}"
        if not node.children:
            return subtree_str
        else:
            for child in node.children:
                subtree_str += self._subtree_string_repr(child, prefix=f"{prefix}\t")
            return subtree_str

    def find_leaf_from_node(self, node: FilterNode) -> str:
        """Find name of leaf from node. if node isn't a leaf, return empty string"""
        try:
            return list(self.leaves.keys())[list(self.leaves.values()).index(node)]
        except KeyError:
            return ""


@dataclass(slots=True)
class CutflowItem:
    """
    Represents a single cut of a cutflow
    """

    npass: int
    eff: float
    ceff: float
    cut: Cut


@dataclass(slots=True)
class Cutflow:
    """
    Cutflow object. `_cutflow` attribute contains list of cutflow items that defines a cutflow
    """

    _cutflow: List[CutflowItem] = field(init=False, default_factory=list)
    logger: Logger = field(default_factory=get_logger)

    def __getitem__(self, idx: int) -> CutflowItem:
        return self._cutflow[idx]

    def __iter__(self) -> Generator[CutflowItem, None, None]:
        yield from self._cutflow

    def __len__(self) -> int:
        return len(self._cutflow)

    def __add__(self, other: Cutflow):
        return deepcopy(self).__iadd__(other)

    def __iadd__(self, other: Cutflow):
        """Sum individual cutflow values (the only important part), leave the rest alone"""

        # cuts must be identical if cutflows are to be merged
        if (self_names := [item.cut for item in self]) != (
            other_names := [item.cut for item in other]
        ):
            raise ValueError(
                "Cuts are not equivalent and cannot be summed. "
                "Got:\n{}".format(
                    "\n".join("{}\t{}".format(x, y) for x, y in zip(self_names, other_names))
                )
            )

        # do inclusive separately
        npass_inc = self[0].npass + other[0].npass
        self._cutflow[0] = CutflowItem(npass=npass_inc, eff=100, ceff=100, cut=self[0].cut)

        npass = npass_inc
        for i in range(1, len(self)):
            self_item = self[i]
            other_item = other[i]

            npass_prev = npass
            npass = self_item.npass + other_item.npass
            self._cutflow[i].npass = npass

            # recalculate efficiencies
            try:
                self._cutflow[i].eff = 100 * npass / npass_prev
            except ZeroDivisionError:
                self._cutflow[i].eff = np.nan
            try:
                self._cutflow[i].ceff = 100 * npass / npass_inc
            except ZeroDivisionError:
                self._cutflow[i].eff = np.nan

        return self

    @property
    def total_events(self) -> int:
        """Return total number of events passed to cutflow"""
        if self._cutflow is None:
            raise AttributeError("Must have generated cutflow in order to obtain number of events")
        return self._cutflow[0].npass

    def gen_cutflow(self, filter_node: FilterNode) -> None:
        """
        Generate cutflow - forces an event loop to run if not already.

        :param filter_node: node of filtered root dataframe
        """
        report = filter_node.df.Report()
        cuts = filter_node.get_cuts()

        if self.logger.level <= logging.DEBUG:
            self.logger.debug(
                "Full report (internal):\n%s",
                "\n".join([f"{cut.name}: {report.At(cut.name).GetPass()}" for cut in cuts]),
            )

        self._cutflow = [
            CutflowItem(
                npass=report.At(cut.name).GetPass(),
                eff=report.At(cut.name).GetEff(),
                ceff=0.0,  # calculate this next
                cut=cut,
            )
            for cut in cuts
        ]
        for i in range(1, len(self._cutflow)):
            try:
                self._cutflow[i].ceff = 100 * self._cutflow[i].npass / self._cutflow[0].npass
            except ZeroDivisionError:
                self._cutflow[i].ceff = np.nan

    def import_cutflow(self, hist: ROOT.TH1I, cuts: List[Cut]) -> None:
        """Import cutflow from histogram and cutfile"""
        if hist.GetNbinsX() - 1 != len(cuts):
            raise ValueError(
                f"Number of cuts passed to cutflow ({len(cuts)}) "
                f"does not match number of cuts in histogram ({hist.GetNbinsX() - 1})"
            )

        # indexing is fucked b/c ROOT histograms are 1-indexed,
        # so the indexing goes:
        # idx  cutflow  hist
        # ------------------
        # 0    cut1
        # 1    cut2     cut1
        # 2             cut2
        cutflow: list[CutflowItem] = []
        for i, cut in enumerate(cuts):
            npass = hist.GetBinContent(i + 1)
            try:
                eff = 100 * npass / hist.GetBinContent(i)
            except ZeroDivisionError:
                eff = np.NAN
            try:
                ceff = 100 * npass / hist.GetBinContent(1)
            except ZeroDivisionError:
                ceff = np.NAN

            cutflow.append(CutflowItem(npass=npass, eff=eff, ceff=ceff, cut=cut))

        self._cutflow = cutflow

    def print(self, latex_path: Path | None = None) -> None:
        """Print cutflow table to console or as latex table to latex file"""
        if latex_path:
            cut_list_truth = [
                [
                    cutflowitem.cut.name,
                    int(cutflowitem.npass),
                    f"{cutflowitem.eff:.3G} \\%",
                    f"{cutflowitem.ceff:.3G} \\%",
                ]
                for cutflowitem in self._cutflow
                if (not cutflowitem.cut.is_reco)
            ]
            if cut_list_truth:
                cut_list_truth[0][0] = r"\hline " + cut_list_truth[0][0]
                cut_list_truth.insert(0, [r"\hline Particle-Level", "", "", ""])

            cut_list_reco = [
                [
                    cutflowitem.cut.name,
                    int(cutflowitem.npass),
                    f"{cutflowitem.eff:.3G} \\%",
                    f"{cutflowitem.ceff:.3G} \\%",
                ]
                for cutflowitem in self._cutflow
                if cutflowitem.cut.is_reco
            ]
            if cut_list_reco:
                cut_list_reco[0][0] = r"\hline " + cut_list_reco[0][0]
                cut_list_reco.insert(0, [r"\hline Detector-Level", "", "", ""])

            table = tabulate(
                cut_list_truth + cut_list_reco,
                headers=["name", "npass", "eff", "cum. eff"],
                tablefmt="latex_raw",
            )

            with open(latex_path, "w") as f:
                f.write(table)

        else:
            table = tabulate(
                [
                    [
                        cut_item.cut.name,
                        int(cut_item.npass),
                        f"{cut_item.eff:.3G} %",
                        f"{cut_item.ceff:.3G} %",
                    ]
                    for cut_item in self._cutflow
                ],
                headers=["name", "npass", "eff", "cum. eff"],
                tablefmt="simple",
            )
            self.logger.info(table)

    def gen_histogram(self, name: str = "") -> ROOT.TH1I:
        """return cutflow histogram"""
        n_items = len(self._cutflow)
        if n_items == 0:
            raise AttributeError("Cutflow empty or uninitialised. Try running gen_cutflow()")

        name = ("cutflow_" + name) if name else "cutflow"
        hist = ROOT.TH1I(name, name, n_items, 0, n_items)

        for i, cut_item in enumerate(self._cutflow):
            hist.SetBinContent(i + 1, cut_item.npass)
            hist.GetXaxis().SetBinLabel(i + 1, cut_item.cut.name)

        return hist


class Cutfile:
    """
    Handles importing cutfiles and extracting variables

    DEPRECIATED - DO NOT USE (will delete in future)
    """

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

        return Cut(name=name, cutstr=cut_str)

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

            cut_variables = cut.included_variables
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
            all_vars_set.update(cut.included_variables)
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
