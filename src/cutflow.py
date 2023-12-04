from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import List, Generator

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore

from src.cutfile import Cut
from src.logger import get_logger


@dataclass(slots=True)
class CutflowItem:
    npass: int
    eff: float
    ceff: float
    cut: Cut


class PCutflow:
    __slots__ = (
        "logger",
        "_n_events_tot",
        "cutflow_labels",
        "cutflow_ratio",
        "cutflow_n_events",
        "cutflow_a_ratio",
        "cutflow_cum",
        "cutflow_str",
        "_cuthist_options",
        "__first_reco_cut",
        "logger",
    )

    def __init__(self, df: pd.DataFrame, cuts: List[Cut], logger: logging.Logger | None = None):
        """
        Generates cutflow object that keeps track of various properties and ratios of selections made on given dataset

        :param df: Input analysis dataframe with boolean cut rows.
        :param cuts: Ordered dict of cuts made.
        :param logger: logger to output to
        """
        if missing_cuts := {cut.name for cut in cuts if "PASS_" + cut.name not in df.columns}:
            raise ValueError(f"Missing cut(s) {missing_cuts} in DataFrame")

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

        # generate cutflow
        self._n_events_tot = len(df.index)
        if (self._n_events_tot == 0) or (len(cuts) == 0):
            self.logger.error("Skipping cutflow; either no cuts or no events.")
            self.cutflow_labels = ["Inclusive"] + [cut_name for cut_name in cuts]
            n = len(self.cutflow_labels)
            self.cutflow_ratio = [1.0] * n
            self.cutflow_n_events = [0] * n
            self.cutflow_a_ratio = [1.0] * n
            self.cutflow_cum = [1.0] * n
            return

        # find first reco cut to separate reco and truth cuts in printout
        self.__first_reco_cut = ""
        for cut in cuts:
            if self.__first_reco_cut and not cut.is_reco:
                raise ValueError("Truth cut after reco cut!")
            elif not self.__first_reco_cut and cut.is_reco:
                self.__first_reco_cut = cut.name

        # list of cutflow labels (necessary for all cutflows)
        self.cutflow_labels = ["Inclusive"] + [cut.name for cut in cuts]
        self.cutflow_str = ["Inclusive"] + [cut.cutstr for cut in cuts]
        self.cutflow_ratio = [1.0]  # contains ratio of each separate cut to inclusive sample
        self.cutflow_n_events = [self._n_events_tot]  # contains number of events passing each cut
        self.cutflow_a_ratio = [1.0]  # contains ratio of each separate cut to inclusive sample
        self.cutflow_cum = [1.0]  # contains ratio of each cut to inclusive sample

        # extract only the cut columns from the dataframe
        df = df[[col for col in df.columns if "PASS_" in col]]

        # generate cutflow
        prev_n = self._n_events_tot  # saves the last cut in loop
        curr_cut_columns = []  # which cuts have already passed

        # loop over individual cuts
        for cut in self.cutflow_labels[1:]:
            curr_cut_columns += ["PASS_" + cut]
            # number of events passing current cut & all previous cuts
            n_events_left = len(df.loc[df[curr_cut_columns].all(1)].index)
            self.cutflow_n_events.append(n_events_left)

            if prev_n == 0:
                ratio = 0
            else:
                ratio = n_events_left / prev_n
            self.cutflow_ratio.append(ratio)

            self.cutflow_cum.append(n_events_left / self._n_events_tot)

            self.cutflow_a_ratio.append(
                len(df[df[["PASS_" + cut]].all(axis=1)].index) / self._n_events_tot
            )

            prev_n = n_events_left

        # assign histogram options for each type
        self._cuthist_options = {
            "ratio": {
                "filepath": "{}cutflow_ratio.png",
                "y_ax_vals": self.cutflow_ratio,
                "ylabel": "Cutflow ratio",
            },
            "cummulative": {
                "filepath": "{}cutflow_cummulative.png",
                "y_ax_vals": self.cutflow_cum,
                "ylabel": "Cummulative cutflow ratio",
            },
            "a_ratio": {
                "filepath": "{}cutflow_acceptance_ratio.png",
                "y_ax_vals": self.cutflow_a_ratio,
                "ylabel": "Cut ratio to accpetance",
            },
            "event": {
                "filepath": "{}cutflow.png",
                "y_ax_vals": self.cutflow_n_events,
                "ylabel": "Events",
            },
        }

        if logger.level == logging.DEBUG:
            self.print()

    def print(self, latex_path: Path | None = None) -> None:
        """Prints out cutflow table to terminal"""
        # lengths of characters needed to get everything to line up properly

        if latex_path:
            self.print_latex_table(latex_path)
            return None

        max_n_len = len(str(self._n_events_tot))
        max_name_len = max([len(cut) for cut in self.cutflow_str])

        # cutflow printout
        self.logger.info("")
        self.logger.info(f"=========== CUTFLOW =============")
        self.logger.info("---------------------------------")
        self.logger.info(
            "Cut "
            + " " * (max_name_len - 3)
            + "Events "
            + " " * (max_n_len - 6)
            + "Ratio A. Ratio Cum. Ratio"
        )
        # first line is inclusive sample
        self.logger.info(
            "Inclusive " + " " * (max_name_len - 9) + f"{self._n_events_tot} -     -        -"
        )

        # print line
        if not (self.cutflow_labels[1] == self.__first_reco_cut):
            # truth cuts
            self.logger.info("TRUTH:")
            self.logger.info("---------------------------------")
        for i, cutname in enumerate(self.cutflow_labels[1:]):
            if cutname == self.__first_reco_cut:
                self.logger.info("RECO:")
                self.logger.info("---------------------------------")
            self.logger.info(
                f"{self.cutflow_str[i + 1]:<{max_name_len}} "
                f"{self.cutflow_n_events[i + 1]:<{max_n_len}} "
                f"{self.cutflow_ratio[i + 1]:.3f} "
                f"{self.cutflow_a_ratio[i + 1]:.3f}    "
                f"{self.cutflow_cum[i + 1]:.3f}"
            )

        self.logger.info("")

    def print_latex_table(self, filepath: Path) -> None:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        with open(filepath, "w") as f:
            f.write("\\begin{tabular}{|c||c|c|c|}\n" "\\hline\n")
            f.write(
                f"Cut & Events & Ratio & Cumulative \\\\\\hline\n"
                f"Inclusive & {self.cutflow_n_events[0]} & — & — \\\\\n"
            )
            for i, cut_name in enumerate(self.cutflow_labels):
                f.write(
                    f"{cut_name} & "
                    f"{self.cutflow_n_events[i]} & "
                    f"{self.cutflow_ratio[i]:.3f} & "
                    f"{self.cutflow_cum[i]:.3f} "
                    f"\\\\\n"
                )
            f.write("\\hline\n\\end{tabular}\n")

    def print_histogram(self, out_path: Path, kind: str, plot_label: str = "", **kwargs) -> None:
        """
        Generates and saves a cutflow histogram

        :param kind: which cutflow type. options:
                    'ratio': ratio of cut to previous cut
                    'cummulative': ratio of all current cuts to acceptance
                    'a_ratio': ratio of only current cut to acceptance
                    'event': number of events passing through each cut
        :param out_path: plot output directory
        :param plot_label: plot title
        :param kwargs: keyword arguments to pass to plt.bar()
        :return: None
        """
        if kind not in self._cuthist_options.keys():
            raise Exception(
                f"Unknown cutflow histogram type {kind}. "
                f"Possible types: {', '.join(self._cuthist_options.keys())}"
            )
        fig, ax = plt.subplots()

        # plot
        ax.bar(
            x=self.cutflow_labels,
            height=self._cuthist_options[kind]["y_ax_vals"],
            color="w",
            edgecolor="k",
            width=1.0,
            **kwargs,
        )
        ax.set_xlabel("Cut")
        ax.tick_params(axis="x", which="both", bottom=False, top=False)  # disable ticks on x axis
        ax.set_xticks(
            ax.get_xticks()
        )  # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - BUG IN MATPLOTLIB 3.3.2
        ax.set_xticklabels(labels=self.cutflow_labels, rotation="-40")
        ax.set_ylabel(self._cuthist_options[kind]["ylabel"])
        hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=plot_label)
        ax.grid(b=True, which="both", axis="y", alpha=0.3)

        filepath = self._cuthist_options[kind]["filepath"].format(out_path)
        fig.savefig(filepath)
        self.logger.info(f"Cutflow histogram saved to {filepath}")


@dataclass(slots=True)
class RCutflow:
    _cutflow: List[CutflowItem] = field(init=False, default_factory=list)
    logger: Logger = field(default_factory=get_logger)

    def __getitem__(self, idx: int) -> CutflowItem:
        return self._cutflow[idx]

    def __iter__(self) -> Generator[CutflowItem, None, None]:
        yield from self._cutflow

    def __len__(self) -> int:
        return len(self._cutflow)

    def __add__(self, other: RCutflow):
        return deepcopy(self).__iadd__(other)

    def __iadd__(self, other: RCutflow):
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
        if self._cutflow is None:
            raise AttributeError("Must generated cutflow in order to obtain number of events")
        return self._cutflow[0].npass

    def gen_cutflow(self, rdf: ROOT.RDataFrame, cuts: List[Cut]) -> None:
        """
        Generate cutflow - forces an event loop to run if not already.

        :param rdf: filtered root dataframe
        :param cuts: List of Cut objects
        """
        report = rdf.Report()
        self._cutflow = [
            CutflowItem(
                npass=report.At(cut.name).GetPass(),
                eff=report.At(cut.name).GetEff(),
                ceff=0.0,  # calculate this next
                cut=cut,
            )
            for cut in cuts
        ]
        self._cutflow = [
            CutflowItem(
                npass=report.At("Inclusive").GetAll(),
                eff=100,
                ceff=100,
                cut=Cut("Inclusive", cutstr="-"),
            )
        ] + self._cutflow
        for i in range(1, len(self._cutflow)):
            self._cutflow[i].ceff = 100 * self._cutflow[i].npass / self._cutflow[0].npass

    def import_cutflow(self, hist: ROOT.TH1I, cuts: List[Cut]) -> None:
        """Import cutflow from histogram and cutfile"""
        self._cutflow = cutflow_from_hist_and_cuts(hist, cuts)

    def print(self, latex_path: Path | None = None) -> None:
        if latex_path:
            cut_list_truth = [
                [
                    cutflowitem.cut.name,
                    int(cutflowitem.npass),
                    f"{cutflowitem.eff:.3G} \\%",
                    f"{cutflowitem.ceff:.3G} \\%",
                ]
                for cutflowitem in self._cutflow
                if (not cutflowitem.cut.is_reco and cutflowitem.cut.name.lower != "inclusive")
            ]
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
                if (cutflowitem.cut.is_reco and cutflowitem.cut.name.lower != "inclusive")
            ]
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

    def gen_histogram(self) -> ROOT.TH1I:
        """return cutflow histogram"""
        n_items = len(self._cutflow)
        if n_items == 0:
            raise AttributeError("Cutflow empty or uninitialised. Try running gen_cutflow()")

        hist = ROOT.TH1I("cutflow", "cutflow", n_items, 0, n_items)

        for i, cut_item in enumerate(self._cutflow):
            hist.SetBinContent(i + 1, cut_item.npass)
            hist.GetXaxis().SetBinLabel(i + 1, cut_item.cut.name)

        return hist


def cutflow_from_hist_and_cuts(hist: ROOT.TH1I, cuts: List[Cut]) -> List[CutflowItem]:
    """Create cutflow object from cutflow histogram"""
    if hist.GetNbinsX() - 1 != len(cuts):
        raise ValueError(
            f"Number of cuts in cutfile ({len(cuts)}) does not match number of cuts in histogram ({hist.GetNbinsX() - 1})"
        )

    cutflow = [
        CutflowItem(
            npass=hist.GetBinContent(1),
            eff=100,
            ceff=100,
            cut=Cut("Inclusive", "-", set(), set(), False),
        )
    ]

    # indexing is fucked b/c ROOT histograms are 1-indexed and 1st bin is inclusive,
    # so the indexing goes:
    # idx  cutfile  hist
    # ------------------
    # 0    cut1
    # 1    cut2     inc.
    # 2             cut1
    # 3             cut2
    for i, cut in enumerate(cuts):
        npass = hist.GetBinContent(i + 2)
        try:
            eff = 100 * npass / hist.GetBinContent(i + 1)
        except ZeroDivisionError:
            eff = np.NAN
        try:
            ceff = 100 * npass / hist.GetBinContent(1)
        except ZeroDivisionError:
            ceff = np.NAN

        cutflow.append(CutflowItem(npass=hist.GetBinContent(i + 2), eff=eff, ceff=ceff, cut=cut))

    return cutflow
