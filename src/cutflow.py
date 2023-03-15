import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import List, Generator

import ROOT  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mplhep as hep  # type: ignore
import pandas as pd  # type: ignore
from tabulate import tabulate  # type: ignore

from src.cutfile import Cut
from src.logger import get_logger


@dataclass(slots=True)
class CutflowItem:
    value: str
    npass: float
    eff: float
    ceff: float


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

    def __init__(
        self, df: pd.DataFrame, cuts: OrderedDict[str, Cut], logger: logging.Logger | None = None
    ):
        """
        Generates cutflow object that keeps track of various properties and ratios of selections made on given dataset

        :param df: Input analysis dataframe with boolean cut rows.
        :param cuts: Ordered dict of cuts made.
        :param logger: logger to output to
        """
        if missing_cuts := {cut_name for cut_name in cuts if "PASS_" + cut_name not in df.columns}:
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
        for cut in cuts.values():
            if self.__first_reco_cut and not cut.is_reco:
                raise ValueError("Truth cut after reco cut!")
            elif not self.__first_reco_cut and cut.is_reco:
                self.__first_reco_cut = cut.name

        # list of cutflow labels (necessary for all cutflows)
        self.cutflow_labels = ["Inclusive"] + [cut.name for cut in cuts.values()]
        self.cutflow_str = ["Inclusive"] + [cut.cutstr for cut in cuts.values()]
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
    rdf: ROOT.RDataFrame
    _cutflow: OrderedDict[str, CutflowItem] = field(init=False, default_factory=OrderedDict)
    report: ROOT.RDF.RCutFlowReport | None = field(init=False, default=None)
    logger: Logger = field(default_factory=get_logger)

    def __getitem__(self, item) -> CutflowItem:
        return self._cutflow[item]

    def __getattr__(self, item) -> CutflowItem:
        return self._cutflow[item]

    def __iter__(self) -> Generator[CutflowItem, None, None]:
        yield from self._cutflow.values()

    def __len__(self) -> int:
        return len(self._cutflow)

    @property
    def total_events(self) -> int:
        if self.report is None:
            raise AttributeError("Must have run one event loop in order to obtain number of events")
        return self.report.At("Inclusive").GetAll()

    def gen_cutflow(self, cuts: List[Cut]) -> None:
        """
        Generate cutflow - forces an event loop to run if not already.

        :param cuts: List of Cut objects to ov
        """
        self.report = self.rdf.Report()
        self._cutflow = OrderedDict(
            (
                (
                    cut.name,
                    CutflowItem(
                        value=cut.cutstr,
                        npass=self.report.At(cut.name).GetPass(),
                        eff=self.report.At(cut.name).GetEff(),
                        ceff=0.0,  # calculate this next
                    ),
                )
                for cut in cuts
            )
        )
        for cut_name in self._cutflow:
            self._cutflow[cut_name].ceff = (
                100 * self._cutflow[cut_name].npass / self.report.At("Inclusive").GetAll()
            )

    def print(self, latex_path: Path | None = None) -> None:
        if self.report is None:
            raise AttributeError("Must first generate cutflow before being able to print")

        table = tabulate(
            [["Inclusive", "-", self.report.At("Inclusive").GetAll(), "-", "-"]]
            + [
                [
                    cut_name,
                    cut.value,
                    cut.npass,
                    f"{cut.eff:.3G} %",
                    f"{cut.ceff:.3G} %",
                ]
                for cut_name, cut in self._cutflow.items()
            ],
            headers=["name", "value", "npass", "eff", "cum. eff"],
            tablefmt="latex" if latex_path else "simple",
        )

        if latex_path:
            with open(latex_path, "w") as f:
                f.write(table)

        else:
            print(table)
