from __future__ import annotations

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

    def gen_cutflow(self, rdf: ROOT.RDataFrame, cuts: List[Cut]) -> None:
        """
        Generate cutflow - forces an event loop to run if not already.

        :param rdf: filtered root dataframe
        :param cuts: List of Cut objects
        """
        report = rdf.Report()

        self.logger.debug(
            "Full report (internal):\n%s",
            "\n".join(
                [
                    f"{cutname}: {report.At(cutname).GetPass()}"
                    for cutname in list(rdf.GetFilterNames())
                ]
            ),
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
        self._cutflow = [
            CutflowItem(
                npass=report.At("Inclusive").GetAll(),
                eff=100,
                ceff=100,
                cut=Cut("Inclusive", cutstr="-"),
            )
        ] + self._cutflow
        for i in range(1, len(self._cutflow)):
            try:
                self._cutflow[i].ceff = 100 * self._cutflow[i].npass / self._cutflow[0].npass
            except ZeroDivisionError:
                self._cutflow[i].ceff = np.nan

    def import_cutflow(self, hist: ROOT.TH1I, cuts: List[Cut]) -> None:
        """Import cutflow from histogram and cutfile"""
        self._cutflow = cutflow_from_hist_and_cuts(hist, cuts)

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
                if (not cutflowitem.cut.is_reco and cutflowitem.cut.name.lower != "inclusive")
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
                if (cutflowitem.cut.is_reco and cutflowitem.cut.name.lower != "inclusive")
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


def cutflow_from_hist_and_cuts(hist: ROOT.TH1I, cuts: List[Cut]) -> List[CutflowItem]:
    """Create cutflow object from cutflow histogram"""
    if hist.GetNbinsX() - 1 != len(cuts):
        raise ValueError(
            f"Number of cuts passed to cutflow ({len(cuts)}) "
            f"does not match number of cuts in histogram ({hist.GetNbinsX() - 1})"
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
