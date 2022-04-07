import logging
from typing import OrderedDict

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

import src.config as config
from src.cutfile import Cut


# TODO: Separate reco and truth cuts
class Cutflow:
    def __init__(self, df: pd.DataFrame,
                 cuts: OrderedDict[str, Cut],
                 logger: logging.Logger,
                 ):
        """
        Generates cutflow object that keeps track of various properties and ratios of selections made on given dataset

        :param df: Input analysis dataframe with boolean cut rows.
        :param cuts: Ordered of cuts made.
        :param logger: logger to output to
        """
        if missing_cuts := {
            cut_name
            for cut_name in cuts
            if cut_name + config.cut_label not in df.columns
        }:
            raise ValueError(f"Missing cut(s) {missing_cuts} in DataFrame")

        self.logger = logger

        # generate cutflow
        self._n_events_tot = len(df.index)

        # set input fields
        self._cuts = cuts

        # list of cutflow labels (necessary for all cutflows)
        self.cutflow_labels = ['Inclusive'] + [cut_name for cut_name in self._cuts]
        self.cutflow_ratio = [1.]  # contains ratio of each separate cut to inclusive sample
        self.cutflow_n_events = [self._n_events_tot]  # contains number of events passing each cut

        # extract only the cut columns from the dataframe
        df = df[[col for col in df.columns if config.cut_label in col]]

        self.cutflow_a_ratio = [1.0]  # contains ratio of each separate cut to inclusive sample
        self.cutflow_cum = [1.0]  # contains ratio of each cut to inclusive sample

        # generate cutflow
        prev_n = self._n_events_tot  # saves the last cut in loop
        curr_cut_columns = []  # which cuts have already passed

        # loop over individual cuts
        for cut in self.cutflow_labels[1:]:
            curr_cut_columns += [cut + config.cut_label]
            # number of events passing current cut & all previous cuts
            n_events_left = len(df.loc[df[curr_cut_columns].all(1)].index)
            self.cutflow_n_events.append(n_events_left)
            self.cutflow_ratio.append(n_events_left / prev_n)
            self.cutflow_cum.append(n_events_left / self._n_events_tot)
            self.cutflow_a_ratio.append(len(df[df[[cut + config.cut_label]].all(1)].index) / self._n_events_tot)
            prev_n = n_events_left

        # assign histogram options for each type
        self._cuthist_options = {
            'ratio': {
                'filepath': '{}cutflow_ratio.png',
                'y_ax_vals': self.cutflow_ratio,
                'ylabel': 'Cutflow ratio',
            },
            'cummulative': {
                'filepath': '{}cutflow_cummulative.png',
                'y_ax_vals': self.cutflow_cum,
                'ylabel': 'Cummulative cutflow ratio',
            },
            'a_ratio': {
                'filepath': '{}cutflow_acceptance_ratio.png',
                'y_ax_vals': self.cutflow_a_ratio,
                'ylabel': 'Cut ratio to accpetance',
            },
            'event': {
                'filepath': '{}cutflow.png',
                'y_ax_vals': self.cutflow_n_events,
                'ylabel': 'Events',
            },
        }

        if logger.level == logging.DEBUG:
            self.printout()

    def printout(self) -> None:
        """
        Prints out cutflow table to terminal
        """
        # lengths of characters needed to get everything to line up properly
        max_n_len = len(str(self._n_events_tot))
        max_name_len = max([len(cut) for cut in self.cutflow_labels])

        # cutflow printout
        self.logger.info('')
        self.logger.info(f"=========== CUTFLOW =============")
        self.logger.info("---------------------------------")
        self.logger.info("Cut " + " " * (max_name_len - 3) +
                         "Events " + " " * (max_n_len - 6) +
                         "Ratio A. Ratio Cum. Ratio")
        # first line is inclusive sample
        self.logger.info("Inclusive " + " " * (max_name_len - 9) + f"{self._n_events_tot} -     -        -")

        # print line
        for i, cutname in enumerate(self.cutflow_labels[1:]):
            self.logger.info(f"{cutname:<{max_name_len}} "
                             f"{self.cutflow_n_events[i + 1]:<{max_n_len}} "
                             f"{self.cutflow_ratio[i + 1]:.3f} "
                             f"{self.cutflow_a_ratio[i + 1]:.3f}    "
                             f"{self.cutflow_cum[i + 1]:.3f}")
        self.logger.info('')

    def print_latex_table(self, filepath: str) -> None:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        with open(filepath, "w") as f:
            f.write("\\begin{tabular}{|c||c|c|c|}\n"
                    "\\hline\n")
            f.write(f"Cut & Events & Ratio & Cumulative \\\\\\hline\n"
                    f"Inclusive & {self.cutflow_n_events[0]} & — & — \\\\\n")
            for i, cut in enumerate(self._cuts.values()):
                f.write(f"{cut.cutstr} & "
                        f"{self.cutflow_n_events[i + 1]} & "
                        f"{self.cutflow_ratio[i + 1]:.3f} & "
                        f"{self.cutflow_cum[i + 1]:.3f} "
                        f"\\\\\n")
            f.write("\\hline\n\\end{tabular}\n")

    def print_histogram(self, out_path: str, kind: str, plot_label: str = '', **kwargs) -> None:
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
            raise Exception(f"Unknown cutflow histogram type {kind}. "
                            f"Possible types: {', '.join(self._cuthist_options.keys())}")
        fig, ax = plt.subplots()

        # plot
        ax.bar(x=self.cutflow_labels, height=self._cuthist_options[kind]['y_ax_vals'],
               color='w', edgecolor='k', width=1.0, **kwargs)
        ax.set_xlabel("Cut")
        ax.tick_params(axis='x', which='both', bottom=False, top=False)  # disable ticks on x axis
        ax.set_xticks(ax.get_xticks())  # REMOVE IN THE FUTURE - PLACED TO AVOID WARNING - BUG IN MATPLOTLIB 3.3.2
        ax.set_xticklabels(labels=self.cutflow_labels, rotation='-40')
        ax.set_ylabel(self._cuthist_options[kind]['ylabel'])
        hep.atlas.label(llabel="Internal", loc=0, ax=ax, rlabel=plot_label)
        ax.grid(b=True, which='both', axis='y', alpha=0.3)

        filepath = self._cuthist_options[kind]['filepath'].format(out_path)
        fig.savefig(filepath)
        self.logger.info(f"Cutflow histogram saved to {filepath}")
