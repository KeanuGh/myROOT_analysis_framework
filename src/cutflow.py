from time import strftime
from typing import Dict, List, OrderedDict, Optional

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

import src.config as config


class Cutflow:
    def __init__(self, df: pd.DataFrame,
                 cut_dicts: List[Dict],
                 cutgroups: Optional[OrderedDict[str, List[str]]] = None,
                 sequential: bool = True):
        """
        Generates cutflow object that keeps track of various properties and ratios of selections made on given dataset

        :param df: Input analysis dataframe with boolean cut rows.
        :param cut_dicts: Dictionary of cufts made.
        :param cutgroups: Optional ordered dictionary of cut groups. If suppled, will organise cutflow in terms of
                          cutgroups rather than individual cuts.
        :param sequential: Whether or not to organise cutflow as each cut happening one after the other
                           or each cut separately. Default True. If false, plots each cut separately.
        """
        # When sequential cuts, apply each cut one after each other in the order they were input into the cutfile
        self._is_sequential = sequential

        # generate cutflow
        self._n_events_tot = len(df.index)

        # if cutgroups are supplied, apply cutflow over groups rather than individual cuts
        self._cutgroups = cutgroups

        # set input fields
        self._cut_dicts = cut_dicts

        # list of cutflow labels (necessary for all cutflows)
        if self._cutgroups:
            self.cutflow_labels = ['Inclusive'] + [group for group in self._cutgroups.keys()]
        else:
            self.cutflow_labels = ['Inclusive'] + [cut['name'] for cut in self._cut_dicts]
        self.cutflow_ratio = [1.]  # contains ratio of each separate cut to inclusive sample
        self.cutflow_n_events = [self._n_events_tot]  # contains number of events passing each cut

        if self._is_sequential:
            # extract only the cut columns from the dataframe
            df = df[[col for col in df.columns if config.cut_label in col]]

            # special variables only for sequential cuts
            self.cutflow_a_ratio = [1.0]  # contains ratio of each separate cut to inclusive sample
            self.cutflow_cum = [1.0]  # contains ratio of each cut to inclusive sample

            # generate cutflow
            prev_n = self._n_events_tot  # saves the last cut in loop
            curr_cut_columns = []  # which cuts have already passed
            if self._cutgroups:
                # loop over groups
                for group in self._cutgroups.values():
                    curr_cut_columns += [cut + config.cut_label for cut in group]
                    # number of events passing current cut & all previous cuts
                    n_events_left = len(df[df[curr_cut_columns].all(1)].index)
                    # append calculations to cutflow arrays
                    self.cutflow_n_events.append(n_events_left)
                    self.cutflow_ratio.append(n_events_left / prev_n)
                    self.cutflow_cum.append(n_events_left / self._n_events_tot)
                    self.cutflow_a_ratio.append(len(df[df[[cut + config.cut_label for cut in group]].all(1)].index) / self._n_events_tot)
                    prev_n = n_events_left
            else:
                # loop over individual cuts
                for cut in self.cutflow_labels:
                    curr_cut_columns.append(cut + config.cut_label)
                    # number of events passing current cut & all previous cuts
                    n_events_left = len(df[df[curr_cut_columns].all(1)].index)
                    # append calculations to cutflow arrays
                    self.cutflow_n_events.append(n_events_left)
                    self.cutflow_ratio.append(n_events_left / prev_n)
                    self.cutflow_cum.append(n_events_left / self._n_events_tot)
                    self.cutflow_a_ratio.append(len(df[df[cut + config.cut_label].all(1)].index) / self._n_events_tot)
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

        else:
            # extract only the cut columns from the dataframe
            df = df[[col for col in df.columns if config.cut_label in col]]

            # if not a sequential cutflow, only print out the number of events passing each cut and the ratio with
            # the inclusive sample
            if self._cutgroups:
                # loop over groups
                for group in self._cutgroups.values():
                    # calculations
                    cut_cols = [cut + config.cut_label for cut in group]
                    n_events_cut = len(df[df[cut_cols].all(1)].index)
                    del cut_cols
                    self.cutflow_n_events.append(n_events_cut)
                    self.cutflow_ratio.append(n_events_cut / self._n_events_tot)
            else:
                for cut in cut_dicts:
                    # calculations
                    n_events_cut = len(df[df[cut['name'] + config.cut_label].all(1)].index)
                    self.cutflow_n_events.append(n_events_cut)
                    self.cutflow_ratio.append(n_events_cut / self._n_events_tot)

            # assign histogram options for each type
            self._cuthist_options = {
                'ratio': {
                    'filepath': '{}cutflow_ratio.png',
                    'y_ax_vals': self.cutflow_ratio,
                    'ylabel': 'Cutflow ratio',
                },
                'event': {
                    'filepath': '{}cutflow.png',
                    'y_ax_vals': self.cutflow_n_events,
                    'ylabel': 'Events',
                },
            }

    def terminal_printout(self) -> None:
        """
        Prints out cutflow table to terminal
        """
        # lengths of characters needed to get everything to line up properly
        max_n_len = len(str(self._n_events_tot))
        max_name_len = max([len(cut) for cut in self.cutflow_labels])

        if self._is_sequential:
            # cutflow printout
            print(f"\n=========== CUTFLOW =============")
            print("Option: Sequential")
            print("---------------------------------")
            print("Cut " + " " * (max_name_len - 3) +
                  "Events " + " " * (max_n_len - 6) +
                  "Ratio A. Ratio Cum. Ratio")
            # first line is inclusive sample
            print("Inclusive " + " " * (max_name_len - 9) + f"{self._n_events_tot} -     -        -")

            # print line
            for i, cutname in enumerate(self.cutflow_labels[1:]):
                n_events = self.cutflow_n_events[i + 1]
                ratio = self.cutflow_ratio[i + 1]
                cum_ratio = self.cutflow_cum[i + 1]
                a_ratio = self.cutflow_a_ratio[i + 1]
                print(f"{cutname:<{max_name_len}} "
                      f"{n_events:<{max_n_len}} "
                      f"{ratio:.3f} "
                      f"{a_ratio:.3f}    "
                      f"{cum_ratio:.3f}")
        else:
            # cutflow printout
            print(f"=========== CUTFLOW =============")
            print("Option: Non-sequential")
            print("---------------------------------")
            print("Cut " + " " * (max_name_len - 3) +
                  "Events " + " " * (max_n_len - 6) +
                  "Ratio")
            # first line is inclusive sample
            print("Inclusive " + " " * (max_name_len - 9) + f"{self._n_events_tot}   -")

            # print line
            for i, cutname in enumerate(self.cutflow_labels[1:]):
                n_events = self.cutflow_n_events[i + 1]
                ratio = self.cutflow_ratio[i + 1]
                print(f"{cutname:<{max_name_len}} "
                      f"{n_events:<{max_n_len}} "
                      f"{ratio:.3f}    ")
        print('')

    def print_histogram(self, kind: str, plot_label: str = '', **kwargs) -> None:
        """
        Generates and saves a cutflow histogram

        :param kind: which cutflow type. options:
                    'ratio': ratio of cut to previous cut
                    'cummulative': ratio of all current cuts to acceptance
                    'a_ratio': ratio of only current cut to acceptance
                    'event': number of events passing through each cut
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

        filepath = self._cuthist_options[kind]['filepath'].format(config.plot_dir)
        fig.savefig(filepath)
        print(f"Cutflow histogram saved to {filepath}")

    def print_latex_table(self, filepath: str, filename_prefix: str = '') -> str:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        latex_filepath = filepath + filename_prefix + "_cutflow_" + strftime("%Y-%m-%d_%H-%M-%S") + ".tex"

        with open(latex_filepath, "w") as f:
            f.write("\\begin{tabular}{|c||c|c|c|}\n"
                    "\\hline\n")
            if self._is_sequential:
                f.write(f"Cut & Events & Ratio & Cumulative \\\\\\hline\n"
                        f"Inclusive & {self._n_events_tot} & — & — \\\\\n")
                for i, cutname in enumerate(self.cutflow_labels[1:]):
                    n_events = self.cutflow_n_events[i+1]
                    ratio = self.cutflow_ratio[i+1]
                    cum_ratio = self.cutflow_cum[i+1]
                    f.write(f"{cutname} & {n_events} & {ratio:.3f} & {cum_ratio:.3f} \\\\\n")
            else:
                f.write(f"Cut & Events & Ratio\\\\\\hline\n"
                        f"Inclusive & {self._n_events_tot} & — \\\\\n")
                for i, cutname in enumerate(self.cutflow_labels[1:]):
                    n_events = self.cutflow_n_events[i+1]
                    ratio = self.cutflow_ratio[i+1]
                    f.write(f"{cutname} & {n_events} & {ratio:.3f} \\\\\n")
            f.write("\\hline\n"
                    "\\end{tabular}\n")

        print(f"Saved LaTeX cutflow table in {latex_filepath}")
        return latex_filepath
