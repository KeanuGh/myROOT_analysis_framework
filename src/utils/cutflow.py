import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from time import strftime


class Cutflow:
    def __init__(self, df: pd.DataFrame,
                 cut_dicts: List[Dict],
                 cut_label: str = ' CUT'
                 ):
        # create copy of dataframe to apply cuts to (just the cut columns)
        df = df[[col for col in df.columns if cut_label in col]]
        cutflow_df = df.copy()

        # set input fields
        self._cut_dicts = cut_dicts
        self._cut_label = cut_label

        # list of cutflow labels
        self.cutflow_labels = ['Inclusive'] + [cut['name'] for cut in self._cut_dicts]

        self.cutflow_a_ratio = []  # contains ratio of each separate cut to inclusive sample
        self.cutflow_ratio = []  # contains ratio of each cut to previous cut
        self.cutflow_cum = []  # contains ratio of each cut to inclusive sample
        self.cutflow_n_events = []  # contains number of events passing each cut

        # generate cutflow
        self._n_events_tot = len(cutflow_df.index)
        self.cutflow_n_events.append(self._n_events_tot)
        self.cutflow_ratio.append(1.0)
        self.cutflow_cum.append(1.0)

        prev_n = self._n_events_tot  # saves the last cut in loop
        for cut in cut_dicts:
            cutflow_df = cutflow_df[cutflow_df[cut['name'] + cut_label]]

            # calculations
            n_events_left = len(cutflow_df.index)
            self.cutflow_n_events.append(n_events_left)
            self.cutflow_ratio.append(n_events_left / prev_n)
            self.cutflow_cum.append(n_events_left / self._n_events_tot)
            self.cutflow_a_ratio.append(len(df[df[cut['name']+cut_label]].index) / self._n_events_tot)
            prev_n = n_events_left

    def terminal_printout(self) -> None:
        """
        Prints out cutflow table to terminal
        """
        max_n_len = len(str(self._n_events_tot))
        max_name_len = max([len(cut['name']) for cut in self._cut_dicts])

        # cutflow printout
        print(f"\n=========== CUTFLOW =============")
        print("Cut " + " " * (max_name_len - 3) +
              "Events " + " " * (max_n_len - 6) +
              "Ratio Cum. Ratio")
        # first line is inclusive sample
        print("Inclusive " + " " * (max_name_len - 9) + f"{self._n_events_tot} -     -")

        # print line
        for i, cutname in enumerate(self.cutflow_labels[1:]):
            n_events = self.cutflow_n_events[i]
            ratio = self.cutflow_ratio[i]
            cum_ratio = self.cutflow_cum[i]

            print(f"{cutname:<{max_name_len}} "
                  f"{n_events:<{max_n_len}} "
                  f"{ratio:.3f} "
                  f"{cum_ratio:.3f}")

    def print_histogram(self, filepath: str, kind: str, **kwargs) -> None:
        """
        Generates and saves a cutflow histogram

        :param filepath: path to directory to save plots into
        :param kind: which cutflow type. options:
                    'ratio': ratio of cut to previous cut
                    'cummulative': ratio of all current cuts to acceptance
                    'a_ratio': ratio of only current cut to acceptance
                    'event': number of events passing through each cut
        :param kwargs: keyword arguments to pass to plt.bar()
        :return: None
        """
        # assign histogram options for each type
        cuthist_options = {
            'ratio': {
                'filepath': filepath + 'cutflow_ratio.png',
                'y_ax_vals': self.cutflow_ratio,
                'ylabel': 'Cutflow ratio',
            },
            'cummulative': {
                'filepath': filepath + 'cutflow_cummulative.png',
                'y_ax_vals': self.cutflow_cum,
                'ylabel': 'Cummulative cutflow ratio',
            },
            'a_ratio': {
                'filepath': filepath + 'cutflow_acceptance_ratio.png',
                'y_ax_vals': self.cutflow_a_ratio,
                'ylabel': 'Cut ratio to accpetance',
            },
            'event': {
                'filepath': filepath + 'cutflow.png',
                'y_ax_vals': self.cutflow_n_events,
                'ylabel': 'Events',
            },
        }
        if kind not in cuthist_options.keys():
            raise Exception(f"Unknown cutflow histogram type {kind}. "
                            f"Possible types: {', '.join(cuthist_options.keys())}")

        fig, ax = plt.subplots()

        # plot
        # TODO: make cut groups the same colour
        ax.bar(x=self.cutflow_labels, height=cuthist_options[kind]['y_ax_vals'],
               color='w', edgecolor='k', width=1.0, **kwargs)
        ax.set_xlabel("cut")
        ax.set_ylabel(cuthist_options[kind]['ylabel'])
        ax.grid(b=True, which='both', axis='y')

        filepath = cuthist_options[kind]['filepath']
        fig.savefig(filepath)
        print(f"Cutflow histogram saved to {filepath}")

    def print_latex_table(self, filepath: str) -> str:
        """
        Prints a latex table containing cutflow to file in filepath with date and time.
        Returns the name of the printed table
        """
        latex_filepath = filepath + "cutflow_" + strftime("%Y-%m-%d_%H-%M-%S") + ".tex"

        with open(latex_filepath, "w") as f:
            f.write("\\begin{tabular}{|c||c|c|c|}\n"
                    "\\hline\n"
                    "Cut & Events & Ratio & Cumulative \\\\\\hline\n"
                    f"Inclusive & {self._n_events_tot} & - & - \\\\\n")
            # print line
            for i, cutname in enumerate(self.cutflow_labels[1:]):
                n_events = self.cutflow_n_events[i]
                ratio = self.cutflow_ratio[i]
                cum_ratio = self.cutflow_cum[i]
                f.write(f"{cutname} & {n_events} & {ratio:.3f} & {cum_ratio:.3f} \\\\\n")
            f.write("\\end{tabular}")

        print(f"Saved LaTeX cutflow table in {latex_filepath}")
        return latex_filepath
