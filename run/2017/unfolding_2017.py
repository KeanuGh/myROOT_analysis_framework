from pathlib import Path

import ROOT

from analysis import Analysis
from datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from utils.ROOT_utils import sum_th1s, get_th1_bin_edges
from utils.helper_functions import smart_join
from utils.variable_names import variable_data

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "TauPt",
    "MTW",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    # "TauBDTEleScore",
    # "TauRNNJetScore",
    # "TauNCoreTracks",
]
measurement_vars = measurement_vars_mass + measurement_vars_unitless
NOMINAL_NAME = "T_s1thv_NOMINAL"
YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
WP = [
    "loose",
    "medium",
    "tight",
]
VARS = [
    "MTW",
    "TauPt",
    "TauEta",
    "MET_met",
]
ITER = [
    1,
    2,
    4,
    8,
]

if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = Analysis(data_dict={}, year=YEAR, analysis_label=Path(__file__).stem)
    base_plotting_dir = analysis.paths.plot_dir
    fakes_colour = next(analysis.c_iter)
    truths = {
        "TauPt": "VisTruthTauPt",
        "MTW": "TruthMTW",
        "MET_met": "TruthNeutrinoPt",
        "TauEta": "VisTruthTauEta",
        "TauPhi": "VisTruthTauPhi",
    }
    symbols = {
        "TauPt": r"p_\mathrm{T}^{\tau_\mathrm{had-vis}}",
        "MTW": r"m^W_\mathrm{T}",
        "MET_met": r"E_\mathrm{T}^\mathrm{miss}",
        "TauEta": r"\eta^{\tau_\mathrm{had-vis}}",
        "TauPhi": r"\phi^{\tau_\mathrm{had-vis}}",
    }

    for wp in WP:
        wp_dir = base_plotting_dir / wp

        for var in VARS:
            if var not in truths:
                continue

            hists: dict[str, dict[str, Histogram1D]] = {}

            # shadow bins difference?
            # ========================================================
            analysis.paths.plot_dir = wp_dir / "shadow_bins"
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"efficiency_and_acceptance/root/efficiency_and_acceptance.root"
                )
            ) as file:
                acc_hist = file.Get(f"{wp}_{var}_acceptance")
                acc_hist.SetDirectory(0)
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"efficiency_and_acceptance_shadow_bin200/root/efficiency_and_acceptance_shadow_bin200.root"
                )
            ) as file:
                acc_hist200 = file.Get(f"{wp}_{var}_acceptance")
                acc_hist200.SetDirectory(0)
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"efficiency_and_acceptance_shadow_bin250/root/efficiency_and_acceptance_shadow_bin250.root"
                )
            ) as file:
                acc_hist250 = file.Get(f"{wp}_{var}_acceptance")
                acc_hist250.SetDirectory(0)
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"efficiency_and_acceptance_shadow_bin300/root/efficiency_and_acceptance_shadow_bin300.root"
                )
            ) as file:
                acc_hist300 = file.Get(f"{wp}_{var}_acceptance")
                acc_hist300.SetDirectory(0)

            analysis.plot(
                val=[acc_hist200, acc_hist250, acc_hist300, acc_hist],
                label=[
                    r"200|100|100 (min $m_\mathrm{T}^W$|$p_\mathrm{T}^\mathrm{had-vis}|E_\mathrm{T}^\mathrm{miss}$)",
                    r"250|125|125 (min $m_\mathrm{T}^W$|$p_\mathrm{T}^\mathrm{had-vis}|E_\mathrm{T}^\mathrm{miss}$)",
                    r"300|150|150 (min $m_\mathrm{T}^W$|$p_\mathrm{T}^\mathrm{had-vis}|E_\mathrm{T}^\mathrm{miss}$)",
                    r"350|170|170 (no shadow bin)",
                ],
                xlabel=(
                    variable_data[var]["name"] + (" [GeV]" if var in measurement_vars_mass else "")
                ),
                colour=["r", "b", "g", "k"],
                kind="overlay",
                do_stat=False,
                do_syst=False,
                title=smart_join(
                    f"{wp.title()} Tau ID",
                    r"$\sqrt{s} = 13$TeV",
                    sep=" | ",
                ),
                ylabel=r"$f_\mathrm{in}$",
                y_axlim=(0.7, 1.1),
                hline_at=1,
                vline_at=[0, 250, 300, 350] if var == "MTW" else None,
                logx=True if var in measurement_vars_mass else False,
                label_params={"llabel": "Preliminary"},
                filename=f"{wp}_{var}_shadow_bins.png",
            )

            # UNFOLD
            # ===============================================================================
            analysis.logger.info(f"Performing unfolding for {var} in {wp}")
            analysis.paths.plot_dir = wp_dir / "unfolded" / var

            # get response
            response = analysis.get_response_histogram(
                varname_reco=var,
                varname_truth=truths[var],
                dataset="wtaunu",
                wp=wp,
                nprong="",
                systematic=NOMINAL_NAME,
            )

            default_args = {
                "systematic": NOMINAL_NAME,
                "label": ["Truth MC", "Unfolded Data"],
                "xlabel": (
                    variable_data[var]["name"] + (" [GeV]" if var in measurement_vars_mass else "")
                ),
                "colour": ["r", "k"],
                "kind": "overlay",
                "do_stat": True,
                "do_syst": False,
                "title": smart_join(
                    f"Unfolding {YEAR}",
                    f"{wp.title()} Tau ID",
                    r"$\sqrt{s} = 13$TeV",
                    sep=" | ",
                ),
                "scale_by_bin_width": True,
                "ylabel": (
                    r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                    + symbols[var]
                    + r"}$"
                    + (" [fb / GeV]" if var in measurement_vars_mass else " [fb]")
                ),
                "logx": True if var in measurement_vars_mass else False,
                "ratio_plot": True,
                "ratio_label": "Data / MC",
                "ratio_axlim": (0.8, 1.2),
                "label_params": {"llabel": "Preliminary", "loc": 1},
            }

            def unfold_bayes(h: ROOT.TH1, i: int) -> ROOT.TH1:
                return ROOT.RooUnfoldBayes(response, h, i).Hreco()

            # get data and signal
            # ----------------------------------------------------------------
            with ROOT.TFile(
                str(analysis.paths.output_dir.parent / f"analysis_simple_2017/root/data.root")
            ) as file:
                data = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(var)
                data.SetDirectory(0)
            with ROOT.TFile(
                str(analysis.paths.output_dir.parent / f"analysis_simple_2017/root/wtaunu.root")
            ) as file:
                signal = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(var)
                signal.SetDirectory(0)

            # get truth
            # ----------------------------------------------------------------
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent / f"efficiency_and_acceptance/root/wtaunu.root"
                )
            ) as file:
                truth = file[f"{NOMINAL_NAME}/truth_tau"].Get(truths[var])
                truth.SetDirectory(0)
            truth = Histogram1D(th1=truth) / LUMI

            # get backgrounds
            # ----------------------------------------------------------------
            bkg_hists = []
            for bkg in ["wlnu", "zll", "top", "diboson"]:
                with ROOT.TFile(
                    str(analysis.paths.output_dir.parent / f"analysis_simple_2017/root/{bkg}.root")
                ) as file:
                    h = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(var)
                    h.SetDirectory(0)
                    bkg_hists.append(h)
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"analysis_fakes_{YEAR}/root/analysis_fakes_{YEAR}.root"
                )
            ) as file:
                fakes_hist = file.Get(f"{wp}_{var}_fakes_bkg_TauPt_src")
                fakes_hist.SetDirectory(0)
            background = sum_th1s(*(bkg_hists + [fakes_hist]))
            data_sig = data - background

            # uncertainties
            # ----------------------------------------------------------------
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"analysis_simple_2017/root/analysis_simple_2017.root"
                )
            ) as file:
                uncert = file.Get(f"{var}_{wp}_wtaunu_uncert")
                uncert.SetDirectory(0)
            uncert_up = data_sig + uncert
            uncert_down = data_sig - uncert

            # efficiency and acceptance
            # ----------------------------------------------------------------
            with ROOT.TFile(
                str(
                    analysis.paths.output_dir.parent
                    / f"efficiency_and_acceptance/root/efficiency_and_acceptance.root"
                )
            ) as file:
                eff_hist = file.Get(f"{wp}_{var}_efficiency")
                eff_hist.SetDirectory(0)
                acc_hist = file.Get(f"{wp}_{var}_acceptance")
                acc_hist.SetDirectory(0)

            shadow_bins = (
                ("no_shadow_bin", acc_hist),
                ("shadow_bin_200", acc_hist200),
                ("shadow_bin_250", acc_hist250),
                ("shadow_bin_300", acc_hist300),
            )

            # fully unfold
            # ----------------------------------------------------------------
            for sh_bin_label, acc in shadow_bins:
                hists[sh_bin_label] = {}

                if (sh_bin_label != "no_shadow_bin") and (var in ("MTW", "TauPt", "MET_met")):
                    # make new histogram that excludes shadow bin
                    acc_new = ROOT.TH1F(
                        acc.GetName(),
                        acc.GetTitle(),
                        acc.GetNbinsX() - 1,
                        get_th1_bin_edges(acc)[1:],
                    )
                    for i in range(acc.GetNbinsX()):
                        acc_new.SetBinContent(i, acc.GetBinContent(i + 1))
                    acc_new = Histogram1D(th1=acc_new)

                else:
                    acc_new = acc.Clone()
                    acc_new = Histogram1D(th1=acc_new)

                for n_iter in ITER:
                    analysis.logger.info(
                        f"Doing unfolding for {var} with {sh_bin_label} and {n_iter} bayesian iterations"
                    )

                    # unfold
                    data_unfolded = Histogram1D(th1=unfold_bayes(data_sig, n_iter))
                    signal_unfolded = Histogram1D(th1=unfold_bayes(signal, n_iter))
                    uncert_up_unfolded = Histogram1D(th1=unfold_bayes(uncert_up, n_iter))
                    uncert_down_unfolded = Histogram1D(th1=unfold_bayes(uncert_down, n_iter))
                    uncert_sym_unfolded = (
                        (uncert_up_unfolded - data_unfolded)
                        + (data_unfolded - uncert_down_unfolded)
                    ) / 2

                    data_unfolded_full = data_unfolded * acc_new / LUMI
                    signal_unfolded_full = signal_unfolded * acc_new / LUMI
                    uncert_unfolded_full = uncert_sym_unfolded * acc_new / LUMI

                    ratio_err = (uncert_unfolded_full / data_unfolded_full).bin_values()
                    default_args.update(
                        {
                            "uncert": uncert_unfolded_full,
                            "ratio_err": ratio_err,
                        }
                    )

                    # save
                    hists[sh_bin_label][n_iter] = data_unfolded_full

                    analysis.plot(
                        val=[truth, data_unfolded_full],
                        **default_args,
                        filename=f"{wp}_{var}_unfolded_{sh_bin_label}_{n_iter}iter.png",
                    )
                    analysis.plot(
                        val=[truth, data_unfolded_full],
                        **default_args,
                        logy=True,
                        filename=f"{wp}_{var}_unfolded_{sh_bin_label}_{n_iter}iter_logy.png",
                    )

            # COMPARISONS
            # ===================================================
            for sh_bin_label, _ in shadow_bins:
                analysis.plot(
                    val=[truth] + [hists[sh_bin_label][i] for i in ITER],
                    label=["Truth"] + [f"{i} iterations" for i in ITER],
                    xlabel=(
                        variable_data[var]["name"]
                        + (" [GeV]" if var in measurement_vars_mass else "")
                    ),
                    kind="overlay",
                    do_stat=True,
                    title=smart_join(
                        sh_bin_label,
                        f"{wp.title()} Tau ID",
                        r"$\sqrt{s} = 13$TeV",
                        sep=" | ",
                    ),
                    scale_by_bin_width=True,
                    ylabel=(
                        r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                        + symbols[var]
                        + r"}$"
                        + (" [fb / GeV]" if var in measurement_vars_mass else " [fb]")
                    ),
                    logx=True if var in measurement_vars_mass else False,
                    label_params={"llabel": "Preliminary", "loc": 1},
                    filename=f"{sh_bin_label}_i_iter_compare.png",
                )

            for i in ITER:
                analysis.plot(
                    val=[truth] + [hists[sh_bin_label][i] for sh_bin_label, _ in shadow_bins],
                    label=["Truth"] + [sh_bin_label for sh_bin_label, _ in shadow_bins],
                    xlabel=(
                        variable_data[var]["name"]
                        + (" [GeV]" if var in measurement_vars_mass else "")
                    ),
                    kind="overlay",
                    do_stat=True,
                    title=smart_join(
                        f"{i} Bayesian Iterations",
                        f"{wp.title()} Tau ID",
                        r"$\sqrt{s} = 13$TeV",
                        sep=" | ",
                    ),
                    scale_by_bin_width=True,
                    ylabel=(
                        r"$\frac{d\sigma_{W\rightarrow\tau\nu\rightarrow\mathrm{had}}}{d"
                        + symbols[var]
                        + r"}$"
                        + (" [fb / GeV]" if var in measurement_vars_mass else " [fb]")
                    ),
                    logx=True if var in measurement_vars_mass else False,
                    label_params={"llabel": "Preliminary", "loc": 1},
                    filename=f"shadow_bin_{i}_iter_compare.png",
                )

    analysis.logger.info("DONE.")
