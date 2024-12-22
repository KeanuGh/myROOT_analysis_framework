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
    "TauBDTEleScore",
    "TauRNNJetScore",
    "TauNCoreTracks",
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
    0,
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
    systematics = {
        "TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL",
        "TAUS_TRUEHADTAU_EFF_RECO_TOTAL",
        "TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718",
        "TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718",
        "TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718",
        "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt",
        "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt",
        "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt",
        "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt",
        "TAUS_TRUEHADTAU_SME_TES_INSITUEXP",
        "TAUS_TRUEHADTAU_SME_TES_INSITUFIT",
        "TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE",
        "TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST",
    }

    def unfolding_label(i: int) -> str:
        if i == 0:
            return "Bin-By-Bin Unfolding"
        else:
            return f"Iterative Unfolding - {i} Iterations"

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
                vline_at=[0, 0, 250, 300] if var == "MTW" else None,
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

            def unfold_bayes(h: ROOT.TH1, i: int) -> ROOT.TH1:
                if i == 0:
                    return ROOT.RooUnfoldBinByBin(response, h)
                return ROOT.RooUnfoldBayes(response, h, i)

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
            sys_hists = {}
            with ROOT.TFile(
                str(analysis.paths.output_dir.parent / f"analysis_simple_2017/root/wtaunu.root")
            ) as file:
                for sys in systematics:
                    uncert = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(f"{var}_{sys}_tot_uncert")
                    uncert.SetDirectory(0)
                    sys_hists[sys] = uncert

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
            acc_no_shadow = acc_hist.Clone()

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

                else:
                    acc_new = acc.Clone()

                for n_iter in ITER:
                    analysis.logger.info(
                        f"Doing unfolding for {var} with {sh_bin_label} and {n_iter} bayesian iterations"
                    )
                    default_args = {
                        "systematic": NOMINAL_NAME,
                        "xlabel": (
                            variable_data[var]["name"]
                            + (" [GeV]" if var in measurement_vars_mass else "")
                        ),
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
                        "ratio_axlim": (0.5, 1.5),
                        "label_params": {"llabel": "Preliminary", "loc": 1},
                    }

                    def unfold_scale(h) -> ROOT.TH1D:
                        h.Scale(1 / LUMI)
                        return h * acc_new / acc_no_shadow

                    def unfold(h, i) -> ROOT.TH1D:
                        h = unfold_bayes(h, i).Hunfold()
                        h.Scale(1 / LUMI)
                        return h * acc_new / acc_no_shadow

                    # unfold
                    data_unfolded = unfold_bayes(data_sig, n_iter)
                    signal_unfolded = unfold_bayes(signal, n_iter)

                    data_unfolded_full = unfold_scale(data_unfolded.Hunfold())
                    signal_unfolded_full = unfold_scale(signal_unfolded.Hunfold())

                    data_response = data_unfolded.response().Hresponse()
                    signal_response = signal_unfolded.response().Hresponse()

                    data_cov = ROOT.TH2D(data_unfolded.Eunfold())
                    signal_cov = ROOT.TH2D(signal_unfolded.Eunfold())

                    # plot
                    analysis.paths.plot_dir = wp_dir / "unfolded" / var
                    hists[sh_bin_label][n_iter] = data_unfolded_full
                    default_args.update(
                        {
                            "label": ["Truth MC", "Unfolded Signal MC", "Unfolded Data"],
                            "colour": ["r", "b", "k"],
                            "histstyle": ["step", "step", "errorbar"],
                        }
                    )
                    analysis.plot(
                        val=[truth, signal_unfolded_full, data_unfolded_full],
                        **default_args,
                        filename=f"{wp}_{var}_unfolded_{sh_bin_label}_{n_iter}iter.png",
                    )
                    analysis.plot(
                        val=[truth, signal_unfolded_full, data_unfolded_full],
                        **default_args,
                        logy=True,
                        filename=f"{wp}_{var}_unfolded_{sh_bin_label}_{n_iter}iter_logy.png",
                    )
                    analysis.plot_2d(
                        data_cov,
                        ylabel=r"Bin($m_\mathrm{T}^W [GeV]$)",
                        xlabel=r"Bin($m_\mathrm{T}^W [GeV]$)",
                        title=rf"$m_\mathrm{{T}}^W$ Covariance | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                        labels=True,
                        label_params={"llabel": "Simulation"},
                        filename=f"{wp}_{var}_{n_iter}iter_cov.png",
                    )

                    # detector systematics
                    # -------------------------------------------------------------------------
                    analysis.paths.plot_dir = wp_dir / "unfolded" / var / "sys"

                    def sys_up(sys) -> Histogram1D:
                        h = (
                            unfold(signal + sys_hists[sys], n_iter) - signal_unfolded_full
                        ) / data_unfolded_full
                        h.Scale(100)
                        return h

                    def sys_down(sys) -> Histogram1D:
                        h = (
                            unfold(signal - sys_hists[sys], n_iter) - signal_unfolded_full
                        ) / data_unfolded_full
                        h.Scale(100)
                        return h

                    default_args = {
                        "logx": True if var in measurement_vars_mass else False,
                        "ylabel": "Syst. Uncert. [%]",
                        "do_syst": False,
                        "do_stat": False,
                        "title": smart_join(
                            f"{wp.title()} Tau ID",
                            r"$\sqrt{s} = 13$TeV",
                            sep=" | ",
                        ),
                    }
                    analysis.plot(
                        val=[
                            sys_up("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt"),
                            sys_up("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt"),
                            sys_up("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt"),
                            sys_up("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt"),
                        ],
                        label=[
                            "Endcap_LowPt",
                            "Endcap_HighPt",
                            "Barrel_LowPt",
                            "Barrel_HighPt",
                        ],
                        colour=[
                            (0.0, 0.0, 0.5, 1.0),
                            (0.0, 0.8333333333333334, 1.0, 1.0),
                            (1.0, 0.9012345679012348, 0.0, 1.0),
                            (0.5, 0.0, 0.0, 1.0),
                        ],
                        linestyle=[
                            "solid",
                            "solid",
                            "solid",
                            "solid",
                        ],
                        **default_args,
                        filename=f"{var}_sys_DETECTOR_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # TRIGGER systematics
                    analysis.plot(
                        val=[
                            sys_up("TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718"),
                            sys_up("TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718"),
                            sys_up("TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718"),
                        ],
                        label=[
                            "TRIGGER_SYST161718",
                            "TRIGGER_STATMC161718_1up",
                            "TRIGGER_STATDATA161718",
                        ],
                        colour=[
                            (0.0, 0.0, 0.5, 1.0),
                            (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                            (0.5, 0.0, 0.0, 1.0),
                        ],
                        linestyle=[
                            "solid",
                            "solid",
                            "solid",
                        ],
                        **default_args,
                        filename=f"{var}_sys_TRIGGER_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # RECO systematic
                    analysis.plot(
                        val=[
                            sys_up("TAUS_TRUEHADTAU_EFF_RECO_TOTAL"),
                        ],
                        label=[
                            "EFF_RECO_TOTAL",
                        ],
                        colour=["r"],
                        linestyle=[
                            "solid",
                        ],
                        **default_args,
                        filename=f"{var}_sys_RECO_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # OTHER systematics
                    analysis.plot(
                        val=[
                            sys_up("TAUS_TRUEHADTAU_SME_TES_INSITUEXP"),
                            sys_up("TAUS_TRUEHADTAU_SME_TES_INSITUFIT"),
                            sys_up("TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE"),
                        ],
                        label=[
                            "INSITUEXP",
                            "INSITUFIT",
                            "MODEL_CLOSURE",
                        ],
                        colour=[
                            (0.0, 0.0, 0.5, 1.0),
                            (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                            (0.5, 0.0, 0.0, 1.0),
                        ],
                        linestyle=[
                            "solid",
                            "solid",
                            "solid",
                        ],
                        **default_args,
                        filename=f"{var}_sys_OTHER_{sh_bin_label}_{n_iter}iter.png",
                    )

            # COMPARISONS
            # ===================================================
            analysis.paths.plot_dir = wp_dir / "unfolded" / "compare"
            for sh_bin_label, _ in shadow_bins:
                analysis.plot(
                    val=[truth] + [hists[sh_bin_label][i] for i in ITER],
                    label=["Truth"] + [unfolding_label(i) for i in ITER],
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
