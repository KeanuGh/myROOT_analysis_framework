"""Run the experimental full shadow-response unfolding.

This implementation loads the response matrices from the shadow-bin
efficiency/acceptance outputs and unfolds with those matrices directly. It is
currently incomplete as a closure method because the measured inputs are still
nominal signal-region histograms, not shadow-region reconstructed inputs.
"""

from pathlib import Path

import ROOT
import tabulate

from src.analysis import Analysis
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from utils import ROOT_utils
from utils.helper_functions import smart_join
from utils.plotting_tools import PlotKwargs
from utils.ROOT_utils import get_th1_bin_edges, sum_th1s
from utils.variable_names import variable_data

ResponseComponents = tuple[ROOT.RooUnfoldResponse, ROOT.TH1, ROOT.TH1, ROOT.TH2]

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
FULL_RESPONSE_SYSTEMATICS = {
    "TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL",
    "TAUS_TRUEHADTAU_EFF_RECO_TOTAL",
    "TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718",
    "TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718",
    "TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718",
}
RECO_ONLY_SYSTEMATICS = {
    "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt",
    "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt",
    "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt",
    "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt",
    "TAUS_TRUEHADTAU_SME_TES_INSITUEXP",
    "TAUS_TRUEHADTAU_SME_TES_INSITUFIT",
    "TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE",
    "TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST",
}
SHADOW_RESPONSE_OUTPUTS = {
    "no_shadow_bin": "efficiency_and_acceptance",
    "shadow_bin_200": "efficiency_and_acceptance_shadow_bin200",
    "shadow_bin_250": "efficiency_and_acceptance_shadow_bin250",
    "shadow_bin_300": "efficiency_and_acceptance_shadow_bin300",
}
BIN_EDGE_TOLERANCE = 1e-6


def systematic_source(systematic: str) -> str:
    """Return the base systematic name without the up/down variation suffix."""
    for suffix in ("__1up", "__1down"):
        if systematic.endswith(suffix):
            return systematic.removesuffix(suffix)
    return systematic


def same_bin_edges(left: ROOT.TH1, right: ROOT.TH1) -> bool:
    """Check whether two ROOT histograms have the same explicit bin edges."""
    left_edges = get_th1_bin_edges(left)
    right_edges = get_th1_bin_edges(right)
    if len(left_edges) != len(right_edges):
        return False
    return all(
        abs(float(left_edge) - float(right_edge)) < BIN_EDGE_TOLERANCE
        for left_edge, right_edge in zip(left_edges, right_edges, strict=True)
    )


def has_leading_shadow_bin(source: ROOT.TH1, target: ROOT.TH1) -> bool:
    """Check whether target equals source with one extra leading shadow bin."""
    source_edges = get_th1_bin_edges(source)
    target_edges = get_th1_bin_edges(target)
    if len(target_edges) != len(source_edges) + 1:
        return False
    return all(
        abs(float(source_edge) - float(target_edge)) < BIN_EDGE_TOLERANCE
        for source_edge, target_edge in zip(source_edges, target_edges[1:], strict=True)
    )


def embed_histogram_in_shadow_binning(source: ROOT.TH1, target: ROOT.TH1, name: str) -> ROOT.TH1:
    """Clone source into target binning by inserting an empty leading shadow bin."""
    if same_bin_edges(source, target):
        clone = source.Clone(name)
        clone.SetDirectory(0)
        return clone

    if not has_leading_shadow_bin(source, target):
        raise ValueError(
            f"Cannot align histogram '{source.GetName()}' to '{target.GetName()}': "
            "the target binning is not identical and does not contain one leading shadow bin."
        )

    target_edges = get_th1_bin_edges(target)
    aligned = ROOT.TH1D(name, source.GetTitle(), len(target_edges) - 1, target_edges)
    aligned.SetDirectory(0)
    for bin_i in range(1, source.GetNbinsX() + 1):
        aligned.SetBinContent(bin_i + 1, source.GetBinContent(bin_i))
        aligned.SetBinError(bin_i + 1, source.GetBinError(bin_i))
    return aligned


def crop_shadow_bin_from_histogram(source: ROOT.TH1, target: ROOT.TH1, name: str) -> ROOT.TH1:
    """Clone source into target binning by dropping a leading shadow bin."""
    if same_bin_edges(source, target):
        clone = source.Clone(name)
        clone.SetDirectory(0)
        return clone

    if not has_leading_shadow_bin(target, source):
        raise ValueError(
            f"Cannot crop histogram '{source.GetName()}' to '{target.GetName()}': "
            "the source binning is not identical and does not contain one leading shadow bin."
        )

    target_edges = get_th1_bin_edges(target)
    cropped = ROOT.TH1D(name, source.GetTitle(), len(target_edges) - 1, target_edges)
    cropped.SetDirectory(0)
    for bin_i in range(1, target.GetNbinsX() + 1):
        cropped.SetBinContent(bin_i, source.GetBinContent(bin_i + 1))
        cropped.SetBinError(bin_i, source.GetBinError(bin_i + 1))
    return cropped


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = Analysis(data_dict={}, year=YEAR, analysis_label=Path(__file__).stem)
    base_plotting_dir = analysis.paths.plot_dir
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

    def load_response_components(
        output_label: str,
        systematic: str,
        response_var: str,
        response_wp: str,
    ) -> ResponseComponents:
        """Load reco, truth, and migration histograms and wrap them as a RooUnfold response."""
        response_file = analysis.paths.output_dir.parent / output_label / "root/wtaunu_had.root"
        reco_path = f"{systematic}/{response_wp}_reco_tau/{response_var}"
        truth_path = f"{systematic}/truth_tau/{truths[response_var]}"
        matrix_path = (
            f"{systematic}/{response_wp}_truth_reco_tau/{response_var}_{truths[response_var]}"
        )

        with ROOT.TFile(str(response_file)) as file:
            response_reco = file.Get(reco_path)
            response_truth = file.Get(truth_path)
            response_matrix = file.Get(matrix_path)
            missing = [
                path
                for path, hist in (
                    (reco_path, response_reco),
                    (truth_path, response_truth),
                    (matrix_path, response_matrix),
                )
                if not hist
            ]
            if missing:
                raise KeyError(
                    f"Missing response object(s) in {response_file}: {', '.join(missing)}"
                )

            response_reco = response_reco.Clone(
                f"{output_label}_{systematic}_{response_wp}_{response_var}_reco"
            )
            response_truth = response_truth.Clone(
                f"{output_label}_{systematic}_{response_wp}_{response_var}_truth"
            )
            response_matrix = response_matrix.Clone(
                f"{output_label}_{systematic}_{response_wp}_{response_var}_matrix"
            )
            response_reco.SetDirectory(0)
            response_truth.SetDirectory(0)
            response_matrix.SetDirectory(0)

        response = ROOT.RooUnfoldResponse(response_reco, response_truth, response_matrix)
        return response, response_reco, response_truth, response_matrix

    def unfolding_label(i: int) -> str:
        """User-facing label for the chosen unfolding method/iteration count."""
        if i == 0:
            return "Bin-By-Bin Unfolding"
        else:
            return f"Iterative Unfolding - {i} Iterations"


    def covariance_from_hist(h: ROOT.TH1, name: str) -> ROOT.TH2D:
        """Build a diagonal covariance matrix from a histogram's bin errors."""
        cov = ROOT.TH2D(
            name,
            name,
            h.GetNbinsX(),
            0,
            h.GetNbinsX(),
            h.GetNbinsX(),
            0,
            h.GetNbinsX(),
        )
        cov.SetDirectory(0)
        for bin_i in range(1, h.GetNbinsX() + 1):
            cov.SetBinContent(bin_i, bin_i, h.GetBinError(bin_i) ** 2)
        return cov


    def unfold_bin_by_bin_with_corrections(
            data_sig: ROOT.TH1,
            signal: ROOT.TH1,
            response_reco: ROOT.TH1,
            response_truth: ROOT.TH1,
            unfold_scale,
            wp: str,
            var: str,
            sh_bin_label: str,
    ) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1, ROOT.TH1, ROOT.TH2D, ROOT.TH2D]:
        """Apply the framework bin-by-bin correction when reco/truth axes match."""
        data_unfolded = analysis.unfold_bin_by_bin(
            data_sig,
            response_reco,
            response_truth,
            name=f"{wp}_{var}_{sh_bin_label}_data",
        )
        signal_unfolded = analysis.unfold_bin_by_bin(
            signal,
            response_reco,
            response_truth,
            name=f"{wp}_{var}_{sh_bin_label}_signal",
        )

        data_unfolded_full = unfold_scale(data_unfolded.unfolded)
        signal_unfolded_full = unfold_scale(signal_unfolded.unfolded)

        data_cov = covariance_from_hist(
            data_unfolded_full,
            f"{wp}_{var}_{sh_bin_label}_bin_by_bin_data_cov",
        )
        signal_cov = covariance_from_hist(
            signal_unfolded_full,
            f"{wp}_{var}_{sh_bin_label}_bin_by_bin_signal_cov",
        )

        analysis.plot(
            val=[data_unfolded.correction],
            label=["Truth / reco"],
            xlabel=(
                    variable_data[var]["name"]
                    + (" [GeV]" if var in measurement_vars_mass else "")
            ),
            ylabel="Bin-by-bin correction",
            title=smart_join(
                sh_bin_label,
                f"{wp.title()} Tau ID",
                r"$\sqrt{s} = 13$TeV",
                sep=" | ",
            ),
            do_stat=True,
            do_syst=False,
            logx=True if var in measurement_vars_mass else False,
            label_params={"llabel": "Simulation", "loc": 1},
            filename=f"{wp}_{var}_{sh_bin_label}_bin_by_bin_correction.png",
        )

        return (
            data_unfolded_full,
            signal_unfolded_full,
            data_unfolded.correction,
            signal_unfolded.correction,
            data_cov,
            signal_cov,
        )


    for wp in WP:
        wp_dir = base_plotting_dir / wp

        for var in VARS:
            if var not in truths:
                continue

            hists: dict[str, dict[str, Histogram1D]] = {}
            response_cache: dict[str, ResponseComponents] = {}

            # Compare acceptance shifts from the alternative shadow-bin definitions.
            # ========================================================
            analysis.paths.plot_dir = wp_dir / "shadow_bins"
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent
                        / "efficiency_and_acceptance/root/efficiency_and_acceptance.root"
                    )
            ) as file:
                acc_hist = file.Get(f"{wp}_{var}_acceptance")
                acc_hist.SetDirectory(0)
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent
                        / "efficiency_and_acceptance_shadow_bin200/root/efficiency_and_acceptance_shadow_bin200.root"
                    )
            ) as file:
                acc_hist200 = file.Get(f"{wp}_{var}_acceptance")
                acc_hist200.SetDirectory(0)
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent
                        / "efficiency_and_acceptance_shadow_bin250/root/efficiency_and_acceptance_shadow_bin250.root"
                    )
            ) as file:
                acc_hist250 = file.Get(f"{wp}_{var}_acceptance")
                acc_hist250.SetDirectory(0)
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent
                        / "efficiency_and_acceptance_shadow_bin300/root/efficiency_and_acceptance_shadow_bin300.root"
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

            # Load the nominal response first; systematic variations reuse it below.
            response, response_reco, response_truth, response_matrix = load_response_components(
                SHADOW_RESPONSE_OUTPUTS["no_shadow_bin"],
                NOMINAL_NAME,
                var,
                wp,
            )
            response_cache[NOMINAL_NAME] = (
                response,
                response_reco,
                response_truth,
                response_matrix,
            )


            def get_response_components(
                    systematic: str,
                    *,
                    cache: dict[str, ResponseComponents] = response_cache,
                    response_var: str = var,
                    response_wp: str = wp,
            ) -> ResponseComponents:
                """Return the response to use for a systematic variation."""
                if systematic not in cache:
                    source = systematic_source(systematic)
                    if source in FULL_RESPONSE_SYSTEMATICS:
                        # Efficiency variations change the response, reco, and truth pieces.
                        cache[systematic] = analysis.get_response_histogram(
                            varname_reco=response_var,
                            varname_truth=truths[response_var],
                            dataset="wtaunu_had",
                            wp=response_wp,
                            nprong="",
                            systematic=systematic,
                            return_histograms=True,
                        )
                    elif source in RECO_ONLY_SYSTEMATICS:
                        # TES variations only shift the reconstructed axis in saved inputs.
                        response_file = (
                                analysis.paths.output_dir.parent
                                / "efficiency_and_acceptance/root/wtaunu_had.root"
                        )
                        reco_path = f"{systematic}/{response_wp}_reco_tau/{response_var}"
                        with ROOT.TFile(str(response_file)) as file:
                            reco_varied = file.Get(reco_path)
                            if not reco_varied:
                                raise KeyError(
                                    f"Expected shifted reco histogram '{reco_path}' "
                                    f"in {response_file}"
                                )
                            reco_varied.SetDirectory(0)

                        _, _, nominal_truth, nominal_matrix = cache[NOMINAL_NAME]
                        nominal_truth_for_response = nominal_truth.Clone(
                            f"{systematic}_{response_wp}_{response_var}_nominal_truth"
                        )
                        nominal_matrix_for_response = nominal_matrix.Clone(
                            f"{systematic}_{response_wp}_{response_var}_nominal_response"
                        )
                        nominal_truth_for_response.SetDirectory(0)
                        nominal_matrix_for_response.SetDirectory(0)

                        response_from_shifted_reco = ROOT.RooUnfoldResponse(
                            reco_varied,
                            nominal_truth_for_response,
                            nominal_matrix_for_response,
                        )
                        cache[systematic] = (
                            response_from_shifted_reco,
                            reco_varied,
                            nominal_truth_for_response,
                            nominal_matrix_for_response,
                        )
                    else:
                        raise ValueError(
                            f"No response-input policy is defined for systematic '{systematic}'. "
                            "Add it to FULL_RESPONSE_SYSTEMATICS or RECO_ONLY_SYSTEMATICS."
                        )
                return cache[systematic]


            def unfold_bayes(
                    h: ROOT.TH1,
                    response_obj: ROOT.RooUnfoldResponse,
                    i: int,
            ) -> ROOT.TH1:
                """Construct the requested RooUnfold object for one histogram."""
                if i == 0:
                    return ROOT.RooUnfoldBinByBin(response_obj, h)
                return ROOT.RooUnfoldBayes(response_obj, h, i)

            # Current measured inputs are nominal SR histograms. Full shadow closure
            # needs matching shadow-region inputs upstream, not just shadow responses.
            # ----------------------------------------------------------------
            with ROOT.TFile(
                    str(analysis.paths.output_dir.parent / "analysis_simple_2017/root/data.root")
            ) as file:
                data = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(var)
                data.SetDirectory(0)
            with ROOT.TFile(
                    str(analysis.paths.output_dir.parent / "analysis_simple_2017/root/wtaunu_had.root")
            ) as file:
                signal = file[f"{NOMINAL_NAME}/{wp}_SR_passID"].Get(var)
                signal.SetDirectory(0)

            # get truth
            # ----------------------------------------------------------------
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent / "efficiency_and_acceptance/root/wtaunu_had.root"
                    )
            ) as file:
                truth_nominal_hist = file[f"{NOMINAL_NAME}/truth_tau"].Get(truths[var])
                truth_nominal_hist.SetDirectory(0)
            truth = Histogram1D(th1=truth_nominal_hist) / LUMI

            # get backgrounds
            # ----------------------------------------------------------------
            bkg_hists = []
            for bkg in ["wtaunu_lep", "wlnu", "zll", "top", "diboson"]:
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

            bin_edges = get_th1_bin_edges(signal)
            signal_fraction_rows = []
            for bin_i in range(1, signal.GetNbinsX() + 1):
                signal_yield = signal.GetBinContent(bin_i)
                total_prediction = signal_yield + background.GetBinContent(bin_i)
                signal_percent = 100 * signal_yield / total_prediction if total_prediction else 0.0
                signal_fraction_rows.append(
                    [
                        f"[{bin_edges[bin_i - 1]:.3g}, {bin_edges[bin_i]:.3g})",
                        f"{signal_yield:.2f}",
                        f"{total_prediction:.2f}",
                        f"{signal_percent:.2f}",
                    ]
                )

            signal_fraction_table_path = (
                    analysis.paths.latex_dir / f"{wp}_{var}_signal_percentage.tex"
            )
            with open(signal_fraction_table_path, "w") as f:
                f.write(
                    tabulate.tabulate(
                        signal_fraction_rows,
                        headers=[
                            rf"{variable_data[var]['name']} bin",
                            r"$W\rightarrow\tau\nu$",
                            "Total prediction",
                            "Signal [\\%]",
                        ],
                        tablefmt="latex_raw",
                    )
                )
            analysis.logger.info(f"Saved table to {signal_fraction_table_path}")

            # Load efficiency/acceptance products used for diagnostics and labels.
            # ----------------------------------------------------------------
            with ROOT.TFile(
                    str(
                        analysis.paths.output_dir.parent
                        / "efficiency_and_acceptance/root/efficiency_and_acceptance.root"
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
            shadow_response_cache = {"no_shadow_bin": response_cache[NOMINAL_NAME]}

            # Unfold with each response definition. Shadow responses can have one
            # extra leading bin; inputs are aligned below before RooUnfold sees them.
            # ----------------------------------------------------------------
            for sh_bin_label, _ in shadow_bins:
                hists[sh_bin_label] = {}

                if sh_bin_label not in shadow_response_cache:
                    shadow_response_cache[sh_bin_label] = load_response_components(
                        SHADOW_RESPONSE_OUTPUTS[sh_bin_label],
                        NOMINAL_NAME,
                        var,
                        wp,
                    )

                (
                    active_response,
                    active_response_reco,
                    active_response_truth,
                    active_response_matrix,
                ) = shadow_response_cache[sh_bin_label]
                # This inserts an empty leading bin for shadow responses. It is a
                # technical alignment step, not a substitute for real shadow-region data.
                input_data_sig = embed_histogram_in_shadow_binning(
                    data_sig,
                    active_response_reco,
                    f"{wp}_{var}_{sh_bin_label}_data_minus_background_input",
                )
                input_signal = embed_histogram_in_shadow_binning(
                    signal,
                    active_response_reco,
                    f"{wp}_{var}_{sh_bin_label}_signal_input",
                )

                for n_iter in ITER:
                    analysis.logger.info(
                        f"Doing unfolding for {var} with {sh_bin_label} and {n_iter} bayesian iterations"
                    )
                    default_args: PlotKwargs = {
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


                    def unfold_scale(
                        h: ROOT.TH1,
                        nominal_truth_bins: ROOT.TH1 = truth_nominal_hist,
                    ) -> ROOT.TH1:
                        """Convert unfolded yields to cross-section and nominal truth bins."""
                        scaled = h.Clone(f"{h.GetName()}_scaled")
                        scaled.SetDirectory(0)
                        scaled.Scale(1 / LUMI)
                        return crop_shadow_bin_from_histogram(
                            scaled,
                            nominal_truth_bins,
                            f"{scaled.GetName()}_nominal_truth_bins",
                        )


                    def unfold(
                        h: ROOT.TH1,
                        i: int,
                        response_obj: ROOT.RooUnfoldResponse = active_response,
                        response_reco_hist: ROOT.TH1 = active_response_reco,
                        response_truth_hist: ROOT.TH1 = active_response_truth,
                    ) -> ROOT.TH1D:
                        """Unfold one histogram with the active response definition."""
                        if i == 0:
                            # Use the local correction helper when axes match exactly.
                            h = analysis.unfold_bin_by_bin(
                                h,
                                response_reco_hist,
                                response_truth_hist,
                            ).unfolded
                        else:
                            h = unfold_bayes(h, response_obj, i).Hunfold()
                        return unfold_scale(h)


                    # unfold
                    analysis.paths.plot_dir = wp_dir / "unfolded" / var
                    if (
                        n_iter == 0
                        and active_response_reco.GetNbinsX() == active_response_truth.GetNbinsX()
                    ):
                        (
                            data_unfolded_full,
                            signal_unfolded_full,
                            data_response,
                            signal_response,
                            data_cov,
                            signal_cov,
                        ) = unfold_bin_by_bin_with_corrections(
                            input_data_sig,
                            input_signal,
                            active_response_reco,
                            active_response_truth,
                            unfold_scale,
                            wp,
                            var,
                            sh_bin_label,
                        )
                    else:
                        if n_iter == 0:
                            analysis.logger.info(
                                "Using RooUnfold bin-by-bin for %s %s %s because the "
                                "shadow response has %s reco bins and %s truth bins.",
                                wp,
                                var,
                                sh_bin_label,
                                active_response_reco.GetNbinsX(),
                                active_response_truth.GetNbinsX(),
                            )
                        data_unfolded = unfold_bayes(input_data_sig, active_response, n_iter)
                        signal_unfolded = unfold_bayes(input_signal, active_response, n_iter)

                        data_unfolded_full = unfold_scale(data_unfolded.Hunfold())
                        signal_unfolded_full = unfold_scale(signal_unfolded.Hunfold())

                        data_response = data_unfolded.response().Hresponse()
                        signal_response = signal_unfolded.response().Hresponse()

                        if n_iter == 0:
                            data_cov = covariance_from_hist(
                                data_unfolded_full,
                                f"{wp}_{var}_{sh_bin_label}_response_bin_by_bin_data_cov",
                            )
                            signal_cov = covariance_from_hist(
                                signal_unfolded_full,
                                f"{wp}_{var}_{sh_bin_label}_response_bin_by_bin_signal_cov",
                            )
                        else:
                            data_cov = ROOT.TH2D(data_unfolded.Eunfold())
                            signal_cov = ROOT.TH2D(signal_unfolded.Eunfold())

                    # plot
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
                    response_plot = active_response_matrix if n_iter == 0 else data_response
                    response_filename = (
                        f"{wp}_{var}_{n_iter}iter_response.png"
                        if sh_bin_label == "no_shadow_bin"
                        else f"{wp}_{var}_{sh_bin_label}_{n_iter}iter_response.png"
                    )
                    analysis.plot_2d(
                        response_plot,
                        ylabel=f"Truth {truths[var]}",
                        xlabel=f"Reco {var}",
                        title=smart_join(
                            sh_bin_label,
                            unfolding_label(n_iter),
                            f"{wp.title()} Tau ID",
                            r"$\sqrt{s} = 13$TeV",
                            sep=" | ",
                        ),
                        labels=True,
                        label_params={"llabel": "Simulation"},
                        filename=response_filename,
                    )

                    if sh_bin_label != "no_shadow_bin":
                        # Shadow-bin systematic response inputs have not been produced,
                        # so only the nominal shadow response can be plotted here.
                        analysis.logger.info(
                            "Skipping response systematic uncertainty plots for %s %s %s: "
                            "saved shadow-bin systematic response inputs are not available.",
                            wp,
                            var,
                            sh_bin_label,
                        )
                        continue

                    # detector systematics
                    # -------------------------------------------------------------------------
                    analysis.paths.plot_dir = wp_dir / "unfolded" / var / "sys"


                    def response_sys_uncertainty(
                            sys: str,
                            *,
                            nominal_unfolded: ROOT.TH1 = data_unfolded_full,
                            input_data: ROOT.TH1 = data_sig,
                            iter_count: int = n_iter,
                            hist_wp: str = wp,
                            hist_var: str = var,
                            hist_shadow_bin: str = sh_bin_label,
                    ) -> ROOT.TH1:
                        """Calculate relative uncertainty from up/down response variations."""
                        response_up, reco_up, truth_up, _ = get_response_components(f"{sys}__1up")
                        response_down, reco_down, truth_down, _ = get_response_components(
                            f"{sys}__1down"
                        )

                        unfolded_up = unfold(
                            input_data,
                            iter_count,
                            response_up,
                            reco_up,
                            truth_up,
                        )
                        unfolded_down = unfold(
                            input_data,
                            iter_count,
                            response_down,
                            reco_down,
                            truth_down,
                        )
                        response_uncertainty = ROOT_utils.th1_max_abs_deviation(
                            unfolded_up,
                            unfolded_down,
                            nominal_unfolded,
                        )
                        return ROOT_utils.th1_relative_uncertainty(
                            response_uncertainty,
                            nominal_unfolded,
                            name=(
                                f"{hist_wp}_{hist_var}_{hist_shadow_bin}_{iter_count}iter_"
                                f"{sys}_response_uncert"
                            ),
                        )


                    default_args: PlotKwargs = {
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

                    def plot_response_systematics(
                            sys_specs,
                            filename: str,
                            *,
                            plot_args: PlotKwargs = default_args,
                    ) -> None:
                        """Plot a group of response systematic uncertainties together."""
                        histograms = []
                        labels = []
                        colours = []
                        linestyles = []
                        for sys_name, label, colour, linestyle in sys_specs:
                            hist = response_sys_uncertainty(sys_name)
                            histograms.append(hist)
                            labels.append(label)
                            colours.append(colour)
                            linestyles.append(linestyle)

                        analysis.plot(
                            val=histograms,
                            label=labels,
                            colour=colours,
                            linestyle=linestyles,
                            **plot_args,
                            filename=filename,
                        )

                    plot_response_systematics(
                        [
                            (
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt",
                                "Endcap_LowPt",
                                (0.0, 0.0, 0.5, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt",
                                "Endcap_HighPt",
                                (0.0, 0.8333333333333334, 1.0, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt",
                                "Barrel_LowPt",
                                (1.0, 0.9012345679012348, 0.0, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt",
                                "Barrel_HighPt",
                                (0.5, 0.0, 0.0, 1.0),
                                "solid",
                            ),
                        ],
                        f"{var}_sys_DETECTOR_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # TRIGGER systematics
                    plot_response_systematics(
                        [
                            (
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718",
                                "TRIGGER_SYST161718",
                                (0.0, 0.0, 0.5, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718",
                                "TRIGGER_STATMC161718_1up",
                                (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718",
                                "TRIGGER_STATDATA161718",
                                (0.5, 0.0, 0.0, 1.0),
                                "solid",
                            ),
                        ],
                        f"{var}_sys_TRIGGER_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # RECO systematic
                    plot_response_systematics(
                        [
                            (
                                "TAUS_TRUEHADTAU_EFF_RECO_TOTAL",
                                "EFF_RECO_TOTAL",
                                "r",
                                "solid",
                            ),
                        ],
                        f"{var}_sys_RECO_{sh_bin_label}_{n_iter}iter.png",
                    )
                    # OTHER systematics
                    plot_response_systematics(
                        [
                            (
                                "TAUS_TRUEHADTAU_SME_TES_INSITUEXP",
                                "INSITUEXP",
                                (0.0, 0.0, 0.5, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_SME_TES_INSITUFIT",
                                "INSITUFIT",
                                (0.4901960784313725, 1.0, 0.4775458570524984, 1.0),
                                "solid",
                            ),
                            (
                                "TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE",
                                "MODEL_CLOSURE",
                                (0.5, 0.0, 0.0, 1.0),
                                "solid",
                            ),
                        ],
                        f"{var}_sys_OTHER_{sh_bin_label}_{n_iter}iter.png",
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
