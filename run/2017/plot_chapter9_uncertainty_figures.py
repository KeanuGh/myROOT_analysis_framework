"""Make thesis-ready Chapter 9 uncertainty plots from cached outputs.

This script intentionally reads only the cached ROOT files under
``outputs/analysis_shadow_unfold``. It does not open the DTA input ntuples or
run event loops.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

OUTPUT_DIR = (
    Path("outputs")
    / "analysis_shadow_unfold"
    / "plots"
    / "no_shadow_bin"
    / "MTW"
    / "chapter9_uncertainties"
)
RESPONSE_ROOT = (
    Path("outputs") / "analysis_shadow_unfold" / "response" / "root" / "wtaunu_had.root"
)
MEASURED_ROOT = (
    Path("outputs")
    / "analysis_shadow_unfold"
    / "measured"
    / "root"
    / "analysis_shadow_unfold_measured.root"
)

NOMINAL_RESPONSE_DIR = "T_s1thv_NOMINAL/no_shadow_bin_medium_truth_reco_tau"
NOMINAL_TRUTH_DIR = "T_s1thv_NOMINAL/no_shadow_bin_truth_tau"

plt.style.use(hep.style.ATLAS)


def clone(root_file: ROOT.TFile, path: str) -> ROOT.TH1:
    hist = root_file.Get(path)
    if not hist:
        raise KeyError(f"Missing histogram: {path}")
    out = hist.Clone(Path(path).name + "_clone")
    out.SetDirectory(0)
    return out


def bin_edges(hist: ROOT.TH1) -> np.ndarray:
    axis = hist.GetXaxis()
    return np.array([axis.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 2)])


def bin_values(hist: ROOT.TH1) -> np.ndarray:
    return np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])


def add_hist(target: ROOT.TH1, source: ROOT.TH1, scale: float = 1.0) -> ROOT.TH1:
    target.Add(source, scale)
    return target


def unfold(hist: ROOT.TH1, response: ROOT.RooUnfoldResponse, iterations: int, name: str) -> ROOT.TH1:
    unfolded = ROOT.RooUnfoldBayes(response, hist, iterations).Hunfold()
    out = unfolded.Clone(name)
    out.SetDirectory(0)
    return out


def relative_envelope(
    nominal: ROOT.TH1,
    shifted: list[ROOT.TH1],
    name: str,
) -> ROOT.TH1:
    out = nominal.Clone(name)
    out.SetDirectory(0)
    out.Reset()
    for i in range(1, nominal.GetNbinsX() + 1):
        nom = nominal.GetBinContent(i)
        if nom == 0:
            continue
        max_shift = max(abs(h.GetBinContent(i) - nom) for h in shifted)
        out.SetBinContent(i, 100.0 * max_shift / abs(nom))
    return out


def plot_group(
    hists: list[ROOT.TH1],
    labels: list[str],
    colours: list[str],
    filename: str,
    *,
    title: str,
    ylabel: str = "Relative uncertainty / %",
    ylim: tuple[float, float] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for hist, label, colour in zip(hists, labels, colours, strict=True):
        ax.stairs(bin_values(hist), bin_edges(hist), label=label, color=colour, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlim(350, 2000)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ymax = max(float(np.nanmax(bin_values(hist))) for hist in hists)
        ax.set_ylim(0, max(1.0, ymax * 1.25))
    ax.set_xlabel(r"$m_{\mathrm{T}}^W$ [GeV]", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    hep.atlas.label(ax=ax, llabel="", loc=0)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def response_uncertainty_plots(response_file: ROOT.TFile) -> None:
    prefix = f"{NOMINAL_RESPONSE_DIR}/MTW_"
    groups = [
        (
            "no_shadow_bin_MTW_response_tes_calibration_uncertainty.png",
            "Tau energy scale: calibration and modelling",
            [
                ("TAUS_TRUEHADTAU_SME_TES_INSITUFIT_pct_uncert", "In-situ fit"),
                ("TAUS_TRUEHADTAU_SME_TES_INSITUEXP_pct_uncert", "In-situ extrapolation"),
                ("TAUS_TRUEHADTAU_SME_TES_PHYSICSLIST_pct_uncert", "Physics-list modelling"),
                ("TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE_pct_uncert", "Model closure"),
            ],
            ["tab:blue", "tab:orange", "tab:green", "tab:red"],
        ),
        (
            "no_shadow_bin_MTW_response_tes_detector_uncertainty.png",
            "Tau energy scale: detector response",
            [
                ("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_LowPt_pct_uncert", "Barrel, low-$p_{\\mathrm{T}}$"),
                ("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt_pct_uncert", "Barrel, high-$p_{\\mathrm{T}}$"),
                ("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_LowPt_pct_uncert", "Endcap, low-$p_{\\mathrm{T}}$"),
                ("TAUS_TRUEHADTAU_SME_TES_DETECTOR_Endcap_HighPt_pct_uncert", "Endcap, high-$p_{\\mathrm{T}}$"),
            ],
            ["tab:blue", "tab:orange", "tab:green", "tab:red"],
        ),
        (
            "no_shadow_bin_MTW_response_reconstruction_uncertainty.png",
            "Tau reconstruction and overlap removal",
            [
                ("TAUS_TRUEHADTAU_EFF_RECO_TOTAL_pct_uncert", "Tau reconstruction"),
                ("TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL_pct_uncert", "Electron overlap removal"),
            ],
            ["tab:blue", "tab:orange"],
        ),
        (
            "no_shadow_bin_MTW_response_trigger_uncertainty.png",
            "Tau trigger efficiency",
            [
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718_pct_uncert", "Statistical, data"),
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718_pct_uncert", "Statistical, simulation"),
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718_pct_uncert", "Systematic"),
            ],
            ["tab:blue", "tab:orange", "tab:green"],
        ),
        (
            "no_shadow_bin_MTW_response_tau_efficiency_uncertainty.png",
            "Tau efficiency",
            [
                ("TAUS_TRUEHADTAU_EFF_RECO_TOTAL_pct_uncert", "Tau reconstruction"),
                ("TAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL_pct_uncert", "Electron overlap removal"),
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_STATDATA161718_pct_uncert", "Trigger statistical, data"),
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_STATMC161718_pct_uncert", "Trigger statistical, simulation"),
                ("TAUS_TRUEHADTAU_EFF_TRIGGER_SYST161718_pct_uncert", "Trigger systematic"),
            ],
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
        ),
    ]
    for filename, title, entries, colours in groups:
        hists = [clone(response_file, prefix + name) for name, _ in entries]
        labels = [label for _, label in entries]
        ylim = (0.0, 10.5) if filename == "no_shadow_bin_MTW_response_tau_efficiency_uncertainty.png" else None
        plot_group(hists, labels, colours, filename, title=title, ylim=ylim)


def fake_source_plots(measured_file: ROOT.TFile, response_file: ROOT.TFile) -> None:
    reco = clone(response_file, f"{NOMINAL_RESPONSE_DIR}/MTW")
    truth = clone(response_file, f"{NOMINAL_TRUTH_DIR}/TruthMTW")
    matrix = response_file.Get(f"{NOMINAL_RESPONSE_DIR}/MTW_TruthMTW")
    if not matrix:
        raise KeyError("Missing nominal response matrix")
    matrix = matrix.Clone("nominal_response_matrix_clone")
    matrix.SetDirectory(0)
    response = ROOT.RooUnfoldResponse(reco, truth, matrix)

    nominal_fakes = clone(measured_file, "no_shadow_bin_MTW_nominal_fakes_reference")

    def nominal_data_sig_from(shifted_data_sig_name: str, shifted_fakes_name: str) -> ROOT.TH1:
        shifted_data_sig = clone(measured_file, shifted_data_sig_name)
        shifted_fakes = clone(measured_file, shifted_fakes_name)
        nominal = shifted_data_sig.Clone(shifted_data_sig_name + "_nominal_rebuilt")
        nominal.SetDirectory(0)
        nominal.Add(shifted_fakes)
        nominal.Add(nominal_fakes, -1.0)
        return nominal

    fake_stat_nominal = nominal_data_sig_from(
        "no_shadow_bin_MTW_JET_FAKE_FF_STAT_up_data_sig",
        "no_shadow_bin_MTW_JET_FAKE_FF_STAT_up_fakes",
    )
    fake_stat_up = clone(measured_file, "no_shadow_bin_MTW_JET_FAKE_FF_STAT_up_data_sig")
    fake_stat_down = clone(measured_file, "no_shadow_bin_MTW_JET_FAKE_FF_STAT_down_data_sig")

    met_window_nominal = nominal_data_sig_from(
        "no_shadow_bin_MTW_JET_FAKE_MET_WINDOW_up_data_sig",
        "no_shadow_bin_MTW_JET_FAKE_MET_WINDOW_up_fakes",
    )
    met_window_up = clone(measured_file, "no_shadow_bin_MTW_JET_FAKE_MET_WINDOW_up_data_sig")
    met_window_down = clone(measured_file, "no_shadow_bin_MTW_JET_FAKE_MET_WINDOW_down_data_sig")

    width_nominal = nominal_data_sig_from(
        "no_shadow_bin_TauTrackWidthPt1000PV_MTW_JET_FAKE_TAU_WIDTH_COMPOSITION_data_sig",
        "no_shadow_bin_TauTrackWidthPt1000PV_MTW_JET_FAKE_TAU_WIDTH_COMPOSITION_fakes",
    )
    width_shifted = clone(
        measured_file,
        "no_shadow_bin_TauTrackWidthPt1000PV_MTW_JET_FAKE_TAU_WIDTH_COMPOSITION_data_sig",
    )

    iter_count = 4
    fake_stat_rel = relative_envelope(
        unfold(fake_stat_nominal, response, iter_count, "fake_stat_nominal_unfolded"),
        [
            unfold(fake_stat_up, response, iter_count, "fake_stat_up_unfolded"),
            unfold(fake_stat_down, response, iter_count, "fake_stat_down_unfolded"),
        ],
        "fake_factor_stat_relative_uncertainty",
    )
    met_window_rel = relative_envelope(
        unfold(met_window_nominal, response, iter_count, "met_window_nominal_unfolded"),
        [
            unfold(met_window_up, response, iter_count, "met_window_up_unfolded"),
            unfold(met_window_down, response, iter_count, "met_window_down_unfolded"),
        ],
        "met_window_relative_uncertainty",
    )
    width_rel = relative_envelope(
        unfold(width_nominal, response, iter_count, "width_nominal_unfolded"),
        [unfold(width_shifted, response, iter_count, "width_shifted_unfolded")],
        "tau_width_relative_uncertainty",
    )

    plot_group(
        [fake_stat_rel],
        ["Fake-factor statistics"],
        ["tab:blue"],
        "no_shadow_bin_MTW_fake_factor_stat_uncertainty_clean.png",
        title="Jet-fake estimate: fake-factor statistics",
    )
    plot_group(
        [met_window_rel],
        [r"Low-$E_{\mathrm{T}}^{\mathrm{miss}}$ control-region transfer"],
        ["tab:green"],
        "no_shadow_bin_MTW_met_window_transfer_uncertainty_clean.png",
        title=r"Jet-fake estimate: low-$E_{\mathrm{T}}^{\mathrm{miss}}$ transfer",
    )
    plot_group(
        [width_rel],
        ["Tau-width composition"],
        ["tab:orange"],
        "no_shadow_bin_MTW_tau_width_composition_uncertainty_clean.png",
        title="Jet-fake estimate: tau-width composition",
    )


def main() -> None:
    with ROOT.TFile.Open(str(RESPONSE_ROOT), "READ") as response_file, ROOT.TFile.Open(
        str(MEASURED_ROOT), "READ"
    ) as measured_file:
        response_uncertainty_plots(response_file)
        fake_source_plots(measured_file, response_file)
    print(f"Wrote Chapter 9 uncertainty plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
