from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binnings import BINNINGS  # noqa: E402
from common import (  # noqa: E402
    FAKES_SOURCE,
    MC_SAMPLES,
    VALIDATION_OUTPUT,
    WP,
    fake_factor_bin_health,
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import get_th1_bin_errors, sum_th1s  # noqa: E402
from utils.variable_names import variable_data  # noqa: E402

YEAR = 2017
VARIABLE = "MTW"
CONFIG_LABEL = "MTW_shadow_bin_300"
MTW_SHADOW_MIN = 300
MTW_NOMINAL_MIN = 350
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
FORCE_REBUILD_HISTS = os.environ.get("VALIDATE_LOW_MET_FORCE_REBUILD") == "1"
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_TRANSFER_COMPARISON = True
PLOT_FAKE_ENRICHMENT = True
PLOT_MEDIUM_FAKE_CONTAMINATION = True
PLOT_THESIS_LIKE_REGION_STACKS = True
PLOT_THESIS_LIKE_PRONG_STACKS = True
PLOT_NOMINAL_LOW_MET_FAKE_FACTORS = True
PLOT_CURRENT_DATA_MC_FAKES_STACKS = True
PLOT_CHAPTER10_CURRENT_STACKS = True
PLOT_CHAPTER10_CURRENT_TABLES_AND_PIES = True
THESIS_LIKE_STACK_VARS = ("TauRNNJetScore", "TauBDTEleScore", "TauNCoreTracks")
THESIS_DATA_MC_STACK_VARS = (
    "MTW",
    "MET_met",
    "TauPt",
    "TauEta",
    "AbsDeltaPhi_tau_met",
    "TauPhi",
)
CHAPTER10_PRONG_STACK_VARS = THESIS_DATA_MC_STACK_VARS
CHAPTER10_CHARGE_STACK_VARS = THESIS_DATA_MC_STACK_VARS
FAKE_CONTAMINATION_VARS = ("MET_met", "TauRNNJetScore", "TauBDTEleScore")
CHAPTER10_TAUETA_BINS = np.array(
    [
        -2.5,
        -2.16666667,
        -1.83333333,
        -1.52,
        -1.37,
        -1.16666667,
        -0.83333333,
        -0.5,
        -0.16666667,
        0.16666667,
        0.5,
        0.83333333,
        1.16666667,
        1.37,
        1.52,
        1.83333333,
        2.16666667,
        2.5,
    ],
    dtype="double",
)
CHAPTER10_BACKGROUND_SAMPLES = ("diboson", "top", "wlnu", "wtaunu_lep", "zll")
CHAPTER10_BACKGROUND_COLOURS = {
    "diboson": "tab:purple",
    "top": "tab:gray",
    "wlnu": "tab:orange",
    "wtaunu_lep": "tab:pink",
    "zll": "tab:red",
}
CHAPTER10_FAKES_COLOUR = "tab:brown"
CHAPTER10_YIELD_TABLE_VARIABLE = "MTW"
CHAPTER10_COMPONENT_LABELS = {
    "signal": r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}$",
    "wtaunu_lep": r"$W\rightarrow\tau\nu\rightarrow\ell\nu$",
    "wlnu": r"$W\rightarrow(e/\mu)\nu$",
    "zll": r"$Z/\gamma^*\rightarrow\ell\ell/\nu\nu$",
    "top": "Top",
    "diboson": "Diboson",
    "fakes": "Jet-to-tau fakes",
}
CHAPTER10_PIE_LABELS = {
    "signal": r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}$",
    "wtaunu_lep": r"$W\rightarrow\tau\nu\rightarrow\ell\nu$",
    "wlnu": r"$W\rightarrow(e/\mu)\nu$",
    "zll": r"$Z/\gamma^*$",
    "top": "Top",
    "diboson": "Diboson",
    "fakes": "Fake jets",
}
CHAPTER10_COMPONENT_COLOURS = {
    "signal": "tab:red",
    "wtaunu_lep": "tab:pink",
    "wlnu": "tab:orange",
    "zll": "tab:green",
    "top": "tab:gray",
    "diboson": "tab:purple",
    "fakes": CHAPTER10_FAKES_COLOUR,
}
OUTPUT_DIR = VALIDATION_OUTPUT / "low_met_fake_region"
SUMMARY_PATH = OUTPUT_DIR / "low_met_fake_region_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_low_met_fake_region.root"
NOMINAL_MEASURED_ROOT = (
    REPO_ROOT
    / "outputs"
    / "analysis_shadow_unfold"
    / "measured"
    / "root"
    / "analysis_shadow_unfold_measured.root"
)


@dataclass(frozen=True)
class FakeMethod:
    key: str
    label: str
    cuts: tuple[Cut, ...]


@dataclass(frozen=True)
class ValidationTarget:
    key: str
    label: str
    cuts: tuple[Cut, ...]


PASS_RECO_PRESELECTION = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) "
    r"&& passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + "
    r"MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
PASS_MEDIUM = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
    r"(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
FAIL_MEDIUM = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
    r"(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
PASS_ETA = Cut(
    r"$|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
CHARGE_SPLITS = (
    ("tauplus", r"$q^\tau = +1$", Cut(r"$q^\tau = +1$", "TauCharge > 0")),
    ("tauminus", r"$q^\tau = -1$", Cut(r"$q^\tau = -1$", "TauCharge < 0")),
)
PASS_TRUETAU = Cut(
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || "
    "MatchedTruthParticle_isMuon == true || "
    "MatchedTruthParticle_isElectron == true",
)


def validation_target(analysis: Analysis, selection: str) -> ROOT.TH1:
    data = analysis.get_hist(
        VARIABLE,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                VARIABLE,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    target = data - nonfake
    target.SetName(f"{selection}_{VARIABLE}_data_minus_nonfake")
    target.SetDirectory(0)
    return target


def fake_enrichment_components(
    analysis: Analysis,
    selection: str,
    variable: str,
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    data = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    fake_like = data - nonfake
    for hist, suffix in (
        (data, "data"),
        (nonfake, "nonfake_mc"),
        (fake_like, "data_minus_nonfake"),
    ):
        hist.SetName(f"{selection}_{variable}_{suffix}")
        hist.SetDirectory(0)
    return data, nonfake, fake_like


def sum_fake_enrichment_components(
    analysis: Analysis,
    selections: tuple[str, ...],
    variable: str,
    name: str,
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    components = [
        fake_enrichment_components(analysis, selection, variable)
        for selection in selections
    ]
    data = sum_th1s(*[component[0] for component in components])
    nonfake = sum_th1s(*[component[1] for component in components])
    fake_like = sum_th1s(*[component[2] for component in components])
    for hist, suffix in (
        (data, "data"),
        (nonfake, "nonfake_mc"),
        (fake_like, "data_minus_nonfake"),
    ):
        hist.SetName(f"{name}_{variable}_{suffix}")
        hist.SetDirectory(0)
    return data, nonfake, fake_like


def load_root_histograms(root_file_path: Path, hist_names: tuple[str, ...]) -> list[ROOT.TH1]:
    loaded_hists: list[ROOT.TH1] = []
    root_file = ROOT.TFile.Open(str(root_file_path), "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT file: {root_file_path}")
    try:
        for hist_name in hist_names:
            hist = root_file.Get(hist_name)
            if not hist:
                raise KeyError(f"Missing histogram '{hist_name}' in {root_file_path}")
            cloned_hist = hist.Clone(hist_name)
            cloned_hist.SetDirectory(0)
            loaded_hists.append(cloned_hist)
    finally:
        root_file.Close()
    return loaded_hists


def variable_label(variable: str) -> str:
    label = variable_data[variable]["name"]
    if variable in {"MTW", "MET_met", "TauPt"}:
        label += " [GeV]"
    return label


def chapter10_axis_options(variable: str) -> tuple[bool, bool, str]:
    """Return log-x/log-y choices and y-axis label for thesis-style Chapter 10 stacks."""

    if variable in {"MTW", "MET_met", "TauPt"}:
        return True, True, "Events / GeV"
    if variable == "AbsDeltaPhi_tau_met":
        return False, True, "Events / Bin Width"
    return False, False, "Events / Bin Width"


def chapter10_yscale_tag(variable: str) -> str:
    return "log" if chapter10_axis_options(variable)[1] else "liny"


def th1_edges(hist: ROOT.TH1) -> np.ndarray:
    return np.asarray(
        [hist.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 2)],
        dtype=float,
    )


def th1_values(hist: ROOT.TH1, *, scale_by_bin_width: bool = False) -> np.ndarray:
    values = np.asarray(
        [hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)],
        dtype=float,
    )
    if scale_by_bin_width:
        values = values / np.diff(th1_edges(hist))
    return values


def th1_errors(hist: ROOT.TH1, *, scale_by_bin_width: bool = False) -> np.ndarray:
    errors = np.asarray(
        [hist.GetBinError(i) for i in range(1, hist.GetNbinsX() + 1)],
        dtype=float,
    )
    if scale_by_bin_width:
        errors = errors / np.diff(th1_edges(hist))
    return errors


def clone_hist(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    return clone


def chapter10_expected_binning(variable: str) -> np.ndarray | None:
    if variable == "TauEta":
        return CHAPTER10_TAUETA_BINS
    return None


def th1_has_expected_binning(hist: ROOT.TH1, variable: str) -> bool:
    expected = chapter10_expected_binning(variable)
    if expected is None:
        return True
    edges = th1_edges(hist)
    return len(edges) == len(expected) and np.allclose(edges, expected, rtol=0, atol=1e-6)


def dataset_has_histogram(
    analysis: Analysis,
    dataset: str,
    selection: str,
    variable: str,
    systematic: str = NOMINAL_NAME,
) -> bool:
    if (
        systematic not in analysis[dataset].histograms
        or selection not in analysis[dataset].histograms[systematic]
        or variable not in analysis[dataset].histograms[systematic][selection]
    ):
        return False
    return th1_has_expected_binning(
        analysis[dataset].histograms[systematic][selection][variable],
        variable,
    )


def analysis_has_histogram(analysis: Analysis, hist_name: str, variable: str) -> bool:
    return hist_name in analysis.histograms and th1_has_expected_binning(
        analysis.histograms[hist_name],
        variable,
    )


def ensure_chapter10_histogram_cache(
    analysis: Analysis,
    *,
    variables: tuple[str, ...],
    signal_selections: tuple[str, ...],
) -> list[str]:
    """Regenerate missing Chapter 10 nominal histograms in batched dataset passes."""

    missing_by_dataset: dict[str, list[tuple[str, str]]] = {}
    for dataset in ("data", "wtaunu_had"):
        for selection in signal_selections:
            for variable in variables:
                if not dataset_has_histogram(analysis, dataset, selection, variable):
                    missing_by_dataset.setdefault(dataset, []).append((selection, variable))

    for dataset in CHAPTER10_BACKGROUND_SAMPLES:
        for selection in signal_selections:
            true_selection = f"trueTau_{selection}"
            for variable in variables:
                if not dataset_has_histogram(analysis, dataset, true_selection, variable):
                    missing_by_dataset.setdefault(dataset, []).append((true_selection, variable))

    regenerated: list[str] = []
    for dataset_name, missing in missing_by_dataset.items():
        dataset = analysis[dataset_name]
        missing_text = ", ".join(
            f"{selection}:{variable}" for selection, variable in missing[:6]
        )
        if len(missing) > 6:
            missing_text += f", ... ({len(missing)} total)"
        analysis.logger.info(
            "Regenerating Chapter 10 cached histograms for %s; missing %s",
            dataset_name,
            missing_text,
        )
        dataset.gen_all_histograms(do_prints=False)
        dataset.export_histograms(analysis.paths.root_dir / f"{dataset_name}.root")
        regenerated.append(dataset_name)

    return regenerated


def plot_chapter10_stack(
    *,
    output_path: Path,
    variable: str,
    title: str,
    data: ROOT.TH1,
    signal: ROOT.TH1,
    backgrounds: list[tuple[str, ROOT.TH1, str]],
    fakes: ROOT.TH1,
) -> None:
    """Draw a Chapter-10-style stack from cached/current histograms."""

    logx, logy, ylabel = chapter10_axis_options(variable)
    xlabel = variable_label(variable)
    edges = th1_edges(data)
    stack_hists = [hist for _label, hist, _colour in backgrounds] + [fakes]
    stack_labels = [label for label, _hist, _colour in backgrounds] + ["Fake Jets"]
    stack_colours = [colour for _label, _hist, colour in backgrounds] + [CHAPTER10_FAKES_COLOUR]
    total_prediction = sum_th1s(*stack_hists, signal)
    total_prediction.SetName(f"{data.GetName()}_{variable}_chapter10_total_prediction")
    total_prediction.SetDirectory(0)

    stack_values = [
        th1_values(hist, scale_by_bin_width=True)
        for hist in stack_hists
    ]
    data_values = th1_values(data, scale_by_bin_width=True)
    data_errors = th1_errors(data, scale_by_bin_width=True)
    total_values = th1_values(total_prediction, scale_by_bin_width=True)
    total_errors = th1_errors(total_prediction, scale_by_bin_width=True)

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    hep.histplot(
        stack_values,
        bins=edges,
        ax=ax,
        stack=True,
        histtype="fill",
        color=stack_colours,
        edgecolor="k",
        linewidth=1,
        alpha=0.85,
        label=stack_labels,
    )
    hep.histplot(
        total_values,
        bins=edges,
        ax=ax,
        histtype="step",
        color="red",
        linewidth=1.4,
        label=r"$W\rightarrow\tau\nu\rightarrow\mathrm{had}$",
    )

    err_top = total_values + total_errors
    err_bottom = np.clip(total_values - total_errors, 0.0, None)
    ax.fill_between(
        edges,
        np.r_[err_bottom, err_bottom[-1]],
        np.r_[err_top, err_top[-1]],
        step="post",
        color="grey",
        alpha=0.3,
        hatch="/",
        label="Stat. Err.",
    )

    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.5 * np.diff(edges)
    ax.errorbar(
        centres,
        data_values,
        xerr=widths,
        yerr=data_errors,
        fmt="o",
        color="k",
        markersize=4,
        linewidth=1,
        capsize=0,
        label="Data",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = np.divide(
            data_values,
            total_values,
            out=np.full_like(data_values, np.nan),
            where=total_values != 0,
        )
        ratio_errors = np.divide(
            data_errors,
            total_values,
            out=np.zeros_like(data_errors),
            where=total_values != 0,
        )
        ratio_band = np.divide(
            total_errors,
            total_values,
            out=np.zeros_like(total_errors),
            where=total_values != 0,
        )

    ratio_ax.fill_between(
        edges,
        np.r_[1.0 - ratio_band, 1.0 - ratio_band[-1]],
        np.r_[1.0 + ratio_band, 1.0 + ratio_band[-1]],
        step="post",
        color="grey",
        alpha=0.3,
        hatch="/",
    )
    ratio_ax.errorbar(
        centres,
        ratio_values,
        xerr=widths,
        yerr=ratio_errors,
        fmt="o",
        color="k",
        markersize=4,
        linewidth=1,
        capsize=0,
    )
    ratio_ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ratio_ax.set_ylim(0.5, 1.5)
    ratio_ax.set_ylabel("Data / MC")

    hep.atlas.label(ax=ax, llabel="", rlabel=title, loc=0)
    ax.set_ylabel(ylabel)
    ratio_ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
        ratio_ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        positive_values = np.concatenate(
            [
                data_values[data_values > 0],
                total_values[total_values > 0],
            ]
        )
        if positive_values.size:
            ax.set_ylim(
                bottom=max(np.nanmin(positive_values) / 20.0, 1e-5),
                top=np.nanmax(np.r_[data_values + data_errors, err_top, total_values]) * 30.0,
            )
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylim(
            top=np.nanmax(np.r_[data_values + data_errors, err_top, total_values]) * 1.35
        )
    ax.set_xlim(edges[0], edges[-1])
    ratio_ax.set_xlim(edges[0], edges[-1])
    handles, labels = ax.get_legend_handles_labels()
    legend_kwargs = {"loc": "upper right", "fontsize": 9}
    if variable == "AbsDeltaPhi_tau_met":
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (0.02, 0.82),
            "fontsize": 8,
        }
    ax.legend(list(reversed(handles)), list(reversed(labels)), **legend_kwargs)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def ensure_histogram_slot(analysis: Analysis, dataset: str, selection: str) -> None:
    dataset_obj = analysis[dataset]
    dataset_obj.histograms.setdefault(NOMINAL_NAME, {})
    dataset_obj.histograms[NOMINAL_NAME].setdefault(selection, {})


def current_data_mc_fakes_components(
    analysis: Analysis,
    selection: str,
    variable: str,
    fake_hists: tuple[ROOT.TH1, ...],
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    ensure_histogram_slot(analysis, "wtaunu_had", selection)
    signal = analysis.get_hist(
        variable,
        dataset="wtaunu_had",
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=False,
    )
    signal.SetName(f"{selection}_{variable}_signal")
    signal.SetDirectory(0)
    simulated_background = sum_th1s(
        *[
            (
                ensure_histogram_slot(
                    analysis,
                    mc_sample,
                    f"trueTau_{selection}",
                )
                or analysis.get_hist(
                    variable,
                    dataset=mc_sample,
                    systematic=NOMINAL_NAME,
                    selection=f"trueTau_{selection}",
                    allow_generation=False,
                )
            )
            for mc_sample in MC_SAMPLES
            if mc_sample != "wtaunu_had"
        ]
    )
    simulated_background.SetName(f"{selection}_{variable}_simulated_background")
    simulated_background.SetDirectory(0)
    fakes = sum_th1s(*fake_hists)
    fakes.SetName(f"{selection}_{variable}_data_driven_jet_fakes")
    fakes.SetDirectory(0)
    ensure_histogram_slot(analysis, analysis.data_sample, selection)
    data = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=False,
    )
    data.SetName(f"{selection}_{variable}_data")
    data.SetDirectory(0)
    return signal, simulated_background, fakes, data


def current_data_mc_fakes_process_components(
    analysis: Analysis,
    selection: str,
    variable: str,
    fake_hists: tuple[ROOT.TH1, ...],
) -> tuple[ROOT.TH1, list[tuple[str, ROOT.TH1, str]], ROOT.TH1, ROOT.TH1]:
    """Build current Chapter 10 components without double-counting jet-fake-like MC."""

    ensure_histogram_slot(analysis, "wtaunu_had", selection)
    signal = clone_hist(
        analysis.get_hist(
            variable,
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection=selection,
            allow_generation=False,
        ),
        f"{selection}_{variable}_signal",
    )
    backgrounds: list[tuple[str, ROOT.TH1, str]] = []
    for sample in CHAPTER10_BACKGROUND_SAMPLES:
        true_selection = f"trueTau_{selection}"
        ensure_histogram_slot(analysis, sample, true_selection)
        hist = clone_hist(
            analysis.get_hist(
                variable,
                dataset=sample,
                systematic=NOMINAL_NAME,
                selection=true_selection,
                allow_generation=False,
            ),
            f"{selection}_{variable}_{sample}_mc_contamination",
        )
        backgrounds.append(
            (
                analysis[sample].label,
                hist,
                CHAPTER10_BACKGROUND_COLOURS.get(sample, analysis[sample].colour),
            )
        )

    fakes = sum_th1s(*fake_hists)
    fakes.SetName(f"{selection}_{variable}_data_driven_jet_fakes")
    fakes.SetDirectory(0)
    data = clone_hist(
        analysis.get_hist(
            variable,
            dataset="data",
            systematic=NOMINAL_NAME,
            selection=selection,
            allow_generation=False,
        ),
        f"{selection}_{variable}_data",
    )
    return signal, backgrounds, fakes, data


def sum_chapter10_process_components(
    components: list[tuple[ROOT.TH1, list[tuple[str, ROOT.TH1, str]], ROOT.TH1, ROOT.TH1]],
    *,
    name_prefix: str,
    variable: str,
) -> tuple[ROOT.TH1, list[tuple[str, ROOT.TH1, str]], ROOT.TH1, ROOT.TH1]:
    """Sum prong-level Chapter 10 components into a split-level plot."""

    signal = sum_th1s(*[component[0] for component in components])
    signal.SetName(f"{name_prefix}_{variable}_signal")
    signal.SetDirectory(0)
    backgrounds: list[tuple[str, ROOT.TH1, str]] = []
    for i, sample in enumerate(CHAPTER10_BACKGROUND_SAMPLES):
        label = components[0][1][i][0]
        colour = components[0][1][i][2]
        hist = sum_th1s(*[component[1][i][1] for component in components])
        hist.SetName(f"{name_prefix}_{variable}_{sample}_mc_contamination")
        hist.SetDirectory(0)
        backgrounds.append((label, hist, colour))
    fakes = sum_th1s(*[component[2] for component in components])
    fakes.SetName(f"{name_prefix}_{variable}_data_driven_jet_fakes")
    fakes.SetDirectory(0)
    data = sum_th1s(*[component[3] for component in components])
    data.SetName(f"{name_prefix}_{variable}_data")
    data.SetDirectory(0)
    return signal, backgrounds, fakes, data


def hist_integral_error(hist: ROOT.TH1) -> float:
    return float(
        np.sqrt(
            sum(
                hist.GetBinError(bin_idx) ** 2
                for bin_idx in range(1, hist.GetNbinsX() + 1)
            )
        )
    )


def yield_text(hist: ROOT.TH1) -> str:
    return f"${hist_integral(hist):.2f} \\pm {hist_integral_error(hist):.2f}$"


def data_yield_text(hist: ROOT.TH1) -> str:
    return f"${hist_integral(hist):.0f} \\pm {hist_integral_error(hist):.2f}$"


def chapter10_component_hists(
    *,
    signal: ROOT.TH1,
    backgrounds: list[tuple[str, ROOT.TH1, str]],
    fakes: ROOT.TH1,
) -> dict[str, ROOT.TH1]:
    components = {"signal": signal, "fakes": fakes}
    for sample, (_label, hist, _colour) in zip(CHAPTER10_BACKGROUND_SAMPLES, backgrounds):
        components[sample] = hist
    return components


def write_chapter10_yield_tables(
    *,
    output_dir: Path,
    prong_components: dict[int, dict[str, ROOT.TH1]],
    prong_data: dict[int, ROOT.TH1],
    charge_components: dict[str, dict[str, ROOT.TH1]],
    charge_data: dict[str, ROOT.TH1],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def total_hist(component_map: dict[str, ROOT.TH1], keys: tuple[str, ...]) -> ROOT.TH1:
        total = sum_th1s(*[component_map[key] for key in keys])
        total.SetDirectory(0)
        return total

    background_keys = (*CHAPTER10_BACKGROUND_SAMPLES, "fakes")
    prediction_keys = ("signal", *background_keys)

    prong_tex = output_dir / "chapter10_medium_event_counts_prongs.tex"
    prong_md = output_dir / "chapter10_medium_event_counts_prongs.md"
    prong_rows = [
        ("signal", CHAPTER10_COMPONENT_LABELS["signal"]),
        ("wtaunu_lep", CHAPTER10_COMPONENT_LABELS["wtaunu_lep"]),
        ("wlnu", CHAPTER10_COMPONENT_LABELS["wlnu"]),
        ("zll", CHAPTER10_COMPONENT_LABELS["zll"]),
        ("top", CHAPTER10_COMPONENT_LABELS["top"]),
        ("diboson", CHAPTER10_COMPONENT_LABELS["diboson"]),
        ("fakes", CHAPTER10_COMPONENT_LABELS["fakes"]),
    ]
    prong_lines = [
        r"\begin{tabular}{l|l|l}",
        r"    \hline",
        r"    Process & 1-prong SR weighted events & 3-prong SR weighted events \\",
        r"    \hline",
    ]
    prong_md_lines = [
        "| Process | 1-prong SR weighted events | 3-prong SR weighted events |",
        "|---|---:|---:|",
    ]
    for key, label in prong_rows:
        one = yield_text(prong_components[1][key])
        three = yield_text(prong_components[3][key])
        prong_lines.append(f"    {label} & {one} & {three} \\\\")
        prong_md_lines.append(f"| {label} | {one} | {three} |")
    for key, label, keys in (
        ("total_background", "Total background", background_keys),
        ("total_prediction", "Total prediction", prediction_keys),
    ):
        _ = key
        one_hist = total_hist(prong_components[1], keys)
        three_hist = total_hist(prong_components[3], keys)
        one = yield_text(one_hist)
        three = yield_text(three_hist)
        prong_lines.append(f"    {label} & {one} & {three} \\\\")
        prong_md_lines.append(f"| {label} | {one} | {three} |")
    prong_lines.append(
        f"    Data 2017 & {data_yield_text(prong_data[1])} & "
        f"{data_yield_text(prong_data[3])} \\\\"
    )
    prong_md_lines.append(
        f"| Data 2017 | {data_yield_text(prong_data[1])} | "
        f"{data_yield_text(prong_data[3])} |"
    )
    prong_lines.extend([r"    \hline", r"\end{tabular}"])
    prong_tex.write_text("\n".join(prong_lines) + "\n")
    write_markdown(prong_md, prong_md_lines)

    charge_tex = output_dir / "chapter10_medium_event_counts_plusminus.tex"
    charge_md = output_dir / "chapter10_medium_event_counts_plusminus.md"
    charge_rows = prong_rows
    charge_lines = [
        r"\begin{tabular}{l|l|l}",
        r"    \hline",
        r"    Process & $\tau^+$ SR weighted events & $\tau^-$ SR weighted events \\",
        r"    \hline",
    ]
    charge_md_lines = [
        "| Process | tau+ SR weighted events | tau- SR weighted events |",
        "|---|---:|---:|",
    ]
    for key, label in charge_rows:
        plus = yield_text(charge_components["tauplus"][key])
        minus = yield_text(charge_components["tauminus"][key])
        charge_lines.append(f"    {label} & {plus} & {minus} \\\\")
        charge_md_lines.append(f"| {label} | {plus} | {minus} |")
    for key, label, keys in (
        ("total_background", "Total background", background_keys),
        ("total_prediction", "Total prediction", prediction_keys),
    ):
        _ = key
        plus_hist = total_hist(charge_components["tauplus"], keys)
        minus_hist = total_hist(charge_components["tauminus"], keys)
        plus = yield_text(plus_hist)
        minus = yield_text(minus_hist)
        charge_lines.append(f"    {label} & {plus} & {minus} \\\\")
        charge_md_lines.append(f"| {label} | {plus} | {minus} |")
    charge_lines.append(
        f"    Data 2017 & {data_yield_text(charge_data['tauplus'])} & "
        f"{data_yield_text(charge_data['tauminus'])} \\\\"
    )
    charge_md_lines.append(
        f"| Data 2017 | {data_yield_text(charge_data['tauplus'])} | "
        f"{data_yield_text(charge_data['tauminus'])} |"
    )
    charge_lines.extend([r"    \hline", r"\end{tabular}"])
    charge_tex.write_text("\n".join(charge_lines) + "\n")
    write_markdown(charge_md, charge_md_lines)

    summary_md = output_dir / "chapter10_current_tables_summary.md"
    write_markdown(
        summary_md,
        [
            "# Chapter 10 Current Yield Tables",
            "",
            "Weighted final-region yields for the Medium tau-identification working point.",
            "The data-driven jet-to-tau fake estimate is included as a background component.",
            "Uncertainties are statistical only.",
            "",
            f"- prong LaTeX table: `{prong_tex}`",
            f"- charge LaTeX table: `{charge_tex}`",
            f"- prong Markdown table: `{prong_md}`",
            f"- charge Markdown table: `{charge_md}`",
        ],
    )
    return prong_tex, charge_tex, summary_md


def plot_chapter10_composition_pie(
    *,
    output_path: Path,
    title: str,
    components: dict[str, ROOT.TH1],
) -> None:
    ordered_keys = ("signal", "wtaunu_lep", "wlnu", "zll", "top", "diboson", "fakes")
    values = np.asarray([hist_integral(components[key]) for key in ordered_keys])
    positive = values > 0
    plot_values = values[positive]
    plot_labels = [
        CHAPTER10_PIE_LABELS[key]
        for key, keep in zip(ordered_keys, positive)
        if keep
    ]
    plot_colours = [
        CHAPTER10_COMPONENT_COLOURS[key]
        for key, keep in zip(ordered_keys, positive)
        if keep
    ]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    wedges, texts, autotexts = ax.pie(
        plot_values,
        labels=plot_labels,
        colors=plot_colours,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 1.0 else "",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 9},
        pctdistance=0.72,
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("white")
    ax.axis("equal")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    derivation_methods = (
        FakeMethod(
            "current_mtw_shadow_metlt170",
            "current MTW-shadow CR, MET < 170",
            (
                Cut(
                    r"shadow $m_T^W$ sideband",
                    f"(MTW >= {MTW_SHADOW_MIN:g}) && (MTW < {MTW_NOMINAL_MIN:g})",
                ),
                Cut("MET < 170", "MET_met < 170"),
            ),
        ),
        FakeMethod(
            "low_met_fake_enriched",
            "low-MET fake-enriched CR, MET < 100",
            (Cut("MET < 100", "MET_met < 100"),),
        ),
    )
    validation_targets = (
        ValidationTarget(
            "nominal_mtw_control_metlt170",
            "MTW >= 350, MET < 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET < 170", "MET_met < 170"),
            ),
        ),
        ValidationTarget(
            "signal_like_metgt170",
            "MTW >= 350, MET >= 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET >= 170", "MET_met >= 170"),
            ),
        ),
    )

    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}

    for prong in PRONGS:
        prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
        for method in derivation_methods:
            prefix = f"{CONFIG_LABEL}_{method.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_derive_passID"] = (
                base_cuts + list(method.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_derive_failID"] = (
                base_cuts + list(method.cuts) + [FAIL_MEDIUM, prong_cut]
            )

        for target in validation_targets:
            prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + list(target.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + list(target.cuts) + [FAIL_MEDIUM, prong_cut]
            )

            for charge_key, _charge_label, charge_cut in CHARGE_SPLITS:
                charge_prefix = (
                    f"{CONFIG_LABEL}_{target.key}_{WP}_{charge_key}_{prong}prong"
                )
                data_selections[f"{charge_prefix}_validate_passID"] = (
                    base_cuts
                    + list(target.cuts)
                    + [PASS_MEDIUM, prong_cut, charge_cut]
                )
                data_selections[f"{charge_prefix}_validate_failID"] = (
                    base_cuts
                    + list(target.cuts)
                    + [FAIL_MEDIUM, prong_cut, charge_cut]
                )

    thesis_like_stack_selections = {
        f"{CONFIG_LABEL}_low_met_{WP}_CR_passID": (
            base_cuts + [Cut("MET < 100", "MET_met < 100"), PASS_MEDIUM]
        ),
        f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID": (
            base_cuts
            + [
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET >= 170", "MET_met >= 170"),
                PASS_MEDIUM,
            ]
        ),
    }
    signal_like_target = next(
        target for target in validation_targets if target.key == "signal_like_metgt170"
    )
    for charge_key, _charge_label, charge_cut in CHARGE_SPLITS:
        thesis_like_stack_selections[f"{CONFIG_LABEL}_signal_like_{WP}_{charge_key}_SR_passID"] = (
            base_cuts + list(signal_like_target.cuts) + [PASS_MEDIUM, charge_cut]
        )
    data_selections.update(thesis_like_stack_selections)

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = FORCE_REBUILD_HISTS or (
        RUN_EVENT_LOOPS_IF_CACHE_MISSING and (not LOAD_SAVED_HISTS or not cache_exists)
    )

    chapter10_binnings = dict(BINNINGS)
    chapter10_binnings["TauEta"] = CHAPTER10_TAUETA_BINS

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_low_met_fake_region",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            VARIABLE,
            FAKES_SOURCE,
            "MET_met",
            "TauEta",
            "AbsDeltaPhi_tau_met",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={
            VARIABLE,
            FAKES_SOURCE,
            *THESIS_LIKE_STACK_VARS,
            *CHAPTER10_CHARGE_STACK_VARS,
            *FAKE_CONTAMINATION_VARS,
        },
        binnings={"": chapter10_binnings},
    )

    loaded_hists = (
        LOAD_SAVED_HISTS
        and not FORCE_REBUILD_HISTS
        and analysis.load_hists_if_available(CACHE_FILE)
    )
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing low-MET fake-region cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    chapter10_signal_selections = (
        f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID",
        *(
            f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{prong}prong_validate_passID"
            for prong in PRONGS
        ),
        *(
            f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{charge_key}_{prong}prong_validate_passID"
            for charge_key, _charge_label, _charge_cut in CHARGE_SPLITS
            for prong in PRONGS
        ),
    )
    chapter10_regenerated_datasets: list[str] = []
    if PLOT_CURRENT_DATA_MC_FAKES_STACKS or PLOT_CHAPTER10_CURRENT_STACKS:
        chapter10_regenerated_datasets = ensure_chapter10_histogram_cache(
            analysis,
            variables=THESIS_DATA_MC_STACK_VARS,
            signal_selections=chapter10_signal_selections,
        )

    lines = [
        "# Low-MET fake-enriched fake-factor validation",
        "",
        "This validation tests an ATLAS-like low-MET fake-enriched fake-factor derivation "
        "against the current MTW-shadow control-region derivation. It is a validation-only "
        "study, not a nominal unfolding change.",
        "",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "- Chapter 10 cache regeneration: "
        + (
            "`" + "`, `".join(chapter10_regenerated_datasets) + "`"
            if chapter10_regenerated_datasets
            else "`none`"
        ),
        "",
        "## Transfer summary",
        "",
        "| Method | Target | Prong | CR numerator | CR denominator | "
        "Negative numerator bins | Tiny/non-positive denominator bins | "
        "Validation fail-ID input | Predicted fakes | Validation target | "
        "Prediction / target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    comparison_hists: dict[tuple[str, int], list[tuple[str, ROOT.TH1]]] = {}
    fake_enrichment_hists: dict[
        tuple[int, str],
        tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1],
    ] = {}

    for target in validation_targets:
        for prong in PRONGS:
            target_prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            target_hist = validation_target(analysis, validate_pass)
            comparison_hists[(target.key, prong)] = [
                ("Data - simulated backgrounds", target_hist)
            ]

            for method in derivation_methods:
                method_prefix = f"{CONFIG_LABEL}_{method.key}_{WP}_{prong}prong"
                derive_pass = f"{method_prefix}_derive_passID"
                derive_fail = f"{method_prefix}_derive_failID"
                estimate_name = f"{method_prefix}_{target.key}_{FAKES_SOURCE}_src"

                if not loaded_hists:
                    analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        (VARIABLE,),
                        derive_pass,
                        derive_fail,
                        validate_pass,
                        validate_fail,
                        f"trueTau_{derive_pass}",
                        f"trueTau_{derive_fail}",
                        f"trueTau_{validate_fail}",
                        name=estimate_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

                numerator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_numerator"]
                denominator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_denominator"]
                validation_fail_input = analysis.histograms[
                    f"{estimate_name}_{FAKES_SOURCE}_FF_fakes_data_est"
                ]
                prediction = analysis.histograms[
                    f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                    numerator, denominator
                )
                predicted_integral = hist_integral(prediction)
                target_integral = hist_integral(target_hist)

                lines.append(
                    f"| {method.label} | {target.label} | {prong} | "
                    f"{hist_integral(numerator):.3f} | {hist_integral(denominator):.3f} | "
                    f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                    f"{hist_integral(validation_fail_input):.3f} | {predicted_integral:.3f} | "
                    f"{target_integral:.3f} | {ratio(predicted_integral, target_integral):.3f} |"
                )
                comparison_hists[(target.key, prong)].append((method.label, prediction))

    low_met_method = next(
        method for method in derivation_methods if method.key == "low_met_fake_enriched"
    )
    lines.extend(
        [
            "",
            "## Low-MET control-region fake enrichment",
            "",
            "The fake-like component is computed as data minus the simulated "
            "backgrounds in the same region. A small simulated-background fraction indicates a "
            "region dominated by jets misidentified as tau candidates.",
            "",
            "| Region | Prong | Data | Simulated backgrounds | "
            "Inferred jet-fake component | Fake-like / data | Simulated background / data |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for prong in PRONGS:
        method_prefix = f"{CONFIG_LABEL}_{low_met_method.key}_{WP}_{prong}prong"
        for region_key, region_label in (
            ("derive_passID", "Pass-ID numerator"),
            ("derive_failID", "Anti-ID denominator"),
        ):
            selection = f"{method_prefix}_{region_key}"
            data_hist, nonfake_hist, fake_like_hist = fake_enrichment_components(
                analysis,
                selection,
                FAKES_SOURCE,
            )
            fake_enrichment_hists[(prong, region_key)] = (
                data_hist,
                nonfake_hist,
                fake_like_hist,
            )
            data_yield = hist_integral(data_hist)
            nonfake_yield = hist_integral(nonfake_hist)
            fake_like_yield = hist_integral(fake_like_hist)
            lines.append(
                f"| {region_label} | {prong} | {data_yield:.3f} | "
                f"{nonfake_yield:.3f} | {fake_like_yield:.3f} | "
                f"{ratio(fake_like_yield, data_yield):.3f} | "
                f"{ratio(nonfake_yield, data_yield):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "The low-MET fake-enriched method is interesting only if it improves the "
            "`Prediction / target` ratio without creating unhealthy source bins. If it "
            "does better in the `MTW >= 350, MET < 170` target but not in the signal-like "
            "`MET >= 170` proxy, the difference should be treated as a transfer uncertainty "
            "rather than a nominal model change.",
            "",
            "The current MTW-shadow control-region row is included so the low-MET method can "
            "be judged against the current fake-factor construction in the same script and "
            "with the same target definitions.",
        ]
    )

    if PLOT_TRANSFER_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "low_met_fake_region"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])
        for (target_key, prong), labelled_hists in comparison_hists.items():
            filename = f"{CONFIG_LABEL}_{target_key}_{prong}prong_low_met_fake_region.png"
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange"],
                histstyle=["step", "step", "step"],
                linestyle=["-", "-", "--"],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{CONFIG_LABEL} | {target_key} | {prong}-prong",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                logx=True,
                ratio_plot=True,
                ratio_label="Prediction / target",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=filename,
            )
            lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_NOMINAL_LOW_MET_FAKE_FACTORS:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "fake_factors"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Nominal low-MET fake-factor plots", ""])
        low_met_fake_factors = load_root_histograms(
            NOMINAL_MEASURED_ROOT,
            (
                "no_shadow_bin_medium_3prong_lowMET_TauPt_FF",
                "no_shadow_bin_medium_1prong_lowMET_TauPt_FF",
            ),
        )
        filename = "no_shadow_bin_medium_TauPt_lowMET_prong_fake_factors.png"
        analysis.plot(
            low_met_fake_factors,
            label=["3-prong", "1-prong"],
            colour=["tab:orange", "tab:blue"],
            histstyle=["step", "step"],
            xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
            ylabel="Fake factor",
            title="No shadow bin | low-MET fake factors",
            kind="overlay",
            do_stat=False,
            do_syst=False,
            label_params={"llabel": "", "loc": 0},
            legend_params={"fontsize": 10, "loc": "upper right"},
            filename=filename,
        )
        lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    built_current_data_mc_fakes = False
    if PLOT_CURRENT_DATA_MC_FAKES_STACKS:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "current_data_mc_fakes_stacks"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Current data/MC plus fake-estimate stacks", ""])

        signal_like_target = next(
            target for target in validation_targets if target.key == "signal_like_metgt170"
        )
        fake_histograms: dict[tuple[str, int], ROOT.TH1] = {}
        for prong in PRONGS:
            method_prefix = f"{CONFIG_LABEL}_{low_met_method.key}_{WP}_{prong}prong"
            target_prefix = f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{prong}prong"
            derive_pass = f"{method_prefix}_derive_passID"
            derive_fail = f"{method_prefix}_derive_failID"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            estimate_name = f"{method_prefix}_{signal_like_target.key}_{FAKES_SOURCE}_src"
            missing_fakes = [
                variable
                for variable in THESIS_DATA_MC_STACK_VARS
                if not analysis_has_histogram(
                    analysis,
                    f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src",
                    variable,
                )
            ]
            if missing_fakes:
                built_current_data_mc_fakes = True
                analysis.do_fakes_estimate(
                    FAKES_SOURCE,
                    tuple(missing_fakes),
                    derive_pass,
                    derive_fail,
                    validate_pass,
                    validate_fail,
                    f"trueTau_{derive_pass}",
                    f"trueTau_{derive_fail}",
                    f"trueTau_{validate_fail}",
                    name=estimate_name,
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )
            for variable in THESIS_DATA_MC_STACK_VARS:
                fake_histograms[(variable, prong)] = analysis.histograms[
                    f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src"
                ]

        stack_selection = f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID"
        for variable in THESIS_DATA_MC_STACK_VARS:
            signal, simulated_background, fakes, data = current_data_mc_fakes_components(
                analysis,
                stack_selection,
                variable,
                tuple(fake_histograms[(variable, prong)] for prong in PRONGS),
            )
            mc_no_fakes = sum_th1s(signal, simulated_background)
            mc_no_fakes.SetName(f"{stack_selection}_{variable}_mc_signal_background")
            mc_no_fakes.SetDirectory(0)
            mc_with_fakes = sum_th1s(signal, simulated_background, fakes)
            mc_with_fakes.SetName(
                f"{stack_selection}_{variable}_mc_signal_background_fakes"
            )
            mc_with_fakes.SetDirectory(0)
            for yscale, logy in (("liny", False), ("log", True)):
                filename = f"{WP}_{variable}_current_signal_background_fakes_{yscale}.png"
                analysis.plot(
                    [data, mc_with_fakes, mc_no_fakes],
                    label=[
                        "Data",
                        "MC (signal + backgrounds) + fakes",
                        "MC (signal + backgrounds), no fakes",
                    ],
                    colour=["k", "tab:orange", "tab:blue"],
                    histstyle=["errorbar", "step", "step"],
                    uncert=[
                        get_th1_bin_errors(data),
                        get_th1_bin_errors(mc_with_fakes),
                        get_th1_bin_errors(mc_no_fakes),
                    ],
                    xlabel=variable_label(variable),
                    ylabel="Events",
                    title="Medium tau ID | signal region",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    ratio_plot=True,
                    ratio_label="Prediction / Data",
                    ratio_axlim=(0.5, 1.5),
                    logy=logy,
                    label_params={"llabel": "", "loc": 1},
                    legend_params={"fontsize": 9, "loc": "upper right"},
                    filename=filename,
                    sort=False,
                    capsize=0,
                )
                lines.append(f"- `{analysis.paths.plot_dir / filename}`")

        prong_plot_dir = analysis.paths.plot_dir / "prong_split"
        prong_plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Current prong-split data/MC plus fake-estimate stacks", ""])
        previous_plot_dir = analysis.paths.plot_dir
        analysis.paths.plot_dir = prong_plot_dir
        for prong in PRONGS:
            prong_selection = (
                f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{prong}prong_validate_passID"
            )
            for variable in CHAPTER10_PRONG_STACK_VARS:
                signal, simulated_background, fakes, data = current_data_mc_fakes_components(
                    analysis,
                    prong_selection,
                    variable,
                    (fake_histograms[(variable, prong)],),
                )
                mc_no_fakes = sum_th1s(signal, simulated_background)
                mc_no_fakes.SetName(f"{prong_selection}_{variable}_mc_signal_background")
                mc_no_fakes.SetDirectory(0)
                mc_with_fakes = sum_th1s(signal, simulated_background, fakes)
                mc_with_fakes.SetName(
                    f"{prong_selection}_{variable}_mc_signal_background_fakes"
                )
                mc_with_fakes.SetDirectory(0)
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = (
                        f"{WP}_{prong}prong_{variable}_current_signal_background_fakes_"
                        f"{yscale}.png"
                    )
                    analysis.plot(
                        [data, mc_with_fakes, mc_no_fakes],
                        label=[
                            "Data",
                            "MC (signal + backgrounds) + fakes",
                            "MC (signal + backgrounds), no fakes",
                        ],
                        colour=["k", "tab:orange", "tab:blue"],
                        histstyle=["errorbar", "step", "step"],
                        uncert=[
                            get_th1_bin_errors(data),
                            get_th1_bin_errors(mc_with_fakes),
                            get_th1_bin_errors(mc_no_fakes),
                        ],
                        xlabel=variable_label(variable),
                        ylabel="Events",
                        title=f"Medium tau ID | signal region | {prong}-prong",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        ratio_plot=True,
                        ratio_label="Prediction / Data",
                        ratio_axlim=(0.5, 1.5),
                        logy=logy,
                        label_params={"llabel": "", "loc": 1},
                        legend_params={"fontsize": 9, "loc": "upper right"},
                        filename=filename,
                        sort=False,
                        capsize=0,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")
        analysis.paths.plot_dir = previous_plot_dir

        charge_plot_dir = analysis.paths.plot_dir / "charge_split"
        charge_plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Current charge-split data/MC plus fake-estimate stacks", ""])
        previous_plot_dir = analysis.paths.plot_dir
        analysis.paths.plot_dir = charge_plot_dir
        all_charge_fake_histograms: dict[tuple[str, str, int], ROOT.TH1] = {}
        for charge_key, charge_label, _charge_cut in CHARGE_SPLITS:
            charge_fake_histograms: dict[tuple[str, int], ROOT.TH1] = {}
            for prong in PRONGS:
                method_prefix = (
                    f"{CONFIG_LABEL}_{low_met_method.key}_{WP}_{prong}prong"
                )
                target_prefix = (
                    f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{charge_key}_{prong}prong"
                )
                derive_pass = f"{method_prefix}_derive_passID"
                derive_fail = f"{method_prefix}_derive_failID"
                validate_pass = f"{target_prefix}_validate_passID"
                validate_fail = f"{target_prefix}_validate_failID"
                estimate_name = (
                    f"{method_prefix}_{charge_key}_{signal_like_target.key}_"
                    f"{FAKES_SOURCE}_src"
                )
                missing_fakes = [
                    variable
                    for variable in CHAPTER10_CHARGE_STACK_VARS
                    if not analysis_has_histogram(
                        analysis,
                        f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src",
                        variable,
                    )
                ]
                if missing_fakes:
                    built_current_data_mc_fakes = True
                    analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        tuple(missing_fakes),
                        derive_pass,
                        derive_fail,
                        validate_pass,
                        validate_fail,
                        f"trueTau_{derive_pass}",
                        f"trueTau_{derive_fail}",
                        f"trueTau_{validate_fail}",
                        name=estimate_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )
                    analysis.save_hists(filename=CACHE_FILE.name)
                for variable in CHAPTER10_CHARGE_STACK_VARS:
                    charge_fake_histograms[(variable, prong)] = analysis.histograms[
                        f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src"
                    ]
                    all_charge_fake_histograms[(charge_key, variable, prong)] = (
                        charge_fake_histograms[(variable, prong)]
                    )

            for variable in CHAPTER10_CHARGE_STACK_VARS:
                prong_components = [
                    current_data_mc_fakes_components(
                        analysis,
                        (
                            f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_"
                            f"{charge_key}_{prong}prong_validate_passID"
                        ),
                        variable,
                        (charge_fake_histograms[(variable, prong)],),
                    )
                    for prong in PRONGS
                ]
                signal = sum_th1s(*[component[0] for component in prong_components])
                simulated_background = sum_th1s(
                    *[component[1] for component in prong_components]
                )
                fakes = sum_th1s(*[component[2] for component in prong_components])
                data = sum_th1s(*[component[3] for component in prong_components])
                charge_selection = (
                    f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_"
                    f"{charge_key}_validate_passID"
                )
                for hist, suffix in (
                    (signal, "signal"),
                    (simulated_background, "simulated_background"),
                    (fakes, "data_driven_jet_fakes"),
                    (data, "data"),
                ):
                    hist.SetName(f"{charge_selection}_{variable}_{suffix}")
                    hist.SetDirectory(0)
                mc_no_fakes = sum_th1s(signal, simulated_background)
                mc_no_fakes.SetName(f"{charge_selection}_{variable}_mc_signal_background")
                mc_no_fakes.SetDirectory(0)
                mc_with_fakes = sum_th1s(signal, simulated_background, fakes)
                mc_with_fakes.SetName(
                    f"{charge_selection}_{variable}_mc_signal_background_fakes"
                )
                mc_with_fakes.SetDirectory(0)
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = (
                        f"{WP}_{charge_key}_{variable}_current_signal_background_fakes_"
                        f"{yscale}.png"
                    )
                    analysis.plot(
                        [data, mc_with_fakes, mc_no_fakes],
                        label=[
                            "Data",
                            "MC (signal + backgrounds) + fakes",
                            "MC (signal + backgrounds), no fakes",
                        ],
                        colour=["k", "tab:orange", "tab:blue"],
                        histstyle=["errorbar", "step", "step"],
                        uncert=[
                            get_th1_bin_errors(data),
                            get_th1_bin_errors(mc_with_fakes),
                            get_th1_bin_errors(mc_no_fakes),
                        ],
                        xlabel=variable_label(variable),
                        ylabel="Events",
                        title=f"Medium tau ID | signal region | {charge_label}",
                        kind="overlay",
                        do_stat=False,
                        do_syst=False,
                        ratio_plot=True,
                        ratio_label="Prediction / Data",
                        ratio_axlim=(0.5, 1.5),
                        logy=logy,
                        label_params={"llabel": "", "loc": 1},
                        legend_params={"fontsize": 9, "loc": "upper right"},
                        filename=filename,
                        sort=False,
                        capsize=0,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")
        analysis.paths.plot_dir = previous_plot_dir

        if PLOT_CHAPTER10_CURRENT_STACKS:
            chapter10_plot_dir = OUTPUT_DIR / "plots" / "chapter10_current_stacks"
            chapter10_plot_dir.mkdir(parents=True, exist_ok=True)
            lines.extend(["", "## Chapter 10 current thesis-style stacks", ""])
            chapter10_yield_prong_components: dict[int, dict[str, ROOT.TH1]] = {}
            chapter10_yield_prong_data: dict[int, ROOT.TH1] = {}
            chapter10_yield_charge_components: dict[str, dict[str, ROOT.TH1]] = {}
            chapter10_yield_charge_data: dict[str, ROOT.TH1] = {}

            inclusive_plot_dir = chapter10_plot_dir / "inclusive"
            stack_selection = f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID"
            for variable in THESIS_DATA_MC_STACK_VARS:
                signal, backgrounds, fakes, data = current_data_mc_fakes_process_components(
                    analysis,
                    stack_selection,
                    variable,
                    tuple(fake_histograms[(variable, prong)] for prong in PRONGS),
                )
                filename = (
                    f"{WP}_{variable}_current_stack_fakes_"
                    f"{chapter10_yscale_tag(variable)}.png"
                )
                plot_chapter10_stack(
                    output_path=inclusive_plot_dir / filename,
                    variable=variable,
                    title="Data 2017 | Medium Tau ID | 44.3 fb$^{-1}$",
                    data=data,
                    signal=signal,
                    backgrounds=backgrounds,
                    fakes=fakes,
                )
                lines.append(f"- `{inclusive_plot_dir / filename}`")

            for prong in PRONGS:
                prong_plot_dir = chapter10_plot_dir / "prong_split" / f"{prong}prong"
                prong_selection = (
                    f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{prong}prong_"
                    "validate_passID"
                )
                for variable in CHAPTER10_PRONG_STACK_VARS:
                    signal, backgrounds, fakes, data = (
                        current_data_mc_fakes_process_components(
                            analysis,
                            prong_selection,
                            variable,
                            (fake_histograms[(variable, prong)],),
                        )
                    )
                    if variable == CHAPTER10_YIELD_TABLE_VARIABLE:
                        chapter10_yield_prong_components[prong] = (
                            chapter10_component_hists(
                                signal=signal,
                                backgrounds=backgrounds,
                                fakes=fakes,
                            )
                        )
                        chapter10_yield_prong_data[prong] = data
                    filename = (
                        f"{WP}_{prong}prong_{variable}_current_stack_fakes_"
                        f"{chapter10_yscale_tag(variable)}.png"
                    )
                    plot_chapter10_stack(
                        output_path=prong_plot_dir / filename,
                        variable=variable,
                        title=(
                            "Data 2017 | Medium Tau ID | "
                            f"{prong}-prong | 44.3 fb$^{{-1}}$"
                        ),
                        data=data,
                        signal=signal,
                        backgrounds=backgrounds,
                        fakes=fakes,
                    )
                    lines.append(f"- `{prong_plot_dir / filename}`")

            for charge_key, charge_label, _charge_cut in CHARGE_SPLITS:
                charge_plot_dir = chapter10_plot_dir / "charge_split" / charge_key
                for variable in CHAPTER10_CHARGE_STACK_VARS:
                    components = [
                        current_data_mc_fakes_process_components(
                            analysis,
                            (
                                f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_"
                                f"{charge_key}_{prong}prong_validate_passID"
                            ),
                            variable,
                            (all_charge_fake_histograms[(charge_key, variable, prong)],),
                        )
                        for prong in PRONGS
                    ]
                    signal, backgrounds, fakes, data = sum_chapter10_process_components(
                        components,
                        name_prefix=(
                            f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_"
                            f"{charge_key}_validate_passID"
                        ),
                        variable=variable,
                    )
                    if variable == CHAPTER10_YIELD_TABLE_VARIABLE:
                        chapter10_yield_charge_components[charge_key] = (
                            chapter10_component_hists(
                                signal=signal,
                                backgrounds=backgrounds,
                                fakes=fakes,
                            )
                        )
                        chapter10_yield_charge_data[charge_key] = data
                    filename = (
                        f"{WP}_{charge_key}_{variable}_current_stack_fakes_"
                        f"{chapter10_yscale_tag(variable)}.png"
                    )
                    plot_chapter10_stack(
                        output_path=charge_plot_dir / filename,
                        variable=variable,
                        title=(
                            "Data 2017 | Medium Tau ID | "
                            f"{charge_label} | 44.3 fb$^{{-1}}$"
                        ),
                        data=data,
                        signal=signal,
                        backgrounds=backgrounds,
                        fakes=fakes,
                    )
                    lines.append(f"- `{charge_plot_dir / filename}`")

            if PLOT_CHAPTER10_CURRENT_TABLES_AND_PIES:
                table_dir = OUTPUT_DIR / "tables" / "chapter10_current"
                pie_dir = OUTPUT_DIR / "plots" / "chapter10_current_pies"
                prong_tex, charge_tex, table_summary = write_chapter10_yield_tables(
                    output_dir=table_dir,
                    prong_components=chapter10_yield_prong_components,
                    prong_data=chapter10_yield_prong_data,
                    charge_components=chapter10_yield_charge_components,
                    charge_data=chapter10_yield_charge_data,
                )
                lines.extend(
                    [
                        "",
                        "## Chapter 10 current yield tables and composition pies",
                        "",
                        f"- `{prong_tex}`",
                        f"- `{charge_tex}`",
                        f"- `{table_summary}`",
                    ]
                )
                pie_specs = [
                    (
                        pie_dir / "prong_split" / "medium_1prong_current_prediction_pie.png",
                        "Medium ID | 1-prong | selected-region prediction",
                        chapter10_yield_prong_components[1],
                    ),
                    (
                        pie_dir / "prong_split" / "medium_3prong_current_prediction_pie.png",
                        "Medium ID | 3-prong | selected-region prediction",
                        chapter10_yield_prong_components[3],
                    ),
                    (
                        pie_dir / "charge_split" / "medium_tauplus_current_prediction_pie.png",
                        r"Medium ID | $\tau^+$ | selected-region prediction",
                        chapter10_yield_charge_components["tauplus"],
                    ),
                    (
                        pie_dir / "charge_split" / "medium_tauminus_current_prediction_pie.png",
                        r"Medium ID | $\tau^-$ | selected-region prediction",
                        chapter10_yield_charge_components["tauminus"],
                    ),
                ]
                for output_path, title, components in pie_specs:
                    plot_chapter10_composition_pie(
                        output_path=output_path,
                        title=title,
                        components=components,
                    )
                    lines.append(f"- `{output_path}`")

    if PLOT_THESIS_LIKE_REGION_STACKS:
        lines.extend(["", "## Thesis-like CR/SR stack plots", ""])
        thesis_plot_dir = OUTPUT_DIR / "plots" / "thesis_like_region_stacks"
        stack_plot_args = {
            "dataset": [analysis.data_sample, *MC_SAMPLES],
            "do_stat": True,
            "do_syst": False,
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
            "label_params": {"llabel": "", "loc": 1},
            "legend_params": {"fontsize": 9, "loc": "upper right"},
        }
        for selection, region_label in (
            (f"{CONFIG_LABEL}_low_met_{WP}_CR_passID", "low-MET control region"),
            (f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID", "signal region"),
        ):
            analysis.paths.plot_dir = thesis_plot_dir / selection
            analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
            for variable in THESIS_LIKE_STACK_VARS:
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = f"{WP}_{variable}_stack_no_fakes_{yscale}.png"
                    analysis.plot(
                        val=variable,
                        selection=selection,
                        xlabel=variable_data[variable]["name"],
                        ylabel="Events",
                        title=f"Medium tau ID | {region_label}",
                        filename=filename,
                        logy=logy,
                        **stack_plot_args,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_THESIS_LIKE_PRONG_STACKS:
        lines.extend(["", "## Thesis-like prong-split signal-region stack plots", ""])
        thesis_prong_plot_dir = OUTPUT_DIR / "plots" / "thesis_like_prong_stacks"
        stack_plot_args = {
            "dataset": [analysis.data_sample, *MC_SAMPLES],
            "do_stat": True,
            "do_syst": False,
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
            "label_params": {"llabel": "", "loc": 1},
            "legend_params": {"fontsize": 9, "loc": "upper right"},
        }
        for prong in PRONGS:
            selection = (
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_passID"
            )
            analysis.paths.plot_dir = (
                thesis_prong_plot_dir
                / f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID_{prong}prong"
            )
            analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
            for variable in ("TauRNNJetScore", "TauBDTEleScore"):
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = f"{WP}_{prong}prong_{variable}_stack_no_fakes_{yscale}.png"
                    analysis.plot(
                        val=variable,
                        selection=selection,
                        xlabel=variable_data[variable]["name"],
                        ylabel="Events",
                        title=f"Medium tau ID | signal region | {prong}-prong",
                        filename=filename,
                        logy=logy,
                        **stack_plot_args,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_FAKE_ENRICHMENT:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "fake_enrichment"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Fake-enrichment plots", ""])
        for (prong, region_key), (
            data_hist,
            nonfake_hist,
            fake_like_hist,
        ) in fake_enrichment_hists.items():
            region_label = (
                "pass-ID numerator"
                if region_key == "derive_passID"
                else "anti-ID denominator"
            )
            filename = (
                f"{CONFIG_LABEL}_low_met_{region_key}_{prong}prong_"
                f"{FAKES_SOURCE}_fake_enrichment.png"
            )
            analysis.plot(
                [fake_like_hist, nonfake_hist],
                label=[
                    "Inferred jet-fake component",
                    "Simulated backgrounds",
                ],
                colour=["tab:orange", "tab:blue"],
                plot_as_data=data_hist,
                data_label="Data",
                xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"Low-MET control region | {region_label} | {prong}-prong",
                kind="stack",
                do_stat=False,
                do_syst=False,
                logx=True,
                logy=True,
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=filename,
                sort=False,
            )
            lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_MEDIUM_FAKE_CONTAMINATION:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "medium_fake_contamination"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Medium fake-contamination plots", ""])
        region_selections = {
            "medium_CR_passID": tuple(
                f"{CONFIG_LABEL}_low_met_fake_enriched_{WP}_{prong}prong_derive_passID"
                for prong in PRONGS
            ),
            "medium_CR_failID": tuple(
                f"{CONFIG_LABEL}_low_met_fake_enriched_{WP}_{prong}prong_derive_failID"
                for prong in PRONGS
            ),
            "medium_SR_passID": tuple(
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_passID"
                for prong in PRONGS
            ),
            "medium_SR_failID": tuple(
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_failID"
                for prong in PRONGS
            ),
        }
        region_titles = {
            "medium_CR_passID": "Low-MET determination region | pass-ID",
            "medium_CR_failID": "Low-MET determination region | anti-ID",
            "medium_SR_passID": "Signal-like application region | pass-ID",
            "medium_SR_failID": "Signal-like application region | anti-ID",
        }
        for region_key, selections in region_selections.items():
            for variable in FAKE_CONTAMINATION_VARS:
                data_hist, nonfake_hist, fake_like_hist = sum_fake_enrichment_components(
                    analysis,
                    selections,
                    variable,
                    region_key,
                )
                filename = f"all_mc_{variable}_{region_key}_fake_fractions.png"
                xlabel = variable_data[variable]["name"]
                if variable in {"MET_met"}:
                    xlabel += " [GeV]"
                analysis.plot(
                    [fake_like_hist, nonfake_hist],
                    label=[
                        "Inferred jet-to-tau component",
                        "Simulated backgrounds",
                    ],
                    colour=["tab:orange", "tab:blue"],
                    plot_as_data=data_hist,
                    data_label="Data",
                    xlabel=xlabel,
                    ylabel="Events",
                    title=region_titles[region_key],
                    kind="stack",
                    do_stat=False,
                    do_syst=False,
                    logy=True,
                    label_params={"llabel": "", "loc": 1},
                    legend_params={"fontsize": 10, "loc": "upper right"},
                    filename=filename,
                    sort=False,
                )
                lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if run_event_loops or FORCE_REBUILD_HISTS or built_current_data_mc_fakes:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
