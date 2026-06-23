from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from common import (
    MC_SAMPLES,
    MEASURED_HIST_CACHE,
    VALIDATION_OUTPUT,
    dataset_hist_path,
    get_root_hist,
    hist_integral,
    ratio,
    sum_hists,
    write_markdown,
)

VARIABLE = "MTW"
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_250")
SOURCE_ROOT_DIR = (
    VALIDATION_OUTPUT.parents[1] / "outputs" / "analysis_shadow_unfold" / "measured" / "root"
)
OUTPUT_DIR = VALIDATION_OUTPUT / "preunfolding_stack_composition"
PLOT_DIR = OUTPUT_DIR / "plots"
SUMMARY_PATH = OUTPUT_DIR / "preunfolding_stack_composition_summary.md"


def dataset_file(dataset: str) -> Path:
    return SOURCE_ROOT_DIR / f"{dataset}.root"


def dataset_hist(dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(dataset_file(dataset), dataset_hist_path(selection, variable))


def analysis_hist(hist_name: str) -> ROOT.TH1:
    return get_root_hist(MEASURED_HIST_CACHE, hist_name)


def empty_like(hist: ROOT.TH1, name: str) -> ROOT.TH1:
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    clone.Reset("ICES")
    return clone


def edges_and_values(hist: ROOT.TH1) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array(
        [hist.GetXaxis().GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 1)]
        + [hist.GetXaxis().GetBinUpEdge(hist.GetNbinsX())],
        dtype=float,
    )
    values = np.array(
        [hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)],
        dtype=float,
    )
    return edges, values


def make_plot(config: str, data: ROOT.TH1, variants: dict[str, ROOT.TH1]) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )
    data_edges, data_values = edges_and_values(data)
    centres = 0.5 * (data_edges[:-1] + data_edges[1:])
    ax.errorbar(
        centres,
        data_values,
        yerr=np.sqrt(np.clip(data_values, 0.0, None)),
        fmt="ko",
        label="data",
        markersize=4,
    )

    colours = {
        "all MC, no fakes": "tab:blue",
        "all MC + data-driven fakes": "tab:red",
        "nonfake MC + data-driven fakes": "tab:green",
        "fake-like MC only": "tab:purple",
    }
    for label, hist in variants.items():
        edges, values = edges_and_values(hist)
        ax.stairs(values, edges, label=label, color=colours.get(label), linewidth=1.6)
        ratio_values = np.divide(
            values,
            data_values,
            out=np.full_like(values, np.nan),
            where=data_values != 0,
        )
        ratio_ax.stairs(ratio_values, edges, color=colours.get(label), linewidth=1.3)

    ax.set_ylabel("Events")
    ax.set_title(f"{config} pre-unfolding stack composition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ratio_ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ratio_ax.set_ylabel("Pred. / data")
    ratio_ax.set_xlabel(r"$m_T^W$ [GeV]")
    ratio_ax.set_ylim(0.0, 2.0)
    ratio_ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = PLOT_DIR / f"{config}_preunfolding_stack_composition.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


if __name__ == "__main__":
    if not SOURCE_ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Missing measured ROOT directory: {SOURCE_ROOT_DIR}")
    if not MEASURED_HIST_CACHE.is_file():
        raise FileNotFoundError(f"Missing analysis histogram cache: {MEASURED_HIST_CACHE}")

    lines = [
        "# Pre-unfolding stack-composition validation",
        "",
        "This cache-only validation asks whether adding the data-driven fake estimate "
        "double-counts fake-like MC already present in the reconstructed MC stack.",
        "",
        f"- source ROOT directory: `{SOURCE_ROOT_DIR.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- analysis fake cache: `{MEASURED_HIST_CACHE.relative_to(VALIDATION_OUTPUT.parents[1])}`",
        f"- target variable: `{VARIABLE}`",
        "",
        "## Integral summary",
        "",
        "| Configuration | Data | All MC, no fakes | All MC + fakes | "
        "Nonfake MC + fakes | Fake-like MC only | Fakes | All MC/data | "
        "(All MC + fakes)/data | (Nonfake + fakes)/data |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    plot_paths: list[Path] = []

    for config in CONFIGS:
        selection = f"{config}_medium_SR_passID"
        data = dataset_hist("data", selection, VARIABLE)
        all_mc = sum_hists(
            [dataset_hist(sample, selection, VARIABLE) for sample in MC_SAMPLES]
        )
        nonfake_mc = sum_hists(
            [dataset_hist(sample, f"trueTau_{selection}", VARIABLE) for sample in MC_SAMPLES]
        )
        fake_like_mc = all_mc - nonfake_mc
        fake_like_mc.SetName(f"{config}_{VARIABLE}_fake_like_mc")
        fake_like_mc.SetDirectory(0)

        fakes = empty_like(data, f"{config}_{VARIABLE}_prong_split_fakes")
        for prong in (1, 3):
            fakes.Add(
                analysis_hist(
                    f"{config}_medium_{prong}prong_lowMET_{VARIABLE}_fakes_bkg_TauPt_src"
                )
            )

        all_mc_plus_fakes = all_mc.Clone(f"{config}_{VARIABLE}_all_mc_plus_fakes")
        all_mc_plus_fakes.SetDirectory(0)
        all_mc_plus_fakes.Add(fakes)
        nonfake_plus_fakes = nonfake_mc.Clone(f"{config}_{VARIABLE}_nonfake_plus_fakes")
        nonfake_plus_fakes.SetDirectory(0)
        nonfake_plus_fakes.Add(fakes)

        data_yield = hist_integral(data)
        lines.append(
            f"| {config} | {data_yield:.3f} | {hist_integral(all_mc):.3f} | "
            f"{hist_integral(all_mc_plus_fakes):.3f} | "
            f"{hist_integral(nonfake_plus_fakes):.3f} | "
            f"{hist_integral(fake_like_mc):.3f} | {hist_integral(fakes):.3f} | "
            f"{ratio(hist_integral(all_mc), data_yield):.3f} | "
            f"{ratio(hist_integral(all_mc_plus_fakes), data_yield):.3f} | "
            f"{ratio(hist_integral(nonfake_plus_fakes), data_yield):.3f} |"
        )

        plot_paths.append(
            make_plot(
                config,
                data,
                {
                    "all MC, no fakes": all_mc,
                    "all MC + data-driven fakes": all_mc_plus_fakes,
                    "nonfake MC + data-driven fakes": nonfake_plus_fakes,
                    "fake-like MC only": fake_like_mc,
                },
            )
        )

    lines.extend(
        [
            "",
            "## Representative plots",
            "",
            *[f"- `{path}`" for path in plot_paths],
            "",
            "## Interpretation guide",
            "",
            "If `all MC + data-driven fakes` overshoots data while `all MC, no fakes` "
            "is already close to data, the fake estimate may be overlapping with a "
            "fake-like component still present in the MC stack. If `nonfake MC + "
            "data-driven fakes` is closer to data, the data-driven fake estimate should "
            "be treated as a replacement for fake-like MC rather than an additive term.",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
