from __future__ import annotations

import ROOT
from common import REPO_ROOT, VALIDATION_OUTPUT, get_root_hist, write_markdown
from samples import NOMINAL_NAME
from shadow_unfold.models import ResponseComponents

from src.analysis import Analysis
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from src.unfolding import closure_metrics, scale_and_crop_unfolded, unfold_histogram
from utils.variable_names import variable_data

YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
VARIABLE = "MTW"
ITERATIONS = (0, 1, 2)
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_200", "MTW_shadow_bin_250", "MTW_shadow_bin_300")

SHADOW_OUTPUT = REPO_ROOT / "outputs" / "analysis_shadow_unfold"
SPLIT_RESPONSE_ROOT = SHADOW_OUTPUT / "split_response" / "root" / "wtaunu_had.root"
SPLIT_PSEUDO_ROOT = SHADOW_OUTPUT / "split_pseudo_data" / "root" / "wtaunu_had.root"
OUTPUT_DIR = VALIDATION_OUTPUT / "split_sample_unfolding_closure"
SUMMARY_PATH = OUTPUT_DIR / "split_sample_unfolding_closure_summary.md"


def split_response_hist(selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(SPLIT_RESPONSE_ROOT, f"{NOMINAL_NAME}/{selection}/{variable}")


def split_pseudo_hist(selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(SPLIT_PSEUDO_ROOT, f"{NOMINAL_NAME}/{selection}/{variable}")


if __name__ == "__main__":
    if not SPLIT_RESPONSE_ROOT.is_file() or not SPLIT_PSEUDO_ROOT.is_file():
        raise FileNotFoundError(
            "Missing split-sample ROOT output. This validation needs the historical "
            "`outputs/analysis_shadow_unfold/split_*` caches or a dedicated producer rerun."
        )

    plotter = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label="validate_split_sample_unfolding_closure",
        output_dir=OUTPUT_DIR,
        log_level=20,
        log_out="both",
    )
    truth_var = variable_data[VARIABLE]["truth"]
    lines = [
        "# Split-sample unfolding closure validation",
        "",
        "This cache-only validation preserves the old split-sample closure check outside the "
        "production script. The response uses even-numbered events and the pseudo-data/truth "
        "target uses odd-numbered events.",
        "",
        "| Config | Iterations | Mean deviation | Max deviation | Integral ratio |",
        "|---|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        truth_selection = f"{config}_truth_tau"
        reco_selection = f"{config}_medium_reco_tau"
        truth_reco_selection = f"{config}_medium_truth_reco_tau"
        reco = split_response_hist(truth_reco_selection, VARIABLE)
        truth_response = split_response_hist(truth_selection, truth_var)
        matrix = split_response_hist(truth_reco_selection, f"{VARIABLE}_{truth_var}")
        response = ResponseComponents(ROOT.RooUnfoldResponse(reco, truth_response, matrix), reco, truth_response, matrix)

        pseudo_all_reco = split_pseudo_hist(reco_selection, VARIABLE)
        pseudo_fiducial_reco = split_pseudo_hist(truth_reco_selection, VARIABLE)
        pseudo_nonfiducial = pseudo_all_reco - pseudo_fiducial_reco
        pseudo_signal = pseudo_all_reco - pseudo_nonfiducial
        nominal_truth = split_pseudo_hist("no_shadow_bin_truth_tau", truth_var)
        truth = Histogram1D(th1=nominal_truth) / LUMI

        for iteration in ITERATIONS:
            unfolded, _ = unfold_histogram(plotter, pseudo_signal, response, iteration)
            unfolded = scale_and_crop_unfolded(
                unfolded,
                nominal_truth,
                f"{config}_{VARIABLE}_{iteration}iter_split_closure",
                LUMI,
            )
            mean_dev, max_dev, integral_ratio = closure_metrics(unfolded, truth.TH1)
            lines.append(
                f"| {config} | {iteration} | {mean_dev:.3f} | {max_dev:.3f} | "
                f"{integral_ratio:.3f} |"
            )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
