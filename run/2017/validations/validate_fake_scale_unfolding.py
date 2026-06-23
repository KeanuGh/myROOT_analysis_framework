from __future__ import annotations

import ROOT
from common import MC_SAMPLES, REPO_ROOT, VALIDATION_OUTPUT, get_root_hist, ratio
from samples import NOMINAL_NAME as TREE_NAME
from shadow_unfold.histograms import closure_metrics, scale_and_crop_unfolded, unfold_histogram
from shadow_unfold.models import ResponseComponents

from src.analysis import Analysis
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from utils.ROOT_utils import sum_th1s
from utils.variable_names import variable_data

YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
VARIABLE = "MTW"
FAKES_SOURCE = "TauPt"
ITERATION = 1
FAKE_SCALES = (0.0, 0.5, 1.0)
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_200", "MTW_shadow_bin_250", "MTW_shadow_bin_300")

SHADOW_OUTPUT = REPO_ROOT / "outputs" / "analysis_shadow_unfold"
MEASURED_ROOT = SHADOW_OUTPUT / "measured" / "root"
RESPONSE_ROOT = SHADOW_OUTPUT / "response" / "root" / "wtaunu_had.root"
OUTPUT_DIR = VALIDATION_OUTPUT / "fake_scale_unfolding"
SUMMARY_PATH = OUTPUT_DIR / "fake_scale_unfolding_summary.md"


def dataset_hist(dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(MEASURED_ROOT / f"{dataset}.root", f"{TREE_NAME}/{selection}/{variable}")


def response_hist(selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(RESPONSE_ROOT, f"{TREE_NAME}/{selection}/{variable}")


def saved_fake_hist(config: str, prong: int) -> ROOT.TH1:
    analysis_root = MEASURED_ROOT / "analysis_shadow_unfold_measured.root"
    candidates = (
        f"{config}_medium_{prong}prong_lowMET_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src",
        f"{config}_medium_{prong}prong_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src",
    )
    for hist_name in candidates:
        try:
            return get_root_hist(analysis_root, hist_name)
        except KeyError:
            continue
    raise KeyError(
        f"Missing prong fake histogram for {config}, {prong}-prong in {analysis_root}. "
        "Run analysis_shadow_unfold.py first."
    )


def build_response(config: str) -> tuple[ResponseComponents, ROOT.TH1, ROOT.TH1]:
    truth_var = variable_data[VARIABLE]["truth"]
    reco = response_hist(f"{config}_medium_truth_reco_tau", VARIABLE)
    truth = response_hist(f"{config}_truth_tau", truth_var)
    matrix = response_hist(f"{config}_medium_truth_reco_tau", f"{VARIABLE}_{truth_var}")
    nominal_truth = response_hist("no_shadow_bin_truth_tau", truth_var)
    return ResponseComponents(ROOT.RooUnfoldResponse(reco, truth, matrix), reco, truth, matrix), truth, nominal_truth


if __name__ == "__main__":
    if not MEASURED_ROOT.is_dir() or not RESPONSE_ROOT.is_file():
        raise FileNotFoundError(
            "Missing analysis_shadow_unfold cached ROOT output. Run "
            "`pixi run python run/2017/analysis_shadow_unfold.py` first."
        )

    plotter = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label="validate_fake_scale_unfolding",
        output_dir=OUTPUT_DIR,
        log_level=20,
        log_out="both",
    )
    lines = [
        "# Fake-scale unfolding validation",
        "",
        "This cache-only validation repeats the old fake-scale unfolded-data diagnostic outside "
        "the production shadow-unfolding script. It varies only the fake subtraction and keeps "
        "the prompt backgrounds, nonfiducial correction, and response fixed.",
        "",
        f"- iteration: `{ITERATION}`",
        f"- fake scales: `{', '.join(str(scale) for scale in FAKE_SCALES)}`",
        f"- measured cache: `{MEASURED_ROOT.relative_to(REPO_ROOT)}`",
        f"- response cache: `{RESPONSE_ROOT.relative_to(REPO_ROOT)}`",
        "",
        "| Config | Fake scale | Data sig | Fid reco signal | Fid reco / data sig | "
        "Unfolded integral | Truth integral | Unfolded / truth | Mean shape dev | Max shape dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        sr_pass = f"{config}_medium_SR_passID"
        reco_selection = f"{config}_medium_reco_tau"
        truth_reco_selection = f"{config}_medium_truth_reco_tau"
        data = dataset_hist("data", sr_pass, VARIABLE)
        prompt_background = sum_th1s(
            *[
                dataset_hist(dataset, sr_pass, VARIABLE)
                for dataset in MC_SAMPLES
                if dataset != "wtaunu_had"
            ]
        )
        fakes = sum_th1s(saved_fake_hist(config, 1), saved_fake_hist(config, 3))
        all_reco_signal = response_hist(reco_selection, VARIABLE)
        fiducial_reco_signal = response_hist(truth_reco_selection, VARIABLE)
        nonfiducial_signal = all_reco_signal - fiducial_reco_signal
        response, _truth_response, nominal_truth = build_response(config)
        truth_xsec = Histogram1D(th1=nominal_truth) / LUMI

        for fake_scale in FAKE_SCALES:
            scaled_fakes = fakes.Clone(f"{config}_{VARIABLE}_fake_scale_{fake_scale:g}")
            scaled_fakes.SetDirectory(0)
            scaled_fakes.Scale(fake_scale)
            data_sig = data - prompt_background - scaled_fakes - nonfiducial_signal
            data_sig.SetName(f"{config}_{VARIABLE}_fake_scale_{fake_scale:g}_data_signal")
            data_unfolded, _ = unfold_histogram(plotter, data_sig, response, ITERATION)
            data_unfolded = scale_and_crop_unfolded(
                data_unfolded,
                nominal_truth,
                f"{config}_{VARIABLE}_fake_scale_{fake_scale:g}_data_unfolded",
                LUMI,
            )
            mean_dev, max_dev, unfolded_over_truth = closure_metrics(
                data_unfolded,
                truth_xsec.TH1,
            )
            lines.append(
                f"| {config} | {fake_scale:.2f} | {data_sig.Integral():.3f} | "
                f"{fiducial_reco_signal.Integral():.3f} | "
                f"{ratio(fiducial_reco_signal.Integral(), data_sig.Integral()):.3f} | "
                f"{data_unfolded.Integral():.3f} | {truth_xsec.TH1.Integral():.3f} | "
                f"{unfolded_over_truth:.3f} | {mean_dev:.3f} | {max_dev:.3f} |"
            )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
