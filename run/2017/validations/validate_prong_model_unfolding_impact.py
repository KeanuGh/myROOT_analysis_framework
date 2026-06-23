from __future__ import annotations

import ROOT
from common import MC_SAMPLES, REPO_ROOT, VALIDATION_OUTPUT, get_root_hist, ratio, write_markdown
from samples import NOMINAL_NAME
from shadow_unfold.models import ResponseComponents

from src.analysis import Analysis
from src.datasetbuilder import LUMI_YEAR
from src.histogram import Histogram1D
from src.unfolding import closure_metrics, scale_and_crop_unfolded, unfold_histogram
from utils.ROOT_utils import sum_th1s
from utils.variable_names import variable_data

YEAR = 2017
LUMI = LUMI_YEAR[YEAR]
VARIABLE = "MTW"
FAKES_SOURCE = "TauPt"
ITERATION = 1
PRONG_VARIATIONS = {"reduce_3prong_20pct": {1: 1.0, 3: 0.8}}
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_200", "MTW_shadow_bin_250", "MTW_shadow_bin_300")

SHADOW_OUTPUT = REPO_ROOT / "outputs" / "analysis_shadow_unfold"
MEASURED_ROOT = SHADOW_OUTPUT / "measured" / "root"
RESPONSE_ROOT = SHADOW_OUTPUT / "response" / "root" / "wtaunu_had.root"
OUTPUT_DIR = VALIDATION_OUTPUT / "prong_model_unfolding_impact"
SUMMARY_PATH = OUTPUT_DIR / "prong_model_unfolding_impact_summary.md"


def measured_hist(dataset: str, selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(MEASURED_ROOT / f"{dataset}.root", f"{NOMINAL_NAME}/{selection}/{variable}")


def response_hist(selection: str, variable: str) -> ROOT.TH1:
    return get_root_hist(RESPONSE_ROOT, f"{NOMINAL_NAME}/{selection}/{variable}")


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
    raise KeyError(f"Missing fake histogram for {config}, {prong}-prong in {analysis_root}")


if __name__ == "__main__":
    if not MEASURED_ROOT.is_dir() or not RESPONSE_ROOT.is_file():
        raise FileNotFoundError(
            "Missing analysis_shadow_unfold measured/response ROOT output. Run "
            "`pixi run python run/2017/analysis_shadow_unfold.py` first."
        )

    plotter = Analysis(
        data_dict={},
        year=YEAR,
        analysis_label="validate_prong_model_unfolding_impact",
        output_dir=OUTPUT_DIR,
        log_level=20,
        log_out="both",
    )
    truth_var = variable_data[VARIABLE]["truth"]
    nominal_truth = response_hist("no_shadow_bin_truth_tau", truth_var)
    truth = Histogram1D(th1=nominal_truth) / LUMI

    lines = [
        "# Prong-model unfolding-impact validation",
        "",
        "This cache-only validation preserves the old `wtaunu_had` prong-composition propagated "
        "unfolding test outside the production script. It rescales the prong-split signal "
        "response pieces and nonfiducial correction, then unfolds the same data input.",
        "",
        f"- iteration: `{ITERATION}`",
        "",
        "| Config | Variation | 1-prong scale | 3-prong scale | Nominal data sig | "
        "Varied data sig | Nominal fid reco / data sig | Varied fid reco / data sig | "
        "Unfolded / truth | Mean shape dev | Max shape dev |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        sr_pass = f"{config}_medium_SR_passID"
        reco_selection = f"{config}_medium_reco_tau"
        truth_reco_selection = f"{config}_medium_truth_reco_tau"
        data = measured_hist("data", sr_pass, VARIABLE)
        prompt_background = sum_th1s(
            *[
                measured_hist(dataset, sr_pass, VARIABLE)
                for dataset in MC_SAMPLES
                if dataset != "wtaunu_had"
            ]
        )
        fakes = sum_th1s(saved_fake_hist(config, 1), saved_fake_hist(config, 3))
        nominal_all_reco = response_hist(reco_selection, VARIABLE)
        nominal_fiducial_reco = response_hist(truth_reco_selection, VARIABLE)
        nominal_nonfiducial = nominal_all_reco - nominal_fiducial_reco
        nominal_data_sig = data - prompt_background - fakes - nominal_nonfiducial
        truth_response = response_hist(f"{config}_truth_tau", truth_var)

        prong_all_reco = {
            prong: response_hist(f"{config}_medium_{prong}prong_reco_tau", VARIABLE)
            for prong in (1, 3)
        }
        prong_fiducial_reco = {
            prong: response_hist(f"{config}_medium_{prong}prong_truth_reco_tau", VARIABLE)
            for prong in (1, 3)
        }
        prong_matrix = {
            prong: response_hist(
                f"{config}_medium_{prong}prong_truth_reco_tau",
                f"{VARIABLE}_{truth_var}",
            )
            for prong in (1, 3)
        }

        for variation_name, prong_scales in PRONG_VARIATIONS.items():
            varied_all_reco = None
            varied_fiducial_reco = None
            varied_matrix = None
            for prong, scale in prong_scales.items():
                scaled_all = prong_all_reco[prong].Clone(
                    f"{config}_{variation_name}_{prong}p_all"
                )
                scaled_all.SetDirectory(0)
                scaled_all.Scale(scale)
                scaled_fid = prong_fiducial_reco[prong].Clone(
                    f"{config}_{variation_name}_{prong}p_fid"
                )
                scaled_fid.SetDirectory(0)
                scaled_fid.Scale(scale)
                scaled_matrix = prong_matrix[prong].Clone(
                    f"{config}_{variation_name}_{prong}p_matrix"
                )
                scaled_matrix.SetDirectory(0)
                scaled_matrix.Scale(scale)

                if varied_all_reco is None:
                    varied_all_reco = scaled_all
                else:
                    varied_all_reco.Add(scaled_all)

                if varied_fiducial_reco is None:
                    varied_fiducial_reco = scaled_fid
                else:
                    varied_fiducial_reco.Add(scaled_fid)

                if varied_matrix is None:
                    varied_matrix = scaled_matrix
                else:
                    varied_matrix.Add(scaled_matrix)

            varied_nonfiducial = varied_all_reco - varied_fiducial_reco
            varied_data_sig = data - prompt_background - fakes - varied_nonfiducial
            response = ResponseComponents(
                ROOT.RooUnfoldResponse(varied_fiducial_reco, truth_response, varied_matrix),
                varied_fiducial_reco,
                truth_response,
                varied_matrix,
            )
            varied_unfolded, _ = unfold_histogram(plotter, varied_data_sig, response, ITERATION)
            varied_unfolded = scale_and_crop_unfolded(
                varied_unfolded,
                nominal_truth,
                f"{config}_{variation_name}_{VARIABLE}_unfolded",
                LUMI,
            )
            mean_dev, max_dev, unfolded_over_truth = closure_metrics(varied_unfolded, truth.TH1)
            lines.append(
                f"| {config} | {variation_name} | {prong_scales[1]:.3f} | "
                f"{prong_scales[3]:.3f} | {nominal_data_sig.Integral():.3f} | "
                f"{varied_data_sig.Integral():.3f} | "
                f"{ratio(nominal_fiducial_reco.Integral(), nominal_data_sig.Integral()):.3f} | "
                f"{ratio(varied_fiducial_reco.Integral(), varied_data_sig.Integral()):.3f} | "
                f"{unfolded_over_truth:.3f} | {mean_dev:.3f} | {max_dev:.3f} |"
            )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
