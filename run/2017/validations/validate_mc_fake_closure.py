from __future__ import annotations

import ROOT
from common import MC_SAMPLES, REPO_ROOT, VALIDATION_OUTPUT, get_root_hist, ratio, write_markdown
from samples import NOMINAL_NAME

from utils.ROOT_utils import sum_th1s

VARIABLE = "TauPt"
CONFIGS = ("no_shadow_bin", "MTW_shadow_bin_200", "MTW_shadow_bin_250", "MTW_shadow_bin_300")
CONTROL_REGION_TAGS = ("lowMET_CR", "CR")
SHADOW_OUTPUT = REPO_ROOT / "outputs" / "analysis_shadow_unfold"
MEASURED_ROOT = SHADOW_OUTPUT / "measured" / "root"
SUMMARY_PATH = VALIDATION_OUTPUT / "mc_fake_closure" / "mc_fake_closure_summary.md"


def dataset_hist(dataset: str, selection: str) -> ROOT.TH1:
    return get_root_hist(MEASURED_ROOT / f"{dataset}.root", f"{NOMINAL_NAME}/{selection}/{VARIABLE}")


def mc_fake(selection: str) -> ROOT.TH1:
    all_mc = sum_th1s(*[dataset_hist(dataset, selection) for dataset in MC_SAMPLES])
    true_mc = sum_th1s(*[dataset_hist(dataset, f"trueTau_{selection}") for dataset in MC_SAMPLES])
    fake = all_mc - true_mc
    fake.SetName(f"{selection}_{VARIABLE}_mc_fake")
    fake.SetDirectory(0)
    return fake


def existing_control_prefix(config: str, prong: str) -> tuple[str, str]:
    prefix = f"{config}_medium" if prong == "inclusive" else f"{config}_medium_{prong}"
    for tag in CONTROL_REGION_TAGS:
        selection = f"{prefix}_{tag}_passID"
        try:
            dataset_hist("data", selection)
            return prefix, tag
        except KeyError:
            continue
    raise KeyError(
        f"Could not find any control-region selections for {config}, {prong}. "
        f"Tried {', '.join(CONTROL_REGION_TAGS)} in {MEASURED_ROOT}."
    )


def predicted_fake(cr_pass: ROOT.TH1, cr_fail: ROOT.TH1, sr_fail: ROOT.TH1) -> ROOT.TH1:
    prediction = sr_fail.Clone(f"{sr_fail.GetName()}_predicted")
    prediction.SetDirectory(0)
    for bin_idx in range(1, prediction.GetNbinsX() + 1):
        den = cr_fail.GetBinContent(bin_idx)
        if den == 0:
            prediction.SetBinContent(bin_idx, 0.0)
            prediction.SetBinError(bin_idx, 0.0)
            continue
        prediction.SetBinContent(bin_idx, sr_fail.GetBinContent(bin_idx) * cr_pass.GetBinContent(bin_idx) / den)
    return prediction


if __name__ == "__main__":
    if not MEASURED_ROOT.is_dir():
        raise FileNotFoundError(
            "Missing analysis_shadow_unfold measured ROOT output. Run "
            "`pixi run python run/2017/analysis_shadow_unfold.py` first."
        )

    lines = [
        "# MC fake-closure validation",
        "",
        "This cache-only validation rebuilds the old MC-only fake closure outside the production "
        "script. It uses the known non-true-tau MC component as both the fake-factor source "
        "and the SR target.",
        "",
        "| Config | Prong | CR pass fake | CR fail fake | SR fail fake | SR pass fake target | "
        "Predicted SR fake | Predicted / target | CR tag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for config in CONFIGS:
        for prong in ("inclusive", "1prong", "3prong"):
            prefix, cr_tag = existing_control_prefix(config, prong)
            cr_pass = mc_fake(f"{prefix}_{cr_tag}_passID")
            cr_fail = mc_fake(f"{prefix}_{cr_tag}_failID")
            sr_fail = mc_fake(f"{prefix}_SR_failID")
            sr_pass = mc_fake(f"{prefix}_SR_passID")
            prediction = predicted_fake(cr_pass, cr_fail, sr_fail)
            lines.append(
                f"| {config} | {prong} | {cr_pass.Integral():.3f} | "
                f"{cr_fail.Integral():.3f} | {sr_fail.Integral():.3f} | "
                f"{sr_pass.Integral():.3f} | {prediction.Integral():.3f} | "
                f"{ratio(prediction.Integral(), sr_pass.Integral()):.3f} | {cr_tag} |"
            )

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
