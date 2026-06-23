from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

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
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import sum_th1s  # noqa: E402

YEAR = 2017
VARIABLE = "MTW"
CONFIGS = {
    "no_shadow_bin": 350,
    "MTW_shadow_bin_250": 250,
}
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
OUTPUT_DIR = VALIDATION_OUTPUT / "photon_nonfake_subtraction"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_photon_nonfake_subtraction.root"
SUMMARY_PATH = OUTPUT_DIR / "photon_nonfake_subtraction_summary.md"
CSV_PATH = OUTPUT_DIR / "photon_nonfake_subtraction_summary.csv"


@dataclass(frozen=True)
class NonfakeModel:
    key: str
    label: str
    cut: Cut


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

CURRENT_NONFAKE = NonfakeModel(
    "current",
    "Current nonfake subtraction: hadronic tau, electron, muon",
    Cut(
        "current nonfake",
        "MatchedTruthParticle_isHadronicTau == true || "
        "MatchedTruthParticle_isMuon == true || "
        "MatchedTruthParticle_isElectron == true",
    ),
)
PHOTON_NONFAKE = NonfakeModel(
    "with_photon",
    "Expanded nonfake subtraction: current + photon",
    Cut(
        "current nonfake plus photon",
        "MatchedTruthParticle_isHadronicTau == true || "
        "MatchedTruthParticle_isMuon == true || "
        "MatchedTruthParticle_isElectron == true || "
        "MatchedTruthParticle_isPhoton == true",
    ),
)
PHOTON_LEPTAU_NONFAKE = NonfakeModel(
    "with_photon_leptonic_tau",
    "Expanded nonfake subtraction: current + photon + leptonic tau",
    Cut(
        "current nonfake plus photon plus leptonic tau",
        "MatchedTruthParticle_isHadronicTau == true || "
        "MatchedTruthParticle_isMuon == true || "
        "MatchedTruthParticle_isElectron == true || "
        "MatchedTruthParticle_isPhoton == true || "
        "MatchedTruthParticle_isLeptonicTau == true",
    ),
)
NONFAKE_MODELS = (CURRENT_NONFAKE, PHOTON_NONFAKE, PHOTON_LEPTAU_NONFAKE)


def selection_name(config: str, prong: int, region: str) -> str:
    return f"{config}_{WP}_{prong}prong_{region}"


def nonfake_selection(model: NonfakeModel, selection: str) -> str:
    return f"{model.key}_nonfake_{selection}"


def fake_prefix(config: str, prong: int, model: NonfakeModel) -> str:
    return f"{config}_{WP}_{prong}prong_lowMET_{model.key}"


def hist_integral_from_analysis(
    analysis: Analysis,
    *,
    dataset: str,
    selection: str,
    variable: str = VARIABLE,
) -> float:
    return hist_integral(
        analysis.get_hist(
            variable,
            dataset=dataset,
            systematic=NOMINAL_NAME,
            selection=selection,
            allow_generation=True,
        )
    )


def summed_mc_integral(analysis: Analysis, selection: str, variable: str = VARIABLE) -> float:
    return hist_integral(
        sum_th1s(
            *[
                analysis.get_hist(
                    variable,
                    dataset=sample,
                    systematic=NOMINAL_NAME,
                    selection=selection,
                    allow_generation=True,
                )
                for sample in MC_SAMPLES
            ]
        )
    )


def analysis_hist_integral(analysis: Analysis, hist_name: str) -> float:
    return hist_integral(analysis.histograms[hist_name])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_exists
    )
    if not cache_exists and not run_event_loops:
        raise FileNotFoundError(
            "Missing photon-nonfake subtraction cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    base_reco_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
    ]
    fake_control_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        Cut(r"Low-$E_T^{\mathrm{miss}}$ fake-enriched region", "MET_met < 100"),
    ]

    for config, mtw_min in CONFIGS.items():
        sr_cuts = [
            *base_reco_cuts,
            Cut(r"$m_T^W$ threshold", f"MTW > {mtw_min:g}"),
            Cut(r"$E_T^{\mathrm{miss}} > 170$", "MET_met > 170"),
        ]
        for prong in PRONGS:
            prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
            selections = {
                "SR_passID": [*sr_cuts, PASS_MEDIUM, prong_cut],
                "SR_failID": [*sr_cuts, FAIL_MEDIUM, prong_cut],
                "lowMET_CR_passID": [*fake_control_cuts, PASS_MEDIUM, prong_cut],
                "lowMET_CR_failID": [*fake_control_cuts, FAIL_MEDIUM, prong_cut],
            }
            for region, cuts in selections.items():
                selection = selection_name(config, prong, region)
                data_selections[selection] = cuts
                mc_selections[selection] = cuts
                for model in NONFAKE_MODELS:
                    mc_selections[nonfake_selection(model, selection)] = [*cuts, model.cut]

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_photon_nonfake_subtraction",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            VARIABLE,
            FAKES_SOURCE,
            "MET_met",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
            "MatchedTruthParticle_isTau",
            "MatchedTruthParticle_isHadronicTau",
            "MatchedTruthParticle_isLeptonicTau",
            "MatchedTruthParticle_isElectron",
            "MatchedTruthParticle_isMuon",
            "MatchedTruthParticle_isPhoton",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={VARIABLE, FAKES_SOURCE},
        binnings={"": BINNINGS},
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists:
        for config in CONFIGS:
            for prong in PRONGS:
                cr_pass = selection_name(config, prong, "lowMET_CR_passID")
                cr_fail = selection_name(config, prong, "lowMET_CR_failID")
                sr_pass = selection_name(config, prong, "SR_passID")
                sr_fail = selection_name(config, prong, "SR_failID")
                for model in NONFAKE_MODELS:
                    analysis.do_fakes_estimate(
                        FAKES_SOURCE,
                        (VARIABLE,),
                        CR_passID_data=cr_pass,
                        CR_failID_data=cr_fail,
                        SR_passID_data=sr_pass,
                        SR_failID_data=sr_fail,
                        CR_passID_mc=nonfake_selection(model, cr_pass),
                        CR_failID_mc=nonfake_selection(model, cr_fail),
                        SR_passID_mc=nonfake_selection(model, sr_pass),
                        SR_failID_mc=nonfake_selection(model, sr_fail),
                        name=fake_prefix(config, prong, model),
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )
        analysis.save_hists(filename=CACHE_FILE.name)

    rows: list[dict[str, object]] = []
    lines = [
        "# Photon nonfake-subtraction validation",
        "",
        "This validation tests whether photon-matched tau candidates should be part of "
        "the MC nonfake subtraction in the fake-factor estimate. It compares the "
        "current subtraction to variants that also subtract photon-matched candidates "
        "and photon plus leptonic-tau candidates.",
        "",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        f"- output CSV: `{CSV_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Fake-yield and stack summary",
        "",
        "| Configuration | Model | Data pass-ID | All MC pass-ID | Total fakes | "
        "All MC + fakes / data | Nonfake MC + fakes / data |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for config in CONFIGS:
        data_pass = sum(
            hist_integral_from_analysis(
                analysis,
                dataset=analysis.data_sample,
                selection=selection_name(config, prong, "SR_passID"),
            )
            for prong in PRONGS
        )
        all_mc_pass = sum(
            summed_mc_integral(analysis, selection_name(config, prong, "SR_passID"))
            for prong in PRONGS
        )
        for model in NONFAKE_MODELS:
            total_fakes = 0.0
            nonfake_mc_pass = 0.0
            for prong in PRONGS:
                prefix = fake_prefix(config, prong, model)
                total_fakes += analysis_hist_integral(
                    analysis,
                    f"{prefix}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src",
                )
                nonfake_mc_pass += summed_mc_integral(
                    analysis,
                    nonfake_selection(model, selection_name(config, prong, "SR_passID")),
                )
            all_mc_plus_fakes = all_mc_pass + total_fakes
            nonfake_plus_fakes = nonfake_mc_pass + total_fakes
            rows.append(
                {
                    "configuration": config,
                    "model": model.key,
                    "data_pass": f"{data_pass:.6g}",
                    "all_mc_pass": f"{all_mc_pass:.6g}",
                    "nonfake_mc_pass": f"{nonfake_mc_pass:.6g}",
                    "total_fakes": f"{total_fakes:.6g}",
                    "all_mc_plus_fakes_over_data": f"{ratio(all_mc_plus_fakes, data_pass):.6g}",
                    "nonfake_plus_fakes_over_data": f"{ratio(nonfake_plus_fakes, data_pass):.6g}",
                }
            )
            lines.append(
                f"| {config} | {model.key} | {data_pass:.3f} | {all_mc_pass:.3f} | "
                f"{total_fakes:.3f} | {ratio(all_mc_plus_fakes, data_pass):.3f} | "
                f"{ratio(nonfake_plus_fakes, data_pass):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Per-prong fake yields",
            "",
            "| Configuration | Prong | Model | Fake yield |",
            "|---|---:|---|---:|",
        ]
    )
    for config in CONFIGS:
        for prong in PRONGS:
            for model in NONFAKE_MODELS:
                prefix = fake_prefix(config, prong, model)
                fake_yield = analysis_hist_integral(
                    analysis,
                    f"{prefix}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src",
                )
                lines.append(f"| {config} | {prong} | {model.key} | {fake_yield:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "If the photon-expanded subtraction substantially reduces the fake estimate "
            "and improves the pre-unfolding stack ratio, photon-matched candidates are "
            "a plausible missing nonfake component. If it has only a small effect, the "
            "main problem is the data fail-ID application population rather than the "
            "photon treatment alone.",
        ]
    )
    write_csv(CSV_PATH, rows)
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
