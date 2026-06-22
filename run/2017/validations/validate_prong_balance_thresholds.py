from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
CONFIG_LABEL = "MTW_shadow_bin_300"
MTW_SHADOW_MIN = 300
MTW_NOMINAL_MIN = 350
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
OUTPUT_DIR = VALIDATION_OUTPUT / "prong_balance_thresholds"
SUMMARY_PATH = OUTPUT_DIR / "prong_balance_thresholds_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_prong_balance_thresholds_mtw300.root"

VALIDATION_BINNINGS = dict(BINNINGS)
VALIDATION_BINNINGS["MTW"] = np.insert(BINNINGS["MTW"], 0, MTW_SHADOW_MIN)


@dataclass(frozen=True)
class FakeMethod:
    key: str
    label: str
    cuts: tuple[Cut, ...]


@dataclass(frozen=True)
class TargetRegion:
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
PASS_TRUETAU = Cut(
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true || "
    "MatchedTruthParticle_isMuon == true || "
    "MatchedTruthParticle_isElectron == true",
)


def target_components(analysis: Analysis, selection: str) -> tuple[float, float, float]:
    data = hist_integral(
        analysis.get_hist(
            VARIABLE,
            dataset=analysis.data_sample,
            systematic=NOMINAL_NAME,
            selection=selection,
            allow_generation=True,
        )
    )
    wtaunu_had = hist_integral(
        analysis.get_hist(
            VARIABLE,
            dataset="wtaunu_had",
            systematic=NOMINAL_NAME,
            selection=f"trueTau_{selection}",
            allow_generation=True,
        )
    )
    other_nonfake = hist_integral(
        sum_th1s(
            *[
                analysis.get_hist(
                    VARIABLE,
                    dataset=mc_sample,
                    systematic=NOMINAL_NAME,
                    selection=f"trueTau_{selection}",
                    allow_generation=True,
                )
                for mc_sample in MC_SAMPLES
                if mc_sample != "wtaunu_had"
            ]
        )
    )
    return data, wtaunu_had, other_nonfake


if __name__ == "__main__":
    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    derivation_methods = (
        FakeMethod(
            "current_mtw_shadow_metlt170",
            "current MTW-shadow CR",
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
            "low-MET fake-enriched CR",
            (Cut("MET < 100", "MET_met < 100"),),
        ),
    )
    target_regions = (
        TargetRegion(
            "mtw_shadow_high_met",
            "300 <= MTW < 350, MET >= 170",
            (
                Cut(
                    r"shadow $m_T^W$ sideband",
                    f"(MTW >= {MTW_SHADOW_MIN:g}) && (MTW < {MTW_NOMINAL_MIN:g})",
                ),
                Cut("MET >= 170", "MET_met >= 170"),
            ),
        ),
        TargetRegion(
            "mtw_ge300_high_met",
            "MTW >= 300, MET >= 170",
            (Cut(r"$m_T^W >= 300$", f"MTW >= {MTW_SHADOW_MIN:g}"), Cut("MET >= 170", "MET_met >= 170")),
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

        for target in target_regions:
            prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + list(target.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + list(target.cuts) + [FAIL_MEDIUM, prong_cut]
            )

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (not LOAD_SAVED_HISTS or not cache_exists)

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_prong_balance_thresholds",
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
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={VARIABLE, FAKES_SOURCE},
        binnings={"": VALIDATION_BINNINGS},
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing prong-balance threshold cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for target in target_regions:
        for prong in PRONGS:
            target_prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            data, wtaunu_had, other_nonfake = target_components(analysis, validate_pass)

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
                        f"trueTau_{validate_pass}",
                        f"trueTau_{validate_fail}",
                        name=estimate_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

                fake_prediction = hist_integral(
                    analysis.histograms[f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"]
                )
                implied_wtaunu_had = data - fake_prediction - other_nonfake
                rows[(target.key, method.key, prong)] = {
                    "data": data,
                    "fake_prediction": fake_prediction,
                    "other_nonfake": other_nonfake,
                    "wtaunu_had": wtaunu_had,
                    "implied_wtaunu_had": implied_wtaunu_had,
                    "scale_factor": ratio(implied_wtaunu_had, wtaunu_had),
                }

    lines = [
        "# High-MET threshold prong-balance scale factors",
        "",
        "This validation adds two high-MET regions that were not available in the cache-only "
        "prong-balance scale test. It asks whether the 3-prong imbalance turns on with high "
        "`MET`, the nominal `MTW >= 350` cut, or the exact signal-like corner.",
        "",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        f"- target variable: `{VARIABLE}`",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- validation `MTW` binning starts at `{MTW_SHADOW_MIN}` so the `300-350` shadow interval is retained",
        "",
        "## Implied scale factors",
        "",
        "| Region | Fake model | Prong | Data | Fake prediction | Other nonfake MC | "
        "wtaunu_had MC | Data - fakes - other nonfake | Implied wtaunu_had SF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in target_regions:
        for method in derivation_methods:
            for prong in PRONGS:
                row = rows[(target.key, method.key, prong)]
                lines.append(
                    f"| {target.label} | {method.label} | {prong} | "
                    f"{row['data']:.3f} | {row['fake_prediction']:.3f} | "
                    f"{row['other_nonfake']:.3f} | {row['wtaunu_had']:.3f} | "
                    f"{row['implied_wtaunu_had']:.3f} | {row['scale_factor']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Prong-scale comparison",
            "",
            "| Region | Fake model | SF 1-prong | SF 3-prong | SF 3-prong / SF 1-prong |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for target in target_regions:
        for method in derivation_methods:
            sf_1p = rows[(target.key, method.key, 1)]["scale_factor"]
            sf_3p = rows[(target.key, method.key, 3)]["scale_factor"]
            lines.append(
                f"| {target.label} | {method.label} | {sf_1p:.3f} | {sf_3p:.3f} | "
                f"{ratio(sf_3p, sf_1p):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the 3-prong scale is already low in `300 <= MTW < 350, MET >= 170`, "
            "then the imbalance is primarily high-MET driven. If it appears only after "
            "`MTW >= 350`, then the exact signal-like high-MTW corner is the stress point.",
        ]
    )

    if run_event_loops:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
