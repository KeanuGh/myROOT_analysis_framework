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
    VALIDATION_OUTPUT,
    get_root_hist,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402

YEAR = 2017
VARIABLE = "TauPt"
LOAD_SAVED_HISTS = True
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
OUTPUT_DIR = VALIDATION_OUTPUT / "note_like_loose_fake_factor"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_note_like_loose_fake_factor.root"
SUMMARY_PATH = OUTPUT_DIR / "note_like_loose_fake_factor_summary.md"
TABLE14_VALUES = {
    ("loose_proxy", 1, "350-500"): 0.362,
    ("loose_proxy", 1, "500-1000"): 0.336,
    ("loose_proxy", 3, "350-500"): 0.116,
    ("loose_proxy", 3, "500-1000"): 0.083,
}


@dataclass(frozen=True)
class IdModel:
    key: str
    label: str
    pass_cut: Cut
    fail_cut: Cut


PASS_RECO_PRESELECTION = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) "
    r"&& passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + "
    r"MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
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
LOOSE_PROXY = IdModel(
    "loose_proxy",
    "Loose numerator / fail Loose plus RNN > 0.01 proxy denominator",
    Cut("Pass Loose tau ID", "TauLooseWP == 1"),
    Cut(
        "Fail Loose tau ID, pass available anti-ID floor",
        "(TauLooseWP == 0) && (TauRNNJetScore > 0.01)",
    ),
)
MEDIUM_REFERENCE = IdModel(
    "medium_reference",
    "Medium numerator / fail Medium plus RNN > 0.01 denominator",
    Cut(
        "Pass Medium tau ID",
        "(TauBDTEleScore > 0.1) && "
        "((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
        "(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
    ),
    Cut(
        "Fail Medium tau ID, pass available anti-ID floor",
        "(TauBDTEleScore > 0.1) && (TauRNNJetScore > 0.01) && "
        "!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + "
        "(TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
    ),
)
ID_MODELS = (LOOSE_PROXY, MEDIUM_REFERENCE)


def hist_name(model: IdModel, prong: int, suffix: str) -> str:
    return f"note_like_{model.key}_{prong}prong_{suffix}"


def read_fake_factor(model: IdModel, prong: int):
    return get_root_hist(CACHE_FILE, f"note_like_{model.key}_{prong}prong_{FAKES_SOURCE}_FF")


if __name__ == "__main__":
    source_binning = dict(BINNINGS)
    source_binning[FAKES_SOURCE] = np.array([170, 200, 250, 300, 350, 500, 1000])

    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", "TauPt > 170"),
        PASS_ETA,
        Cut(r"Low-$E_T^{\mathrm{miss}}$ fake-factor region", "MET_met < 100"),
    ]
    data_selections: dict[str, list[Cut]] = {}
    mc_selections: dict[str, list[Cut]] = {}
    for model in ID_MODELS:
        for prong in (1, 3):
            prong_cut = Cut(f"{prong}-prong", f"TauNCoreTracks == {prong}")
            pass_selection = hist_name(model, prong, "passID")
            fail_selection = hist_name(model, prong, "failID")
            data_selections[pass_selection] = base_cuts + [model.pass_cut, prong_cut]
            data_selections[fail_selection] = base_cuts + [model.fail_cut, prong_cut]
            mc_selections[pass_selection] = data_selections[pass_selection]
            mc_selections[fail_selection] = data_selections[fail_selection]
            mc_selections[f"trueTau_{pass_selection}"] = (
                data_selections[pass_selection] + [PASS_TRUETAU]
            )
            mc_selections[f"trueTau_{fail_selection}"] = (
                data_selections[fail_selection] + [PASS_TRUETAU]
            )

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = RUN_EVENT_LOOPS_IF_CACHE_MISSING and (
        not LOAD_SAVED_HISTS or not cache_exists
    )

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_note_like_loose_fake_factor",
        output_dir=OUTPUT_DIR,
        log_level=10,
        log_out="both" if run_event_loops else "console",
        extract_vars={
            FAKES_SOURCE,
            "MET_met",
            "TauEta",
            "TauBDTEleScore",
            "TauRNNJetScore",
            "TauNCoreTracks",
            "TauCharge",
            "TauLooseWP",
        },
        import_missing_columns_as_nan=True,
        snapshot=False,
        histogram_vars={FAKES_SOURCE},
        binnings={"": source_binning},
    )

    loaded_hists = LOAD_SAVED_HISTS and analysis.load_hists_if_available(CACHE_FILE)
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing note-like fake-factor cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    if not loaded_hists:
        for model in ID_MODELS:
            for prong in (1, 3):
                pass_selection = hist_name(model, prong, "passID")
                fail_selection = hist_name(model, prong, "failID")
                analysis.do_fakes_estimate(
                    FAKES_SOURCE,
                    (FAKES_SOURCE,),
                    CR_passID_data=pass_selection,
                    CR_failID_data=fail_selection,
                    SR_passID_data=pass_selection,
                    SR_failID_data=fail_selection,
                    CR_passID_mc=f"trueTau_{pass_selection}",
                    CR_failID_mc=f"trueTau_{fail_selection}",
                    SR_failID_mc=f"trueTau_{fail_selection}",
                    name=f"note_like_{model.key}_{prong}prong",
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )
        analysis.save_hists(filename=CACHE_FILE.name)

    lines = [
        "# Note-like Loose fake-factor comparison",
        "",
        "This validation compares this framework's fake factors to Table 14 of the "
        "ATLAS high-mass `tau + MET` technical note. It is validation-only and does "
        "not change the nominal unfolding fake model.",
        "",
        "- note reference: `/home/keanu/Uni_Stuff_Queen_Mary/Reading list/High-mass "
        "resonances to taunu may 21 NOTE.pdf`, Sec. 6.1 and Table 14",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "- anti-ID caveat: no exact VeryLoose branch was found, so this uses "
        "`TauRNNJetScore > 0.01` as the available anti-ID floor proxy.",
        "",
        "## High-pT fake-factor comparison",
        "",
        "| ID model | Prong | TauPt bin [GeV] | This validation FF | ATLAS note FF | "
        "This / ATLAS note |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ID_MODELS:
        for prong in (1, 3):
            ff_hist = read_fake_factor(model, prong)
            for bin_label, bin_idx in (("350-500", 5), ("500-1000", 6)):
                value = ff_hist.GetBinContent(bin_idx)
                atlas_value = TABLE14_VALUES.get((model.key, prong, bin_label), float("nan"))
                lines.append(
                    f"| {model.label} | {prong} | {bin_label} | {value:.5f} | "
                    f"{atlas_value:.3f} | {ratio(value, atlas_value):.3f} |"
                )

    lines.extend(
        [
            "",
            "## Full source-bin fake factors",
            "",
            "| ID model | Prong | TauPt bin [GeV] | Fake factor |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in ID_MODELS:
        for prong in (1, 3):
            ff_hist = read_fake_factor(model, prong)
            for bin_idx in range(1, ff_hist.GetNbinsX() + 1):
                low_edge = ff_hist.GetXaxis().GetBinLowEdge(bin_idx)
                high_edge = ff_hist.GetXaxis().GetBinUpEdge(bin_idx)
                lines.append(
                    f"| {model.label} | {prong} | {low_edge:.0f}-{high_edge:.0f} | "
                    f"{ff_hist.GetBinContent(bin_idx):.5f} |"
                )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "If the Loose-proxy fake factors approach the ATLAS note values while the "
            "medium-reference factors remain much smaller, the discrepancy is mainly a "
            "working-point and anti-ID-definition effect. If both remain far below the "
            "note, the difference is more likely driven by phase space, trigger/sample "
            "composition, or the available VeryLoose proxy.",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
