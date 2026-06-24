from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import ROOT

RUN_2017_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RUN_2017_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_2017_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import VALIDATION_OUTPUT, write_markdown  # noqa: E402
from samples import DTA_PATH, NOMINAL_NAME  # noqa: E402

OUTPUT_DIR = VALIDATION_OUTPUT / "tes_response_truth_axis"
SUMMARY_PATH = OUTPUT_DIR / "tes_response_truth_axis_summary.md"
CSV_PATH = OUTPUT_DIR / "tes_response_truth_axis_tree_checks.csv"
RESPONSE_CACHE = (
    REPO_ROOT / "outputs" / "analysis_shadow_unfold" / "response" / "root" / "wtaunu_had.root"
)

MAX_FILES = 1
TREE_PREFIX = "T_s1thv_"
TES_TREE_SUFFIXES = (
    "TAUS_TRUEHADTAU_SME_TES_DETECTOR_Barrel_HighPt__1up",
    "TAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE__1up",
)
REQUIRED_TRUTH_INPUTS = (
    "VisTruthTauPt",
    "VisTruthTauPhi",
    "TruthNeutrinoPt",
    "TruthNeutrinoPhi",
    "VisTruthTauEta",
    "TruthTau_nChargedTracks",
    "TruthTau_isHadronic",
    "passTruth",
)
REQUIRED_RECO_INPUTS = (
    "TauPt",
    "TauPhi",
    "MET_met",
    "MET_phi",
    "TauEta",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TauNCoreTracks",
    "passReco",
)
EVENT_KEY_INPUTS = ("mcChannel", "eventNumber")


def event_key(tree: ROOT.TTree) -> tuple[int, int]:
    return int(tree.mcChannel), int(tree.eventNumber)


def branch_names(tree: ROOT.TTree) -> set[str]:
    return {branch.GetName() for branch in tree.GetListOfBranches()}


def sample_files() -> list[Path]:
    patterns = [
        DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
        DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(path) for path in glob.glob(str(pattern)))
    return files[:MAX_FILES]


def response_cache_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not RESPONSE_CACHE.is_file():
        return rows
    with ROOT.TFile(str(RESPONSE_CACHE), "READ") as file:
        systematics = [
            key.GetName()
            for key in file.GetListOfKeys()
            if key.GetClassName() == "TDirectoryFile"
            and key.GetName().startswith("TAUS_TRUEHADTAU_SME_TES_")
        ]
        for systematic in sorted(systematics):
            for selection in (
                "no_shadow_bin_medium_truth_reco_tau",
                "MTW_shadow_bin_250_medium_truth_reco_tau",
            ):
                reco = file.Get(f"{systematic}/{selection}/MTW")
                matrix = file.Get(f"{systematic}/{selection}/MTW_TruthMTW")
                rows.append(
                    {
                        "systematic": systematic,
                        "selection": selection,
                        "reco_integral": float(reco.Integral()) if reco else float("nan"),
                        "matrix_integral": float(matrix.Integral()) if matrix else float("nan"),
                    }
                )
    return rows


def inspect_trees() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in sample_files():
        root_file = ROOT.TFile.Open(str(file_path), "READ")
        if not root_file:
            continue
        nominal_tree = root_file.Get(NOMINAL_NAME)
        if not nominal_tree:
            continue
        nominal_branches = branch_names(nominal_tree)
        nominal_missing_truth = [
            branch for branch in REQUIRED_TRUTH_INPUTS if branch not in nominal_branches
        ]
        nominal_tree.SetBranchStatus("*", 0)
        for branch in EVENT_KEY_INPUTS:
            nominal_tree.SetBranchStatus(branch, 1)
        nominal_keys: set[tuple[int, int]] = set()
        for entry_idx in range(nominal_tree.GetEntries()):
            nominal_tree.GetEntry(entry_idx)
            nominal_keys.add(event_key(nominal_tree))

        for suffix in TES_TREE_SUFFIXES:
            tree_name = f"{TREE_PREFIX}{suffix}"
            tes_tree = root_file.Get(tree_name)
            if not tes_tree:
                rows.append(
                    {
                        "file": str(file_path),
                        "tree": tree_name,
                        "exists": False,
                    }
                )
                continue
            tes_branches = branch_names(tes_tree)
            missing_truth = [
                branch for branch in REQUIRED_TRUTH_INPUTS if branch not in tes_branches
            ]
            missing_reco = [
                branch for branch in REQUIRED_RECO_INPUTS if branch not in tes_branches
            ]
            missing_key = [
                branch for branch in EVENT_KEY_INPUTS if branch not in tes_branches
            ]
            entries = int(tes_tree.GetEntries())
            matched = 0
            if not missing_key:
                tes_tree.SetBranchStatus("*", 0)
                for branch in EVENT_KEY_INPUTS:
                    tes_tree.SetBranchStatus(branch, 1)
                for entry_idx in range(entries):
                    tes_tree.GetEntry(entry_idx)
                    if event_key(tes_tree) in nominal_keys:
                        matched += 1
            rows.append(
                {
                    "file": str(file_path),
                    "tree": tree_name,
                    "exists": True,
                    "entries": entries,
                    "nominal_missing_truth_inputs": ",".join(nominal_missing_truth),
                    "tes_missing_truth_inputs": ",".join(missing_truth),
                    "tes_missing_reco_inputs": ",".join(missing_reco),
                    "tes_missing_key_inputs": ",".join(missing_key),
                    "tes_entries_matched_to_nominal": matched,
                    "nominal_truth_inputs_available": not nominal_missing_truth,
                    "matched_fraction": matched / entries if entries else float("nan"),
                }
            )
        root_file.Close()
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tree_rows = inspect_trees()
    cache_rows = response_cache_rows()
    write_csv(CSV_PATH, tree_rows)

    representative = tree_rows[0] if tree_rows else {}
    matched_fractions = [
        float(row["matched_fraction"])
        for row in tree_rows
        if row.get("exists") and row.get("matched_fraction") == row.get("matched_fraction")
    ]
    cache_matrix_nonzero = sum(
        1 for row in cache_rows if abs(float(row["matrix_integral"])) > 0.0
    )
    cache_reco_nonzero = sum(1 for row in cache_rows if abs(float(row["reco_integral"])) > 0.0)

    lines = [
        "# TES response truth-axis diagnostic",
        "",
        "Question: why do TES shifted response matrices have zero integral?",
        "",
        "Implementation:",
        f"- inspected up to `{MAX_FILES}` Wtaunu input files in read-only mode",
        f"- tree-level CSV: `{CSV_PATH}`",
        f"- response cache inspected: `{RESPONSE_CACHE}`",
        "",
        "Result:",
        "",
        "| Check | Value | Interpretation |",
        "|---|---:|---|",
        f"| Representative TES missing truth inputs | `{representative.get('tes_missing_truth_inputs', '-')}` | shifted TES trees do not carry the truth-axis inputs needed for `TruthMTW` |",
        f"| Representative TES missing reco inputs | `{representative.get('tes_missing_reco_inputs', '-')}` | shifted TES trees do carry the reconstructed inputs needed for `MTW` if this is empty |",
        f"| TES entries matched to nominal event keys | `{min(matched_fractions) if matched_fractions else float('nan'):.3f}-{max(matched_fractions) if matched_fractions else float('nan'):.3f}` | nominal truth lookup by `(mcChannel,eventNumber)` is feasible if close to 1 |",
        f"| TES response reco histograms with non-zero integral | `{cache_reco_nonzero}` | shifted reco projections exist in the current cache |",
        f"| TES response matrices with non-zero integral | `{cache_matrix_nonzero}` | currently zero means full shifted migration matrices are unavailable |",
        "",
        "Interpretation:",
        "The TES shifted trees contain shifted reconstructed variables but not the "
        "truth variables needed to calculate the truth-axis value. The existing "
        "helper can select a nominal fiducial event set using event masks, but it "
        "does not attach nominal `TruthMTW` values to the shifted events. The "
        "result is a finite shifted reco projection and an empty `MTW_TruthMTW` "
        "matrix.",
        "",
        "Recommended fix:",
        "Build a nominal truth lookup keyed by `(mcChannel,eventNumber)` for the "
        "fiducial event set, attach `TruthMTW` to the TES shifted RDataFrame, and "
        "then fill the shifted `MTW_TruthMTW` matrix. This is a response-builder "
        "extension, not a fake-estimate or weighting correction.",
    ]
    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
