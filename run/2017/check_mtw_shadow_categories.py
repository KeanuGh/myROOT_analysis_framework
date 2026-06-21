from pathlib import Path

import ROOT
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, signal_sample

from src.datasetbuilder import DatasetBuilder
from src.dsid_meta import DatasetMetadata
from src.logger import get_logger
from utils import ROOT_utils

YEAR = 2017
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / Path(__file__).stem


ROOT_utils.load_ROOT_settings()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(name=Path(__file__).stem, log_out="console")
logger.info("Building nominal wtaunu_had dataframe for MTW shadow-category diagnostic...")

metadata = DatasetMetadata(logger=logger)
metadata.read_metadata(DSID_METADATA_CACHE)
sumws = [(dsid, meta.sumw) for dsid, meta in metadata]
pmgfs = [
    (dsid, meta.cross_section * meta.kfactor * meta.filter_eff) for dsid, meta in metadata
]
ROOT.gInterpreter.Declare(
    f"""
        std::map<int, float> dsid_sumw{{{','.join(f'{{{dsid}, {sumw}}}' for dsid, sumw in sumws)}}};
        std::map<int, float> dsid_pmgf{{{','.join(f'{{{dsid}, {pmgf}}}' for dsid, pmgf in pmgfs)}}};
    """
)
logger.info("Loaded DSID metadata cache and declared ROOT weight maps.")

sample = signal_sample()
builder = DatasetBuilder(
    name="wtaunu_had",
    ttree=NOMINAL_NAME,
    year=YEAR,
    label=sample["label"],
    hard_cut=sample["hard_cut"],
    is_signal=sample.get("is_signal", False),
    nominal_tree_name=NOMINAL_NAME,
    logger=logger,
)
dataset = builder.build(
    data_path=sample["data_path"],
    selections={},
    extract_vars={
        "passTruth",
        "TruthTau_isHadronic",
        "TruthTau_nChargedTracks",
        "VisTruthTauPt",
        "VisTruthTauEta",
        "TruthMTW",
        "TruthNeutrinoPt",
        "passReco",
        "TauBaselineWP",
        "TauCharge",
        "passMetTrigger",
        "badJet",
        "MatchedTruthParticle_isTau",
        "MatchedTruthParticle_isElectron",
        "MatchedTruthParticle_isMuon",
        "MatchedTruthParticle_isPhoton",
        "TauNCoreTracks",
        "TauBDTEleScore",
        "TauRNNJetScore",
        "TauEta",
        "TauPt",
        "MTW",
        "MET_met",
        "truth_weight",
        "reco_weight",
    },
)
rdf = dataset.rdataframes[NOMINAL_NAME]

truth_fiducial = (
    "(passTruth == 1)"
    " && TruthTau_isHadronic"
    " && ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))"
    " && (VisTruthTauPt > 170)"
    " && (TruthMTW > 350)"
    " && (TruthNeutrinoPt > 170)"
    " && (((abs(VisTruthTauEta) < 1.37) || (1.52 < abs(VisTruthTauEta)))"
    " && (abs(VisTruthTauEta) < 2.47))"
)

reco_prerequisites = (
    "(passReco == 1)"
    " && (TauBaselineWP == 1)"
    " && (abs(TauCharge) == 1)"
    " && passMetTrigger"
    " && (badJet == 0)"
    " && ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron"
    " + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    " && ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))"
    " && (((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))"
    " && (TauBDTEleScore > 0.1)"
    " && ((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1)"
    " + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))"
)

pass_mtw = "MTW > 350"
pass_taupt = "TauPt > 170"
pass_met = "MET_met > 170"

logger.info("Scheduling category counts. ROOT should evaluate these in one event loop...")
truth_rdf = rdf.Filter(truth_fiducial, "truth fiducial")

total_count = truth_rdf.Count()
total_truth_weight = truth_rdf.Sum("truth_weight")
total_reco_weight = truth_rdf.Sum("reco_weight")

category_expressions = {
    "pass_all_nominal_reco_cuts": (
        f"({reco_prerequisites}) && ({pass_mtw}) && ({pass_taupt}) && ({pass_met})"
    ),
    "fail_reco_prerequisites": f"!({reco_prerequisites})",
    "fail_only_mtw": (
        f"({reco_prerequisites}) && !({pass_mtw}) && ({pass_taupt}) && ({pass_met})"
    ),
    "fail_only_taupt": (
        f"({reco_prerequisites}) && ({pass_mtw}) && !({pass_taupt}) && ({pass_met})"
    ),
    "fail_only_met": (
        f"({reco_prerequisites}) && ({pass_mtw}) && ({pass_taupt}) && !({pass_met})"
    ),
    "fail_mtw_and_taupt": (
        f"({reco_prerequisites}) && !({pass_mtw}) && !({pass_taupt}) && ({pass_met})"
    ),
    "fail_mtw_and_met": (
        f"({reco_prerequisites}) && !({pass_mtw}) && ({pass_taupt}) && !({pass_met})"
    ),
    "fail_taupt_and_met": (
        f"({reco_prerequisites}) && ({pass_mtw}) && !({pass_taupt}) && !({pass_met})"
    ),
    "fail_mtw_taupt_and_met": (
        f"({reco_prerequisites}) && !({pass_mtw}) && !({pass_taupt}) && !({pass_met})"
    ),
}

actions = {}
for category, expression in category_expressions.items():
    category_rdf = truth_rdf.Filter(expression, category)
    actions[category] = {
        "raw": category_rdf.Count(),
        "truth_weight": category_rdf.Sum("truth_weight"),
        "reco_weight": category_rdf.Sum("reco_weight"),
    }

total_raw = int(total_count.GetValue())
total_truth = float(total_truth_weight.GetValue())
total_reco = float(total_reco_weight.GetValue())

rows = []
for category, category_actions in actions.items():
    raw = int(category_actions["raw"].GetValue())
    truth_weight = float(category_actions["truth_weight"].GetValue())
    reco_weight = float(category_actions["reco_weight"].GetValue())
    rows.append(
        {
            "category": category,
            "raw": raw,
            "raw_fraction": raw / total_raw if total_raw else 0.0,
            "truth_weight": truth_weight,
            "truth_weight_fraction": truth_weight / total_truth if total_truth else 0.0,
            "reco_weight": reco_weight,
            "reco_weight_fraction": reco_weight / total_reco if total_reco else 0.0,
        }
    )

category_fraction = {row["category"]: row["truth_weight_fraction"] for row in rows}
mtw_only_fraction = category_fraction["fail_only_mtw"]
other_threshold_fraction = (
    category_fraction["fail_only_taupt"]
    + category_fraction["fail_only_met"]
    + category_fraction["fail_taupt_and_met"]
)
mixed_with_mtw_fraction = (
    category_fraction["fail_mtw_and_taupt"]
    + category_fraction["fail_mtw_and_met"]
    + category_fraction["fail_mtw_taupt_and_met"]
)

logger.info("")
logger.info("Truth-fiducial denominator:")
logger.info("  raw events: %.0f", total_raw)
logger.info("  truth_weight sum: %.6g", total_truth)
logger.info("  reco_weight sum: %.6g", total_reco)
logger.info("")
logger.info("Reco-category breakdown, fractions normalised to truth-fiducial denominator:")
for row in rows:
    logger.info(
        "  %-28s raw=%10d  truth_frac=%8.4f  reco_frac=%8.4f",
        row["category"],
        row["raw"],
        row["truth_weight_fraction"],
        row["reco_weight_fraction"],
    )

csv_path = OUTPUT_DIR / "mtw_shadow_category_breakdown.csv"
with csv_path.open("w") as csv_file:
    csv_file.write(
        "category,raw,raw_fraction,truth_weight,truth_weight_fraction,"
        "reco_weight,reco_weight_fraction\n"
    )
    for row in rows:
        csv_file.write(
            f"{row['category']},{row['raw']},{row['raw_fraction']:.10g},"
            f"{row['truth_weight']:.10g},{row['truth_weight_fraction']:.10g},"
            f"{row['reco_weight']:.10g},{row['reco_weight_fraction']:.10g}\n"
        )

markdown_path = OUTPUT_DIR / "mtw_shadow_category_breakdown.md"
with markdown_path.open("w") as md_file:
    md_file.write("# MTW Shadow-Category Diagnostic\n\n")
    md_file.write(
        "This checks whether a 1D `MTW` shadow bin is likely to cover the main "
        "reconstructed-selection losses for truth-fiducial `wtaunu_had` events.\n\n"
    )
    md_file.write("Truth-fiducial denominator:\n\n")
    md_file.write(f"- Raw events: `{total_raw}`\n")
    md_file.write(f"- Truth-weight sum: `{total_truth:.6g}`\n")
    md_file.write(f"- Reco-weight sum: `{total_reco:.6g}`\n\n")
    md_file.write("| Category | Raw events | Truth-weight fraction | Reco-weight fraction |\n")
    md_file.write("|---|---:|---:|---:|\n")
    for row in rows:
        md_file.write(
            f"| `{row['category']}` | {row['raw']} | "
            f"{row['truth_weight_fraction']:.6f} | {row['reco_weight_fraction']:.6f} |\n"
        )
    md_file.write("\n## Reading\n\n")
    md_file.write(
        f"- Fail only `MTW`: `{mtw_only_fraction:.6f}` of the truth-weighted "
        "truth-fiducial denominator.\n"
    )
    md_file.write(
        f"- Fail only `TauPt`, only `MET`, or both without failing `MTW`: "
        f"`{other_threshold_fraction:.6f}`.\n"
    )
    md_file.write(
        f"- Fail `MTW` together with `TauPt` and/or `MET`: "
        f"`{mixed_with_mtw_fraction:.6f}`.\n\n"
    )
    md_file.write(
        "If the non-`MTW` and mixed categories are sizeable, a pure 1D `MTW` "
        "shadow bin is probably incomplete. In that case, the next useful test "
        "would be an `MTW` unfolding with explicit reconstructed pass/fail "
        "categories for the other threshold cuts.\n"
    )

logger.info("")
logger.info("Saved CSV summary to %s", csv_path)
logger.info("Saved markdown summary to %s", markdown_path)
