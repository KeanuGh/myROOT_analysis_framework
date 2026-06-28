from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import ROOT

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
    fake_factor_bin_health,
    hist_integral,
    ratio,
    write_markdown,
)
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, analysis_samples  # noqa: E402

from src.analysis import Analysis  # noqa: E402
from src.cutting import Cut  # noqa: E402
from utils.ROOT_utils import get_th1_bin_errors, sum_th1s  # noqa: E402
from utils.variable_names import variable_data  # noqa: E402

YEAR = 2017
VARIABLE = "MTW"
CONFIG_LABEL = "MTW_shadow_bin_300"
MTW_SHADOW_MIN = 300
MTW_NOMINAL_MIN = 350
TAUPT_MIN = 170
PRONGS = (1, 3)
LOAD_SAVED_HISTS = True
FORCE_REBUILD_HISTS = os.environ.get("VALIDATE_LOW_MET_FORCE_REBUILD") == "1"
RUN_EVENT_LOOPS_IF_CACHE_MISSING = True
PLOT_TRANSFER_COMPARISON = True
PLOT_FAKE_ENRICHMENT = True
PLOT_MEDIUM_FAKE_CONTAMINATION = True
PLOT_THESIS_LIKE_REGION_STACKS = True
PLOT_THESIS_LIKE_PRONG_STACKS = True
PLOT_NOMINAL_LOW_MET_FAKE_FACTORS = True
PLOT_CURRENT_DATA_MC_FAKES_STACKS = True
THESIS_LIKE_STACK_VARS = ("TauRNNJetScore", "TauBDTEleScore", "TauNCoreTracks")
THESIS_DATA_MC_STACK_VARS = (
    "MTW",
    "MET_met",
    "TauPt",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TauNCoreTracks",
)
FAKE_CONTAMINATION_VARS = ("MET_met", "TauRNNJetScore", "TauBDTEleScore")
OUTPUT_DIR = VALIDATION_OUTPUT / "low_met_fake_region"
SUMMARY_PATH = OUTPUT_DIR / "low_met_fake_region_summary.md"
CACHE_FILE = OUTPUT_DIR / "root" / "validate_low_met_fake_region.root"
NOMINAL_MEASURED_ROOT = (
    REPO_ROOT
    / "outputs"
    / "analysis_shadow_unfold"
    / "measured"
    / "root"
    / "analysis_shadow_unfold_measured.root"
)


@dataclass(frozen=True)
class FakeMethod:
    key: str
    label: str
    cuts: tuple[Cut, ...]


@dataclass(frozen=True)
class ValidationTarget:
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


def validation_target(analysis: Analysis, selection: str) -> ROOT.TH1:
    data = analysis.get_hist(
        VARIABLE,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                VARIABLE,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    target = data - nonfake
    target.SetName(f"{selection}_{VARIABLE}_data_minus_nonfake")
    target.SetDirectory(0)
    return target


def fake_enrichment_components(
    analysis: Analysis,
    selection: str,
    variable: str,
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    data = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    nonfake = sum_th1s(
        *[
            analysis.get_hist(
                variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
        ]
    )
    fake_like = data - nonfake
    for hist, suffix in (
        (data, "data"),
        (nonfake, "nonfake_mc"),
        (fake_like, "data_minus_nonfake"),
    ):
        hist.SetName(f"{selection}_{variable}_{suffix}")
        hist.SetDirectory(0)
    return data, nonfake, fake_like


def sum_fake_enrichment_components(
    analysis: Analysis,
    selections: tuple[str, ...],
    variable: str,
    name: str,
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    components = [
        fake_enrichment_components(analysis, selection, variable)
        for selection in selections
    ]
    data = sum_th1s(*[component[0] for component in components])
    nonfake = sum_th1s(*[component[1] for component in components])
    fake_like = sum_th1s(*[component[2] for component in components])
    for hist, suffix in (
        (data, "data"),
        (nonfake, "nonfake_mc"),
        (fake_like, "data_minus_nonfake"),
    ):
        hist.SetName(f"{name}_{variable}_{suffix}")
        hist.SetDirectory(0)
    return data, nonfake, fake_like


def load_root_histograms(root_file_path: Path, hist_names: tuple[str, ...]) -> list[ROOT.TH1]:
    loaded_hists: list[ROOT.TH1] = []
    root_file = ROOT.TFile.Open(str(root_file_path), "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT file: {root_file_path}")
    try:
        for hist_name in hist_names:
            hist = root_file.Get(hist_name)
            if not hist:
                raise KeyError(f"Missing histogram '{hist_name}' in {root_file_path}")
            cloned_hist = hist.Clone(hist_name)
            cloned_hist.SetDirectory(0)
            loaded_hists.append(cloned_hist)
    finally:
        root_file.Close()
    return loaded_hists


def variable_label(variable: str) -> str:
    label = variable_data[variable]["name"]
    if variable in {"MTW", "MET_met", "TauPt"}:
        label += " [GeV]"
    return label


def current_data_mc_fakes_components(
    analysis: Analysis,
    selection: str,
    variable: str,
    fake_hists: tuple[ROOT.TH1, ...],
) -> tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1, ROOT.TH1]:
    signal = analysis.get_hist(
        variable,
        dataset="wtaunu_had",
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    signal.SetName(f"{selection}_{variable}_signal")
    signal.SetDirectory(0)
    simulated_background = sum_th1s(
        *[
            analysis.get_hist(
                variable,
                dataset=mc_sample,
                systematic=NOMINAL_NAME,
                selection=f"trueTau_{selection}",
                allow_generation=True,
            )
            for mc_sample in MC_SAMPLES
            if mc_sample != "wtaunu_had"
        ]
    )
    simulated_background.SetName(f"{selection}_{variable}_simulated_background")
    simulated_background.SetDirectory(0)
    fakes = sum_th1s(*fake_hists)
    fakes.SetName(f"{selection}_{variable}_data_driven_jet_fakes")
    fakes.SetDirectory(0)
    data = analysis.get_hist(
        variable,
        dataset=analysis.data_sample,
        systematic=NOMINAL_NAME,
        selection=selection,
        allow_generation=True,
    )
    data.SetName(f"{selection}_{variable}_data")
    data.SetDirectory(0)
    return signal, simulated_background, fakes, data


if __name__ == "__main__":
    base_cuts = [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau > 170$", f"TauPt > {TAUPT_MIN:g}"),
        PASS_ETA,
    ]
    derivation_methods = (
        FakeMethod(
            "current_mtw_shadow_metlt170",
            "current MTW-shadow CR, MET < 170",
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
            "low-MET fake-enriched CR, MET < 100",
            (Cut("MET < 100", "MET_met < 100"),),
        ),
    )
    validation_targets = (
        ValidationTarget(
            "nominal_mtw_control_metlt170",
            "MTW >= 350, MET < 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET < 170", "MET_met < 170"),
            ),
        ),
        ValidationTarget(
            "signal_like_metgt170",
            "MTW >= 350, MET >= 170",
            (
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET >= 170", "MET_met >= 170"),
            ),
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

        for target in validation_targets:
            prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            data_selections[f"{prefix}_validate_passID"] = (
                base_cuts + list(target.cuts) + [PASS_MEDIUM, prong_cut]
            )
            data_selections[f"{prefix}_validate_failID"] = (
                base_cuts + list(target.cuts) + [FAIL_MEDIUM, prong_cut]
            )

    thesis_like_stack_selections = {
        f"{CONFIG_LABEL}_low_met_{WP}_CR_passID": (
            base_cuts + [Cut("MET < 100", "MET_met < 100"), PASS_MEDIUM]
        ),
        f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID": (
            base_cuts
            + [
                Cut(r"$m_T^W >= 350$", f"MTW >= {MTW_NOMINAL_MIN:g}"),
                Cut("MET >= 170", "MET_met >= 170"),
                PASS_MEDIUM,
            ]
        ),
    }
    data_selections.update(thesis_like_stack_selections)

    for selection, cuts in data_selections.items():
        mc_selections[selection] = cuts
        mc_selections[f"trueTau_{selection}"] = cuts + [PASS_TRUETAU]

    cache_exists = CACHE_FILE.is_file()
    run_event_loops = FORCE_REBUILD_HISTS or (
        RUN_EVENT_LOOPS_IF_CACHE_MISSING and (not LOAD_SAVED_HISTS or not cache_exists)
    )

    analysis = Analysis(
        analysis_samples(mc_selections, data_selections=data_selections, snapshot=False),
        year=YEAR,
        rerun=run_event_loops,
        regen_histograms=run_event_loops,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label="validate_low_met_fake_region",
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
        histogram_vars={
            VARIABLE,
            FAKES_SOURCE,
            *THESIS_LIKE_STACK_VARS,
            *FAKE_CONTAMINATION_VARS,
        },
        binnings={"": BINNINGS},
    )

    loaded_hists = (
        LOAD_SAVED_HISTS
        and not FORCE_REBUILD_HISTS
        and analysis.load_hists_if_available(CACHE_FILE)
    )
    if not loaded_hists and not run_event_loops:
        raise FileNotFoundError(
            "Missing low-MET fake-region cache and event loops are disabled: "
            f"{CACHE_FILE}"
        )

    lines = [
        "# Low-MET fake-enriched fake-factor validation",
        "",
        "This validation tests an ATLAS-like low-MET fake-enriched fake-factor derivation "
        "against the current MTW-shadow control-region derivation. It is a validation-only "
        "study, not a nominal unfolding change.",
        "",
        f"- configuration: `{CONFIG_LABEL}`",
        f"- fake source variable: `{FAKES_SOURCE}`",
        f"- target variable: `{VARIABLE}`",
        f"- cache file: `{CACHE_FILE.relative_to(REPO_ROOT)}`",
        f"- event loops run in this invocation: `{run_event_loops}`",
        "",
        "## Transfer summary",
        "",
        "| Method | Target | Prong | CR numerator | CR denominator | "
        "Negative numerator bins | Tiny/non-positive denominator bins | "
        "Validation fail-ID input | Predicted fakes | Validation target | "
        "Prediction / target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    comparison_hists: dict[tuple[str, int], list[tuple[str, ROOT.TH1]]] = {}
    fake_enrichment_hists: dict[
        tuple[int, str],
        tuple[ROOT.TH1, ROOT.TH1, ROOT.TH1],
    ] = {}

    for target in validation_targets:
        for prong in PRONGS:
            target_prefix = f"{CONFIG_LABEL}_{target.key}_{WP}_{prong}prong"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            target_hist = validation_target(analysis, validate_pass)
            comparison_hists[(target.key, prong)] = [
                ("Data - simulated backgrounds", target_hist)
            ]

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
                        f"trueTau_{validate_fail}",
                        name=estimate_name,
                        systematic=NOMINAL_NAME,
                        save_intermediates=True,
                    )

                numerator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_numerator"]
                denominator = analysis.histograms[f"{estimate_name}_{FAKES_SOURCE}_FF_denominator"]
                validation_fail_input = analysis.histograms[
                    f"{estimate_name}_{FAKES_SOURCE}_FF_fakes_data_est"
                ]
                prediction = analysis.histograms[
                    f"{estimate_name}_{VARIABLE}_fakes_bkg_{FAKES_SOURCE}_src"
                ]
                negative_numerator_bins, tiny_denominator_bins = fake_factor_bin_health(
                    numerator, denominator
                )
                predicted_integral = hist_integral(prediction)
                target_integral = hist_integral(target_hist)

                lines.append(
                    f"| {method.label} | {target.label} | {prong} | "
                    f"{hist_integral(numerator):.3f} | {hist_integral(denominator):.3f} | "
                    f"{negative_numerator_bins} | {tiny_denominator_bins} | "
                    f"{hist_integral(validation_fail_input):.3f} | {predicted_integral:.3f} | "
                    f"{target_integral:.3f} | {ratio(predicted_integral, target_integral):.3f} |"
                )
                comparison_hists[(target.key, prong)].append((method.label, prediction))

    low_met_method = next(
        method for method in derivation_methods if method.key == "low_met_fake_enriched"
    )
    lines.extend(
        [
            "",
            "## Low-MET control-region fake enrichment",
            "",
            "The fake-like component is computed as data minus the simulated "
            "backgrounds in the same region. A small simulated-background fraction indicates a "
            "region dominated by jets misidentified as tau candidates.",
            "",
            "| Region | Prong | Data | Simulated backgrounds | "
            "Inferred jet-fake component | Fake-like / data | Simulated background / data |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for prong in PRONGS:
        method_prefix = f"{CONFIG_LABEL}_{low_met_method.key}_{WP}_{prong}prong"
        for region_key, region_label in (
            ("derive_passID", "Pass-ID numerator"),
            ("derive_failID", "Anti-ID denominator"),
        ):
            selection = f"{method_prefix}_{region_key}"
            data_hist, nonfake_hist, fake_like_hist = fake_enrichment_components(
                analysis,
                selection,
                FAKES_SOURCE,
            )
            fake_enrichment_hists[(prong, region_key)] = (
                data_hist,
                nonfake_hist,
                fake_like_hist,
            )
            data_yield = hist_integral(data_hist)
            nonfake_yield = hist_integral(nonfake_hist)
            fake_like_yield = hist_integral(fake_like_hist)
            lines.append(
                f"| {region_label} | {prong} | {data_yield:.3f} | "
                f"{nonfake_yield:.3f} | {fake_like_yield:.3f} | "
                f"{ratio(fake_like_yield, data_yield):.3f} | "
                f"{ratio(nonfake_yield, data_yield):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "The low-MET fake-enriched method is interesting only if it improves the "
            "`Prediction / target` ratio without creating unhealthy source bins. If it "
            "does better in the `MTW >= 350, MET < 170` target but not in the signal-like "
            "`MET >= 170` proxy, the difference should be treated as a transfer uncertainty "
            "rather than a nominal model change.",
            "",
            "The current MTW-shadow control-region row is included so the low-MET method can "
            "be judged against the current fake-factor construction in the same script and "
            "with the same target definitions.",
        ]
    )

    if PLOT_TRANSFER_COMPARISON:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "low_met_fake_region"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Representative plots", ""])
        for (target_key, prong), labelled_hists in comparison_hists.items():
            filename = f"{CONFIG_LABEL}_{target_key}_{prong}prong_low_met_fake_region.png"
            analysis.plot(
                [hist for _label, hist in labelled_hists],
                label=[label for label, _hist in labelled_hists],
                colour=["k", "tab:blue", "tab:orange"],
                histstyle=["step", "step", "step"],
                linestyle=["-", "-", "--"],
                xlabel=variable_data[VARIABLE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"{CONFIG_LABEL} | {target_key} | {prong}-prong",
                kind="overlay",
                do_stat=True,
                do_syst=False,
                logx=True,
                ratio_plot=True,
                ratio_label="Prediction / target",
                ratio_axlim=(0.0, 2.0),
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=filename,
            )
            lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_NOMINAL_LOW_MET_FAKE_FACTORS:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "fake_factors"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Nominal low-MET fake-factor plots", ""])
        low_met_fake_factors = load_root_histograms(
            NOMINAL_MEASURED_ROOT,
            (
                "no_shadow_bin_medium_3prong_lowMET_TauPt_FF",
                "no_shadow_bin_medium_1prong_lowMET_TauPt_FF",
            ),
        )
        filename = "no_shadow_bin_medium_TauPt_lowMET_prong_fake_factors.png"
        analysis.plot(
            low_met_fake_factors,
            label=["3-prong", "1-prong"],
            colour=["tab:orange", "tab:blue"],
            histstyle=["step", "step"],
            xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
            ylabel="Fake factor",
            title="No shadow bin | low-MET fake factors",
            kind="overlay",
            do_stat=False,
            do_syst=False,
            label_params={"llabel": "", "loc": 0},
            legend_params={"fontsize": 10, "loc": "upper right"},
            filename=filename,
        )
        lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    built_current_data_mc_fakes = False
    if PLOT_CURRENT_DATA_MC_FAKES_STACKS:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "current_data_mc_fakes_stacks"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Current data/MC plus fake-estimate stacks", ""])

        signal_like_target = next(
            target for target in validation_targets if target.key == "signal_like_metgt170"
        )
        fake_histograms: dict[tuple[str, int], ROOT.TH1] = {}
        for prong in PRONGS:
            method_prefix = f"{CONFIG_LABEL}_{low_met_method.key}_{WP}_{prong}prong"
            target_prefix = f"{CONFIG_LABEL}_{signal_like_target.key}_{WP}_{prong}prong"
            derive_pass = f"{method_prefix}_derive_passID"
            derive_fail = f"{method_prefix}_derive_failID"
            validate_pass = f"{target_prefix}_validate_passID"
            validate_fail = f"{target_prefix}_validate_failID"
            estimate_name = f"{method_prefix}_{signal_like_target.key}_{FAKES_SOURCE}_src"
            missing_fakes = [
                variable
                for variable in THESIS_DATA_MC_STACK_VARS
                if f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src"
                not in analysis.histograms
            ]
            if missing_fakes:
                built_current_data_mc_fakes = True
                analysis.do_fakes_estimate(
                    FAKES_SOURCE,
                    tuple(THESIS_DATA_MC_STACK_VARS),
                    derive_pass,
                    derive_fail,
                    validate_pass,
                    validate_fail,
                    f"trueTau_{derive_pass}",
                    f"trueTau_{derive_fail}",
                    f"trueTau_{validate_fail}",
                    name=estimate_name,
                    systematic=NOMINAL_NAME,
                    save_intermediates=True,
                )
            for variable in THESIS_DATA_MC_STACK_VARS:
                fake_histograms[(variable, prong)] = analysis.histograms[
                    f"{estimate_name}_{variable}_fakes_bkg_{FAKES_SOURCE}_src"
                ]

        stack_selection = f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID"
        for variable in THESIS_DATA_MC_STACK_VARS:
            signal, simulated_background, fakes, data = current_data_mc_fakes_components(
                analysis,
                stack_selection,
                variable,
                tuple(fake_histograms[(variable, prong)] for prong in PRONGS),
            )
            mc_no_fakes = sum_th1s(signal, simulated_background)
            mc_no_fakes.SetName(f"{stack_selection}_{variable}_mc_signal_background")
            mc_no_fakes.SetDirectory(0)
            mc_with_fakes = sum_th1s(signal, simulated_background, fakes)
            mc_with_fakes.SetName(
                f"{stack_selection}_{variable}_mc_signal_background_fakes"
            )
            mc_with_fakes.SetDirectory(0)
            for yscale, logy in (("liny", False), ("log", True)):
                filename = f"{WP}_{variable}_current_signal_background_fakes_{yscale}.png"
                analysis.plot(
                    [data, mc_with_fakes, mc_no_fakes],
                    label=[
                        "Data",
                        "MC (signal + backgrounds) + fakes",
                        "MC (signal + backgrounds), no fakes",
                    ],
                    colour=["k", "tab:orange", "tab:blue"],
                    histstyle=["errorbar", "step", "step"],
                    uncert=[
                        get_th1_bin_errors(data),
                        get_th1_bin_errors(mc_with_fakes),
                        get_th1_bin_errors(mc_no_fakes),
                    ],
                    xlabel=variable_label(variable),
                    ylabel="Events",
                    title="Medium tau ID | signal region",
                    kind="overlay",
                    do_stat=False,
                    do_syst=False,
                    ratio_plot=True,
                    ratio_label="Prediction / Data",
                    ratio_axlim=(0.5, 1.5),
                    logy=logy,
                    label_params={"llabel": "", "loc": 1},
                    legend_params={"fontsize": 9, "loc": "upper right"},
                    filename=filename,
                    sort=False,
                    capsize=0,
                )
                lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_THESIS_LIKE_REGION_STACKS:
        lines.extend(["", "## Thesis-like CR/SR stack plots", ""])
        thesis_plot_dir = OUTPUT_DIR / "plots" / "thesis_like_region_stacks"
        stack_plot_args = {
            "dataset": [analysis.data_sample, *MC_SAMPLES],
            "do_stat": True,
            "do_syst": False,
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
            "label_params": {"llabel": "", "loc": 1},
            "legend_params": {"fontsize": 9, "loc": "upper right"},
        }
        for selection, region_label in (
            (f"{CONFIG_LABEL}_low_met_{WP}_CR_passID", "low-MET control region"),
            (f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID", "signal region"),
        ):
            analysis.paths.plot_dir = thesis_plot_dir / selection
            analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
            for variable in THESIS_LIKE_STACK_VARS:
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = f"{WP}_{variable}_stack_no_fakes_{yscale}.png"
                    analysis.plot(
                        val=variable,
                        selection=selection,
                        xlabel=variable_data[variable]["name"],
                        ylabel="Events",
                        title=f"Medium tau ID | {region_label}",
                        filename=filename,
                        logy=logy,
                        **stack_plot_args,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_THESIS_LIKE_PRONG_STACKS:
        lines.extend(["", "## Thesis-like prong-split signal-region stack plots", ""])
        thesis_prong_plot_dir = OUTPUT_DIR / "plots" / "thesis_like_prong_stacks"
        stack_plot_args = {
            "dataset": [analysis.data_sample, *MC_SAMPLES],
            "do_stat": True,
            "do_syst": False,
            "ratio_plot": True,
            "ratio_axlim": (0.8, 1.2),
            "kind": "stack",
            "label_params": {"llabel": "", "loc": 1},
            "legend_params": {"fontsize": 9, "loc": "upper right"},
        }
        for prong in PRONGS:
            selection = (
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_passID"
            )
            analysis.paths.plot_dir = (
                thesis_prong_plot_dir
                / f"{CONFIG_LABEL}_signal_like_{WP}_SR_passID_{prong}prong"
            )
            analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
            for variable in ("TauRNNJetScore", "TauBDTEleScore"):
                for yscale, logy in (("liny", False), ("log", True)):
                    filename = f"{WP}_{prong}prong_{variable}_stack_no_fakes_{yscale}.png"
                    analysis.plot(
                        val=variable,
                        selection=selection,
                        xlabel=variable_data[variable]["name"],
                        ylabel="Events",
                        title=f"Medium tau ID | signal region | {prong}-prong",
                        filename=filename,
                        logy=logy,
                        **stack_plot_args,
                    )
                    lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_FAKE_ENRICHMENT:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "fake_enrichment"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Fake-enrichment plots", ""])
        for (prong, region_key), (
            data_hist,
            nonfake_hist,
            fake_like_hist,
        ) in fake_enrichment_hists.items():
            region_label = (
                "pass-ID numerator"
                if region_key == "derive_passID"
                else "anti-ID denominator"
            )
            filename = (
                f"{CONFIG_LABEL}_low_met_{region_key}_{prong}prong_"
                f"{FAKES_SOURCE}_fake_enrichment.png"
            )
            analysis.plot(
                [fake_like_hist, nonfake_hist],
                label=[
                    "Inferred jet-fake component",
                    "Simulated backgrounds",
                ],
                colour=["tab:orange", "tab:blue"],
                plot_as_data=data_hist,
                data_label="Data",
                xlabel=variable_data[FAKES_SOURCE]["name"] + " [GeV]",
                ylabel="Events",
                title=f"Low-MET control region | {region_label} | {prong}-prong",
                kind="stack",
                do_stat=False,
                do_syst=False,
                logx=True,
                logy=True,
                label_params={"llabel": "", "loc": 1},
                legend_params={"fontsize": 10, "loc": "upper right"},
                filename=filename,
                sort=False,
            )
            lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if PLOT_MEDIUM_FAKE_CONTAMINATION:
        analysis.paths.plot_dir = OUTPUT_DIR / "plots" / "medium_fake_contamination"
        analysis.paths.plot_dir.mkdir(parents=True, exist_ok=True)
        lines.extend(["", "## Medium fake-contamination plots", ""])
        region_selections = {
            "medium_CR_passID": tuple(
                f"{CONFIG_LABEL}_low_met_fake_enriched_{WP}_{prong}prong_derive_passID"
                for prong in PRONGS
            ),
            "medium_CR_failID": tuple(
                f"{CONFIG_LABEL}_low_met_fake_enriched_{WP}_{prong}prong_derive_failID"
                for prong in PRONGS
            ),
            "medium_SR_passID": tuple(
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_passID"
                for prong in PRONGS
            ),
            "medium_SR_failID": tuple(
                f"{CONFIG_LABEL}_signal_like_metgt170_{WP}_{prong}prong_validate_failID"
                for prong in PRONGS
            ),
        }
        region_titles = {
            "medium_CR_passID": "Low-MET determination region | pass-ID",
            "medium_CR_failID": "Low-MET determination region | anti-ID",
            "medium_SR_passID": "Signal-like application region | pass-ID",
            "medium_SR_failID": "Signal-like application region | anti-ID",
        }
        for region_key, selections in region_selections.items():
            for variable in FAKE_CONTAMINATION_VARS:
                data_hist, nonfake_hist, fake_like_hist = sum_fake_enrichment_components(
                    analysis,
                    selections,
                    variable,
                    region_key,
                )
                filename = f"all_mc_{variable}_{region_key}_fake_fractions.png"
                xlabel = variable_data[variable]["name"]
                if variable in {"MET_met"}:
                    xlabel += " [GeV]"
                analysis.plot(
                    [fake_like_hist, nonfake_hist],
                    label=[
                        "Inferred jet-to-tau component",
                        "Simulated backgrounds",
                    ],
                    colour=["tab:orange", "tab:blue"],
                    plot_as_data=data_hist,
                    data_label="Data",
                    xlabel=xlabel,
                    ylabel="Events",
                    title=region_titles[region_key],
                    kind="stack",
                    do_stat=False,
                    do_syst=False,
                    logy=True,
                    label_params={"llabel": "", "loc": 1},
                    legend_params={"fontsize": 10, "loc": "upper right"},
                    filename=filename,
                    sort=False,
                )
                lines.append(f"- `{analysis.paths.plot_dir / filename}`")

    if run_event_loops or FORCE_REBUILD_HISTS or built_current_data_mc_fakes:
        analysis.save_hists(filename=CACHE_FILE.name)

    write_markdown(SUMMARY_PATH, lines)
    print(f"Wrote {SUMMARY_PATH}")
