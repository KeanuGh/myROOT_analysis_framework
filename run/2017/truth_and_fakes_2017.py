from pathlib import Path

import numpy as np
from samples import DSID_METADATA_CACHE, NOMINAL_NAME, mc_samples

from src.analysis import Analysis
from src.cutting import Cut
from utils.helper_functions import smart_join
from utils.plotting_tools import ProfileOpts, get_axis_labels

YEAR = 2017
LOAD_SAVED_HISTS = True

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger && (badJet == 0)"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
pass_taupt170 = Cut(
    r"$p_T^\tau > 170$",
    r"TauPt > 170",
)
pass_eta = Cut(
    r"$|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || 1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(TauEta) < 1.37) || (1.52 < abs(TauEta))) && (abs(TauEta) < 2.47))",
)
pass_mtw350 = Cut(
    r"$m_T^W > 350$",
    r"MTW > 350",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
fail_loose = Cut(
    r"\mathrm{Fail Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
fail_medium = Cut(
    r"\mathrm{Fail Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
fail_tight = Cut(
    r"\mathrm{Fail Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"(TauRNNJetScore > 0.01) && "
    r"!((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_met170 = Cut(
    r"$E_T^{\mathrm{miss}} > 170$",
    r"MET_met > 170",
)
pass_150met = Cut(
    r"$E_T^{\mathrm{miss}} < 150$",
    r"MET_met < 150",
)

selections_loose: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_loose,
        pass_met170,
    ],
    "loose_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_loose,
        pass_met170,
    ],
    "loose_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_loose,
        pass_150met,
    ],
    "loose_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_loose,
        pass_150met,
    ],
}
selections_medium: dict[str, list[Cut]] = {
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_medium,
        pass_met170,
    ],
    "medium_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_medium,
        pass_met170,
    ],
    "medium_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_medium,
        pass_150met,
    ],
    "medium_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_medium,
        pass_150met,
    ],
}
selections_tight: dict[str, list[Cut]] = {
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_tight,
        pass_met170,
    ],
    "tight_SR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_tight,
        pass_met170,
    ],
    "tight_CR_passID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        pass_tight,
        pass_150met,
    ],
    "tight_CR_failID": [
        pass_presel,
        pass_taupt170,
        pass_eta,
        pass_mtw350,
        fail_tight,
        pass_150met,
    ],
}
selections = selections_loose | selections_medium | selections_tight

# VARIABLES
# ========================================================================
wanted_variables = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TauNCoreTracks",
    "MatchedTruthParticle_isTau",
    "MatchedTruthParticle_isHadronicTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isPhoton",
    "MatchedTruthParticle_isJet",
    "badJet",
}
measurement_vars_mass = [
    "TauPt",
    "MTW",
    "MET_met",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "TauNCoreTracks",
    "TauRNNJetScore",
    "TauBDTEleScore",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
origin_vars = [
    "MatchedTruthParticle_isHadronicTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isJet",
]
profiles = {
    f"{measurement_var}_{origin_var}": ProfileOpts(
        x=measurement_var,
        y=origin_var,
        weight="reco_weight",
    )
    for measurement_var in measurement_vars
    for origin_var in origin_vars
}
binnings = {
    "": {
        "MTW": np.geomspace(350, 1000, 21),
        "TauPt": np.geomspace(170, 1000, 21),
        "TauEta": np.linspace(-2.5, 2.5, 21),
        "MET_met": np.geomspace(170, 1000, 21),
        "TauPhi": np.linspace(-3.5, 3.5, 21),
        "TauRNNJetScore": np.linspace(0, 1, 51),
        "TauBDTEleScore": np.linspace(0, 1, 51),
        "TauNCoreTracks": np.linspace(0, 4, 5),
    },
    ".*_CR_.*ID": {
        "MET_met": np.geomspace(1, 150, 51),
    },
    "loose_.*failID": {
        "TauRNNJetScore": np.linspace(0, 0.3, 41),
    },
    "medium_.*failID": {
        "TauRNNJetScore": np.linspace(0, 0.45, 41),
    },
    "tight_.*failID": {
        "TauRNNJetScore": np.linspace(0, 0.6, 41),
    },
}
datasets = mc_samples(selections)


def run_analysis() -> Analysis:
    """Build the truth-origin diagnostic analysis."""
    return Analysis(
        datasets,
        year=YEAR,
        rerun=not LOAD_SAVED_HISTS,
        regen_histograms=not LOAD_SAVED_HISTS,
        do_systematics=False,
        metadata_cache=DSID_METADATA_CACHE,
        ttree=NOMINAL_NAME,
        analysis_label=Path(__file__).stem,
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        profiles=profiles,
        binnings=binnings,
    )


if __name__ == "__main__":
    analysis = run_analysis()
    base_plotting_dir = analysis.paths.plot_dir
    origin_colours = [
        "#5DA5DA",  # Jet fakes
        "#FDB515",  # Muon fakes
        "#D65F3A",  # Electron fakes
        "#B8C4C0",  # Hadronic taus
    ]
    origin_labels = [
        "Jet fakes",
        "Muon fakes",
        "Electron fakes",
        "Hadronic taus",
    ]
    origin_suffixes = [
        "MatchedTruthParticle_isJet",
        "MatchedTruthParticle_isMuon",
        "MatchedTruthParticle_isElectron",
        "MatchedTruthParticle_isHadronicTau",
    ]
    mc_sample_names = list(datasets)

    for selection in selections:
        wp_dir = base_plotting_dir / selection
        fake_fraction_y_axlim = (0, 1)

        for variable in measurement_vars:
            xlabel = get_axis_labels(variable)[0]
            profile_hists = [
                analysis.sum_hists(
                    [
                        analysis.get_hist(
                            f"{variable}_{origin_suffix}",
                            dataset=dataset,
                            selection=selection,
                            systematic=NOMINAL_NAME,
                        )
                        for dataset in mc_sample_names
                    ],
                    f"all_mc_{variable}_{origin_suffix}_{selection}_PROFILE",
                )
                for origin_suffix in origin_suffixes
            ]
            analysis.paths.plot_dir = wp_dir / "fakes_distributions"
            analysis.plot(
                profile_hists,
                label=origin_labels,
                systematic=NOMINAL_NAME,
                sort=False,
                do_stat=False,
                colour=origin_colours,
                title=smart_join(
                    ("Fail " if "fail" in selection else "")
                    + ("Loose" if "loose" in selection else ("Medium" if "medium" in selection else "Tight"))
                    + (" SR" if "SR" in selection else " CR"),
                    "MC 2017",
                    f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    sep=" | ",
                ),
                y_axlim=fake_fraction_y_axlim,
                kind="stack",
                xlabel=xlabel,
                logx=variable in measurement_vars_mass,
                label_params={"llabel": "Simulation"},
                ylabel="Fraction of reconstructed tau origin in MC",
                filename=f"all_mc_{variable}_{selection}_fake_fractions.png",
            )

            for dataset in mc_sample_names:
                args = {
                    "dataset": dataset,
                    "selection": selection,
                    "systematic": NOMINAL_NAME,
                }
                analysis.paths.plot_dir = base_plotting_dir / selection / dataset
                analysis.plot(
                    [
                        analysis.get_hist(f"{variable}_{origin_suffix}", **args)
                        for origin_suffix in origin_suffixes
                    ],
                    label=origin_labels,
                    sort=False,
                    do_stat=False,
                    systematic=NOMINAL_NAME,
                    colour=origin_colours,
                    title=f"Truth-origin fractions for {variable} in {dataset}",
                    y_axlim=fake_fraction_y_axlim,
                    kind="stack",
                    xlabel=xlabel,
                    logx=variable in measurement_vars_mass,
                    label_params={"llabel": "Simulation"},
                    ylabel="Fraction of reconstructed tau origin in MC",
                    filename=f"{dataset}_{variable}_{selection}_fake_fractions.png",
                )

    analysis.logger.info("DONE.")
