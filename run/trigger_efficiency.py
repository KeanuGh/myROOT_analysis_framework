from pathlib import Path
from typing import Dict

import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")
DO_SYS = False

datasets: Dict[str, Dict] = {
    # SIGNAL
    # ====================================================================
    "wtaunu": {
        "data_path": {
            "lm_cut": DTA_PATH / "*Sh_2211_Wtaunu_*_maxHTpTV2*/*.root",
            "full": DTA_PATH / "*Sh_2211_Wtaunu_mW_120*/*.root",
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$W\rightarrow\tau\nu$",
        "is_signal": True,
    },
}

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger"
    # r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)",
)
pass_trigger = Cut(
    r"Pass trigger",
    r"passTrigger",
)
pass_met_trigger = Cut(
    r"Pass met trigger",
    r"passMetTrigger",
)
pass_truth = Cut(
    r"Pass Truth",
    r"(passTruth == 1)",
)
truth_tau = Cut(
    r"Truth Hadronic Tau",
    r"TruthTau_isHadronic",
)
truth_tau_1prong = Cut(
    r"1-prong truth",
    r"TruthTau_nChargedTracks == 1",
)
truth_tau_3prong = Cut(
    r"3-prong truth",
    r"TruthTau_nChargedTracks == 3",
)
selections: dict[str, list[Cut]] = {
    "truth_tau": [
        pass_truth,
    ],
    "1prong_truth_tau": [
        pass_truth,
        truth_tau,
        truth_tau_1prong,
    ],
    "3prong_truth_tau": [
        pass_truth,
        truth_tau,
        truth_tau_3prong,
    ],
    "1prong_truth_tau_trigger": [
        pass_truth,
        truth_tau,
        truth_tau_1prong,
        pass_trigger,
    ],
    "3prong_truth_tau_trigger": [
        pass_truth,
        truth_tau,
        truth_tau_3prong,
        pass_trigger,
    ],
    "1prong_truth_tau_mettrigger": [
        pass_truth,
        truth_tau,
        truth_tau_1prong,
        pass_met_trigger,
    ],
    "3prong_truth_tau_mettrigger": [
        pass_truth,
        truth_tau,
        truth_tau_3prong,
        pass_met_trigger,
    ],
}

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "TauPt",
    "TruthTauPt",
    "VisTruthTauPt",
    "TruthNeutrinoPt",
    "MET_met",
    "MTW",
    "TruthMTW",
    "TruthBosonM",
    "TruthDilepM",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "TruthTauEta",
    "TruthNeutrinoEta",
    "VisTruthTauEta",
    "VisTruthTauPhi",
    "nJets",
    "TruthTau_nChargedTracks",
    "TruthTau_nNeutralTracks",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=DO_SYS,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        selections=selections,
        analysis_label="trigger_efficiency",
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        binnings={
            "": {
                "MTW": np.geomspace(100, 1000, 11),
                "TruthMTW": np.geomspace(100, 1000, 11),
                "TauPt": np.geomspace(100, 1000, 11),
                "TruthTauPt": np.geomspace(100, 1000, 11),
                "VisTruthTauPt": np.geomspace(100, 1000, 11),
                "TruthTau_nChargedTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                "TruthTau_nNeutralTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]),
                "TauEta": np.linspace(-2.5, 2.5, 11),
                "TruthTauEta": np.linspace(-2.5, 2.5, 11),
                "VisTruthTauEta": np.linspace(-2.5, 2.5, 11),
                "MET_met": np.geomspace(1, 1000, 11),
                "TruthNeutrinoPt": np.geomspace(1, 1000, 11),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 11),
                "TruthDeltaPhi_tau_met": np.linspace(0, 3.5, 11),
                "TauPt_div_MET": np.linspace(0, 3, 21),
                "TruthTauPt_div_MET": np.linspace(0, 3, 21),
                "TauNCoreTracks": np.linspace(0, 4, 5),
            },
        },
    )


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()
    analysis.full_cutflow_printout(datasets=["wtaunu"])
    base_plotting_dir = analysis.paths.plot_dir

    # print histograms
    for dataset in analysis:
        dataset.histogram_printout(to_file="txt", to_dir=analysis.paths.latex_dir)

    trigger_efficiencies = {
        nprong: {
            var: (
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau_trigger")
                / analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau")
            )
            for var in measurement_vars
        }
        for nprong in ("1prong_", "3prong_")
    }
    met_trigger_efficiencies = {
        nprong: {
            var: (
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau_mettrigger")
                / analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau")
            )
            for var in measurement_vars
        }
        for nprong in ("1prong_", "3prong_")
    }

    default_args = {
        "dataset": "wtaunu",
        "title": f"mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "do_stat": False,
        "do_syst": False,
        "selection": "",
        "ratio_plot": False,
        "stats_box": False,
        "label_params": {"llabel": "Simulation"},
        "ylabel": r"Trigger Efficiency $\epsilon_\mathrm{trigger}$",
    }

    # TRIGGER EFFICIENCY
    # ========================================================================
    # reco
    analysis.paths.plot_dir = base_plotting_dir / "trigger_efficiency"
    for s, effs in (("trigger", trigger_efficiencies), ("met_trigger", met_trigger_efficiencies)):
        analysis.plot(
            val=[
                effs["1prong_"]["TauPt"],
                effs["3prong_"]["TauPt"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=True,
            xlabel=variable_data["TauPt"]["name"] + " [GeV]",
            filename=f"TauPt_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["TauEta"],
                effs["3prong_"]["TauEta"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=False,
            xlabel=variable_data["TauEta"]["name"],
            filename=f"TauEta_{s}_efficiency_1prong_3prong.png",
        )
        # truth
        analysis.plot(
            val=[
                effs["1prong_"]["TruthTauPt"],
                effs["3prong_"]["TruthTauPt"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=True,
            xlabel=variable_data["TruthTauPt"]["name"] + " [GeV]",
            filename=f"TruthTauPt_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["VisTruthTauPt"],
                effs["3prong_"]["VisTruthTauPt"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=True,
            xlabel=variable_data["VisTruthTauPt"]["name"] + " [GeV]",
            filename=f"VisTruthTauPt_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["TruthTauEta"],
                effs["3prong_"]["TruthTauEta"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=False,
            xlabel=variable_data["TruthTauEta"]["name"],
            filename=f"TruthTauEta_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["VisTruthTauEta"],
                effs["3prong_"]["VisTruthTauEta"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=False,
            xlabel=variable_data["VisTruthTauEta"]["name"],
            filename=f"VisTruthTauEta_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["TruthNeutrinoEta"],
                effs["3prong_"]["TruthNeutrinoEta"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=False,
            xlabel=variable_data["TruthNeutrinoEta"]["name"],
            filename=f"TruthNeutrinoEta_{s}_efficiency_1prong_3prong.png",
        )
        analysis.plot(
            val=[
                effs["1prong_"]["TruthTau_nNeutralTracks"],
                effs["3prong_"]["TruthTau_nNeutralTracks"],
            ],
            label=["1-prong", "3-prong"],
            **default_args,
            logx=False,
            xlabel=variable_data["TruthTau_nNeutralTracks"]["name"],
            filename=f"TruthTau_nNeutralTracks_{s}_efficiency_1prong_3prong.png",
        )

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
