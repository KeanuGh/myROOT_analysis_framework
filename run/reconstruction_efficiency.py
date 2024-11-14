from pathlib import Path
from typing import Dict

import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")

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
reco_tau = Cut(
    r"Reconstructed Hadronic Tau",
    r"TauPt > 0",
)
reco_tau_1prong = Cut(
    r"1-prong Reconstructed Hadronic Tau",
    "TauNCoreTracks == 1",
)
reco_tau_3prong = Cut(
    r"3-prong Reconstructed Hadronic Tau",
    "TauNCoreTracks == 3",
)
pass_loose = Cut(
    r"\mathrm{Pass Loose ID}",
    r"(TauBDTEleScore > 0.05) && "
    r"((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
)
pass_medium = Cut(
    r"\mathrm{Pass Medium ID}",
    r"(TauBDTEleScore > 0.1) && "
    r"((TauRNNJetScore > 0.25) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.4) * (TauNCoreTracks == 3))",
)
pass_tight = Cut(
    r"\mathrm{Pass Tight ID}",
    r"(TauBDTEleScore > 0.15) && "
    r"((TauRNNJetScore > 0.4) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.55) * (TauNCoreTracks == 3))",
)
pass_SR_reco = Cut(
    r"Pass SR Reco",
    r"(TauPt > 170) && (MET_met > 150) && (MTW > 150)"
    r"&& ((TauNCoreTracks == 1) || (TauNCoreTracks == 3))",
)
pass_SR_truth = Cut(
    r"Pass SR Truth",
    r"(VisTruthTauPt > 170) && (TruthMTW > 150) && (TruthNeutrinoPt > 150)"
    r"&& ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))",
)
pass_matched_truth_tau = Cut(
    r"Matched Truth Tau",
    r"(MatchedTruthParticle_isHadronicTau == true) && (MatchedTruthParticlePt == TruthTauPt)",
)
selections: dict[str, list[Cut]] = {
    "truth_tau": [
        pass_truth,
        # pass_SR_truth,
    ],
    "1prong_truth_tau": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_1prong,
    ],
    "3prong_truth_tau": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_3prong,
    ],
    "1prong_truth_tau_trigger": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_1prong,
        pass_trigger,
    ],
    "3prong_truth_tau_trigger": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_3prong,
        pass_trigger,
    ],
    "1prong_truth_tau_mettrigger": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_1prong,
        pass_met_trigger,
    ],
    "3prong_truth_tau_mettrigger": [
        pass_truth,
        pass_SR_truth,
        truth_tau,
        truth_tau_3prong,
        pass_met_trigger,
    ],
    "vl_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        reco_tau,
    ],
    "vl_1prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        reco_tau_1prong,
    ],
    "vl_3prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        reco_tau_3prong,
    ],
    "loose_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_loose,
        reco_tau,
    ],
    "loose_1prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_loose,
        reco_tau_1prong,
    ],
    "loose_3prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_loose,
        reco_tau_3prong,
    ],
    "medium_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_medium,
        reco_tau,
    ],
    "medium_1prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_medium,
        reco_tau_1prong,
    ],
    "medium_3prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_medium,
        reco_tau_3prong,
    ],
    "tight_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_tight,
        reco_tau,
    ],
    "tight_1prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_tight,
        reco_tau_1prong,
    ],
    "tight_3prong_reco_tau": [
        pass_presel,
        pass_matched_truth_tau,
        pass_SR_truth,
        pass_tight,
        reco_tau_3prong,
    ],
}

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    # "TauPt",
    "TruthTauPt",
    "VisTruthTauPt",
    # "MTW",
    "TruthMTW",
    # "TruthBosonM",
    # "TruthDilepM",
]
measurement_vars_unitless = [
    # "TauEta",
    # "TauPhi",
    "TruthTauEta",
    "TruthTauPhi",
    "VisTruthTauEta",
    "VisTruthTauPhi",
    "TruthNeutrinoPt",
    "TruthNeutrinoEta",
    # "nJets",
    "TruthTau_nChargedTracks",
    "TruthTau_nNeutralTracks",
    # "TauNCoreTracks",
    # "TauRNNJetScore",
    # "TauBDTEleScore",
    # "DeltaPhi_tau_met",
    # "TruthDeltaPhi_tau_met",
    # "TauPt_div_MET",
    # "TruthTauPt_div_MET",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"


def run_analysis() -> Analysis:
    """Run analysis"""

    nedges = 11
    return Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=False,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        selections=selections,
        analysis_label="reconstruction_efficiency",
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        snapshot=False,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, nedges),
                "TruthMTW": np.geomspace(150, 1000, nedges),
                "TauPt": np.geomspace(170, 1000, nedges),
                "TruthTauPt": np.geomspace(170, 1000, nedges),
                "VisTruthTauPt": np.geomspace(170, 1000, nedges),
                "TruthNeutrinoPt": np.geomspace(170, 1000, nedges),
                "TruthTau_nChargedTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                "TruthTau_nNeutralTracks": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                "TauEta": np.linspace(-2.5, 2.5, nedges),
                "TruthTauEta": np.linspace(-2.5, 2.5, nedges),
                "TruthTauPhi": np.linspace(-2.5, 2.5, nedges),
                "VisTruthTauEta": np.linspace(-2.5, 2.5, nedges),
                "VisTruthTauPhi": np.linspace(-2.5, 2.5, nedges),
                "TruthNeutrinoEta": np.linspace(-2.5, 2.5, nedges),
                "MET_met": np.geomspace(150, 1000, nedges),
                "DeltaPhi_tau_met": np.linspace(0, 2 * np.pi, nedges),
                "TruthDeltaPhi_tau_met": np.linspace(0, 3.5, nedges),
                "TauPt_div_MET": np.linspace(0, 3, nedges),
                "TruthTauPt_div_MET": np.linspace(0, 3, nedges),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TauPt_res": np.linspace(-1, 1, nedges),
                "TauPt_diff": np.linspace(-300, 300, nedges),
            },
            ".*_CR_.*ID": {
                "MET_met": np.geomspace(1, 100, 51),
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

    # calculate efficiencies
    recon_efficiencies = {
        wp: {
            nprong: {
                var: (
                    analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{wp}_{nprong}reco_tau")
                    / analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau")
                )
                for var in measurement_vars
            }
            for nprong in ("", "1prong_", "3prong_")
        }
        for wp in ("vl", "loose", "medium", "tight")
    }
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
        "systematic": NOMINAL_NAME,
        "title": f"mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
        "do_stat": True,
        "do_syst": False,
        "ratio_err": "binom",
        "label_params": {"llabel": "Simulation", "loc": 1},
    }

    # DIRECT FRACTIONS
    # =======================================================================
    analysis.paths.plot_dir = base_plotting_dir / "ratio_overlay"
    for wp in ("vl", "loose", "medium", "tight"):
        for nprong in ("", "1prong_", "3prong_"):
            for v in measurement_vars:
                if v in measurement_vars_mass:
                    default_args.update(
                        {"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"}
                    )
                elif v in measurement_vars_unitless:
                    default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})

                analysis.plot(
                    val=v,
                    selection=[f"{nprong}truth_tau", f"{wp}_{nprong}reco_tau"],
                    label=["Particle-Level", "Detector-Level"],
                    **default_args,
                    filename=f"{wp}_{v}_{nprong}_overlay_compare.png",
                    ylabel="Events",
                    ratio_label=r"$\epsilon_\mathrm{reco}$",
                    ratio_plot=True,
                    ratio_fit=True,
                    ratio_axlim=(0, 1),
                )

    # TRIGGER EFFICIENCY
    # ========================================================================
    default_args["selection"] = ""
    default_args["ylabel"] = r"Trigger Efficiency $\epsilon_\mathrm{trigger}$"

    analysis.paths.plot_dir = base_plotting_dir / "trigger_efficiency"
    for s, effs in (("trigger", trigger_efficiencies), ("met_trigger", met_trigger_efficiencies)):
        for v in measurement_vars:
            if v in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[
                    effs["1prong_"][v],
                    effs["3prong_"][v],
                ],
                label=["1-prong", "3-prong"],
                **default_args,
                filename=f"{v}_{s}_efficiency_1prong_3prong.png",
            )

    # START OF PRONG LOOP
    # ========================================================================
    default_args["ylabel"] = r"Reconstruction Efficiency $\epsilon$"
    wps = ("vl", "loose", "medium", "tight")
    for nprong in ("", "1prong_", "3prong_"):
        analysis.paths.plot_dir = base_plotting_dir / nprong

        # RECONSTRUCTION EFFICIENCY
        # ========================================================================
        for v in measurement_vars:
            if v in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[recon_efficiencies[wp][nprong][v] for wp in wps],
                label=["Very Loose", "Loose", "Medium", "Tight"],
                **default_args,
                filename=f"{v}_{nprong}_efficiency_wp_compare.png",
            )

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in ("vl", "loose", "medium", "tight"):
        analysis.paths.plot_dir = base_plotting_dir / wp

        # RECONSTRUCTION EFFICIENCY
        # ========================================================================
        for v in measurement_vars:
            if v in measurement_vars_mass:
                default_args.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                default_args.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[recon_efficiencies[wp]["1prong_"][v], recon_efficiencies[wp]["3prong_"][v]],
                label=["1-prong", "3-prong"],
                **default_args,
                filename=f"{v}_efficiency_{wp}_1prong_3prong.png",
            )

    analysis.histogram_printout(to_file="txt")
    analysis.logger.info("DONE.")
