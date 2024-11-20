from pathlib import Path
from typing import Dict

import numpy as np
from tabulate import tabulate

from src.analysis import Analysis
from src.cutting import Cut
from utils.ROOT_utils import bayes_divide
from utils.plotting_tools import Hist2dOpts
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
pass_matched_reco = Cut(
    r"Pass preselection",
    r"(passReco == 1) && "
    r"(TauBaselineWP == 1) && "
    r"(abs(TauCharge) == 1) && "
    r"passMetTrigger && "
    r"((TauNCoreTracks == 1) || (TauNCoreTracks == 3)) &&"
    r"(MatchedTruthParticle_isHadronicTau == true) && "
    r"(MatchedTruthParticlePt == TruthTauPt)",
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
    r"TruthTau_isHadronic && ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))",
)
truth_tau_1prong = Cut(
    r"1-prong truth",
    r"TruthTau_nChargedTracks == 1",
)
truth_tau_3prong = Cut(
    r"3-prong truth",
    r"TruthTau_nChargedTracks == 3",
)
reco_tau_1prong = Cut(
    r"1-prong Reconstructed Hadronic Tau",
    "(TauNCoreTracks == 1) && (TruthTau_nChargedTracks == 1)",
)
reco_tau_3prong = Cut(
    r"3-prong Reconstructed Hadronic Tau",
    "(TauNCoreTracks == 3) && (TruthTau_nChargedTracks == 3)",
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
pass_SR = pass_SR_truth
truth_cuts = [
    pass_truth,
    pass_SR,
    truth_tau,
]
selections: dict[str, list[Cut]] = {
    "truth_tau": truth_cuts,
    "1prong_truth_tau": truth_cuts
    + [
        truth_tau_1prong,
    ],
    "3prong_truth_tau": truth_cuts
    + [
        truth_tau_3prong,
    ],
    "1prong_truth_tau_trigger": truth_cuts
    + [
        truth_tau_1prong,
        pass_trigger,
    ],
    "3prong_truth_tau_trigger": truth_cuts
    + [
        truth_tau_3prong,
        pass_trigger,
    ],
    "1prong_truth_tau_mettrigger": truth_cuts
    + [
        truth_tau_1prong,
        pass_met_trigger,
    ],
    "3prong_truth_tau_mettrigger": truth_cuts
    + [
        truth_tau_3prong,
        pass_met_trigger,
    ],
    "vl_reco_tau": truth_cuts
    + [
        pass_matched_reco,
    ],
    "vl_1prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        reco_tau_1prong,
    ],
    "vl_3prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        reco_tau_3prong,
    ],
    "loose_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_loose,
    ],
    "loose_1prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_loose,
        reco_tau_1prong,
    ],
    "loose_3prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_loose,
        reco_tau_3prong,
    ],
    "medium_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_medium,
    ],
    "medium_1prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_medium,
        reco_tau_1prong,
    ],
    "medium_3prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_medium,
        reco_tau_3prong,
    ],
    "tight_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_tight,
    ],
    "tight_1prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_tight,
        reco_tau_1prong,
    ],
    "tight_3prong_reco_tau": truth_cuts
    + [
        pass_matched_reco,
        pass_tight,
        reco_tau_3prong,
    ],
}

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "TauPt",
    "MET_met",
    "TruthTauPt",
    "VisTruthTauPt",
    "TruthNeutrinoPt",
    "MTW",
    "TruthMTW",
    # "TruthBosonM",
    # "TruthDilepM",
]
measurement_vars_unitless = [
    "TauEta",
    "TauPhi",
    "MET_phi",
    "TruthTauEta",
    "TruthTauPhi",
    "VisTruthTauEta",
    "VisTruthTauPhi",
    "TruthNeutrinoEta",
    # "nJets",
    "TruthTau_nChargedTracks",
    "TruthTau_nNeutralTracks",
    "TauPt_res_frac",
    "TauPt_res",
    # "TauNCoreTracks",
    # "TauRNNJetScore",
    # "TauBDTEleScore",
    # "DeltaPhi_tau_met",
    # "TruthDeltaPhi_tau_met",
    # "TauPt_div_MET",
    # "TruthTauPt_div_MET",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
truth_measurement_vars = [v for v in measurement_vars if variable_data[v]["tag"] == "truth"]
reco_measurement_vars = [
    v
    for v in measurement_vars
    if (variable_data[v]["tag"] == "reco") and (v not in ("TauPt_res_frac", "TauPt_res"))
]

# define 2d histograms
hists_2d = {
    "TauPt_TruthTauPt": Hist2dOpts("TauPt", "TruthTauPt"),
    "TauPt_VisTruthTauPt": Hist2dOpts("TauPt", "VisTruthTauPt"),
    "TauEta_TruthTauEta": Hist2dOpts("TauEta", "TruthTauEta"),
    "TauEta_VisTruthTauEta": Hist2dOpts("TauEta", "VisTruthTauEta"),
    "MTW_TauPt": Hist2dOpts("MTW", "TauPt"),
    "MET_met_TruthNeutrinoPt": Hist2dOpts("MET_met", "TruthNeutrinoPt"),
}
for v in reco_measurement_vars:
    hists_2d[f"{v}_TauPt_res_frac"] = Hist2dOpts(
        x=v,
        y="TauPt_res_frac",
        weight="reco_weight",
    )
    hists_2d[f"{v}_TauPt_res"] = Hist2dOpts(
        x=v,
        y="TauPt_res",
        weight="reco_weight",
    )
NOMINAL_NAME = "T_s1thv_NOMINAL"


def run_analysis() -> Analysis:
    """Run analysis"""

    nedges = 21
    return Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=False,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        selections=selections,
        analysis_label="signal_reconstruction",
        log_level=10,
        log_out="both",
        extract_vars=measurement_vars + ["TauPt_res_frac"],
        import_missing_columns_as_nan=True,
        snapshot=False,
        hists_2d=hists_2d,
        do_weights=False,
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
                "TauPhi": np.linspace(-2.5, 2.5, nedges),
                "TruthTauEta": np.linspace(-2.5, 2.5, nedges),
                "TruthTauPhi": np.linspace(-2.5, 2.5, nedges),
                "VisTruthTauEta": np.linspace(-2.5, 2.5, nedges),
                "VisTruthTauPhi": np.linspace(-2.5, 2.5, nedges),
                "TruthNeutrinoEta": np.linspace(-2.5, 2.5, nedges),
                "MET_met": np.geomspace(150, 1000, nedges),
                "MET_eta": np.linspace(-2.5, 2.5, nedges),
                "MET_phi": np.linspace(-2.5, 2.5, nedges),
                "DeltaPhi_tau_met": np.linspace(0, 2 * np.pi, nedges),
                "TruthDeltaPhi_tau_met": np.linspace(0, 3.5, nedges),
                "TauPt_div_MET": np.linspace(0, 3, nedges),
                "TruthTauPt_div_MET": np.linspace(0, 3, nedges),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TauPt_res_frac": np.linspace(-1, 1, 31),
                "TauPt_res": np.linspace(-300, 300, nedges),
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

    working_points = ("vl", "loose", "medium", "tight")
    working_prongs = ("", "1prong_", "3prong_")

    # CALCULATE EFFICIENCIES
    # ========================================================================
    recon_efficiencies = {
        wp: {
            nprong: {
                var: bayes_divide(
                    analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{wp}_{nprong}reco_tau"),
                    analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau"),
                )
                for var in measurement_vars
            }
            for nprong in working_prongs
        }
        for wp in working_points
    }
    trigger_efficiencies = {
        nprong: {
            var: bayes_divide(
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau_trigger"),
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau"),
            )
            for var in measurement_vars
        }
        for nprong in ("1prong_", "3prong_")
    }
    met_trigger_efficiencies = {
        nprong: {
            var: bayes_divide(
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau_mettrigger"),
                analysis.get_hist(var, "wtaunu", NOMINAL_NAME, f"{nprong}truth_tau"),
            )
            for var in measurement_vars
        }
        for nprong in ("1prong_", "3prong_")
    }

    args_eff = {
        "dataset": "wtaunu",
        "systematic": NOMINAL_NAME,
        "title": f"Reconstruction Efficiency | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
        "do_stat": True,
        "do_syst": False,
        "ratio_err": "binom",
        "label_params": {"llabel": "Simulation", "loc": 1},
    }
    args_res = {
        "dataset": "wtaunu",
        "systematic": NOMINAL_NAME,
        "title": f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
        "do_stat": False,
        "do_syst": False,
        "label_params": {"llabel": "Simulation", "loc": 1},
    }

    # TRIGGER EFFICIENCY
    # ========================================================================
    args_eff["selection"] = ""
    args_eff["ylabel"] = r"Trigger Efficiency $\epsilon_\mathrm{trigger}$"
    args_eff["do_stat"] = False

    analysis.paths.plot_dir = base_plotting_dir / "trigger_efficiency"
    for s, effs in (("trigger", trigger_efficiencies), ("met_trigger", met_trigger_efficiencies)):
        for v in truth_measurement_vars:
            if v in measurement_vars_mass:
                args_eff.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                args_eff.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[
                    effs["1prong_"][v],
                    effs["3prong_"][v],
                ],
                label=["1-prong", "3-prong"],
                **args_eff,
                filename=f"{v}_{s}_efficiency_1prong_3prong.png",
            )

    # BEEEG LOOP
    # =======================================================================
    args_eff["do_stat"] = True
    for wp in working_points:
        for nprong in working_prongs:
            # DIRECT FRACTIONS
            # =======================================================================
            analysis.paths.plot_dir = base_plotting_dir / "efficiency/full" / wp / nprong
            args_eff["selection"] = [f"{nprong}truth_tau", f"{wp}_{nprong}reco_tau"]
            args_eff["ylabel"] = f"Events"
            for v in truth_measurement_vars:
                if v in measurement_vars_mass:
                    args_eff.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
                elif v in measurement_vars_unitless:
                    args_eff.update({"logx": False, "xlabel": variable_data[v]["name"]})

                analysis.plot(
                    val=v,
                    label=["Particle-Level", "Detector-Level"],
                    **args_eff,
                    filename=f"{wp}_{v}_{nprong}_overlay_compare.png",
                    ratio_label=r"$\epsilon_\mathrm{reco}$",
                    ratio_plot=True,
                    ratio_fit=True,
                    ratio_axlim=(0, 1),
                )

            # 2D RESOLUTION PLOTS
            # ========================================================================
            analysis.paths.plot_dir = base_plotting_dir / "resolution/full"
            selection = f"{wp}_{nprong}reco_tau"
            # profiles
            for v in reco_measurement_vars:
                analysis.plot_2d(
                    v,
                    "TauPt_res_frac",
                    dataset="wtaunu",
                    systematic=NOMINAL_NAME,
                    selection=selection,
                    ylabel=r"$p_\mathrm{T,res-frac}^\tau$",
                    title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    norm="log",
                    logx=True if v in measurement_vars_mass else False,
                    logy=False,
                    label_params={"llabel": "Simulation"},
                    filename=f"{v}_TauPt_res_frac_2D_{selection}.png",
                )
                analysis.plot_2d(
                    v,
                    "TauPt_res",
                    dataset="wtaunu",
                    systematic=NOMINAL_NAME,
                    selection=selection,
                    ylabel=r"$p_\mathrm{T,res}^\tau$ [GeV]",
                    title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                    norm="log",
                    logx=True if v in measurement_vars_mass else False,
                    logy=False,
                    label_params={"llabel": "Simulation"},
                    filename=f"{v}_TauPt_res_2D_{selection}.png",
                )
            analysis.plot_2d(
                "TauPt",
                "TruthTauPt",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                xlabel=r"$p_\mathrm{T}^\mathrm{reco}$ [GeV]",
                ylabel=r"$p_\mathrm{T}^\mathrm{true}$ [GeV]",
                logx=True,
                logy=True,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"TauPt_TruthTauPt_2D_{selection}.png",
            )
            analysis.plot_2d(
                "TauPt",
                "VisTruthTauPt",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                xlabel=r"$p_\mathrm{T}^\mathrm{reco}$ [GeV]",
                ylabel=r"$p_\mathrm{T}^\mathrm{had-vis}$ [GeV]",
                logx=True,
                logy=True,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"TauPt_VisTruthTauPt_2D_{selection}.png",
            )
            analysis.plot_2d(
                "TauEta",
                "TruthTauEta",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                logx=False,
                logy=False,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"TauEta_TruthTauEta_res_2D_{selection}.png",
            )
            analysis.plot_2d(
                "TauEta",
                "VisTruthTauEta",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                logx=False,
                logy=False,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"TauEta_VisTruthTauEta_res_2D_{selection}.png",
            )
            analysis.plot_2d(
                "MTW",
                "TauPt",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                logx=True,
                logy=True,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"MTW_TauPt_2D_{selection}.png",
            )
            analysis.plot_2d(
                "MET_met",
                "TruthNeutrinoPt",
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                logx=True,
                logy=True,
                title=f"Tau $p_T$ resolution | mc16d | {analysis.global_lumi / 1000:.3g}fb$^{{-1}}$",
                norm="log",
                label_params={"llabel": "Simulation"},
                filename=f"MET_met_TruthNeutrinoPt_2D_{selection}.png",
            )

    # START OF PRONG LOOP
    # ========================================================================
    args_eff["ylabel"] = r"Reconstruction Efficiency $\epsilon$"
    for nprong in working_prongs:
        # RECONSTRUCTION EFFICIENCY
        # ========================================================================
        analysis.paths.plot_dir = base_plotting_dir / "efficiency" / nprong
        args_eff["selection"] = None
        for v in truth_measurement_vars:
            if v in measurement_vars_mass:
                args_eff.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                args_eff.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[recon_efficiencies[wp][nprong][v] for wp in working_points],
                label=["Very Loose", "Loose", "Medium", "Tight"],
                **args_eff,
                filename=f"{v}_{nprong}_efficiency_wp_compare.png",
            )

        # RESOLUTION
        # ========================================================================
        analysis.paths.plot_dir = base_plotting_dir / "resolution" / nprong
        selection = [f"{wp}_{nprong}reco_tau" for wp in working_points]
        # profiles
        for v in reco_measurement_vars:
            if v in measurement_vars_mass:
                args_res.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                args_res.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=f"{v}_TauPt_res",
                label=["Very Loose", "Loose", "Medium", "Tight"],
                selection=selection,
                **args_res,
                logy=False,
                filename=f"{v}_TauPt_res_wp_compare.png",
            )
            analysis.plot(
                v,
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                label=["Very Loose", "Loose", "Medium", "Tight"],
                logx=v in measurement_vars_mass,
                logy=False,
                filename=f"{v}_{nprong}wp.png",
            )
        analysis.plot(
            "TauPt_res",
            dataset="wtaunu",
            systematic=NOMINAL_NAME,
            selection=selection,
            label=["Very Loose", "Loose", "Medium", "Tight"],
            logx=False,
            logy=False,
            filename=f"TauPt_res_{nprong}wp.png",
        )
        analysis.plot(
            "TauPt_res_frac",
            dataset="wtaunu",
            systematic=NOMINAL_NAME,
            selection=selection,
            label=["Very Loose", "Loose", "Medium", "Tight"],
            logx=False,
            logy=False,
            filename=f"TauPt_res_frac_{nprong}wp.png",
        )

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in working_points:
        # RECONSTRUCTION EFFICIENCY
        # ========================================================================
        analysis.paths.plot_dir = base_plotting_dir / "efficiency" / wp
        for v in truth_measurement_vars:
            if v in measurement_vars_mass:
                args_eff.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                args_eff.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=[recon_efficiencies[wp]["1prong_"][v], recon_efficiencies[wp]["3prong_"][v]],
                label=["1-prong", "3-prong"],
                **args_eff,
                filename=f"{v}_efficiency_{wp}_1prong_3prong.png",
            )

        # RESOLUTION
        # ========================================================================
        analysis.paths.plot_dir = base_plotting_dir / "resolution" / wp
        selection = [f"{wp}_{nprong}_reco_tau" for nprong in ("1prong", "3prong")]
        # profiles
        for v in reco_measurement_vars:
            if v in measurement_vars_mass:
                args_res.update({"logx": True, "xlabel": variable_data[v]["name"] + " [GeV]"})
            elif v in measurement_vars_unitless:
                args_res.update({"logx": False, "xlabel": variable_data[v]["name"]})

            analysis.plot(
                val=f"{v}_TauPt_res",
                selection=selection,
                label=["1-prong", "3-prong"],
                ylabel=r"$\left<p_\mathrm{T,res}^\tau\right>$ [GeV]",
                **args_res,
                filename=f"{v}_TauPt_res_prong_compare_profile.png",
            )
            analysis.plot(
                val=f"{v}_TauPt_res_frac",
                selection=selection,
                label=["1-prong", "3-prong"],
                ylabel=r"$\left<p_\mathrm{T,res-frac}^\tau\right>$",
                **args_res,
                filename=f"{v}_TauPt_res_frac_prong_compare_profile.png",
            )
            analysis.plot(
                v,
                dataset="wtaunu",
                systematic=NOMINAL_NAME,
                selection=selection,
                label=["1-prong", "3-prong"],
                logx=v in measurement_vars_mass,
                logy=True,
                filename=f"{v}_{wp}_prong_compare.png",
            )
        analysis.plot(
            "TauPt_res",
            dataset="wtaunu",
            systematic=NOMINAL_NAME,
            selection=selection,
            label=["1-prong", "3-prong"],
            logx=False,
            logy=False,
            filename=f"TauPt_res_{wp}_prong_compare.png",
        )
        analysis.plot(
            "TauPt_res_frac",
            dataset="wtaunu",
            systematic=NOMINAL_NAME,
            selection=selection,
            label=["1-prong", "3-prong"],
            logx=False,
            logy=False,
            filename=f"TauPt_res_frac_{wp}_prong_compare.png",
        )

    # just a quick calculation for overall resolution
    res_table = []
    frac_table = []
    header = [
        r"$\left<p_\mathrm{T,res}^{\tau}\right>$ [GeV]",
        r"$\left<p_\mathrm{T,res}^{\tau_{1p}}\right>$ [GeV]",
        r"$\left<p_\mathrm{T,res}^{\tau_{3p}}\right>$ [GeV]",
    ]
    for wp in working_points:
        row_res = [wp]
        row_frac = [wp]

        for nprong in ("", "1prong_", "3prong_"):
            pt = (
                analysis["wtaunu"]
                .filters[NOMINAL_NAME][f"{wp}_{nprong}reco_tau"]
                .df.AsNumpy(["TauPt_res", "TauPt_res_frac"])
            )
            mean_frac = abs(pt["TauPt_res_frac"]).mean()
            mean_res = abs(pt["TauPt_res"]).mean()
            analysis.logger.info(f"{wp}, {nprong}: {mean_res:.3f}, {mean_frac:.3f}")
            row_res.append(f"{mean_res:.3g}")
            row_frac.append(f"{mean_frac:.3g}")
        res_table.append(row_res)
        frac_table.append(row_frac)

    with open(analysis.paths.latex_dir / "resolution_res.tex", "w") as f:
        f.write(tabulate(res_table, headers=header, tablefmt="latex_raw"))
    with open(analysis.paths.latex_dir / "resolution_frac.tex", "w") as f:
        f.write(tabulate(frac_table, headers=header, tablefmt="latex_raw"))

    analysis.logger.info("DONE.")
