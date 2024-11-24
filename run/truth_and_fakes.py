from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from src.dataset import ProfileOpts
from utils.plotting_tools import get_axis_labels

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
    # BACKGROUNDS
    # ====================================================================
    # W -> light lepton
    "wlnu": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Wmunu_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Wmunu_mW_120*/*.root",
                DTA_PATH / "*Sh_2211_Wenu_mW_120*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$W\rightarrow (e/\mu)\nu$",
    },
    # Z -> ll
    "zll": {
        "data_path": {
            "lm_cut": [
                DTA_PATH / "*Sh_2211_Ztautau_*_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zee_maxHTpTV2*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_maxHTpTV2*/*.root",
            ],
            "full": [
                DTA_PATH / "*Sh_2211_Ztautau_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zmumu_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Zee_mZ_120*/*.root",
                DTA_PATH / "*Sh_2211_Znunu_pTV2*/*.root",
            ],
        },
        "hard_cut": {"lm_cut": "TruthBosonM < 120"},
        "label": r"$Z\rightarrow (l/\nu)(l/\nu)$",
    },
    "top": {
        "data_path": [
            DTA_PATH / "*PP8_singletop*/*.root",
            DTA_PATH / "*PP8_tchan*/*.root",
            DTA_PATH / "*PP8_Wt_DR_dilepton*/*.root",
            DTA_PATH / "*PP8_ttbar_hdamp258p75*/*.root",
        ],
        "label": "Top",
    },
    # DIBOSON
    "diboson": {
        "data_path": [
            DTA_PATH / "*Sh_2212_llll*/*.root",
            DTA_PATH / "*Sh_2212_lllv*/*.root",
            DTA_PATH / "*Sh_2212_llvv*/*.root",
            DTA_PATH / "*Sh_2212_lvvv*/*.root",
            DTA_PATH / "*Sh_2212_vvvv*/*.root",
            DTA_PATH / "*Sh_2211_ZqqZll*/*.root",
            DTA_PATH / "*Sh_2211_ZbbZll*/*.root",
            DTA_PATH / "*Sh_2211_WqqZll*/*.root",
            DTA_PATH / "*Sh_2211_WlvWqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZqq*/*.root",
            DTA_PATH / "*Sh_2211_WlvZbb*/*.root",
        ],
        "label": "Diboson",
    },
}

# CUTS & SELECTIONS
# ========================================================================
pass_presel = Cut(
    r"Pass preselection",
    r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1) && passMetTrigger"
    r"&& ((MatchedTruthParticle_isTau + MatchedTruthParticle_isElectron + MatchedTruthParticle_isMuon + MatchedTruthParticle_isPhoton) <= 1)",
)
pass_taupt170 = Cut(
    r"$p_T^\tau > 170$",
    r"TauPt > 170",
)
pass_mtw150 = Cut(
    r"$m_T^W > 150$",
    r"MTW > 150",
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
pass_met150 = Cut(
    r"$E_T^{\mathrm{miss}} > 150$",
    r"MET_met > 150",
)
pass_truetau = Cut(
    r"True Tau",
    "MatchedTruthParticle_isHadronicTau == true",
)

# selections
selections: dict[str, list[Cut]] = {
    "loose_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_loose,
        pass_met150,
    ],
    "medium_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_medium,
        pass_met150,
    ],
    "tight_SR_passID": [
        pass_presel,
        pass_taupt170,
        pass_mtw150,
        pass_tight,
        pass_met150,
    ],
}
# define selection for MC samples
selections_list = list(selections.keys())
selections_cuts = list(selections.values())
for selection, cut_list in zip(selections_list, selections_cuts):
    selections[f"trueTau_{selection}"] = cut_list + [pass_truetau]

# VARIABLES
# ========================================================================
wanted_variables = {
    "TauEta",
    "TauPhi",
    "TauPt",
    "MET_met",
    "MET_phi",
    "MTW",
    "DeltaPhi_tau_met",
    "TauPt_div_MET",
    "TauRNNJetScore",
    "TauBDTEleScore",
    "TruthTauPt",
    "TruthTauEta",
    "TruthTauPhi",
    "TauNCoreTracks",
    "TauPt_res",
    "TauPt_diff",
    "MatchedTruthParticlePt",
    "MatchedTruthParticle_isTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isPhoton",
    "MatchedTruthParticle_isJet",
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
    "DeltaPhi_tau_met",
    "TauPt_div_MET",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
profile_vars = [
    "TauPt_res",
    "TauPt_diff",
    "MatchedTruthParticlePt",
    "MatchedTruthParticle_isTau",
    "MatchedTruthParticle_isElectron",
    "MatchedTruthParticle_isMuon",
    "MatchedTruthParticle_isPhoton",
    "MatchedTruthParticle_isJet",
]
# define which profiles to calculate
profiles: dict[str, ProfileOpts] = dict()
for measurement_var in measurement_vars:
    for prof_var in profile_vars:
        profiles[f"{measurement_var}_{prof_var}"] = ProfileOpts(
            x=measurement_var,
            y=prof_var,
            weight="" if "MatchedTruthParticle" in prof_var else "reco_weight",
        )
NOMINAL_NAME = "T_s1thv_NOMINAL"


def run_analysis() -> Analysis:
    """Run analysis"""
    return Analysis(
        datasets,
        year=2017,
        # rerun=True,
        # regen_histograms=True,
        do_systematics=False,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        selections=selections,
        analysis_label="truth_and_fakes",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        profiles=profiles,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 21),
                "TauPt": np.geomspace(170, 1000, 21),
                "MatchedTruthParticlePt": np.geomspace(170, 1000, 21),
                "EleEta": np.linspace(-2.5, 2.5, 21),
                "MuonEta": np.linspace(-2.5, 2.5, 21),
                "MET_met": np.geomspace(150, 1000, 21),
                "DeltaPhi_tau_met": np.linspace(0, 3.5, 21),
                "TauPt_div_MET": np.linspace(0, 3, 61),
                "TauRNNJetScore": np.linspace(0, 1, 51),
                "TauBDTEleScore": np.linspace(0, 1, 51),
                "TruthTauPt": np.geomspace(1, 1000, 21),
                "TauNCoreTracks": np.linspace(0, 4, 5),
                "TauPt_res": np.linspace(-1, 1, 51),
                "TauPt_diff": np.linspace(-300, 300, 51),
                "badJet": (2, 0, 2),
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
    base_plotting_dir = analysis.paths.plot_dir
    mc_samples = list(datasets.keys())

    # START OF WP LOOP
    # ========================================================================
    # loop over each working point
    for wp in ("loose", "medium", "tight"):
        wp_dir = base_plotting_dir / wp

        # PLOT TRUTH (for mental health) (no selection)
        # ========================================================================
        analysis.paths.plot_dir = wp_dir / "truth"
        default_args = {
            "dataset": mc_samples,
            "title": f"Truth Taus | mc16d | {analysis.global_lumi / 1000:.3g}" + r"fb$^{-1}$",
            "do_stat": True,
            "do_syst": False,
            "selection": "",
            "ratio_plot": False,
            "stats_box": False,
            "label_params": {"llabel": "Simulation"},
        }
        analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
        analysis.plot(val="TruthTauPt", **default_args, logx=True)
        analysis.plot(val="TruthTauEta", **default_args)
        analysis.plot(val="TruthTauPhi", **default_args)

        default_args["selection"] = f"trueTau_{wp}_SR_passID"
        analysis.plot(val="MatchedTruthParticlePt", **default_args, logx=True)
        analysis.plot(val="TruthTauPt", **default_args, logx=True)
        analysis.plot(val="TruthTauEta", **default_args)
        analysis.plot(val="TruthTauPhi", **default_args)

        # tau pt resolution
        for selection in ("", f"{wp}_SR_passID", f"trueTau_{wp}_SR_passID"):
            analysis.plot(
                val="TauPt_res",
                dataset="wtaunu",
                xlabel=r"$(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
                selection=selection,
                title=(
                    r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                    + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
                ),
                filename=f"wtaunu_taupt_{selection}_resolution.png",
            )
            analysis.plot(
                val="TauPt_diff",
                dataset="wtaunu",
                xlabel=r"$p_T^\mathrm{true} - p_T^\mathrm{reco}$ [GeV]",
                title=(
                    r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                    + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
                ),
                filename=f"wtaunu_taupt_{selection}_truthrecodiff.png",
            )
            analysis.plot(
                val="MTW_TauPt_res",
                dataset="wtaunu",
                selection=f"{wp}_SR_passID",
                ylabel=r"Mean $(p_T^\mathrm{true} - p_T^\mathrm{reco}) / p_T^\mathrm{true}$",
                xlabel=r"$m_W^T$ [GeV]",
                title=(
                    r"Tau $p_T$ resolution in $W\rightarrow\tau\nu$ | mc16d | "
                    + f"{analysis.global_lumi / 1000:.3g}fb$^{{-1}}$"
                ),
                y_axlim=(-10, 10),
                filename=f"wtaunu_mtw_taupt_{selection}_profile.png",
            )

        # Fakes distribution across kinematic variable for signal MC
        # -----------------------------------------------------------------------
        analysis.paths.plot_dir = wp_dir / "fakes_distributions"
        for var in measurement_vars:
            xlabel = get_axis_labels(var)[0]
            sel = f"{wp}_SR_passID"
            mc = "wtaunu"

            # for all MC
            wtaunu_el_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isElectron",
                        dataset=d,
                        selection=sel,
                        systematic=NOMINAL_NAME,
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isElectron_{sel}_PROFILE",
            )
            wtaunu_mu_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isMuon",
                        dataset=d,
                        selection=sel,
                        systematic=NOMINAL_NAME,
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isMuon_{sel}_PROFILE",
            )
            wtaunu_ph_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isPhoton",
                        dataset=d,
                        selection=sel,
                        systematic=NOMINAL_NAME,
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isPhoton_{sel}_PROFILE",
            )
            wtaunu_jet_fakes = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isJet",
                        dataset=d,
                        selection=sel,
                        systematic=NOMINAL_NAME,
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isJet_{sel}_PROFILE",
            )
            wtaunu_true_taus = analysis.sum_hists(
                [
                    analysis.get_hist(
                        f"{var}_MatchedTruthParticle_isTau",
                        dataset=d,
                        selection=sel,
                        systematic=NOMINAL_NAME,
                    )
                    for d in mc_samples
                ],
                f"{wp}_all_mc_{var}_MatchedTruthParticle_isTau_{sel}_PROFILE",
            )
            analysis.plot(
                [
                    wtaunu_jet_fakes,
                    wtaunu_ph_fakes,
                    wtaunu_mu_fakes,
                    wtaunu_el_fakes,
                    wtaunu_true_taus,
                ],
                label=[
                    "Jet-matched fake taus",
                    "Photon-matched fake taus",
                    "Muon-matched fake taus",
                    "electron-matched fake taus",
                    "True taus",
                ],
                systematic=NOMINAL_NAME,
                sort=False,
                do_stat=False,
                colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
                title=f"Fake fractions for {var} in {sel} for all MC in SR",
                y_axlim=(0, 1),
                kind="stack",
                xlabel=xlabel,
                ylabel="Fraction of fake matched taus in signal MC",
                filename=f"{wp}_all_mc_{var}_{sel}_fake_fractions.png",
            )

            # for all MC
            args = dict(dataset=mc, selection=sel, systematic=NOMINAL_NAME)
            el_fakes = analysis.get_hist(f"{var}_MatchedTruthParticle_isElectron", **args)
            mu_fakes = analysis.get_hist(f"{var}_MatchedTruthParticle_isMuon", **args)
            ph_fakes = analysis.get_hist(f"{var}_MatchedTruthParticle_isPhoton", **args)
            jet_fakes = analysis.get_hist(f"{var}_MatchedTruthParticle_isJet", **args)
            true_taus = analysis.get_hist(f"{var}_MatchedTruthParticle_isTau", **args)

            sel_hist = analysis.get_hist(
                var, "wtaunu", selection=sel, systematic=NOMINAL_NAME, TH1=True
            )
            nbins = sel_hist.GetNbinsX()

            analysis.plot(
                [jet_fakes, ph_fakes, mu_fakes, el_fakes, true_taus],
                label=[
                    "Jet Fakes",
                    "Photon Fakes",
                    "Muon Fakes",
                    "Electron Fakes",
                    "True taus",
                ],
                sort=False,
                do_stat=False,
                systematic=NOMINAL_NAME,
                colour=list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:5],
                title=f"Fake fractions for {var} in {mc} for SR",
                y_axlim=(0, 1),
                kind="stack",
                xlabel=xlabel,
                label_params={"llabel": "Simulation"},
                ylabel="Fraction of fake matched taus in signal MC",
                filename=f"{wp}_{mc}_{var}_{sel}_fake_fractions.png",
            )
