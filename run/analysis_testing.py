from pathlib import Path
from typing import Dict

import numpy as np

from src.analysis import Analysis
from src.cutting import Cut
from src.dataset import ProfileOpts

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-08-28/")

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        # SIGNAL
        # ====================================================================
        "wtaunu": {
            "data_path": "/mnt/D/data/DTA_outputs/2024-08-28/user.kghorban.Sh_2211_Wtaunu_H_maxHTpTV2_BFilter.e8351.MC16d.v1.2024-08-28_histograms.root/user.kghorban.40997756._000001.histograms.root",
            "label": r"$W\rightarrow\tau\nu$",
            "is_signal": True,
        },
    }

    # CUTS & SELECTIONS
    # ========================================================================
    pass_presel = Cut(
        r"Pass preselection",
        r"(passReco == 1) && (TauBaselineWP == 1) && (abs(TauCharge) == 1)"
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
    fail_loose = Cut(
        r"\mathrm{Fail Loose ID}",
        "(TauBDTEleScore > 0.05) && "
        "!((TauRNNJetScore > 0.15) * (TauNCoreTracks == 1) + (TauRNNJetScore > 0.25) * (TauNCoreTracks == 3))",
    )
    pass_met150 = Cut(
        r"$E_T^{\mathrm{miss}} > 150$",
        r"MET_met > 150",
    )
    pass_100met = Cut(
        r"$E_T^{\mathrm{miss}} < 100$",
        r"MET_met < 100",
    )
    # this is a multijet background estimation. Tau is "True" in this case if it is a lepton
    pass_truetau = Cut(
        r"True Tau",
        "MatchedTruthParticle_isHadronicTau == true || MatchedTruthParticle_isMuon == true || MatchedTruthParticle_isElectron == true",
    )

    # selections
    selections: dict[str, list[Cut]] = {
        "SR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
        ],
        "SR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
        ],
        "CR_passID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
        ],
        "CR_failID": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
        ],
        # for MC
        "SR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_met150,
            pass_truetau,
        ],
        "SR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_met150,
            pass_truetau,
        ],
        "CR_passID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            pass_loose,
            pass_100met,
            pass_truetau,
        ],
        "CR_failID_trueTau": [
            pass_presel,
            pass_taupt170,
            pass_mtw150,
            fail_loose,
            pass_100met,
            pass_truetau,
        ],
    }

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
        "TruthTau_nChargedTracks",
        "TruthTau_nNeutralTracks",
        "TauPt_res",
        "TauPt_diff",
        "MatchedTruthParticlePt",
        "MatchedTruthParticle_isTau",
        "MatchedTruthParticle_isElectron",
        "MatchedTruthParticle_isMuon",
        "MatchedTruthParticle_isPhoton",
        "MatchedTruthParticle_isJet",
        "nJets",
    }
    measurement_vars_mass = [
        "TauPt",
        "MTW",
        "MET_met",
    ]
    measurement_vars_unitless = [
        "TauEta",
        "TauPhi",
        "nJets",
        "TauNCoreTracks",
        "TauRNNJetScore",
        "TauBDTEleScore",
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

    # RUN
    # ========================================================================
    analysis = Analysis(
        datasets,
        year=2017,
        regen_histograms=True,
        do_systematics=True,
        # regen_metadata=True,
        ttree="T_s1thv_NOMINAL",
        cuts=selections,
        analysis_label="analysis_test",
        dataset_type="dta",
        log_level=10,
        log_out="both",
        extract_vars=wanted_variables,
        import_missing_columns_as_nan=True,
        profiles=profiles,
        binnings={
            "": {
                "MTW": np.geomspace(150, 1000, 21),
                "TauPt": np.geomspace(170, 1000, 21),
                "TauEta": np.linspace(-2.5, 2.5, 21),
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
            "CR_failID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_failID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
            "CR_passID_trueTau": {
                "MET_met": np.geomspace(1, 100, 51),
            },
        },
    )
    all_samples = analysis.mc_samples
    mc_samples = analysis.mc_samples
    analysis["wtaunu"].calculate_systematic_uncertainties()

    # analysis.histogram_printout()
    analysis.logger.info("DONE.")
