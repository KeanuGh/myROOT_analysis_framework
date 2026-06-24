"""Shared production selections for the 2017 shadow-unfold workflow."""

from shadow_unfold.models import ShadowConfig
from src.cutting import Cut

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
PASS_TRUTH = Cut(r"Pass Truth", r"(passTruth == 1)")
PASS_TRUTH_ETA = Cut(
    r"Truth $|\eta^{\tau_\mathrm{had-vis}}| < 1.37 || "
    r"1.52 < |\eta^{\tau_\mathrm{had-vis}}| < 2.47$",
    r"(((abs(VisTruthTauEta) < 1.37) || (1.52 < abs(VisTruthTauEta))) "
    r"&& (abs(VisTruthTauEta) < 2.47))",
)
TRUTH_HAD_TAU = Cut(
    r"Truth Hadronic Tau",
    r"TruthTau_isHadronic && ((TruthTau_nChargedTracks == 1) || "
    r"(TruthTau_nChargedTracks == 3))",
)


def build_reco_sr_cuts(config: ShadowConfig) -> list[Cut]:
    """Return reconstructed SR cuts for one shadow-bin configuration."""
    return [
        PASS_RECO_PRESELECTION,
        Cut(r"$p_T^\tau$ threshold", f"TauPt > {config.taupt_min:g}"),
        PASS_ETA,
        Cut(r"$m_T^W$ threshold", f"MTW > {config.mtw_min:g}"),
        Cut(r"$E_T^{\mathrm{miss}}$ threshold", f"MET_met > {config.met_min:g}"),
    ]


def build_fiducial_truth_cuts(config: ShadowConfig) -> list[Cut]:
    """Return fiducial truth cuts for one shadow-bin configuration."""
    return [
        PASS_TRUTH,
        Cut(
            r"Pass truth fiducial region",
            f"(VisTruthTauPt > {config.taupt_min:g}) && "
            f"(TruthMTW > {config.mtw_min:g}) && "
            f"(TruthNeutrinoPt > {config.met_min:g})"
            r"&& ((TruthTau_nChargedTracks == 1) || (TruthTau_nChargedTracks == 3))",
        ),
        PASS_TRUTH_ETA,
        TRUTH_HAD_TAU,
    ]


def fiducial_truth_hard_cut(config: ShadowConfig) -> str:
    """Return the fiducial truth phase-space cut as one nominal-mask string."""
    return " && ".join(cut.cutstr for cut in build_fiducial_truth_cuts(config))
