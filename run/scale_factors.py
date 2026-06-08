from pathlib import Path

import numpy as np

from src.analysis import Analysis
from utils.plotting_tools import ProfileOpts
from utils.variable_names import variable_data

DTA_PATH = Path("/mnt/D/data/DTA_outputs/2024-09-19/")
# DTA_PATH = Path("/eos/home-k/kghorban/DTA_OUT/2024-02-05/")

datasets: dict[str, dict] = {
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

# VARIABLES
# ========================================================================
measurement_vars_mass = [
    "TauPt",
    "MET_met",
    "MTW",
]
measurement_vars_unitless = [
    # "TauEta",
    "TauPhi",
    "MET_phi",
]
measurement_vars = measurement_vars_unitless + measurement_vars_mass
NOMINAL_NAME = "T_s1thv_NOMINAL"
profiles = {
    "mtw_mcWeight": ProfileOpts(x="MTW", y="mcWeight"),
    "mtw_prwWeight": ProfileOpts(x="MTW", y="prwWeight"),
}


def run_analysis() -> Analysis:
    """Run analysis"""

    nedges = 51
    return Analysis(
        datasets,
        year=2017,
        rerun=True,
        regen_histograms=True,
        do_systematics=False,
        # regen_metadata=True,
        ttree=NOMINAL_NAME,
        analysis_label="scale_factors",
        log_level=10,
        log_out="both",
        profiles=profiles,
        extract_vars=measurement_vars,
        import_missing_columns_as_nan=True,
        snapshot=False,
        do_weights=False,
        binnings={
            "": {
                "MTW": np.geomspace(1, 1000, nedges),
                "mcWeight": np.geomspace(150, 1000, nedges),
                "prwWeight": np.geomspace(150, 1000, nedges),
            },
        },
    )


if __name__ == "__main__":
    # RUN
    # ========================================================================
    analysis = run_analysis()

    mtw = analysis["wtaunu"].histograms[NOMINAL_NAME][""]["MTW"]
    mtw_mcweighted = mtw.Clone()
    mtw_mcweighted.Multiply(analysis["wtaunu"].histograms[NOMINAL_NAME][""]["mtw_mcWeight"])

    mtw_prweighted = mtw.Clone()
    mtw_prweighted.Multiply(analysis["wtaunu"].histograms[NOMINAL_NAME][""]["mtw_prwWeight"])

    default_args = dict(
        ylabel="Events",
        xlabel="Reconstructed " + variable_data["MTW"]["name"] + " [GeV]",
        ratio_plot=True,
        ratio_label="weighted/unweighted",
        label_params={"llabel": "Simulation", "loc": 1},
    )
    analysis.plot(
        val=[mtw, mtw_mcweighted],
        label=["Unweighted", "mcWeight weighted"],
        **default_args,
        filename="mc_weighted.png",
    )
    analysis.plot(
        val=[mtw, mtw_mcweighted],
        label=["Unweighted", "mcWeight weighted"],
        **default_args,
        logx=True,
        logy=True,
        filename="mc_weighted_log.png",
    )
    analysis.plot(
        val=[mtw, mtw_prweighted],
        label=["Unweighted", "PRW weighted"],
        **default_args,
        filename="prw_weighted.png",
    )
