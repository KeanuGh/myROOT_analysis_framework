from typing import Dict

import matplotlib.pyplot as plt
from numpy import pi

from src.analysis import Analysis
from utils import plotting_tools

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-05-05/"
ANALYSISTOP_PATH = "/mnt/D/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/mnt/D/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1tmv_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1tev_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
        },
        "wtaunu_CVetoBVeto_dta": {
            "data_path": DTA_PATH + "/*CVetoBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"CVetoBVeto",
        },
        "wtaunu_CFilterBVeto_dta": {
            "data_path": DTA_PATH + "/*CFilterBVeto*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"CFilterBVeto",
        },
        "wtaunu_BFilter_dta": {
            "data_path": DTA_PATH + "/*BFilter*/*.root",
            "cutfile": "../options/DTA_cuts/dta_full.txt",
            "TTree_name": {"T_s1thv_NOMINAL", "T_s1tev_NOMINAL", "T_s1tmv_NOMINAL"},
            # "regen_histograms": False,
            "label": r"BFilter",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # regen_histograms=True,
        analysis_label="dta_internal_comparisons",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
    )
    # my_analysis.merge_datasets('wtaunu_e_dta', 'wtaunu_e_dta_peak')

    BR = len(my_analysis["wtaunu_mu_dta"]) / len(my_analysis["wtaunu_e_dta"])
    my_analysis.logger.info(f"BRANCHING RATIO tau->munu / tau->enu:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_CFilterBVeto_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"CFilterBVeto / CVetoBVeto:  {BR:.5f}")

    BR = len(my_analysis["wtaunu_BFilter_dta"]) / len(my_analysis["wtaunu_CVetoBVeto_dta"])
    my_analysis.logger.info(f"BFilter / CVetoBVeto:  {BR:.5f}")

    # my_analysis.apply_cuts(truth=True)

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    # argument dicts
    lepton_ds = ["wtaunu_mu_dta", "wtaunu_e_dta", "wtaunu_h_dta"]
    flavour_ds = ["wtaunu_BFilter_dta", "wtaunu_CFilterBVeto_dta", "wtaunu_CVetoBVeto_dta"]

    ratio_args = {
        # "ratio_axlim": 1.5,
        "stats_box": False,
        "ratio_fit": True,
    }
    truth_mass_args = {
        "bins": (30, 1, 5000),
        "logbins": True,
        "logx": True,
        # "ratio_axlim": 1.5,
    }
    truth_weighted_args = {
        "weight": "truth_weight",
        "prefix": "truth_inclusive",
        "title": "truth - 36.2fb$^{-1}$",
        "normalise": False,
    }

    my_analysis.plot_hist(
        ["wtaunu_mu_dta", "wtaunu_mu_dta"],
        ["MuonPt", "cut_MuonPt"],
        bins=(30, 1, 5000),
        logx=True,
        logy=True,
        labels=["Muon Pt reco", "pass trigger"],
        weight="reco_weight",
        **ratio_args,
    )

    # TRUTH
    # -----------------------------------
    # leptons
    my_analysis.plot_hist(
        lepton_ds,
        "TruthMTW",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthBosonM",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    # truth taus
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauPt",
        **truth_mass_args,
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauEta",
        bins=(30, -5, 5),
        **truth_weighted_args,
        **ratio_args,
    )
    my_analysis.plot_hist(
        lepton_ds,
        "TruthTauPhi",
        bins=(30, -pi, pi),
        logy=False,
        **truth_weighted_args,
        **ratio_args,
    )

    # # incoming quark flavour
    my_analysis.plot_hist(
        flavour_ds,
        "TruthTauPt",
        bins=(30, 1, 5000),
        weight="base_weight",
        ratio_axlim=1.5,
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TruthTauEta",
        bins=(30, -5, 5),
        weight="base_weight",
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TruthTauPhi",
        bins=(30, -pi, pi),
        weight="base_weight",
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TruthMTW",
        bins=(30, 1, 5000),
        weight="base_weight",
        ratio_axlim=1.5,
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TruthBosonM",
        bins=(30, 1, 5000),
        weight="base_weight",
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TruthMTW",
        bins=(30, 1, 5000),
        weight="base_weight",
        ratio_axlim=1.5,
        title="truth - 36.2fb$^{-1}$",
        normalise=False,
        logx=True,
        logbins=True,
        **ratio_args,
    )

    # MET/neutrinos
    met_vars = ["TruthNeutrinoPt", "ImplicitMetPt", "TruthMetPt"]
    for dataset in lepton_ds:
        fig, (ax, ratio_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        hists = []
        colours = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for i, var in enumerate(met_vars):
            hist = my_analysis[dataset].plot_hist(var=var, ax=ax, label=var)
            hists.append(hist)
            colour = next(colours)

            ax.legend(fontsize=10, loc="upper right")
            plotting_tools.set_axis_options(ax, var, (30, 1, 5000), logx=True, logy=True)

            if len(hists) > 1:
                ratio_hist = hists[0].plot_ratio(
                    hists[-1],
                    ax=ratio_ax,
                    yerr=True,
                    label=f"{met_vars[0]}/{var}",
                    display_stats=True,
                    color=colour,
                )

            fig.tight_layout()
            fig.subplots_adjust(hspace=0.1, wspace=0)
            ax.set_xticklabels([])
            ax.set_xlabel("")

            ratio_ax.legend(fontsize=10, loc=1)
            plotting_tools.set_axis_options(
                axis=ratio_ax,
                var_name=var,
                bins=(30, 1, 5000),
                ylabel="Ratio",
                xlabel="Missing $E_T$",
                title="",
                logx=True,
                logy=False,
                label=False,
            )
            filename = my_analysis.paths.plot_dir / (dataset + "_" + "_".join(met_vars) + ".png")

            fig.savefig(filename, bbox_inches="tight")
            my_analysis.logger.info(f"Saved overlay plot of {var} to {filename}")
            plt.close(fig)

    # RECO
    # -----------------------------------
    my_analysis.plot_hist(
        lepton_ds,
        "MuonPt",
        bins=(30, 1, 5000),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MuonEta",
        bins=(30, -5, 5),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MuonPhi",
        bins=(30, -pi, pi),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        lepton_ds,
        "ElePt",
        bins=(30, 1, 5000),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "EleEta",
        bins=(30, -5, 5),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "ElePhi",
        bins=(30, -pi, pi),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        lepton_ds,
        "TauPt",
        bins=(30, 1, 5000),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TauEta",
        bins=(30, -5, 5),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "TauPhi",
        bins=(30, -pi, pi),
        weight="lep_reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MET_met",
        bins=(30, 1, 5000),
        weight="reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        logx=True,
        logbins=True,
        **ratio_args,
    )

    my_analysis.plot_hist(
        flavour_ds,
        "MET_phi",
        bins=(30, -pi, pi),
        weight="reco_weight",
        normalise=False,
        title="reco - 36.2fb$^{-1}$",
        **ratio_args,
    )

    my_analysis.histogram_printout()

    my_analysis.logger.info("DONE.")
