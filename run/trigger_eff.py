from typing import Dict

from histogram import Histogram1D
from src.analysis import Analysis

# import matplotlib.pyplot as plt

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-06-29/"
ANALYSISTOP_PATH = "/mnt/D/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/mnt/D/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tmv_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tev_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
        },
        "wtaunu_h_mettrig_dta": {
            "data_path": DTA_PATH + "/user.kghorban.MET_TRIG.Sh_2211_Wtaunu_H*/*.root",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$ MET triggers",
        },
    }

    truth_cuts = """ passTruth
        && TruthBosonM > 130
        && (isnan(TruthElePt)
        && (isnan(TruthEleEta) || (TruthEleEta < 1.37 || 1.52 < TruthEleEta < 2.47))
        && (isnan(TruthMuonPt) || TruthMuonPt > 20)
        && (isnan(TruthMuonEta) || (TruthMuonEta < 1.37 || 1.52 < TruthMuonEta < 2.47))
        && TruthTauPt > 80
    """
    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        regen_histograms=True,
        analysis_label="trigger_efficiencies",
        cutfile="../options/DTA_cuts/dta_full.txt",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="console",
        # hard_cut=truth_cuts,
    )

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    ratio_args = {
        "ratio_axlim": (0, 1.1),
        "stats_box": False,
        "ratio_fit": True,
        "ratio_label": "Efficiency",
        "ratio_err": "binom",
    }

    # PT
    my_analysis.plot_hist(
        ["wtaunu_e_dta", "wtaunu_e_dta"],
        ["ElePt", "cut_ElePt"],
        logx=True,
        logy=True,
        labels=[r"Electron $p_T$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow e\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_mu_dta", "wtaunu_mu_dta"],
        ["MuonPt", "cut_MuonPt"],
        logx=True,
        logy=True,
        labels=[r"Muon $p_T$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mu\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_dta", "wtaunu_h_dta"],
        ["TauPt", "cut_TauPt"],
        logx=True,
        logy=True,
        labels=[r"Tau $p_T$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadrons}$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_mettrig_dta"],
        ["TauPt", "cut_TauPt"],
        logx=True,
        logy=True,
        labels=[r"Tau $p_T$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadron}$ (MET triggers)",
        weight="reco_weight",
        **ratio_args,
    )

    # eta
    my_analysis.plot_hist(
        ["wtaunu_mu_dta", "wtaunu_mu_dta"],
        ["MuonEta", "cut_MuonEta"],
        labels=[r"Muon $\eta$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mu\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_e_dta", "wtaunu_e_dta"],
        ["EleEta", "cut_EleEta"],
        labels=[r"Electron $\eta$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow e\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_dta", "wtaunu_h_dta"],
        ["TauEta", "cut_TauEta"],
        labels=[r"Tau $\eta$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadron}$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_mettrig_dta"],
        ["TauEta", "cut_TauEta"],
        labels=[r"Tau $\eta$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadron}$ (MET triggers)",
        weight="reco_weight",
        **ratio_args,
    )

    # MTW
    my_analysis.plot_hist(
        ["wtaunu_mu_dta", "wtaunu_mu_dta"],
        ["MTW", "cut_MTW"],
        labels=[r"Muon $m_T^W$ reco", "pass trigger"],
        weight="reco_weight",
        title=r"$W\rightarrow\tau\nu\rightarrow\mu\nu$",
        **ratio_args,
        logx=True,
        logy=True,
    )
    my_analysis.plot_hist(
        ["wtaunu_e_dta", "wtaunu_e_dta"],
        ["MTW", "cut_MTW"],
        labels=[r"Electron $m_T^W$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow e\nu$",
        weight="reco_weight",
        **ratio_args,
        logx=True,
        logy=True,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_dta", "wtaunu_h_dta"],
        ["MTW", "cut_MTW"],
        labels=[r"Tau $m_T^W$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadrons}$",
        weight="reco_weight",
        **ratio_args,
        logx=True,
        logy=True,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_mettrig_dta"],
        ["MTW", "cut_MTW"],
        labels=[r"Tau $m_T^W$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadrons}$ (MET triggers)",
        weight="reco_weight",
        **ratio_args,
        logx=True,
        logy=True,
    )

    # MET trigger comparisons
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_dta"],
        [
            "wtaunu_h_mettrig_dta_wtaunu_h_mettrig_dta_cut_TauPt_ratio",
            "wtaunu_h_dta_wtaunu_h_dta_cut_TauPt_ratio",
        ],
        labels=["Tau + MET trig. eff.", "Tau trig. eff."],
        xlabel="Tau $p_T$ [GeV]",
        ylabel="Efficiency",
        stats_box=False,
        ratio_fit=True,
        logy=False,
        logx=True,
        ratio_axlim=(0, 1.1),
        ratio_err="binom",
        y_axlim=(0, 1.1),
        gridopts=(True, "both", "y"),
    )
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_dta"],
        [
            "wtaunu_h_mettrig_dta_wtaunu_h_mettrig_dta_cut_MTW_ratio",
            "wtaunu_h_dta_wtaunu_h_dta_cut_MTW_ratio",
        ],
        labels=["Tau + MET trig. eff.", "Tau trig. eff."],
        xlabel="$m_T^W$ [GeV]",
        ylabel="Efficiency",
        stats_box=False,
        ratio_fit=True,
        logy=False,
        logx=True,
        ratio_axlim=(0, 1.1),
        ratio_err="binom",
        y_axlim=(0, 1.1),
        gridopts=(True, "both", "y"),
    )

    my_analysis.histogram_printout()
    my_analysis.save_histograms()

    Histogram1D()

    # h_taupt_eff = my_analysis.histograms["wtaunu_h_dta_TauPt"]
    # h_taupt_eff.Divide(
    #     my_analysis.histograms["wtaunu_h_dta_cut_TauPt"],
    #     my_analysis.histograms["wtaunu_h_dta_TauPt"],
    #     1,
    #     1,
    #     "B",
    # )
    # h_taupt_eff = Histogram1D(th1=h_taupt_eff)
    #
    # h_taupt_mettrig_eff = my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"]
    # h_taupt_mettrig_eff.Divide(
    #     my_analysis.histograms["wtaunu_h_mettrig_dta_cut_TauPt"],
    #     my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"],
    #     1,
    #     1,
    #     "B",
    # )
    # h_taupt_mettrig_eff = Histogram1D(th1=h_taupt_mettrig_eff)
    #
    # fig, ax = plt.subplots()
    # ax.errorbar(
    #     h_taupt_mettrig_eff.bin_centres,
    #     h_taupt_mettrig_eff.bin_values(),
    #     xerr=h_taupt_mettrig_eff.bin_widths / 2,
    #     yerr=h_taupt_mettrig_eff.error(),
    #     linestyle="None",
    #     label="Tau + MET trigger efficiency",
    # )
    # ax.errorbar(
    #     h_taupt_eff.bin_centres,
    #     h_taupt_eff.bin_values(),
    #     xerr=h_taupt_eff.bin_widths / 2,
    #     yerr=h_taupt_eff.error(),
    #     linestyle="None",
    #     label="Tau trigger efficiency",
    # )
    # ax.set_ylim(0, 1)
    # ax.grid(True, "both", "y")
    # ax.set_xlabel("Tau $p_T$ [GeV]")
    # ax.set_ylabel("Efficiency")
    # ax.semilogx()
    # ax.legend()
    # fig.savefig(my_analysis.paths.plot_dir / "taupt_eff.png", bbox_inches="tight")
    # plt.close(fig)
    #
    # my_analysis.logger.info("DONE.")
