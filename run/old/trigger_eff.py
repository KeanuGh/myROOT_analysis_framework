from typing import Dict

import matplotlib.pyplot as plt  # type: ignore

from histogram import Histogram1D  # type: ignore
from src.analysis import Analysis

# DTA_PATH = '/data/keanu/ditau_output/'
# ANALYSISTOP_PATH = '/data/atlas/HighMassDrellYan/mc16a'
# DATA_OUT_DIR = '/data/keanu/framework_outputs/'
DTA_PATH = "/data/DTA_outputs/2023-06-29/"
DTA_PATH_H = "/data/DTA_outputs/2023-07-17/user.kghorban.Sh_2211_Wtaunu_H*/*.root"
ANALYSISTOP_PATH = "/mnt/D/data/analysistop_out/mc16a/"
DATA_OUT_DIR = "/mnt/D/data/dataset_pkl_outputs/"

if __name__ == "__main__":
    datasets: Dict[str, Dict] = {
        "wtaunu_mu_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tmv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_mu.txt",
            # "cutfile": "../options/DTA_cuts/passtrigger.txt",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
            "hard_cut": """
                (MTW > 150)
                && (isnan(MuonPt) || MuonPt > 20)
                && (isnan(MuonEta) || (abs(MuonEta) < 1.37 || 1.52 < abs(MuonEta) < 2.47))
                && (TauPt > 170)
            """,
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "TTree_name": "T_s1tev_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_e.txt",
            # "cutfile": "../options/DTA_cuts/passtrigger.txt",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
            "hard_cut": """
                (MTW > 150)
                && (isnan(ElePt) || ElePt > 20)
                && (isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))
                && (TauPt > 170)
            """,
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH_H,
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had.txt",
            # "cutfile": "../options/DTA_cuts/passtrigger.txt",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
            "hard_cut": """
                (MTW > 150)
                && (isnan(ElePt) || ElePt > 20)
                && (isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))
                && (isnan(MuonPt) || MuonPt > 20)
                && (isnan(MuonEta) || (abs(MuonEta) < 1.37 || 1.52 < abs(MuonEta) < 2.47))
                && (TauPt > 170)
            """,
        },
        "wtaunu_h_mettrig_dta": {
            "data_path": DTA_PATH_H,
            "TTree_name": "T_s1thv_NOMINAL",
            "cutfile": "../options/DTA_cuts/dta_full_had_mettrig.txt",
            # "cutfile": "../options/DTA_cuts/passmettrigger.txt",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$ MET triggers",
            "hard_cut": """
                (MTW > 150)
                && (isnan(ElePt) || ElePt > 20)
                && (isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))
                && (isnan(MuonPt) || MuonPt > 20)
                && (isnan(MuonEta) || (abs(MuonEta) < 1.37 || 1.52 < abs(MuonEta) < 2.47))
                && (TauPt > 170)
            """,
        },
    }

    # truth_cuts = """
    #     (TruthBosonM > 150)
    #     && (isnan(TruthElePt) || TruthElePt > 20)
    #     && (isnan(TruthEleEta) || (abs(TruthEleEta) < 1.37 || 1.52 < abs(TruthEleEta) < 2.47))
    #     && (isnan(TruthMuonPt) || TruthMuonPt > 20)
    #     && (isnan(TruthMuonEta) || (abs(TruthMuonEta) < 1.37 || 1.52 < abs(TruthMuonEta) < 2.47))
    #     && (TruthTauPt > 170)
    # """
    reco_cuts = """
        (MTW > 150)
        && (isnan(ElePt) || ElePt > 20)
        && (isnan(EleEta) || (abs(EleEta) < 1.37 || 1.52 < abs(EleEta) < 2.47))
        && (isnan(MuonPt) || MuonPt > 20)
        && (isnan(MuonEta) || (abs(MuonEta) < 1.37 || 1.52 < abs(MuonEta) < 2.47))
        && (TauPt > 170)
    """
    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        regen_histograms=True,
        analysis_label="trigger_efficiencies",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
        # hard_cut=truth_cuts,
    )
    my_analysis.cutflow_printout(latex=True)

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

    # phi
    # eta
    my_analysis.plot_hist(
        ["wtaunu_mu_dta", "wtaunu_mu_dta"],
        ["MuonPhi", "cut_MuonPhi"],
        labels=[r"Muon $\phi$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mu\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_e_dta", "wtaunu_e_dta"],
        ["ElePhi", "cut_ElePhi"],
        labels=[r"Electron $\phi$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow e\nu$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_dta", "wtaunu_h_dta"],
        ["TauPhi", "cut_TauPhi"],
        labels=[r"Tau $\phi$ reco", "pass trigger"],
        title=r"$W\rightarrow\tau\nu\rightarrow\mathrm{Hadron}$",
        weight="reco_weight",
        **ratio_args,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_mettrig_dta", "wtaunu_h_mettrig_dta"],
        ["TauPhi", "cut_TauPhi"],
        labels=[r"Tau $\phi$ reco", "pass trigger"],
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

    # Manual efficiency comparisons
    h_taupt_eff = my_analysis.histograms["wtaunu_h_dta_TauPt"].Clone()
    h_taupt_eff.Divide(
        my_analysis.histograms["wtaunu_h_dta_cut_TauPt"],
        my_analysis.histograms["wtaunu_h_dta_TauPt"],
        1,
        1,
        "b",
    )
    h_taupt_eff = Histogram1D(th1=h_taupt_eff)

    # error calculation
    for i in range(h_taupt_eff.n_bins):
        b2 = my_analysis.histograms["wtaunu_h_dta_TauPt"].GetBinContent(i + 1)
        b2sq = b2 * b2
        e2sq = my_analysis.histograms["wtaunu_h_dta_TauPt"].GetBinError(i + 1) ** 2
        b1 = my_analysis.histograms["wtaunu_h_dta_cut_TauPt"].GetBinContent(i + 1)
        b1sq = b1 * b1
        e1sq = my_analysis.histograms["wtaunu_h_dta_cut_TauPt"].GetBinError(i + 1) ** 2
        # err = (1 / N) * np.sqrt(k * (1 - k / N))
        err = abs(((1.0 - 2.0 * b1 / b2) * e1sq + b1sq * e2sq / b2sq) / b2sq)
        print(f"calculated error: {err}")
        e = h_taupt_eff.get_error(i)
        print(f"histogram error: {e}")
        h_taupt_eff.TH1.SetBinError(i + 1, err)

    h_taupt_mettrig_eff = my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"].Clone()
    h_taupt_mettrig_eff.Divide(
        my_analysis.histograms["wtaunu_h_mettrig_dta_cut_TauPt"],
        my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"],
        1,
        1,
        "b",
    )
    h_taupt_mettrig_eff = Histogram1D(th1=h_taupt_mettrig_eff)

    # error calculation
    for i in range(h_taupt_eff.n_bins):
        b2 = my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"].GetBinContent(i + 1)
        b2sq = b2 * b2
        e2sq = my_analysis.histograms["wtaunu_h_mettrig_dta_TauPt"].GetBinError(i + 1) ** 2
        b1 = my_analysis.histograms["wtaunu_h_mettrig_dta_cut_TauPt"].GetBinContent(i + 1)
        b1sq = b1 * b1
        e1sq = my_analysis.histograms["wtaunu_h_mettrig_dta_cut_TauPt"].GetBinError(i + 1) ** 2
        # err = (1 / N) * np.sqrt(k * (1 - k / N))
        err = abs(((1.0 - 2.0 * b1 / b2) * e1sq + b1sq * e2sq / b2sq) / b2sq)
        print(f"calculated error: {err}")
        e = h_taupt_eff.get_error(i)
        print(f"histogram error: {e}")
        h_taupt_mettrig_eff.TH1.SetBinError(i + 1, err)

    fig, (ax, diff_ax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    ax.errorbar(
        h_taupt_mettrig_eff.bin_centres,
        h_taupt_mettrig_eff.bin_values(),
        xerr=h_taupt_mettrig_eff.bin_widths / 2,
        yerr=h_taupt_mettrig_eff.error(),
        linestyle="None",
        label="Tau + MET trigger efficiency",
    )
    ax.errorbar(
        h_taupt_eff.bin_centres,
        h_taupt_eff.bin_values(),
        xerr=h_taupt_eff.bin_widths / 2,
        yerr=h_taupt_eff.error(),
        linestyle="None",
        label="Tau trigger efficiency",
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(170, 5000)
    ax.grid(True, "both", "y")
    ax.set_xlabel("Tau $p_T$ [GeV]")
    ax.set_ylabel("Efficiency")
    ax.semilogx()
    ax.legend()

    # diff
    test_bin = h_taupt_mettrig_eff.TH1.GetBinContent(3) - h_taupt_eff.TH1.GetBinContent(3)
    print(f"{test_bin=}")

    h_diff = h_taupt_eff.TH1.Clone()
    h_diff.Add(h_taupt_mettrig_eff.TH1, h_taupt_eff.TH1, 1, -1)

    h_diff = Histogram1D(th1=h_diff)
    diff_ax.errorbar(
        h_diff.bin_centres,
        h_diff.bin_values(),
        xerr=h_diff.bin_widths / 2,
        yerr=h_diff.error(),
        linestyle="None",
    )
    diff_ax.grid(True, "both", "y")
    diff_ax.set_xlabel("Tau $p_T$ [GeV]")
    diff_ax.set_ylabel("Difference")
    diff_ax.semilogx()
    # diff_ax.set_ylim(0, 1)
    diff_ax.set_xlim(170, 5000)
    result_bin = h_diff.TH1.GetBinContent(3)
    print(f"{result_bin=}")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0)
    ax.set_xticklabels([])
    ax.set_xlabel("")

    fig.savefig(my_analysis.paths.plot_dir / "taupt_eff.png", bbox_inches="tight")
    plt.close(fig)

    my_analysis.logger.info("DONE.")
