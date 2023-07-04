from typing import Dict

from src.analysis import Analysis

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
            "cutfile": "../options/DTA_cuts/passtrigger.txt",
            "TTree_name": "T_s1tmv_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mu\nu$",
        },
        "wtaunu_e_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_L*/*.root",
            "cutfile": "../options/DTA_cuts/passtrigger.txt",
            "TTree_name": "T_s1tev_NOMINAL",
            # "regen_histograms": False,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow e\nu$",
        },
        "wtaunu_h_dta": {
            "data_path": DTA_PATH + "/user.kghorban.Sh_2211_Wtaunu_H*/*.root",
            "cutfile": "../options/DTA_cuts/passtrigger.txt",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$",
        },
        "wtaunu_h_mettrig_dta": {
            "data_path": DTA_PATH + "/user.kghorban.MET_TRIG.Sh_2211_Wtaunu_H*/*.root",
            "cutfile": "../options/DTA_cuts/passtrigger.txt",
            "TTree_name": "T_s1thv_NOMINAL",
            # "regen_histograms": True,
            "label": r"Sherpa 2211 $W\rightarrow\tau\nu\rightarrow \mathrm{had}\nu$ MET triggers",
        },
    }

    my_analysis = Analysis(
        datasets,
        data_dir=DATA_OUT_DIR,
        year="2015+2016",
        # regen_histograms=True,
        analysis_label="trigger_efficiencies",
        lepton="tau",
        dataset_type="dta",
        # log_level=10,
        log_out="both",
        hard_cut="MTW > 130",
    )

    # HISTORGRAMS
    # ==================================================================================================================
    # TRUTH
    # -----------------------------------
    ratio_args = {
        # "ratio_axlim": 1.5,
        "stats_box": False,
        "ratio_fit": True,
        "ratio_label": "Efficiency",
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
        ["wtaunu_h_dta", "wtaunu_h_mettrig_dta"],
        [
            "wtaunu_h_dta_wtaunu_h_dta_cut_TauPt_ratio",
            "wtaunu_h_mettrig_dta_wtaunu_h_mettrig_dta_cut_TauPt_ratio",
        ],
        labels=["Tau trig. eff.", "Tau + MET trig. eff."],
        xlabel="Tau $p_T$ [GeV]",
        ylabel="Efficiency",
        stats_box=False,
        ratio_fit=True,
        logy=False,
        logx=True,
        ratio_axlim=1.5,
    )
    my_analysis.plot_hist(
        ["wtaunu_h_dta", "wtaunu_h_mettrig_dta"],
        [
            "wtaunu_h_dta_wtaunu_h_dta_cut_MTW_ratio",
            "wtaunu_h_mettrig_dta_wtaunu_h_mettrig_dta_cut_MTW_ratio",
        ],
        labels=["Tau trig. eff.", "Tau + MET trig. eff."],
        xlabel="$m_T^W$ [GeV]",
        ylabel="Efficiency",
        stats_box=False,
        ratio_fit=True,
        logy=False,
        logx=True,
        ratio_axlim=1.5,
    )

    my_analysis.histogram_printout()

    my_analysis.logger.info("DONE.")
