from src.analysis import Analysis

if __name__ == "__main__":
    datasets = {
        # NEGATIVE
        "wminmunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wminmunu_*/*.root",
            "cutfile": "../options/jesal_cutflow/DY_HM.txt",
            "lepton": "muon",
            "label": r"$W^-\rightarrow\mu\nu$",
        },
        "wmintaunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu_*/*.root",
            "cutfile": "../options/jesal_cutflow/DY_HM.txt",
            "lepton": "tau",
            "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wminmunu": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wminmunu/*.root",
            "cutfile": "../options/jesal_cutflow/DY_peak.txt",
            "lepton": "muon",
            "label": r"$W^-\rightarrow\mu\nu$",
        },
        "wmintaunu": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu/*.root",
            "cutfile": "../options/jesal_cutflow/DY_peak.txt",
            "lepton": "tau",
            "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        # POSITIVE
        "wplusmunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu_*/*.root",
            "cutfile": "../options/jesal_cutflow/DY_HM.txt",
            "lepton": "muon",
            "label": r"$W^+\rightarrow\mu\nu$",
        },
        "wplustaunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu_*/*.root",
            "cutfile": "../options/jesal_cutflow/DY_HM.txt",
            "lepton": "tau",
            "label": r"$W^+\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wplusmunu": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu/*.root",
            "cutfile": "../options/jesal_cutflow/DY_peak.txt",
            "lepton": "muon",
            "label": r"$W^+\rightarrow\mu\nu$",
        },
        "wplustaunu": {
            "data_path": "/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu/*.root",
            "cutfile": "../options/jesal_cutflow/DY_peak.txt",
            "lepton": "tau",
            "label": r"$W^+\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
    }

    analysis = Analysis(
        datasets,
        "mutau_compare",
        log_level=10,
        log_out="both",
        timedatelog=False,
        year="2015+2016",
        force_rebuild=False,
        TTree_name="nominal_Loose",
    )

    analysis.merge_datasets("wminmunu", "wminmunu_hm", verify=True)
    analysis.merge_datasets("wmintaunu", "wmintaunu_hm", verify=True)
    analysis.merge_datasets("wplusmunu", "wplusmunu_hm", verify=True)
    analysis.merge_datasets("wplustaunu", "wplustaunu_hm", verify=True)

    # normalised
    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "met_met",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "met_met",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_pt",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_pt",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_d0sig",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, -3.5, 3.5),
        logy=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_d0sig",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, -3.5, 3.5),
        logy=True,
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_delta_z0_sintheta",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, -1, 1),
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_delta_z0_sintheta",
        weight="reco_weight",
        title="reco 139fb$^{-1}$",
        bins=(30, -1, 1),
        lepton="muon",
        normalise="lumi",
        yerr="rsumw2",
    )

    # un-normalised
    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "met_met",
        weight="reco_weight",
        title="reco",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "met_met",
        weight="reco_weight",
        title="reco",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_pt",
        weight="reco_weight",
        title="reco",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_pt",
        weight="reco_weight",
        title="reco",
        bins=(30, 150, 5000),
        logbins=True,
        logx=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
        scale_by_bin_width=True,
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_d0sig",
        weight="reco_weight",
        title="reco",
        bins=(30, -3.5, 3.5),
        logy=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_d0sig",
        weight="reco_weight",
        title="reco",
        bins=(30, -3.5, 3.5),
        logy=True,
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mu_delta_z0_sintheta",
        weight="reco_weight",
        title="reco",
        bins=(30, -1, 1),
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mu_delta_z0_sintheta",
        weight="reco_weight",
        title="reco",
        bins=(30, -1, 1),
        normalise=False,
        lepton="muon",
        yerr="rsumw2",
    )
