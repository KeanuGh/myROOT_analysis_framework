from src.analysis import Analysis

if __name__ == "__main__":
    datasets = {
        # NEGATIVE
        "wminmunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wminmunu_*/*.root",
            "cutfile": "../options/joanna_cutflow/DY_HM.txt",
            "lepton": "muon",
            "label": r"$W^-\rightarrow\mu\nu$",
        },
        "wmintaunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wmintaunu_*/*.root",
            "cutfile": "../options/joanna_cutflow/DY_HM.txt",
            "lepton": "tau",
            "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wminmunu": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wminmunu/*.root",
            "cutfile": "../options/joanna_cutflow/DY_peak.txt",
            "lepton": "muon",
            "label": r"$W^-\rightarrow\mu\nu$",
        },
        "wmintaunu": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wmintaunu/*.root",
            "cutfile": "../options/joanna_cutflow/DY_peak.txt",
            "lepton": "tau",
            "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        # POSITIVE
        "wplusmunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wplusmunu_*/*.root",
            "cutfile": "../options/joanna_cutflow/DY_HM.txt",
            "lepton": "muon",
            "label": r"$W^+\rightarrow\mu\nu$",
        },
        "wplustaunu_hm": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wplustaunu_*/*.root",
            "cutfile": "../options/joanna_cutflow/DY_HM.txt",
            "lepton": "tau",
            "label": r"$W^+\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
        "wplusmunu": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wplusmunu/*.root",
            "cutfile": "../options/joanna_cutflow/DY_peak.txt",
            "lepton": "muon",
            "label": r"$W^+\rightarrow\mu\nu$",
        },
        "wplustaunu": {
            "data_path": "/data/atlas/HighMassDrellYan/mc16a/wplustaunu/*.root",
            "cutfile": "../options/joanna_cutflow/DY_peak.txt",
            "lepton": "tau",
            "label": r"$W^+\rightarrow\tau\nu\rightarrow\mu\nu$",
        },
    }

    analysis = Analysis(
        datasets,
        "mutau_compare_full",
        data_dir="/data/keanu/framework_outputs/mutau_compare_full/",
        log_level=10,
        log_out="both",
        timedatelog=True,
        dataset_type="analysistop",
        year="2015+2016",
        force_rebuild=False,
        TTree_name="truth",
        hard_cut="M_W",
    )

    analysis.merge_datasets("wminmunu", "wminmunu_hm")
    analysis.merge_datasets("wmintaunu", "wmintaunu_hm")
    analysis.merge_datasets("wplusmunu", "wplusmunu_hm")
    analysis.merge_datasets("wplustaunu", "wplustaunu_hm")

    # # =========================
    # # ===== TRUTH - UNCUT =====
    # # =========================
    # # normalised
    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mt_born', weight='truth_weight', title='truth - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # # 36.2fb$^{-1}$
    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, -4, 4), lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    analysis.plot_hist(
        ["wminmunu", "wmintaunu"],
        "mt_born",
        weight="truth_weight",
        title="truth - 36.2fb$^{-1}$",
        bins=(30, 50, 5000),
        logbins=True,
        logx=True,
        ratio_fit=True,
        stats_box=True,
        normalise=False,
        lepton="muon",
    )

    analysis.plot_hist(
        ["wplusmunu", "wplustaunu"],
        "mt_born",
        weight="truth_weight",
        title="truth - 36.2fb$^{-1}$",
        bins=(30, 50, 5000),
        logbins=True,
        logx=True,
        ratio_fit=True,
        stats_box=True,
        normalise=False,
        lepton="muon",
    )

    # cut
    analysis.apply_cuts()

    # # =========================
    # # ====== RECO - CUTS ======
    # # =========================
    # # normalised
    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'met_met', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'met_met', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_pt', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_pt', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, 50, 1000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_eta', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_eta', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_d0sig', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_d0sig', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -1, 1), ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - normalised to unity$',
    #                    bins=(30, -1, 1), ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=True)

    # # 36.2fb$^{-1}$
    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'met_met', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'met_met', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_pt', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 5000), logbins=True, logx=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_pt', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, 50, 1000), logbins=True, logx=True, lepton='muon', ratio_fit=True, stats_box=True,
    #                    normalise=False)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_eta', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=False)

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_eta', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    lepton='muon', normalise=False)

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_d0sig', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_d0sig', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -3.5, 3.5), logy=True, ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -1, 1), ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')

    # analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - 36.2fb$^{-1}$',
    #                    bins=(30, -1, 1), ratio_fit=True, stats_box=True,
    #                    normalise=False, lepton='muon')
