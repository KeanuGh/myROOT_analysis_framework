from src.analysis import Analysis

if __name__ == "__main__":
    datasets = {
        "wminmunu": {
            "data_path": "../tests/resources/test_analysistop_mcwmintaunu.root",
            "cutfile": "../tests/resources/cutfile_EXAMPLE.txt",
            "TTree_name": "truth",
            "lepton": "muon",
            "label": r"$W^-\rightarrow\mu\nu$",
        },
        # "wmintaunu": {
        #     "data_path": "/data/test_mc16a_wmintaunu/*/*.root",
        #     # 'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu_*/*.root',
        #     "cutfile": "../options/cutfile_EXAMPLE.txt",
        #     "TTree_name": "truth",
        #     "hard_cut": r"Muon $|#eta|$",
        #     "lepton": "tau",
        #     "label": r"$W^-\rightarrow\tau\nu\rightarrow\mu\nu$",
        # },
    }

    my_analysis = Analysis(
        datasets,
        analysis_label="test_analysis",
        force_rebuild=True,
        dataset_type="analysistop",
        log_level=10,
        log_out="both",
        timedatelog=False,
        separate_loggers=False,
        year="2015+2016",
    )
    # my_analysis.print_latex_table(["wminmunu", "wmintaunu"])

    # my_analysis.apply_cuts()
    # my_analysis.merge_datasets('wminmunu', 'wmintaunu', apply_cuts=r'Muon $|#eta|$')

    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZmu_el_eta_born",
    #     bins=(30, -5, 5),
    #     weight="truth_weight",
    #     normalise="lumi",
    #     lepton="muon",
    #     ratio_plot=True,
    #     stats_box=True,
    #     ratio_fit=True,
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZ_dilep_m_born",
    #     bins=(50, 120, 5000),
    #     weight="truth_weight",
    #     lepton="muon",
    #     title="test plot",
    #     normalise=True,
    #     ratio_plot=True,
    #     stats_box=True,
    #     ratio_fit=True,
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "mu_pt",
    #     bins=(50, 1, 5000),
    #     weight="reco_weight",
    #     logbins=True,
    #     logx=True,
    #     normalise="lumi",
    #     lepton="muon",
    #     stats_box=True,
    #     ratio_fit=True,
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZmu_el_pt_born",
    #     bins=(50, 1, 5000),
    #     weight="truth_weight",
    #     logbins=True,
    #     logx=True,
    #     normalise="lumi",
    #     lepton="muon",
    #     stats_box=True,
    #     ratio_fit=True,
    # )
    #
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZmu_el_eta_born",
    #     bins=(30, -5, 5),
    #     weight="truth_weight",
    #     normalise=False,
    #     lepton="muon",
    #     ratio_plot=True,
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZ_dilep_m_born",
    #     bins=(50, 120, 5000),
    #     weight="truth_weight",
    #     lepton="muon",
    #     title="test plot",
    #     normalise=False,
    #     ratio_plot=True,
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "mu_pt",
    #     bins=(50, 1, 5000),
    #     weight="reco_weight",
    #     logbins=True,
    #     logx=True,
    #     normalise=False,
    #     lepton="muon",
    # )
    # my_analysis.plot_hist(
    #     ["wminmunu", "wmintaunu"],
    #     "MC_WZmu_el_pt_born",
    #     bins=(50, 1, 5000),
    #     weight="truth_weight",
    #     logbins=True,
    #     logx=True,
    #     normalise=False,
    #     lepton="muon",
    # )
    #
    # my_analysis["wminmunu"].plot_cut_overlays(
    #     "MC_WZ_dilep_m_born",
    #     bins=(50, 120, 5000),
    #     weight="truth_weight",
    #     lepton="muon",
    #     title="test plot",
    #     normalise=True,
    # )

    # my_analysis.plot_mass_slices('wmintaunu', 'mt_born', weight='truth_weight', bins=(50, 200, 5000))
    # my_analysis.plot_mass_slices('wmintaunu', 'MC_WZ_dilep_m_born', weight='truth_weight', bins=(50, 200, 10000), logbins=True, logx=True)
    # my_analysis.plot_mass_slices('wmintaunu', 'mu_pt', weight='reco_weight', bins=(50, 200, 5000), logbins=True, logx=True)
    # my_analysis['wmintaunu'].profile_plot('MC_WZ_dilep_m_born', 'weight_KFactor', c='k', s=0.5, logx=True,
    #                                       xlim=(100, 600), ylim=(0.98, 1.025))
    # my_analysis.make_all_cutgroup_2dplots('wminmunu')
    # my_analysis.plot_with_cuts('wminmunu')
