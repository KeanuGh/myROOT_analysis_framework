from src.analysis import Analysis

if __name__ == '__main__':

    datasets = {
        # NEGATIVE
        'wminmunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wminmunu_*/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'muon',
            'label': r'$W^-\rightarrow\mu\nu$',
        },
        'wmintaunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wmintaunu_*/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'tau',
            'label': r'$W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wminmunu': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wminmunu/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_peak.txt',
            'lepton': 'muon',
            'label': r'$W^-\rightarrow\mu\nu$',
        },
        'wmintaunu': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wmintaunu/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_peak.txt',
            'lepton': 'tau',
            'label': r'$W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },

        # POSITIVE
        'wplusmunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wplusmunu_*/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'muon',
            'label': r'$W^+\rightarrow\mu\nu$',
        },
        'wplustaunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wplustaunu_*/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_HM.txt',
            'lepton': 'tau',
            'label': r'$W^+\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wplusmunu': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wplusmunu/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_peak.txt',
            'lepton': 'muon',
            'label': r'$W^+\rightarrow\mu\nu$',
        },
        'wplustaunu': {
            'data_path': '/data/atlas/HighMassDrellYan/mc16a/wplustaunu/*.root',
            'cutfile_path': '../options/joanna_cutflow/DY_peak.txt',
            'lepton': 'tau',
            'label': r'$W^+\rightarrow\tau\nu\rightarrow\mu\nu$',
        }
    }

    analysis = Analysis(
        datasets,
        'mutau_compare_full',
        data_dir='/data/keanu/framework_outputs/mutau_compare_full/',
        log_level=10,
        log_out='both',
        timedatelog=True,
        year='2015+2016',
        force_rebuild=False,
        TTree_name='truth'
    )
    logger = analysis.logger

    analysis.merge_datasets("wminmunu",   "wminmunu_hm", verify=True)
    analysis.merge_datasets("wmintaunu",  "wmintaunu_hm", verify=True)
    analysis.merge_datasets("wplusmunu",  "wplusmunu_hm", verify=True)
    analysis.merge_datasets("wplustaunu", "wplustaunu_hm", verify=True)
    
    # some checks
    

    # =========================
    # ======= NORMALISED ======
    # =========================
    # reco plots
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'met_met', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'met_met', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_pt', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_pt', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, 150, 1000), logbins=True, logx=True, lepton='muon', 
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_d0sig', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, -3.5, 3.5), logy=True,
                       lepton='muon', normalise='lumi', yerr='rsumw2')
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_d0sig', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, -3.5, 3.5), logy=True,
                       lepton='muon', normalise='lumi', yerr='rsumw2')
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, -1, 1),
                       lepton='muon', normalise='lumi', yerr='rsumw2')
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - 139fb$^{-1}$',
                       bins=(30, -1, 1),
                       lepton='muon', normalise='lumi', yerr='rsumw2')

   # truth plots
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon', apply_cuts='M_W',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wminmunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)
    
    analysis.plot_hist(['wminmunu', 'wminmunu'], 'mt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'mt_born', weight='truth_weight', title='truth - 139fb$^{-1}$',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       lepton='muon', normalise='lumi', yerr='rsumw2', scale_by_bin_width=True)

    # =========================
    # ===== UN-NORMALISED =====
    # =========================
    # reco plots
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_mt_reco', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'met_met', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'met_met', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_pt', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True,
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_pt', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, 150, 1000), logbins=True, logx=True, lepton='muon',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_d0sig', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, -3.5, 3.5), logy=True,
                       normalise=False, lepton='muon', yerr='rsumw2')

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_d0sig', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, -3.5, 3.5), logy=True,
                       normalise=False, lepton='muon', yerr='rsumw2')

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, -1, 1),
                       normalise=False, lepton='muon', yerr='rsumw2')

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'mu_delta_z0_sintheta', weight='reco_weight', title='reco - un-normalised',
                       bins=(30, -1, 1),
                       normalise=False, lepton='muon', yerr='rsumw2')

    # truth plots
    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, lepton='muon', apply_cuts='M_W',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZ_dilep_m_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZneutrino_eta_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wmintaunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'MC_WZneutrino_pt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplustaunu'], 'MC_WZmu_el_eta_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, -4, 4), lepton='muon', apply_cuts='M_W',
                       normalise=False, yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wminmunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'MC_WZmu_el_pt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wminmunu', 'wminmunu'], 'mt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)

    analysis.plot_hist(['wplusmunu', 'wplusmunu'], 'mt_born', weight='truth_weight', title='truth - un-normalised',
                       bins=(30, 150, 5000), logbins=True, logx=True, apply_cuts='M_W',
                       normalise=False, lepton='muon', yerr='rsumw2', scale_by_bin_width=True)
