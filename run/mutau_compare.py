from src.analysis import Analysis

if __name__ == '__main__':
    
    datasets = {
        # NEGATIVE
        'wminmunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wminmunu_*/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_HM.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'muon',
            'force_rebuild': False,
            'label': r'$W^-\rightarrow\mu\nu$',
        },
        'wmintaunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu_*/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_HM.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'label': r'$W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wminmunu': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wminmunu/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_peak.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'muon',
            'force_rebuild': False,
            'label': r'$W^-\rightarrow\mu\nu$',
        },
        'wmintaunu': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wmintaunu/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_peak.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'label': r'$W^-\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        
        # POSITIVE
        'wplusmunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu_*/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_HM.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'muon',
            'force_rebuild': False,
            'label': r'$W^+\rightarrow\mu\nu$',
        },
        'wplustaunu_hm': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu_*/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_HM.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'label': r'$W^+\rightarrow\tau\nu\rightarrow\mu\nu$',
        },
        'wplusmunu': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wplusmunu/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_peak.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'muon',
            'force_rebuild': False,
            'label': r'$W^+\rightarrow\mu\nu$',
        },
        'wplustaunu': {
            'data_path': '/data/atlas/HighMassDrellYan/test_mc16a/wplustaunu/*.root',
            'cutfile_path': '../options/jesal_cutflow/DY_peak.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'label': r'$W^+\rightarrow\tau\nu\rightarrow\mu\nu$',
        }
    }
    
    analysis = Analysis(datasets, 'mutau_compare', log_level=10, log_out='console')
    
    analysis.merge("wminmunu",   "wminmunu_hm",   to_pkl=True)
    analysis.merge("wmintaunu",  "wmintaunu_hm",  to_pkl=True)
    analysis.merge("wplusmunu",  "wplusmunu_hm",  to_pkl=True)
    analysis.merge("wplustaunu", "wplustaunu_hm", to_pkl=True)
    
    analysis.plot_hist_overlay(['wminmunu, wmintaunu'],   'met_met', bins=(30, 0, 5000), lepton='muon')
    analysis.plot_hist_overlay(['wplusmunu, wplustaunu'], 'met_met', bins=(30, 0, 5000), lepton='muon')
    analysis.plot_hist_overlay(['wminmunu, wmintaunu'],   'mu_pt', bins=(30, 0, 5000), lepton='muon')
    analysis.plot_hist_overlay(['wplusmunu, wplustaunu'], 'mu_pt', bins=(30, 0, 5000), lepton='muon')
    analysis.plot_hist_overlay(['wminmunu, wmintaunu'],   'mu_d0sig', bins=(30, -3.5, 3.5), logy=False, lepton='muon')
    analysis.plot_hist_overlay(['wplusmunu, wplustaunu'], 'mu_d0sig', bins=(30, -3.5, 3.5), logy=False, lepton='muon')
    analysis.plot_hist_overlay(['wminmunu, wmintaunu'],   'mu_delta_z0_sintheta', bins=(30, -1, 1), lepton='muon')
    analysis.plot_hist_overlay(['wplusmunu, wplustaunu'], 'mu_delta_z0_sintheta', bins=(30, -1, 1), lepton='muon')
