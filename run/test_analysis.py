#!/users/keanu/miniconda3/envs/root_env/bin/python

from src.analysis import Analysis

if __name__ == '__main__':

    datasets = {
        # 'wminmunu': {
        #     'data_path': '../data/wminmunu_MC.root',
        #     'cutfile_path': '../options/cutfile_EXAMPLE.txt',
        #     'TTree_name': 'truth',
        #     'year': '2015+2016',
        #     'lepton': 'muon',
        #     'force_rebuild': False,
        #     'label': r'$W^-\rightarrow\mu\nu$',
        # },
        'wmintaunu': {
            'data_path': '../data/test_mc16a_wmintaunu/*/*.root',
            'cutfile_path': '../options/cutfile_reco.txt',
            'TTree_name': 'nominal_Loose',
            'year': '2015+2016',
            'lepton': 'tau',
            'force_rebuild': False,
            'label': r'$W^-\rightarrow\tau\nu$',
        }
    }

    my_analysis = Analysis(datasets, analysis_label='test_analysis', log_level=10, log_out='both')

    # my_analysis.plot_mass_slices('wmintaunu', 'mt_born', weight='truth_weight', bins=(50, 200, 5000))
    # my_analysis.plot_mass_slices('wmintaunu', 'MC_WZ_dilep_m_born', weight='truth_weight', bins=(50, 200, 5000))
    # my_analysis.plot_mass_slices('wmintaunu', 'mu_pt', weight='reco_weight', bins=(50, 200, 5000))
    my_analysis['wmintaunu'].profile_plot('MC_WZ_dilep_m_born', 'weight_KFactor', c='k', s=0.5)
    # my_analysis.make_all_cutgroup_2dplots('wminmunu')
    # my_analysis.plot_with_cuts('wminmunu')
